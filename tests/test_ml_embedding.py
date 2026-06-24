"""Contract tests for the learned-embedding retrieval (`ml` backend) on a synthetic mini-DB.

Reuses the synthetic database/tensor builders from ``test_fragment_memory`` so these run without
the machine-local fragment database.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_fragment_memory import (_make_varied_db_dir, _make_tensor_npz,  # noqa: E402
                                  _backbone_coords)

from openawsem.memory.retrieval import FragmentIndex
from openawsem.memory.retrieval.index import ATOM_PAIRS, _DECODE
from openawsem.memory.retrieval.embedding import EmbeddingIndex
from openawsem.memory.ml import fingerprint as fp
from openawsem.memory.generators.ml import Ml
from openawsem.memory.memory import MemoryWells, WELLS_COLUMNS

CHAINS = {"C1": "AKLVRSTEDM", "C2": "EDMWFAKLRS", "C3": "LRSTEDMWFA"}   # len-10, L=5 windows
SEQ = "AKLVRSTEDM"
L = 5
CACB = {("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB")}


def test_fingerprint_roundtrips_get_fragment(tmp_path):
    """Feature: the structural fingerprint must equal the database's stored pair distances."""
    idx = FragmentIndex.load(_make_varied_db_dir(tmp_path, CHAINS))
    rows = np.asarray(idx._encoded_for_length(L)[0])[:4]
    S, mask, (pa, pb, _) = fp.assemble(idx, rows, L)
    pair_pos = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(pa, pb))}
    for m, r in enumerate(rows):
        cu = idx.chain_names[int(idx.chain_code[r])]
        start = int(idx.internal_index[r])
        frag = idx.get_fragment(cu, start, L)
        assert len(frag) > 0
        for row in frag.itertuples(index=False):
            k = pair_pos[(int(row.internal_i) - start, int(row.internal_j) - start)]
            for c, (X, Y) in enumerate(ATOM_PAIRS):
                ref = getattr(row, f"d_{X}_{Y}")
                if not np.isnan(ref):
                    assert abs(ref - S[m, k, c] * 10.0) < 1e-3      # nm -> A


def test_dstruct_zero_for_identical_fragment(tmp_path):
    """Feature: d_struct of a fragment with itself is zero (well-defined metric)."""
    idx = FragmentIndex.load(_make_varied_db_dir(tmp_path, CHAINS))
    rows = np.asarray(idx._encoded_for_length(L)[0])[:3]
    F = fp.weighted_features(idx, rows, L)
    assert fp.d_struct(F[0], F[0]) == 0.0
    assert fp.d_struct(F[0], F[1]) > 0.0


def test_v0_embedding_self_retrieval_and_roundtrip(tmp_path):
    """Feature: an exact query window self-retrieves at cosine ~1; save/load is identity."""
    idx = FragmentIndex.load(_make_varied_db_dir(tmp_path, CHAINS))
    emb = EmbeddingIndex.build(idx, "v0", L=L)
    assert emb.version == "v0" and emb.Z.shape[1] == L * 20
    r = int(np.asarray(idx._encoded_for_length(L)[0])[0])
    win = "".join(_DECODE[idx.ctx[r, :L]])
    rows, sims = emb.search(win, top=3)
    assert r in rows and sims[0] > 0.999
    emb.save(str(tmp_path / "embdir"))
    emb2 = EmbeddingIndex.load(str(tmp_path / "embdir"))
    r2, _ = emb2.search(win, top=3)
    assert np.array_equal(rows, r2)


def test_ml_generate_schema(tmp_path):
    """Feature: ml.generate returns a valid non-empty MemoryWells of only CA/CB wells."""
    db = _make_varied_db_dir(tmp_path, CHAINS)
    mem = Ml(db, encoder="v0", fragment_length=L, min_seq_sep=3, max_seq_sep=4,
             top_k=5, backoff=False).generate(SEQ)
    assert isinstance(mem, MemoryWells) and len(mem) > 0
    assert list(mem.columns) == WELLS_COLUMNS
    assert set(zip(mem["name_i"], mem["name_j"])) <= CACB
    assert (mem["sigma"] > 0).all() and (mem["weight"] > 0).all()
    assert mem["r_mean"].between(0.05, 5.0).all()
    assert mem.provenance["method"] == "ml" and mem.provenance["n_backoff"] == 0


def test_ml_backoff_fills_low_coverage(tmp_path):
    """Feature: when coverage is thin, wells come from the marginal tensor instead."""
    db = _make_varied_db_dir(tmp_path, CHAINS)
    tensor = _make_tensor_npz(tmp_path / "tensors.npz")
    mem = Ml(db, encoder="v0", fragment_length=L, min_seq_sep=3, max_seq_sep=4,
             backoff=True, backoff_sim=2.0, marginal_db=str(tensor)).generate(SEQ)   # 2.0 forces backoff
    assert isinstance(mem, MemoryWells) and len(mem) > 0
    assert list(mem.columns) == WELLS_COLUMNS
    assert mem.provenance["n_backoff"] == mem.provenance["n_windows"] > 0
    assert (mem["sigma"] > 0).all() and (mem["weight"] > 0).all()


def test_ml_registered_non_experimental():
    """Feature: 'ml' is a usable backend, not an experimental stub."""
    from openawsem.memory import available_methods
    assert "ml" in available_methods()


def test_hnsw_index_self_retrieves_and_persists(tmp_path):
    """Feature: the HNSW backend retrieves correctly and persists its graph across save/load."""
    idx = FragmentIndex.load(_make_varied_db_dir(tmp_path, CHAINS))
    hnsw = EmbeddingIndex.build(idx, "v0", L=L, index_type="hnsw", ef_search=64)
    r = int(np.asarray(idx._encoded_for_length(L)[0])[0])
    win = "".join(_DECODE[idx.ctx[r, :L]])
    rows, sims = hnsw.search(win, top=3)
    assert r in rows and sims[0] > 0.999
    d = str(tmp_path / "hnsw")
    hnsw.save(d)
    assert os.path.isfile(os.path.join(d, "embedding_index", "ann.faiss"))
    h2 = EmbeddingIndex.load(d)
    assert h2.index_type == "hnsw"
    r2, _ = h2.search(win, top=3)
    assert np.array_equal(rows, r2)


def test_ml_mutation_engine_matches_bruteforce(tmp_path):
    """Feature: engine.dE equals the full mutant-memory energy difference (a mutation only changes
    the windows covering it)."""
    db = _make_varied_db_dir(tmp_path, CHAINS)
    coords = _backbone_coords(len(SEQ))
    ml = Ml(db, encoder="v0", fragment_length=L, min_seq_sep=3, max_seq_sep=4,
            top_k=5, temp=0.07, backoff=False)
    engine = ml.mutation_engine(SEQ, coords)
    e_wt = ml.generate(SEQ).energy_at(coords)
    assert abs(engine.total_energy() - e_wt) < 1e-5
    for pos, aa in [(2, "W"), (5, "G"), (8, "P"), (3, "A")]:
        if SEQ[pos - 1] == aa:
            continue
        de = engine.dE(pos, aa)
        mut = SEQ[:pos - 1] + aa + SEQ[pos:]
        oracle = ml.generate(mut).energy_at(coords) - e_wt
        assert abs(de - oracle) < 1e-4, (pos, aa, de, oracle)


def test_v3_esm_encoder_db_pathway(tmp_path):
    """Feature: the v3 ESM head encodes DB fragments by row (pool memmap -> head -> normalized) and
    round-trips save/load.  Uses a synthetic feature memmap so it does not load the ESM model."""
    import pytest
    pytest.importorskip("torch")
    from openawsem.memory.ml.esm_encoder import Esm2Encoder, load_esm_encoder
    idx = FragmentIndex.load(_make_varied_db_dir(tmp_path, CHAINS))
    n_res = len(np.asarray(idx.segment_id))
    feat = np.random.default_rng(0).standard_normal((n_res, 16)).astype(np.float16)
    enc = Esm2Encoder(feat, model_key="t30_150M", layer=30, d_esm=16, L=L, hidden=8, dim=4)
    rows = np.asarray(idx._encoded_for_length(L)[0])
    Z = enc.encode_db(idx, rows)
    assert Z.shape == (len(rows), 4)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-4)        # L2-normalized
    assert enc.train_inputs(idx, rows).shape == (len(rows), 16)          # pooled features
    emb = EmbeddingIndex.build(idx, enc, L=L)
    assert emb.version == "v3" and emb.Z.shape == (len(rows), 4)
    r, s = emb.search_vectors(Z[:2], top=3)
    assert r.shape == (2, 3) and s[0, 0] > 0.999                          # self top-1
    p = str(tmp_path / "v3head.pt")
    enc.save(p)
    enc2 = load_esm_encoder(p, esm_feat=feat)
    assert np.allclose(Z[:5], enc2.encode_db(idx, rows[:5]), atol=1e-5)


def test_ml_mutation_engine_commit(tmp_path):
    """Feature: commit applies a mutation so later dE/energy are relative to it."""
    db = _make_varied_db_dir(tmp_path, CHAINS)
    coords = _backbone_coords(len(SEQ))
    engine = Ml(db, encoder="v0", fragment_length=L, min_seq_sep=3, max_seq_sep=4,
                top_k=5, backoff=False).mutation_engine(SEQ, coords)
    e0 = engine.total_energy()
    de = engine.dE(4, "W")
    engine.commit(4, "W")
    assert engine.seq[3] == "W"
    assert abs(engine.total_energy() - (e0 + de)) < 1e-4
    assert engine.dE(4, "W") == 0.0          # already W -> no-op
