"""Tests for the openawsem.memory submodule (MemoryWells / MemoryTable + Registry).

Pure-pandas tabulation math (synthetic, exact), invariants, JSON round-trip, cache-key
invalidation, the molscene gro read, and an array-level golden comparison of the legacy
frags.mem -> MemoryWells -> table path against a frozen pre-refactor reference (needs
OpenMM + molscene).
"""
import warnings

warnings.filterwarnings("ignore")
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openawsem.memory import FragmentMemory, MemoryTable, MemoryWells, available_methods
from openawsem.memory.generators import LocalBlast, Registry, SingleStructure, selection

data_path = Path("tests") / "data"


def _wells(rows):
    return MemoryWells(pd.DataFrame(rows))


def _gauss(r, r_mean, sigma, weight=1.0):
    return weight * np.exp(-((r - r_mean) ** 2) / (2 * sigma ** 2))


# --------------------------------------------------------------------------- #
# Tabulation math (synthetic, exact)
# --------------------------------------------------------------------------- #
def test_single_gaussian_is_exact():
    m = _wells([dict(resid_i=1, name_i="CA", resid_j=5, name_j="CA", r_mean=0.5, sigma=0.13, weight=2.0)])
    pairs, table = m._tabulate_arrays({("CA", 1): 10, ("CA", 5): 20}, 0.0, 5.0, 0.01)
    assert pairs == [(10, 20)]
    np.testing.assert_allclose(table[0], _gauss(np.arange(0, 5, 0.01), 0.5, 0.13, 2.0), atol=1e-12)


def test_components_on_same_pair_sum():
    m = _wells([
        dict(resid_i=1, name_i="CA", resid_j=6, name_j="CB", r_mean=0.4, sigma=0.12, weight=1.0),
        dict(resid_i=1, name_i="CA", resid_j=6, name_j="CB", r_mean=0.6, sigma=0.12, weight=1.0),
    ])
    pairs, table = m._tabulate_arrays({("CA", 1): 0, ("CB", 6): 7}, 0.0, 5.0, 0.01)
    r = np.arange(0, 5, 0.01)
    np.testing.assert_allclose(table[0], _gauss(r, 0.4, 0.12) + _gauss(r, 0.6, 0.12), atol=1e-12)


def test_unresolved_atoms_are_dropped():
    # CB on a (target) glycine has no atom index -> the restraint is dropped.
    m = _wells([dict(resid_i=2, name_i="CB", resid_j=6, name_j="CA", r_mean=0.5, sigma=0.13, weight=1.0)])
    pairs, table = m._tabulate_arrays({("CA", 6): 5}, 0.0, 5.0, 0.01)
    assert pairs == [] and table.shape[0] == 0


def test_memory_table_reproduces_wells():
    """MemoryWells.tabulate() -> MemoryTable that yields the same per-pair curves."""
    m = _wells([
        dict(resid_i=1, name_i="CA", resid_j=5, name_j="CA", r_mean=0.5, sigma=0.13, weight=2.0),
        dict(resid_i=1, name_i="CA", resid_j=5, name_j="CA", r_mean=0.7, sigma=0.10, weight=1.0),
        dict(resid_i=2, name_i="CB", resid_j=6, name_j="CB", r_mean=0.4, sigma=0.12, weight=1.0),
    ])
    target = {("CA", 1): 0, ("CA", 5): 4, ("CB", 2): 9, ("CB", 6): 13}
    mt = m.tabulate(target)
    assert isinstance(mt, MemoryTable)
    p1, t1 = m._tabulate_arrays(target, 0.0, 5.0, 0.01)
    p2, t2 = mt._tabulate_arrays(target, 0.0, 5.0, 0.01)
    assert p1 == p2
    np.testing.assert_allclose(t1, t2, atol=1e-9)


# --------------------------------------------------------------------------- #
# from_fragments + invariants
# --------------------------------------------------------------------------- #
def _linear_fragment(resids, names, spacing_nm=0.38, weight=1.0):
    n = len(resids)
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n) * spacing_nm
    return pd.DataFrame({"target_resid": resids, "name": names, "weight": weight,
                         "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]})


def test_seq_sep_window_and_sigma_law():
    frag = _linear_fragment([1, 2, 3, 4, 5], ["CA"] * 5)
    m = MemoryWells.from_fragments([frag], min_seq_sep=3, max_seq_sep=9, well_width=0.1)
    seps = (m["resid_j"] - m["resid_i"]).to_numpy()
    assert seps.min() >= 3 and seps.max() <= 9
    assert len(m) == 3  # (1,4),(1,5),(2,5)
    np.testing.assert_allclose(m["sigma"].to_numpy(), 0.1 * seps.astype(float) ** 0.15)
    assert isinstance(m, MemoryWells)


def test_pair_canonicalization():
    frag = _linear_fragment([9, 1], ["CA", "CA"], spacing_nm=0.5)
    m = MemoryWells.from_fragments([frag], min_seq_sep=3, max_seq_sep=9, well_width=0.1)
    assert len(m) == 1
    assert int(m["resid_i"].iloc[0]) <= int(m["resid_j"].iloc[0])


def test_slicing_preserves_class():
    m = _wells([dict(resid_i=1, name_i="CA", resid_j=5, name_j="CA", r_mean=0.5, sigma=0.13, weight=1.0)])
    assert isinstance(m.iloc[:1], MemoryWells)
    assert isinstance(m[m["resid_i"] == 1], MemoryWells)


# --------------------------------------------------------------------------- #
# IO + cache key + registry
# --------------------------------------------------------------------------- #
def test_memory_json_roundtrip(tmp_path):
    m = _wells([
        dict(resid_i=1, name_i="CA", resid_j=5, name_j="CA", r_mean=0.5, sigma=0.13, weight=2.0),
        dict(resid_i=2, name_i="CA", resid_j=6, name_j="CB", r_mean=0.4, sigma=0.12, weight=1.0),
    ]).with_metadata(params={"well_width": 0.1}, provenance={"method": "test"})
    p = tmp_path / "mem.json"
    m.save(p)
    loaded = MemoryWells.read(p)
    assert isinstance(loaded, MemoryWells)
    pd.testing.assert_frame_equal(pd.DataFrame(m).reset_index(drop=True),
                                  pd.DataFrame(loaded).reset_index(drop=True))
    assert loaded.params["well_width"] == 0.1 and loaded.provenance["method"] == "test"


def test_cache_key_invalidation():
    m = _wells([dict(resid_i=1, name_i="CA", resid_j=5, name_j="CA", r_mean=0.5, sigma=0.13, weight=1.0)])
    k = m.cache_key(rmin=0, rmax=5, dr=0.01)
    assert k == m.cache_key(rmin=0, rmax=5, dr=0.01)            # stable
    assert k != m.cache_key(rmin=0, rmax=5, dr=0.02)            # grid change
    m2 = _wells([dict(resid_i=1, name_i="CA", resid_j=5, name_j="CA", r_mean=0.6, sigma=0.13, weight=1.0)])
    assert k != m2.cache_key(rmin=0, rmax=5, dr=0.01)           # data change


def test_registry_usable_and_experimental():
    methods = available_methods()
    assert "single" in methods and "local_blast" in methods
    assert isinstance(Registry.create("single"), SingleStructure)
    with pytest.raises(NotImplementedError):
        Registry.create("alphafold")  # experimental stub
    with pytest.raises(ValueError):
        Registry.create("nonexistent_method")


# --------------------------------------------------------------------------- #
# selection (deterministic ranking + exactly-N)
# --------------------------------------------------------------------------- #
def _hit(query_start, k, evalue=0.1, bitscore=100.0):
    return dict(query_start=query_start, query_end=query_start + 8, pdb_id=f"p{query_start}{k}",
                chain="A", subject_start=k + 1, subject_end=k + 9, evalue=evalue, bitscore=bitscore)


def test_selection_exactly_n_per_position():
    recs = [_hit(q, k, evalue=0.1 * k, bitscore=100 - k)
            for q, count in [(1, 5), (2, 1), (3, 4)] for k in range(count)]
    kept, shortfall = selection.select_n_per_position(pd.DataFrame(recs), 3, report=True)
    counts = kept.groupby("query_start").size().to_dict()
    assert counts == {1: 3, 2: 1, 3: 3}
    assert set(shortfall.index) == {2}  # only position 2 has < 3 hits


def test_selection_is_deterministic_under_shuffle():
    hits = pd.DataFrame([_hit(1, k, evalue=0.01 * k, bitscore=100 - k) for k in range(10)])
    a = selection.select_n_per_position(hits, 4)
    b = selection.select_n_per_position(hits.sample(frac=1, random_state=1).reset_index(drop=True), 4)
    pd.testing.assert_frame_equal(a, b)


def test_selection_deduplicates():
    hits = pd.DataFrame([_hit(1, 0)] * 3)
    assert len(selection.select_n_per_position(hits, 20)) == 1


def test_search_windows_do_not_cross_record_boundaries(monkeypatch):
    """A multi-record query is windowed record by record (no junction-spanning windows),
    with target residues numbered continuously across records."""
    lb = LocalBlast(database="x", fragment_length=3, max_gaps=0, evalue=10000)
    calls = []

    def fake_psiblast(query):
        calls.append(query)
        return pd.DataFrame([dict(sseqid="pdb|1abc|A", qstart=1, qend=len(query), sstart=1,
                                  send=len(query), qseq=query, sseq=query, length=len(query),
                                  gaps=0, bitscore=50.0, evalue=0.1, pdb_id="1abc", chain="A")])

    monkeypatch.setattr(lb, "_psiblast", fake_psiblast)

    hits = lb.search(["ACDEF", "GHIKL"])  # two 5-mers, fragment_length 3
    # windows stay inside each record: ACD,CDE,DEF then GHI,HIK,IKL -- never DEF|GHI etc.
    assert calls == ["ACD", "CDE", "DEF", "GHI", "HIK", "IKL"]
    # numbering is continuous: record 2 (len-5 record 1 before it) starts at target 6, not 4/5
    assert sorted(hits["query_start"].unique()) == [1, 2, 3, 6, 7, 8]

    # a plain string is still a single record
    calls.clear()
    lb.search("ACDEFG")
    assert calls == ["ACD", "CDE", "DEF", "EFG"]


def test_create_fragment_memories_passes_records_not_concatenation(tmp_path, monkeypatch):
    """The bridge hands the backend the per-record sequences, not one concatenated string,
    so windows cannot span chains."""
    from openawsem.memory import projects

    fasta = tmp_path / "two_chains.fasta"
    fasta.write_text(">chainA\nACDEFGHIK\n>chainB\nLMNPQRSTV\n")
    captured = {}

    class _StubMem:
        provenance: dict = {}

        def to_frags_mem(self, output):
            Path(output).write_text("")

    def fake_generate(self, sequence, gro_dir=None):
        captured["sequence"] = sequence
        return _StubMem()

    monkeypatch.setattr(projects.LocalBlast, "generate", fake_generate)
    projects.create_fragment_memories(database="x", fasta_file=str(fasta),
                                      output=str(tmp_path / "frags.mem"),
                                      frag_lib_dir=str(tmp_path / "fraglib"))
    assert captured["sequence"] == ["ACDEFGHIK", "LMNPQRSTV"]


def test_localblast_parse_sseqid():
    assert LocalBlast._parse_sseqid("pdb|4V12|A") == ("4v12", "A")
    assert LocalBlast._parse_sseqid("4v12_A") == ("4v12", "A")


def _stub_backend(windows):
    """A StructureBackend whose search/fetch/materialize are driven by ``windows``
    (``{pdb_id: fragment DataFrame}``), for exercising the generate() pipeline offline."""
    from openawsem.memory.generators.base import StructureBackend

    class _FakeScene:
        def write_awsem_gro(self, path, chain=None):
            Path(path).write_text("placeholder")

    class _Stub(StructureBackend):
        n_mem = 20

        def search(self, sequence):
            return pd.DataFrame([
                dict(pdb_id=pid, chain="A", query_start=int(w["target_resid"].min()),
                     query_end=int(w["target_resid"].min()) + 8, subject_start=1,
                     subject_end=9, evalue=0.1 * (i + 1), bitscore=50 - i)
                for i, (pid, w) in enumerate(windows.items())])

        def fetch(self, hit):
            return _FakeScene()

        def materialize(self, hit, scene):
            return windows[hit["pdb_id"]].copy()

    return _Stub()


def _ca_window(resids, target_start, weight=1.0):
    n = len(resids)
    return pd.DataFrame({"chain": "A", "resid": resids, "name": "CA", "resname": "ALA",
                         "target_resid": [target_start + k for k in range(n)], "weight": weight,
                         "x": np.arange(n) * 0.38, "y": 0.0, "z": 0.0})


def test_generate_returned_memory_matches_recorded_fragments(tmp_path):
    """When writing a frags.mem, the returned MemoryWells must contain exactly the
    fragments that were recorded (to_frags_mem reproducible): a non-contiguous fragment
    that no frags.mem line can index is excluded from both, not just from the file."""
    windows = {
        "p0aa": _ca_window([10, 11, 12, 13, 14], target_start=1),   # contiguous -> recordable
        "p1aa": _ca_window([20, 22, 24, 26, 28], target_start=2),   # gapped -> not recordable
    }
    backend = _stub_backend(windows)

    mem = backend.generate("X" * 12, gro_dir=tmp_path)
    # exactly one fragment recorded, and it is the contiguous one
    assert len(mem.provenance["fragments"]) == 1
    assert mem.provenance["fragments"][0]["fragment_start"] == 10
    # returned wells equal those of the recorded fragment alone (no orphan restraints)
    keys = ["resid_i", "name_i", "resid_j", "name_j"]
    got = mem.sort_values(keys).reset_index(drop=True)[keys]
    exp = MemoryWells.from_fragments([windows["p0aa"]]).sort_values(keys).reset_index(drop=True)[keys]
    pd.testing.assert_frame_equal(pd.DataFrame(got), pd.DataFrame(exp))
    # the non-contiguous fragment's target residue 6 never leaks into the returned object
    assert 6 not in set(mem["resid_i"]) | set(mem["resid_j"])

    # with no gro_dir there is no file to diverge from -> all materialized fragments are kept
    mem_all = backend.generate("X" * 12)
    assert 6 in set(mem_all["resid_i"]) | set(mem_all["resid_j"])
    assert "fragments" not in mem_all.provenance


def _backfill_backend(n_mem, failing_prefix, cand):
    from openawsem.memory.generators.base import StructureBackend

    windows = {pid: _ca_window([10, 11, 12, 13, 14], target_start=1) for pid in cand}

    class _FakeScene:
        def write_awsem_gro(self, path, chain=None):
            Path(path).write_text("placeholder")

    class _Stub(StructureBackend):
        pass

    _Stub.n_mem = n_mem

    def search(self, sequence):
        return pd.DataFrame([
            dict(pdb_id=pid, chain="A", query_start=1, query_end=9, subject_start=1,
                 subject_end=9, evalue=0.1 * (i + 1), bitscore=50 - i)
            for i, pid in enumerate(cand)])

    def fetch(self, hit):
        if hit["pdb_id"].startswith(failing_prefix):
            raise RuntimeError("simulated fetch failure")
        return _FakeScene()

    def materialize(self, hit, scene):
        return windows[hit["pdb_id"]].copy()

    _Stub.search, _Stub.fetch, _Stub.materialize = search, fetch, materialize
    return _Stub()


def test_generate_backfills_past_failed_fetches(tmp_path):
    """A position must reach n_mem usable fragments even when its best-ranked hits fail to
    fetch: selection counts materialized fragments, not pre-truncated hits."""
    # ranked by evalue: fa < fb < uc < ud ; fa/fb fail to fetch, uc/ud are usable
    cand = ["faaa", "fbbb", "uccc", "uddd"]
    backend = _backfill_backend(n_mem=2, failing_prefix="f", cand=cand)
    mem = backend.generate("X" * 12, gro_dir=tmp_path)
    used = mem.provenance["fragments"]
    assert len(used) == 2  # backfilled to N despite the two best hits failing
    assert {Path(u["location"]).name for u in used} == {"ucccA.gro", "udddA.gro"}


def test_generate_respects_n_mem_cap(tmp_path):
    """Backfill does not overshoot: with every hit usable, only the top n_mem are kept."""
    cand = ["uaaa", "ubbb", "uccc", "uddd"]
    backend = _backfill_backend(n_mem=2, failing_prefix="none", cand=cand)
    mem = backend.generate("X" * 12, gro_dir=tmp_path)
    used = mem.provenance["fragments"]
    assert len(used) == 2  # the two best-ranked usable hits
    assert {Path(u["location"]).name for u in used} == {"uaaaA.gro", "ubbbA.gro"}


def test_brain_damage_fails_closed_on_homolog_blast_failure(monkeypatch):
    """If the homolog BLAST errors, brain_damage must fail closed (raise), not silently
    treat every hit as a non-homolog and keep them all."""
    hits = pd.DataFrame([
        dict(pdb_id="aaaa", chain="A", query_start=1, subject_start=1, subject_end=9, evalue=0.1),
        dict(pdb_id="bbbb", chain="A", query_start=1, subject_start=1, subject_end=9, evalue=0.2),
    ])
    seq = "ACDEFGHIK"
    # a non-existent executable makes the underlying subprocess.run raise FileNotFoundError
    lb = LocalBlast(database="x", brain_damage=1, blast_exe="definitely-not-a-real-blast-exe")
    with pytest.raises(RuntimeError):
        lb._filter_hits(hits, seq)

    # brain_damage=0 never consults homologs, so a broken homolog BLAST is harmless there
    lb0 = LocalBlast(database="x", brain_damage=0, blast_exe="definitely-not-a-real-blast-exe")
    assert len(lb0._filter_hits(hits, seq)) == len(hits)

    # a homolog BLAST that *runs* but finds nothing is a valid empty set: keep all for bd=1
    monkeypatch.setattr(lb, "_homologs", lambda s: {})
    assert set(lb._filter_hits(hits, seq)["pdb_id"]) == {"aaaa", "bbbb"}


def test_parse_blast_keeps_only_final_iteration():
    """PSI-BLAST -outfmt 7 concatenates every round; only the last (converged) round's
    alignments must survive parsing — intermediate-round-only hits are dropped."""
    lb = LocalBlast(database="x", max_gaps=0, evalue=10000)

    def row(pdb, evalue):
        # sseqid qstart qend sstart send qseq sseq length gaps bitscore evalue
        return "\t".join(["pdb|" + pdb + "|A", "1", "9", "1", "9",
                          "ACDEFGHIK", "ACDEFGHIK", "9", "0", "50.0", str(evalue)])

    stdout = "\n".join([
        "# PSIBLAST 2.13.0+",
        "# Iteration: 1",
        "# Fields: subject id, ...",
        row("1aaa", 0.001),          # appears only in round 1 -> must be dropped
        row("1bbb", 0.5),
        "# Iteration: 2",
        "# Fields: subject id, ...",
        row("1bbb", 0.002),          # final round
        row("1ccc", 0.003),
        "",
        "Search has CONVERGED!",
        "# BLAST processed 1 queries",
    ])
    df = lb._parse_blast(stdout)
    assert set(df["pdb_id"]) == {"1bbb", "1ccc"}
    assert "1aaa" not in set(df["pdb_id"])
    # a single-round output (no Iteration markers) still yields all its rows
    single = "\n".join([row("1ddd", 0.01), row("1eee", 0.02)])
    assert set(lb._parse_blast(single)["pdb_id"]) == {"1ddd", "1eee"}


def test_local_blast_brain_damage_filtering(monkeypatch):
    hits = pd.DataFrame([
        dict(pdb_id="aaaa", chain="A", query_start=1, subject_start=1, subject_end=9, evalue=0.1),
        dict(pdb_id="bbbb", chain="A", query_start=1, subject_start=1, subject_end=9, evalue=0.2),
        dict(pdb_id="cccc", chain="A", query_start=1, subject_start=1, subject_end=9, evalue=0.3),
    ])
    homologs = {"aaaa": 99.0, "bbbb": 60.0}  # aaaa = self(>90), bbbb = homolog, cccc = not a homolog

    def backend(brain_damage):
        lb = LocalBlast(database="x", brain_damage=brain_damage, cutoff_identical=90)
        monkeypatch.setattr(lb, "_homologs", lambda seq: homologs)
        return lb

    seq = "ACDEFGHIK"
    assert set(backend(0)._filter_hits(hits, seq)["pdb_id"]) == {"aaaa", "bbbb", "cccc"}  # all
    assert set(backend(1)._filter_hits(hits, seq)["pdb_id"]) == {"cccc"}                  # exclude homologs
    assert set(backend(2)._filter_hits(hits, seq)["pdb_id"]) == {"bbbb"}                  # homologs except self
    assert set(backend(0.5)._filter_hits(hits, seq)["pdb_id"]) == {"aaaa"}                # self only


# --------------------------------------------------------------------------- #
# SingleStructure backend
# --------------------------------------------------------------------------- #
def test_single_structure_functional():
    pytest.importorskip("molscene")
    pdb = "tests/data/1mbn-crystal_structure.pdb"
    if not Path(pdb).exists():
        pytest.skip("test structure missing")
    mem = SingleStructure(pdb, weight=20).generate()
    assert isinstance(mem, MemoryWells) and len(mem) > 0
    seps = (mem["resid_j"] - mem["resid_i"]).to_numpy()
    assert seps.min() >= 3 and seps.max() <= 9
    assert set(mem["name_i"]).issubset({"CA", "CB"}) and (mem["weight"] == 20).all()


def test_single_structure_matches_legacy_geometry():
    """SingleStructure(pdb) reproduces the legacy single memory's pair set and geometry
    (up to the .gro's 3-decimal coordinate rounding)."""
    pytest.importorskip("molscene")
    pdb = "tests/data/1mbn-crystal_structure.pdb"
    mem_file = data_path / "1mbn-single_frags.mem"
    if not (Path(pdb).exists() and mem_file.exists()):
        pytest.skip("inputs missing")
    keys = ["resid_i", "name_i", "resid_j", "name_j"]
    new = SingleStructure(pdb, weight=20).generate().sort_values(keys).reset_index(drop=True)
    legacy = MemoryWells.from_frags_mem(mem_file).sort_values(keys).reset_index(drop=True)
    merged = new.merge(legacy, on=keys, suffixes=("_new", "_leg"))
    assert len(merged) == len(new) == len(legacy)  # identical restraint set
    np.testing.assert_allclose(merged["r_mean_new"], merged["r_mean_leg"], atol=2e-3)  # gro rounding


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_single(tmp_path):
    pytest.importorskip("molscene")
    pdb = "tests/data/1mbn-crystal_structure.pdb"
    if not Path(pdb).exists():
        pytest.skip("test structure missing")
    from openawsem.memory import cli
    out = tmp_path / "mem.json"
    cli.main(["--method", "single", "--structure", pdb, "--weight", "20", "--out", str(out)])
    assert out.exists()
    loaded = MemoryWells.read(out)
    assert isinstance(loaded, MemoryWells) and len(loaded) > 0


def test_cli_list_methods(capsys):
    from openawsem.memory import cli
    cli.main(["--list-methods"])
    out = capsys.readouterr().out
    assert "single" in out and "local_blast" in out


# --------------------------------------------------------------------------- #
# molscene gro read + golden
# --------------------------------------------------------------------------- #
def test_molscene_gro_read_matches_legacy_pandas():
    pytest.importorskip("molscene")
    from openawsem.memory._molscene import read_awsem_gro

    gro = "examples/1r69/fraglib/4e5sa.gro"
    if not Path(gro).exists():
        pytest.skip("fraglib gro missing")
    legacy = pd.read_csv(gro, skiprows=2, sep=r"\s+", header=None,
                         names=["Res_id", "Res", "Type", "i", "x", "y", "z"])
    legacy = legacy[legacy["Type"].isin(["CA", "CB"])].sort_values(["Res_id", "Type"])
    new = read_awsem_gro(gro).sort_values(["resid", "name"])
    assert len(legacy) == len(new)
    np.testing.assert_allclose(legacy[["x", "y", "z"]].to_numpy(),
                               new[["x", "y", "z"]].to_numpy(), atol=1e-6)


@pytest.fixture(scope="module")
def oa_1r69():
    import openawsem
    pdb = "examples/1r69/1r69-openmmawsem.pdb"
    chain = openawsem.helperFunctions.myFunctions.getAllChains(pdb)
    seq = openawsem.helperFunctions.myFunctions.read_fasta("examples/1r69/1r69.fasta")
    return openawsem.OpenMMAWSEMSystem(pdb, chains=chain, k_awsem=1.0,
                                       xml_filename=openawsem.xml, seqFromPdb=seq,
                                       includeLigands=False)


def test_golden_1r69_table_matches_legacy(oa_1r69):
    golden_file = data_path / "1r69-frags.golden_frag_table.npz"
    if not golden_file.exists():
        pytest.skip("golden fixture missing")
    m = MemoryWells.from_frags_mem("examples/1r69/frags.mem")
    pairs, table = m._tabulate_arrays(oa_1r69, 0.0, 5.0, 0.01)
    index = {p: k for k, p in enumerate(pairs)}
    g = np.load(golden_file)
    golden = {tuple(int(x) for x in p): row for p, row in zip(g["pairs"], g["table"])}
    assert set(index) == set(golden)
    for pair, k in index.items():
        np.testing.assert_allclose(table[k], golden[pair], atol=1e-4)


# --------------------------------------------------------------------------- #
# align_fragments (moved into the memory module) + score
# --------------------------------------------------------------------------- #
def test_align_fragments_score_1r69():
    import openawsem.memory.align as align
    fasta, frags = "examples/1r69/1r69.fasta", "examples/1r69/frags.mem"
    if not (Path(fasta).exists() and Path(frags).exists()):
        pytest.skip("1r69 inputs missing")
    df = align.align_fragments(fasta, frags, working_directory="examples/1r69")
    n_lines = len(pd.read_csv(frags, skiprows=4, sep=r"\s+", comment="#",
                              names=["a", "b", "c", "d", "e"]))
    assert len(df) == n_lines > 0
    assert ((df["score"] >= 0) & (df["score"] <= 1)).all()
    assert df["score"].mean() > 0.5           # BLAST fragments match the target well
    assert (df["score"] == 1.0).any()          # some fragments are identical to the target
    # deterministic
    pd.testing.assert_frame_equal(df, align.align_fragments(fasta, frags, working_directory="examples/1r69"))
    # score == independent identity recompute, for the first fragment
    target = "".join(align.parse_fasta(fasta).values())
    first = df.iloc[0]
    span = target[first.seq_init - 1: first.seq_init - 1 + first.length]
    expected = sum(a == b for a, b in zip(first.fragment, span)) / first.length
    assert abs(first.score - expected) < 1e-12


# --------------------------------------------------------------------------- #
# Phase 4 bridge: term consumes a memory object/JSON; frags.mem round-trip
# --------------------------------------------------------------------------- #
def test_term_accepts_memory_object_and_json(tmp_path, oa_1r69):
    import openawsem.functionTerms.templateTerms as templateTerms
    memory = MemoryWells.from_frags_mem("examples/1r69/frags.mem")
    json_path = tmp_path / "mem.json"
    memory.save(json_path)
    f_obj = templateTerms.fragment_memory_term(oa_1r69, fragment_memory=memory,
                                               UseSavedFragTable=False, npy_frag_table=None)
    f_json = templateTerms.fragment_memory_term(oa_1r69, fragment_memory=str(json_path),
                                                UseSavedFragTable=False, npy_frag_table=None)
    assert f_obj.getNumBonds() == f_json.getNumBonds() > 0


def test_frags_mem_roundtrip_1r69(tmp_path, oa_1r69):
    m1 = MemoryWells.from_frags_mem("examples/1r69/frags.mem")
    out = tmp_path / "frags.mem"
    m1.to_frags_mem(out)  # writes absolute gro paths from the fragment provenance
    m2 = MemoryWells.from_frags_mem(out)
    p1, t1 = m1._tabulate_arrays(oa_1r69, 0.0, 5.0, 0.01)
    p2, t2 = m2._tabulate_arrays(oa_1r69, 0.0, 5.0, 0.01)
    assert set(p1) == set(p2)
    i1 = {p: k for k, p in enumerate(p1)}
    i2 = {p: k for k, p in enumerate(p2)}
    for pair in p1:
        np.testing.assert_allclose(t1[i1[pair]], t2[i2[pair]], atol=1e-9)


# --------------------------------------------------------------------------- #
# local_blast against the local cullpdb DB (gated on DB + psiblast; network opt-in)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cullpdb_db():
    import openawsem
    db = str(openawsem.data_path.blast)
    if not Path(db).exists():
        pytest.skip("cullpdb BLAST database not present")
    if shutil.which("psiblast") is None:
        pytest.skip("psiblast not on PATH")
    return db


def test_local_blast_search_deterministic(cullpdb_db):
    import openawsem
    seq = openawsem.helperFunctions.myFunctions.read_fasta("examples/1r69/1r69.fasta")
    lb = LocalBlast(database=cullpdb_db, fragment_length=9, evalue=10000)
    h1 = lb._psiblast(seq[:9])
    assert h1 is not None and len(h1) > 0
    assert {"pdb_id", "chain", "qstart", "sstart", "sseq", "evalue", "bitscore"} <= set(h1.columns)
    assert (h1["pdb_id"] == "1r69").any()  # the target's own chain is in the DB
    pd.testing.assert_frame_equal(h1.reset_index(drop=True),
                                  lb._psiblast(seq[:9]).reset_index(drop=True))  # deterministic


def test_local_blast_homologs_real(cullpdb_db):
    import openawsem
    seq = openawsem.helperFunctions.myFunctions.read_fasta("examples/1r69/1r69.fasta")
    homologs = LocalBlast(database=cullpdb_db)._homologs(seq)
    assert "1r69" in homologs and homologs["1r69"] > 90  # the target's own chain is a self-homolog


def test_local_blast_generate_smoke(cullpdb_db):
    if not os.environ.get("OPENAWSEM_TEST_NETWORK"):
        pytest.skip("set OPENAWSEM_TEST_NETWORK=1 to run the network-dependent generate smoke")
    import openawsem
    seq = openawsem.helperFunctions.myFunctions.read_fasta("examples/1r69/1r69.fasta")
    mem = LocalBlast(database=cullpdb_db, n_mem=1, fragment_length=9, evalue=10000).generate(seq[:11])
    assert isinstance(mem, MemoryWells) and len(mem) > 0
    assert int(mem["resid_i"].min()) >= 1 and int(mem["resid_j"].max()) <= 11
    seps = (mem["resid_j"] - mem["resid_i"]).to_numpy()
    assert seps.min() >= 3 and seps.max() <= 9


# --------------------------------------------------------------------------- #
# structure fetch (local mmCIF) + project bridges
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def local_cif_dir():
    path = "/home/cb/Databases/mmCIF"
    if not Path(path).exists():
        pytest.skip("local mmCIF database not present")
    return path


def test_fetch_local_cif(local_cif_dir):
    pytest.importorskip("molscene")
    from openawsem.memory.structures import fetch_structure, local_cif_path
    if local_cif_path("1r69", local_cif_dir) is None:
        pytest.skip("1r69 not in local mmCIF database")
    scene = fetch_structure("1r69", cif_dir=local_cif_dir)
    assert len(scene) > 0 and "A" in set(scene["chain"])


def test_create_single_memory_bridge(tmp_path):
    pytest.importorskip("molscene")
    pdb = "tests/data/1mbn-crystal_structure.pdb"
    if not Path(pdb).exists():
        pytest.skip("test structure missing")
    from openawsem.helperFunctions import create_single_memory
    out = tmp_path / "single_frags.mem"
    create_single_memory(pdb, "-1", output=str(out), gro_dir=str(tmp_path))
    assert out.exists() and (tmp_path / "1mbn-crystal_structure_A.gro").exists()
    m = MemoryWells.from_frags_mem(out)
    assert len(m) > 0 and (m["weight"] == 20).all()


def test_local_blast_frags_mem_roundtrip(tmp_path, cullpdb_db, local_cif_dir):
    """Fully local (no network): generate -> gro library + frags.mem -> read back."""
    import openawsem
    seq = openawsem.helperFunctions.myFunctions.read_fasta("examples/1r69/1r69.fasta")
    mem = LocalBlast(database=cullpdb_db, n_mem=2, fragment_length=9,
                     cif_dir=local_cif_dir).generate(seq[:15], gro_dir=tmp_path / "fraglib")
    assert len(mem.provenance.get("fragments", [])) > 0
    out = tmp_path / "frags.mem"
    mem.to_frags_mem(out)
    reread = MemoryWells.from_frags_mem(out)
    assert len(reread) > 0
    seps = (reread["resid_j"] - reread["resid_i"]).to_numpy()
    assert seps.min() >= 3 and seps.max() <= 9
