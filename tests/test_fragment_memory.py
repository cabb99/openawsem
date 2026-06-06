"""Tests for the openawsem.memory submodule (MemoryWells / MemoryTable + Registry).

Pure-pandas tabulation math (synthetic, exact), invariants, JSON round-trip, cache-key
invalidation, the molscene gro read, and an array-level golden comparison of the legacy
frags.mem -> MemoryWells -> table path against a frozen pre-refactor reference (needs
OpenMM + molscene).
"""
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openawsem.memory import FragmentMemory, MemoryTable, MemoryWells, available_methods
from openawsem.memory.generators import Registry

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


def test_registry_experimental_raises():
    assert available_methods() == []  # nothing usable yet (Phase 1B fills single/local_blast)
    with pytest.raises(NotImplementedError):
        Registry.create("local_blast")
    with pytest.raises(ValueError):
        Registry.create("nonexistent_method")


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
