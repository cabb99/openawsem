"""Fragment memory as a pandas DataFrame.

A fragment memory is a tidy table of pairwise CA/CB distance restraints in **target
residue numbering** and **nanometers**.  Two concrete representations subclass the
shared base :class:`FragmentMemory` (itself a ``pandas.DataFrame``):

* :class:`MemoryWells` -- parametric: one Gaussian well per row
  (``r_mean``, ``sigma``, ``weight``).  This is the AWSEM fragment-memory potential.
* :class:`MemoryTable` -- non-parametric: a sampled potential, long form
  (one ``(r, energy)`` sample per row), for learned/sampled methods (ML, atomistic).

Both reduce, via :meth:`FragmentMemory.tabulate`, to the same per-pair table sampled on
a distance grid -- the "tabulated Gaussian" the OpenMM term consumes.  Subclassing
DataFrame (with ``_constructor`` = ``type(self)``) keeps the class through slicing so a
memory can later ride inside a coarse-grained molscene ``Scene`` and be sliced/merged
with it.  Run parameters and provenance live in ``df.attrs``.
"""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

WELLS_COLUMNS = ["resid_i", "name_i", "resid_j", "name_j", "r_mean", "sigma", "weight"]
TABLE_COLUMNS = ["resid_i", "name_i", "resid_j", "name_j", "r", "energy"]
ATOM_RANK = {"CA": 0, "CB": 1}

#: bump when the math/ordering changes so cached tables invalidate.
TABULATOR_VERSION = "2"


# =========================================================================== #
# base
# =========================================================================== #
class FragmentMemory(pd.DataFrame):
    """Base class: a DataFrame of pairwise CA/CB restraints (target resid space, nm).

    Subclasses define the row schema (:data:`WELLS_COLUMNS` / :data:`TABLE_COLUMNS`) via
    ``_DTYPES`` and implement :meth:`_tabulate_arrays`.  Behavior shared here:
    target-atom resolution, the OpenMM force build (+ legacy-compatible cache), JSON IO,
    and the molscene Scene bridge.
    """

    _DTYPES: dict = {}

    @property
    def _constructor(self):
        return type(self)

    # ----- metadata (pandas-native attrs) ------------------------------- #
    @property
    def params(self) -> dict:
        return self.attrs.setdefault("params", {})

    @property
    def provenance(self) -> dict:
        return self.attrs.setdefault("provenance", {})

    def with_metadata(self, *, params=None, provenance=None) -> "FragmentMemory":
        if params:
            self.attrs.setdefault("params", {}).update(params)
        if provenance:
            self.attrs.setdefault("provenance", {}).update(provenance)
        return self

    # ----- target atom-index resolution --------------------------------- #
    @staticmethod
    def atom_index_map(target) -> dict:
        """Map ``(name, resid)`` -> global atom index from ``target``.

        ``target`` may be a mapping (used as-is), an ``OpenMMAWSEMSystem`` (uses
        ``ca``/``cb``/``resi``; resid is 1-based), or a molscene ``Scene`` (uses
        ``name``/``resid``/``index`` columns).
        """
        if isinstance(target, dict):
            return target
        if hasattr(target, "ca") and hasattr(target, "cb") and hasattr(target, "resi"):
            mapping = {}
            ca, cb, resi = set(target.ca), set(target.cb), target.resi
            for i in range(target.natoms):
                if i in ca:
                    mapping[("CA", 1 + int(resi[i]))] = i
                if i in cb:
                    mapping[("CB", 1 + int(resi[i]))] = i
            return mapping
        frame = pd.DataFrame(target)
        sub = frame[frame["name"].isin(("CA", "CB"))]
        return {(n, int(r)): int(ix) for n, r, ix in zip(sub["name"], sub["resid"], sub["index"])}

    # ----- tabulation (subclass-specific) ------------------------------- #
    def _tabulate_arrays(self, target, rmin, rmax, dr):
        """Return ``(pairs, frag_table)``: sorted ``(atom_i, atom_j)`` pairs and the
        ``(n_pairs, r_size)`` potential sampled on ``np.arange(rmin, rmax, dr)``."""
        raise NotImplementedError

    def tabulate(self, target, *, rmin=0.0, rmax=5.0, dr=0.01) -> "MemoryTable":
        """Return a :class:`MemoryTable` (long-form sampled potential) for ``target``."""
        pairs, table = self._tabulate_arrays(target, rmin, rmax, dr)
        inv = {v: k for k, v in self.atom_index_map(target).items()}
        r = np.arange(rmin, rmax, dr)
        rows = []
        for (ai, aj), curve in zip(pairs, table):
            (name_i, resid_i), (name_j, resid_j) = inv[ai], inv[aj]
            rows.append(pd.DataFrame({
                "resid_i": resid_i, "name_i": name_i, "resid_j": resid_j, "name_j": name_j,
                "r": r, "energy": curve,
            }))
        out = MemoryTable(pd.concat(rows, ignore_index=True) if rows else MemoryTable.empty())
        out.attrs.update(self.attrs)
        return out

    # ----- energy of a structure ---------------------------------------- #
    def _value_at(self, sub, r) -> float:
        """The per-pair tabulated value at distance ``r`` (nm) for the rows of one pair."""
        raise NotImplementedError

    @staticmethod
    def _structure_coords(structure, *, virtual_cb=False) -> dict:
        """``{(name, resid): xyz_nm}`` for CA/CB, from a dict, ``OpenMMAWSEMSystem``,
        molscene ``Scene`` or a structure file path.

        ``virtual_cb=True`` fills a virtual CB (built in Angstrom from backbone N/CA/C) at residues
        with no observed CB (glycine), matching the fragment database.  Use it ONLY with an engine that
        masks CB by the current sequence (e.g. :class:`FullDBMutationEnergy`); without that masking it
        would add CB wells to glycines.  Only the molscene/file path can build it (needs N/CA/C); a
        coords dict or an ``OpenMMAWSEMSystem`` (no backbone N/C beads) is returned unchanged.
        """
        if isinstance(structure, dict):
            return structure
        if hasattr(structure, "ca") and hasattr(structure, "cb") and hasattr(structure, "resi"):
            try:
                from openmm.unit import nanometer
            except ModuleNotFoundError:  # pragma: no cover
                from simtk.unit import nanometer
            pos = np.asarray(structure.pdb.getPositions().value_in_unit(nanometer))
            ca, cb, resi = set(structure.ca), set(structure.cb), structure.resi
            out = {}
            for i in range(structure.natoms):
                if i in ca:
                    out[("CA", 1 + int(resi[i]))] = pos[i]
                if i in cb:
                    out[("CB", 1 + int(resi[i]))] = pos[i]
            return out
        from openawsem.memory import _molscene
        scene = _molscene.load_structure(structure) if isinstance(structure, (str, Path)) else structure
        atoms = _molscene.cacb_with_virtual_cb(scene) if virtual_cb else _molscene.cacb_nm(scene)
        return {(n, int(r)): np.array([x, y, z], dtype=float) for n, r, x, y, z in
                zip(atoms["name"], atoms["resid"], atoms["x"], atoms["y"], atoms["z"])}

    def energy_at(self, structure, *, k_fm=0.04184) -> float:
        """Fragment-memory energy ``-k_fm * sum_pairs T_p(r_p)`` of ``structure`` under this
        memory, where ``r_p`` is the CA/CB distance of each pair in ``structure``.  Pairs with
        a missing atom (e.g. a GLY CB) are skipped."""
        if len(self) == 0:
            return 0.0
        coords = self._structure_coords(structure)
        keys = ["resid_i", "name_i", "resid_j", "name_j"]
        total = 0.0
        for (ri, ni, rj, nj), sub in pd.DataFrame(self).groupby(keys, sort=False):
            ci, cj = coords.get((ni, int(ri))), coords.get((nj, int(rj)))
            if ci is None or cj is None:
                continue
            total += self._value_at(sub, float(np.linalg.norm(np.asarray(ci) - np.asarray(cj))))
        return -float(k_fm) * total

    # ----- OpenMM force -------------------------------------------------- #
    def to_openmm_force(self, oa, *, k_fm=0.04184, caOnly=False, forceGroup=23,
                        rmin=0.0, rmax=5.0, dr=0.01, cache=None, use_cache=False, debug=False):
        """Build the ``CustomCompoundBondForce`` for system ``oa`` (lazy openmm import).

        ``cache`` is an optional path; it is byte-compatible with the legacy
        ``(frag_table, interaction_list, pair_to_index)`` pickle, with a ``.meta.json``
        sidecar carrying a key so a stale cache is rebuilt.
        """
        if use_cache and cache and Path(cache).is_file():
            pairs, table, (rmin, rmax, dr) = self._load_cache(cache)
        else:
            pairs, table = self._tabulate_arrays(oa, rmin, rmax, dr)
            if cache:
                self._save_cache(cache, pairs, table, rmin, rmax, dr,
                                 key=self.cache_key(rmin=rmin, rmax=rmax, dr=dr))
        return self._build_force(pairs, table, oa, k_fm=k_fm, caOnly=caOnly, forceGroup=forceGroup,
                                 rmin=rmin, rmax=rmax, dr=dr, debug=debug)

    def cache_key(self, *, rmin, rmax, dr) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(f"{type(self).__name__}|{TABULATOR_VERSION}|{rmin}|{rmax}|{dr}".encode())
        h.update(f"|{self.params}".encode())
        cols = [c for c in (WELLS_COLUMNS + ["r", "energy"]) if c in self.columns]
        if len(self):
            h.update(pd.util.hash_pandas_object(self[cols], index=False).values.tobytes())
        return h.hexdigest()

    # ----- IO ------------------------------------------------------------ #
    def save(self, path) -> Path:
        """Persist as a self-contained JSON (data + class + params + provenance)."""
        path = Path(path)
        payload = {
            "schema": "openawsem.memory.v2",
            "class": type(self).__name__,
            "params": self.params,
            "provenance": self.provenance,
            "data": pd.DataFrame(self).to_dict(orient="list"),
        }
        with open(path, "w") as f:
            json.dump(payload, f, default=self._json_default)
        return path

    @classmethod
    def read(cls, path) -> "FragmentMemory":
        with open(path) as f:
            payload = json.load(f)
        klass = cls._class_named(payload.get("class"))
        frame = pd.DataFrame(payload["data"])
        for col, dtype in klass._DTYPES.items():
            if col in frame.columns and len(frame):
                frame[col] = frame[col].astype(dtype)
        obj = klass(frame)
        obj.attrs["params"] = payload.get("params", {})
        obj.attrs["provenance"] = payload.get("provenance", {})
        return obj

    @classmethod
    def _class_named(cls, name) -> type:
        """Resolve a stored class name to a FragmentMemory subclass.

        Search the whole subclass tree rooted at :class:`FragmentMemory` (not ``cls``) so
        ``read`` reconstructs the right class regardless of which class it is called on
        (e.g. ``MemoryTable.read`` of a MemoryTable payload).
        """
        def walk(klass):
            for sub in klass.__subclasses__():
                if sub.__name__ == name:
                    return sub
                found = walk(sub)
                if found is not None:
                    return found
            return None
        return walk(FragmentMemory) or MemoryWells

    # ----- coarse-grained Scene bridge ---------------------------------- #
    def attach_to_scene(self, scene):
        """Attach this memory to a molscene ``Scene`` as metadata so selecting/slicing
        the scene carries (and, once molscene reindexes resids, slides) the restraints."""
        scene._meta["fragment_memory"] = self
        return scene

    @staticmethod
    def from_scene(scene) -> "FragmentMemory":
        return scene._meta["fragment_memory"]

    # ----- internal helpers --------------------------------------------- #
    @staticmethod
    def _build_force(pairs, table, oa, *, k_fm, caOnly, forceGroup, rmin, rmax, dr, debug=False):
        try:
            from openmm import CustomCompoundBondForce, Discrete2DFunction
        except ModuleNotFoundError:  # pragma: no cover
            from simtk.openmm import CustomCompoundBondForce, Discrete2DFunction
        r_size = table.shape[1]
        max_r_index_1 = r_size - 2
        pair_to_index = {p: k for k, p in enumerate(pairs)}
        fm = CustomCompoundBondForce(2, f"-{k_fm}*((v2-v1)*r+v1*r_2-v2*r_1)/(r_2-r_1); \
                                    v1=frag_table(index, r_index_1);\
                                    v2=frag_table(index, r_index_2);\
                                    r_1={rmin}+{dr}*r_index_1;\
                                    r_2={rmin}+{dr}*r_index_2;\
                                    r_index_2=r_index_1+1;\
                                    r_index_1=min({max_r_index_1}, floor(r/{dr}));\
                                    r=distance(p1, p2);")
        ca = set(oa.ca)
        for (i, j) in pairs:
            if caOnly and ((i not in ca) or (j not in ca)):
                continue
            fm.addBond([i, j], [pair_to_index[(i, j)]])
        fm.addPerBondParameter("index")
        fm.addTabulatedFunction("frag_table", Discrete2DFunction(len(pairs), r_size, table.T.flatten()))
        if debug:
            x = np.arange(rmin, rmax, dr)
            pd.DataFrame(-k_fm * table, columns=x,
                         index=[f"{i}-{j}" for (i, j) in pairs]).to_csv("frag_table_debug.csv")
        if getattr(oa, "periodic_box", False):
            fm.setUsesPeriodicBoundaryConditions(True)
        fm.setForceGroup(forceGroup)
        return fm

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".meta.json")

    @classmethod
    def _save_cache(cls, path, pairs, table, rmin, rmax, dr, key=None):
        path = Path(path)
        pair_to_index = {p: k for k, p in enumerate(pairs)}
        with open(path, "wb") as f:
            pickle.dump((table, list(pairs), pair_to_index), f)
        with open(cls._meta_path(path), "w") as f:
            json.dump({"cache_key": key, "rmin": rmin, "rmax": rmax, "dr": dr,
                       "tabulator_version": TABULATOR_VERSION}, f, indent=2)

    @classmethod
    def _load_cache(cls, path):
        path = Path(path)
        with open(path, "rb") as f:
            table, interaction_list, _pair_to_index = pickle.load(f)
        rmin, rmax, dr = 0.0, 5.0, 0.01
        if cls._meta_path(path).exists():
            with open(cls._meta_path(path)) as f:
                meta = json.load(f)
            rmin, rmax, dr = meta.get("rmin", 0.0), meta.get("rmax", 5.0), meta.get("dr", 0.01)
        return [tuple(p) for p in interaction_list], np.asarray(table), (rmin, rmax, dr)

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


# =========================================================================== #
# parametric: Gaussian wells
# =========================================================================== #
class MemoryWells(FragmentMemory):
    """Fragment memory as a sum of Gaussian wells (rows = :data:`WELLS_COLUMNS`)."""

    _DTYPES = {"resid_i": "int64", "name_i": object, "resid_j": "int64", "name_j": object,
               "r_mean": "float64", "sigma": "float64", "weight": "float64"}

    @classmethod
    def empty(cls) -> "MemoryWells":
        return cls(pd.DataFrame({c: pd.Series(dtype=t) for c, t in cls._DTYPES.items()}))

    @classmethod
    def from_fragments(cls, fragments, *, min_seq_sep=3, max_seq_sep=9, well_width=0.1) -> "MemoryWells":
        """Build wells from aligned fragments.

        Each fragment is a DataFrame of CA/CB atoms with ``target_resid``, ``name``,
        ``x``/``y``/``z`` (nm) and an optional ``weight`` (defaults 1).  Every pair in the
        seq-sep window contributes one Gaussian (r_mean = distance, sigma =
        well_width*seq_sep**0.15).
        """
        parts = [cls._fragment_wells(f, min_seq_sep, max_seq_sep, well_width) for f in fragments]
        parts = [p for p in parts if len(p)]
        frame = pd.concat(parts, ignore_index=True) if parts else cls.empty()
        out = cls(frame)
        out.attrs["params"] = {"min_seq_sep": min_seq_sep, "max_seq_sep": max_seq_sep,
                               "well_width": well_width, "units": "nm", "cb_policy": "legacy"}
        return out

    @classmethod
    def from_frags_mem(cls, path, *, min_seq_sep=3, max_seq_sep=9, well_width=0.1) -> "MemoryWells":
        """Read a legacy ``frags.mem`` + its ``.gro`` library (gro via molscene)."""
        from openawsem.memory import _molscene
        path = str(path)
        base = Path(path).parent
        table = pd.read_csv(path, skiprows=4, sep=r"\s+", comment="#",
                            names=["location", "target_start", "fragment_start", "frag_len", "weight"])
        gro_cache: dict = {}
        fragments = []
        for row in table.itertuples(index=False):
            gro = str(base / row.location)
            if gro not in gro_cache:
                gro_cache[gro] = _molscene.read_awsem_gro(gro)
            atoms = gro_cache[gro]
            window = atoms[(atoms["resid"] >= row.fragment_start) &
                           (atoms["resid"] < row.fragment_start + row.frag_len)].copy()
            window["target_resid"] = window["resid"] - int(row.fragment_start) + int(row.target_start)
            window["weight"] = float(row.weight)
            fragments.append(window)
        out = cls.from_fragments(fragments, min_seq_sep=min_seq_sep, max_seq_sep=max_seq_sep,
                                 well_width=well_width)
        # record the fragments used (absolute gro paths so the memory re-reads from anywhere)
        used = [{"location": str((base / row.location).resolve()),
                 "target_start": int(row.target_start), "fragment_start": int(row.fragment_start),
                 "frag_len": int(row.frag_len), "weight": float(row.weight)}
                for row in table.itertuples(index=False)]
        out.provenance.update({"method": "legacy_frags_mem", "source": path,
                               "n_fragments": int(len(table)), "fragments": used})
        return out

    def to_frags_mem(self, path, gro_dir=None, target="query") -> Path:
        """Write a legacy ``frags.mem`` for the fragments behind this memory.

        Requires fragment provenance (``provenance['fragments']``), recorded by
        :meth:`from_frags_mem` and by the structure backends.  With ``gro_dir`` the gro
        files are copied there and referenced relatively; otherwise their stored
        (absolute) paths are written.
        """
        used = self.provenance.get("fragments")
        if not used:
            raise ValueError("no fragment provenance to write; build the memory from a frags.mem "
                             "or a structure backend that records fragments")
        path = Path(path)
        if gro_dir is not None:
            gro_dir = Path(gro_dir)
            gro_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(f"[Target]\n{target}\n\n[Memories]\n")
            for fr in used:
                location = fr["location"]
                if gro_dir is not None:
                    dest = gro_dir / Path(location).name
                    shutil.copy(location, dest)
                    location = str(dest)
                fh.write(f"{location} {fr['target_start']} {fr['fragment_start']} "
                         f"{fr['frag_len']} {fr['weight']}\n")
        return path

    def deduplicate(self) -> "MemoryWells":
        return type(self)(self.drop_duplicates(subset=WELLS_COLUMNS).reset_index(drop=True))

    def to_table(self, target, *, rmin=0.0, rmax=5.0, dr=0.01) -> "MemoryTable":
        """Alias for :meth:`tabulate` (Gaussian wells -> sampled MemoryTable)."""
        return self.tabulate(target, rmin=rmin, rmax=rmax, dr=dr)

    def _value_at(self, sub, r) -> float:
        rm = sub["r_mean"].to_numpy()
        sg = sub["sigma"].to_numpy()
        w = sub["weight"].to_numpy()
        return float((w * np.exp(-((r - rm) ** 2) / (2.0 * sg ** 2))).sum())

    def _tabulate_arrays(self, target, rmin, rmax, dr):
        amap = self.atom_index_map(target)
        r = np.arange(rmin, rmax, dr)
        if len(self) == 0:
            return [], np.zeros((0, len(r)))
        ai = np.array([amap.get((n, int(rid)), -1) for n, rid in zip(self["name_i"], self["resid_i"])])
        aj = np.array([amap.get((n, int(rid)), -1) for n, rid in zip(self["name_j"], self["resid_j"])])
        mask = (ai >= 0) & (aj >= 0)
        lo = np.minimum(ai[mask], aj[mask])
        hi = np.maximum(ai[mask], aj[mask])
        work = pd.DataFrame({"i": lo, "j": hi, "r_mean": self["r_mean"].to_numpy()[mask],
                            "sigma": self["sigma"].to_numpy()[mask], "weight": self["weight"].to_numpy()[mask]})
        pairs = sorted({(int(i), int(j)) for i, j in zip(work["i"], work["j"])})
        index = {p: k for k, p in enumerate(pairs)}
        table = np.zeros((len(pairs), len(r)))
        for (i, j), grp in work.groupby(["i", "j"], sort=False):
            rm = grp["r_mean"].to_numpy()[:, None]
            sg = grp["sigma"].to_numpy()[:, None]
            w = grp["weight"].to_numpy()[:, None]
            table[index[(int(i), int(j))]] = (w * np.exp(-((r[None, :] - rm) ** 2) / (2.0 * sg ** 2))).sum(axis=0)
        return pairs, table

    @classmethod
    def _fragment_wells(cls, fragment, min_seq_sep, max_seq_sep, well_width) -> pd.DataFrame:
        f = pd.DataFrame(fragment)
        n = len(f)
        if n < 2 or "target_resid" not in f.columns:
            return cls.empty()
        resids = f["target_resid"].to_numpy()
        names = f["name"].to_numpy(dtype=object)
        coords = f[["x", "y", "z"]].to_numpy(dtype=float)
        weight = float(f["weight"].iloc[0]) if "weight" in f.columns else 1.0
        # order by (resid, CA<CB) so triu pairs are canonical and seq_sep >= 0
        order = np.lexsort((np.array([ATOM_RANK.get(t, 99) for t in names]), resids))
        resids, names, coords = resids[order], names[order], coords[order]
        a, b = np.triu_indices(n, k=1)
        sep = resids[b] - resids[a]
        keep = (sep >= min_seq_sep) & (sep <= max_seq_sep)
        a, b, sep = a[keep], b[keep], sep[keep]
        if len(a) == 0:
            return cls.empty()
        return pd.DataFrame({
            "resid_i": resids[a].astype("int64"), "name_i": names[a],
            "resid_j": resids[b].astype("int64"), "name_j": names[b],
            "r_mean": np.linalg.norm(coords[a] - coords[b], axis=1),
            "sigma": well_width * sep.astype(float) ** 0.15,
            "weight": np.full(len(a), weight, dtype=float),
        })


# =========================================================================== #
# non-parametric: sampled table
# =========================================================================== #
class MemoryTable(FragmentMemory):
    """Fragment memory as a sampled potential, long form (rows = :data:`TABLE_COLUMNS`)."""

    _DTYPES = {"resid_i": "int64", "name_i": object, "resid_j": "int64", "name_j": object,
               "r": "float64", "energy": "float64"}

    @classmethod
    def empty(cls) -> "MemoryTable":
        return cls(pd.DataFrame({c: pd.Series(dtype=t) for c, t in cls._DTYPES.items()}))

    def _value_at(self, sub, r) -> float:
        rr = sub["r"].to_numpy()
        order = np.argsort(rr)
        return float(np.interp(r, rr[order], sub["energy"].to_numpy()[order]))

    def _tabulate_arrays(self, target, rmin, rmax, dr):
        amap = self.atom_index_map(target)
        r = np.arange(rmin, rmax, dr)
        if len(self) == 0:
            return [], np.zeros((0, len(r)))
        ai = np.array([amap.get((n, int(rid)), -1) for n, rid in zip(self["name_i"], self["resid_i"])])
        aj = np.array([amap.get((n, int(rid)), -1) for n, rid in zip(self["name_j"], self["resid_j"])])
        mask = (ai >= 0) & (aj >= 0)
        lo = np.minimum(ai[mask], aj[mask])
        hi = np.maximum(ai[mask], aj[mask])
        work = pd.DataFrame({"i": lo, "j": hi, "r": self["r"].to_numpy()[mask],
                            "energy": self["energy"].to_numpy()[mask]})
        pairs = sorted({(int(i), int(j)) for i, j in zip(work["i"], work["j"])})
        index = {p: k for k, p in enumerate(pairs)}
        table = np.zeros((len(pairs), len(r)))
        for (i, j), grp in work.groupby(["i", "j"], sort=False):
            order = np.argsort(grp["r"].to_numpy())
            table[index[(int(i), int(j))]] = np.interp(r, grp["r"].to_numpy()[order], grp["energy"].to_numpy()[order])
        return pairs, table
