"""``fragment_db`` backend: database-averaged statistics -> a sampled distance memory.

Instead of searching for explicit structural fragments, this backend looks up, for each
target residue pair, the database-averaged inter-residue distance density conditioned on the
two residue identities, the sequence separation and the atom-pair type.  Those densities are
a precomputed tensor (``tensors_openawsem.npz``): one Gaussian-smoothed, unit-integral curve
per ``(atom_pair, sep, aa_i, aa_j)`` cell on a fixed distance grid, with the same width law
(``sigma = 0.1 * sep**0.15`` nm) the conventional Gaussian-well memory uses.  Each looked-up
curve is emitted verbatim as one pair's sampled potential, so the result is a
:class:`MemoryTable` that flows through the same tabulate/OpenMM path as ``local_blast``.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from openawsem.memory.memory import MemoryTable
from openawsem.memory.generators.base import FragmentBackend, Registry

logger = logging.getLogger(__name__)

#: env var holding a default tensor database path (so library code carries no user path).
DEFAULT_DB_ENV = "OPENAWSEM_FRAGMENT_DB"


@Registry.register("fragment_db")
class FragmentDB(FragmentBackend):
    """Database-averaged statistics -> a :class:`MemoryTable` (sampled distance potential).

    Parameters
    ----------
    database:
        Path to a ``tensors_openawsem.npz`` (keys ``local_density`` [atom_pair, sep, aa_i,
        aa_j, r], ``local_grid`` [sep, r] in Angstrom, ``local_count``, ``aa``,
        ``atom_pairs``, ``seps``).  Falls back to ``$OPENAWSEM_FRAGMENT_DB``.
    min_seq_sep, max_seq_sep:
        Sequence-separation window to emit (matches the conventional memory's 3..9).
    weighting:
        How a pair's unit-integral density is scaled into a well.

        * ``"multiplicity"`` -- reproduce the conventional fragment-memory force.  The
          conventional table for a pair is a sum of ``N(i,j) = n_mem * n_windows(i,j)``
          unit-height Gaussians, where ``n_windows`` is the number of overlapping
          fragment windows containing the pair (``= fragment_length - sep`` in the chain
          interior, fewer at the tails and for large ``sep``).  If the homolog fragments
          sample the same distribution the tensor averages, then
          ``E[T_conv(r)] = N(i,j) * sigma * sqrt(2*pi) * f(r)``, so each curve is weighted by
          ``N(i,j) * sigma_A(sep) * sqrt(2*pi)`` (``sigma_A`` from the tensor's smoothing law).
          Pairs with ``sep >= fragment_length`` (``n_windows = 0``) are dropped, matching
          conventional coverage.
        * ``"uniform"`` -- the simplified model: every pair's curve is scaled equally (by
          ``scale``), so each multi-Gaussian contributes the same force regardless of how
          many fragments would have supported it conventionally.
    scale:
        Global residual multiplier on every curve.  For ``"uniform"`` it sets the overall
        well depth (e.g. ~168 to match a target's median conventional depth).  For
        ``"multiplicity"`` the absolute depth is already set by ``N(i,j)*sigma*sqrt(2pi)``;
        leave ``scale=1`` (use it only to fine-tune).
    n_mem, fragment_length, well_width:
        Conventional-memory parameters used by ``"multiplicity"`` (defaults match
        ``local_blast``: 20 memories per 9-residue window; ``well_width`` 0.1 nm is the
        conventional Gaussian width ``sigma = well_width*sep^0.15``).  ``well_width`` sets the
        area-match factor and is a property of the potential, not of the tensor smoothing, so
        the same value is correct for any tensor (e.g. the finely-smoothed ``tensors_fine``).
    weight_by_count:
        If true, additionally weight each curve by ``log1p(count)/log1p(count_max)`` so
        better-sampled cells give (relatively) deeper wells.  Off by default.
    """

    _ATOM_PAIRS = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))

    def __init__(self, *, database=None, min_seq_sep=3, max_seq_sep=9,
                 weighting="uniform", scale=1.0, n_mem=20, fragment_length=9,
                 well_width=0.1, weight_by_count=False):
        self.database = str(database or os.environ.get(DEFAULT_DB_ENV) or "")
        if not self.database:
            raise ValueError("fragment_db needs a tensor database: pass database=... or set "
                             f"${DEFAULT_DB_ENV}")
        if weighting not in ("uniform", "multiplicity"):
            raise ValueError(f"weighting must be 'uniform' or 'multiplicity', got {weighting!r}")
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.weighting = weighting
        self.scale = float(scale)
        self.n_mem = int(n_mem)
        self.fragment_length = int(fragment_length)
        self.well_width = float(well_width)
        self.weight_by_count = bool(weight_by_count)
        self._tensor = None

    # ----- tensor access ------------------------------------------------- #
    @property
    def tensor(self) -> dict:
        if self._tensor is None:
            self._tensor = self._load(self.database)
        return self._tensor

    @staticmethod
    def _load(path) -> dict:
        data = np.load(path, allow_pickle=True)
        aa = [str(x) for x in data["aa"].tolist()]
        atom_pairs = [str(x) for x in data["atom_pairs"].tolist()]
        seps = [int(x) for x in data["seps"].tolist()]
        return {
            "local_density": data["local_density"],
            "local_count": data["local_count"],
            "local_grid": data["local_grid"],
            "aa_index": {a: i for i, a in enumerate(aa)},
            "ap_index": {a: i for i, a in enumerate(atom_pairs)},
            "sep_index": {s: i for i, s in enumerate(seps)},
            "sigma_coeff": float(data["sigma_coeff"]),
            "sigma_exp": float(data["sigma_exp"]),
            "sigma_scale": float(data["sigma_scale"]),
        }

    @staticmethod
    def _n_windows(pos_i, pos_j, n, length) -> int:
        """Number of length-``length`` windows (1-based starts within a chain of ``n``
        residues) that contain both 1-based positions ``pos_i < pos_j``.  Interior value is
        ``length - (pos_j - pos_i)``; the chain ends and ``sep >= length`` reduce it."""
        first = max(1, pos_j - length + 1)
        last = min(pos_i, n - length + 1)
        return max(0, last - first + 1)

    @staticmethod
    def _as_records(sequence):
        """One record per chain; a plain string is a single record (cf. ``local_blast``)."""
        return [sequence] if isinstance(sequence, str) else list(sequence)

    def _pair_weight(self, sep, pos_i, pos_j, n) -> float:
        """Per-pair scalar applied to the density curve (see ``weighting``).  Returns 0 to
        drop a pair (multiplicity with no overlapping window)."""
        if self.weighting == "uniform":
            return self.scale
        n_windows = self._n_windows(pos_i, pos_j, n, self.fragment_length)
        if n_windows <= 0:
            return 0.0
        # Area-match the unit-integral (over Angstrom) density to the conventional sum of
        # N = n_mem*n_windows unit-height Gaussians of width sigma_conv = well_width*sep^0.15.
        # sigma_conv is the *potential's* width (nm); in Angstrom it is 10*well_width*sep^0.15.
        sigma_conv_a = 10.0 * self.well_width * sep ** 0.15
        return self.scale * self.n_mem * n_windows * sigma_conv_a * np.sqrt(2.0 * np.pi)

    # ----- generation ---------------------------------------------------- #
    def generate(self, sequence) -> MemoryTable:
        t = self.tensor
        density, count, grid = t["local_density"], t["local_count"], t["local_grid"]
        aa_index, ap_index, sep_index = t["aa_index"], t["ap_index"], t["sep_index"]
        ap_lookup = [(ni, nj, ap_index[f"{ni}-{nj}"]) for ni, nj in self._ATOM_PAIRS]
        log_count_max = np.log1p(float(count.max())) if self.weight_by_count else 1.0

        cols: dict = {k: [] for k in ("resid_i", "name_i", "resid_j", "name_j", "r", "energy")}
        n_skipped = 0
        base = 0
        for record in self._as_records(sequence):
            n = len(record)
            codes = [aa_index.get(c, -1) for c in record]
            for i in range(n):
                ci = codes[i]
                for sep in range(self.min_seq_sep, self.max_seq_sep + 1):
                    j = i + sep
                    if j >= n:
                        break
                    cj = codes[j]
                    sidx = sep_index.get(sep)
                    if sidx is None:
                        continue
                    if ci < 0 or cj < 0:  # non-standard residue: no statistics for this pair
                        n_skipped += 1
                        continue
                    weight = self._pair_weight(sep, i + 1, j + 1, n)
                    if weight == 0.0:     # multiplicity: pair not in any fragment window
                        continue
                    grid_nm = grid[sidx] / 10.0
                    resi, resj = base + i + 1, base + j + 1
                    for ni, nj, apidx in ap_lookup:
                        if (ni == "CB" and record[i] == "G") or (nj == "CB" and record[j] == "G"):
                            continue
                        curve = density[apidx, sidx, ci, cj].astype(np.float64) * weight
                        if self.weight_by_count:
                            curve = curve * (np.log1p(float(count[apidx, sidx, ci, cj])) / log_count_max)
                        m = curve.shape[0]
                        cols["resid_i"].append(np.full(m, resi, dtype=np.int64))
                        cols["name_i"].append(np.full(m, ni, dtype=object))
                        cols["resid_j"].append(np.full(m, resj, dtype=np.int64))
                        cols["name_j"].append(np.full(m, nj, dtype=object))
                        cols["r"].append(grid_nm)
                        cols["energy"].append(curve)
            base += n

        n_pairs = len(cols["energy"])
        if n_pairs:
            frame = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
            memory = MemoryTable(frame)
        else:
            memory = MemoryTable.empty()
        memory.attrs["params"] = {
            "min_seq_sep": self.min_seq_sep, "max_seq_sep": self.max_seq_sep,
            "weighting": self.weighting, "scale": self.scale, "n_mem": self.n_mem,
            "fragment_length": self.fragment_length,
            "weight_by_count": self.weight_by_count, "units": "nm",
        }
        memory.provenance.update({"method": "fragment_db", "database": self.database,
                                  "n_pairs": int(n_pairs), "n_skipped": int(n_skipped)})
        return memory
