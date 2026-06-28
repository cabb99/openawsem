"""``local_db`` backend: real fragments retrieved from the local residue database.

Like ``local_blast`` (PSI-BLAST + structure fetch) but single-sequence and fetch-free: for
each sliding window of the target, a BLOSUM62 k-mer search over a residue-centric database
(:class:`openawsem.memory.retrieval.FragmentDatabase`) returns the most sequence-similar real
fragments, and their **stored** pairwise distances become Gaussian wells.  This places each
well on the distances of actual homologous fragments (target-specific geometry), unlike
``fragment_db`` which uses the database-average distance per residue-pair type.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from openawsem.memory.memory import MemoryWells, WELLS_COLUMNS
from openawsem.memory.retrieval.blosum import encode_sequence
from openawsem.memory.retrieval.index import ATOM_PAIRS
from openawsem.memory.generators import homology
from openawsem.memory.generators.base import FragmentBackend, Registry

logger = logging.getLogger(__name__)

_CACB_PAIRS = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))


def accumulate_soft_wells(cols, idx, record, start, base, rows, weights, *,
                          fragment_length, min_seq_sep, max_seq_sep, well_width,
                          atom_pairs=_CACB_PAIRS):
    """Append the score-weighted CA/CB wells of one window to ``cols`` (shared by ``local_db``
    soft and the ``ml`` backend).

    ``rows`` are database fragment **start** rows and ``weights`` their per-candidate weights;
    pointwise distances are gathered from ``idx.dist`` (Angstrom -> nm), the GLY/CB pairs are
    dropped using the (possibly mutated) query ``record``, and ``sigma = well_width*sep**0.15``.
    """
    rows = np.asarray(rows)
    apidx = {p: ATOM_PAIRS.index(p) for p in atom_pairs}
    for a in range(fragment_length):
        for b in range(a + 1, fragment_length):
            sep = b - a
            if not (min_seq_sep <= sep <= max_seq_sep) or sep not in idx._sep_index:
                continue
            si, sigma = idx._sep_index[sep], well_width * sep ** 0.15
            resid_i, resid_j = base + start + a + 1, base + start + b + 1
            gly_i, gly_j = record[start + a] == "G", record[start + b] == "G"
            for name_i, name_j in atom_pairs:
                if (name_i == "CB" and gly_i) or (name_j == "CB" and gly_j):
                    continue
                d = idx.dist[rows + a, si, apidx[(name_i, name_j)]] / 10.0
                ok = ~np.isnan(d)
                m = int(ok.sum())
                if not m:
                    continue
                cols["resid_i"].extend([resid_i] * m); cols["name_i"].extend([name_i] * m)
                cols["resid_j"].extend([resid_j] * m); cols["name_j"].extend([name_j] * m)
                cols["r_mean"].extend(d[ok].tolist()); cols["sigma"].extend([sigma] * m)
                cols["weight"].extend(weights[ok].tolist())


def emit_mixture_wells(cols, record, start, base, *, fragment_length, min_seq_sep, max_seq_sep,
                       pair_a, pair_b, pair_sep, pi, mu, sigma, n_mem, atom_pairs=_CACB_PAIRS):
    """Append the wells of a *predicted* Gaussian mixture for one window (the ``ml_gen``/v4 path).

    Mirror of :func:`accumulate_soft_wells`, but the per-component ``(weight, r_mean, sigma)`` are
    *predicted* (``weight_k = n_mem*pi_k``) rather than gathered from a database fragment.
    ``pi``/``mu``/``sigma`` are ``(P, C, K)`` aligned to ``(pair_a, pair_b, pair_sep)`` and
    ``atom_pairs``; the GLY/CB drop and resid bookkeeping are identical, and non-finite or
    zero-weight components are skipped.
    """
    P = len(pair_a)
    for p in range(P):
        a, b, sep = int(pair_a[p]), int(pair_b[p]), int(pair_sep[p])
        if not (min_seq_sep <= sep <= max_seq_sep):
            continue
        resid_i, resid_j = base + start + a + 1, base + start + b + 1
        gly_i, gly_j = record[start + a] == "G", record[start + b] == "G"
        for c, (name_i, name_j) in enumerate(atom_pairs):
            if (name_i == "CB" and gly_i) or (name_j == "CB" and gly_j):
                continue
            for k in range(pi.shape[-1]):
                w, rm, sg = float(n_mem * pi[p, c, k]), float(mu[p, c, k]), float(sigma[p, c, k])
                if not (np.isfinite(rm) and sg > 0 and w > 0):
                    continue
                cols["resid_i"].append(resid_i); cols["name_i"].append(name_i)
                cols["resid_j"].append(resid_j); cols["name_j"].append(name_j)
                cols["r_mean"].append(rm); cols["sigma"].append(sg); cols["weight"].append(w)


@Registry.register("local_db")
class LocalDB(FragmentBackend):
    """Database-retrieved real fragments -> a :class:`MemoryWells`.

    Parameters
    ----------
    database:
        Path to a ``fragdb`` output directory containing ``residues.csv[.gz]``.
    n_mem, fragment_length:
        Fragments kept per window and the sliding-window length (defaults match
        ``local_blast``: top-20 hits of 9-mers).
    min_seq_sep, max_seq_sep, well_width, weight:
        Same well parameters as ``local_blast`` (``sigma = well_width*sep**0.15``).
    weighting, soft_temp, soft_pool, soft_seed_word:
        ``"hard"`` (default) keeps the top ``n_mem`` fragments per window, each weighted equally.
        ``"soft"`` instead score-weights a wider candidate pool: every well of the top
        ``soft_pool`` candidates contributes ``softmax(score/soft_temp)`` (summing to ``n_mem`` so
        the depth matches the hard memory).  ``soft_temp`` controls peakedness: the default ``2.0``
        folds 1r69 to ``Q=0.555`` (the best of hard 0.517, conventional 0.515, and the softer temp=3
        0.497) and keeps the softmax narrow enough that the mutation-ΔE pool stays small.  The soft
        memory is tie-free, which makes the mutation-ΔE engine (:meth:`mutation_engine`) well defined.
        ``soft`` requires a :class:`FragmentIndex` (the pointwise distance store).
    brain_damage, cutoff_identical, homolog_evalue, homolog_database, blast_exe, num_threads,
    homolog_oversample:
        Homolog handling, identical flags/semantics to ``local_blast`` (0 all | 1 exclude
        homologs | 2 homologs except self | 0.5 self only).  Homologs are found with one
        full-sequence PSI-BLAST against ``homolog_database`` (default ``openawsem.data_path.blast``,
        since this method's own ``database`` is the residue DB, not a BLAST DB) and fragments
        are filtered by ``pdb_id``; ``homolog_oversample`` widens the per-window search so
        ``n_mem`` survives the exclusion.
    """

    _ATOM_PAIRS = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))
    _db_cache: dict = {}

    def __init__(self, database, *, n_mem=20, fragment_length=9, min_seq_sep=3,
                 max_seq_sep=9, well_width=0.1, weight=1.0, weighting="hard", soft_temp=2.0,
                 soft_pool=80, soft_seed_word=2, brain_damage=0,
                 cutoff_identical=90, homolog_evalue=0.005, homolog_database=None,
                 blast_exe="psiblast", num_threads=1, homolog_oversample=10, seeded=True):
        self.database = str(database)
        self.seeded = bool(seeded)
        self.n_mem = int(n_mem)
        self.fragment_length = int(fragment_length)
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.well_width = float(well_width)
        self.weight = float(weight)
        if weighting not in ("hard", "soft"):
            raise ValueError(f"weighting must be 'hard' or 'soft', got {weighting!r}")
        self.weighting = weighting
        self.soft_temp = float(soft_temp)
        self.soft_pool = int(soft_pool)
        self.soft_seed_word = int(soft_seed_word)
        self.brain_damage = brain_damage
        self.cutoff_identical = cutoff_identical
        self.homolog_evalue = homolog_evalue
        self.homolog_database = homolog_database
        self.blast_exe = blast_exe
        self.num_threads = num_threads
        self.homolog_oversample = int(homolog_oversample)

    @property
    def db(self):
        if self.database not in LocalDB._db_cache:
            from openawsem.memory.retrieval import FragmentDatabase, FragmentIndex
            if FragmentIndex.exists(self.database):
                logger.info("loading prebuilt fragment index for %s", self.database)
                LocalDB._db_cache[self.database] = FragmentIndex.load(self.database)
            else:
                LocalDB._db_cache[self.database] = FragmentDatabase.load(self.database)
        return LocalDB._db_cache[self.database]

    @staticmethod
    def _as_records(sequence):
        """One record per chain; a plain string is a single record (cf. ``local_blast``)."""
        return [sequence] if isinstance(sequence, str) else list(sequence)

    def _resolve_homolog_database(self):
        if self.homolog_database:
            return str(self.homolog_database)
        import openawsem
        return str(openawsem.data_path.blast)

    def _homologs(self, sequence) -> dict:
        """Target's homologs ``{pdb_id: max pident}`` (empty when brain_damage off)."""
        if not self.brain_damage:
            return {}
        return homology.blast_homologs(self._as_records(sequence),
                                       database=self._resolve_homolog_database(),
                                       blast_exe=self.blast_exe, homolog_evalue=self.homolog_evalue,
                                       num_threads=self.num_threads, brain_damage=self.brain_damage)

    def _search(self, window, top):
        """Search the DB for ``window``; use the seed index when available (FragmentIndex)."""
        from openawsem.memory.retrieval import FragmentIndex
        if self.seeded and isinstance(self.db, FragmentIndex):
            return self.db.search(window, L=self.fragment_length, top=top, seeded=True)
        return self.db.search(window, L=self.fragment_length, top=top)

    def _window_hits(self, window, homologs):
        """Search a window, apply brain_damage by ``pdb_id``, keep the top ``n_mem`` survivors."""
        top = self.n_mem * self.homolog_oversample if self.brain_damage else self.n_mem
        hits = self._search(window, top)
        if self.brain_damage and len(hits):
            hits = hits.assign(pdb_id=[str(c).split("_")[0] for c in hits["chain_uid"]])
            hits = homology.apply_brain_damage(hits, homologs, brain_damage=self.brain_damage,
                                               cutoff_identical=self.cutoff_identical)
        return hits.head(self.n_mem)

    def _window_wells(self, start, base, window, homologs, cols) -> int:
        """Append the wells of one window (query=``window``) to ``cols``; return n_hits used."""
        n_hits = 0
        for hit in self._window_hits(window, homologs).itertuples(index=False):
            frag = self.db.get_fragment(hit.chain_uid, int(hit.start), self.fragment_length)
            if len(frag) and self._accumulate(cols, frag, window, start, base, int(hit.start)):
                n_hits += 1
        return n_hits

    def generate(self, sequence) -> MemoryWells:
        if self.weighting == "soft":
            return self._generate_soft(sequence)
        L = self.fragment_length
        homologs = self._homologs(sequence)
        cols: dict = {c: [] for c in WELLS_COLUMNS}
        n_windows = n_hits = 0
        base = 0
        for record in self._as_records(sequence):
            n = len(record)
            for start in range(n - L + 1):
                n_hits += self._window_wells(start, base, record[start:start + L], homologs, cols)
                n_windows += 1
            base += n

        frame = pd.DataFrame(cols)
        memory = MemoryWells(frame) if len(frame) else MemoryWells.empty()
        memory.attrs["params"] = {"min_seq_sep": self.min_seq_sep, "max_seq_sep": self.max_seq_sep,
                                  "well_width": self.well_width, "units": "nm", "cb_policy": "legacy"}
        memory.provenance.update({"method": "local_db", "database": self.database,
                                  "brain_damage": self.brain_damage,
                                  "n_windows": int(n_windows), "n_hits": int(n_hits)})
        return memory

    def _keep_rows(self, sequence):
        """Boolean mask over index rows for brain_damage (drop homolog/self PDBs), or ``None``."""
        if not self.brain_damage:
            return None
        keep = homology.keep_predicate(self._homologs(sequence), brain_damage=self.brain_damage,
                                       cutoff_identical=self.cutoff_identical)
        if keep is None:
            return None
        idx = self.db
        chain_keep = np.array([keep(str(name).split("_")[0]) for name in idx.chain_names])
        return chain_keep[np.asarray(idx.chain_code)]

    def _generate_soft(self, sequence) -> MemoryWells:
        """Score-weighted memory: per window, ``softmax(score/soft_temp)`` over the top
        ``soft_pool`` candidates (scaled to ``n_mem``), wells gathered pointwise from the index."""
        from openawsem.memory.retrieval import FragmentIndex
        idx = self.db
        if not isinstance(idx, FragmentIndex):
            raise TypeError("weighting='soft' needs a FragmentIndex; build one with "
                            "`python -m openawsem.memory.retrieval.index <database>`")
        L = self.fragment_length
        keep_rows = self._keep_rows(sequence)
        cols: dict = {c: [] for c in WELLS_COLUMNS}
        n_windows = 0
        base = 0
        for record in self._as_records(sequence):
            n = len(record)
            for start in range(n - L + 1):
                q = encode_sequence(record[start:start + L])
                rows, _, scores = idx.topk_pool(q, L, self.soft_pool,
                                                seed_word=self.soft_seed_word, keep_rows=keep_rows)
                w = np.exp((scores - scores.max()) / self.soft_temp)
                w = self.n_mem * w / w.sum()
                self._accumulate_soft(cols, record, start, base, rows, w)
                n_windows += 1
            base += n

        frame = pd.DataFrame(cols)
        memory = MemoryWells(frame) if len(frame) else MemoryWells.empty()
        memory.attrs["params"] = {"min_seq_sep": self.min_seq_sep, "max_seq_sep": self.max_seq_sep,
                                  "well_width": self.well_width, "units": "nm", "cb_policy": "legacy"}
        memory.provenance.update({"method": "local_db_soft", "database": self.database,
                                  "brain_damage": self.brain_damage, "soft_temp": self.soft_temp,
                                  "soft_pool": self.soft_pool, "n_windows": int(n_windows)})
        return memory

    def _accumulate_soft(self, cols, record, start, base, rows, weights):
        """Append the score-weighted wells of one window (candidate ``rows`` with ``weights``)."""
        accumulate_soft_wells(cols, self.db, record, start, base, rows, weights,
                              fragment_length=self.fragment_length, min_seq_sep=self.min_seq_sep,
                              max_seq_sep=self.max_seq_sep, well_width=self.well_width,
                              atom_pairs=self._ATOM_PAIRS)

    def mutation_engine(self, sequence, structure, *, method="fulldb", k_fm=0.04184,
                        cache_commit=True):
        """A reusable soft-memory mutation-ΔE engine on a fixed ``structure``: ``engine.dE(position,
        new_aa)`` for a trial, ``engine.commit(position, new_aa)`` to advance a Monte-Carlo walk.

        method:
          ``'fulldb'`` (default, EXACT) -- evaluates the whole database.  O(N) precompute (needs a
            :class:`FragmentIndex`; ~minutes and GBs of RAM on a large DB), then ~tens of thousands of
            exact trial-ΔE/s; ``commit`` advances a walk in O(N) (~seconds/accept).  The faithful-MC
            choice.  A long MC is commit-bound; a smaller curated DB cuts the precompute/commit linearly.
          ``'anchored'`` (fast, APPROXIMATE) -- a fixed top-``soft_pool`` pool, no precompute, but it
            cannot see fragments outside the pool, so it distorts the marginal moves of a walk.  A fallback
            when the exact precompute/commit is unaffordable, not a faithful sampler.
          ``'faithful'`` (exact, slow, no precompute) -- re-search the mutant's windows; superseded by
            ``'fulldb'``.
        ``cache_commit`` applies to ``'fulldb'`` only (``cache_commit=False`` skips the walk Gaussian
        cache for a scan-only run)."""
        from openawsem.memory.mutation import MutationEnergy, FullDBMutationEnergy
        if method == "fulldb":
            return FullDBMutationEnergy(self, sequence, structure, k_fm=k_fm, cache_commit=cache_commit)
        if method in ("anchored", "faithful"):
            return MutationEnergy(self, sequence, structure, method=method, k_fm=k_fm)
        raise ValueError(f"method must be 'fulldb', 'anchored', or 'faithful', got {method!r}")

    def mutation_energy(self, sequence, structure, position, new_aa, *, method="faithful",
                        k_fm=0.04184) -> float:
        """One-shot soft mutation ΔE on a fixed ``structure``.  Defaults to ``method='faithful'`` (exact,
        no precompute -- best for a single mutation).  For many mutations or a walk use
        :meth:`mutation_engine` (``method='fulldb'``) to amortize the precompute; ``'anchored'`` is the
        fast-but-approximate fallback."""
        return self.mutation_engine(sequence, structure, method=method, k_fm=k_fm).dE(position, new_aa)

    def _accumulate(self, cols, frag, window, start, base, hit_start) -> bool:
        """Append wells for one retrieved fragment; return True if it contributed any.

        ``window`` is the (possibly mutated) query letters of this window; the GLY test uses
        the query identity so a mutation to/from G drops/adds the right CB pairs.
        """
        contributed = False
        for row in frag.itertuples(index=False):
            sep = int(row.sep)
            if not (self.min_seq_sep <= sep <= self.max_seq_sep):
                continue
            a = int(row.internal_i) - hit_start
            b = int(row.internal_j) - hit_start
            resid_i = base + start + a + 1
            resid_j = base + start + b + 1
            gly_i = window[a] == "G"
            gly_j = window[b] == "G"
            sigma = self.well_width * sep ** 0.15
            for name_i, name_j in self._ATOM_PAIRS:
                if (name_i == "CB" and gly_i) or (name_j == "CB" and gly_j):
                    continue
                d = getattr(row, f"d_{name_i}_{name_j}")
                if d is None or (isinstance(d, float) and math.isnan(d)) or d == "":
                    continue
                cols["resid_i"].append(resid_i)
                cols["name_i"].append(name_i)
                cols["resid_j"].append(resid_j)
                cols["name_j"].append(name_j)
                cols["r_mean"].append(float(d) / 10.0)   # Angstrom -> nm
                cols["sigma"].append(sigma)
                cols["weight"].append(self.weight)
                contributed = True
        return contributed
