"""``local_blast`` backend: PSI/BLAST a sliding window against a local database.

Requires a BLAST executable, a local database, and (for fetch) network access, so it is
exercised via gated/e2e tests; the deterministic hit handling it relies on lives in
:mod:`openawsem.memory.generators.selection` (unit-tested there).
"""
from __future__ import annotations

import logging
import subprocess
import tempfile

import pandas as pd

from openawsem.memory.generators.base import Registry, StructureBackend

logger = logging.getLogger(__name__)


@Registry.register("local_blast")
class LocalBlast(StructureBackend):
    """Sliding-window PSI/BLAST -> deterministic hits; structures fetched via molscene."""

    _OUTFMT = "7 sseqid qstart qend sstart send qseq sseq length gaps bitscore evalue"

    def __init__(self, database, *, n_mem=20, fragment_length=9, evalue=10000.0,
                 min_seq_sep=3, max_seq_sep=9, well_width=0.1, weight=1.0,
                 blast_exe="psiblast", num_iterations=5, max_gaps=0, num_threads=1,
                 cif_dir=None, download_format="cif", brain_damage=0, cutoff_identical=90,
                 homolog_evalue=0.005):
        self.database = str(database)
        self.n_mem = n_mem
        self.fragment_length = fragment_length
        self.evalue = evalue
        self.min_seq_sep = min_seq_sep
        self.max_seq_sep = max_seq_sep
        self.well_width = well_width
        self.weight = weight
        self.blast_exe = blast_exe
        self.num_iterations = num_iterations
        self.max_gaps = max_gaps
        self.num_threads = num_threads
        self.cif_dir = cif_dir            # local mmCIF database (wwPDB divided layout)
        self.download_format = download_format  # 'cif' (RCSB) or 'pdb' (PDBFixer) when not local
        # homolog handling (brain_damage): 0 all hits | 1 exclude homologs |
        # 2 homologs except self(>cutoff) | 0.5 self only(>cutoff)
        self.brain_damage = brain_damage
        self.cutoff_identical = cutoff_identical
        self.homolog_evalue = homolog_evalue

    def search(self, sequence) -> pd.DataFrame:
        windows = len(sequence) - self.fragment_length + 1
        if windows < 1:
            raise ValueError(f"sequence shorter than fragment_length={self.fragment_length}")
        frames = []
        for start in range(windows):
            hits = self._psiblast(sequence[start:start + self.fragment_length])
            if hits is None or len(hits) == 0:
                continue
            # offset the (1-based) alignment coordinates into the full target sequence
            hits["query_start"] = hits["qstart"] + start
            hits["query_end"] = hits["qend"] + start
            hits["subject_start"] = hits["sstart"]
            hits["subject_end"] = hits["send"]
            frames.append(hits)
        if not frames:
            return pd.DataFrame(columns=["pdb_id", "chain", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"])
        return pd.concat(frames, ignore_index=True)

    def fetch(self, hit):
        from openawsem.memory.structures import fetch_structure
        return fetch_structure(hit["pdb_id"], cif_dir=self.cif_dir, download=self.download_format)

    # ----- homolog handling (brain_damage) ------------------------------ #
    def _filter_hits(self, hits, sequence):
        if not self.brain_damage or len(hits) == 0:
            return hits
        homologs = self._homologs(sequence)

        def is_homolog(pid):
            return pid in homologs

        def is_self(pid):
            return homologs.get(pid, 0.0) > self.cutoff_identical

        if self.brain_damage == 1:        # exclude all homologs
            keep = lambda pid: not is_homolog(pid)
        elif self.brain_damage == 2:      # homologs only, but not self (>cutoff)
            keep = lambda pid: is_homolog(pid) and not is_self(pid)
        elif self.brain_damage == 0.5:    # self only (>cutoff)
            keep = is_self
        else:
            return hits
        mask = hits["pdb_id"].map(keep)
        logger.info("brain_damage=%s: kept %d/%d hits (%d homologs found, cutoff %s%%)",
                    self.brain_damage, int(mask.sum()), len(hits), len(homologs), self.cutoff_identical)
        return hits[mask].reset_index(drop=True)

    def _homologs(self, sequence) -> dict:
        """Full-sequence BLAST -> ``{pdb_id: max percent identity}`` (the target's homologs)."""
        df = self._blast_full_sequence(sequence)
        homologs: dict = {}
        for pid, pident in zip(df["pdb_id"], df["pident"]):
            homologs[pid] = max(homologs.get(pid, 0.0), float(pident))
        return homologs

    def _blast_full_sequence(self, sequence) -> pd.DataFrame:
        fields = "sseqid pident evalue bitscore"
        with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=True) as fh:
            fh.write(f">query\n{sequence}\n")
            fh.flush()
            cmd = [self.blast_exe, "-query", fh.name, "-db", self.database,
                   "-num_iterations", "1", "-word_size", "3", "-evalue", str(self.homolog_evalue),
                   "-num_threads", str(self.num_threads), "-outfmt", f"6 {fields}"]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                raise RuntimeError(
                    f"homolog BLAST failed; refusing to apply brain_damage={self.brain_damage} "
                    f"on an empty homolog set (that would silently keep/drop the wrong hits)"
                ) from exc
        lines = [ln for ln in result.stdout.splitlines() if ln.strip() and not ln.startswith("Search has")]
        df = pd.DataFrame([ln.split("\t") for ln in lines], columns=fields.split())
        if len(df):
            df["pident"] = pd.to_numeric(df["pident"], errors="coerce")
            pdb_id, chain = zip(*df["sseqid"].map(self._parse_sseqid))
            return df.assign(pdb_id=list(pdb_id), chain=list(chain))
        return df.assign(pdb_id=[], chain=[])

    def _psiblast(self, query) -> pd.DataFrame:
        with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=True) as fh:
            fh.write(f">query\n{query}\n")
            fh.flush()
            cmd = [self.blast_exe, "-query", fh.name, "-db", self.database,
                   "-num_iterations", str(self.num_iterations), "-comp_based_stats", "0",
                   "-word_size", "2", "-threshold", "9", "-window_size", "0",
                   "-matrix", "BLOSUM62", "-evalue", str(self.evalue),
                   "-num_threads", str(self.num_threads), "-outfmt", self._OUTFMT]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                logger.error("BLAST failed: %s", exc)
                return None
        return self._parse_blast(result.stdout)

    @staticmethod
    def _final_iteration_rows(stdout):
        """Tab-separated data rows of the *final* PSI-BLAST round only.

        With ``-outfmt 7`` each round is preceded by a ``# Iteration: N`` comment, so the
        last round is the data after the last such marker.  Earlier rounds are intermediate
        profiles, not the converged result, and must not be mixed into the hit set.  Without
        any marker (e.g. a single-iteration run) every data row is returned.
        """
        lines = stdout.splitlines()
        last_iter = max((i for i, ln in enumerate(lines) if ln.startswith("# Iteration:")),
                        default=-1)
        return [ln for ln in lines[last_iter + 1:]
                if ln.strip() and not ln.startswith("#") and not ln.startswith("Search has")]

    def _parse_blast(self, stdout) -> pd.DataFrame:
        rows = self._final_iteration_rows(stdout)
        names = self._OUTFMT.split()[1:]
        df = pd.DataFrame([ln.split("\t") for ln in rows], columns=names)
        for col in ["qstart", "qend", "sstart", "send", "length", "gaps"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["bitscore"] = pd.to_numeric(df["bitscore"], errors="coerce")
        df["evalue"] = pd.to_numeric(df["evalue"], errors="coerce")
        df = df[(df["gaps"] <= self.max_gaps) & (df["evalue"] <= self.evalue)]
        pdb_id, chain = zip(*df["sseqid"].map(self._parse_sseqid)) if len(df) else ([], [])
        return df.assign(pdb_id=list(pdb_id), chain=list(chain))

    @staticmethod
    def _parse_sseqid(sseqid):
        """``pdb|4V12|A`` or ``4v12_A`` -> (pdb_id, chain)."""
        if sseqid.startswith("pdb|"):
            parts = sseqid.split("|")
            return parts[1].lower(), (parts[2] if len(parts) > 2 else None)
        if "_" in sseqid:
            pid, ch = sseqid.split("_", 1)
            return pid.lower(), ch
        return sseqid[:4].lower(), (sseqid[4:5] or None)
