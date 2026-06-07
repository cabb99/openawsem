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

    _OUTFMT = "6 sseqid qstart qend sstart send qseq sseq length gaps bitscore evalue"

    def __init__(self, database, *, n_mem=20, fragment_length=9, evalue=10000.0,
                 min_seq_sep=3, max_seq_sep=9, well_width=0.1, weight=1.0,
                 blast_exe="psiblast", num_iterations=5, max_gaps=0, num_threads=1,
                 cif_dir=None, download_format="cif"):
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

    def _parse_blast(self, stdout) -> pd.DataFrame:
        lines = [ln for ln in stdout.splitlines() if ln.strip() and not ln.startswith("Search has")]
        names = self._OUTFMT.split()[1:]
        df = pd.DataFrame([ln.split("\t") for ln in lines], columns=names)
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
