"""Homolog detection + brain_damage hit filtering, shared by the fragment-memory backends.

brain_damage modes (as in ``local_blast``): ``0`` all hits | ``1`` exclude homologs |
``2`` homologs except self (>cutoff) | ``0.5`` self only (>cutoff).  Homologs are found by one
full-sequence PSI-BLAST per record (the same call ``local_blast`` uses), keyed by ``pdb_id``;
``local_db`` reuses this so its brain_damage excludes exactly the same PDBs.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)


def parse_sseqid(sseqid):
    """``pdb|4V12|A`` or ``4v12_A`` -> ``(pdb_id, chain)`` (pdb_id lowercased)."""
    if sseqid.startswith("pdb|"):
        parts = sseqid.split("|")
        return parts[1].lower(), (parts[2] if len(parts) > 2 else None)
    if "_" in sseqid:
        pid, ch = sseqid.split("_", 1)
        return pid.lower(), ch
    return sseqid[:4].lower(), (sseqid[4:5] or None)


def _blast_full_sequence(sequence, *, database, blast_exe, homolog_evalue, num_threads,
                         brain_damage) -> pd.DataFrame:
    fields = "sseqid pident evalue bitscore"
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=True) as fh:
        fh.write(f">query\n{sequence}\n")
        fh.flush()
        cmd = [blast_exe, "-query", fh.name, "-db", database,
               "-num_iterations", "1", "-word_size", "3", "-evalue", str(homolog_evalue),
               "-num_threads", str(num_threads), "-outfmt", f"6 {fields}"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"homolog BLAST failed; refusing to apply brain_damage={brain_damage} on an "
                f"empty homolog set (that would silently keep/drop the wrong hits)") from exc
    lines = [ln for ln in result.stdout.splitlines()
             if ln.strip() and not ln.startswith("Search has")]
    df = pd.DataFrame([ln.split("\t") for ln in lines], columns=fields.split())
    if len(df):
        df["pident"] = pd.to_numeric(df["pident"], errors="coerce")
        pdb_id, chain = zip(*df["sseqid"].map(parse_sseqid))
        return df.assign(pdb_id=list(pdb_id), chain=list(chain))
    return df.assign(pdb_id=[], chain=[])


def blast_homologs(records, *, database, blast_exe="psiblast", homolog_evalue=0.005,
                   num_threads=1, brain_damage=1) -> dict:
    """Full-sequence BLAST per record -> ``{pdb_id: max percent identity}`` (the target's
    homologs).  Each record is blasted on its own so a multi-chain query does not produce
    spurious junction-spanning homolog alignments."""
    homologs: dict = {}
    for record in records:
        df = _blast_full_sequence(record, database=database, blast_exe=blast_exe,
                                  homolog_evalue=homolog_evalue, num_threads=num_threads,
                                  brain_damage=brain_damage)
        for pid, pident in zip(df["pdb_id"], df["pident"]):
            homologs[pid] = max(homologs.get(pid, 0.0), float(pident))
    return homologs


def keep_predicate(homologs, *, brain_damage, cutoff_identical):
    """The pid->bool keep rule for a brain_damage mode, or ``None`` to keep everything."""
    def is_homolog(pid):
        return pid in homologs

    def is_self(pid):
        return homologs.get(pid, 0.0) > cutoff_identical

    if brain_damage == 1:        # exclude all homologs
        return lambda pid: not is_homolog(pid)
    if brain_damage == 2:        # homologs only, but not self (>cutoff)
        return lambda pid: is_homolog(pid) and not is_self(pid)
    if brain_damage == 0.5:      # self only (>cutoff)
        return is_self
    return None


def apply_brain_damage(hits, homologs, *, brain_damage, cutoff_identical, pdb_col="pdb_id"):
    """Filter a hits DataFrame by the brain_damage rule (no-op for mode 0 / unknown)."""
    keep = keep_predicate(homologs, brain_damage=brain_damage, cutoff_identical=cutoff_identical)
    if keep is None or len(hits) == 0:
        return hits
    mask = hits[pdb_col].map(keep)
    logger.info("brain_damage=%s: kept %d/%d hits (%d homologs found, cutoff %s%%)",
                brain_damage, int(mask.sum()), len(hits), len(homologs), cutoff_identical)
    return hits[mask].reset_index(drop=True)
