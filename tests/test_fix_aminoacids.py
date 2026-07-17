import pytest
import shutil
from pathlib import Path
from openawsem.helperFunctions.convertOpenmmTrajectoryToStandardMovie import (
    get_seq_dic,
    convert_openMM_to_standard_pdb,
)

data_path = Path(__file__).parent / 'data' / 'DNA_reconstruction'


def parse_pdb_residues(pdb_text):
    """Parse PDB text and return a dict mapping (chain, resnum) -> resname."""
    residues = {}
    for line in pdb_text.splitlines():
        if line.startswith('ATOM') and len(line) > 26:
            resname = line[17:20].strip()
            chain = line[21]
            resnum = int(line[22:26])
            residues[(chain, resnum)] = resname
    return residues


def assert_residues_converted(pdb_text):
    """Assert that all AWSEM placeholder residues were converted."""
    assert 'NGP' not in pdb_text
    assert 'IGL' not in pdb_text
    assert 'IPR' not in pdb_text


def assert_chain_boundary_residues(residues):
    """Assert correct residues at chain boundaries."""
    assert residues.get(('A', 1)) == 'MET'
    assert residues.get(('A', 274)) == 'ASP'
    assert residues.get(('B', 1)) == 'MET'
    assert residues.get(('B', 313)) == 'GLU'

def assert_chain_boundary_residues_resnum_offset(residues):
    """Similar to assert_chain_boundary_residues,
       but uses the residue numbers and names from
       the resnum_offset test system"""
    with open('test_foo.txt','w') as f:
        for thing in residues.items():
            f.write(f'{thing}\n')
    assert residues.get(('A', 1)) == 'ALA', residues
    assert residues.get(('A', 302)) == 'GLY'
    assert residues.get(('B', 303)) == 'MET'
    assert residues.get(('B', 615)) ==  'GLU'


class TestFixAminoacids:
    """Tests for the fix_aminoacids reconstruction workflow."""

    def test_reconstruction_with_multiple_chain_fasta(self, tmp_path):
        """Test reconstruction with properly formatted multi-chain FASTA."""
        pdb_copy = tmp_path / 'test.pdb'
        shutil.copy(data_path / 'bp40_frame5000_awsem.pdb', pdb_copy)
        
        seq_dic = get_seq_dic(fasta=data_path / 'protein_multiple.seq')
        convert_openMM_to_standard_pdb(fileName=pdb_copy, seq_dic=seq_dic, back=False)
        
        result = pdb_copy.read_text()
        assert_residues_converted(result)
        assert_chain_boundary_residues(parse_pdb_residues(result))

    def test_reconstruction_with_multiple_chain_fasta_resnum_offset(self, tmp_path):
        """Test reconstruction with properly formatted multi-chain FASTA.
           The key difference between this test and test_reconstruction_with_multiple_chain_fasta
           is that the chain B residue numbering starts at 303 instead of 1."""
        pdb_copy = tmp_path / 'test.pdb'

        # these files don't have DNA in them, but we'll use the same data_path
        # folder that we used for the previous test just to keep things organized
        shutil.copy(data_path / 'NFKB_chainB303.pdb', pdb_copy)
        seq_dic = get_seq_dic(fasta=data_path / 'NFKB_chainB303.fasta')
        convert_openMM_to_standard_pdb(fileName=pdb_copy, seq_dic=seq_dic, back=False)
        
        result = pdb_copy.read_text()
        assert_residues_converted(result)
        assert_chain_boundary_residues_resnum_offset(parse_pdb_residues(result))

    def test_reconstruction_with_single_sequence_fasta(self, tmp_path):
        """Test reconstruction with deprecated single-sequence file (no chain headers)."""
        pdb_copy = tmp_path / 'test.pdb'
        shutil.copy(data_path / 'bp40_frame5000_awsem.pdb', pdb_copy)
        
        with pytest.warns(DeprecationWarning, match="No '>CRYSTAL_STRUCTURE:' chain headers found"):
            seq_dic = get_seq_dic(fasta=data_path / 'protein_single.seq')
        
        assert 'Unknown' in seq_dic
        assert len(seq_dic) == 1
        
        convert_openMM_to_standard_pdb(fileName=pdb_copy, seq_dic=seq_dic, back=False)
        
        result = pdb_copy.read_text()
        assert_residues_converted(result)
        assert_chain_boundary_residues(parse_pdb_residues(result))


class TestGetSeqDic:
    """Tests for the get_seq_dic function."""

    def test_multiple_chains(self):
        """Test parsing a valid multi-chain FASTA file."""
        seq_dic = get_seq_dic(fasta=data_path / 'protein_multiple.seq')

        assert len(seq_dic) == 2
        assert len(seq_dic['A']) == 274
        assert len(seq_dic['B']) == 313
        first_7_A = seq_dic['A'][0] + seq_dic['A'][1] + seq_dic['A'][2] + seq_dic['A'][3] +\
            seq_dic['A'][4] + seq_dic['A'][5] + seq_dic['A'][6]
        assert first_7_A == 'MAYVEII'
        first_7_B = seq_dic['B'][274] + seq_dic['B'][275] + seq_dic['B'][276] + seq_dic['B'][277] +\
            seq_dic['B'][278] + seq_dic['B'][279] + seq_dic['B'][280]
        assert first_7_B == 'MGPYLQI'

    def test_single_sequence_deprecated(self):
        """Test that plain sequence files emit deprecation warning."""
        with pytest.warns(DeprecationWarning, match="No '>CRYSTAL_STRUCTURE:' chain headers found"):
            seq_dic = get_seq_dic(fasta=data_path / 'protein_single.seq')
        
        assert 'Unknown' in seq_dic
        assert len(seq_dic) == 1
        assert len(seq_dic['Unknown']) == 587  # 274 + 313
        start_7 = seq_dic['Unknown'][0] + seq_dic['Unknown'][1] + seq_dic['Unknown'][2] +\
            seq_dic['Unknown'][3] + seq_dic['Unknown'][4] + seq_dic['Unknown'][5] + seq_dic['Unknown'][6]
        assert start_7 == 'MAYVEII', f"Sequence starts incorrectly: {start_7}"
        end_4 = seq_dic['Unknown'][583] + seq_dic['Unknown'][584] + seq_dic['Unknown'][585] + seq_dic['Unknown'][586] 
        assert end_4 == 'YYPE', f"Sequence ends incorrectly: {end_4}"
