#!/usr/bin/env python3
try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except ModuleNotFoundError:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
import sys
from pdbfixer import * # this gets us PDBFile and PDBxFile
import mdtraj as md
from Bio.PDB.Polypeptide import *
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBList
from Bio.PDB import PDBIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import textwrap
import shutil
from pathlib import Path
import openawsem.helperFunctions
import openawsem.functionTerms

__author__ = 'Carlos Bueno'

# Application location
__location__ = Path(__file__).resolve().parent

data_path = openawsem.helperFunctions.DataPath(default_location = __location__,
                                               default_config_path = __location__ / 'config.ini', 
                                               custom_config_path = Path.home() / '.awsem' / 'config.ini')

_AWSEMresidues = ['IPR', 'IGL', 'NGP']
xml = data_path.topology/'awsem.xml'

three_to_one = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
                'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
                'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
                'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}

def get_openmm_io_class(file_type):
    if file_type == "pdb":
        io_class = PDBFile
    elif file_type == "cif":
        io_class = PDBxFile
    else:
        raise ValueError(f"Expected file_type 'pdb' or 'cif' but got file_type={file_type}") 
    return io_class

def parsePDB(pdb_file):
    '''Reads a pdb file and outputs a pandas DataFrame'''
    def pdb_line(line):
        return dict(recname=str(line[0:6]).strip(),
                    serial=int(line[6:11]),
                    name=str(line[12:16]).strip(),
                    altLoc=str(line[16:17]),
                    resname=str(line[17:20]).strip(),
                    chainID=str(line[21:22]),
                    resSeq=int(line[22:26]),
                    iCode=str(line[26:27]),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=0.0 if line[54:60].strip() == '' else float(line[54:60]),
                    tempFactor=0.0 if line[60:66].strip() == '' else float(line[60:66]),
                    element=str(line[76:78]),
                    charge=str(line[78:80]))

    with open(pdb_file, 'r') as pdb:
        lines = []
        for line in pdb:
            if len(line) > 6 and line[:6] in ['ATOM  ', 'HETATM']:
                lines += [pdb_line(line)]
    pdb_atoms = pd.DataFrame(lines)
    pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                           'resname', 'chainID', 'resSeq', 'iCode',
                           'x', 'y', 'z', 'occupancy', 'tempFactor',
                           'element', 'charge']]
    return pdb_atoms
    
def writePDB(atoms,pdb_file):
    '''Reads a pandas DataFrame of atoms and outputs a pdb file'''
    with open(pdb_file, 'w+') as pdb:
        for i, atom in atoms.iterrows():
            pdb_line = f'{atom.recname:<6}{atom.serial:>5} {atom["name"]:^4}{atom.altLoc:1}'+\
                       f'{atom.resname:<3} {atom.chainID:1}{atom.resSeq:>4}{atom.iCode:1}   '+\
                       f'{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}' +\
                       f'{atom.occupancy:>6.2f}{atom.occupancy:>6.2f}'+' ' * 10 +\
                       f'{atom.element:>2}{atom.charge:>2}'
            assert len(pdb_line) == 80, f'An item in the atom table is longer than expected ({len(pdb_line)})\n{pdb_line}'
            pdb.write(pdb_line + '\n')


def parseConfigTable(config_section):
    """Parses a section of the configuration file as a table"""

    def readData(config_section, a):
        """Filters comments and returns values as a list"""
        temp = config_section.get(a).split('#')[0].split()
        l = []
        for val in temp:
            val = val.strip()
            try:
                x = int(val)
                l += [x]
            except ValueError:
                try:
                    y = float(val)
                    l += [y]
                except ValueError:
                    l += [val]
        return l

    data = []
    for a in config_section:
        if a == 'name':
            columns = readData(config_section, a)
        elif len(a) > 3 and a[:3] == 'row':
            data += [readData(config_section, a)]
        else:
            logging.warning(f'Unexpected row {readData(config_section, a)}')
    return pd.DataFrame(data, columns=columns)


def copy_parameter_files():
    src = __location__ / "parameters"
    dest = '.'
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)

def create_single_memory(fixed, memory_file="fixed.pdb", chain=-1):
    """Creates a single memory file from a openmm pdb file"""
    PDBFile.writeFile(fixed.topology, fixed.positions, open(memory_file, 'w'))
    openawsem.helperFunctions.create_single_memory(memory_file,chain)
    
def save_protein_sequence(Coarse,sequence_file='protein.seq'):
    """Saves protein sequence to a file from table"""
    protein_data=Coarse[Coarse.resname.isin(_AWSEMresidues)].copy()
    resix = (protein_data.chainID + '_' + protein_data.resSeq.astype(str))
    res_unique = resix.unique()
    protein_data['resID'] = resix.replace(dict(zip(res_unique, range(len(res_unique)))))
    protein_sequence=[r.iloc[0]['real_resname'] for i, r in protein_data.groupby('resID')]
    protein_sequence_one = [three_to_one[a] for a in protein_sequence]

    with open(sequence_file,'w+') as ps:
        ps.write(''.join(protein_sequence_one))

class BaseError(Exception):
    pass


class Protein(object):
    def __init__(self, atoms, sequence, k_awsem=1):
        self.atoms = atoms

        #Include real residue name in atoms
        atoms = self.atoms.copy()
        atoms['chain_res'] = atoms['chainID'].astype(str) + '_' + atoms['resSeq'].astype(str)
        sel = atoms[atoms['resname'].isin(_AWSEMresidues)]
        resix = sel['chain_res'].unique()
        assert len(resix) == len(sequence), \
            f'The number of residues {len(resix)} does not agree with the length of the sequence {len(sequence)}'
        atoms.index = atoms['chain_res']
        for r, s in zip(resix, sequence):
            atoms.loc[r, 'real_resname'] = s
        atoms.index = range(len(atoms))
        self.atoms = atoms

        protein_data = atoms[atoms.resname.isin(_AWSEMresidues)].copy()
        # renumber residues
        resix = (protein_data.chainID + '_' + protein_data.resSeq.astype(str))
        res_unique = resix.unique()
        protein_data['resID'] = resix.replace(dict(zip(res_unique, range(len(res_unique)))))
        # renumber atom types
        atom_types_table = {'N': 'n', 'H': 'h', 'CA': 'ca', 'C': 'c', 'O': 'o', 'CB': 'cb'}
        protein_data['atom_list'] = protein_data['name'].replace(atom_types_table)
        protein_data['idx'] = protein_data.index.astype(int)
        self.protein_data = protein_data
        self.atom_lists = protein_data.pivot(index='resID', columns='atom_list', values='idx').fillna(-1).astype(int)
        self.n = self.atom_lists['n'].tolist()
        self.h = self.atom_lists['h'].tolist()
        self.ca = self.atom_lists['ca'].tolist()
        self.c = self.atom_lists['c'].tolist()
        self.o = self.atom_lists['o'].tolist()
        self.cb = self.atom_lists['cb'].tolist()
        self.nres = len(self.atom_lists)
        self.k_awsem = k_awsem
        self.res_type = [r.iloc[0]['resname'] for i, r in protein_data.groupby('resID')]
        self.chain_starts = [c.iloc[0].resID for i, c in protein_data.groupby('chainID')]
        self.chain_ends = [c.iloc[-1].resID for i, c in protein_data.groupby('chainID')]
        self.natoms = len(atoms)
        self.bonds = self._setup_bonds()
        self.seq = sequence
        self.resi = pd.merge(self.atoms, self.protein_data, how='left').resID.fillna(-1).astype(int).tolist()
        pass

    def _setup_bonds(self):
        bonds = []
        for i in range(self.nres):
            bonds.append((self.ca[i], self.o[i]))
            if not self.res_type[i] == "IGL":
                bonds.append((self.ca[i], self.cb[i]))
            if i not in self.chain_ends:
                bonds.append((self.ca[i], self.ca[i + 1]))
                bonds.append((self.o[i], self.ca[i + 1]))

        for i in range(self.nres):
            if i not in self.chain_starts and not self.res_type[i] == "IGL":
                bonds.append((self.n[i], self.cb[i]))
            if i not in self.chain_ends and not self.res_type[i] == "IGL":
                bonds.append((self.c[i], self.cb[i]))
            if i not in self.chain_starts and i not in self.chain_ends:
                bonds.append((self.n[i], self.c[i]))
        return bonds

    def setup_virtual_sites(self, system, ):
        # set virtual sites
        for i in range(self.nres):
            if i not in self.chain_starts:
                n_virtual_site = ThreeParticleAverageSite(self.ca[i - 1], self.ca[i], self.o[i - 1],
                                                                       0.48318, 0.70328, -0.18643)
                system.setVirtualSite(self.n[i], n_virtual_site)
                if not self.res_type[i] == "IPR":
                    h_virtual_site = ThreeParticleAverageSite(self.ca[i - 1], self.ca[i], self.o[i - 1],
                                                                           0.84100, 0.89296, -0.73389)
                    system.setVirtualSite(self.h[i], h_virtual_site)
            if i not in self.chain_ends:
                c_virtual_site = ThreeParticleAverageSite(self.ca[i], self.ca[i + 1], self.o[i],
                                                                       0.44365, 0.23520, 0.32115)
                # print("Virtual", c[i])
                system.setVirtualSite(self.c[i], c_virtual_site)

    @classmethod
    def fromPDB(cls, pdb, pdbout='CoarseProtein.pdb'):
        """ Initializes a protein form a pdb, making all the atoms coarse-grained"""
        pass

    @classmethod
    def fromCoarsePDB(cls, pdb_file, sequence):
        """ Initializes the protein from an already coarse grained pdb"""
        atoms = parsePDB(pdb_file)
        return cls(atoms, sequence)

    def parseConfigurationFile(self):
        """ Parses the AWSEM configuration file to use for the topology and to set the forces"""
        pass

    def computeTopology(self):
        """ Compute the bonds and angles from the pdb"""
        pass

    @staticmethod
    def CoarseGrain(pdb_table):
        """ Selects AWSEM atoms from a pdb table and returns a table containing only the coarse-grained atoms for AWSEM """
        protein_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                            'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                            'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                            'SER', 'THR', 'TRP', 'TYR', 'VAL']
        awsem_atoms = ["N", "H", "CA", "C", "O", "CB"]

        # Select coarse grained atoms
        selection = pdb_table[pdb_table.resname.isin(protein_residues) & pdb_table.name.isin(awsem_atoms)].copy()

        # Remove virtual atoms at the end or begining of the chain
        drop_list = []
        for chain in selection.chainID.unique():
            sel = selection[selection.chainID == chain]
            drop_list += list(sel[(sel.resSeq == sel.resSeq.min()) & sel['name'].isin(['N', 'H'])].index)
            drop_list += list(sel[(sel.resSeq == sel.resSeq.max()) & sel['name'].isin(['C'])].index)
        selection = selection.drop(drop_list)

        # Replace resnames
        selection['real_resname'] = selection.resname.copy()
        resname = selection.resname.copy()
        resname[:] = 'NGP'
        resname[selection.resname == 'PRO'] = 'IPR'
        resname[selection.resname == 'GLY'] = 'IGL'
        selection.resname = resname

        # CB element is B
        selection.loc[selection['name'] == 'CB', 'element'] = 'B'

        # Reorder atoms
        selection.name = pd.Categorical(selection.name, awsem_atoms)
        selection.sort_values(['chainID', 'resSeq', 'name'])

        # Prepare virtual sites
        for c, chain in selection.groupby('chainID'):
            first = chain.resSeq.min()
            last = chain.resSeq.max()
            for i, residue in chain.groupby('resSeq'):
                idx = dict(zip(residue.name, residue.index))
                pos = dict(zip(residue.name, [residue.loc[i, ['x', 'y', 'z']] for i in residue.index]))

                if i != first:
                    if 'N' in idx.keys():
                        selection.loc[idx['N'], ['x', 'y', 'z']] = 0.48318 * pos_im['CA'] + 0.70328 * pos[
                            'CA'] - 0.18643 * pos_im['O']
                    if 'H' in idx.keys():
                        selection.loc[idx['H'], ['x', 'y', 'z']] = 0.84100 * pos_im['CA'] + 0.89296 * pos[
                            'CA'] - 0.73389 * pos_im['O']
                    if 'C' in idx.keys():
                        selection.loc[idx_im['C'], ['x', 'y', 'z']] = 0.44365 * pos_im['CA'] + 0.23520 * pos[
                            'CA'] + 0.32115 * pos_im['O']

                pos_im = pos.copy()
                idx_im = idx.copy()
        # Renumber
        selection['serial'] = range(len(selection))
        return selection

    @staticmethod
    def write_sequence(Coarse, seq_file='protein.seq'):
        protein_data = Coarse[Coarse.resname.isin(_AWSEMresidues)].copy()
        resix = (protein_data.chainID + '_' + protein_data.resSeq.astype(str))
        res_unique = resix.unique()
        protein_data['resID'] = resix.replace(dict(zip(res_unique, range(len(res_unique)))))
        protein_sequence = [r.iloc[0]['real_resname'] for i, r in protein_data.groupby('resID')]
        protein_sequence_one = [three_to_one[a] for a in protein_sequence]

        with open(seq_file, 'w+') as ps:
            ps.write(''.join(protein_sequence_one))

def addNonBondedExclusions(oa, force):
    cb_fixed = [x if x > 0 else y for x, y in zip(oa.cb, oa.ca)]
    none_cb_fixed = [i for i in range(oa.natoms) if i not in cb_fixed]
    for e1 in none_cb_fixed:
        for e2 in none_cb_fixed:
            if e1 > e2:
                continue
            force.addExclusion(e1, e2)
    for e1 in none_cb_fixed:
        for e2 in cb_fixed:
            force.addExclusion(e1, e2)
    for e1 in none_cb_fixed:
        for e2 in none_cb_fixed:
            if e1 > e2:
                continue
            force.addExclusion(e1, e2)
    for e1 in none_cb_fixed:
        for e2 in cb_fixed:
            force.addExclusion(e1, e2)


se_map_3_letter = {'ALA': 0,  'PRO': 1,  'LYS': 2,  'ASN': 3,  'ARG': 4,
                   'PHE': 5,  'ASP': 6,  'GLN': 7,  'GLU': 8,  'GLY': 9,
                   'ILE': 10, 'HIS': 11, 'LEU': 12, 'CYS': 13, 'MET': 14,
                   'SER': 15, 'THR': 16, 'TYR': 17, 'VAL': 18, 'TRP': 19}

def line_number():
    return sys._getframe(1).f_lineno


def prepare_pdb(pdb_filename, chains_to_simulate, use_cis_proline=False, keepIds=False, removeHeterogens=True):
    """
        This function cleans an input PDB or PDBx/mmCIF file and prepares it for an OpenAWSEM simulation. 
        In the mm_create_project.py workflow, the calls to this function occur after a call to 
        helperFunctions.myFunctions.cleanPdb, which does some preliminary structure cleaning. 
        I don't know why the cleaning protocol is split like this, but there may be a good reason for it.

        This function uses the openmm PDBFile/PDBxFile/Topology infrastructure directly and also uses 
        the PDBFixer class, which is basically the same PDBFile/PDBxFile/Topology functionality
        packaged with some convenient Topology editing methods. 

        -- Finley Clark, 1/15/25
    """

    # configure local variables
    file_type = pdb_filename[-3:]
    io_class = get_openmm_io_class(file_type)
    cleaned_pdb_filename = f"{pdb_filename[:-4]}-cleaned.{file_type}" 
    input_pdb_filename = f"{pdb_filename[:-4]}-openmmawsem.{file_type}"
    print(pdb_filename) 

    # clean the pdb
    fixer = PDBFixer(filename=pdb_filename)
    # remove unwanted chains
    chains = list(fixer.topology.chains())
    chains_to_remove = [i for i, x in enumerate(chains) if x.id not in chains_to_simulate]
    fixer.removeChains(chains_to_remove)
    #Identify Missing Residues
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    #Replace Nonstandard Residues
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    #Remove Heterogens
    if removeHeterogens:
        fixer.removeHeterogens(keepWater=False)
    #Add Missing Heavy Atoms
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    #Add Missing Hydrogens
    fixer.addMissingHydrogens(7.0)
    # write crystal_structure-cleaned structure file
    with open(cleaned_pdb_filename,'w') as f:
        io_class.writeFile(fixer.topology, fixer.positions, f, keepIds=keepIds)

    # prepare the pdb for openawsem simulation
    # identify terminal residues--returns something like {'A': (0, 114), 'B': (115, 229)}
    terminal_residues = {}
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        residue_indices = [int(residue.index) for residue in residues]
        terminal_residues.update({chain.id:(residue_indices[0],residue_indices[-1])})
    # get chains and residues from old topology, 
    # raising an error if we encounter any noncanonical residues,
    # and excluding non-awsem atoms from acceptable residues.
    # "non-awsem atoms" means everything besides N, H, CA, CB, O, and C for non-terminal residues,
    # everything besides CA, CB, O, and C for N-terminal residues,
    # and everything besides N, H, CA, CB, and O for C-terminal residues
    remove_atoms = []
    chain_info = {}
    for chain in fixer.topology.chains():
        chain_info.update({chain.id:{}}) # chain.id is "A", "B", etc.
        for residue in chain.residues():
            chain_info[chain.id].update({residue.index:[residue.name,residue.chain,residue.id,residue.insertionCode]})
            if residue.name in ["ALA","CYS","ASP","GLU","PHE","HIS","ILE","LYS","LEU","MET","ASN","PRO","GLN","ARG",
                                      "SER","THR","VAL","TRP","TYR"]:
                awsem_atoms = ["CA","O","CB","C","H","N"]
            elif residue.name == "GLY":
                awsem_atoms = ["CA","O","C","H","N"]
            else:
                raise ValueError(f"Unrecognized residue name: {residue.name}")
            for atom in residue.atoms():
                if atom.name not in awsem_atoms:
                    remove_atoms.append(atom)
                elif terminal_residues[residue.chain.id][0] == residue.index: # the N terminus
                    if atom.name in ["N","H"]: # the awsem force field doesn't include these atoms at the N terminal residue
                        remove_atoms.append(atom)
                elif terminal_residues[residue.chain.id][1] == residue.index: # the C terminus
                    if atom.name == "C": # the awsem force field doesn't include this atom at the C terminal residue
                        remove_atoms.append(atom)
    # set up new topology with same chains and residues
    new_topology = topology.Topology() # openmm.app.topology.Topology
    awsem_resnames = {"ALA":"NGP","CYS":'NGP','ASP':'NGP','GLU':'NGP','PHE':'NGP','GLY':'IGL','HIS':'NGP',
        'ILE':'NGP','LYS':'NGP','LEU':'NGP','MET':'NGP','ASN':'NGP','PRO':'IPR','GLN':'NGP','ARG':'NGP',
        'SER':'NGP','THR':'NGP','VAL':'NGP','TRP':'NGP','TYR':'NGP'}
    for chain_name in chain_info.keys():
        chain = new_topology.addChain(id=chain_name)
        # We expect the residue.index to be an integer representing the position of the residue in the chain,
        # starting at 0 and counting up.
        # But wait, this won't be true for a multichain file
        # So we'll only run this assert statement if there's only one chain
        if len(chain_info.keys()) == 1:
            assert list(chain_info[chain_name].keys()) == list(range(len(chain_info[chain_name].keys()))), f"list(chain_info[chain_name].keys()): {list(chain_info[chain_name].keys())}"
        for residue_index in chain_info[chain_name].keys():
            new_topology.addResidue(awsem_resnames[chain_info[chain_name][residue_index][0]], # name
                                    chain, # need to pass in our newly created chain, not the chain from the pdbfixer-loaded topology
                                    chain_info[chain_name][residue_index][2], # id
                                    chain_info[chain_name][residue_index][3], # inscode
                                   )
    # add AWSEM atoms to the new topology
    for atom in fixer.topology.atoms():
        if atom not in remove_atoms:
            if atom.name == "CB":
                atom_element = Element.getBySymbol("B") # i don't know why we do this. Maybe to make selecting things easier later? Hopefully nothing in the simulation workflow relies on it
            else:
                atom_element = atom.element
            new_topology.addAtom(atom.name,atom_element,list(new_topology.residues())[atom.residue.index]) # avoid specifying id to allow it to reorder as appropriate
    # get positions of retained atoms    
    keep_atom_indices = [atom.index for atom in fixer.topology.atoms() if atom not in remove_atoms]
    keep_atom_vectors = [vector/angstrom for counter,vector in enumerate(fixer.positions) if counter in keep_atom_indices] 
    keep_atom_vectors = np.array([[vector.x,vector.y,vector.z] for vector in keep_atom_vectors])
    #Fix Virtual Site Coordinates 
    # we're not going to remove the coordinates of the removed atoms yet because
    final_top, final_pos = prepare_virtual_sites_v2_openmm(new_topology, keep_atom_vectors,
                               chain_info, use_cis_proline=use_cis_proline)
    # write -openmmawsem file
    # this one only has AWSEM atoms 
    # (meaning ["CA","O","CB","C","H","N"] for non-glycine and ["CA","O","C","H","N"] for glycine)
    # and also excludes N-terminal residue N and H, C-terminal residue C
    write_fh = open(input_pdb_filename,'w')
    with open(input_pdb_filename,'w') as f:
        io_class.writeFile(final_top, final_pos, f, keepIds=True) 
    # add a '#' on a new line at the end of the cif,
    # which seems to be necessary for vmd to be able to read the output file.
    # This isn't necessary to create an openawsem project but is a nice feature
    if file_type == "cif":  
        lines = []
        with open(input_pdb_filename,'r') as f:
            for line in f:
                lines.append(line)
        lines.append('#')
        with open(input_pdb_filename,'w') as f:
            for line in lines:
                f.write(line)

    return input_pdb_filename, cleaned_pdb_filename

def correct_atom_position(atom, pos, use_cis_proline=False):

    return pos

def prepare_virtual_sites_v2_openmm(top, pos, chain_info, use_cis_proline=False):
    # check on our data before working with it
    if not pos.shape == (len(list(top.atoms())),3):
        raise ValueError(f'pos.shape: {pos.shape}, len(list(top.residues())): {len(list(top.residues()))}')
    if not len(list(top.residues())) == len([ 'foo' for atom in top.atoms() if atom.name == "CA"]) \
      == len([ 'foo' for atom in top.atoms() if atom.name == "O"]):
        raise ValueError(f"""{len(list(top.residues()))} residues,
                            {len([ 'foo' for atom in top.atoms() if atom.name == "CA"])} CA atoms,
                            {len([ 'foo' for atom in top.atoms() if atom.name == "O"])} O atoms,
                            but they should all be equal""")
    # get indices of all CA and O atoms
    # every residue has exactly one CA and exactly one O
    CA_indices = []
    O_indices = []
    for counter,atom in enumerate(top.atoms()):
        assert atom.index == counter, f"atom.index: {atom.index}, counter: {counter}"
        if atom.name == "CA":
           CA_indices.append(atom.index)
        elif atom.name == "O":
           O_indices.append(atom.index)
    # get chain start and end residue indices
    chain_start_res_indices = [item[0] for item in chain_info.items()]
    chain_end_res_indices = [item[1] for item in chain_info.items()]
    # edit coordinates
    atom_count = 0
    atom_counter = 0
    for residue_counter,residue in enumerate(top.residues()):  
        residue_atoms = list(residue.atoms()) 
        if use_cis_proline and residue.name=="IPR":
            for atom_counter, atom in enumerate(residue_atoms):
                # N terminal residues won't have N and H, while C terminal residues won't have C,
                # which is fine
                if atom.name == "H":  
                    pos[atom_count+atom_counter,:] = -0.9871*pos[CA_indices[residue_counter-start_offset-1]] \
                                                     +0.9326*pos[CA_indices[residue_counter-start_offset]] \
                                                     +1.0604*pos[O_indices[residue_counter-start_offset-1]]
                elif atom.name == "C":
                    pos[atom_count+atom_counter,:] = 0.2196*pos[CA_indices[residue_counter-end_offset]] \
                                                    +0.2300*pos[CA_indices[residue_counter-end_offset+1]]\
                                                    +0.5507*pos[O_indices[residue_counter-end_offset]]
                elif atom.name == "N":
                    pos[atom_count+atom_counter,:] = -0.2094*pos[CA_indices[residue_counter-1]] + 0.6908*pos[CA_indices[residue_counter]] + 0.5190*pos[O_indices[residue_counter-1]]
        else:
            for atom_counter, atom in enumerate(residue_atoms): 
                # N terminal residues won't have N and H, while C terminal residues won't have C,
                # which is fine
                if atom.name == "N":
                    pos[atom_count+atom_counter,:] = 0.48318*pos[CA_indices[residue_counter-1]] + 0.70328*pos[CA_indices[residue_counter]] - 0.18643*pos[O_indices[residue_counter-1]]
                elif atom.name == "C":
                    pos[atom_count+atom_counter,:] = 0.44365*pos[CA_indices[residue_counter]] + 0.23520*pos[CA_indices[residue_counter+1]] + 0.32115*pos[O_indices[residue_counter]]
                elif atom.name == "H":
                    pos[atom_count+atom_counter,:] = 0.84100*pos[CA_indices[residue_counter-1]] + 0.89296*pos[CA_indices[residue_counter]] - 0.73389*pos[O_indices[residue_counter-1]]
        # keep track of number of atoms iterated over so we can properly index coordinate array
        atom_count += len(residue_atoms)
    return (top,pos)



def prepare_virtual_sites_v2(pdb_file, use_cis_proline=False, file_type='pdb'): 
    lines = []
    with open(pdb_file,'r') as f:
        for line in f:
            if line[:6] == "HETATM":
                lines.append(f'ATOM  {line[6:]}')
            else:
                lines.append(line)
    with open(pdb_file,'w') as f:
        for line in lines:
            f.write(line)
    if file_type == "pdb":
        parser = PDBParser(QUIET=True)
    elif file_type == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Expected file_type 'pdb' or 'cif' but got file_type={file_type}")
    structure = parser.get_structure('X',pdb_file)
    res = list(structure.get_residues())
    model = structure[0]
    output_file = pdb_file
    f = open(pdb_file)
    all_lines = f.readlines()
    f.close()
    output = open(output_file, "w")
    index = 0
    for line in all_lines:
        # print(line)
        splitline = line.split()
        if len(line)>4 and line[0:4] == "ATOM":
            try:
                atom_index=line[6:11].strip()
                atom_type=line[12:16].strip()
                res_type=line[17:20].strip()
                chain=line[21].strip()
                res_index=line[22:26].strip()
                x=line[30:38].strip()
                y=line[38:46].strip()
                z=line[46:54].strip()
                element = line[76:78].strip()
            except ValueError:
                print(line)
                raise
        else:
            continue
        res_index = int(res_index)

        try:
            r_im = model[chain][res_index-1]
        except KeyError:
            r_im = model[chain][res_index] # won't be used
        r_i = model[chain][res_index]
        try:
            r_ip = model[chain][res_index+1]
        except KeyError:
            r_ip = model[chain][res_index]  # won't be used
        if use_cis_proline and res_type == "IPR":
            n_coord = -0.2094*r_im['CA'].get_coord()+ 0.6908*r_i['CA'].get_coord() + 0.5190*r_im['O'].get_coord()
            c_coord = 0.2196*r_i['CA'].get_coord()+ 0.2300*r_ip['CA'].get_coord() + 0.5507*r_i['O'].get_coord()
            h_coord = -0.9871*r_im['CA'].get_coord()+ 0.9326*r_i['CA'].get_coord() + 1.0604*r_im['O'].get_coord()
        else:
            n_coord = 0.48318*r_im['CA'].get_coord()+ 0.70328*r_i['CA'].get_coord()- 0.18643 *r_im['O'].get_coord()
            c_coord = 0.44365*r_i['CA'].get_coord()+ 0.23520*r_ip['CA'].get_coord()+ 0.32115 *r_i['O'].get_coord()
            h_coord = 0.84100*r_im['CA'].get_coord()+ 0.89296*r_i['CA'].get_coord()- 0.73389 *r_im['O'].get_coord()

        line_list=list(line)
        index += 1
        line_list[6:11] = "{:5d}".format(index)
        if atom_type == "N":
            line_list[30:38] = '{:.8s}'.format('{:8.3f}'.format(n_coord[0]))
            line_list[38:46] = '{:.8s}'.format('{:8.3f}'.format(n_coord[1]))
            line_list[46:54] = list('{:.8s}'.format('{:8.3f}'.format(n_coord[2])))
        if atom_type == "C":
            line_list[30:38] = '{:.8s}'.format('{:8.3f}'.format(c_coord[0]))
            line_list[38:46] = '{:.8s}'.format('{:8.3f}'.format(c_coord[1]))
            line_list[46:54] = list('{:.8s}'.format('{:8.3f}'.format(c_coord[2])))
        if atom_type == "H":
            line_list[30:38] = '{:.8s}'.format('{:8.3f}'.format(h_coord[0]))
            line_list[38:46] = '{:.8s}'.format('{:8.3f}'.format(h_coord[1]))
            line_list[46:54] = list('{:.8s}'.format('{:8.3f}'.format(h_coord[2])))
        new_line=''.join(line_list)
        output.write(new_line)
    index += 1
    line_list[6:11] = "{:5d}".format(index)
    line_list[0:4] = "TER "
    line_list[30:78] = " " * 48
    new_line=''.join(line_list)
    output.write(new_line)
    output.write("END\n")
    output.close()
    

def prepare_virtual_sites(pdb_file, use_cis_proline=False):
    parser = PDBParser(QUIET=True)
    structure=parser.get_structure('X',pdb_file,)
    for model in structure:
        for chain in model:
            r_im={}
            r_i={}
            for residue in chain:
                r_im=r_i
                r_i={}
                for atom in residue:
                    r_i[atom.get_name()]=atom
                if use_cis_proline and residue.get_resname() == "IPR":
                    if 'N' in r_i:
                        r_i['N'].set_coord(-0.2094*r_im['CA'].get_coord()+ 0.6908*r_i['CA'].get_coord() + 0.5190*r_im['O'].get_coord())
                    if 'C' in r_im:
                        r_im['C'].set_coord(0.2196*r_im['CA'].get_coord()+ 0.2300*r_i['CA'].get_coord() + 0.5507*r_im['O'].get_coord())
                    if 'H' in r_i:
                        r_i['H'].set_coord(-0.9871*r_im['CA'].get_coord()+ 0.9326*r_i['CA'].get_coord() + 1.0604*r_im['O'].get_coord())
                else:
                    if 'N' in r_i:
                        r_i['N'].set_coord(0.48318*r_im['CA'].get_coord()+ 0.70328*r_i['CA'].get_coord()- 0.18643 *r_im['O'].get_coord())
                    if 'C' in r_im:
                        r_im['C'].set_coord(0.44365*r_im['CA'].get_coord()+ 0.23520*r_i['CA'].get_coord()+ 0.32115 *r_im['O'].get_coord())
                    if 'H' in r_i:
                        r_i['H'].set_coord(0.84100*r_im['CA'].get_coord()+ 0.89296*r_i['CA'].get_coord()- 0.73389 *r_im['O'].get_coord())
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)

def build_lists_of_atoms(nres, residues):
    # build lists of atoms, residue types, and bonds
    atom_types=['n', 'h', 'ca', 'c', 'o', 'cb']
    res_types=[]
    atom_lists=dict(zip(atom_types,[[] for i in range(len(atom_types))]))
    for residue in residues:
        res_types.append(residue.name)
        atom_types=['n', 'h', 'ca', 'c', 'o', 'cb']
        residue_atoms = [x.index for x in residue.atoms()]
        if residue.index == 0:
            atom_types.remove('n')
            atom_lists['n'].append(-1)
            atom_types.remove('h')
            atom_lists['h'].append(-1)
        if residue.index == nres-1:
            atom_types.remove('c')
            atom_lists['c'].append(-1)
            pass
        if residue.name == "IGL" and 'cb' in atom_types:
            atom_types.remove('cb')
            atom_lists['cb'].append(-1)
        if residue.name in "IPR" and 'h' in atom_types:
            atom_types.remove('h')
            atom_lists['h'].append(-1)
        assert len(residue_atoms)==len(atom_types), '%s\n%s'%(str(residue_atoms),str(atom_types))
        atom_types=[a.name.lower() for a in residue._atoms] #Sometimes the atom order may be different
        for atom, atype in zip(residue_atoms, atom_types):
                atom_lists[atype].append(atom)

    return atom_lists, res_types

def build_lists_of_atoms_2(nres, residues, atoms):
    res_id = 0
    n = h = ca = c = o = cb = -1
    atom_types=['n', 'h', 'ca', 'c', 'o', 'cb']
    atom_types_table = {'N':'n', 'H':'h', 'CA':'ca', 'C':'c', 'O':'o', 'CB':'cb'}
    res_types=[]
    for residue in residues:
        res_types.append(residue.name)

    atom_lists=dict(zip(atom_types,[[-1]*nres for i in range(len(atom_types))]))
    for atom in atoms:
        atom_lists[atom_types_table[atom.name]][atom.residue.index] = atom.index

    return atom_lists, res_types


def ensure_atom_order(input_pdb_filename, quiet=1):
    # ensure order of ['n', 'h', 'ca', 'c', 'o', 'cb']
    # to be more specific, this ensure 'ca' always show up before 'c'.
    '''
    def first(t):
        return t[0]
    order_table = {'N':0, 'H':1, 'CA':2, 'C':3, 'O':4, 'CB':5}
    one_residue = []
    with open("tmp.pdb", "w") as out:
        with open(input_pdb_filename, "r") as f:
            all_lines = f.readlines()
            pre_res_id = all_lines[0].split()[5]
            for line in all_lines:
                info = line.split()
                if info[0]!="ATOM":
                    sorted_residue = sorted(one_residue, key=first)
                    for a in sorted_residue:
                        out.write(a[1])
                    one_residue = []
                    out.write(line)
                    continue
                res_id = info[5]
                if res_id != pre_res_id:
                    pre_res_id = res_id
                    # sort them in order
                    sorted_residue = sorted(one_residue, key=first)
                    for a in sorted_residue:
                        out.write(a[1])
                    if sorted_residue != one_residue:
                        if not quiet:
                            print("Reorder atom position")
                            print("Original, Changed to")
                        for i, t in zip(one_residue, sorted_residue):
                            if not quiet:
                                print(i[2], i[3], ", ", t[2], t[3])
                    one_residue = []
                atomType = info[2]
                one_residue.append((order_table[atomType], line, info[1], atomType))
            sorted_residue = sorted(one_residue, key=first)
            for a in sorted_residue:
                out.write(a[1])
    os.system(f"mv tmp.pdb {input_pdb_filename}")
    '''
    file_type = input_pdb_filename[-3:]
    io_class = get_openmm_io_class(file_type)
    temp = io_class(input_pdb_filename)
    top = temp.getTopology()
    pos = temp.getPositions(asNumpy=True)
    del temp
    # create new topology to fix atom order
    new_top = Topology()
    for old_chain in top.chains():
        # add chain to new topology
        new_chain = new_top.addChain(id=old_chain.id)
        for old_residue in old_chain.residues():
            # add new residue to new topology
            new_residue = new_top.addResidue(old_residue.name,new_chain,id=old_residue.id)
            # get atom info
            atoms = [atom for atom in old_residue.atoms()]
            names = [atom.name for atom in atoms]
            indices = [atom.index for atom in atoms]
            # get atom positions
            temp_pos = pos[indices,:]
            # check that old_residue.atoms() iterator returns atoms ordered by index
            strictly_increasing = [indices[i+1]>indices[i] for i in range(len(indices)-1)]
            if False in strictly_increasing:
                raise AssertionError(f"list(old_residue.atoms()) ({atoms}) not returned in order of increasing index")
            # reorder atoms, if necessary
            atom_count = 0
            if "N" in names:
                new_top.addAtom("N",atoms[names.index("N")].element,new_residue,id=indices[atom_count])
                temp_pos[atom_count,:] = pos[indices[names.index("N")]]
                atom_count += 1
            if "H" in names:
                new_top.addAtom("H",atoms[names.index("H")].element,new_residue,id=indices[atom_count])
                temp_pos[atom_count,:] = pos[indices[names.index("H")]]
                atom_count += 1   
            if "CA" in names:
                new_top.addAtom("CA",atoms[names.index("CA")].element,new_residue,id=indices[atom_count])
                temp_pos[atom_count,:] = pos[indices[names.index("CA")]]
                atom_count += 1   
            if "C" in names:
                new_top.addAtom("C",atoms[names.index("C")].element,new_residue,id=indices[atom_count])
                temp_pos[atom_count,:] = pos[indices[names.index("C")]]
                atom_count += 1         
            if "O" in names:
                new_top.addAtom("O",atoms[names.index("O")].element,new_residue,id=indices[atom_count])
                temp_pos[atom_count,:] = pos[indices[names.index("O")]]
                atom_count += 1
            if "CB" in names:
                new_top.addAtom("CB",atoms[names.index("CB")].element,new_residue,id=indices[atom_count])
                temp_pos[atom_count,:] = pos[indices[names.index("CB")]]
                atom_count += 1   
            # update atom positions
            pos[indices,:] = temp_pos
    io_class.writeFile(new_top,pos,open(input_pdb_filename,'w'),keepIds=True)        
    # check that we have the correct atom order (really just a test of the above code)
    for residue in new_top.residues():
        atom_names = [atom.name for atom in residue.atoms()]
        assert atom_names in [["N",'H','CA','C','O','CB'],['N','H','CA','C','O'],['N','CA','C','O','CB'],
                               # normal non-GLY/non-PRO     normal GLY            normal PRO
                                                  ['CA','C','O','CB'],['CA','C','O'],["N",'H','CA','O','CB'],["N",'H','CA','O'],["N",'CA','O','CB']], atom_names
                                 #                   N-terminus non-GLY N-terminus GLY C-term non-GLY/non-PRO     C-term GLY          C-term pro
    




def get_chain_starts_and_ends(all_res):
    chain_starts = []
    chain_ends = []
    pos = -1
    for i, res in enumerate(all_res):
        if res.chain.index != pos:
            chain_starts.append(i)
            pos = res.chain.index
            if i > 0:
                chain_ends.append(i-1)
    chain_ends.append(len(all_res)-1)
    return chain_starts, chain_ends

def setup_virtual_sites(nres, system, n, h, ca, c, o, cb, res_type, chain_starts, chain_ends, use_cis_proline=False):
    # set virtual sites
    for i in range(nres):
        if use_cis_proline and res_type[i] == "IPR":
            if i not in chain_starts:
                n_virtual_site = ThreeParticleAverageSite(ca[i-1], ca[i], o[i-1],
                                                         -0.2094, 0.6908, 0.5190)
                system.setVirtualSite(n[i], n_virtual_site)
                if not res_type[i] == "IPR":
                    h_virtual_site = ThreeParticleAverageSite(ca[i-1], ca[i], o[i-1],
                                                            -0.9871, 0.9326, 1.0604)
                    system.setVirtualSite(h[i], h_virtual_site)
            if i not in chain_ends:
                c_virtual_site = ThreeParticleAverageSite(ca[i], ca[i+1], o[i],
                                                        0.2196, 0.2300, 0.5507)
                # print("Virtual", c[i])
                system.setVirtualSite(c[i], c_virtual_site)
        else:
            if i not in chain_starts:
                n_virtual_site = ThreeParticleAverageSite(ca[i-1], ca[i], o[i-1],
                                                        0.48318, 0.70328, -0.18643)
                system.setVirtualSite(n[i], n_virtual_site)
                if not res_type[i] == "IPR":
                    h_virtual_site = ThreeParticleAverageSite(ca[i-1], ca[i], o[i-1],
                                                            0.84100, 0.89296, -0.73389)
                    system.setVirtualSite(h[i], h_virtual_site)
            if i not in chain_ends:
                c_virtual_site = ThreeParticleAverageSite(ca[i], ca[i+1], o[i],
                                                        0.44365, 0.23520, 0.32115)
                # print("Virtual", c[i])
                system.setVirtualSite(c[i], c_virtual_site)

def setup_bonds(nres, n, h, ca, c, o, cb, res_type, chain_starts, chain_ends):
    bonds = []
    for i in range(nres):
        bonds.append((ca[i], o[i]))
        if not res_type[i] == "IGL":
            bonds.append((ca[i], cb[i]))
        if  i not in chain_ends:
            bonds.append((ca[i], ca[i+1]))
            bonds.append((o[i], ca[i+1]))

    for i in range(nres):
        if i not in chain_starts and not res_type[i] == "IGL":
            bonds.append((n[i], cb[i]))
        if i not in chain_ends and not res_type[i] == "IGL":
            bonds.append((c[i], cb[i]))
        if i not in chain_starts and i not in chain_ends:
            bonds.append((n[i], c[i]))
    return bonds

def read_fasta(fastaFile):
    with open(fastaFile) as input_data:
        data = ""
        for line in input_data:
            if(line[0] == ">"):
                print(line)
            elif(line == "\n"):
                pass
            else:
                data += line.strip("\n")
    return data

def formatResidue_ThreeLetterCodeToOne(residue):
    if type(residue) == topology.Residue: # openmm.app.topology.Residue
        residue_name = residue.name
    else:
        residue_name = residue.get_resname()
    try:
        oneLetter = three_to_one[residue_name]
    except:
        print(residue.__dict__)
        print(f"Unknown residue: {residue.get_full_id()}, treat as ALA")
        residue_name = "ALA"
    return three_to_one[residue_name]

def getSeqFromCleanPdb(input_pdb_filename, chains='A', writeFastaFile=False):
    cleaned_pdb_filename = input_pdb_filename.replace("openmmawsem", "cleaned")
    pdb = ''.join(input_pdb_filename.split('-')[:-1])
    fastaFile = pdb + ".fasta"
    file_type = cleaned_pdb_filename[-3:]
    io_class = get_openmm_io_class(file_type)
    top = io_class(cleaned_pdb_filename).getTopology()
    m = {chain.id:chain for chain in top.chains()} # matching variable name from older version of the function
    assert list(m.keys()) == list(chains), f'list(m.keys): {list(m.keys())}, list(chains): {list(chains)}'
    seq = ""
    if writeFastaFile:
        with open(fastaFile, "w") as out:
            for chain in chains:
                out.write(f">{pdb.upper()}:{chain}\n")
                c = m[chain]
                chain_seq = ""
                for residue in c.residues():
                    chain_seq += formatResidue_ThreeLetterCodeToOne(residue)
                out.write("\n".join(textwrap.wrap(chain_seq, width=80))+"\n")
                seq += chain_seq
    else:
        for chain in chains:
            c = m[chain]
            chain_seq = ""
            for residue in c.residues():
                chain_seq += formatResidue_ThreeLetterCodeToOne(residue)
            seq += chain_seq
    return seq

def getSeq(input_pdb_filename, chains='A', writeFastaFile=False, fromPdb=False, fromFasta=None):
    if fromPdb:
        # Path.replace() doesn't work like str.replace(), so we have to converts Paths to strs
        # and then back again.
        # Fortunately, Path.__str__() returns the appropriately formatted string for the OS
        cleaned_pdb_filename = Path(str(input_pdb_filename).replace("openmmawsem.pdb", "cleaned.pdb"))
        pdb = Path(str(input_pdb_filename).replace("-openmmawsem.pdb", "")) 
        fastaFile = Path(f'{str(pdb)}.fasta') # returns a new Path 

        s = PDBParser().get_structure("X", cleaned_pdb_filename)
        m = s[0] # model 0
        seq = ""
        if writeFastaFile:
            with open(fastaFile, "w") as out:
                for chain in chains:
                    out.write(f">{pdb.upper()}:{chain}\n")
                    c = m[chain]
                    chain_seq = ""
                    for residue in c:
                        chain_seq += formatResidue_ThreeLetterCodeToOne(residue)
                    out.write("\n".join(textwrap.wrap(chain_seq, width=80))+"\n")
                    seq += chain_seq
        else:
            for chain in chains:
                c = m[chain]
                chain_seq = ""
                for residue in c:
                    chain_seq += formatResidue_ThreeLetterCodeToOne(residue)
                seq += chain_seq
    elif fromFasta:
        seq = ""
        with open(fromFasta) as f:
            for line in f:
                if line[0] == ">":
                    pass
                else:
                    # print(line)
                    seq += line.strip()
    return seq

def fixPymolPdb(location):
    # location = "1r69.pdb"
    with open("tmp", "w") as out:
        with open(location, "r") as f:
            for line in f:
                info = list(line)
                if len(info) > 21:
                    info[21] = "A"
                out.write("".join(info))
    os.system(f"mv tmp {location}")

def download(pdb_id):
    if not os.path.isfile(f"{pdb_id}.pdb"):
        PDBList().retrieve_pdb_file(pdb_id.lower(), pdir='.', file_format='pdb')
        os.rename("pdb%s.ent" % pdb_id, f"{pdb_id}.pdb")

class OpenMMAWSEMSystem:
    def __init__(self, pdb_filename, chains='A', xml_filename=xml, k_awsem=1.0, seqFromPdb=None, includeLigands=False, periodic=False):
        # read PDB
        if type(pdb_filename) is not Path:
            pdb_filename = Path(pdb_filename)
        io_class = get_openmm_io_class(pdb_filename.suffix[1:])
        self.pdb = io_class(str(pdb_filename))
        self.forcefield = ForceField(str(xml_filename))
        self.periodic = periodic
        if not includeLigands:
            self.system = self.forcefield.createSystem(self.pdb.topology)
            # define convenience variables
            self.nres = self.pdb.topology.getNumResidues()
            self.natoms = self.pdb.topology.getNumAtoms()
            self.residues = list(self.pdb.topology.residues())
            self.resi = [x.residue.index for x in list(self.pdb.topology.atoms())]
            self.resids = [x.residue.id for x in list(self.pdb.topology.atoms())]
            self.resnames = {residue.index+1:residue.name for residue in self.residues} # use 1-indexed residue index
            # build lists of atoms and residue types
            # self.atom_lists,self.res_type=build_lists_of_atoms(self.nres, self.residues)
            self.atom_lists,self.res_type=build_lists_of_atoms_2(self.nres, self.residues, self.pdb.topology.atoms())
        if includeLigands:
            print(self.pdb.topology)
            [templates, names] = self.forcefield.generateTemplatesForUnmatchedResidues(self.pdb.topology)
            # a = templates[0]
            for a in templates:
                for a1 in a.atoms:
                    # we can define the types of ligand atoms using their names.
                    a1.type = a1.name
                    # if a1.element.symbol == "C":
                    #     a1.type = "CA"
                    # else:
                    #     a1.type = a1.element.symbol
                    # a1.type = a1.element.symbol
                self.forcefield.registerResidueTemplate(a)

            res_list = list(self.pdb.topology.residues())
            atom_list = list(self.pdb.topology.atoms())
            protein_resNames = ["NGP", "IGL", "IPR"]
            DNA_resNames = ["DA", "DC", "DT", "DG"]
            protein_res_list = []
            DNA_res_list = []
            ligand_res_list = []
            for res in res_list:
                if res.name in protein_resNames:
                    protein_res_list.append(res)
                elif res.name in DNA_resNames:
                    DNA_res_list.append(res)
                else:
                    ligand_res_list.append(res)

            protein_atom_list = []
            DNA_atom_list = []
            ligand_atom_list = []
            for atom in atom_list:
                if atom.residue.name in protein_resNames:
                    protein_atom_list.append(atom)
                elif atom.residue.name in DNA_resNames:
                    DNA_atom_list.append(atom)
                else:
                    ligand_atom_list.append(atom)
            self.ligand_res_list = ligand_res_list
            self.ligand_atom_list = ligand_atom_list
            self.protein_atom_list = protein_atom_list
            self.system = self.forcefield.createSystem(self.pdb.topology)
            self.nres = len(protein_res_list)
            self.residues = protein_res_list
            print(f"Number of residues: {self.nres}, Number of ligands: {len(ligand_res_list)}")
            # self.natoms = len(protein_atom_list)
            self.natoms = self.pdb.topology.getNumAtoms()
            self.resi = [x.residue.index if x in protein_atom_list else -1 for x in atom_list]
            self.atom_lists,self.res_type=build_lists_of_atoms_2(self.nres, self.residues, protein_atom_list)


        # print(self.atom_lists,self.res_type)
        self.n =self.atom_lists['n']
        self.h =self.atom_lists['h']
        self.ca=self.atom_lists['ca']
        self.c =self.atom_lists['c']
        self.o =self.atom_lists['o']
        self.cb=self.atom_lists['cb']

        self.chain_starts, self.chain_ends = get_chain_starts_and_ends(self.residues)
        # setup virtual sites
        # if Segmentation fault in setup_virtual_sites, use ensure_atom_order
        setup_virtual_sites(self.nres, self.system, self.n, self.h, self.ca, self.c, self.o,
                            self.cb, self.res_type, self.chain_starts, self.chain_ends)
        # setup bonds
        self.bonds = setup_bonds(self.nres, self.n, self.h, self.ca, self.c, self.o,
                            self.cb, self.res_type, self.chain_starts, self.chain_ends)
        # set overall scaling
        self.k_awsem = k_awsem
        # keep track of force names for output purposes
        self.force_names = []
        # save seq info
        if seqFromPdb is None:
            self.seq = getSeq(pdb_filename, chains=chains, fromPdb=True)
        else:
            self.seq = seqFromPdb
        # elif seqFromPdb == 0:
        #     self.seq = getSeq(pdb_filename, chains=chains, fromFasta=True)

    def addForces(self, forces):
        for i, (force) in enumerate(forces):
            self.addForce(force)
            force.setForceGroup(i+1)

    def addForce(self, force):
        self.system.addForce(force)

    def addForcesWithDefaultForceGroup(self, forces):
        for i, (force) in enumerate(forces):
            self.addForce(force)



def read_trajectory_pdb_positions(pdb_trajectory_filename):
    import uuid, os
    pdb_trajectory_contents = open(pdb_trajectory_filename).read().split("MODEL")[1:]
    pdb_trajectory_contents = ['\n'.join(x.split('\n')[1:]) for x in pdb_trajectory_contents]
    pdb_trajectory = []
    for i, pdb_contents in enumerate(pdb_trajectory_contents):
        temporary_file_name = str(uuid.uuid4())
        temporary_pdb_file = open(temporary_file_name, 'w')
        temporary_pdb_file.write(pdb_contents)
        temporary_pdb_file.close()
        pdb = PDBFile(temporary_file_name)
        pdb_trajectory.append(pdb)
        os.remove(temporary_file_name)
    return pdb_trajectory

def compute_order_parameters(openmm_awsem_pdb_file, pdb_trajectory_filename, order_parameters,
            platform_name='CPU', k_awsem=1.0, compute_mdtraj=False, rmsd_reference_structure=None,
            compute_total_energy=True, energy_columns=None, xml_filename="awsem.xml"):
    pdb_trajectory = read_trajectory_pdb_positions(pdb_trajectory_filename)
    order_parameter_values = []
    for i, order_parameter in enumerate(order_parameters):
        order_parameter_values.append([])
        oa = OpenMMAWSEMSystem(openmm_awsem_pdb_file, k_awsem=k_awsem, xml_filename=xml_filename)
        platform = Platform.getPlatformByName(platform_name) # OpenCL, CUDA, CPU, or Reference
        integrator = VerletIntegrator(2*femtoseconds)
        oa.addForce(order_parameter)
        simulation = Simulation(oa.pdb.topology, oa.system, integrator, platform)
        for pdb in pdb_trajectory:
            simulation.context.setPositions(pdb.positions)
            state = simulation.context.getState(getEnergy=True)
            order_parameter_values[i].append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
    if compute_mdtraj:
        md_traj_order_parameters = compute_mdtraj_order_parmeters(pdb_trajectory_filename, rmsd_reference_structure=rmsd_reference_structure)
        for key, value in md_traj_order_parameters.items():
            if len(value.flatten()) == len(pdb_trajectory):
                order_parameter_values.append(value)
    if compute_total_energy:
        order_parameter_values.append([])
        for i in range(len(pdb_trajectory)):
            order_parameter_values[-1].append(np.sum([order_parameter_values[x-1][i] for x in energy_columns]))
    if not compute_mdtraj:
        return np.array(order_parameter_values)
    else:
        return np.array(order_parameter_values), md_traj_order_parameters

def compute_mdtraj_order_parmeters(trajectory_file, rmsd_reference_structure=None):
    # documentation: http://mdtraj.org/1.8.0/analysis.html#
    trajectory = md.load(trajectory_file)

    return_values = []
    return_value_names = []

    if not rmsd_reference_structure == None:
        reference = md.load(rmsd_reference_structure)
        rmsd = md.rmsd(trajectory, reference)
        return_values.append(rmsd)
        return_value_names.append("RMSD")

    hydrogen_bonds = np.array([np.sum(x) for x in md.kabsch_sander(trajectory)])
    return_values.append(hydrogen_bonds)
    return_value_names.append("HBondEnergy")

    ss = md.compute_dssp(trajectory)
    shape = ss.shape
    transdict = dict(zip(list(set(list(ss.flatten()))),range(len(list(set(list(ss.flatten())))))))
    ss = np.array([transdict[x] for x in ss.flatten()]).reshape(shape).T
    return_values.append(ss)
    return_value_names.append("SecondaryStructure")

    rg = md.compute_rg(trajectory)
    return_values.append(rg)
    return_value_names.append("Rg")

    distances, residue_pairs = md.compute_contacts(trajectory, scheme='ca')
    contacts = md.geometry.squareform(distances, residue_pairs)
    return_values.append(contacts)
    return_value_names.append("Contacts")

    return dict(zip(return_value_names, return_values))

def plot_free_energy(order_parameter_file, labels, bins=20, discard=.2, twodlimits=[[0,1],[0,1]]):
    import matplotlib as mpl
    import scipy.ndimage
    mpl.rcParams['font.size'] = 24

    if len(labels) > 2:
        print("Too many labels to plot.")
        return
    order_parameters = np.loadtxt(order_parameter_file).T
    data_labels = open(order_parameter_file).readlines()[0].split()[1:]
    data = []
    for label in labels:
        raw_data = order_parameters[data_labels.index(label)]
        data_after_discard = raw_data[int(len(raw_data)*discard):]
        comparison_data_set_size = (1.0-discard)/2
        data_set_1 = raw_data[int(len(raw_data)*discard):int(len(raw_data)*(discard+comparison_data_set_size))]
        data_set_2 = raw_data[int(len(raw_data)*(discard+comparison_data_set_size)):]
        data.append(data_after_discard)
    if len(labels) == 1:
        hist, bins = np.histogram(data[0], density=True, bins=bins)
        hist /= np.sum(hist)
        f = -np.log(hist)
        bin_centers = (bins[:-1]+bins[1:])/2
        hist, bins = np.histogram(data_set_1, density=True, bins=bins)
        hist /= np.sum(hist)
        f_1 = -np.log(hist)
        bin_centers_1 = (bins[:-1]+bins[1:])/2
        hist, bins = np.histogram(data_set_2, density=True, bins=bins)
        hist /= np.sum(hist)
        f_2 = -np.log(hist)
        bin_centers_2 = (bins[:-1]+bins[1:])/2
        plt.figure(figsize=(12,8))
        plt.plot(bin_centers, f, label="All data")
        plt.plot(bin_centers_1, f_1, label="Split data 1")
        plt.plot(bin_centers_2, f_2, label="Split data 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(labels[0])
        plt.xlabel(labels[0])
        plt.ylabel("Free Energy (kT)")
    if len(labels) == 2:
        H, xedges, yedges = np.histogram2d(data[0], data[1], bins=bins)
        xcenters = (xedges[:-1] + xedges[1:])/2.0
        ycenters = (yedges[:-1] + yedges[1:])/2.0
        H /= np.sum(H)
        H = -np.log(H)
        H -= np.min(H)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        ax.set_title("%s vs. %s" % (labels[0], labels[1]))
        ax.set_xlim(twodlimits[0])
        ax.set_ylim(twodlimits[1])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        X, Y = np.meshgrid(xcenters, ycenters)
        plt.contourf(X, Y, H, cmap=plt.cm.get_cmap('jet'))

        plt.colorbar()
        plt.tight_layout()
        plt.show()

def compute_perturbed_energies(openmm_awsem_pdb_file, pdb_trajectory_filename, perturbations, order_parameter_values, platform_name='CPU', k_awsem=1.0, total_energy_column=15):
    pdb_trajectory = read_trajectory_pdb_positions(pdb_trajectory_filename)
    all_perturbed_energies = []
    for i, perturbation in enumerate(perturbations):
        # compute new energies
        perturbed_energies = []
        for j, energy_term in enumerate(perturbation):
            perturbed_energies.append([])
            oa = OpenMMAWSEMSystem(openmm_awsem_pdb_file, k_awsem=k_awsem)
            platform = Platform.getPlatformByName(platform_name) # OpenCL, CUDA, CPU, or Reference
            integrator = VerletIntegrator(2*femtoseconds)
            oa.addForce(energy_term[1])
            simulation = Simulation(oa.pdb.topology, oa.system, integrator, platform)
            for pdb in pdb_trajectory:
                simulation.context.setPositions(pdb.positions)
                state = simulation.context.getState(getEnergy=True)
                perturbed_energies[j].append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        perturbed_energies = np.array(perturbed_energies)
        total_perturbed_energies = np.sum(perturbed_energies, axis=0)
        # compute new total energy by subtracting original value and adding new value
        new_total_energy = np.array(order_parameter_values[:, total_energy_column-1])
        for j, energy_term in enumerate(perturbation):
            new_total_energy -= order_parameter_values[:, energy_term[0]-1]
        new_total_energy += total_perturbed_energies
        all_perturbed_energies.append(new_total_energy)
    all_perturbed_energies = np.array(all_perturbed_energies)
    return all_perturbed_energies.T

def pick_structures(label, conditions_string, metadata_file, reference_structure, num_snapshots=6000, order_parameter_file_name="order_parameters.txt", pdb_trajectory_filename="output.pdb"):
    # parse restrictions (including which file to pull columns from)
    # read in data (including extra files if necessary)
    # go through the data and filter out snapshots that do not satisfy the criteria
    # select a subset of the structures that satisfy the constraints

    def parse_conditions_string(conditions_string):
        conditions = []
        condition_signs = []
        conditions_string = conditions_string.split()
        for condition in conditions_string:
            if "gt" in condition:
                condition_signs.append("+")
                condition = condition.split("gt")
            if "lt" in condition:
                condition_signs.append("-")
                condition = condition.split("lt")
            conditions.append(condition)
        return conditions, condition_signs

    def load_all_data(metadata_file, conditions):
        data_files = list(np.loadtxt(metadata_file, dtype=str)[:,0])
        num_files = len(data_files)
        num_conditions = len(conditions)
        # Load all data into array
        data_array = np.zeros((num_files, num_conditions, num_snapshots))
        for i, data_file in enumerate(data_files):
            all_order_parameters = np.loadtxt(data_file)
            for j, condition in enumerate(conditions):
                data_array[i][j] = all_order_parameters[0:num_snapshots,int(condition[0])-1]

        data_array = np.swapaxes(data_array,1,2)
        return data_array

    def write_selected_pdbs(pdb_files, snapshots_in_pdb_files):
        structure_index = 1
        selected_pdbs = open("%s.pdb" % label, 'w')
        for pdb_file in pdb_files:
            pdb_trajectory_contents = open(pdb_file).read().split("MODEL")[1:]
            pdb_trajectory_contents = ['\n'.join(x.split('\n')[1:]) for x in pdb_trajectory_contents]
            for snapshot in snapshots_in_pdb_files[pdb_files.index(pdb_file)]:
                selected_pdbs.write("MODEL        %d\n" % structure_index)
                selected_pdbs.write(pdb_trajectory_contents[snapshot])
                structure_index += 1
        selected_pdbs.close()

    # Lists
    files_array = list(np.loadtxt(metadata_file, dtype=str)[:,0])
    conditions, condition_signs = parse_conditions_string(conditions_string)
    data_array = load_all_data(metadata_file, conditions)

    # File names and parameters
    output_file_name = label + ".dat"

    structure_index = 1

    # loop over data and output those points that satisfy all conditions
    output_file = open(output_file_name, "w")
    pdb_files = []
    snapshots_in_pdb_files = []

    # loop over files
    for i, data_file in enumerate(files_array):
        # loop over snapshots
        for j, snapshot in enumerate(data_array[i]):
            bad_condition = False
            # loop over conditions
            for k, condition in enumerate(conditions):
                condition_boundary = float(condition[1])
                # If all conditions are satisfied, print out the data
                if condition_signs[k] == "+":
                    if not data_array[i][j][k] > condition_boundary: bad_condition = True
                elif condition_signs[k] == "-":
                    if not data_array[i][j][k] < condition_boundary: bad_condition = True
                else:
                    print("Bad condition argument.")
                    sys.exit()
            if not bad_condition:
                output_file.write("%d %s %s\n" % (structure_index, data_file, j+1))
                pdb_file = data_file.replace(order_parameter_file_name, pdb_trajectory_filename)
                if not pdb_file in pdb_files:
                    pdb_files.append(pdb_file)
                    snapshots_in_pdb_files.append([])
                snapshots_in_pdb_files[pdb_files.index(pdb_file)].append(j)
                structure_index += 1
    output_file.close()
    write_selected_pdbs(pdb_files, snapshots_in_pdb_files)

    pymol_script = open("%s.pml" % label, 'w')
    pymol_script.write("load %s\n" % reference_structure)
    pymol_script.write("load_traj %s.pdb\n" % label)
    object_name = os.path.basename(reference_structure)[:-4]
    pymol_script.write("intra_fit %s\n" % object_name)
    pymol_script.write("smooth\n")
    pymol_script.close()



def test_Protein_fromCoarsePDB():
    pass


def test_Protein_fromPDB():
    pass


def test_Protein_parseConfigurationFile():
    pass


def test_Protein_computeTopology():
    pass

