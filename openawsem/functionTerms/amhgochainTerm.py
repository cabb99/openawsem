from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from Bio.PDB.PDBParser import PDBParser
from itertools import product, combinations
import numpy as np
import pandas as pd

def read_amhgo_chain_structure(oa, pdb_file, ref_head, amhgo_min_seq_sep=4, amhgo_contact_threshold=0.8*nanometers, amhgo_well_width=0.1):
    structure_interactions = []
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_file)
    chains=structure[0]
    residues = []
    for chain in chains:
     for x in chain:
        residues.append(x)
    print("amh_go chain model applied to ", len(residues), "res")
    for i, residue_i in enumerate(residues):
         for j, residue_j in enumerate(residues):
             ca_list = []
             cb_list = []
             atom_list_i = []
             atom_list_j = []
             if i-j >= amhgo_min_seq_sep:  # taking the signed value to avoid double counting
                 ca_i = residue_i['CA']
                 ca_list.append(ca_i)
                 atom_list_i.append(ca_i)
                 ca_j = residue_j['CA']
                 ca_list.append(ca_j)
                 atom_list_j.append(ca_j)
                 if not residue_i.get_resname() == "GLY":
                     cb_i = residue_i['CB']
                     cb_list.append(cb_i)
                     atom_list_i.append(cb_i)
                 if not residue_j.get_resname() == "GLY":
                     cb_j = residue_j['CB']
                     cb_list.append(cb_j)
                     atom_list_j.append(cb_j)
                 for atom_i, atom_j in product(atom_list_i, atom_list_j):
                     r_ijN = abs(atom_i - atom_j)/10.0*nanometers # convert to nm
                     if r_ijN <= amhgo_contact_threshold:
                         sigma_ij = amhgo_well_width*abs(i-j)**0.15 # 0.1 nm = 1 A
                         gamma_ij = 1.0
                         if atom_i in ca_list:
                             i_index = oa.ca[i+ ref_head]
                         if atom_i in cb_list:
                             i_index = oa.cb[i+ ref_head]
                         if atom_j in ca_list:
                             j_index = oa.ca[j+ ref_head]
                         if atom_j in cb_list:
                             j_index = oa.cb[j+ ref_head]
                         structure_interaction = [i_index, j_index, [gamma_ij, r_ijN, sigma_ij]]
                         structure_interactions.append(structure_interaction)
#                         print(i, j, structure_interaction)
    return structure_interactions

def additive_amhgo_chain_term(oa, pdb_file, ref_head, k_amhgo=4.184, amhgo_min_seq_sep=3, amhgo_contact_threshold=0.8*nanometers, amhgo_well_width=0.1, forceGroup=24):
    import itertools
    # multiply interaction strength by overall scaling
    k_amhgo *= oa.k_awsem
    # create contact force
    amhgo = CustomBondForce(f"-{k_amhgo}*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    # # add global parameters
    #amhgo.addGlobalParameter("k_amhgo", k_amhgo)
    amhgo.addPerBondParameter("gamma_ij")
    amhgo.addPerBondParameter("r_ijN")
    amhgo.addPerBondParameter("sigma_ij")
    # create bonds
    structure_interactions = read_amhgo_chain_structure(oa, pdb_file, ref_head, amhgo_min_seq_sep, amhgo_contact_threshold, amhgo_well_width=amhgo_well_width)
    for structure_interaction in structure_interactions:
        amhgo.addBond(*structure_interaction)
    amhgo.setForceGroup(forceGroup)
    print("AMH-GO chain structure based term is ON")
    return amhgo
