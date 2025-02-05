try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except ModuleNotFoundError:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
import numpy as np
import pandas as pd
import pickle
#from .fragmentMemoryTerms import *

###################################
#### MODIFIED FOR templateTerms ONLY FOR BACKWARDS COMPATIBILITY WITH GRO FRAGMEMS
def get_openmm_io_class(file_type,full_name=None):
    if file_type == "pdb":
        io_class = PDBFile
    elif file_type == "cif":
        io_class = PDBxFile
    elif file_type == "gro":
        # for backwards compatibility with fragment method
        df = pd.read_csv(full_name, skiprows=2, sep="\s+", header=None, names=["Res_id", "Res", "Type", "i", "x", "y", "z"])
        top = Topology()
        chain = top.addChain() # we don't care about chain id if we're just reading things in for a fragment memory
        pos = []
        res_id_dict = {}
        for index, row in df.iterrows():
            if row['Res_id'] not in res_id_dict.keys():
                residue = top.addResidue(row['Res'], chain, id=row['Res_id'])
                res_id_dict.update({row['Res_id']:residue})
            top.addAtom(row['Type'],element.Element.getBySymbol(row['Type'][0]),res_id_dict[row['Res_id']],id=row['i'])
            pos.append([float(row['x']),float(row['y']),float(row['z'])])
        pos = np.array(pos)
        pos = Quantity(pos,unit=nanometer)
        class DummyOpenmmReader:
            def __init__(self,top,pos):
                self.top = top
                self.pos = pos
            def getTopology(self):
                return top
            def getPositions(self,asNumpy=False):
                return pos
            def __call__(self,_):
                return self
        io_class = DummyOpenmmReader(top,pos)
    else:
        raise ValueError(f"Expected file_type 'pdb' or 'cif' but got file_type={file_type}") 
    return io_class
###################################

def read_reference_structure_for_q_calculation_4(oa, contact_threshold,rnative_dat, min_seq_sep=3, max_seq_sep=np.inf):
    # use contact matrix for Q calculation
    # this change use the canonical Qw/Qo calculation for reference Q
    # for Qw calculation is 0; Qo is 1;
    in_rnative = np.loadtxt(rnative_dat)  # read in rnative_dat file for Q calculation
    structure_interactions = []
    chain_start = 0
    count = 0
    for i in range(oa.nres):
        chain_start += count
        count = 0
        for j in range(oa.nres):
            count +=1
            # if abs(i-j) >= min_seq_sep and abs(i-j) <= max_seq_sep:  # taking the signed value to avoid double counting
            if j-i >= min_seq_sep and j-i <= max_seq_sep:  # taking the signed value to avoid double counting
                r_ijN = in_rnative[i][j]/10.0 * nanometers  # convert to nm
                if r_ijN < contact_threshold:
                    continue
                sigma_ij = 0.1*abs(i-j)**0.15  # 0.1 nm = 1 A
                gamma_ij = 1.0
                i_index = oa.ca[i]
                j_index = oa.ca[j]
                structure_interaction = [i_index, j_index, [gamma_ij, r_ijN, sigma_ij]]
                # print(i, j, r_ijN)
                structure_interactions.append(structure_interaction)
    return structure_interactions



def q_value_dat(oa, contact_threshold, rnative_dat="rnative.dat", min_seq_sep=3, max_seq_sep=np.inf):
    ### Added by Mingchen
    ### this function is solely used for template based modelling from rnative.dat file
    ### for details, refer to Chen, Lin & Lu Wolynes JCTC 2018
    structure_interactions_tbm_q = read_reference_structure_for_q_calculation_4(oa, contact_threshold=contact_threshold,rnative_dat=rnative_dat, min_seq_sep=min_seq_sep, max_seq_sep=max_seq_sep)
    normalization = len(structure_interactions_tbm_q)
    qvalue_dat = CustomBondForce(f"(1/{normalization})*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    qvalue_dat.addPerBondParameter("gamma_ij")
    qvalue_dat.addPerBondParameter("r_ijN")
    qvalue_dat.addPerBondParameter("sigma_ij")

    for structure_interaction_tbm_q in structure_interactions_tbm_q:
        qvalue_dat.addBond(*structure_interaction_tbm_q)
    return qvalue_dat


def tbm_q_term(oa, k_tbm_q, rnative_dat="rnative.dat", tbm_q_min_seq_sep=3, tbm_q_cutoff=0.2*nanometers, tbm_q_well_width=0.1, target_q=1.0, forceGroup=26):
    ### Added by Mingchen Chen
    ### this function is solely used for template based modelling from rnative.dat file
    ### for details, refer to Chen, Lin & Lu Wolynes JCTC 2018
    print("TBM_Q term ON")
    tbm_q = CustomCVForce(f"{k_tbm_q}*(q-{target_q})^2")
    q = q_value_dat(oa, contact_threshold=tbm_q_cutoff, rnative_dat=rnative_dat, min_seq_sep=tbm_q_min_seq_sep, max_seq_sep=np.inf)
    tbm_q.addCollectiveVariable("q", q)
    tbm_q.setForceGroup(forceGroup)
    return tbm_q



def fragment_memory_term(oa, k_fm=0.04184, frag_file_list_file="./frag.mem", npy_frag_table="./frag_table.npy",
                    min_seq_sep=3, max_seq_sep=9, fm_well_width=0.1, UseSavedFragTable=True, caOnly=False, forceGroup=23,
                    testing=False):
    # 0.8368 = 0.01 * 4.184 # in kJ/mol, converted from default value in LAMMPS AWSEM
    k_fm *= oa.k_awsem
    frag_table_rmin = 0
    frag_table_rmax = 5  # in nm
    frag_table_dr = 0.01
    r_array = np.arange(frag_table_rmin, frag_table_rmax, frag_table_dr)
    number_of_atoms = oa.natoms
    r_table_size = int((frag_table_rmax - frag_table_rmin)/frag_table_dr)  # 500 here.
    raw_frag_table = np.zeros((number_of_atoms, 6*(1+max_seq_sep), r_table_size))
    raw_frag_table_count = np.zeros((number_of_atoms, 6*(1+max_seq_sep), r_table_size))
    data_dic = {}
    # data_dic maps CA and CB atoms in our target structure to atom types and residue indices
    # ?? replace variable name res_id with res_index to better match distinction between Topology.Residue.id vs. Topology.Residue.index? ??
    is_glycine = [] # keep track of whether atom is part of a glycine residue
    for i in range(oa.natoms):
        if i in oa.ca:
            one_indexed_res_index = oa.resi[i]+1    # oa.resi start with 0, but pdb residue id start with 1
                                                    # (oa.resi is a wrapper for openmm.topology.Residue.index)
            data_dic[("CA", one_indexed_res_index)] = i
            res_id = oa.resids[i]
            if oa.resnames[one_indexed_res_index] == "GLY": # oa.resnames gives the name of each residue
                                                            # where the length of the dictionary is equal to the number of residues
                                                            # note that keys are residues ids instead of residue indices
                is_glycine.append(True)
            else:
                is_glycine.append(False)
        if i in oa.cb:
            one_indexed_res_index = oa.resi[i]+1
            data_dic[("CB", one_indexed_res_index)] = i
            is_glycine.append(False) # glycine doesn't have a CB
    if testing: # log interactions for each atom
        full_interaction_info = {atom_index:'' for atom_index in data_dic.values()}
    frag_location_pre = os.path.dirname(frag_file_list_file)
    # frag_file_list_file = frag_location_pre + "frags.mem"
    # frag_table_file = frag_location_pre + "frag_table.npy"
    frag_table_file = npy_frag_table

    if os.path.isfile(frag_table_file) and UseSavedFragTable:
        print(f"Reading Fragment table from {frag_table_file}.")
        # frag_table, interaction_list, interaction_pair_to_bond_index = np.load(frag_table_file, allow_pickle=True)
        with open(frag_table_file, 'rb') as f:
            frag_table, interaction_list, interaction_pair_to_bond_index = pickle.load(f)
        print(f"Fragment table loaded, number of bonds: {len(interaction_list)}")
        frag_file_list = []
    else:
        print(f"Loading Fragment files")
        frag_file_list = pd.read_csv(frag_file_list_file, skiprows=4, sep="\s+", names=["location", "target_start", "fragment_start", "frag_len", "weight"])
        interaction_list = set()
    




    for frag_index in range(len(frag_file_list)):
        if frag_index % 100 == 0:
            print(f"frag_index: {frag_index}")
        location = frag_file_list["location"].iloc[frag_index]
        frag_name = os.path.join(frag_location_pre, location)
        frag_len = frag_file_list["frag_len"].iloc[frag_index]
        weight = frag_file_list["weight"].iloc[frag_index]
        target_start = frag_file_list["target_start"].iloc[frag_index]  # residue id
        fragment_start = frag_file_list["fragment_start"].iloc[frag_index]  # residue id

        io_class = get_openmm_io_class(frag_name[-3:],full_name=frag_name)
        temp = io_class(frag_name)
        frag_top = temp.getTopology()
        frag_pos = temp.getPositions(asNumpy=True)
        frag_ca_cb_indices = []
        frag_ca_cb_residue_ids = []
        frag_ca_cb_atom_types = []
        for residue in frag_top.residues():
            #if fragment_start <= int(residue.index)+1 < fragment_start+frag_len: 
                # Assuming target (the thing we're going to simulate) starts at residue 1, then correcting 0-indexed memory residue.index to compare.
                # Instead of int(residue.index)+1, we could have said residue.id because we're assuming 1-indexed id
            # ^^^ wait, but we're looping over frag, not target! int(residue.index)+1 != residue.id in general for frags!
            if fragment_start <= int(residue.id) < fragment_start+frag_len: 
                found_CA = False
                found_CB = False
                for atom in residue.atoms():
                    if atom.name == "CA":
                        found_CA = True
                        frag_ca_cb_indices.append(atom.index)
                        #frag_ca_cb_residue_ids.append(residue.index+1)
                        frag_ca_cb_residue_ids.append(residue.id)
                        frag_ca_cb_atom_types.append(atom.name)
                    elif atom.name == "CB":
                        found_CB = True
                        frag_ca_cb_indices.append(atom.index)
                        #frag_ca_cb_residue_ids.append(residue.index+1)
                        frag_ca_cb_residue_ids.append(residue.id)
                        frag_ca_cb_atom_types.append(atom.name)
                if not (found_CA and found_CB) and residue.name != "GLY":
                    raise AssertionError(f"missing atom in fragment {frag_name} residue index (0-indexed) {residue.index}, number {residue.id}, {residue.name}")
        f = frag_pos[frag_ca_cb_indices,:]
        assert f.shape[0] == len(frag_ca_cb_indices)
        assert f.shape[0] == len(frag_ca_cb_atom_types)
        w_m = weight
        gamma_ij = 1
        for i in range(f.shape[0]):
            for j in range(i, f.shape[0]):
                res_id_i = frag_ca_cb_residue_ids[i] #frag["Res_id"].iloc[i]
                res_id_j = frag_ca_cb_residue_ids[j] #frag["Res_id"].iloc[j]
                target_res_id_i = int(res_id_i) - fragment_start + target_start
                target_res_id_j = int(res_id_j) - fragment_start + target_start
                seq_sep = int(res_id_j) - int(res_id_i)
                if seq_sep > max_seq_sep:
                    continue
                if seq_sep < min_seq_sep:
                    continue
                # data_dic holds the information from the system that we are going to simulate.
                # We need to figure out the separation of the residues in our target system 
                # that correspond to the residues in our memory so that we can compute the fm potential.
                #  
                # For the short fragments that we typically use in openawsem, the sequence separation
                # in our target is the same as the sequence separation in our memory (either the alignment
                # algorithm doesn't consider gaps or the gap penalty is large enough to prevent gaps).
                # But I guess we are enabling future flexibility by checking the sequence separation of 
                # target residues instead of fragment residues
                i_type = frag_ca_cb_atom_types[i] #frag["Type"].iloc[i]
                j_type = frag_ca_cb_atom_types[j] #frag["Type"].iloc[j]
                #print(i_type)
                #print(len(i_type))
                i_corresponds_to_GLY_CB = (i_type == "CB" and oa.resnames[target_res_id_i] == "IGL")
                try:
                    j_corresponds_to_GLY_CB = (j_type == "CB" and oa.resnames[target_res_id_j] == "IGL")
                except KeyError:
                    print(oa.resnames)
                    print(f'res_id_i: {res_id_i}')
                    print(f'res_id_j: {res_id_j}')
                    print(f'fragment start: {fragment_start}')
                    print(f'target start: {target_start}')
                    print(target_res_id_i)
                    print(target_res_id_j)
                    raise
                if i_corresponds_to_GLY_CB or j_corresponds_to_GLY_CB:
                    continue # no CB in glycine residues
                try:   
                    correspond_target_i = data_dic[(i_type, int(target_res_id_i))]
                    correspond_target_j = data_dic[(j_type, int(target_res_id_j))]
                    correspond_target_i = int(correspond_target_i)
                    correspond_target_j = int(correspond_target_j)
                    i_j_sep = int(correspond_target_j - correspond_target_i) #why do we take int(a-b) if a and b are already ints?
                    #if int(target_res_id_i) == 11 and int(target_res_id_j) == 20:
                    #    print(frag_name,fragment_start,frag_len)
                    #    print('11 and 20')
                    #    print(i_type,j_type)
                    #    print(i,j)
                    #    print(frag_ca_cb_atom_types[i:i+4])
                    #if int(target_res_id_i) == 23 and int(target_res_id_j) == 32:
                    #    print(frag_name,fragment_start,frag_len)
                    #    print('23 and 32')
                    #    print(i_type,j_type)
                    #    print(i,j)
                    #    print(frag_ca_cb_atom_types[i:i+4])
                    #if int(target_res_id_i) == 29 and int(target_res_id_j) == 38:
                    #    print(frag_name,fragment_start,frag_len)
                    #    print('29 to 38')
                    #    print(i_type,j_type)
                    #    print(i,j)
                    #    print(frag_ca_cb_atom_types[i:i+4])
                except Exception as e:
                    print(data_dic)
                    print("\n\n\n\n\n\n\n\n\n\n\n\n\nTHAT EXCEPTION")
                    print(frag_index)
                    print(frag_name)
                    print(i)
                    print(j)
                    print(target_res_id_i,target_res_id_j)
                    print(f'fragment start: {fragment_start}')
                    print(f'target start: {target_start}')
                    print(i_type,j_type)
                    print(oa.resnames[target_res_id_i], oa.resnames[target_res_id_j])
                    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
                    print(oa.resnames)
                    raise
                
                fi_x = round(f[i,0].value_in_unit(nanometer),3) #f[i][4] 
                fi_y = round(f[i,1].value_in_unit(nanometer),3)#f[i][5]
                fi_z = round(f[i,2].value_in_unit(nanometer),3)#f[i][6]

                fj_x = round(f[j,0].value_in_unit(nanometer),3)#f[j][4]
                fj_y = round(f[j,1].value_in_unit(nanometer),3)#f[j][5]
                fj_z = round(f[j,2].value_in_unit(nanometer),3)#f[j][6]
                # print("----", fi_x, fi_y, fi_z, fj_x, fj_y, fj_z)
                sigma_ij = fm_well_width*seq_sep**0.15
                rm = ((fi_x-fj_x)**2 + (fi_y-fj_y)**2 + (fi_z-fj_z)**2)**0.5

                #if correspond_target_i in [95,98,113,116,131,134,148,151,154,157,324,327]:
                #with open(f'{correspond_target_i}_frags_new.txt','a') as number_write_file:
                #    number_write_file.write(f'{frag_name[:-4]}, i: {i}, j: {j}\n')
                #with open(f'{correspond_target_i}_energy_new.txt','a') as write_file:
                #    write_file.write(str(w_m*gamma_ij*np.exp((r_array-rm)**2/(-2.0*sigma_ij**2))))
                #with open(f'{correspond_target_i}_rm_new.txt','a') as write_file:
                #    write_file.write(f'{rm}\n')
                if testing:
                    full_interaction_info[correspond_target_i] += f'{frag_name[:-4]}, i: {i}, j: {j}\n'
                # w_m is the weight of the memory, gamma_ij is weight of the pairwise interaction, sigma_ij is basically the width of the well
                # typically, we set all w_m=1 for all m and gamma_ij=1 for all m, i, and j
                raw_frag_table[correspond_target_i][i_j_sep] += w_m*gamma_ij*np.exp((r_array-rm)**2/(-2.0*sigma_ij**2))
                raw_frag_table_count[correspond_target_i][i_j_sep] += 1
                interaction_list.add((correspond_target_i, correspond_target_j))
        




    if (not os.path.isfile(frag_table_file)) or (not UseSavedFragTable):
        # Reduce memory usage.
        print("Saving fragment table as npy file to speed up future calculation.")
        number_of_bonds = len(interaction_list)
        frag_table = np.zeros((number_of_bonds, r_table_size))
        interaction_pair_to_bond_index = {}
        for index, (i, j) in enumerate(interaction_list):
            ij_sep = j - i
            assert(ij_sep > 0)
            frag_table[index] = raw_frag_table[i][ij_sep]
            interaction_pair_to_bond_index[(i,j)] = index
        np.save(frag_table_file, np.array((frag_table, interaction_list, interaction_pair_to_bond_index),dtype=object))
        #with open(frag_table_file, 'wb') as f:
        #    pickle.dump((frag_table, interaction_list, interaction_pair_to_bond_index), f)
        #with open(f'tests/data/new_raw_frag_table.npy','wb') as f:
        #    pickle.dump(raw_frag_table,f)
        #with open(f'tests/data/new_raw_frag_table_count.npy','wb') as f:
        #    pickle.dump(raw_frag_table_count,f)
        print(f"All gro files information have been stored in the {frag_table_file}. \
            \nYou might want to set the 'UseSavedFragTable'=True to speed up the loading next time. \
            \nBut we recommend removing the .npy file if you modify the .mem file \
              so that you don't accidentally keep loading the old frag memeories in the .npy file.")
    # fm = CustomNonbondedForce(f"-k_fm*((v2-v1)*r+v1*r_2-v2*r_1)/(r_2-r_1); \
    #                             v1=frag_table(index_smaller, sep, r_index_1);\
    #                             v2=frag_table(index_smaller, sep, r_index_2);\
    #                             index_smaller=min(index1,index2);\
    #                             sep=abs(index1-index2);\
    #                             r_1=frag_table_rmin+frag_table_dr*r_index_1;\
    #                             r_2=frag_table_rmin+frag_table_dr*r_index_2;\
    #                             r_index_2=r_index_1+1;\
    #                             r_index_1=floor(r/frag_table_dr);")
    # for i in range(oa.natoms):
    #     fm.addParticle([i])

    # # add interaction that are cutoff away
    # # print(sorted(interaction_list))
    # for (i, j) in interaction_list:
    #     fm.addInteractionGroup([i], [j])
    # # add per-particle parameters
    # fm.addPerParticleParameter("index")



    # Even though the physical energies represented by the fragment memory term arise from many-body interactions,
    # the fragment term is mathematically expressed as a sum of pairwise interactions,
    # parameterized by a set of memories.
    # 
    # To start, we define the form of the force governing the interaction between a pair of atoms.
    # The force depends on the distance (r) between the two atoms, p1 and p2,
    # and the parameter "index" that depends on the atom indices but not their configuration.
    # the differences between frag_table at different "index" values account for 
    # the different memory distance(s) and weights for that atom pair, which will determine
    # the center and deepness of each gaussian well in the total fm potential for that atom pair,
    # which is a sum of all these gaussian wells. 
    #
    # currently, the gaussians are stored in frag_table and we discretize the distances for some reason,
    # causing frag_table to be two-dimensional and resulting in this interpolation formula
    # -{k_fm}*((v2-v1)*r+v1*r_2-v2*r_1)/(r_2-r_1) in the CustomCompoundBondForce definition
    #
    # for edge case, that r > frag_table_rmax # NOT SURE WHAT THIS MEANS
    max_r_index_1 = r_table_size - 2
    # at the time of writing these comments, frag_table_rmin==0 and frag_table_dr==0.01 and 
    fm = CustomCompoundBondForce(2, f"-{k_fm}*((v2-v1)*r+v1*r_2-v2*r_1)/(r_2-r_1); \
                                v1=frag_table(index, r_index_1);\
                                v2=frag_table(index, r_index_2);\
                                r_1={frag_table_rmin}+{frag_table_dr}*r_index_1;\
                                r_2={frag_table_rmin}+{frag_table_dr}*r_index_2;\
                                r_index_2=r_index_1+1;\
                                r_index_1=min({max_r_index_1}, floor(r/{frag_table_dr}));\
                                r=distance(p1, p2);")
    # openmm will only apply the force to pairs of atoms that we pass to the addBond method
    for (i, j) in interaction_list:
        if caOnly and ((i not in oa.ca) or (j not in oa.ca)):
            continue
        # [i,j] is a pair of interacting atoms
        # [interaction_pair_to_bond_index[(i,j)]] is what openmm calls a "per-bond parameter"
        fm.addBond([i, j], [interaction_pair_to_bond_index[(i,j)]])
    # give a name to the parameter [interaction_pair_to_bond_index[(i,j)]] that was passed to each bond 
    fm.addPerBondParameter("index")
    # this parameter, "index", is used to look up the appropriate configuration-independent parameter
    # in a table called frag_table, which is passed to the fm force object on the following line
    fm.addTabulatedFunction("frag_table",Discrete2DFunction(len(interaction_list), r_table_size, frag_table.T.flatten()))
    
    if oa.periodic:
        fm.setUsesPeriodicBoundaryConditions(True)
        print('\nfragment_memory_term is periodic')

    fm.setForceGroup(forceGroup)

    if not testing:
        return fm
    else:
        return fm, full_interaction_info

def read_memory(oa, pdb_file, chain_name, target_start, fragment_start, length, weight, min_seq_sep, max_seq_sep, am_well_width=0.1):
    memory_interactions = []

    # if not os.path.isfile(pdb_file):
    #     pdbl = PDBList()
    #     pdbl.retrieve_pdb_file(pdb_file.split('.')[0].lower(), pdir='.')
    #     os.rename("pdb%s.ent" % pdb_id, "%s.pdb" % pdb_id)

    parser = PDBParser()
    structure = parser.get_structure('X', pdb_file)
    chain = structure[0][chain_name]
    residues = [x for x in chain if x.get_full_id()[3][1] in range(fragment_start,fragment_start+length-1)]
    for i, residue_i in enumerate(residues):
        for j, residue_j in enumerate(residues):
            if abs(i-j) > max_seq_sep:
                continue
            target_index_i = target_start + i - 1
            target_index_j = target_start + j - 1
            atom_list_i = []
            target_atom_list_i = []
            atom_list_j = []
            target_atom_list_j = []
            if i-j >= min_seq_sep: # taking the signed value to avoid double counting
                ca_i = residue_i['CA']
                atom_list_i.append(ca_i)
                target_atom_list_i.append(oa.ca[target_index_i])
                ca_j = residue_j['CA']
                atom_list_j.append(ca_j)
                target_atom_list_j.append(oa.ca[target_index_j])
                if not residue_i.get_resname() == "GLY" and oa.cb[target_index_i] >= 0:
                    cb_i = residue_i['CB']
                    atom_list_i.append(cb_i)
                    target_atom_list_i.append(oa.cb[target_index_i])
                if not residue_j.get_resname() == "GLY" and oa.cb[target_index_j] >= 0:
                    cb_j = residue_j['CB']
                    atom_list_j.append(cb_j)
                    target_atom_list_j.append(oa.cb[target_index_j])
            for atom_i, atom_j in product(atom_list_i, atom_list_j):
                particle_1 = target_atom_list_i[atom_list_i.index(atom_i)]
                particle_2 = target_atom_list_j[atom_list_j.index(atom_j)]
                r_ijm = abs(atom_i - atom_j)/10.0 # convert to nm
                sigma_ij = am_well_width*abs(i-j)**0.15 # 0.1 nm = 1 A
                gamma_ij = 1.0
                w_m = weight
                memory_interaction = [particle_1, particle_2, [w_m, gamma_ij, r_ijm, sigma_ij]]
                memory_interactions.append(memory_interaction)
    return memory_interactions

def associative_memory_term(oa, memories, k_am=0.8368, min_seq_sep=3, max_seq_sep=9, am_well_width=0.1):
    # 0.8368 = 0.2 * 4.184 # in kJ/mol, converted from default value in LAMMPS AWSEM
    #pdbid #chain #target #fragment #length #weight
    # multiply interaction strength by overall scaling
    k_am *= oa.k_awsem
    am_function = '-k_am*w_m*gamma_ij*exp(-(r-r_ijm)^2/(2*sigma_ij^2))'
    am = CustomBondForce(am_function)
    am.addGlobalParameter('k_am', k_am)
    am.addPerBondParameter('w_m')
    am.addPerBondParameter('gamma_ij')
    am.addPerBondParameter('r_ijm')
    am.addPerBondParameter('sigma_ij')
    for memory in memories:
        memory_interactions = read_memory(oa, *memory, min_seq_sep, max_seq_sep, am_well_width=am_well_width)
        for memory_interaction in memory_interactions:
            am.addBond(*memory_interaction)
    return am



def density_dependent_associative_memory_term(oa, memories, k_am_dd=1.0, am_dd_min_seq_sep=3, am_dd_max_seq_sep=9, eta_density=50, r_density_min=.45, r_density_max=.65, density_alpha=1.0, density_normalization=2.0, rho0=2.6, am_well_width=0.1, density_min_seq_sep=10, density_only_from_native_contacts=False, density_pdb_file=None, density_chain_name=None, density_native_contact_min_seq_sep=4, density_native_contact_threshold=0.8*nanometers):

    k_am_dd *= oa.k_awsem

    am_dd = CustomGBForce()

    # add all particles to force
    for i in range(oa.natoms):
        am_dd.addParticle([i])

    # add per-particle parameters
    am_dd.addPerParticleParameter("index")

    # add global parameters
    am_dd.addGlobalParameter("k_am_dd", k_am_dd)
    am_dd.addGlobalParameter("eta_density", eta_density)
    am_dd.addGlobalParameter("r_density_min", r_density_min)
    am_dd.addGlobalParameter("r_density_max", r_density_max)
    am_dd.addGlobalParameter("density_alpha", density_alpha)
    am_dd.addGlobalParameter("density_normalization", density_normalization)
    am_dd.addGlobalParameter("rho0", rho0)

    # if density_only_from_native_contacts, read structure to get native contacts
    if density_only_from_native_contacts:
        structure_interactions = read_amhgo_structure(oa, pdb_file=density_pdb_file, chain_name=density_chain_name, amhgo_min_seq_sep=density_native_contact_min_seq_sep, amhgo_contact_threshold=density_native_contact_threshold, amhgo_well_width=0.1) # the well width is not used, so the value doesn't matter

        native_contacts = []
        for interaction in structure_interactions:
            i_index, j_index, [gamma_ij, r_ijN, sigma_ij] = interaction
            native_contacts.append((i_index, j_index))
            native_contacts.append((j_index, i_index))

    # setup tabulated functions and interactions
    density_gamma_ij = [0.0]*oa.natoms*oa.natoms
    for i in range(oa.natoms):
        for j in range(oa.natoms):
            if (i in oa.cb or (oa.res_type[oa.resi[i]] == "IGL" and i in oa.ca)) and (j in oa.cb or (oa.res_type[oa.resi[j]] == "IGL" and i in oa.ca)) and abs(oa.resi[i]-oa.resi[j])>=density_min_seq_sep:
                if not density_only_from_native_contacts or (i, j) in native_contacts or (j, i) in native_contacts:
                    density_gamma_ij[i+j*oa.natoms] = 1.0
                    density_gamma_ij[j+i*oa.natoms] = 1.0
    am_dd.addTabulatedFunction("density_gamma_ij", Discrete2DFunction(oa.natoms, oa.natoms, density_gamma_ij))

    gamma_ij = [0.0]*oa.natoms*oa.natoms*len(memories)
    sigma_ij = [0.1]*oa.natoms*oa.natoms*len(memories)
    r_ijm = [0.0]*oa.natoms*oa.natoms*len(memories)
    for k, memory in enumerate(memories):
        memory_interactions = read_memory(oa, *memory, am_dd_min_seq_sep, am_dd_max_seq_sep, am_well_width=am_well_width)
        for memory_interaction in memory_interactions:
            i, j, (w_m, gamma, r, sigma) = memory_interaction
            gamma_ij[i+j*oa.natoms+k*oa.natoms*oa.natoms] = gamma
            gamma_ij[j+i*oa.natoms+k*oa.natoms*oa.natoms] = gamma
            sigma_ij[i+j*oa.natoms+k*oa.natoms*oa.natoms] = sigma
            sigma_ij[j+i*oa.natoms+k*oa.natoms*oa.natoms] = sigma
            r_ijm[i+j*oa.natoms+k*oa.natoms*oa.natoms] = r
            r_ijm[j+i*oa.natoms+k*oa.natoms*oa.natoms] = r
    am_dd.addTabulatedFunction("gamma_ij", Discrete3DFunction(oa.natoms, oa.natoms, len(memories), gamma_ij))
    am_dd.addTabulatedFunction("sigma_ij", Discrete3DFunction(oa.natoms, oa.natoms, len(memories), sigma_ij))
    am_dd.addTabulatedFunction("r_ijm", Discrete3DFunction(oa.natoms, oa.natoms, len(memories), r_ijm))

    # add computed values
    # compute the density
    am_dd.addComputedValue("rho", "0.25*density_gamma_ij(index1, index2)*(1+tanh(eta_density*(r-r_density_min)))*(1+tanh(eta_density*(r_density_max-r)))", CustomGBForce.ParticlePair)

    # function that determines how the AM term depends on density
    #f_string = "0.25*(1-tanh(eta_density*(rho0-rho1)))*(1-tanh(eta_density*(rho0-rho2)))" # both residues must be buried for the interaction to be active
    f_string = "1-(0.25*(1-tanh(eta_density*(rho1-rho0)))*(1-tanh(eta_density*(rho2-rho0))))" # one residue being buried is enough for the interaction to be active

    # add energy term for each memory
    for k, memory in enumerate(memories):
        memory_interactions = read_memory(oa, *memory, am_dd_min_seq_sep, am_dd_max_seq_sep, am_well_width=am_well_width)
        for memory_interaction in memory_interactions:
            i, j, (w_m, gamma, r, sigma) = memory_interaction
        am_dd.addEnergyTerm("-k_am_dd*(density_alpha*f*density_normalization*beta_ij+(1-density_alpha)*beta_ij);\
        beta_ij=%f*gamma_ij(index1,index2,%d)*exp(-(r-r_ijm(index1,index2,%d))^2/(2*sigma_ij(index1,index2,%d)^2));\
        f=%s" % (w_m, k, k, k, f_string), CustomGBForce.ParticlePair)

    return am_dd

def read_amhgo_structure(oa, pdb_file, chain_name, amhgo_min_seq_sep=4, amhgo_contact_threshold=0.8*nanometers, amhgo_well_width=0.1):
    structure_interactions = []
    from Bio.PDB import PDBParser
    import itertools
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_file)
    chain = structure[0][chain_name]
    residues = [x for x in chain]
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
                if (residue_i.get_resname() != "GLY") and (residue_i.get_resname() != "IGL"):
                    cb_i = residue_i['CB']
                    cb_list.append(cb_i)
                    atom_list_i.append(cb_i)
                if (residue_j.get_resname() != "GLY") and (residue_j.get_resname() != "IGL"):
                    cb_j = residue_j['CB']
                    cb_list.append(cb_j)
                    atom_list_j.append(cb_j)
                for atom_i, atom_j in itertools.product(atom_list_i, atom_list_j):
                    r_ijN = abs(atom_i - atom_j)/10.0*nanometers # convert to nm
                    if r_ijN <= amhgo_contact_threshold:
                        sigma_ij = amhgo_well_width*abs(i-j)**0.15 # 0.1 nm = 1 A
                        gamma_ij = 1.0
                        if atom_i in ca_list:
                            i_index = oa.ca[i]
                        if atom_i in cb_list:
                            i_index = oa.cb[i]
                        if atom_j in ca_list:
                            j_index = oa.ca[j]
                        if atom_j in cb_list:
                            j_index = oa.cb[j]
                        structure_interaction = [i_index, j_index, [gamma_ij, r_ijN, sigma_ij]]
                        # print(i_index, j_index, gamma_ij, r_ijN, sigma_ij)
                        structure_interactions.append(structure_interaction)
    return structure_interactions

def additive_amhgo_term(oa, pdb_file, chain_name, k_amhgo=4.184, amhgo_min_seq_sep=3, amhgo_contact_threshold=0.8*nanometers, amhgo_well_width=0.1, forceGroup=22):
    import itertools
    # multiply interaction strength by overall scaling
    print("AMH-GO structure based term is ON")
    k_amhgo *= oa.k_awsem
    # create contact force
    amhgo = CustomBondForce(f"-{k_amhgo}*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    # # add global parameters
    amhgo.addPerBondParameter("gamma_ij")
    amhgo.addPerBondParameter("r_ijN")
    amhgo.addPerBondParameter("sigma_ij")
    # create bonds
    structure_interactions = read_amhgo_structure(oa, pdb_file, chain_name, amhgo_min_seq_sep, amhgo_contact_threshold, amhgo_well_width=amhgo_well_width)
    # print(structure_interactions)
    for structure_interaction in structure_interactions:
        # print(structure_interaction)
        amhgo.addBond(*structure_interaction)
    # amhgo.setForceGroup(22)
    amhgo.setForceGroup(forceGroup)
    return amhgo

def er_term(oa, k_er=4.184, er_min_seq_sep=2, er_cutoff=99.0, er_well_width=0.1, forceGroup=25):
    ### this is a structure prediction related term; Adapted from Sirovitz Schafer Wolynes 2017 Protein Science;
    ### See original papers for reference: Make AWSEM AWSEM-ER with Evolutionary restrictions
    ### ER restrictions can be obtained from multiple sources (RaptorX, deepcontact, and Gremlin)
    ### term modified from amh-go term, and the current strength seems to be high, and needs to be lowered somehow.
    ### amh-go normalization factor will be added soon. Based on Eastwood Wolynes 2000 JCP
    print("ER term is ON")
    import itertools
    k_er *= oa.k_awsem
    # create contact force
    er = CustomBondForce("-k_er*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    # # add global parameters
    er.addGlobalParameter("k_er", k_er)
    er.addPerBondParameter("gamma_ij")
    er.addPerBondParameter("r_ijN")
    er.addPerBondParameter("sigma_ij")
    structure_interactions_er = []
    ### read in dat files from contact predictions;
    in_rnativeCACA = np.loadtxt('go_rnativeCACA.dat')
    in_rnativeCACB = np.loadtxt('go_rnativeCACB.dat')
    in_rnativeCBCB = np.loadtxt('go_rnativeCBCB.dat')
    for i in range(oa.nres):
        for j in range(oa.nres):
            if abs(i-j) >= er_min_seq_sep and in_rnativeCACA[i][j]<er_cutoff:
                sigma_ij = er_well_width*abs(i-j)**0.15 # 0.1 nm = 1 A
                gamma_ij = 1.0
                r_ijN = in_rnativeCACA[i][j]/10.0*nanometers
                structure_interactions_er.append([oa.ca[i], oa.ca[j], [gamma_ij, r_ijN, sigma_ij]])
            if abs(i-j) >= er_min_seq_sep and in_rnativeCACB[i][j]<er_cutoff and oa.cb[j]!= -1:
                sigma_ij = er_well_width*abs(i-j)**0.15 # 0.1 nm = 1 A
                gamma_ij = 1.0
                r_ijN = in_rnativeCACB[i][j]/10.0*nanometers
                structure_interactions_er.append([oa.ca[i], oa.cb[j], [gamma_ij, r_ijN, sigma_ij]])
            if abs(i-j) >= er_min_seq_sep and in_rnativeCBCB[i][j]<er_cutoff and oa.cb[j]!= -1 and oa.cb[i]!= -1:#oa.res_type[oa.resi[i]] != "IGL" and oa.res_type[oa.resi[j]] != "IGL":
                sigma_ij = er_well_width*abs(i-j)**0.15 # 0.1 nm = 1 A
                gamma_ij = 1.0
                r_ijN = in_rnativeCBCB[i][j]/10.0*nanometers
                structure_interactions_er.append([oa.cb[i], oa.cb[j], [gamma_ij, r_ijN, sigma_ij]])
                # print([i, j, oa.res_type[oa.resi[i]], oa.res_type[oa.resi[j]],oa.cb[i], oa.cb[j], [gamma_ij, r_ijN, sigma_ij]])
    # create bonds
    for structure_interaction_er in structure_interactions_er:
        er.addBond(*structure_interaction_er)
    er.setForceGroup(forceGroup)
    return er

def machine_learning_term(oa, k=1*kilocalorie_per_mole, dataFile="dist.npz", UseSavedFile=False, saved_file="ml_data.npz", forceGroup=4):
    k_ml = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_ml = k_ml * oa.k_awsem

    x = [0.0, 2.0, 3.5, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75]
    num_of_points = 100

    if UseSavedFile and os.path.isfile(saved_file):
        data = np.load(saved_file)
        index_array = data["index_array"]
        interaction_array = data["interaction_array"]
    else:
        # spline fit
        a = np.load(dataFile)
        distspline = a['distspline']

        n = distspline.shape[0]
        interaction_list = []
        index_list = []
        xnew = np.linspace(min(x), max(x), num=num_of_points, endpoint=True)
        for i in range(n):
            for j in range(i+1, n):
                if np.alltrue(distspline[i][j] == 0):
                    continue
                y = distspline[i][j]
                f = interp1d(x, y)
                ynew = f(xnew)
                interaction_list.append(ynew)
                index_list.append([i, j])
        index_array = np.array(index_list)
        interaction_array = np.array(interaction_list)
        np.savez(saved_file, index_array=index_array, interaction_array=interaction_array)

    interaction_n = index_array.shape[0]

    r_max = max(x)
    r_min = min(x)
    dr = (r_max-r_min)/(num_of_points-1)

    max_r_index_1 = num_of_points - 2

    ml = CustomCompoundBondForce(2, f"{k_ml}*((v2-v1)*r+v1*r_2-v2*r_1)/(r_2-r_1); \
                                v1=ml_table(index, r_index_1);\
                                v2=ml_table(index, r_index_2);\
                                r_1={r_min}+{dr}*r_index_1;\
                                r_2={r_min}+{dr}*r_index_2;\
                                r_index_2=r_index_1+1;\
                                r_index_1=min({max_r_index_1}, floor(r/{dr}));\
                                r=min(r_raw, {r_max});\
                                r_raw=distance(p1, p2)*10;")

    cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]

    for idx, index_pair in enumerate(index_array):
        resi,resj = index_pair
        i = cb_fixed[resi]
        j = cb_fixed[resj]

        ml.addBond([i, j], [idx])

    ml.addPerBondParameter("index")

    ml.addTabulatedFunction("ml_table",
            Discrete2DFunction(interaction_n, num_of_points, interaction_array.T.flatten()))


    ml.setForceGroup(forceGroup)
    return ml



def machine_learning_dihedral_omega_angle_term(oa, k=1*kilocalorie_per_mole, dataFile="omega.npz", UseSavedFile=False, saved_file="ml_data.npz", forceGroup=4):
    k_ml_angle = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_ml_angle = k_ml_angle * oa.k_awsem


    omega = np.load(dataFile)


    omegaspline = omega["omegaspline"]


    omega = "-3.53429174 -3.27249235 -3.01069296 -2.74889357 -2.48709418 -2.2252948\
    -1.96349541 -1.70169602 -1.43989663 -1.17809725 -0.91629786 -0.65449847\
    -0.39269908 -0.13089969  0.13089969  0.39269908  0.65449847  0.91629786\
    1.17809725  1.43989663  1.70169602  1.96349541  2.2252948   2.48709418\
    2.74889357  3.01069296  3.27249235  3.53429174"

    omega_x = [float(a) for a in omega.split()]


    # spline fit
    x = omega_x
    spline = omegaspline

    num_of_points = 100
    n = spline.shape[0]
    interaction_list = []
    index_list = []

    xnew = np.linspace(min(x), max(x), num=num_of_points, endpoint=True)
    for i in range(n):
        for j in range(i+1, n):
            if np.alltrue(spline[i][j] == 0):
                continue
            y = spline[i][j]
            f = interp1d(x, y, kind='cubic')
            ynew = f(xnew)
            interaction_list.append(ynew)
            index_list.append([i, j])
    index_array = np.array(index_list)
    interaction_array = np.array(interaction_list)

    angle_max = max(x)
    angle_min = min(x)
    dangle = (angle_max-angle_min)/(num_of_points-1)

    max_angle_index_1 = num_of_points - 2
    interaction_n = index_array.shape[0]

    ml = CustomCompoundBondForce(4, f"{k_ml_angle}*omegaEnergy;\
                                omegaEnergy=((v2-v1)*angle+v1*angle_2-v2*angle_1)/(angle_2-angle_1); \
                                v1=ml_table(index, angle_index_1);\
                                v2=ml_table(index, angle_index_2);\
                                angle_1={angle_min}+{dangle}*angle_index_1;\
                                angle_2={angle_min}+{dangle}*angle_index_2;\
                                angle_index_2=angle_index_1+1;\
                                angle_index_1=min({max_angle_index_1}, floor((angle-{angle_min})/{dangle}));\
                                angle=dihedral(p1, p2, p3, p4);")


    for idx, index_pair in enumerate(index_array):
        
        resi,resj = index_pair
        p0 = oa.ca[resi]
        p1 = oa.cb[resi]
        p2 = oa.cb[resj]
        p3 = oa.ca[resj]
        if p1 == -1 or p2 == -1:
            continue
        ml.addBond([p0, p1, p2, p3], [idx])

    ml.addPerBondParameter("index")

    ml.addTabulatedFunction("ml_table",
            Discrete2DFunction(interaction_n, num_of_points, interaction_array.T.flatten()))


    ml.setForceGroup(forceGroup)
    return ml


def machine_learning_dihedral_theta_angle_term(oa, k=1*kilocalorie_per_mole, dataFile="theta.npz", forceGroup=4):
    k_ml_angle = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_ml_angle = k_ml_angle * oa.k_awsem

    theta = np.load(dataFile)


    thetaspline = theta["thetaspline"]


    theta = "-3.53429174 -3.27249235 -3.01069296 -2.74889357 -2.48709418 -2.2252948\
    -1.96349541 -1.70169602 -1.43989663 -1.17809725 -0.91629786 -0.65449847\
    -0.39269908 -0.13089969  0.13089969  0.39269908  0.65449847  0.91629786\
    1.17809725  1.43989663  1.70169602  1.96349541  2.2252948   2.48709418\
    2.74889357  3.01069296  3.27249235  3.53429174"

    theta_x = [float(a) for a in theta.split()]


    # spline fit
    x = theta_x
    spline = thetaspline

    num_of_points = 100
    n = spline.shape[0]
    interaction_list = []
    index_list = []

    xnew = np.linspace(min(x), max(x), num=num_of_points, endpoint=True)
    for i in range(n):
        for j in range(i+1, n):
            if np.alltrue(spline[i][j] == 0):
                continue
            y = spline[i][j]
            f = interp1d(x, y, kind='cubic')
            ynew = f(xnew)
            interaction_list.append(ynew)
            index_list.append([i, j])
    index_array = np.array(index_list)
    interaction_array = np.array(interaction_list)

    angle_max = max(x)
    angle_min = min(x)
    dangle = (angle_max-angle_min)/(num_of_points-1)

    max_angle_index_1 = num_of_points - 2
    interaction_n = index_array.shape[0]

    ml = CustomCompoundBondForce(4, f"{k_ml_angle}*omegaEnergy;\
                                omegaEnergy=((v2-v1)*angle+v1*angle_2-v2*angle_1)/(angle_2-angle_1); \
                                v1=ml_table(index, angle_index_1);\
                                v2=ml_table(index, angle_index_2);\
                                angle_1={angle_min}+{dangle}*angle_index_1;\
                                angle_2={angle_min}+{dangle}*angle_index_2;\
                                angle_index_2=angle_index_1+1;\
                                angle_index_1=min({max_angle_index_1}, floor((angle-{angle_min})/{dangle}));\
                                angle=dihedral(p1, p2, p3, p4);")


    for idx, index_pair in enumerate(index_array):
        
        resi,resj = index_pair
        p0 = oa.n[resi]
        p1 = oa.ca[resi]
        p2 = oa.cb[resi]
        p3 = oa.cb[resj]
        if p0 == -1 or p2 == -1 or p3 == -1:
            continue
        ml.addBond([p0, p1, p2, p3], [idx])

    ml.addPerBondParameter("index")

    ml.addTabulatedFunction("ml_table",
            Discrete2DFunction(interaction_n, num_of_points, interaction_array.T.flatten()))


    ml.setForceGroup(forceGroup)
    return ml


def machine_learning_dihedral_phi_angle_term(oa, k=1*kilocalorie_per_mole, dataFile="phi.npz", forceGroup=4):
    k_ml_angle = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_ml_angle = k_ml_angle * oa.k_awsem

    phi = np.load(dataFile)


    phispline = phi["phispline"]


    phi = "-0.39269908 -0.13089969  0.13089969  0.39269908  0.65449847  0.91629786\
    1.17809725  1.43989663  1.70169602  1.96349541  2.2252948   2.48709418\
    2.74889357  3.01069296  3.27249235  3.53429174"


    phi_x = [float(a) for a in phi.split()]


    # spline fit
    x = phi_x
    spline = phispline

    num_of_points = 100
    n = spline.shape[0]
    interaction_list = []
    index_list = []

    xnew = np.linspace(min(x), max(x), num=num_of_points, endpoint=True)
    for i in range(n):
        for j in range(i+1, n):
            if np.alltrue(spline[i][j] == 0):
                continue
            y = spline[i][j]
            f = interp1d(x, y, kind='cubic')
            ynew = f(xnew)
            interaction_list.append(ynew)
            index_list.append([i, j])
    index_array = np.array(index_list)
    interaction_array = np.array(interaction_list)

    angle_max = max(x)
    angle_min = min(x)
    dangle = (angle_max-angle_min)/(num_of_points-1)

    max_angle_index_1 = num_of_points - 2
    interaction_n = index_array.shape[0]

    ml = CustomCompoundBondForce(3, f"{k_ml_angle}*omegaEnergy;\
                                omegaEnergy=((v2-v1)*angle+v1*angle_2-v2*angle_1)/(angle_2-angle_1); \
                                v1=ml_table(index, angle_index_1);\
                                v2=ml_table(index, angle_index_2);\
                                angle_1={angle_min}+{dangle}*angle_index_1;\
                                angle_2={angle_min}+{dangle}*angle_index_2;\
                                angle_index_2=angle_index_1+1;\
                                angle_index_1=min({max_angle_index_1}, floor((angle-{angle_min})/{dangle}));\
                                angle=angle(p1, p2, p3);")


    for idx, index_pair in enumerate(index_array):
        
        resi,resj = index_pair
        p0 = oa.ca[resi]
        p1 = oa.cb[resi]
        p2 = oa.cb[resj]
        if p1 == -1 or p2 == -1:
            continue
        ml.addBond([p0, p1, p2], [idx])

    ml.addPerBondParameter("index")

    ml.addTabulatedFunction("ml_table",
            Discrete2DFunction(interaction_n, num_of_points, interaction_array.T.flatten()))


    ml.setForceGroup(forceGroup)
    return ml

'''
# will be deleted in the future.
def read_reference_structure_for_q_calculation(oa, pdb_file, chain_name, min_seq_sep=3, max_seq_sep=np.inf, contact_threshold=0.8*nanometers):
    structure_interactions = []
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_file)
    chain = structure[0][chain_name]
    residues = [x for x in chain]
    for i, residue_i in enumerate(residues):
        for j, residue_j in enumerate(residues):
            ca_list = []
            cb_list = []
            atom_list_i = []
            atom_list_j = []
            if i-j >= min_seq_sep and i-j <= max_seq_sep:  # taking the signed value to avoid double counting
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
                    if r_ijN <= contact_threshold:
                        sigma_ij = 0.1*abs(i-j)**0.15 # 0.1 nm = 1 A
                        gamma_ij = 1.0
                        if atom_i in ca_list:
                            i_index = oa.ca[i]
                        if atom_i in cb_list:
                            i_index = oa.cb[i]
                        if atom_j in ca_list:
                            j_index = oa.ca[j]
                        if atom_j in cb_list:
                            j_index = oa.cb[j]
                        structure_interaction = [i_index, j_index, [gamma_ij, r_ijN, sigma_ij]]
                        structure_interactions.append(structure_interaction)

    return structure_interactions

def read_reference_structure_for_q_calculation_2(oa, pdb_file, min_seq_sep=3, max_seq_sep=np.inf, contact_threshold=0.8*nanometers):
    # default use all chains in pdb file.
    structure_interactions = []
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_file)
    model = structure[0]
    chain_start = 0
    count = 0
    for chain in model.get_chains():
        chain_start += count
        count = 0
        for i, residue_i in enumerate(chain.get_residues()):
            count += 1
            #  print(i, residue_i)
            for j, residue_j in enumerate(chain.get_residues()):
                ca_list = []
                cb_list = []
                atom_list_i = []
                atom_list_j = []
                if i-j >= min_seq_sep and i-j <= max_seq_sep:  # taking the signed value to avoid double counting
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
                        if r_ijN <= contact_threshold:
                            sigma_ij = 0.1*abs(i-j)**0.15 # 0.1 nm = 1 A
                            gamma_ij = 1.0
                            if atom_i in ca_list:
                                i_index = oa.ca[i+chain_start]
                            if atom_i in cb_list:
                                i_index = oa.cb[i+chain_start]
                            if atom_j in ca_list:
                                j_index = oa.ca[j+chain_start]
                            if atom_j in cb_list:
                                j_index = oa.cb[j+chain_start]
                            structure_interaction = [i_index, j_index, [gamma_ij, r_ijN, sigma_ij]]
                            structure_interactions.append(structure_interaction)

    return structure_interactions
'''
