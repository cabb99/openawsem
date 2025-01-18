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

###################################
#### COPIED FROM openAWSEM.py
def get_openmm_io_class(file_type):
    if file_type == "pdb":
        io_class = PDBFile
    elif file_type == "cif":
        io_class = PDBxFile
    else:
        raise ValueError(f"Expected file_type 'pdb' or 'cif' but got file_type={file_type}") 
    return io_class
###################################

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

        io_class = get_openmm_io_class(frag_name[-3:])
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
        # np.save(frag_table_file, (frag_table, interaction_list, interaction_pair_to_bond_index))
        #with open(frag_table_file, 'wb') as f:
        #    pickle.dump((frag_table, interaction_list, interaction_pair_to_bond_index), f)
        #with open(f'tests/data/new_raw_frag_table.npy','wb') as f:
        #    pickle.dump(raw_frag_table,f)
        #with open(f'tests/data/new_raw_frag_table_count.npy','wb') as f:
        #    pickle.dump(raw_frag_table_count,f)
        print(f"All gro files information have been stored in the {frag_table_file}. \
            \nYou might want to set the 'UseSavedFragTable'=True to speed up the loading next time. \
            \nBut be sure to remove the .npy file if you modify the .mem file. otherwise it will keep using the old frag memeory.")
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