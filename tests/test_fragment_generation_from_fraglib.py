from openawsem import *
from openawsem.functionTerms import fragmentMemoryTerms
import os
import inspect
from pathlib import Path
import pickle

data_path = Path('tests')/'data'

def definitely_correct(oa, k_fm=0.04184, frag_file_list_file="./frag.mem", npy_frag_table="./frag_table.npy",
                    min_seq_sep=3, max_seq_sep=9, fm_well_width=0.1, UseSavedFragTable=True, caOnly=False, forceGroup=23):
    """
    This function is used exclusively to help me debug the test that I wrote. Once the test is working, it will be unnecessary.
    """
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
    for i in range(oa.natoms):
        if i in oa.ca:
            res_id = oa.resi[i]    # oa.resi start with 0, but pdb residue id start with 1
            data_dic[("CA", 1+int(res_id))] = i
        if i in oa.cb:
            res_id = oa.resi[i]
            data_dic[("CB", 1+int(res_id))] = i
    # print(oa.res_type)
    # print(oa.resi)
    # print(data_dic)
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
        print(f"Loading Fragment files(Gro files)")
        frag_file_list = pd.read_csv(frag_file_list_file, skiprows=4, sep="\s+", names=["location", "target_start", "fragment_start", "frag_len", "weight"])
        interaction_list = set()
    for frag_index in range(len(frag_file_list)):
        location = frag_file_list["location"].iloc[frag_index]
        frag_name = os.path.join(frag_location_pre, location)
        frag_len = frag_file_list["frag_len"].iloc[frag_index]
        weight = frag_file_list["weight"].iloc[frag_index]
        target_start = frag_file_list["target_start"].iloc[frag_index]  # residue id
        fragment_start = frag_file_list["fragment_start"].iloc[frag_index]  # residue id
        frag = pd.read_csv(frag_name, skiprows=2, sep="\s+", header=None, names=["Res_id", "Res", "Type", "i", "x", "y", "z"])
        frag = frag.query(f"Res_id >= {fragment_start} and Res_id < {fragment_start+frag_len} and (Type == 'CA' or Type == 'CB')")
        w_m = weight
        gamma_ij = 1
        f = frag.values
        for i in range(len(frag)):
            for j in range(i, len(frag)):
                res_id_i = frag["Res_id"].iloc[i]
                res_id_j = frag["Res_id"].iloc[j]
                target_res_id_i = frag["Res_id"].iloc[i] - fragment_start + target_start
                target_res_id_j = frag["Res_id"].iloc[j] - fragment_start + target_start
                seq_sep = res_id_j - res_id_i
                if seq_sep > max_seq_sep:
                    continue
                if seq_sep < min_seq_sep:
                    continue
                try:
                    i_type = frag["Type"].iloc[i]
                    j_type = frag["Type"].iloc[j]
                    correspond_target_i = data_dic[(i_type, int(target_res_id_i))]
                    correspond_target_j = data_dic[(j_type, int(target_res_id_j))]
                    correspond_target_i = int(correspond_target_i)
                    correspond_target_j = int(correspond_target_j)
                except Exception as e:
                    continue
                
                fi_x = f[i][4]
                fi_y = f[i][5]
                fi_z = f[i][6]

                fj_x = f[j][4]
                fj_y = f[j][5]
                fj_z = f[j][6]
                # print("----", fi_x, fi_y, fi_z, fj_x, fj_y, fj_z)
                sigma_ij = fm_well_width*seq_sep**0.15
                rm = ((fi_x-fj_x)**2 + (fi_y-fj_y)**2 + (fi_z-fj_z)**2)**0.5

                i_j_sep = int(correspond_target_j - correspond_target_i)

                raw_frag_table[correspond_target_i][i_j_sep] += w_m*gamma_ij*np.exp((r_array-rm)**2/(-2.0*sigma_ij**2))
                raw_frag_table_count[correspond_target_i][i_j_sep] += 1
                #print(correspond_target_i)
                #if correspond_target_i in [95,98,113,116,131,134,148,151,154,157,324,327]:
                #with open(f'{correspond_target_i}_frags.txt','a') as number_write_file:
                #    number_write_file.write(f'{frag_name[:-4]}, i: {i}, j: {j}\n')
                with open(f'{correspond_target_i}_rm.txt','a') as number_write_file:
                    number_write_file.write(f'{rm}\n')
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
        with open(frag_table_file, 'wb') as f:
            pickle.dump((frag_table, interaction_list, interaction_pair_to_bond_index), f)
        #with open(f'{data_path}/old_raw_frag_table.npy', 'wb') as f:
        #    pickle.dump(raw_frag_table, f)
        #with open(f'{data_path}/old_raw_frag_table_count.npy','wb') as f:
        #    pickle.dump(raw_frag_table_count, f)
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

    # for edge case, that r > frag_table_rmax
    max_r_index_1 = r_table_size - 2
    fm = CustomCompoundBondForce(2, f"-{k_fm}*((v2-v1)*r+v1*r_2-v2*r_1)/(r_2-r_1); \
                                v1=frag_table(index, r_index_1);\
                                v2=frag_table(index, r_index_2);\
                                r_1={frag_table_rmin}+{frag_table_dr}*r_index_1;\
                                r_2={frag_table_rmin}+{frag_table_dr}*r_index_2;\
                                r_index_2=r_index_1+1;\
                                r_index_1=min({max_r_index_1}, floor(r/{frag_table_dr}));\
                                r=distance(p1, p2);")
    for (i, j) in interaction_list:
        if caOnly and ((i not in oa.ca) or (j not in oa.ca)):
            continue
        fm.addBond([i, j], [interaction_pair_to_bond_index[(i,j)]])

    fm.addPerBondParameter("index")

    fm.addTabulatedFunction("frag_table",
            Discrete2DFunction(len(interaction_list), r_table_size, frag_table.T.flatten()))
    
    if oa.periodic:
        fm.setUsesPeriodicBoundaryConditions(True)
        print('\nfragment_memory_term is periodic')

    fm.setForceGroup(forceGroup)
    return fm

def test_fragment_generation_from_fraglib():

    print("Configuring test and checking for necessary files")

    # set up list of test functions
    # note that we exclude single memory
    # eventually, we might have a structural alignment-based fragment memory function
    # in addition to the usual one
    test_funcs = ['fragment_memory_term']
    test_funcs = {name:func for name,func in inspect.getmembers(fragmentMemoryTerms) if name in test_funcs}

    # set up list of test proteins
    # we should have at least 1r69 and one multi-chain protein
    test_proteins = ['1r69',]#'4tlt']
    # just make sure we don't run into any issues with uppercase
    test_proteins = [test_protein.lower() for test_protein in test_proteins]

    # create paths to all reference fraglibs and memory files
    # Generally, this can vary with protein or fragment generation method
    fraglib_names = []
    memory_names = []
    for protein in test_proteins:
        for func_name in test_funcs.keys():
            fraglib_names.append(f'fraglib-{protein}-{func_name}')
            memory_names.append(f'{protein}-{func_name}.mem')
    
    # make sure that all necessary fraglibs and *frags.mem files actually exist
    fail_message = lambda fraglib_name, memory_name: f"""
       This test ensures that the FM interaction list fed into the force field
       does not change as the code is updated. We try to cover diverse situations by computing this list
       for multiple fragment generation algorithms for multiple proteins.
       
       Since the fraglib creation is often slow and often involves some amount of randomness, 
       we provide a reference fraglib for each combination of test protein and fragment generation function.
         
       Therefore, this test only covers the last step of the fragment generation process (the creation of the potential
       in-memory during the forces_setup phase), which is probably less error-prone and less likely to be modified than
       other stages. However, I had to modify this part of the code, so I'm still writing a test for it.

       This error has been raised because the script could not find a reference fraglib called data/{fraglib_name}
       and/or an associated memory file data/{memory_name},
       which should be present given the test protein set {test_proteins} 
       and the test function set {test_funcs}.

       -- Finley Clark, 1/3/2025
       
    """ 
    fraglib_dirs = [filename for filename in os.listdir(data_path) if 'fraglib' in filename]
    memory_files = [filename for filename in os.listdir(data_path) if '.mem' == filename[-4:]]
    for fraglib_name, memory_name in zip(fraglib_names,memory_names):
        assert fraglib_name in fraglib_dirs, fail_message(fraglib_name,memory_name)
        assert memory_name in memory_files, fail_message(fraglib_name,memory_name)
 
    # make sure that all protein structure files exist
    # right now, all reference protein structure files for the test are assumed to be pdbs
    for pdbid in test_proteins:
        assert f'{pdbid}-openmmawsem.pdb' in os.listdir(data_path), f"-openmmawsem.pdb for protein {pdbid} does not exist in data_path {data_path}"    
        
    # make sure that numpy files containing correctly computed interaction list, etc. exist
    for pdbid in test_proteins:
        for name in test_funcs.keys():
            assert os.path.isfile(f'{data_path}/{pdbid}-{name}.npy'), f"numpy file for correctly computed interaction list, {data_path}/{pdbid}-{name}.npy, not found"

    # now it's time to test our functions
    print(f"testing functions {test_funcs} on proteins {test_proteins}")

    for pdbid in test_proteins:
        chain = helperFunctions.myFunctions.getAllChains(f'{data_path}/{pdbid}-openmmawsem.pdb')
        oa = OpenMMAWSEMSystem(f'{data_path}/{pdbid}-openmmawsem.pdb', k_awsem=1.0, chains=chain, xml_filename=openawsem.xml, seqFromPdb=None, includeLigands=False)  
        for name,func in test_funcs.items():
            # run function with testing switch, which returns extra logging information to full_interaction_info
            force_object, full_interaction_info = func(oa,k_fm=0.04184, 
                frag_file_list_file=f'{data_path}/{protein}-{name}.mem',npy_frag_table=f'{data_path}/{pdbid}-{name}-test.npy',
                min_seq_sep=3, max_seq_sep=9, fm_well_width=0.1, UseSavedFragTable=False, caOnly=False, forceGroup=23,testing=True)
            # load in frag table written during both normal (non-testing) and testing runs of the function
            fm_old = np.load(f'{data_path}/{pdbid}-{name}.npy',allow_pickle=True)
            fm_new = np.load(f'{data_path}/{pdbid}-{name}-test.npy',allow_pickle=True)
            # First, we compare the interaction_list and interaction_pair_to_bond_index saved as part of each frag table.
            # These two assert statements must pass if the full_interaction_info (see below) assert statements pass, 
            # but we include these for clarity and to give another piece of information in case the full_interaction_info assert fails
            interaction_list_old = fm_old[1]
            interaction_list_new = fm_new[1]
            assert interaction_list_old == interaction_list_new
            interaction_pair_to_bond_index_old = fm_old[2]
            interaction_pair_to_bond_index_new = fm_new[2]
            assert interaction_pair_to_bond_index_old == interaction_pair_to_bond_index_new
            # Next, to be sure that each atom has the same interactions coming from the same memories
            # (something that is consistent with but not proven by our first two assertions),
            # we work with the extra logging info in full_interaction_info
            for key in full_interaction_info.keys(): # key is an integer corresponding to an atom index
                if full_interaction_info[key] == '':
                    continue # there won't be a -frags.txt file if the atom is not part of any memories
                             # note that in full_interaction_info, we're actually just recording pairs
                             # of each atom i and interacting atoms with a larger index, 
                             # so the last 3 residues will never have any interactions listed here
                             # (assuming min_seq_sep==3)
                lines = ''
                with open(f'{data_path}/{key}-{pdbid}-{name}-frags.txt','r') as f:
                    for line in f:
                        lines += line
                assert full_interaction_info[key] == lines
            # finally, we assert that the fragment tables are exactly the same
            frag_table_old = fm_old[0]
            frag_table_new = fm_new[0]
            difference = frag_table_new - frag_table_old
            assert (frag_table_old == frag_table_new).all()
            '''
            raw_frag_table_old = np.load(f'{data_path}/old_raw_frag_table.npy',allow_pickle=True)
            raw_frag_table_new = np.load(f'{data_path}/new_raw_frag_table.npy',allow_pickle=True)
            raw_frag_table_old_count = np.load(f'{data_path}/old_raw_frag_table_count.npy',allow_pickle=True)
            raw_frag_table_new_count = np.load(f'{data_path}/new_raw_frag_table_count.npy',allow_pickle=True)
            '''
            #import pdb; pdb.set_trace()
            #print(fm_old)
            #print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            #print(fm_new)

            '''
                        # remove log files so we can write new ones for this test
            for filename in os.listdir(data_path):
                if '_new.txt' == filename[-8:] and f'{protein}-name' in filename:
                    os.remove(filename)
            '''
            '''
            assert (raw_frag_table_old_count==raw_frag_table_new_count).all()
            assert (raw_frag_table_old==raw_frag_table_new).all()
            assert np.allclose(frag_table_old,frag_table_new)
            assert (frag_table_old==frag_table_new).all()
            '''
            #interaction_list_old = fm_old[1]
            #interaction_list_new = fm_new[1]
            #difference1 = interaction_list_new - interaction_list_old
            #difference2 = interaction_list_old - interaction_list_new
            #with open('difference1.txt','w') as f:
            #    for thing in sorted(difference1):
            #        f.write(f'{thing}\n')
            #with open('difference2.txt','w') as f:
            #    for thing in sorted(difference2):
            #        f.write(f'{thing}\n')
            #with open('old.txt','w') as f:
            #    for thing in sorted(interaction_list_old):
            #        f.write(f'{thing}\n')
            #with open('new.txt','w') as f:
            #    for thing in sorted(interaction_list_new):
            #        f.write(f'{thing}\n')
            #exit()
            #assert interaction_list_old == interaction_list_new
            #interaction_pair_to_bond_index_old = fm_old[2]
            #interaction_pair_to_bond_index_new = fm_new[2]
            #assert interaction_pair_to_bond_index_old == interaction_pair_to_bond_index_new

if __name__ == "__main__":
    #pass
    test_fragment_generation_from_fraglib()
