import argparse
from pathlib import Path
import sys
import os
from .IndexPdb import *
from .Pdb2GroLib import *
import subprocess
import argparse
from pathlib import Path
import logging
import requests
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from openmm.app import *

def update_failed_pdb_list(failed_pdb, failed_pdb_list_file):
    try:
        with open(failed_pdb_list_file, 'r') as file:
            existing_pdb_ids = sorted(line.strip() for line in file)
    except FileNotFoundError:
        existing_pdb_ids = []

    # Collect failed pdbIDs that are marked as True
    new_pdb_ids = sorted(pdbID for pdbID, failed in failed_pdb.items() if failed)

    # Merge two sorted lists while removing duplicates
    merged_pdb_ids = []
    i, j = 0, 0
    while i < len(existing_pdb_ids) and j < len(new_pdb_ids):
        if existing_pdb_ids[i] == new_pdb_ids[j]:
            merged_pdb_ids.append(existing_pdb_ids[i])
            i += 1
            j += 1
        elif existing_pdb_ids[i] < new_pdb_ids[j]:
            merged_pdb_ids.append(existing_pdb_ids[i])
            i += 1
        else:
            merged_pdb_ids.append(new_pdb_ids[j])
            j += 1

    # Append remaining items if any list was not exhausted
    if i < len(existing_pdb_ids):
        merged_pdb_ids.extend(existing_pdb_ids[i:])
    if j < len(new_pdb_ids):
        merged_pdb_ids.extend(new_pdb_ids[j:])

    # Write the merged list back to the file
    with open(failed_pdb_list_file, 'w') as file:
        for pdbID in merged_pdb_ids:
            file.write(f"{pdbID}\n")

def process_window(i, record, fragment_length, evalue_threshold, database, residue_base):
    rangeStart = i - 1
    rangeEnd = i + fragment_length - 1
    logging.debug(f"window position::: {i}")
    subrange = str(record[rangeStart:rangeEnd].seq)
    logging.debug(f"fragment subrange::: {subrange}")

    # Use a fixed file name as specified

    with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_fragment_file:
        temp_fragment_file.write(">query\n")
        temp_fragment_file.write(subrange)
        temp_fragment_file.flush()
        
        # Constructing the PSI-BLAST command
        exeline = f"psiblast -num_iterations 5 -comp_based_stats 0 -word_size 2 -evalue {evalue_threshold} " \
                f"-outfmt '6 sseqid qstart qend sstart send qseq sseq length gaps bitscore evalue' -matrix BLOSUM62 -threshold 9 -window_size 0 " \
                f"-db {database} -query {temp_fragment_file.name}"
        logging.debug(f"executing::: {exeline}")

        psiblastOut = os.popen(exeline).read().splitlines()
    
    if psiblastOut and psiblastOut[-1] == 'Search has CONVERGED!':
        psiblastOut = psiblastOut[:-2]  # exclude last two lines for the text

    # Filter last iteration. 
    # If there are multiple iterations, it can be catched if the e_score decreases and the pdbID changes
    N_blast = len(psiblastOut)
    N_start_new = 1
    line_count = 0
    e_score_old = 0
    pdbID_old = 'BBBBB'   # set initial variables for processing PSI-BLAST output
    for line in psiblastOut:  # For PSI-BLAST with multiple Iterations, find N_start_new for the starting position of the output alignments of the final round
        line_count += 1
        if line_count >= N_blast:
            break
        that = line.split()
        pdbID = that[0]
        e_score = float(that[10])
        if e_score < e_score_old:
            N_start_new = line_count
        if pdbID != pdbID_old:
            e_score_old = e_score
            pdbID_old = pdbID
    logging.debug(f"Number of searched PDBs: {N_blast - N_start_new + 1}")
    
    # Convert psiblastOut to a list, sorted by evalue
    psilist = [None] * (N_blast - N_start_new + 1)
    line_count = 0
    kk = 0
    for line in psiblastOut:
        line_count += 1
        if line_count < N_start_new:
            continue
        if line_count > N_blast:
            break
        that = line.split()
        list_tmp = list()
        for ii in range(0, 11):  # PSI-BLAST output has 11 columns
            if not ii == 10:
                list_tmp.append(that[ii])  # column 10 is evalue
            else:
                list_tmp.append(float(that[ii]))
        psilist[kk] = list_tmp
        kk += 1
        logging.debug(list_tmp[:20])
    psilist.sort(key=lambda x: x[10])


    result = []
    # write output alignments to match file
    for jj in range(0, N_blast - N_start_new + 1):
        this = psilist[jj]
        this[10] = str(this[10])
        this.append(str(i))
        queryStart = int(this[1]) + rangeStart + residue_base
        queryEnd = int(this[2]) + rangeStart + residue_base
        this[1] = str(queryStart)
        this[2] = str(queryEnd)
        out = ' '.join(this)
        gaps = this[8]
        if(gaps == '0'):
            result+=[out]  # skip gapped alignments
    
    return result


def download_pdb(pdbID, pdb_dir, max_retries=5):
    pdbID_lower = pdbID.lower()
    filename = f"{pdbID_lower.upper()}.pdb"
    filepath = Path(pdb_dir) / filename

    if filepath.exists() and filepath.stat().st_size > 0:
        logging.info(f"File {filename} already exists, skipping download.")
        return True

    download_url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/{pdbID_lower[1:3]}/pdb{pdbID_lower}.ent.gz"
    temp_gz_path = Path(pdb_dir) / f"{pdbID_lower}.ent.gz"

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Download the file with timeout
            response = requests.get(download_url, stream=True, timeout=10)
            response.raise_for_status()  # Check for HTTP errors

            # Write to temporary file
            with open(temp_gz_path, 'wb') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            # Unzip the file
            with gzip.open(temp_gz_path, 'rb') as gz_file:
                with open(filepath, 'wb') as output_file:
                    output_file.write(gz_file.read())

            logging.info(f"Successfully downloaded and saved {filename} to {filepath}")
            return True

        except requests.exceptions.HTTPError as e:
            logging.warning(f"HTTP error on retry {retry_count + 1} for {pdbID}: {e}")
        except requests.exceptions.ConnectionError as e:
            logging.warning(f"Connection error on retry {retry_count + 1} for {pdbID}: {e}")
        except requests.exceptions.Timeout as e:
            logging.warning(f"Timeout occurred on retry {retry_count + 1} for {pdbID}: {e}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request error on retry {retry_count + 1} for {pdbID}: {e}")
        except IOError as e:
            logging.warning(f"File I/O error on retry {retry_count + 1} for {pdbID}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred on retry {retry_count + 1} for {pdbID}: {e}")
        finally:
            retry_count += 1
            if temp_gz_path.exists():
                temp_gz_path.unlink()

    logging.error(f"Failed to download {filename} after {max_retries} retries.")
    return None

def download_pdbs(pdb_ids, pdb_dir):
    failed_pdb = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pdb = {executor.submit(download_pdb, pdb_id, pdb_dir): pdb_id for pdb_id in pdb_ids}
        
        for future in as_completed(future_to_pdb):
            pdb_id = future_to_pdb[future]
            try:
                result = future.result()
                if result:
                    logging.info(f"Download succeeded for {pdb_id}")
                    failed_pdb[pdb_id] = False
                else:
                    logging.info(f"Download failed for {pdb_id}")
                    failed_pdb[pdb_id] = True
            except Exception as e:
                logging.error(f"Error processing {pdb_id}: {str(e)}")
                failed_pdb[pdb_id] = True

    return failed_pdb

def download_pdb_seqres(pdb_seqres):
    logging.debug(f"Checking if {pdb_seqres} exists")
    pdb_seqres=Path(pdb_seqres)
    if not pdb_seqres.exists():
        import urllib
        logging.warning("pdb_seqres.txt was not found. Attempting download from https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt")
        url = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"
        try:
            urllib.request.urlretrieve(url, pdb_seqres)
            logging.info(f"Download complete. Saved to {pdb_seqres}")
            return pdb_seqres
        except urllib.error.URLError as e:
            logging.error(f"Error downloading file: {e.reason}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        raise FileNotFoundError(f"Failed to download {pdb_seqres}. Make sure it exists in {pdb_seqres} or provide a valid path.")

def get_openmm_io_class(file_type):
    if file_type == "pdb":
        io_class = PDBFile
    elif file_type == "cif":
        io_class = PDBxFile
    else:
        raise ValueError(f"Expected file_type 'pdb' or 'cif' but got file_type={file_type}")
    return io_class

def is_regular_res(residue):
    names = [atom.name for atom in residue.atoms()]
    has_N = "N" in names
    has_CA = "CA" in names
    has_C = "C" in names
    has_CB = "CB" in names
    if residue.name == "GLY":
        condition = (has_N and has_CA and has_C)
    else:
        #condition = (has_N and has_CA and has_C and has_CB)
        condition = (has_N and has_CA and has_C)
    return condition

def create_index_files(iter, line, N_mem, brain_damage,count, failed_pdb,homo, homo_count, weight, frag_lib_dir, pdb_dir, index_dir, pdb_seqres):
    canonical_resnames = ["ALA",'CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS',
                 'LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
    # add a few noncanonical residues that are very easy to work with from the perspective of fragmem construction
    # these are all included in the pdb2gro script
    canonical_resnames.append("MSE") # selenomethionine
    canonical_resnames.append("3ML") # N-trimethyl lysine
    canonical_resnames.append("CAS") # a weird arsenic-modified cysteine, https://www.rcsb.org/ligand/CAS
    # there are lots of others, such as selenocysteine (rcsb.org/ligand/SE7, possibly others) and pyrrollysine,
    # but we'll stick with the convention currently found in pdb2gro.py

    # define function to check if we're dealing with a "regular" residue containing at least an N, CA, and C


    Missing_count=0
    Missing_pdb={}
    out=''
    out1=''
    if not(iter == 0):
        entries = line.split()
        windows_index_str = entries[11]
        if count[windows_index_str] >= N_mem:
            return out, out1, Missing_pdb, Missing_count
        # pdbfull = str(entries[0])
        entry = entries[0]
        if entry[:3] == "pdb":
            # for example 'pdb|4V12|A'
            pdbfull = str(entry[4:8]) + str(entry[9:])
        else:
            pdbfull = str(entry)
        pdbID = pdbfull[0:4].lower() # turns something like "1R69A" into "1r69"
        chainID = pdbfull[4:5] # grabs chain "A"
        groFile = frag_lib_dir + pdbID + chainID + ".gro"
        groName = pdbID + chainID + ".gro"
        pdbFile = pdb_dir + pdbID.upper() + ".pdb"
        indexFile = index_dir + pdbID + chainID + ".index"

        if failed_pdb[pdbID]:
            return out, out1, Missing_pdb, Missing_count  # failed-downloaded ones are still in matchlines, need to be ignored
        if brain_damage == 2:
            if homo[pdbID]:
                homo_count[pdbID] += 1
                pass
            else:
                print(pdbID, "is not a homolog, discard")
                return out, out1, Missing_pdb, Missing_count

        if homo[pdbID]:
            if brain_damage == 0:
                print(pdbID, " Using  a homolog.")
                homo_count[pdbID] += 1
            if brain_damage == 1:
                print(pdbID, " is a homolog, discard")
                return out, out1, Missing_pdb, Missing_count
        residue_list = entries[6]  # sseq (the sequence of the blast hit)

        res_Start = int(entries[3])
        res_End = int(entries[4])
        print(pdbFile, "start: ", res_Start, "end: ", res_End)
        length = res_End - res_Start + 1
        # Do I have the index file?  If No, write it

        if not os.path.isfile(indexFile):
            # generate fasta file
            fastaFile = pdbID + '_' + chainID
            with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:
                exeline = "grep -A1 " + fastaFile + " " + pdb_seqres + " > " + temp_file.name
                print("generating fastaFile: ", fastaFile)
                # p = os.popen(exeline)
                subprocess.Popen(exeline, shell=True).wait()
                # p_status = p.wait()
                if os.path.getsize(temp_file.name) > 0:
                    writeIndexFile(temp_file.name, pdbFile,
                                indexFile, chainID)
                    print("Writing indexFile: ", indexFile)
        else:
            print(indexFile, "exist, no need to create.")

        if not os.path.isfile(indexFile):
            print("Can't create index file, ignore and go on!")
            return out, out1, Missing_pdb, Missing_count

        # Read index file
        index = open(indexFile, 'r')
        # create new_index for frag_seq starting position
        line_count = 0
        flag = ' '
        index_shift = 0
        # read and get the flag
        indexlines = list()
        for index_line in index.readlines():
            indexlines.append(index_line)
            line_count += 1
            tmp_line = index_line.split()
            if line_count == 1:  # first line is the flag
                flag = tmp_line[0]  # index_line
            if flag == "SHIFT" and line_count == 2:
                index_shift = int(tmp_line[0])
                print("shift: ", tmp_line[0])

        r_list = ''  # list()
        if flag == "SKIP":
            Missing_pdb[pdbID] = 1
            Missing_count += 1
            print("***********", flag)
            print("SKIP pdb:", pdbID + chainID)
            return out, out1, Missing_pdb, Missing_count
        elif flag == "FULLMATCH":
            new_index = int(entries[3])
            r_list = residue_list
            print("***********", flag)
        elif flag == "SHIFT":
            new_index = int(entries[3]) + index_shift
            r_list = residue_list
            print("***********", flag)
        elif flag == "INDEXED":
            print("***********", flag)
            # check if there is gaps for the fragment of sequence
            count_flag = 0
            line_count1 = 0
            for index_line in indexlines:
                line_count1 += 1
                if not line_count1 == 1:
                    index_entries = index_line.split()
                    seq_id = int(index_entries[0])
                    res_id = int(index_entries[1])
                    if seq_id < res_Start:
                        continue
                    if seq_id > res_End:
                        break
                    if res_id == -1:
                        print("Missing residues in PDB: ", pdbID + chainID)
                        break
                    if count_flag == 0:
                        new_index = res_id
                        count_flag += 1
                    res_nm = index_entries[2]
                    r_list += res_nm
        else: # the index file does not follow expected syntax
            print("Skip wrongly written index file ", indexFile)
            return out, out1, Missing_pdb, Missing_count

        if r_list != residue_list:
            print("Missing residues: ", pdbID + chainID, residue_list, " incomplete: ", r_list)
            Missing_pdb[pdbID] = 1
            Missing_count += 1
            return out, out1, Missing_pdb, Missing_count

        if os.path.isfile(pdbFile):
            #print("if")
            '''
            if not os.path.isfile(groFile):
                print("converting...... " + pdbFile + " --> " + groFile + " chainID: " + chainID)
                Pdb2Gro(pdbFile, groFile, chainID)
            else:
                print("Exist " + groFile)
            '''
            # groFile used to represent a .gro file
            # now, it just represents the path and name we want to use for our file
            # and we will replace '.gro' with the appropriate extension 
            extension = pdbFile[-3:]
            if not os.path.isfile(f'{groFile[:-4]}.{extension}'):
                print("converting...... " + pdbFile + " --> " + groFile[:-4] + "." + extension + " chainID: " + chainID)
            else:
                print("Exist " + groFile[:-4] + "." + extension)
            io_class = get_openmm_io_class(extension)
            temp = io_class(pdbFile)  
            # sometimes, molecules with the same chain ID can be read in as separate chains
            # if one of these chains is water/ions, we can just ignore it. 
            # Otherwise, we should raise an error
            file_chain = None
            chains_with_correct_id = [chain for chain in temp.getTopology().chains() if chain.id == chainID]
            if len(chains_with_correct_id) > 1:
                for chain in chains_with_correct_id:
                    residue_info = [1 if (residue.name in canonical_resnames and is_regular_res(residue))
                                         else 0 for residue in chain.residues()]
                    if sum(residue_info) > 0: # if True, then we have canonical residues in our segment
                        if file_chain: # if we have previously assigned file_chain, then we have two chains with canonical residues in them and the same id
                                       # this is getting weird, so we're going to return our empty tuple and just not get fragmems from this file
                            return out, out1, Missing_pdb, Missing_count
                        else: # we can assign this chain to file_chain (the chain from the file that we want) if we havent done that already
                            file_chain = chain
            else:
                file_chain = chains_with_correct_id[0]
            # make sure this chain has canonical amino acids
            assert sum([1 if (residue.name in canonical_resnames and is_regular_res(residue)) 
                             else 0 for residue in file_chain.residues()]) > 0    
            #for file_chain in temp.getTopology().chains():
            #    print(file_chain.id)
            #    print(pdbFile)
            #    exit()
                # we only want to work with one chain, but we have to get it by iterating over all of them and checking the ID
                #if chainID == file_chain.id:
            #print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nfound\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            new_top = Topology() # really an openmm.app.Topology
            new_chain = new_top.addChain(id=file_chain.id)
            pos_indices = []
            visited_residue_ids = [] # see below
            added_residues = 0 # see below
            
            residues = [residue for residue in file_chain.residues()]
            for residue in residues:
                #print(residue.id)
                #print(residue.name)
                # We use this to eliminate pdb/cif files with multiple insertion codes at a single residue id
                # (rare; for example, see 3vpg resid 220).
                # Eliminating these files maintains the old convention (from pdb2gro) and also makes a lot of sense,
                # although you could argue that the best possible thing to do would be to create two seprate memories,
                # one for each mutant/coordinate set.
                if residue.id in visited_residue_ids:
                    break
                else:
                    visited_residue_ids.append(residue.id)
                # make sure we don't add water, cofactor, or nonstandard residues to new topology
                if residue.name not in canonical_resnames or (not is_regular_res(residue)):
                    #print(residue.name)
                    continue
                # add residue
                new_residue = new_top.addResidue(residue.name, new_chain, id=residue.id, insertionCode=residue.insertionCode)
                atom_names = []
                for atom in residue.atoms():
                    if atom.name in atom_names:
                        if atom.element == element.hydrogen and "H1" not in atom_names and residue.index==residues[0].index:
                            # openmm sometimes doesn't parse H atom names correctly (maybe related to the N-terminus?)
                            # for example, in pdb 1MUW, atom id 13 with name H1
                            # in the pdb file gets read in as having name H,
                            # which causes an issue because there is another atom in
                            # that residue with the name H
                            new_top.addAtom('H1', atom.element, new_residue, id=atom.id)
                            pos_indices.append(atom.index)
                            atom_names.append(atom.name)
                        else:
                            raise ValueError(f"""
                                                Duplicate atom name in residue {residue}, id {residue.id}, chain {residue.chain}.
                                                pdbFile is {pdbFile}, atom index is {atom.index}. Full residue atom info is:
                                                [index, id, name]
                                                {[[atom_again.index, atom_again.id, atom_again.name] for atom_again in residue.atoms()]}                     
                                            """)
                            '''
                            WE ACTUALLY DON'T EXPECT THIS BLOCK TO EVER RUN BECAUSE ALTLOCS ARE BASICALLY REMOVED BY 
                            OPENMM WHEN LOADING THE Topology (IT JUST TAKES THE FIRST OCCURENCE OF EACH ATOM)
                            THIS IS DIFFERENT FROM pdb2gro, WHICH WILL WRITE THE LAST OCCURENCE OF EACH ATOM
                            print('found')
                            print([atom for atom in residue.atoms()])
                            # if we want to keep the last one in the event of an atom-level altloc
                            # we don't want to add another Atom to the topology,
                            # but we need to change the coordinates of the old one
                            where = -(len(atom_names) - atom_names.index(atom.name))
                            pos_indices[where] = atom.index
                            # if we want to keep the first one in the event of an atom-level altloc
                            #continue 
                            '''
                    else:
                        new_top.addAtom(atom.name, atom.element, new_residue, id=atom.id)
                        pos_indices.append(atom.index)
                        atom_names.append(atom.name)
                # if residue is a part of the fragment hit, increment our counter
                if res_Start <= int(residue.id) <= res_End:
                    added_residues += 1
                    #print(f'added_residues: {added_residues}')
                else:
                    pass
                    #print((res_Start,res_End,int(residue.id)))
                    #print(f'added_residues: {added_residues}')
                
            else: # executes if the for loop exits without encountering break
                if added_residues == length:
                    pos = temp.getPositions(asNumpy=True)[pos_indices,:]
                    #print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nwrit4\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
                    io_class.writeFile(new_top, pos, f'{groFile[:-4]}.{extension}', keepIds=True)    
                    #print('changing return')
                    # increment our counter so that we don't take too many memories
                    count[windows_index_str] += 1
                    out = f'{groFile[:-4]}.{extension} {entries[1]} ' # queue start
                    out += str(new_index) + ' ' + str(length) + ' ' + \
                        str(weight) + "\n"  # frag_seq start
                    out1 = windows_index_str + ' ' + str(count[windows_index_str])
                    out1 += f' {entries[9]} {entries[10]} {groName[:-4]}.{extension}' # groName has pdbid and chain info
                    out1 += ' ' + entries[1] + ' ' + str(new_index) + ' ' + str(
                        length) + ' ' + str(weight) + ' 0' + entries[5] + ' 0' + entries[6] + "\n"
                else: # unexpected, but could happen due to skipped noncanonical amino acid in backbone fragment,
                      #
                      # or a gap in the residue numbering in the pdb (which someone who probably never had to run an MD simulation
                      # https://bioinformatics.stackexchange.com/questions/11587/what-is-the-aim-of-insertion-codes-in-the-pdb-file-format
                      # thinks is a very reasonable thing that all code should handle)
                      #
                      # another source of a gap would be an incomplete residue (one that failed to pass is_regular_res)
                      #
                    pass # we will not try to create memories out of fragments with gaps  
                         # so we allow the function to return the information-lacking tuple (out, out1, Missing_pdb, Missing_counter)
             
        else:
            print(pdbFile, "does not exist! Go figure...")
    
    return out, out1, Missing_pdb, Missing_count


def create_fragment_memories(database, fasta_file, memories_per_position, brain_damage, fragment_length, 
         pdb_dir, index_dir, frag_lib_dir, failed_pdb_list_file, pdb_seqres,
         weight, evalue_threshold, cutoff_identical):
        # set up directories
    
    download_pdb_seqres(pdb_seqres)

    pdb_dir.mkdir(exist_ok=True)
    index_dir.mkdir(exist_ok=True)
    frag_lib_dir.mkdir(exist_ok=True)

    if not pdb_dir.exists() or not frag_lib_dir.exists() or not index_dir.exists():
        print("Can't create necessary directories")
        exit()

    failed_download_pdb_list = []
    if failed_pdb_list_file.exists():
        with failed_pdb_list_file.open() as f:
            failed_download_pdb_list = [line.strip() for line in f]

    #Convert paths to strings
    database = str(database)
    pdb_dir=str(pdb_dir)+'/'
    index_dir=str(index_dir)+'/'
    frag_lib_dir=str(frag_lib_dir)+'/'
    failed_pdb_list_file=str(failed_pdb_list_file)
    pdb_seqres=str(pdb_seqres)

    handle = open(fasta_file, "r")
    N_mem=memories_per_position
    residue_base=0

    with open('frags.mem', 'w') as LAMWmatch, open('log.mem', 'w') as log_match:
        LAMWmatch.write('[Target]' + "\n")
        LAMWmatch.write("query" + "\n\n" + '[Memories]' + "\n")
        for record in SeqIO.parse(handle, "fasta"):
            # Part I, BLAST fragments
            if(len(record.seq) < fragment_length):
                print("Exception::query sequence is shorter than " + str(fragment_length) + " residues. Exit.")
                sys.exit()

            query = str(record.name)[0:4]
            chain = record.name.split(":")[-1]
            print('processing sequence:', record.name)


            # FRAGMENT GENERATION LOOP
            iterations = len(record.seq) - fragment_length + \
                1  # number of sliding windows

            with ThreadPoolExecutor(max_workers=12) as executor:
                futures={}
                for i in range(1, iterations + 1):
                    future = executor.submit(process_window, i, record, fragment_length, evalue_threshold, database, residue_base)
                    futures[future] = i-1
                results = [None] * len(futures)
                for future in as_completed(futures):
                    i = futures[future]
                    results[i] = future.result()
                
            # Writing the results to the match file in the order of execution
            with open('prepFrags.match', 'w') as match:
                match.write(query + "\n")
                for result in results:
                    for line in result:
                        match.write(line + '\n')

            # loop2 close
            with open('prepFrags.match', 'r') as match:
                # list unique PDB IDs for downloading later
                matchlines = list()
                keys = {}
                for line in match.readlines():
                    matchlines.append(line)
                    entries = line.split()
                    entry = entries[0]
                    if entry[:3] == "pdb":
                        # for example 'pdb|4V12|A'
                        pdbfull = str(entry[4:8]) + str(entry[9:])
                    else:
                        pdbfull = str(entry)
                    keys[pdbfull] = 1
                unique = list(keys.keys())

            # pdbparse=PDBParser(PERMISSIVE=1)

            # Part II, BLAST the whole sequence to find homologs
            print(record.seq)
            fragment = open('fragment.fasta', 'w')
            fragment.write(str(record.seq))
            fragment.close()
            
            #Download PDBs
            failed_pdb = {pdb_id[0:4].lower():1 for pdb_id in unique}
            failed_pdb.update(download_pdbs([pdb_id[0:4].lower() for pdb_id in unique if pdb_id[0:4].lower() not in failed_download_pdb_list], pdb_dir))
            homo = {pdbID:0 for pdbID in failed_pdb if not failed_pdb[pdbID]}
            homo_count = {pdbID:0 for pdbID in failed_pdb if not failed_pdb[pdbID]}
            update_failed_pdb_list(failed_pdb,failed_pdb_list_file)

            # blast the whole sequence to identify homologs Evalue 0.005
            exeline = "psiblast -num_iterations 1 -word_size 3 -evalue 0.005"
            exeline += " -outfmt '6 sseqid slen bitscore score evalue pident' -matrix BLOSUM62 -db " + \
                database + " -query fragment.fasta"
            print("finding homologs")
            print("executing::: " + exeline)
            homoOut = os.popen(exeline).read()
            homoOut = homoOut.splitlines()  # now an array
            for line in homoOut:
                entries = line.split()
                print("homologues: ", entries)
                if len(entries):
                    pdbfull = entries[0]
                    pdbID = pdbfull[0:4].lower()
                    if brain_damage == 2:
                        identity = float(entries[5])
                        # exclude self(>90% identity)
                        if identity <= cutoff_identical:
                            homo[pdbID] = 1
                            homo_count[pdbID] = 0
                    if brain_damage == 0.5:
                        identity = float(entries[5])
                        # check identity, add only self (>90% identity) to homo[]
                        if identity > cutoff_identical:
                            homo[pdbID] = 1
                            homo_count[pdbID] = 0
                    else:
                        homo[pdbID] = 1
                        homo_count[pdbID] = 0

            # Part III, Write memories
            count = {}
            for i in range(1, iterations + 1):
                count[str(i)] = 0  # count number of mem per fragments
            fastFile = "./tmp.fasta"
            Missing_pdb={}
            Missing_count=0

            with ThreadPoolExecutor(max_workers=12) as executor:
                futures={}
                for iteration_number,line in enumerate(matchlines):
                    future = executor.submit(create_index_files,iteration_number, line, N_mem, brain_damage,count, failed_pdb,homo, homo_count, weight, frag_lib_dir, pdb_dir, index_dir, pdb_seqres)
                    futures[future] = iteration_number
                    #if iteration_number==1:
                    #    break
                results = [None] * len(futures)
                for future in as_completed(futures):
                    i = futures[future]
                    results[i] = future.result()
            #print('results:')
            #print(results)
            #print(bool(results[0]))
            # Writing the results to the match file in the order of execution
            
            for result in results:
                if result:
                    out, out1, Missing_pdb_out, Missing_count_out = result
                    LAMWmatch.write(out)
                    log_match.write(out1)
                    Missing_pdb.update(Missing_pdb_out)
                    Missing_count+=Missing_count_out

            print("HOMOLOGS:::")
            total_homo_count = 0
            for line in homoOut:
                entries = line.split()
                print("sseqid slen bitscore score evalue pident")
                print(entries)
                entry = entries[0]
                if entry[:3] == "pdb":
                    # for example 'pdb|4V12|A'
                    pdbfull = str(entry[4:8]) + str(entry[9:])
                else:
                    pdbfull = str(entry)
                # pdbfull = entries[0]
                pdbID = pdbfull[0:4].lower()
                if brain_damage == 0 or brain_damage == 2:
                    total_homo_count += homo_count[pdbID]
                    print("Homolog count =", homo_count[pdbID])

            if brain_damage == 0 or brain_damage == 2:
                print("Total homolog count = ", total_homo_count, round(total_homo_count / iterations, 2))

            print("Memories per position that are fewer than expected:")
            for i in count:
                if count[i] < N_mem:
                    print(i, count[i])

            print("Number of blasted PDB: ", len(failed_pdb))
            print("Number of failed downloaded PDB: ", sum(failed_pdb.values()))
            print("Number of PDB with Missing atoms: ", len(Missing_pdb))
            print("Discarded fragments with Missing atoms: ", Missing_count)
            residue_base += len(record.seq)
            for line in homoOut:
                entries = line.split()
                if len(entries):
                    entry = entries[0]
                    if entry[:3] == "pdb":
                        # for example 'pdb|4V12|A'
                        pdbfull = str(entry[4:8]) + str(entry[9:])
                    else:
                        pdbfull = str(entry)
                    # pdbfull = entries[0]
                    pdbID = pdbfull[0:4].lower()
                    print(pdbID)

    # loop1 close

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("database_prefix")
    parser.add_argument("fasta_file", type=Path)
    parser.add_argument("--memories_per_position", type=int, default=20)
    parser.add_argument("--brain_damage_flag", type=float, default=1)
    parser.add_argument("--frag_length", type=int, default = 10)
    parser.add_argument("--pdb_dir", type=Path, default=Path("PDBs"))
    parser.add_argument("--index_dir", type=Path, default=Path("Indices"))
    parser.add_argument("--frag_lib_dir", type=Path, default=Path("Gros"))
    parser.add_argument("--failed_pdb_list_file", type=Path, default=Path("notExistPDBsList"))
    parser.add_argument("--pdb_seqres", type=Path, default=Path("pdb_seqres.txt"))
    parser.add_argument("--weight", type=int, default=1)
    parser.add_argument("--evalue_threshold", type=int, default=10000)
    parser.add_argument("--cutoff_identical", type=int, default=90)

    args = parser.parse_args()

    create_fragment_memories(args.database_prefix, args.fasta_file, args.N_mem, args.brain_damage_flag, 
         args.frag_length, args.pdb_dir, args.index_dir, args.frag_lib_dir, args.failed_pdb_list_file, args.pdb_seqres,
         args.weight, args.evalue_threshold, args.cutoff_identical)
