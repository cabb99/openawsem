import argparse

import sys
import os
from .IndexPdb import *
from .Pdb2GroLib import *
from Bio import SeqIO
from pathlib import Path
import subprocess
import concurrent
import threading



import requests
import gzip
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize a lock for thread-safe file writing
file_write_lock = threading.Lock()

def download_pdb(pdbID, pdb_dir, max_retries=3):
    pdbID_lower = pdbID.lower()
    filename = f"{pdbID_lower.upper()}.pdb"
    filepath = Path(pdb_dir) / filename

    if filepath.exists():
        logging.info(f"File {filename} already exists, skipping download.")
        return True

    download_url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/{pdbID_lower[1:3]}/pdb{pdbID_lower}.ent.gz"
    temp_gz_path = Path(pdb_dir) / f"{pdbID_lower}.ent.gz"

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Download the file with timeout
            response = requests.get(download_url, stream=True, timeout=10)
            response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

            with open(temp_gz_path, 'wb') as f:
                f.write(response.content)

            # Unzip the file
            with gzip.open(temp_gz_path, 'rb') as gz_file:
                with open(filepath, 'wb') as f_out:
                    f_out.write(gz_file.read())

            logging.info(f"Downloaded and saved {filename}")
            return True

        except Exception as e:
            logging.warning(f"Retry {retry_count + 1} for {pdbID_lower.upper()}: {e}")
            retry_count += 1
        finally:
            # Clean up any created temporary files
            if temp_gz_path.exists():
                temp_gz_path.unlink()

    logging.error(f"Failed to download after {max_retries} retries: {pdbID_lower.upper()}")
    return False


def process_fragment(fragment_sequence, rangeStart , evalue_threshold, database, residue_base, match_file, file_write_lock):
    
    print("fragment subrange:::" + fragment_sequence)

    # Write fragment to a temporary file
    temp_fragment_file = Path(f"temp_fragment_{fragment_sequence}.fasta")
    with temp_fragment_file.open('w') as fragment:
        fragment.write(fragment_sequence)

    # Construct PSI-BLAST command
    exeline = f"psiblast -num_iterations 5 -comp_based_stats 0 -word_size 2 -evalue {evalue_threshold} " + \
              f"-outfmt '6 sseqid qstart qend sstart send qseq sseq length gaps bitscore evalue' -matrix BLOSUM62 -threshold 9 -window_size 0 " + \
              f"-db {database} -query {temp_fragment_file}"

    # Execute PSI-BLAST and process output
    
    try:
        print("executing:::" + exeline)
        result = subprocess.run(exeline, shell=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"PSI-BLAST execution failed: {e}")
    finally:
        # Clean up temporary file
        temp_fragment_file.unlink()

    psiblastOut = result.stdout.splitlines()  # Splitting the output into lines
    N_blast = len(psiblastOut)
    # print psiblastOut
    if psiblastOut[N_blast - 1] == 'Search has CONVERGED!':
        N_blast = N_blast - 2  # exclude last two lines for the text

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
    print("Number of searched PDBs:  ", N_blast, N_start_new)

    # convert psiblastOut to a list, sorted by evalue
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
        #print(list_tmp)
    psilist.sort(key=lambda x: x[10])

    # write output alignments to match file
    for jj in range(0, N_blast - N_start_new + 1):
        this = psilist[jj]
        this[10] = str(this[10])
        this.append(str(rangeStart+1))
        queryStart = int(this[1]) + rangeStart + residue_base
        queryEnd = int(this[2]) + rangeStart + residue_base
        this[1] = str(queryStart)
        this[2] = str(queryEnd)
        out = ' '.join(this)
        out += '\n'
        gaps = this[8]
        if(gaps == '0'):
            # Thread-safe writing to the match file
            with file_write_lock:
                with open(match_file, "a") as match:
                    match.write(out)


    return True


def create_fragment_memories(database, fasta_file, memories_per_position, brain_damage, fragment_length, 
         pdb_dir, index_dir, frag_lib_dir, failed_pdb_list_file, pdb_seqres,
         weight, evalue_threshold, cutoff_identical):
    
    # set up directories
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

    # set up out
    LAMWmatch = open('frags.mem', 'w')
    LAMWmatch.write('[Target]' + "\n")
    LAMWmatch.write("query" + "\n\n" + '[Memories]' + "\n")
    log_match = open('log.mem', 'w')

    #Convert paths to strings
    database = str(database)
    index_dir=str(index_dir)+'/'
    frag_lib_dir=str(frag_lib_dir)+'/'
    failed_pdb_list_file=str(failed_pdb_list_file)
    pdb_seqres=str(pdb_seqres)

    handle = open(fasta_file, "r")
    N_mem=memories_per_position
    residue_base=0


    for record in SeqIO.parse(handle, "fasta"):
        # Part I, BLAST fragments
        if(len(record.seq) < fragment_length):
            print("Exception::query sequence is shorter than " + str(fragment_length) + " residues. Exit.")
            sys.exit()

        query = str(record.name)[0:4]
        print('processing sequence:', record.name)
        with file_write_lock:
            with open('prepFrags.match', "w") as match:
                match.write(query + "\n")

        # FRAGMENT GENERATION LOOP
        iterations = len(record.seq) - fragment_length + 1  # number of sliding windows
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(1, iterations + 1):  # loop2, run through all sliding windows in a given chain
                rangeStart = i - 1
                rangeEnd = i + fragment_length - 1
                fragment_sequence = str(record[rangeStart:rangeEnd].seq)

                # Submit each fragment processing task to the thread pool
                future = executor.submit(process_fragment, fragment_sequence, rangeStart, evalue_threshold, database, residue_base, 'prepFrags.match', file_write_lock)
                futures.append(future)
            
            # Wait for all futures to complete
            for future in futures:
                result = future.result()
                
        # loop2 close
        match = open('prepFrags.match', 'r')  # match is read-only now

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
        failed_pdb = {}
        homo = {}
        homo_count = {}

        for pdbfull in unique:
            pdbID = pdbfull[0:4].lower()
            failed_pdb[pdbID] = 0
            homo[pdbID] = 0
            homo_count[pdbID] = 0

        # ThreadPoolExecutor for downloading PDB files
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_pdbID = {executor.submit(download_pdb, pdbfull[0:4].lower(), pdb_dir): pdbfull[0:4].lower() for pdbfull in unique}

            # Process the results of the futures as they complete
            for future in concurrent.futures.as_completed(future_to_pdbID):
                pdbID = future_to_pdbID[future]
                try:
                    success = future.result()
                    if not success:
                        failed_pdb[pdbID] = 1
                except Exception as exc:
                    logging.error(f"{pdbID} generated an exception: {exc}")
                    failed_pdb[pdbID] = 1



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
        iter = 0
        count = {}
        for i in range(1, iterations + 1):
            count[str(i)] = 0  # count number of mem per fragments
        Missing_count = 0
        Missing_pdb = {}
        fastFile = "./tmp.fasta"

        for line in matchlines:  # loop3
            iter += 1
            if not(iter == 1):
                entries = line.split()
                windows_index_str = entries[11]
                if count[windows_index_str] >= N_mem:
                    continue
                # pdbfull = str(entries[0])
                entry = entries[0]
                if entry[:3] == "pdb":
                    # for example 'pdb|4V12|A'
                    pdbfull = str(entry[4:8]) + str(entry[9:])
                else:
                    pdbfull = str(entry)
                pdbID = pdbfull[0:4].lower()
                pdbIDsecond = pdbfull[1:2].lower()
                pdbIDthird = pdbfull[2:3].lower()
                chainID = pdbfull[4:5].lower()
                groFile = frag_lib_dir + pdbID + chainID + ".gro"
                groName = pdbID + chainID + ".gro"
                pdbFile = pdb_dir/ f"{pdbID.upper()}.pdb"
                indexFile = index_dir + pdbID + chainID + ".index"

                if failed_pdb[pdbID]:
                    continue  # failed-downloaded ones are still in matchlines, need to be ignored
                if brain_damage == 2:
                    if homo[pdbID]:
                        homo_count[pdbID] += 1
                        pass
                    else:
                        print(pdbID, "is not a homolog, discard")
                        continue

                if homo[pdbID]:
                    if brain_damage == 0:
                        print(pdbID, " Using  a homolog.")
                        homo_count[pdbID] += 1
                    if brain_damage == 1:
                        print(pdbID, " is a homolog, discard")
                        continue
                residue_list = entries[6]  # sseq

                res_Start = int(entries[3])
                res_End = int(entries[4])
                print(pdbFile, "start: ", res_Start, "end: ", res_End)
                # Do I have the index file?  If No, write it

                if not os.path.isfile(indexFile):
                    # generate fasta file
                    if not os.path.isfile(pdb_seqres):
                        print(pdb_seqres)
                        print("Need to download pdb_seqres.txt from PDB!")
                        print("ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt")
                        print("Copy to $HOME/opt/script/")
                        exit()
                    fastaFile = pdbID + '_' + chainID.upper()
                    exeline = "grep -A1 " + fastaFile + " " + pdb_seqres + " > ./tmp.fasta"
                    print("generating fastaFile: ", fastaFile)
                    # p = os.popen(exeline)
                    subprocess.Popen(exeline, shell=True).wait()
                    # p_status = p.wait()
                    if os.path.getsize('./tmp.fasta') > 0:
                        writeIndexFile(fastFile, pdbFile,
                                    indexFile, chainID.upper())
                        print("Writing indexFile: ", indexFile)
                else:
                    print(indexFile, "exist, no need to create.")

                if not os.path.isfile(indexFile):
                    print("Can't create index file, ignore and go on!")
                    continue

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
                    continue
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
                else:
                    print("Skip wrongly written index file ", indexFile)
                    continue

                if r_list != residue_list:
                    print("Missing residues: ", pdbID + chainID, residue_list, " incomplete: ", r_list)
                    Missing_pdb[pdbID] = 1
                    Missing_count += 1
                    continue

                if os.path.isfile(pdbFile):
                    if not os.path.isfile(groFile):
                        Pdb2Gro(pdbFile, groFile, chainID.upper())
                        print("converting...... " + pdbFile + " --> " + groFile)
                    else:
                        print("Exist " + groFile)
                    count[windows_index_str] += 1

                    length = res_End - res_Start + 1
                    out = groFile + ' ' + entries[1] + ' '  # queue start
                    out += str(new_index) + ' ' + str(length) + ' ' + \
                        str(weight) + "\n"  # frag_seq start
                    LAMWmatch.write(out)
                    out1 = windows_index_str + ' ' + str(count[windows_index_str])
                    out1 += ' ' + entries[9] + ' ' + entries[10] + ' ' + groName
                    out1 += ' ' + entries[1] + ' ' + str(new_index) + ' ' + str(
                        length) + ' ' + str(weight) + ' 0' + entries[5] + ' 0' + entries[6] + "\n"
                    log_match.write(out1)
                else:
                    print(pdbFile, "does not exist! Go figure...")
            # loop3 ends

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

        print("memories per position that is fewer than expected:")
        for i in count:
            if count[i] < N_mem:
                print(i, count[i])

        print("Number of blasted PDB: ", len(failed_pdb))
        print("Number of failed downloaded PDB: ", sum(failed_pdb.values()))
        print("Number of PDB with Missing atoms: ", len(Missing_pdb))
        print("Discarded fragments with Missing atoms: ", Missing_count)
        print("Failed PDB saved in: ", str(failed_pdb_list_file))
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
                
# Sample test setup
def test_process_fragment():
    evalue_threshold = 10000
    database = "path_to_your_database"
    residue_base = 0
    match_file = "path_to_match_file"
    file_write_lock = threading.Lock()

    result = process_fragment('AAAAAAAAA', evalue_threshold, database, residue_base, match_file, file_write_lock)
    print(result)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Script for processing protein sequences and generating fragment memories.")

    parser.add_argument("database_prefix",
                        help="Specifies the prefix for the database to be used, forming the initial part of the database path or name.")

    parser.add_argument("fasta_file", type=Path,
                        help="Path to the FASTA file containing protein sequences for processing.")

    parser.add_argument("--memories_per_position", type=int, default=20,
                        help="Number of memory fragments to generate per position in the protein sequence. Default is 20.")

    parser.add_argument("--brain_damage_flag", type=float, default=1,
                        help="Floating-point flag to control algorithm behavior, especially in handling homologs and sequence identity. Default is 1.")

    parser.add_argument("--frag_length", type=int, default=9,
                        help="Length of the fragment (sliding window size) to consider in the protein sequence. Default is 9.")

    parser.add_argument("--pdb_dir", type=Path, default=Path("PDBs"),
                        help="Directory path for storing or retrieving PDB files. Default is 'PDBs'.")

    parser.add_argument("--index_dir", type=Path, default=Path("Indices"),
                        help="Directory path for storing index files. Default is 'Indices'.")

    parser.add_argument("--frag_lib_dir", type=Path, default=Path("Gros"),
                        help="Directory path for storing fragment library files. Default is 'Gros'.")

    parser.add_argument("--failed_pdb_list_file", type=Path, default=Path("notExistPDBsList"),
                        help="File path listing PDB IDs that failed download or processing. Default is 'notExistPDBsList'.")

    parser.add_argument("--pdb_seqres", type=Path, default=Path("pdb_seqres.txt"),
                        help="Path to 'pdb_seqres.txt' containing sequence information for PDB entries. Default is 'pdb_seqres.txt'.")

    parser.add_argument("--weight", type=int, default=1,
                        help="Weight to be assigned to fragments during processing. Default is 1.")

    parser.add_argument("--evalue_threshold", type=int, default=10000,
                        help="E-value threshold for PSI-BLAST searches. Default is 10000.")

    parser.add_argument("--cutoff_identical", type=int, default=90,
                        help="Cutoff for sequence identity percentage for filtering homologous sequences. Default is 90.")

    parser.add_argument("--test", action='store_true', default=False,
                        help="Run the script in test mode.")
    
    
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("database_prefix")
    parser.add_argument("fasta_file", type=Path)
    parser.add_argument("--memories_per_position", type=int, default=20)
    parser.add_argument("--brain_damage_flag", type=float, default=1)
    parser.add_argument("--frag_length", type=int, default = 9)
    parser.add_argument("--pdb_dir", type=Path, default=Path("PDBs"))
    parser.add_argument("--index_dir", type=Path, default=Path("Indices"))
    parser.add_argument("--frag_lib_dir", type=Path, default=Path("Gros"))
    parser.add_argument("--failed_pdb_list_file", type=Path, default=Path("notExistPDBsList"))
    parser.add_argument("--pdb_seqres", type=Path, default=Path("pdb_seqres.txt"))
    parser.add_argument("--weight", type=int, default=1)
    parser.add_argument("--evalue_threshold", type=int, default=10000)
    parser.add_argument("--cutoff_identical", type=int, default=90)
    parser.add_argument("--test", action='store_true',help='Tests this script', default=False)

    args = parser.parse_args()

    print(args)
    if args.test:
        test_process_fragment()
    else:
        create_fragment_memories(args.database_prefix, args.fasta_file, args.N_mem, args.brain_damage_flag, 
            args.frag_length, args.pdb_dir, args.index_dir, args.frag_lib_dir, args.failed_pdb_list_file, args.pdb_seqres,
            args.weight, args.evalue_threshold, args.cutoff_identical)
