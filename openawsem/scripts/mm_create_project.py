#!/usr/bin/env python3
import os
import argparse
import sys
import openawsem
import pandas as pd
from pathlib import Path
import contextlib
import subprocess
import logging
from Bio import BiopythonWarning
import warnings

__location__ = openawsem.__location__
__author__ = 'Wei Lu, with modifications by the OpenAWSEM contributors'

class AWSEMSimulationProject:
    def __init__(self,args):
        self.data_path=__location__
        self.base_folder=Path.cwd() #Project folder
        self.args=args
                
    def run_command(self, command, stdout=None):
        logging.debug('Command: '+ str(command))
        subprocess.run(command, check=True, shell=False, stdout=(open(stdout, "w") if stdout else None))

    @contextlib.contextmanager
    def change_directory(self,path):
        logging.debug(f"Changing directory to {path}")
        old_path = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(old_path)

            
    def log_commandline_args(self, log_file = 'create_project_commandline_args.txt'):
        logging.debug("Logging commandline arguments")
        logging.debug(' '.join(sys.argv))
        with open(log_file, 'a') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')

    def prepare_input_files_from_pdb(self, parent_folder="."):
        """
        Prepare input files from a PDB file.
        
        This function checks if the PDB file exists, extracts the name of the protein,
        and copies the PDB file to the 'original_pdbs' directory.
        
        Returns:
            name (str): Protein name without the file extension.
            pdb (str): PDB filename.
        """
        logging.info("Creating simulation folder from PDB file.")
        protein_path = Path(self.args.protein)
        parent_folder = Path(parent_folder)
        assert protein_path.exists(), f"The protein path {str(protein_path)} does not exist"
        
        #Get the name of the pdb
        name = protein_path.stem
        pdb = protein_path.name
        
        #Create a pdb backup
        original_pdbs_dir = parent_folder / "original_pdbs"
        original_pdbs_dir.mkdir(parents=True, exist_ok=True)
        (original_pdbs_dir / pdb).write_bytes(protein_path.read_bytes())
        
        return name, pdb
    
    def prepare_input_files_from_cif(self, parent_folder="."):
        """
        Prepare input files from a mmCIF/PDBx file.
        
        This function checks if the .cif file exists, extracts the name of the protein,
        and copies the .cif file to the 'original_pdbs' directory.
        
        Returns:
            name (str): Protein name without the file extension.
            cif (str): filename.
        """
        logging.info("Creating simulation folder from mmCIF/PDBx file.")
        protein_path = Path(self.args.protein)
        parent_folder = Path(parent_folder)
        assert protein_path.exists(), f"The protein path {str(protein_path)} does not exist"
        
        #Get the name of the pdb
        name = protein_path.stem
        cif = protein_path.name
        
        #Create a pdb backup
        original_pdbs_dir = parent_folder / "original_pdbs"
        original_pdbs_dir.mkdir(parents=True, exist_ok=True)
        (original_pdbs_dir / cif).write_bytes(protein_path.read_bytes())
        
        return name, cif

    def prepare_input_files_from_fasta(self, parent_folder="."):
        """
        Prepare input files from a FASTA file.
        
        This function creates a simulation folder from a FASTA file, extracts the name of the protein,
        generates a PDB file, and copies both the FASTA and PDB files to their respective directories.
        
        Returns:
            name (str): Protein name without the file extension.
            pdb (str): PDB filename.
        """
        logging.info("Creating simulation folder from FASTA file.")
        protein_path = Path(self.args.protein)
        name = protein_path.stem

        try:
            openawsem.helperFunctions.create_extended_pdb_from_fasta(self.args.protein, output_file_name=f"{name}.pdb")
            openawsem.helperFunctions.add_chain_to_pymol_pdb(f"{name}.pdb")
        except Exception as e:
            logging.error(f"Failed to convert FASTA to PDB. Exception: {e}")
            exit()

        original_fasta_dir = Path(parent_folder) / "original_fasta"
        original_pdbs_dir = Path(parent_folder) / "original_pdbs"
        original_fasta_dir.mkdir(parents=True, exist_ok=True)
        original_pdbs_dir.mkdir(parents=True, exist_ok=True)
        fasta_file_path = (original_fasta_dir / protein_path.name)
        fasta_file_path.write_bytes(protein_path.read_bytes())
        pdb_path = Path(f"{name}.pdb")
        crystal_structure_path=(Path(parent_folder) / "crystal_structure.pdb")
        crystal_structure_path.write_bytes(pdb_path.read_bytes())
        pdb = pdb_path.name
        
        return name, pdb

    def prepare_input_files_from_name(self, parent_folder="."):
        """
        Prepare input files from a protein name.
        
        This function downloads the PDB file for the given protein name.
        
        Returns:
            name (str): Protein name without the file extension.
            pdb (str): PDB filename.
        """
        
        name = self.args.protein
        pdb = f"{name}.pdb"
        pdb_list = [name]
        parent_folder = Path(parent_folder)
        
        logging.info("Downloading PDB file {pdb}")
        try:
            openawsem.helperFunctions.downloadPdb(pdb_list, location=parent_folder/'original_pdbs')
        except Exception as e:
            logging.error(f"Failed to download PDB file. Exception: {e}")
            exit()

        return name, pdb

    def process_pdb_files(self,is_cif=False):
        """
        Process the PDB or mmCIF/PDBx files by cleaning, preparing, and generating additional required files.
        """

        removeHeterogens = False if self.args.keepLigands is True else True
        chain = self.args.chain

        if is_cif:
            extension = "cif"
        else:
            extension = "pdb"
        if not Path(f"crystal_structure.{extension}").exists():
            logging.info(f"Creating crystal_structure.{extension} file")
            openawsem.helperFunctions.cleanPdb( # see openawsem/helperFunctions/myFunctions.py
                [self.name],
                chain=chain,
                toFolder="cleaned_pdbs",
                verbose=self.args.verbose,
                keepIds=True,
                removeHeterogens=removeHeterogens,
                extension=extension
            )
            cleaned_pdb_path = Path(f"cleaned_pdbs/{self.pdb}")
            shutil.copy(cleaned_pdb_path,f"crystal_structure.{extension}")
        else:
            logging.info(f"Using existing crystal_structure.{extension} file")

        if chain == "-1":
            logging.info(f"Reading chains info from crystal_structure.{extension}")
            chain = openawsem.helperFunctions.getAllChains(
                f"crystal_structure.{extension}",
                removeDNAchains=True
            )
            logging.info(f"Chains info read from crystal_structure.{extension}, chains to simulate: ", chain)
        else:
            logging.info(f"Selected chains: {chain}")


        # for compute Q
        input_pdb_filename, cleaned_pdb_filename = openawsem.prepare_pdb( # function is in openAWSEM.py
            f"crystal_structure.{extension}",
            chain,
            use_cis_proline=False,
            keepIds=self.args.keepIds,
            removeHeterogens=removeHeterogens
        )
        logging.info(f"Created {input_pdb_filename} and {cleaned_pdb_filename} files")
        
        logging.info("Ensuring AWSEM atom order")
        openawsem.ensure_atom_order(input_pdb_filename)
        
        self.input_pdb_filename = input_pdb_filename # crystal_structure-openmmawsem structure file
        self.cleaned_pdb_filename = cleaned_pdb_filename
        
        ###############################################################################
        # after the prepare_pdb call, we have both a crystal_structure-cleaned.pdb/cif
        # and a crystal_structure-openmmawsem.pdb/cif
        # i don't know why we call this function on the crystal_structure-cleaned.pdb file
        # and at this point in the program
        self.chain = openawsem.helperFunctions.getAllChains(f"crystal_structure-cleaned.{extension}")  
        openawsem.getSeqFromCleanPdb(input_pdb_filename, chains=self.chain, writeFastaFile=True)
        shutil.copy('crystal_structure.fasta',f'{self.name}.fasta')
        ###############################################################################
        if self.args.extended:
            if extension == "cif":
                raise NotImplementedError("Still need to implement cif support for extended pdbs")
            # print("Trying to create the extended structure extended.pdb using pymol, please ensure that pymol is installed and callable using 'pymol' in terminal.")
            # self.run_command(["python", f"{__location__}/helperFunctions/fasta2pdb.py", "extended", "-f", f"{self.name}.fasta"])
            # # print("If you has multiple chains, please use other methods to generate the extended structures.")
            # openawsem.helperFunctions.myFunctions.add_chain_to_pymol_pdb("extended.pdb")  # only work for one chain only now
            logging.info("Creating extended structure using PyMOL")
            if self.chain != "A":
                logging.error("Multiple chains detected. Please use other methods to generate the extended structures or fix this function.")
                exit()
            openawsem.helperFunctions.create_extended_pdb_from_fasta(f"{self.name}.fasta", output_file_name="extended.pdb") #should do f"extended.{extension}"
            input_pdb_filename, cleaned_pdb_filename = openawsem.prepare_pdb("extended.pdb", "A", use_cis_proline=False, keepIds=self.args.keepIds, removeHeterogens=removeHeterogens)
            openawsem.ensure_atom_order(input_pdb_filename)
        
        logging.info(f"Copying crystal_structure-cleaned.{extension} to {self.pdb}")
        print(f"Copying crystal_structure-cleaned.{extension} to {self.pdb}")
        shutil.copy(f'crystal_structure-cleaned.{extension}',f'{self.pdb}')
        
        if self.args.keepLigands:
            if extension == "cif":
                raise NotImplementedError("Still need to implement cif support for extended pdbs")
            # cleaned_pdb_filename = f"{name}-cleaned.pdb"
            # input_pdb_filename = f"{name}-openmmawsem.pdb"
            # do(f"grep 'ATOM' {input_pdb_filename} > tmp.pdb")
            # do(f"grep 'HETATM' {cleaned_pdb_filename} >> tmp.pdb")
            # do(f"mv tmp.pdb {input_pdb_filename}")
            logging.info(f"Copying HETATM records from crystal_structure-cleaned.{extension} to {self.name}-openmmawsem.{extension}")
            shutil.copy(f"crystal_structure-cleaned.{extension}", f"{self.name}-cleaned.{extension}")
            with open("tmp.pdb", "w") as output_file:
                with open("crystal_structure-openmmawsem.pdb", "r") as input_file:
                    for line in input_file:
                        if "ATOM" in line:
                            output_file.write(line)
                with open("crystal_structure-cleaned.pdb", "r") as input_file:
                    for line in input_file:
                        if "HETATM" in line:
                            output_file.write(line)
            os.rename("tmp.pdb", f"{self.name}-openmmawsem.pdb")
        else:
            logging.info(f"Creating openmmawsem.{extension} file")
            input_pdb_filename, cleaned_pdb_filename = openawsem.prepare_pdb(self.pdb, self.chain, keepIds=self.args.keepIds, removeHeterogens=removeHeterogens)
            openawsem.ensure_atom_order(input_pdb_filename)

    def generate_ssweight_from_stride(self,is_cif=False):
        """
        Generate the secondary structure weight file (ssweight) using stride or Predict_Property.
        """
        if is_cif:
            extension = "cif"
        else:
            extension = "pdb"
        logging.info("Generating ssweight from stride")
        logging.info(f"stride crystal_structure.{extension}")
        self.run_command(["stride", f"crystal_structure.{extension}"], stdout="ssweight.stride")
        logging.info(f'python {__location__/"helperFunctions"/"stride2ssweight.py"}')
        self.run_command(["python", __location__/"helperFunctions"/"stride2ssweight.py"], stdout="ssweight")
        protein_length = openawsem.helperFunctions.getFromTerminal("wc ssweight").split()[0]
        if int(protein_length) == 0:
            seq = openawsem.helperFunctions.read_fasta(f"{self.name}.fasta")
            protein_length = len(seq)
            logging.warning("Imposing no secondary bias. You might want to install Predict_Property and use the predict_ssweight_from_fasta option.")
            with open("ssweight", "w") as out:
                for i in range(protein_length):
                    out.write("0.0 0.0\n")

    def generate_ssweight_from_fasta(self):

        # Another option for secondary prediction bias generation is using "Predict_Property.sh -i {name}.fasta" to predict from fasta file.
        # but you need install it from https://github.com/realbigws/Predict_Property.
        self.run_command(["Predict_Property.sh", "-i", f"{self.name}.fasta"])
        
        from_secondary = f"{self.name}_PROP/{self.name}.ss3"
        toPre = "."
        to_ssweight = f"{toPre}/ssweight"
        logging.info("Generating ssweight from fasta")
        data = pd.read_csv(from_secondary, comment="#", names=["i", "Res", "ss3", "Helix", "Sheet", "Coil"], sep="\s+")
        with open(to_ssweight, "w") as out:
            for i, line in data.iterrows():
                if line["ss3"] == "H":
                    out.write("1.0 0.0\n")
                if line["ss3"] == "E":
                    out.write("0.0 1.0\n")
                if line["ss3"] == "C":
                    out.write("0.0 0.0\n")
                        
    def prepare_membrane_files(self):
        """
        Prepare the required membrane-related files (zim and zimPosition) if the membrane or hybrid options are specified.
        """
        logging.info("Preparing membrane files")
        self.run_command(["grep", "-E", "CB|CA  GLY", "crystal_structure-cleaned.pdb"], stdout="cbs.data")
        self.run_command(["awk", "{if($9>15) print \"1\"; else if($9<-15) print \"3\"; else print \"2\"}", "cbs.data"], stdout="zimPosition")

        openawsem.helperFunctions.create_zim(f"crystal_structure.fasta", tableLocation=__location__/"helperFunctions")

    def generate_fragment_memory(self,database = "cullpdb_pc80_res3.0_R1.0_d160504_chains29712",fasta = None,N_mem = 20,brain_damage = 1.0,fragmentLength = 10,cutoff_identical=90,fragmem_structure=None):
        """
        Generate the fragment memory file if the frag option is specified.
        """
        logging.info("Generating fragment memories")
        if fasta is None:
            fasta = f"{self.name}.fasta"

        openawsem.helperFunctions.create_fragment_memories(database=database, fasta_file=fasta, memories_per_position=N_mem, 
                                                           brain_damage=brain_damage, fragment_length=fragmentLength, pdb_dir=openawsem.data_path.pdb, 
                                                           index_dir=openawsem.data_path.index, frag_lib_dir=openawsem.data_path.gro,
                                                           failed_pdb_list_file=openawsem.data_path.pdbfail, pdb_seqres=openawsem.data_path.pdbseqres,
                                                           weight=1, evalue_threshold=10000, cutoff_identical=cutoff_identical)

        # self.run_command([
        #     "python", __location__/"helperFunctions"/"MultCha_prepFrags_index.py",
        #     database, fasta, str(N_mem), str(brain_damage), str(fragmentLength)
        # ], stdout="logfile")

        # Check and correct the fragment memory file
        #openawsem.helperFunctions.check_and_correct_fragment_memory("frags.mem",
        #                fragmem_structure=fragmem_structure,openawsem_location=__location__)

        # Relocate the file to the fraglib folder
        openawsem.helperFunctions.relocate(fileLocation="frags.mem", toLocation="fraglib")

        # Replace the file path in frags.mem
        # openawsem.helperFunctions.replace(f"frags.mem", f"{__location__}//Gros/", "./fraglib/") #original
        openawsem.helperFunctions.replace(f"frags.mem", f"{__location__}/data/Gros/", "./fraglib/") #Rebekah edited 03072024
        self.run_command(["cp", "frags.mem", "frag_memory.mem"])

    def generate_single_memory(self,is_cif=False):
        logging.info("Generating single memory file")
        if is_cif:
            extension = "cif"
        else:
            extension = "pdb"
        io_class = openawsem.get_openmm_io_class(extension) 
        temp = io_class(f"crystal_structure-cleaned.{extension}")
        old_top = temp.getTopology()
        pos = temp.getPositions(asNumpy=True)
        for c in self.chain:
            # print(f"convert chain {c} of crystal structure to Gro file")
            #self.run_command(["python", f"{__location__}/helperFunctions/Pdb2Gro.py", f"crystal_structure-cleaned.{extension}", f"{self.name}_{c}.gro", f"{c}"])
            for file_chain in old_top.chains():
                if c == file_chain.id:
                    new_top = openawsem.Topology() # really an openmm.app.Topology
                    new_chain = new_top.addChain(id=file_chain.id)
                    pos_indices = []
                    for residue in file_chain.residues():
                        # maybe add this check?
                        #if residue.insertionCode != ' ':
                        #    raise ValueError(f"""Chain {file_chain.id} residue id {residue.id} has insertion code {residue.insertionCode}.
                        #                         Non-empty insertion codes are not allowed for single memory structures.
                        #                         Check your crystal_structure-cleaned.{extension} file.""")
                        #new_residue = new_top.addResidue(residue.name, new_chain, id=str(residue.index+1)) #,insertionCode=residue.insertionCode)
                        new_residue = new_top.addResidue(residue.name, new_chain, id=residue.id) #,insertionCode=residue.insertionCode)
                        for atom in residue.atoms():
                            new_top.addAtom(atom.name, atom.element, new_residue, id=atom.id)
                            pos_indices.append(atom.index)
                    chain_pos = pos[pos_indices,:]
                    io_class.writeFile(new_top, chain_pos, f'{self.name}_{c}.{extension}', keepIds=True)
            
        # something like [('A', 10, 115), ('B', 10, 115)] or [('A', 1, 115), ('B', 1, 115)], depending on --resetIds
        top_for_seq_data = io_class(f"crystal_structure-cleaned.{extension}").getTopology()
        seq_data = openawsem.helperFunctions.seq_length_from_pdb(top_for_seq_data, self.chain)
        with open("single_frags.mem", "w") as out:
            out.write("[Target]\nquery\n\n[Memories]\n")
            chain_index_start = 1
            for (chain_name, chain_start_residue_index, seq_length) in seq_data:
                # print(f"write chain {chain_name}")
                out.write(f"{self.name}_{chain_name}.gro {chain_index_start} {chain_start_residue_index} {seq_length} 20\n")   # single_memory is always read starting from 1.
                chain_index_start += seq_length

    def generate_charges(self):
        logging.info("Generating charges")
        openawsem.helperFunctions.generate_charge_array(Path(f"{self.name}.fasta"),Path('charge.txt'))

    def copy_parameters(self, destination_folder='.'):
        # Copy the files using shutil.copy()
        logging.info(f"Copying parameters to {destination_folder}")
        shutil.copy(__location__/"parameters"/"burial_gamma.dat", destination_folder)
        shutil.copy(__location__/"parameters"/"gamma.dat", destination_folder)
        shutil.copy(__location__/"parameters"/"membrane_gamma.dat", destination_folder)
        shutil.copy(__location__/"parameters"/"anti_HB", destination_folder)
        shutil.copy(__location__/"parameters"/"anti_NHB", destination_folder)
        shutil.copy(__location__/"parameters"/"anti_one", destination_folder)
        shutil.copy(__location__/"parameters"/"para_HB", destination_folder)
        shutil.copy(__location__/"parameters"/"para_one", destination_folder)
    
    def copy_scripts(self,destination_folder='.'):
        """
        Copy required scripts and files to the current working directory.
        """
        logging.info(f"Copying scripts to {destination_folder}")
        mm_run_path = __location__ /"scripts"/ "mm_run.py"
        mm_analysis_path = __location__ /"scripts"/ "mm_analyze.py"
        forces_setup_path = __location__ /"scripts"/ "forces_setup.py"

        shutil.copy(mm_run_path, destination_folder)
        shutil.copy(mm_analysis_path, destination_folder)
        shutil.copy(forces_setup_path, destination_folder)


    def run(self):

        """
        Execute the main workflow of the AWSEMSimulationProject class.
        """
        
        #Log the command used to run this module.
        self.log_commandline_args()

        for protein in self.args.proteins:
            self.change_directory(self.base_folder)
            self.args.protein = protein
            if self.args.to:
                project_folder = Path(self.args.to)
            else:
                project_folder = Path(os.path.splitext(os.path.basename(protein))[0])
            project_folder.mkdir(parents=True, exist_ok=True)
            
            # Prepare the input files
            if self.args.protein[-4:] == '.pdb':
                self.name, self.pdb = self.prepare_input_files_from_pdb(project_folder)
                is_cif = False
            elif self.args.protein[-4:] == ".cif":
                self.name, self.pdb = self.prepare_input_files_from_cif(project_folder)
                is_cif = True
            elif self.args.protein[-6:] == ".fasta":
                self.name, self.pdb = self.prepare_input_files_from_fasta(project_folder)
                is_cif = False # should eventually update this to download cif file
            else:
                self.name, self.pdb = self.prepare_input_files_from_name(project_folder)
                is_cif = False # should eventually update this to download cif file
                
            logging.info(f"Protein name: {self.name}, PDB or mmCIF/PDBx file: {self.pdb}")
            
            #Change the directory
            with self.change_directory(project_folder):
            
                # Process the PDB files (clean, extract chains, generate extended structure)
                self.process_pdb_files(is_cif=is_cif)
                
                # Generate the secondary structure weight file (ssweight)
                if self.args.predict_ssweight_from_fasta:
                    self.generate_ssweight_from_fasta()
                else:
                    self.generate_ssweight_from_stride(is_cif=is_cif)
                
                # Prepare the membrane-related files if membrane or hybrid option is enabled
                if self.args.membrane or self.args.hybrid:
                    self.prepare_membrane_files()
                
                # Generate single memory file
                self.generate_single_memory(is_cif=is_cif)

                # Generate fragment memory files if the frag option is enabled
                if self.args.frag:
                    self.generate_fragment_memory(database=self.args.frag_database, fasta=self.args.frag_fasta, 
                        N_mem=self.args.frag_N_mem, brain_damage=self.args.frag_brain_damage, 
                        fragmentLength=self.args.frag_fragmentLength, cutoff_identical=self.args.frag_cutoff_identical, 
                        fragmem_structure=self.args.frag_fragmem_structure)

                #Generate charges
                self.generate_charges()
                
                # Copy required scripts to the current working directory
                self.copy_scripts()

                # Copy required parameters to the current working directory
                self.copy_parameters()

                logging.info(f"{project_folder} project folder created")
                logging.warning("Please modify the forces_setup.py if we want to change what energy terms to be used.")

import unittest
import tempfile
import shutil
from unittest.mock import MagicMock

class TestAWSEMSimulationProject(unittest.TestCase):

    def setUp(self):
        self.args = MagicMock()
        self.args.proteins = ['1r69']
        self.args.protein = '1r69'
        self.args.chain = "-1"
        self.args.debug = False
        self.args.keepLigands = False
        self.args.keepIds = False
        self.args.membrane = False
        self.args.hybrid = False
        self.args.frag = False
        self.args.predict_ssweight_from_fasta = False
        self.args.extended = False
        self.args.verbose = False
        
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_folder = __location__.parent/'tests'/'data'
        self.project = AWSEMSimulationProject(self.args)
        self.project.base_folder = self.temp_dir



    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_prepare_input_files_from_pdb(self):
        pdb_path = Path(self.data_folder) / "1r69.pdb"
        self.args.protein = str(pdb_path)
        name, pdb = self.project.prepare_input_files_from_pdb(self.temp_dir)
        self.assertEqual(name, "1r69")
        self.assertEqual(pdb, "1r69.pdb")

    def test_prepare_input_files_from_fasta(self):
        fasta_path = Path(self.data_folder) / "1r69.fasta"
        self.args.protein = str(fasta_path)
        name, pdb = self.project.prepare_input_files_from_fasta(self.temp_dir)
        self.assertEqual(name, "1r69")
        self.assertEqual(pdb, "1r69.pdb")

    def test_prepare_input_files_from_name(self):
        self.args.protein = "1r69"
        name, pdb = self.project.prepare_input_files_from_name(self.temp_dir)
        self.assertEqual(name, "1r69")
        self.assertEqual(pdb, "1r69.pdb")

    def test_copy_scripts(self):
        self.project.copy_scripts(destination_folder=self.temp_dir)
        copied_files = ["mm_run.py", "mm_analysis.py", "forces_setup.py"]
        for file in copied_files:
            logging.info(self.project.base_folder/file)
            self.assertTrue((self.temp_dir/file).exists())

    def test_run_command(self):
        with self.assertRaises(Exception):
            self.project.run_command(["ls", "non_existent_file"])
    

def main(args=None):
    # Create an argument parser with a description for the script
    parser = argparse.ArgumentParser(
        description=f"This Python 3 script automatically creates a protein simulation project template quickly and efficiently. Written by {__author__}"
    )

    # Define the expected arguments and their help messages
    parser.add_argument("proteins", nargs="*", help="Provide the names of the proteins (e.g., 1r69) or the target PDB files for the simulation, separated by spaces.")
    parser.add_argument("-c", "--chain", default="-1", help="Specify the chains to be simulated (e.g., 'ABC').")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("-f", "--frag", "--fragment", action="store_true", default=False, help="Generate fragment memories.")
    parser.add_argument("-e","--extended", action="store_true", default=False, help="Start from an extended structure generated using PyMOL (ensure it's installed). Supports single chain only.")
    parser.add_argument("-m","--membrane", action="store_true", default=False, help="Enable membrane protein simulations.")
    parser.add_argument("--hybrid", action="store_true", default=False, help="Enable hybrid simulations.")
    parser.add_argument("-v", "--verbose", type=int, default=0, const=1, nargs='?', help="Set verbosity level. Default is 0 (no output). Use --verbose to enable info output, and --verbose 2 for debug output.")
    parser.add_argument("-s","--predict_ssweight_from_fasta", action="store_true", default=False, help="Predict secondary structure weight from FASTA sequence.")
    parser.add_argument("-i","--resetIds", action="store_true", default=False, help="Rewrite chain and residue index. By default, chains will be renamed from 'A' and indices will start from 1.")
    parser.add_argument("--keepLigands", action="store_true", default=False, help="Preserve ligands in the protein structure.")
    parser.add_argument("--to", default=None, help="Folder to create the project in. Default is the name of the protein")
    parser.add_argument("--test", action="store_true", default=False, help="Tests the current module")

      # Create a subparser for frag-related arguments
    frag_parser = parser.add_argument_group("frag", "Arguments for fragment memory generation. Only used if --frag is specified")
    frag_parser.add_argument("--frag_fragmem_structure", default=None, help='known structure to filter out fragmems corresponding to non-native conformations')
    frag_parser.add_argument("--frag_database", default=openawsem.data_path.blast, help="Specify the database for fragment generation.")
    frag_parser.add_argument("--frag_fasta", default=None, help="Provide the FASTA file for fragment generation.")
    frag_parser.add_argument("--frag_N_mem", type=int, default=20, help="Number of memories to generate per fragment.")
    frag_parser.add_argument("--frag_brain_damage", type=float, choices=[0, 0.5, 1, 2], default=0, help="Control the inclusion or exclusion of homologous protein structures for generating fragment memories.\n 0: Homologs allowed; include all hits\n 0.5: Self-only; Include only homologs with >90%% similarity\n 1: Homologs excluded; Exclude all homologs (any similarity percent)\n 2: Homologs only; Include only homologous structures (except >90%% similarity)")
    frag_parser.add_argument("--frag_fragmentLength", type=int, default=10, help="Length of the fragments to be generated.")
    frag_parser.add_argument("--frag_cutoff_identical", type=int, default=90, help="Identity cutoff for self-structures")


    # Parse and return the command-line arguments
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    args.keepIds = not args.resetIds

    # Set the verbosity level for logging
    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s' )
    elif args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
        warnings.simplefilter('ignore', BiopythonWarning)

    logging.info("Logging started")

    logging.debug(f"Arguments parsed: {args}")

    if args.test:
        unittest.main(argv=['first-arg-is-program-name'], exit=False) 
        exit()
    else:
        project = AWSEMSimulationProject(args)
        project.run()

if __name__=="__main__":
    main()
