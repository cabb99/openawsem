#!/bin/bash
source ~/.bashrc
for PDBID in "2ohx_A"; do
    n=374 # 2ohx_A
    #n=145 # 8j47_IGECA 
    #n=120 # 6rb9_AB_barrel # NOT WORKING BECAUSE AWSEM_CREATE INCORRECTLY REMOVES CHAIN B HAD TO MANUALLY COPY LAMMPS_MOVIE.PDB OVER TO 6RB9_AB_BARREL-OPENMMAWSEM.PDB
    cd "make_test_for_${PDBID}"
    conda activate openawsem310                                                                  # openawsem installation required                           
    awsem_create $PDBID.pdb --chain A # needed for 2ohx_A
    for ((i=0; i<$n; i++)); do
        echo "0.0 1.0" >> ssweight
    done
    cp ssweight ../"${PDBID}-ssweight"
    cp $PDBID/crystal_structure.pdb ../$PDBID-crystal_structure.pdb   
    cp $PDBID/crystal_structure.fasta ../$PDBID-crystal_structure.fasta 
    cp $PDBID/$PDBID-openmmawsem.pdb ../$PDBID-openmmawsem.pdb
    conda deactivate
    conda activate lammps_awsem                                                                  # suggested to configure dependencies
    bash ../lammps_setup_scripts/NewPdb2Lammps.sh $PDBID $PDBID                                    # requires python=2.7.15, Bio=1.76
    cp ../lammps_setup_parameters/* .
    sed -i -e 's/peptide/awsemmd/g' $PDBID.in
    sed -i -e 's/run\t\t10000/run\t\t0/g' $PDBID.in                                               # 0 or whatever number of steps you want to run
    ~/lammps24/lammps-29Aug2024/src/lmp_serial < $PDBID.in > lammps_stdout.txt                    # or whatever the path to your lammps binary is 
    python3 ../BuildAllAtomsFromLammps_openAWSEM.py dump.lammpstrj lammps_movie.pdb $PDBID.seq    # requires python=3.6.8 (other versions probably okay)
    vmd lammps_movie.pdb -e ../convert_pdb_to_dcd.tcl
    cp lammps_movie.dcd ../$PDBID-movie.dcd
    #TAKE THE INFORMATION FROM energy.log AND PUT IT IN ../PDBID_energies.csv, FOLLOWING THE FORMAT OF OTHER PDBID_energies.csv FILES
    #DON'T FORGET TO ADD PDBID TO THE PROTEINS LIST IN THE TEST SCRIPT, test_implementation_of_lammps_hbond_energies.py
    cd ..
done
