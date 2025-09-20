#!/usr/bin/env python3
import os
import sys
import time
import argparse
import importlib.util
import subprocess

from openawsem import *
from openawsem.helperFunctions.myFunctions import *

do = os.system
cd = os.chdir

def run(args):
    simulation_platform = args.platform
    platform = Platform.getPlatformByName(simulation_platform)
    if simulation_platform == "CPU":
        if args.thread != -1:
            platform.setPropertyDefaultValue("Threads", str(args.thread))
        print(f"{simulation_platform}: {platform.getPropertyDefaultValue('Threads')} threads")
    elif simulation_platform=="OpenCL":
        platform.setPropertyDefaultValue('OpenCLPlatformIndex', '0')
        platform.setPropertyDefaultValue('DeviceIndex', str(args.device))
    elif simulation_platform=="CUDA":
        platform.setPropertyDefaultValue('DeviceIndex', str(args.device))

    # if mm_run.py is not at the same location of your setup folder.
    setupFolderPath = os.path.dirname(args.protein)
    setupFolderPath = "." if setupFolderPath == "" else setupFolderPath
    proteinName = pdb_id = os.path.basename(args.protein)

    pwd = os.getcwd()
    toPath = os.path.abspath(args.to)
    checkPointPath = None if args.fromCheckPoint is None else os.path.abspath(args.fromCheckPoint)
    forceSetupFile = None if args.forces is None else os.path.abspath(args.forces)
    parametersLocation = "." if args.parameters is None else os.path.abspath(args.parameters)
    os.chdir(setupFolderPath)



    # chain=args.chain
    chain=args.chain
    pdb = f"{pdb_id}.pdb"

    if chain == "-1":
        chain = getAllChains("crystal_structure.pdb")
        print("Chains to simulate: ", chain)


    if args.to != "./":
        # os.system(f"mkdir -p {args.to}")
        os.makedirs(toPath, exist_ok=True)
        os.system(f"cp {forceSetupFile} {toPath}/forces_setup.py")
        os.system(f"cp crystal_structure.fasta {toPath}/")
        os.system(f"cp crystal_structure.pdb {toPath}/")
        # os.system(f"cp {pdb} {args.to}/{pdb}")
        # pdb = os.path.join(args.to, pdb)

    if args.fromOpenMMPDB:
        input_pdb_filename = proteinName
        seq=read_fasta("crystal_structure.fasta")
        print(f"Using Seq:\n{seq}")
    else:
        suffix = '-openmmawsem.pdb'
        if pdb_id[-len(suffix):] == suffix:
            input_pdb_filename = pdb_id
        else:
            input_pdb_filename = f"{pdb_id}-openmmawsem.pdb"
        seq=None

    if args.fasta == "":
        seq = None
    else:
        seq = seq=read_fasta(args.fasta)
        print(f"Using Seq:\n{seq}")
    # start simulation
    collision_rate = 5.0 / picoseconds

    # check for atoms whose positions are intended to be fixed
    if args.fixed_residue_indices:
        with open(args.fixed_residue_indices,'r') as f:
            for line in f: # only expect 1 line
                fixed_residue_indices = line.strip().split(',') #expecting a one-line csv
                fixed_residue_indices = [int(item) for item in fixed_residue_indices]
                break
    else:
        fixed_residue_indices = []

    # assign annealing parameters
    Tstart = args.tempStart
    Tend = args.tempEnd

    print(f"using force setup file from {forceSetupFile}")
    spec = importlib.util.spec_from_file_location("forces", forceSetupFile)
    # print(spec)
    forces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forces)


    oa = OpenMMAWSEMSystem(input_pdb_filename, k_awsem=1.0, chains=chain, xml_filename=openawsem.xml, seqFromPdb=seq, 
                           includeLigands=args.includeLigands, periodic_box=args.periodic_box,
                           fixed_residue_indices=fixed_residue_indices)  # k_awsem is an overall scaling factor that will affect the relevant temperature scales
    myForces = forces.set_up_forces(oa, submode=args.subMode, contactParameterLocation=parametersLocation,)
    # print(forces)
    # oa.addForces(myForces)

    if args.removeCMMotionRemover:
        oa.system.removeForce(0)
    oa.addForcesWithDefaultForceGroup(myForces)

    if args.fromCheckPoint:
        integrator = LangevinIntegrator(Tstart*kelvin, 1/picosecond, args.timeStep*femtoseconds)
        simulation = Simulation(oa.pdb.topology, oa.system, integrator, platform)
        simulation.loadCheckpoint(checkPointPath)
        reporter_append = True
    else:
        # output the native and the structure after minimization
        integrator = CustomIntegrator(0.001)
        simulation = Simulation(oa.pdb.topology, oa.system, integrator, platform)
        simulation.context.setPositions(oa.pdb.positions)  # set the initial positions of the atoms
        simulation.reporters.append(PDBReporter(os.path.join(toPath, "native.pdb"), 1))
        simulation.reporters.append(DCDReporter(os.path.join(toPath, "movie.dcd"), 1))
        simulation.step(int(1))
        simulation.minimizeEnergy()  # first, minimize the energy to a local minimum to reduce any large forces that might be present
        simulation.step(int(1))
        reporter_append = False


        # print("------------------Folding-------------------")
        # oa = OpenMMAWSEMSystem(input_pdb_filename, k_awsem=1.0, chains=chain, xml_filename=OPENAWSEM_LOCATION+"awsem.xml")  # k_awsem is an overall scaling factor that will affect the relevant temperature scales
        # myForces = forces.set_up_forces(oa, submode=args.subMode, contactParameterLocation=parametersLocation)
        # oa.addForces(myForces)

        integrator = LangevinIntegrator(Tstart*kelvin, 1/picosecond, args.timeStep*femtoseconds)
        # integrator.setRandomNumberSeed(A_NUMBER_AS_RANDOM_SEED)
        # integrator = CustomIntegrator(0.001)
        simulation = Simulation(oa.pdb.topology, oa.system, integrator, platform)
        # simulation.loadState(os.path.join(toPath, 'output.xml'))
        simulation.context.setPositions(oa.pdb.positions)  # set the initial positions of the atoms
        simulation.context.setVelocitiesToTemperature(Tstart*kelvin)  # set the initial velocities of the atoms according to the desired starting temperature
        # simulation.context.setVelocitiesToTemperature(Tstart*kelvin, A_RANDOM_SEED_NUMBER)
        simulation.minimizeEnergy()  # first, minimize the energy to a local minimum to reduce any large forces that might be present


    print("report_interval", args.reportInterval)
    print("num_frames", args.numFrames)
    if reporter_append:
        if not os.path.exists(os.path.join(toPath, "output.log")):
            raise AssertionError(f"Could not find file {os.path.join(toPath, "output.log")} to append new info to. This is probably a bug in this script")
        else:
            counter = 1
            while True:
                if not os.path.exists(os.path.join(toPath, f"output_{counter}.log")):
                    print(f"making backup log file output_{counter}.log")
                    subprocess.run(['cp', os.path.join(toPath, "output.log"), os.path.join(toPath, f"output_{counter}.log")
                    break
                else:
                    counter += 1                
    simulation.reporters.append(StateDataReporter(sys.stdout, args.reportInterval, step=True, potentialEnergy=True, temperature=True, append=reporter_append))  # output energy and temperature during simulation
    simulation.reporters.append(StateDataReporter(os.path.join(toPath, "output.log"), args.reportInterval, step=True, potentialEnergy=True, temperature=True, append=reporter_append)) # output energy and temperature to a file
    simulation.reporters.append(PDBReporter(os.path.join(toPath, "movie.pdb"), reportInterval=args.reportInterval))  # output PDBs of simulated structures; appending not supported by openmm
    simulation.reporters.append(DCDReporter(os.path.join(toPath, "movie.dcd"), reportInterval=args.reportInterval, append=True))  # output PDBs of simulated structures
    # simulation.reporters.append(DCDReporter(os.path.join(args.to, "movie.dcd"), 1))  # output PDBs of simulated structures
    # simulation.reporters.append(PDBReporter(os.path.join(args.to, "movie.pdb"), 1))  # output PDBs of simulated structures
    simulation.reporters.append(CheckpointReporter(os.path.join(toPath, args.checkpointFile), args.checkpointInterval))  # save progress during the simulation

    if args.dryRun:
        if args.simulation_mode == 1: # test temperature setting
            deltaT = (Tend - Tstart) / args.numFrames
            for i in range(args.numFrames):
                integrator.setTemperature((Tstart + deltaT*i)*kelvin)
        raise SystemExit("Simulation configured successfully")

    print("Simulation Starts")
    start_time = time.time()

    if args.simulation_mode == 0:
        simulation.step(int(args.steps))
    elif args.simulation_mode == 1:
        deltaT = (Tend - Tstart) / args.numFrames
        for i in range(args.numFrames):
            integrator.setTemperature((Tstart + deltaT*i)*kelvin)
            simulation.step(args.reportInterval) 

            # simulation.saveCheckpoint('step_%d.chk' % i)
            # simulation.context.setParameter("k_membrane", 0)
            # if i < snapShotCount/2:
            #     simulation.context.setParameter("k_membrane", (i % 2) * k_mem)
            #     simulation.context.setParameter("k_single_helix_orientation_bias", (i % 2) * k_single_helix_orientation_bias)
            # else:
            #     simulation.context.setParameter("k_membrane", k_mem)
            #     simulation.context.setParameter("k_single_helix_orientation_bias", k_single_helix_orientation_bias)

            # simulation.context.setParameter("k_membrane", (i)*(k_mem/snapShotCount))
            # simulation.context.setParameter("k_single_helix_orientation_bias", (i)*(k_single_helix_orientation_bias/snapShotCount))
            # print(simulation.context.getParameter("k_membrane"))


    # simulation.step(int(1e6))

    time_taken = time.time() - start_time  # time_taken is in seconds
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)
    print(f"---{hours} hours {minutes} minutes {seconds} seconds ---")

    timeFile = os.path.join(toPath, "time.dat")
    with open(timeFile, "w") as out:
        out.write(str(time_taken)+"\n")

    # accompany with analysis run
    simulation = None
    time.sleep(10)
    os.chdir(pwd)
    print(os.getcwd())
    if args.fasta == "":
        analysis_fasta = ""
    else:
        analysis_fasta = f"--fasta {args.fasta}"
    additional_cmd = ""
    if args.includeLigands:
        additional_cmd += "--includeLigands "
    if args.periodic_box:
        additional_cmd += f"--periodic_box {' '.join(map(str, args.periodic_box))} "
    if args.fixed_residue_indices:
        additional_cmd += f"--fixed_residue_indices {fixed_residue_indices} "
    if args.fromOpenMMPDB:
        additional_cmd += f"--fromOpenMMPDB "
    os.system(f"{sys.executable} mm_analyze.py {args.protein} -t {os.path.join(toPath, 'movie.dcd')} --subMode {args.subMode} -f {args.forces} {analysis_fasta} {additional_cmd} -c {chain}")




def main(args=None):
    parser = argparse.ArgumentParser(
        description="This is a python3 script to automatically copy the template file and run simulations")
    # default=False with action="store_true" is redundant but doesn't hurt us
    parser.add_argument("protein", help="The name of the protein")
    parser.add_argument("--name", default="simulation", help="Name of the simulation")
    parser.add_argument("--to", default="./", help="location of movie file")
    parser.add_argument("-c", "--chain", type=str, default="-1")
    parser.add_argument("-t", "--thread", type=int, default=-1, help="default is using all that is available")
    parser.add_argument("-p", "--platform", type=str, default="OpenCL", choices=["OpenCL", "CPU", "HIP", "Reference", "CUDA"], help="Platform to run the simulation.")
    parser.add_argument("-s", "--steps", type=float, default=1e7, help="step size, default 1e7")
    parser.add_argument("--tempStart", type=float, default=800, help="Starting temperature")
    parser.add_argument("--tempEnd", type=float, default=200, help="Ending temperature")
    parser.add_argument("--fromCheckPoint", type=str, default=None, help="The checkpoint file you want to start from")
    parser.add_argument("-m", "--simulation_mode", type=int, default=1,
                    help="default 1,\
                            0: constant temperature,\
                            1: temperature annealing")
    parser.add_argument("--subMode", type=int, default=-1)
    parser.add_argument("-f", "--forces", default="forces_setup.py")
    parser.add_argument("--parameters", default=None)
    parser.add_argument("-r", "--reportInterval", "--reportFrequency", type=float, default=None, help="Number of steps between each frame recorded")
    parser.add_argument("--checkpointInterval", type=float, default=None, help="Number of steps between each frame recorded")
    parser.add_argument("--checkpointFile", type=str, default="checkpoint.chk", help="Name of the checkpoint file")
    parser.add_argument("--numFrames", type=int, default=400, help="Number of frames to record. Can be overridden by --reportInterval")
    parser.add_argument("--fromOpenMMPDB", action="store_true", default=False)
    parser.add_argument("--fasta", type=str, default="crystal_structure.fasta")
    parser.add_argument("--timeStep", type=int, default=2, help="time step in femtoseconds")
    parser.add_argument("--includeLigands", action="store_true", default=False)
    parser.add_argument('--device', default=0, help='OpenCL/CUDA device index')
    parser.add_argument('--removeCMMotionRemover', action="store_true", default=False, help='Removes CMMotionRemover. Recommended for periodic boundary conditions and membrane simulations')
    parser.add_argument('--fixed_residue_indices', type=str, default='', help='csv file with indices (not "ids" or "resnums") of residues whose positions should be fixed)')
    parser.add_argument('--periodic_box', type=float, nargs=3, metavar=('X', 'Y', 'Z'), help='Enable periodic boundary conditions with box dimensions in x, y, z (nanometers)')
    parser.add_argument('--dryRun',action="store_true",default=False,help="Return the configuration and exit without running the simulation")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    with open('commandline_args.txt', 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    print(' '.join(sys.argv))

    # Correct number of timesteps if negative
    if args.steps <= 0:
        logging.warning("--steps must be a positive integer. Reverting to default 1e7")
        args.steps = 1e7
    else:
        # Convert steps to integer
        args.steps = int(args.steps)

    # Adds a deprecation warning if the deprecated option '--reportFrequency' is used
    if '--reportFrequency' in sys.argv:
        logging.warning("The '--reportFrequency' option is deprecated. Please use '--reportInterval' instead.", DeprecationWarning)

    if args.reportInterval is None and args.numFrames>0: 
        #Report interval is not specified and number of frames is reasonable
        if args.numFrames > args.steps:
            logging.warning("Number of frames --numFrames is more than number of steps. Setting number of frames to {args.steps}.")
            args.numFrames = args.steps

        args.reportInterval = math.ceil(args.steps / args.numFrames)
        new_steps = int(args.reportInterval * args.numFrames)
        if new_steps != args.steps:
            logging.warning(f"Number of frames --reportInterval does not divide number of steps --steps exactly. Adjusting number of steps from {args.steps} to {new_steps}")
        args.steps = new_steps
    elif args.reportInterval is not None and args.reportInterval > 0: 
        #Report interval is specified and is reasonable
        if '--numFrames' in sys.argv:
            logging.warning("Ignoring user-specified --numFrames. --reportInterval takes priority over --numFrames. ")
        args.numFrames = math.ceil(args.steps / args.reportInterval)
        new_steps = int(args.numFrames * args.reportInterval)
        if new_steps != args.steps:
            logging.warning(f"Number of frames --numFrames does not divide number of steps --steps exactly. Adjusting number of steps from {args.steps} to {new_steps}")
        args.steps = new_steps
    elif args.numFrames == 0 and args.reportInterval is None:
        logging.info("No frames will be recorded. Simulation will run for the specified number of steps.")
    else:
        logging.error("Invalid values: Either --reportInterval or --numFrames must be provided with positive values.")
        raise ValueError("Both --reportInterval and --numFrames cannot be missing or zero. Please provide valid inputs.")

    args.reportInterval = int(args.reportInterval)
    args.checkpointInterval = args.reportInterval if args.checkpointInterval is None else int(args.checkpointInterval)

    if args.dryRun:
        print("Dry run mode. Simulation will not run.")
        print(args)
        return args
    
    run(args)

if __name__=="__main__":
    main()
