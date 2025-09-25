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

one_to_three={'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE',
            'G':'GLY', 'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU',
            'M':'MET', 'N':'ASN', 'P':'PRO', 'Q':'GLN', 'R':'ARG',
            'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'}

three_to_one = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
                'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
                'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
                'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}


def con_term(oa, k_con=50208, bond_lengths=[.3816, .240, .276, .153], forceGroup=20):
    # add con forces
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_con *= oa.k_awsem
    con = HarmonicBondForce()

    if oa.periodic_box:
        con.setUsesPeriodicBoundaryConditions(True)
        print('\ncon_term is periodic')
    
    for i in range(oa.nres):
        if i in oa.fixed_residue_indices:
            continue
        con.addBond(oa.ca[i], oa.o[i], bond_lengths[1], k_con)
        if not oa.res_type[i] == "IGL":  # OpenAWSEM system doesn't have CB for glycine, so the following bond is not exist for Glycine, but LAMMPS include this bond by using virtual HB as CB.
            con.addBond(oa.ca[i], oa.cb[i], bond_lengths[3], k_con)
        if i not in oa.chain_ends:
            con.addBond(oa.ca[i], oa.ca[i+1], bond_lengths[0], k_con)
            con.addBond(oa.o[i], oa.ca[i+1], bond_lengths[2], k_con)
    con.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return con


def chain_term(oa, k_chain=50208, bond_k=[1, 1, 1], bond_lengths=[0.2459108, 0.2519591, 0.2466597], forceGroup=20):
    # add chain forces
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_chain *= oa.k_awsem
    chain = HarmonicBondForce()
    
    if oa.periodic_box:
        chain.setUsesPeriodicBoundaryConditions(True)
        print('\nchain_term is periodic')

    for i in range(oa.nres):
        if i in oa.fixed_residue_indices:
            continue
        if i not in oa.chain_starts and not oa.res_type[i] == "IGL":
            chain.addBond(oa.n[i], oa.cb[i], bond_lengths[0], k_chain*bond_k[0])
        if i not in oa.chain_ends and not oa.res_type[i] == "IGL":
            chain.addBond(oa.c[i], oa.cb[i], bond_lengths[1], k_chain*bond_k[1])
        if i not in oa.chain_starts and i not in oa.chain_ends:
            chain.addBond(oa.n[i], oa.c[i], bond_lengths[2], k_chain*bond_k[2])
    chain.setForceGroup(forceGroup)
    return chain

def chi_term(oa, k_chi=251.04, chi0=-0.71, forceGroup=20):
    # add chi forces
    # The sign of the equilibrium value is opposite and magnitude differs slightly
    # 251.04 = 60 * 4.184 kJ, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_chi *= oa.k_awsem
    chi = CustomCompoundBondForce(4, f"{k_chi}*(chi*norm-{chi0})^2;"
                                        "chi=crossproduct_x*r_cacb_x+crossproduct_y*r_cacb_y+crossproduct_z*r_cacb_z;"
                                        "crossproduct_x=(u_y*v_z-u_z*v_y);"
                                        "crossproduct_y=(u_z*v_x-u_x*v_z);"
                                        "crossproduct_z=(u_x*v_y-u_y*v_x);"
                                        "norm=1/((u_x*u_x+u_y*u_y+u_z*u_z)*(v_x*v_x+v_y*v_y+v_z*v_z)*(r_cacb_x*r_cacb_x+r_cacb_y*r_cacb_y+r_cacb_z*r_cacb_z))^0.5;"
                                        "r_cacb_x=x1-x4;"
                                        "r_cacb_y=y1-y4;"
                                        "r_cacb_z=z1-z4;"
                                        "u_x=x1-x2; u_y=y1-y2; u_z=z1-z2;"
                                        "v_x=x3-x1; v_y=y3-y1; v_z=z3-z1;")

    if oa.periodic_box:
        chi.setUsesPeriodicBoundaryConditions(True)
        print('\nchi_term is periodic')

    for i in range(oa.nres):
        if i in oa.fixed_residue_indices:
            continue
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL":
            chi.addBond([oa.ca[i], oa.c[i], oa.n[i], oa.cb[i]])
    chi.setForceGroup(forceGroup)
    return chi

def excl_term(oa, k_excl=8368, r_excl=0.35, excludeCB=False, forceGroup=20):
    # add excluded volume
    # Still need to add element specific parameters
    # 8368 = 20 * 4.184 * 100 kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    # Openawsem doesn't have the distance range (r_excl) change from 0.35 to 0.45 when the sequence separtation more than 5
    k_excl *= oa.k_awsem
    excl = CustomNonbondedForce(f"{k_excl}*step({r_excl}-r)*(r-{r_excl})^2")

    if oa.periodic_box:
        excl.setNonbondedMethod(excl.CutoffPeriodic)
        print('\nexcl_term is periodic')
    else:
        excl.setNonbondedMethod(excl.CutoffNonPeriodic)

    pos = oa.pdb.positions
    for i in range(oa.natoms):
        excl.addParticle()
    excl.addInteractionGroup(oa.ca, oa.ca)
    if not excludeCB:
        excl.addInteractionGroup([x for x in oa.cb if x > 0], [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.ca, [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.o, oa.o)

    excl.setCutoffDistance(r_excl)
    excl.createExclusionsFromBonds(oa.bonds, 1)
    excl.setForceGroup(forceGroup)
    return excl


def excl_term_v2(oa, k_excl=8368, r_excl=0.35, periodic=False, excludeCB=False, forceGroup=20):
    # this version remove the use of "createExclusionsFromBonds", which could potentially conflict with other CustomNonbondedForce and causing "All forces must have the same exclusion".
    # add excluded volume
    # Still need to add element specific parameters
    # 8368 = 20 * 4.184 * 100 kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    # Openawsem doesn't have the distance range (r_excl) change from 0.35 to 0.45 when the sequence separtation more than 5
    k_excl *= oa.k_awsem
    excl = CustomNonbondedForce(f"{k_excl}*step(abs(res1-res2)-2+isChainEdge1*isChainEdge2+isnot_Ca1+isnot_Ca2)*step({r_excl}-r)*(r-{r_excl})^2")
    
    if oa.periodic_box:
        excl.setNonbondedMethod(excl.CutoffPeriodic)
        print("\nexcel_term is periodic")
    else:
        excl.setNonbondedMethod(excl.CutoffNonPeriodic)

    excl.addPerParticleParameter("res")
    excl.addPerParticleParameter("isChainEdge")
    excl.addPerParticleParameter("isnot_Ca")
    for i in range(oa.natoms):
        # print(oa.resi[i])
        if (i in oa.chain_ends) or (i in oa.chain_starts):
            # print(i)
            isChainEdge = 1
        else:
            isChainEdge = 0
        if (i in oa.ca):
            isnot_Ca = 0
        else:
            isnot_Ca = 1
        excl.addParticle([oa.resi[i], isChainEdge, isnot_Ca])
    # print(oa.ca)
    # print(oa.bonds)
    # print(oa.cb)
    excl.addInteractionGroup(oa.ca, oa.ca)
    if not excludeCB:
        excl.addInteractionGroup([x for x in oa.cb if x > 0], [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.ca, [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.o, oa.o)

    excl.setCutoffDistance(r_excl)

    # excl.setNonbondedMethod(excl.CutoffNonPeriodic)
    # print(oa.bonds)
    # print(len(oa.bonds))
    # excl.createExclusionsFromBonds(oa.bonds, 1)
    excl.setForceGroup(forceGroup)
    return excl

def rama_term(oa, k_rama=8.368, num_rama_wells=3, w=[1.3149, 1.32016, 1.0264], sigma=[15.398, 49.0521, 49.0954], omega_phi=[0.15, 0.25, 0.65], phi_i=[-1.74, -1.265, 1.041], omega_psi=[0.65, 0.45, 0.25], psi_i=[2.138, -0.318, 0.78], forceGroup=21):
    # add Rama potential
    # 8.368 = 2 * 4.184 kJ/mol, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_rama *= oa.k_awsem
    rama_function = ''.join(["w%d*exp(-sigma%d*(omega_phi%d*phi_term%d^2+omega_psi%d*psi_term%d^2))+" \
                            % (i, i, i, i, i, i) for i in range(num_rama_wells)])[:-1]
    rama_function = '-k_rama*(' + rama_function + ");"
    rama_parameters = ''.join([f"phi_term{i}=cos(phi_{i}-phi0{i})-1; phi_{i}=dihedral(p1, p2, p3, p4);\
                            psi_term{i}=cos(psi_{i}-psi0{i})-1; psi_{i}=dihedral(p2, p3, p4, p5);"\
                            for i in range(num_rama_wells)])
    rama_string = rama_function+rama_parameters
    rama = CustomCompoundBondForce(5, rama_string)

    if oa.periodic_box:
        rama.setUsesPeriodicBoundaryConditions(True)
        print('\nrama_term is periodic')

    for i in range(num_rama_wells):
        rama.addGlobalParameter(f"k_rama", k_rama)
        rama.addGlobalParameter(f"w{i}", w[i])
        rama.addGlobalParameter(f"sigma{i}", sigma[i])
        rama.addGlobalParameter(f"omega_phi{i}", omega_phi[i])
        rama.addGlobalParameter(f"omega_psi{i}", omega_psi[i])
        rama.addGlobalParameter(f"phi0{i}", phi_i[i])
        rama.addGlobalParameter(f"psi0{i}", psi_i[i])
    for i in range(oa.nres):
        if i in oa.fixed_residue_indices and i-1 in oa.fixed_residue_indices and i+1 in oa.fixed_residue_indices:
            continue
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type[i] == "IPR":
            rama.addBond([oa.c[i-1], oa.n[i], oa.ca[i], oa.c[i], oa.n[i+1]])
    rama.setForceGroup(forceGroup)
    return rama

def rama_proline_term(oa, k_rama_proline=8.368, num_rama_proline_wells=2, w=[2.17, 2.15], sigma=[105.52, 109.09], omega_phi=[1.0, 1.0], phi_i=[-1.153, -0.95], omega_psi=[0.15, 0.15], psi_i=[2.4, -0.218], forceGroup=21):
    # add Rama potential for prolines
    # 8.368 = 2 * 4.184 kJ/mol, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_rama_proline *= oa.k_awsem
    rama_function = ''.join(["w_P%d*exp(-sigma_P%d*(omega_phi_P%d*phi_term%d^2+omega_psi_P%d*psi_term%d^2))+" \
                            % (i, i, i, i, i, i) for i in range(num_rama_proline_wells)])[:-1]
    rama_function = '-k_rama_proline*(' + rama_function + ");"
    rama_parameters = ''.join([f"phi_term{i}=cos(phi_{i}-phi0_P{i})-1; phi_{i}=dihedral(p1, p2, p3, p4);\
                            psi_term{i}=cos(psi_{i}-psi0_P{i})-1; psi_{i}=dihedral(p2, p3, p4, p5);"\
                            for i in range(num_rama_proline_wells)])
    rama_string = rama_function+rama_parameters
    rama = CustomCompoundBondForce(5, rama_string)

    if oa.periodic_box:
        rama.setUsesPeriodicBoundaryConditions(True)
        print('\nrama_proline_term is periodic')

    for i in range(num_rama_proline_wells):
        rama.addGlobalParameter(f"k_rama_proline", k_rama_proline)
        rama.addGlobalParameter(f"w_P{i}", w[i])
        rama.addGlobalParameter(f"sigma_P{i}", sigma[i])
        rama.addGlobalParameter(f"omega_phi_P{i}", omega_phi[i])
        rama.addGlobalParameter(f"omega_psi_P{i}", omega_psi[i])
        rama.addGlobalParameter(f"phi0_P{i}", phi_i[i])
        rama.addGlobalParameter(f"psi0_P{i}", psi_i[i])
    for i in range(oa.nres):
        if i in oa.fixed_residue_indices and i-1 in oa.fixed_residue_indices and i+1 in oa.fixed_residue_indices:
            continue
        if i not in oa.chain_starts and i not in oa.chain_ends and oa.res_type[i] == "IPR":
            rama.addBond([oa.c[i-1], oa.n[i], oa.ca[i], oa.c[i], oa.n[i+1]])
    rama.setForceGroup(forceGroup)
    return rama

def rama_ssweight_term(oa, k_rama_ssweight=8.368, num_rama_wells=2, w=[2.0, 2.0],
                    sigma=[419.0, 15.398], omega_phi=[1.0, 1.0], phi_i=[-0.995, -2.25],
                    omega_psi=[1.0, 1.0], psi_i=[-0.82, 2.16], ssweight_file="ssweight", forceGroup=21):
    # add RamaSS potential
    # 8.368 = 2 * 4.184 kJ/mol, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_rama_ssweight *= oa.k_awsem
    rama_function = ''.join(["wSS%d*ssweight(%d,resId)*exp(-sigmaSS%d*(omega_phiSS%d*phi_term%d^2+omega_psiSS%d*psi_term%d^2))+" \
                            % (i, i, i, i, i, i, i) for i in range(num_rama_wells)])[:-1]
    rama_function = f'-{k_rama_ssweight}*(' + rama_function + ");"
    rama_parameters = ''.join([f"phi_term{i}=cos(phi_{i}-phi0SS{i})-1; phi_{i}=dihedral(p1, p2, p3, p4);\
                            psi_term{i}=cos(psi_{i}-psi0SS{i})-1; psi_{i}=dihedral(p2, p3, p4, p5);"\
                            for i in range(num_rama_wells)])
    rama_string = rama_function+rama_parameters
    ramaSS = CustomCompoundBondForce(5, rama_string)
    
    if oa.periodic_box:
        ramaSS.setUsesPeriodicBoundaryConditions(True)
        print('\nrama_ssweight_term is periodic')

    ramaSS.addPerBondParameter("resId")    
    for i in range(num_rama_wells):
        ramaSS.addGlobalParameter(f"wSS{i}", w[i])
        ramaSS.addGlobalParameter(f"sigmaSS{i}", sigma[i])
        ramaSS.addGlobalParameter(f"omega_phiSS{i}", omega_phi[i])
        ramaSS.addGlobalParameter(f"omega_psiSS{i}", omega_psi[i])
        ramaSS.addGlobalParameter(f"phi0SS{i}", phi_i[i])
        ramaSS.addGlobalParameter(f"psi0SS{i}", psi_i[i])
    for i in range(oa.nres):
        if i in oa.fixed_residue_indices and i-1 in oa.fixed_residue_indices and i+1 in oa.fixed_residue_indices:
            continue
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
            ramaSS.addBond([oa.c[i-1], oa.n[i], oa.ca[i], oa.c[i], oa.n[i+1]], [i])
    ssweight = np.loadtxt(ssweight_file)
    ramaSS.addTabulatedFunction("ssweight", Discrete2DFunction(2, oa.nres, ssweight.flatten()))
    ramaSS.setForceGroup(forceGroup)
    return ramaSS

def rama_AM_term_oldparams(oa, memory_angles, memory_weights=None, k_rama=8.368, forceGroup=21):
    fudge_factor = 1/120
    if memory_weights is not None:
        memory_weights *= fudge_factor

    # associative memory ramachandran potential
    # memory_angles: np.ndarray with shape (oa.nres-2, 2, number memories)
    #     where indices [:,0,:] describe the phi values of each memory
    #       and indices [:,1,:] describe the psi values of each memory 

    #TODO: this method is still under construction
    if oa.fixed_residue_indices:
        raise NotImplementedError("rama_AM_term is not yet supported for systems where certain absolute residue positions are fixed in space")
    if len(oa.chain_starts) > 1:
        raise NotImplementedError("rama_AM_term is not yet supported for systems with more than one chain")

    # check input
    if np.max(memory_angles) > np.pi:
        raise ValueError(f"memory_angles should be in radians over the interval (-pi,pi], but found {np.max(memory_angles)}")
    if np.min(memory_angles) < -np.pi:
        raise ValueError(f"memory_angles should be in radians over the interval (-pi,pi], but found {np.min(memory_angles)}")
    if not memory_angles.shape[0] == oa.nres - 2:
        raise ValueError(f"The size of memory_angles along axis 0 should be {oa.nres-2} (oa.nres-2) but was {memory_angles.shape[0]}.\
            We expect oa.nres-2 residues because every residue except the first and last in the chain should have memories")
    # should add check on the shapes of memory_weights, sigma, omega_phi, and omega_psi if we eventually make those 
    #     independent function parameters instead of hard-coding a particular choice of these values (see below)

    # load/calculate basic parameters
    num_residues = memory_angles.shape[0] # assign to descriptively named variable to clarify meaning of this axis
    num_memories = memory_angles.shape[2] # assign to descriptively named variable to clarify meaning of this axis
    if memory_weights is None:
        memory_weights = np.ones((num_residues, num_memories))
    else:
        if not memory_weights.shape == ((num_residues, num_memories)):
            raise ValueError(f"memory_weights should have shape (num_residues,num_memories)=={(num_residues, num_memories)}, \
                               but was {memory_weights.shape}")
    sigma = np.ones((num_residues, num_memories))*100
    omega_phi = np.ones((num_residues, num_memories))
    omega_psi = np.ones((num_residues, num_memories))
    k_rama *= oa.k_awsem

    """
    # set up strings representing memory terms
    phi_term = []
    psi_term = []
    for i in range(num_residues):
        phi_row = []
        psi_row = []
        for k in range(num_memories):
            phi_row.append(f"(cos(dihedral(p1,p2,p3,p4)-{memory_angles[i,0,k]})-1)")
            psi_row.append(f"(cos(dihedral(p2,p3,p4,p5)-{memory_angles[i,1,k]})-1)")
        phi_term.append(phi_row)
        psi_term.append(psi_row)

    # set up the complete rama function
    rama_function = ""
    for i in range(num_residues):
        for k in range(num_memories):
            rama_function += f"{memory_weights[i,k]}*exp(-{sigma[i,k]}*({omega_phi[i,k]}*{phi_term[i][k]}^2+{omega_psi[i,k]}*{psi_term[i][k]}^2))+"
    rama_function = f'-{k_rama}*(' + rama_function[:-1] + ")" # cut off trailing "+" from rama_function
    """
    # the other rama functions set w to 2 by default, but we expect it here to be normalized to 1,
    # so we multiply w by 2 in the energy string
    rama_string_base = "".join([f"2*w(res_index,{mem_index})*exp(-(sigma(res_index,{mem_index})*\
        (omega_phi(res_index,{mem_index})*(cos(dihedral(p1,p2,p3,p4)-phi_memories(res_index,{mem_index}))-1)^2\
         +omega_psi(res_index,{mem_index})*(cos(dihedral(p2,p3,p4,p5)-psi_memories(res_index,{mem_index}))-1)^2)))+" for mem_index in range(num_memories)])
    rama_string_full = f"-2*{k_rama}*({rama_string_base[:-1]})" # removes trailing + from energy expression
                           # multiplying by 2 because we normally stack the base and bias on top of each other, 
                           # but this Force is intended to be used alone
    ramaAM = CustomCompoundBondForce(5, rama_string_full)
    ramaAM.addPerBondParameter("res_index") # will be used to look up the row holding all of our memories for each residue
    ramaAM.addTabulatedFunction("w", Discrete2DFunction(num_residues, num_memories, memory_weights.T.flatten()))
    ramaAM.addTabulatedFunction("sigma", Discrete2DFunction(num_residues, num_memories, sigma.T.flatten()))
    ramaAM.addTabulatedFunction("omega_phi", Discrete2DFunction(num_residues, num_memories, omega_phi.T.flatten()))
    ramaAM.addTabulatedFunction("omega_psi", Discrete2DFunction(num_residues, num_memories, omega_psi.T.flatten()))
    ramaAM.addTabulatedFunction("phi_memories", Discrete2DFunction(num_residues, num_memories, memory_angles[:,0,:].T.flatten()))
    ramaAM.addTabulatedFunction("psi_memories", Discrete2DFunction(num_residues, num_memories, memory_angles[:,1,:].T.flatten()))
    for res_index in range(num_residues):
        # we add 1 to res_index for the purposes of indexing atoms in our structure because num_residues begins counting
        # from the second residue in the structure (because the first doesn't get a rama bias)
        ramaAM.addBond([oa.c[(1+res_index)-1], oa.n[1+res_index], oa.ca[1+res_index], oa.c[1+res_index], oa.n[1+res_index+1]], [res_index])
    if oa.periodic_box:
        ramaAM.setUsesPeriodicBoundaryConditions(True)
        print('\nrama_ssweight_term is periodic') 
    ramaAM.setForceGroup(forceGroup)
    
    return ramaAM

def rama_AM_term(oa, memory_angles, memory_weights=None, k_rama=4.184, forceGroup=21):
    memory_weights[1:,:] *= 0
    fudge_factor = 1/20
    if memory_weights is not None:
        memory_weights *= fudge_factor

    # associative memory ramachandran potential
    # memory_angles: np.ndarray with shape (oa.nres-2, 2, number memories)
    #     where indices [:,0,:] describe the phi values of each memory
    #       and indices [:,1,:] describe the psi values of each memory 

    #TODO: this method is still under construction
    if oa.fixed_residue_indices:
        raise NotImplementedError("rama_AM_term is not yet supported for systems where certain absolute residue positions are fixed in space")
    if len(oa.chain_starts) > 1:
        raise NotImplementedError("rama_AM_term is not yet supported for systems with more than one chain")

    # check input
    if np.max(memory_angles) > np.pi:
        raise ValueError(f"memory_angles should be in radians over the interval (-pi,pi], but found {np.max(memory_angles)}")
    if np.min(memory_angles) < -np.pi:
        raise ValueError(f"memory_angles should be in radians over the interval (-pi,pi], but found {np.min(memory_angles)}")
    if not memory_angles.shape[0] == oa.nres - 2:
        raise ValueError(f"The size of memory_angles along axis 0 should be {oa.nres-2} (oa.nres-2) but was {memory_angles.shape[0]}.\
            We expect oa.nres-2 residues because every residue except the first and last in the chain should have memories")
    # should add check on the shapes of memory_weights, sigma, omega_phi, and omega_psi if we eventually make those 
    #     independent function parameters instead of hard-coding a particular choice of these values (see below)

    # load/calculate basic parameters
    num_residues = memory_angles.shape[0] # assign to descriptively named variable to clarify meaning of this axis
    num_memories = memory_angles.shape[2] # assign to descriptively named variable to clarify meaning of this axis
    if memory_weights is None:
        memory_weights = np.ones((num_residues, num_memories))
    else:
        if not memory_weights.shape == ((num_residues, num_memories)):
            raise ValueError(f"memory_weights should have shape (num_residues,num_memories)=={(num_residues, num_memories)}, \
                               but was {memory_weights.shape}")
    sigma = np.ones((num_residues, num_memories))*500#5000
    omega_phi = np.ones((num_residues, num_memories))
    omega_psi = np.ones((num_residues, num_memories))
    k_rama *= oa.k_awsem

    # set up potential
    rama_string_base = "".join([f"w(res_index,{mem_index})*exp(-(sigma(res_index,{mem_index})*\
        (omega_phi(res_index,{mem_index})*(cos(dihedral(p1,p2,p3,p4)-phi_memories(res_index,{mem_index}))-1)^2\
         +omega_psi(res_index,{mem_index})*(cos(dihedral(p2,p3,p4,p5)-psi_memories(res_index,{mem_index}))-1)^2)))+" for mem_index in range(num_memories)])
    rama_string_full = f"-{k_rama}*({rama_string_base[:-1]})" # removes trailing + from energy expression
                           # multiplying by 2 because we normally stack the base and bias on top of each other, 
                           # but this Force is intended to be used alone
    ramaAM = CustomCompoundBondForce(5, rama_string_full)
    ramaAM.addPerBondParameter("res_index") # will be used to look up the row holding all of our memories for each residue
    ramaAM.addTabulatedFunction("w", Discrete2DFunction(num_residues, num_memories, memory_weights.T.flatten()))
    ramaAM.addTabulatedFunction("sigma", Discrete2DFunction(num_residues, num_memories, sigma.T.flatten()))
    ramaAM.addTabulatedFunction("omega_phi", Discrete2DFunction(num_residues, num_memories, omega_phi.T.flatten()))
    ramaAM.addTabulatedFunction("omega_psi", Discrete2DFunction(num_residues, num_memories, omega_psi.T.flatten()))
    ramaAM.addTabulatedFunction("phi_memories", Discrete2DFunction(num_residues, num_memories, memory_angles[:,0,:].T.flatten()))
    ramaAM.addTabulatedFunction("psi_memories", Discrete2DFunction(num_residues, num_memories, memory_angles[:,1,:].T.flatten()))
    for res_index in range(num_residues):
        # we add 1 to res_index for the purposes of indexing atoms in our structure because num_residues begins counting
        # from the second residue in the structure (because the first doesn't get a rama bias)
        ramaAM.addBond([oa.c[(1+res_index)-1], oa.n[1+res_index], oa.ca[1+res_index], oa.c[1+res_index], oa.n[1+res_index+1]], [res_index])
    if oa.periodic_box:
        ramaAM.setUsesPeriodicBoundaryConditions(True)
        print('\nrama_ssweight_term is periodic') 
    ramaAM.setForceGroup(forceGroup)
    
    return ramaAM

def AM_rama_vram_overflow(oa, k_rama=4.184, forceGroup=21, map_dir='/scratch/AM_rama/rulisek_tripeptide/cmaps_90/data'):
    #TODO: this method is still under construction
    if oa.fixed_residue_indices:
        raise NotImplementedError("AM_rama is not yet supported for systems where certain absolute residue positions are fixed in space")
    if len(oa.chain_starts) > 1:
        raise NotImplementedError("AM_rama is not yet supported for systems with more than one chain")
    # get all 3-letter sequences of the 20 common amino acids
    seq_list = []
    for aa1 in one_to_three.keys():
        for aa2 in one_to_three.keys():
            for aa3 in one_to_three.keys():  
                seq_list.append(f'{aa1}{aa2}{aa3}')
    # initialize Force
    ramaAM = CMAPTorsionForce()
    # load the PES for each sequence into the Force
    for seq in seq_list:
        ramaAM.addMap(90,k_rama*np.load(f'{map_dir}/{aa1}{aa2}{aa3}.npy')) # column-major flattened 360x360 grid
    # add dihedrals to the Force
    assert len(oa.seq) == oa.nres
    for res_index in range(1,oa.nres-1):
        three_segment = oa.seq[res_index-1:res_index+2]
        map_index = seq_list.index(three_segment)
        phi1 = oa.c[res_index-1]
        phi2 = psi1 = oa.n[res_index]
        phi3 = psi2 = oa.ca[res_index]
        phi4 = psi3 = oa.c[res_index]
        psi4 = oa.n[res_index+1]
        ramaAM.addTorsion(map_index, phi1, phi2, phi3, phi4, psi1, psi2, psi3, psi4) 
    # finish Force setup
    if oa.periodic_box:
        ramaAM.setUsesPeriodicBoundaryConditions(True)
        print('\nAM_rama is periodic') 
    ramaAM.setForceGroup(forceGroup)
    return ramaAM

def AM_rama(oa, k_rama=4.184, forceGroup=21, map_dir=f'{os.environ.get("OPENAWSEM_LOCATION")}/parameters/rama'):
    #TODO: this method is still under construction
    if oa.fixed_residue_indices:
        raise NotImplementedError("AM_rama is not yet supported for systems where certain absolute residue positions are fixed in space")
    if len(oa.chain_starts) > 1:
        raise NotImplementedError("AM_rama is not yet supported for systems with more than one chain")
    # initialize Force
    ramaAM = CMAPTorsionForce()
    # add map and dihedrals to the Force
    assert len(oa.seq) == oa.nres
    for res_index in range(1,oa.nres-1):
        # one map for each set of 3 -- could be reduced for reduntant sequences
        # TODO: reuse previously configured Map for second, third, etc. occurrences of each 3-letter motif
        # If we have a super diverse/long sequence, this will still require too much RAM/vRAM, but I think
        # we can get away with this strategy for most systems
        ramaAM.addMap(90,k_rama*np.load(f'{map_dir}/{oa.seq[res_index-1:res_index+2]}.npy')) # column-major flattened 90x90 grid
        three_segment = oa.seq[res_index-1:res_index+2]
        phi1 = oa.c[res_index-1]
        phi2 = psi1 = oa.n[res_index]
        phi3 = psi2 = oa.ca[res_index]
        phi4 = psi3 = oa.c[res_index]
        psi4 = oa.n[res_index+1]
        ramaAM.addTorsion(res_index-1, phi1, phi2, phi3, phi4, psi1, psi2, psi3, psi4) 
    # finish Force setup
    if oa.periodic_box:
        ramaAM.setUsesPeriodicBoundaryConditions(True)
        print('\nAM_rama is periodic') 
    ramaAM.setForceGroup(forceGroup)
    return ramaAM

def side_chain_term(oa, k=1*kilocalorie_per_mole, gmmFileFolder="/Users/weilu/opt/parameters/side_chain", forceGroup=25):
    # add chi forces
    # The sign of the equilibrium value is opposite and magnitude differs slightly
    # 251.04 = 60 * 4.184 kJ, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k = k.value_in_unit(kilojoule_per_mole)
    k_side_chain = k * oa.k_awsem
    n_components = 3

    means_all_res = np.zeros((20, 3, 3))
    precisions_chol_all_res = np.zeros((20, 3, 3, 3))
    log_det_all_res = np.zeros((20, 3))
    weights_all_res = np.zeros((20, 3))
    mean_dot_precisions_chol_all_res = np.zeros((20, 3, 3))

    res_type_map_letters = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                            'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
    for i, res_type_one_letter in enumerate(res_type_map_letters):
        res_type = one_to_three[res_type_one_letter]
        if res_type == "GLY":
            weights_all_res[i] = np.array([1/3, 1/3, 1/3])
            continue

        means = np.loadtxt(f"{gmmFileFolder}/{res_type}_means.txt")
        precisions_chol = np.loadtxt(f"{gmmFileFolder}/{res_type}_precisions_chol.txt").reshape(3,3,3)
        log_det = np.loadtxt(f"{gmmFileFolder}/{res_type}_log_det.txt")
        weights = np.loadtxt(f"{gmmFileFolder}/{res_type}_weights.txt")
        means_all_res[i] = means

        precisions_chol_all_res[i] = precisions_chol
        log_det_all_res[i] = log_det
        weights_all_res[i] = weights


        for j in range(n_components):
            mean_dot_precisions_chol_all_res[i][j] = np.dot(means[j], precisions_chol[j])

    means_all_res = means_all_res.reshape(20, 9)
    precisions_chol_all_res = precisions_chol_all_res.reshape(20, 27)
    mean_dot_precisions_chol_all_res = mean_dot_precisions_chol_all_res.reshape(20, 9)

    log_weights = np.log(weights_all_res)
    sumexp_line = "+".join([f"exp(log_gaussian_and_weights_{i}-c)" for i in range(n_components)])
    const = 3 * np.log(2 * np.pi)
    side_chain = CustomCompoundBondForce(4, f"-{k_side_chain}*(log({sumexp_line})+c);\
                                        c=max(log_gaussian_and_weights_0,max(log_gaussian_and_weights_1,log_gaussian_and_weights_2));\
                                        log_gaussian_and_weights_0=log_gaussian_prob_0+log_weights(res,0);\
                                        log_gaussian_and_weights_1=log_gaussian_prob_1+log_weights(res,1);\
                                        log_gaussian_and_weights_2=log_gaussian_prob_2+log_weights(res,2);\
                                        log_gaussian_prob_0=-.5*({const}+log_prob_0)+log_det(res,0);\
                                        log_gaussian_prob_1=-.5*({const}+log_prob_1)+log_det(res,1);\
                                        log_gaussian_prob_2=-.5*({const}+log_prob_2)+log_det(res,2);\
                                        log_prob_0=((r1*pc(res,0)+r2*pc(res,3)+r3*pc(res,6)-mdpc(res,0))^2+\
                                        (r1*pc(res,1)+r2*pc(res,4)+r3*pc(res,7)-mdpc(res,1))^2+\
                                        (r1*pc(res,2)+r2*pc(res,5)+r3*pc(res,8)-mdpc(res,2))^2);\
                                        log_prob_1=((r1*pc(res,9)+r2*pc(res,12)+r3*pc(res,15)-mdpc(res,3))^2+\
                                        (r1*pc(res,10)+r2*pc(res,13)+r3*pc(res,16)-mdpc(res,4))^2+\
                                        (r1*pc(res,11)+r2*pc(res,14)+r3*pc(res,17)-mdpc(res,5))^2);\
                                        log_prob_2=((r1*pc(res,18)+r2*pc(res,21)+r3*pc(res,24)-mdpc(res,6))^2+\
                                        (r1*pc(res,19)+r2*pc(res,22)+r3*pc(res,25)-mdpc(res,7))^2+\
                                        (r1*pc(res,20)+r2*pc(res,23)+r3*pc(res,26)-mdpc(res,8))^2);\
                                        r1=10*distance(p1,p4);\
                                        r2=10*distance(p2,p4);\
                                        r3=10*distance(p3,p4)")
    
    if oa.periodic_box:
        side_chain.setUsesPeriodicBoundaryConditions(True)

    side_chain.addPerBondParameter("res")
    side_chain.addTabulatedFunction("pc", Discrete2DFunction(20, 27, precisions_chol_all_res.T.flatten()))
    side_chain.addTabulatedFunction("log_weights", Discrete2DFunction(20, 3, log_weights.T.flatten()))
    side_chain.addTabulatedFunction("log_det", Discrete2DFunction(20, 3, log_det_all_res.T.flatten()))
    side_chain.addTabulatedFunction("mdpc", Discrete2DFunction(20, 9, mean_dot_precisions_chol_all_res.T.flatten()))
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL":
            side_chain.addBond([oa.n[i], oa.ca[i], oa.c[i], oa.cb[i]], [gamma_se_map_1_letter[oa.seq[i]]])
    side_chain.setForceGroup(forceGroup)
    return side_chain

def chain_no_cb_constraint_term(oa, k_chain=50208, bond_lengths=[0.2459108, 0.2519591, 0.2466597], forceGroup=20):
    # add chain forces
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_chain *= oa.k_awsem
    chain = HarmonicBondForce()

    if oa.periodic_box:
        chain.setUsesPeriodicBoundaryConditions(True)

    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends:
            chain.addBond(oa.n[i], oa.c[i], bond_lengths[2], k_chain)
    chain.setForceGroup(forceGroup)
    return chain

def con_no_cb_constraint_term(oa, k_con=50208, bond_lengths=[.3816, .240, .276, .153], forceGroup=20):
    # add con forces
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_con *= oa.k_awsem
    con = HarmonicBondForce()

    if oa.periodic_box:
        con.setUsesPeriodicBoundaryConditions(True)
    
    for i in range(oa.nres):
        con.addBond(oa.ca[i], oa.o[i], bond_lengths[1], k_con)
        if ((i in oa.chain_starts) or (i in oa.chain_ends)) and (not oa.res_type[i] == "IGL"):
            # start doesn't have N, end doesn't have C. so only give a naive bond
            con.addBond(oa.ca[i], oa.cb[i], bond_lengths[3], k_con)
        if i not in oa.chain_ends:
            con.addBond(oa.ca[i], oa.ca[i+1], bond_lengths[0], k_con)
            con.addBond(oa.o[i], oa.ca[i+1], bond_lengths[2], k_con)
    con.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return con



def cbd_excl_term(oa, k=1*kilocalorie_per_mole, r_excl=0.7, fileLocation='cbd_cbd_real_contact_symmetric.csv', forceGroup=24):
    # Cb domain Excluded volume
    # With residue specific parameters
    # a harmonic well with minimum at the database histogram peak.
    # and 1 kT(0.593 kcal/mol) penalty when the distance is at r_min of the database.
    # multiply interaction strength by overall scaling
    # Openawsem doesn't have the distance range (r_excl) change from 0.35 to 0.45 when the sequence separtation more than 5
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_excl = k * oa.k_awsem
    excl = CustomNonbondedForce(f"{k_excl}*step(r_max(res1,res2)-r)*((r-r_max(res1,res2))/(r_max(res1,res2)-r_min(res1,res2)))^2")
    excl.addPerParticleParameter("res")

    gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
    for i in range(oa.natoms):
        excl.addParticle([gamma_se_map_1_letter[oa.seq[oa.resi[i]]]])
    # print(oa.ca)
    # print(oa.bonds)
    # print(oa.cb)

    r_min_table = np.zeros((20,20))
    r_max_table = np.zeros((20,20))
    # fileLocation = '/Users/weilu/Research/server/mar_2020/cmd_cmd_exclude_volume/cbd_cbd_real_contact_symmetric.csv'
    df = pd.read_csv(fileLocation)
    for i, line in df.iterrows():
        res1 = line["ResName1"]
        res2 = line["ResName2"]
        r_min_table[gamma_se_map_1_letter[three_to_one[res1]]][gamma_se_map_1_letter[three_to_one[res2]]] = line["r_min"] / 10.0   # A to nm
        r_min_table[gamma_se_map_1_letter[three_to_one[res2]]][gamma_se_map_1_letter[three_to_one[res1]]] = line["r_min"] / 10.0
        r_max_table[gamma_se_map_1_letter[three_to_one[res1]]][gamma_se_map_1_letter[three_to_one[res2]]] = line["r_max"] / 10.0
        r_max_table[gamma_se_map_1_letter[three_to_one[res2]]][gamma_se_map_1_letter[three_to_one[res1]]] = line["r_max"] / 10.0

    excl.addTabulatedFunction("r_min", Discrete2DFunction(20, 20, r_min_table.T.flatten()))
    excl.addTabulatedFunction("r_max", Discrete2DFunction(20, 20, r_max_table.T.flatten()))
    excl.addInteractionGroup([x for x in oa.cb if x > 0], [x for x in oa.cb if x > 0])

    excl.setCutoffDistance(r_excl)
    if oa.periodic_box:
        excl.setNonbondedMethod(excl.CutoffPeriodic)
    else:
        excl.setNonbondedMethod(excl.CutoffNonPeriodic)

    # excl.setNonbondedMethod(excl.CutoffNonPeriodic)
    excl.createExclusionsFromBonds(oa.bonds, 1)
    excl.setForceGroup(forceGroup)
    print("cb domain exlcude volume term On")
    return excl
