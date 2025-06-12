try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except ModuleNotFoundError:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
import numpy as np
from pathlib import Path
import openawsem

se_map_1_letter = {'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                   'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                   'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                   'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

def isChainStart(residueId, chain_starts, n=2):
    # return true if residue is near chain starts.
    # n=0 means always return False
    # n=1 means only return True if residue is the the first residue of a chain.
    # n=2 means return True if residue is the first or the one nearest to the first residue of a chain.
    atBegin = False
    for i in range(n):
        if (residueId-i) in chain_starts:
            atBegin = True
    return atBegin

def isChainEnd(residueId, chain_ends, n=2):
    # return true if residue is near chain ends.
    # n=0 means always return False
    # n=1 means only return True if residue is the the last residue of a chain.
    # n=2 means return True if residue is the last or the one nearest to the last residue of a chain.
    atEnd = False
    for i in range(n):
        if (residueId+i) in chain_ends:
            atEnd = True
    return atEnd
def isChainEdge(residueId, chain_starts, chain_ends, n=2):
    # n is how far away from the two ends count as in chain edge.
    return (isChainStart(residueId, chain_starts, n) or isChainEnd(residueId, chain_ends, n))

def inWhichChain(residueId, chain_ends):
    chain_table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    for i, end_of_chain_resId in enumerate(chain_ends):
        if end_of_chain_resId < residueId:
            pass
        else:
            return chain_table[i]

def read_beta_parameters(parametersLocation=None):
    if parametersLocation is None:
        parametersLocation=openawsem.data_path.parameters
    parametersLocation=Path(parametersLocation)
    assert parametersLocation.is_dir(), 'Directory does not exist'
    in_anti_HB = open(parametersLocation/"anti_HB", 'r').readlines()
    in_anti_NHB = open(parametersLocation/"anti_NHB", 'r').readlines()
    in_para_HB = open(parametersLocation/"para_HB", 'r').readlines()
    in_para_one = open(parametersLocation/"para_one", 'r').readlines()
    in_anti_one = open(parametersLocation/"anti_one", 'r').readlines()

    p_par = np.zeros((20))
    p_anti = np.zeros((20))
    p_antihb = np.zeros((20,20,2))
    p_antinhb = np.zeros((20,20,2))
    p_parhb = np.zeros((20,20,2))

    for i in range(20):
        p_par[i] = float(in_para_one[i].strip())
        p_anti[i] = float(in_anti_one[i].strip())
        for j in range(20):
            p_antihb[i][j][0] = float(in_anti_HB[i].strip().split()[j])
            p_antinhb[i][j][0] = float(in_anti_NHB[i].strip().split()[j])
            p_parhb[i][j][0] = float(in_para_HB[i].strip().split()[j])

    for i in range(20):
        for j in range(20):
            p_antihb[i][j][1] = float(in_anti_HB[i+21].strip().split()[j])
            p_antinhb[i][j][1] = float(in_anti_NHB[i+21].strip().split()[j])
            p_parhb[i][j][1] = float(in_para_HB[i+21].strip().split()[j])
    return p_par, p_anti, p_antihb, p_antinhb, p_parhb

def inSameChain(i,j,chain_starts,chain_ends):
    # determine whether residues are in the same chain
    #
    # sometimes, one of the residues might not exist
    # we'll treat not existing as being part of a different chain
    # but it shouldn't really affect anything
    if i<0 or j<0:
        if i<0 and j<0:
            raise AssertionError(f"Residues i and j do not exist! i: {i}, j: {j}")
        else:
            return False
    if i>chain_ends[-1] or j>chain_ends[-1]:
        if i>chain_ends[-1] and j>chain_ends[-1]:
            raise AssertionError(f"Residues i and j do not exist! i: {i}, j: {j}")
        else:
            return False
    # if we've made it this far, we know that both residues exist
    wombat = [int(chain_start<=i and i<=chain_end) for chain_start,chain_end in zip(chain_starts,chain_ends)]
    assert sum(wombat) == 1, f"i: {i}, chain_starts: {chain_starts}, chain_ends: {chain_ends}, list: {wombat}"
    chain_index_1 = wombat.index(1)
    wombat = [int(chain_start<=j and j<=chain_end) for chain_start,chain_end in zip(chain_starts,chain_ends)]
    assert sum(wombat) == 1, f"j: {j}, chain_starts: {chain_starts}, chain_ends: {chain_ends}, list: {wombat}"
    chain_index_2 = wombat.index(1)
    same_chain = chain_index_1==chain_index_2
    return same_chain

def get_lambda_by_index(i, j, lambda_i, chain_starts, chain_ends, ssweight="ssweight"):
    lambda_table = [[1.37, 1.36, 1.17], #lambda1 values as a function of sequence separation
                    [3.49, 3.50, 3.52], # lambda2 values as a function of sequence separation
                    [0.00, 3.47, 3.62]] # lambda3 values as a function of sequence separation
    # determine whether residues are in the same chain
    same_chain = inSameChain(i,j,chain_starts,chain_ends)
    assert ((same_chain is True) or (same_chain is False)), f"same_chain should be bool but was {same_chain}"
    # treat residues from different chains as intrachain residues of the largest sequence separation class
    if same_chain==False:
        return lambda_table[lambda_i][2]
    # for intrachain pairs, proceed as before
    elif abs(j-i) >= 4 and abs(j-i) < 18: # 3 instead of 4! the real sequence separation is different from that defined in the paper!
        return lambda_table[lambda_i][0]
    elif abs(j-i) >= 18 and abs(j-i) < 45:
        return lambda_table[lambda_i][1]
    elif abs(j-i) >= 45:
        return lambda_table[lambda_i][2]
    else:
        return 0 # need to do this for the regular hydrogen bond terms, while the lammps ones won't reach this else block

def get_alpha_by_index(i, j, alpha_i, chain_starts, chain_ends):
    alpha_table = [[1.30, 1.30, 1.30], # alpha1 for short seq sep, alpha 1 for medium seq sep, alpha 1 for long seq sep
                    [1.32, 1.32, 1.32], # alpha2 for short seq sep, alpha 2 for medium seq sep, alpha 2 for long seq sep
                    [1.22, 1.22, 1.22], # alpha3 for short seq sep, alpha 3 for medium seq sep, alpha 3 for long seq sep
                    [0.00, 0.33, 0.33], # alpha4 for short seq sep, alpha 4 for medium seq sep, alpha 4 for long seq sep
                    [0.00, 1.01, 1.01]] # alpha5 for short seq sep, alpha 5 for medium seq sep, alpha 5 for long seq sep
    # determine whether residues are in the same chain
    same_chain = inSameChain(i,j,chain_starts,chain_ends)
    # treat residues from different chains as intrachain residues of the largest sequence separation class
    if same_chain==False:
        return alpha_table[alpha_i][2]
    if abs(j-i) >= 4 and abs(j-i) < 18:
        return alpha_table[alpha_i][0]
    elif abs(j-i) >= 18 and abs(j-i) < 45:
        return alpha_table[alpha_i][1]
    elif abs(j-i) >= 45:
        return alpha_table[alpha_i][2]
    else:
        return 0

def get_pap_gamma_APH(donor_idx, acceptor_idx, chain_i, chain_j, gamma_APH):
    # if chain_i == chain_j and abs(j-i) < 13 or abs(j-i) > 16:
    # if abs(j-i) < 13 or abs(j-i) > 16:
    # if i-j < 13 or i-j > 16:
    # if (donor_idx - acceptor_idx >= 13 and donor_idx - acceptor_idx <= 16) or chain_i != chain_j:
    if (donor_idx - acceptor_idx >= 13 and donor_idx - acceptor_idx <= 16) and chain_i == chain_j:
        return gamma_APH
    else:
        return 0

def get_pap_gamma_AP(donor_idx, acceptor_idx, chain_i, chain_j, gamma_AP, ssweight):
    if ssweight[donor_idx][1] == 1 and ssweight[acceptor_idx][1] == 1:
        additional_scale = 1.5
    else:
        additional_scale = 1.0
    # if (donor_idx - acceptor_idx >= 17):
    if (donor_idx - acceptor_idx >= 17) or chain_i != chain_j:
        return additional_scale * gamma_AP
    else:
        return 0

def get_pap_gamma_P(donor_idx, acceptor_idx, chain_i, chain_j, gamma_P, ssweight):
    if ssweight[donor_idx][1] == 1 and ssweight[acceptor_idx][1] == 1:
        additional_scale = 1.5
    else:
        additional_scale = 1.0
    if (donor_idx - acceptor_idx >= 9) or chain_i != chain_j:
        return additional_scale * gamma_P
    else:
        return 0

def get_weight_and_p_by_index(i,j,chain_starts,chain_ends,alpha_label,p_array):
    # function to get the overall weight and probability parameter to multiply by alpha
    # i and j are of course our residue indices
    # alpha_label is an int from 1 to 5 depending on which alpha term we're getting the weight and probability for
    # p_array has two elements and we choose the correct one based on sequence separation
    weights = [0.5,0.25,1,1,1]
    weight = weights[alpha_label-1] # alpha_label is 1-indexed
    # determine whether residues are in the same chain
    same_chain = inSameChain(i,j,chain_starts,chain_ends)
    # treat residues from different chains as intrachain residues of the largest sequence separation class
    if same_chain==False:
        return weight*p_array[1]
    if abs(j-i) >= 4 and abs(j-i) < 18:
        return weight*p_array[0]
    elif abs(j-i) >= 18 and abs(j-i) < 45:
        return weight*p_array[1]
    elif abs(j-i) >= 45:
        return weight*p_array[1]
    else:
        return 0    

def get_Lambda_2(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, chain_starts, chain_ends):
    Lambda = get_lambda_by_index(i, j, 1, chain_starts, chain_ends)
    Lambda += get_alpha_by_index(i, j, 0, chain_starts, chain_ends)*get_weight_and_p_by_index(i,j,chain_starts,chain_ends,1,p_antihb[a[i],a[j]])
    Lambda += get_alpha_by_index(i, j, 1, chain_starts, chain_ends)*get_weight_and_p_by_index(i,j,chain_starts,chain_ends,2,p_antinhb[a[i+1],a[j-1]]+p_antinhb[a[i-1],a[j+1]])
    Lambda += get_alpha_by_index(i, j, 2, chain_starts, chain_ends)*(p_anti[a[i]] + p_anti[a[j]])*1 # weight of 1
    return Lambda

def get_Lambda_2_old(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, chain_starts, chain_ends, ssweight):
    Lambda = get_lambda_by_index(i, j, 1, chain_starts, chain_ends, ssweight='ssweight')
    if Lambda==0: # we zeroed it out due to sequence separation or something
        return 0
    #print(f"lambda2: {Lambda}")
    Lambda += 0.5*get_alpha_by_index(i, j, 0, chain_starts, chain_ends)*p_antihb[a[i], a[j]][0]#*0
    #print(f"alpha1: {get_alpha_by_index(i, j, 0, chain_starts, chain_ends)}")
    print(f"p_antihb[a[i], a[j]][0]: {p_antihb[a[i], a[j]][0]}")
    try:
        Lambda += 0.25*get_alpha_by_index(i, j, 1, chain_starts, chain_ends)*(p_antinhb[a[i+1], a[j-1]][0] + p_antinhb[a[i-1], a[j+1]][0])#*0.530203*4
    except IndexError:
        print(f"i: {i}")
        print(f"j: {j}")
        print(f"chain_starts: {chain_starts}")
        print(f"chain_ends: {chain_ends}")
        raise
    #print(f"alpha2: {get_alpha_by_index(i, j, 1, chain_starts, chain_ends)}")
    print(f"p_antinhb[a[i+1], a[j-1]][0] + p_antinhb[a[i-1], a[j+1]][0]: {p_antinhb[a[i+1], a[j-1]][0] + p_antinhb[a[i-1], a[j+1]][0]}")
    #print(f"p_antinhb[a[i+1], a[j-1]][0]: {p_antinhb[a[i+1], a[j-1]][0]}")
    #print(f"p_antinhb[a[i-1], a[j+1]][0]: {p_antinhb[a[i-1], a[j+1]][0]}")
    Lambda += get_alpha_by_index(i, j, 2, chain_starts, chain_ends)*(p_anti[a[i]] + p_anti[a[j]])#*0.0915795
    #print(f"alpha3: {get_alpha_by_index(i, j, 2, chain_starts, chain_ends)}")
    print(f"(p_anti[a[i]] + p_anti[a[j]]): {(p_anti[a[i]] + p_anti[a[j]])}")
    #print(f"p_anti[a[i]]: {p_anti[a[i]]}")
    #print(f"p_anti[a[j]]: {p_anti[a[j]]}")    
    #print(f"Lambda: {Lambda}")
    return Lambda

def get_Lambda_3(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, chain_starts, chain_ends):
    Lambda = get_lambda_by_index(i, j, 2, chain_starts, chain_ends)
    Lambda += get_alpha_by_index(i, j, 3, chain_starts, chain_ends)*get_weight_and_p_by_index(i,j,chain_starts,chain_ends,4,p_parhb[a[i+1],a[j]])
    Lambda += get_alpha_by_index(i, j, 4, chain_starts, chain_ends)*p_par[a[i+1]]*1 # weight of 1
    Lambda += get_alpha_by_index(i, j, 4, chain_starts, chain_ends)*p_par[a[j]]*1 # weight of 1
    return Lambda

def convert_units(k):
    if isinstance(k, float) or isinstance(k, int):
        k = k   # just for backward comptable
    elif isinstance(k, Quantity):
        k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    else:
        print(f"Unknown input, {k}, {type(k)}")
    return k

def beta_term_1(oa, k=0.5*kilocalories_per_mole, forceGroup=27):
    print("beta_1 term ON")
    k_beta = convert_units(k) * oa.k_awsem
    nres, n, h, ca, o, res_type = oa.nres, oa.n, oa.h, oa.ca, oa.o, oa.res_type
    # print(lambda_1)
    r_ON = .298
    sigma_NO = .068
    r_OH = .206
    sigma_HO = .076

    lambda_1 = np.zeros((nres, nres))
    for residue1 in oa.residues:
        i = residue1.index
        for residue2 in oa.residues:
            j = residue2.index
            lambda_1[i][j] = get_lambda_by_index(i, j, 0, oa.chain_starts, oa.chain_ends)
    theta_ij = f"exp(-(r_Oi_Nj-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hj-{r_OH})^2/(2*{sigma_HO}^2))"
    mu_1 = 10  # nm^-1
    # mu_2 = 5   # nm^-1
    rcHB = 1.2  # in nm
    # v1i ensures the hydrogen bonding does not occur when five residue segment is shorter than 12 A
    # v1i = f"0.5*(1+tanh({mu_1}*(distance(a2,a3)-{rcHB})))"
    v1i = "1"
    beta_string_1 = f"-{k_beta}*lambda_1(res_i,res_j)*theta_ij*v1i;theta_ij={theta_ij};v1i={v1i};r_Oi_Nj=distance(a1,d1);r_Oi_Hj=distance(a1,d2);"
    beta_1 = CustomHbondForce(beta_string_1)

    # Set PBC. 02082024 Rebekah Added. --- Start
    if oa.periodic:
        beta_1.setNonbondedMethod(beta_1.CutoffPeriodic)
        print('\nbeta_term_1 is in PBC')
    else:
        beta_1.setNonbondedMethod(beta_1.CutoffNonPeriodic)
    # Set PBC. 02082024 Rebekah Added. --- End

    beta_1.addPerDonorParameter("res_i")
    beta_1.addPerAcceptorParameter("res_j")
    beta_1.addTabulatedFunction("lambda_1", Discrete2DFunction(nres, nres, lambda_1.T.flatten()))
    # print(lambda_1)
    # print(len(oa.o), nres)
    for i in range(nres):
        if oa.o[i]!= -1:
            ca_i_minus_2 = oa.ca[0] if i <= 2 else oa.ca[i-2]
            ca_i_plus_2 = oa.ca[-1] if i+2 >= nres else oa.ca[i+2]
            # beta_1.addAcceptor(oa.o[i], ca_i_minus_2, ca_i_plus_2, [i])
            beta_1.addAcceptor(oa.o[i], -1, -1, [i])
        if oa.n[i]!=-1 and oa.h[i]!=-1:
            beta_1.addDonor(oa.n[i], oa.h[i], -1, [i])
    # beta_1.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
    beta_1.setCutoffDistance(1.0)
    beta_1.setForceGroup(forceGroup)
    # beta_2.setForceGroup(24)
    # beta_3.setForceGroup(25)
    return beta_1

def beta_term_2(oa, k=0.5*kilocalories_per_mole, forceGroup=27):
    print("beta_2 term ON")
    k_beta = convert_units(k) * oa.k_awsem
    nres, n, h, ca, o, res_type = oa.nres, oa.n, oa.h, oa.ca, oa.o, oa.res_type
    # print(lambda_1)
    r_ON = .298
    sigma_NO = .068
    r_OH = .206
    sigma_HO = .076
    eta_beta_1 = 10.0
    eta_beta_2 = 5.0
    # r_HB_c = 0.4
    r_HB_c = 1.2
    p_par, p_anti, p_antihb, p_antinhb, p_parhb = read_beta_parameters()

    # for lookup table.
    a = []
    for ii in range(oa.nres):
        a.append(se_map_1_letter[oa.seq[ii]])

    lambda_2 = np.zeros((nres, nres))
    for i in range(nres):
        for j in range(nres):
            if isChainEdge(i, oa.chain_starts, oa.chain_ends, n=1) or \
                    isChainEdge(j, oa.chain_starts, oa.chain_ends, n=1):
                continue
            lambda_2[i][j] = get_Lambda_2(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, oa.chain_starts, oa.chain_ends)
    theta_ij = f"exp(-(r_Oi_Nj-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hj-{r_OH})^2/(2*{sigma_HO}^2))"
    theta_ji = f"exp(-(r_Oj_Ni-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oj_Hi-{r_OH})^2/(2*{sigma_HO}^2))"
    beta_string_2 = f"-{k_beta}*lambda_2(res_i,res_j)*theta_ij*theta_ji;\
                        theta_ij={theta_ij};r_Oi_Nj=distance(a1,d1);r_Oi_Hj=distance(a1,d2);\
                        theta_ji={theta_ji};r_Oj_Ni=distance(d3,a2);r_Oj_Hi=distance(d3,a3);"
    beta_2 = CustomHbondForce(beta_string_2)

    # Set PBC. 02082024 Rebekah Added. --- Start
    if oa.periodic:
        beta_2.setNonbondedMethod(beta_2.CutoffPeriodic)
        print('\nbeta_term_2 is in PBC')
    else:
        beta_2.setNonbondedMethod(beta_2.CutoffNonPeriodic)
    # Set PBC. 02082024 Rebekah Added. --- End

    beta_2.addPerDonorParameter("res_i")
    beta_2.addPerAcceptorParameter("res_j")
    beta_2.addTabulatedFunction("lambda_2", Discrete2DFunction(nres, nres, lambda_2.T.flatten()))
    # print(lambda_1)
    # print(len(oa.o), nres)
    for i in range(nres):
        if o[i]!= -1 and n[i]!=-1 and h[i]!=-1:
            beta_2.addAcceptor(o[i], n[i], h[i], [i])
            beta_2.addDonor(n[i], h[i], o[i], [i])
    # beta_2.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
    beta_2.setCutoffDistance(1.0)
    # beta_1.setForceGroup(23)
    beta_2.setForceGroup(forceGroup)
    # beta_3.setForceGroup(25)

    return beta_2


def beta_term_3(oa, k=0.5*kilocalories_per_mole, forceGroup=27):
    print("beta_3 term ON")
    k_beta = convert_units(k) * oa.k_awsem
    nres, n, h, ca, o, res_type = oa.nres, oa.n, oa.h, oa.ca, oa.o, oa.res_type
    # print(lambda_1)
    r_ON = .298
    sigma_NO = .068
    r_OH = .206
    sigma_HO = .076
    eta_beta_1 = 10.0
    eta_beta_2 = 5.0
    # r_HB_c = 0.4
    r_HB_c = 1.2
    p_par, p_anti, p_antihb, p_antinhb, p_parhb = read_beta_parameters()

    # for lookup table.
    a = []
    for ii in range(oa.nres):
        a.append(se_map_1_letter[oa.seq[ii]])

    lambda_3 = np.zeros((nres, nres))
    for i in range(nres):
        for j in range(nres):
            if isChainEdge(i, oa.chain_starts, oa.chain_ends, n=1) or \
                    isChainEdge(j, oa.chain_starts, oa.chain_ends, n=1):
                continue
            lambda_3[i][j] = get_Lambda_3(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, oa.chain_starts, oa.chain_ends)

    theta_ij = f"exp(-(r_Oi_Nj-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hj-{r_OH})^2/(2*{sigma_HO}^2))"
    theta_jip2 = f"exp(-(r_Oj_Nip2-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oj_Hip2-{r_OH})^2/(2*{sigma_HO}^2))"

    beta_string_3 = f"-{k_beta}*lambda_3(res_i,res_j)*theta_ij*theta_jip2;\
                        theta_ij={theta_ij};r_Oi_Nj=distance(a1,d1);r_Oi_Hj=distance(a1,d2);\
                        theta_jip2={theta_jip2};r_Oj_Nip2=distance(d3,a2);r_Oj_Hip2=distance(d3,a3);"
    beta_3 = CustomHbondForce(beta_string_3)

    # Set PBC. 02082024 Rebekah Added. --- Start
    if oa.periodic:
        beta_3.setNonbondedMethod(beta_3.CutoffPeriodic)
        print('\nbeta_term_3 is in PBC')
    else:
        beta_3.setNonbondedMethod(beta_3.CutoffNonPeriodic)
    # Set PBC. 02082024 Rebekah Added. --- End
        
    beta_3.addPerDonorParameter("res_i")
    beta_3.addPerAcceptorParameter("res_j")
    beta_3.addTabulatedFunction("lambda_3", Discrete2DFunction(nres, nres, lambda_3.T.flatten()))
    # print(lambda_1)
    # print(len(oa.o), nres)
    for i in range(nres):
        if isChainEdge(i, oa.chain_starts, oa.chain_ends, n=2):
            continue
        if o[i] != -1 and n[i+2] !=-1 and h[i+2] !=-1:
            beta_3.addAcceptor(o[i], n[i+2], h[i+2], [i])
        if o[i] != -1 and n[i] !=-1 and h[i] !=-1:
            beta_3.addDonor(n[i], h[i], o[i], [i])
    # beta_3.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
    beta_3.setCutoffDistance(1.0)
    # beta_1.setForceGroup(23)
    # beta_2.setForceGroup(24)
    beta_3.setForceGroup(forceGroup)

    return beta_3


def pap_term_1(oa, k=0.5*kilocalories_per_mole, dis_i_to_i4=1.2, forceGroup=28, ssweightFileName="ssweight"):
    print("pap_1 term ON")
    k_pap = convert_units(k) * oa.k_awsem
    # dis_i_to_i4 should be in nm, it disfavor hydrogen bond when ca_i and ca_i+4 are 1.2 nm apart away.
    nres, ca = oa.nres, oa.ca
    # r0 = 2.0 # nm
    r0 = 0.8  # nm
    eta_pap = 70  # nm^-1
    gamma_aph = 1.0
    gamma_ap = 0.4
    gamma_p = 0.4

    if not os.path.exists(ssweightFileName):
        print("No ssweight given, assume all zero")
        ssweight = np.zeros((nres, 2))
    else:
        ssweight = np.loadtxt(ssweightFileName)

    gamma_1 = np.zeros((nres, nres))
    gamma_2 = np.zeros((nres, nres))
    for i in range(nres):
        for j in range(nres):
            resId1 = i
            chain1 = inWhichChain(resId1, oa.chain_ends)
            resId2 = j
            chain2 = inWhichChain(resId2, oa.chain_ends)
            gamma_1[i][j] = get_pap_gamma_APH(i, j, chain1, chain2, gamma_aph)
            gamma_2[i][j] = get_pap_gamma_AP(i, j, chain1, chain2, gamma_ap, ssweight)

    constraint_i_and_i4 = f"0.5*(1+tanh({eta_pap}*(distance(a1,a2)-{dis_i_to_i4})))"

    pap_function = f"-{k_pap}*(gamma_1(donor_idx,acceptor_idx)+gamma_2(donor_idx,acceptor_idx))\
                        *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
                        *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))\
                        *{constraint_i_and_i4}"

    # pap_function = f"-{k_pap}*(gamma_1(donor_idx,acceptor_idx))\
    #                     *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
    #                     *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))"

    # pap_function = f"-{k_pap}*distance(a1,d1)"
    pap = CustomHbondForce(pap_function)
    
    # Set PBC. 02082024 Rebekah Added. --- Start
    if oa.periodic:
        pap.setNonbondedMethod(pap.CutoffPeriodic)
        print('\npap_1 is in PBC')
    else:
        pap.setNonbondedMethod(pap.CutoffNonPeriodic)
    # Set PBC. 02082024 Rebekah Added. --- End
    pap.addPerDonorParameter("donor_idx")
    pap.addPerAcceptorParameter("acceptor_idx")
    pap.addTabulatedFunction("gamma_1", Discrete2DFunction(nres, nres, gamma_1.T.flatten()))
    pap.addTabulatedFunction("gamma_2", Discrete2DFunction(nres, nres, gamma_2.T.flatten()))
    # print(ca)
    # count = 0;
    i = 0

    for i in range(nres):
        if not isChainEnd(i, oa.chain_ends, n=4):
            pap.addAcceptor(ca[i], ca[i+4], -1, [i])

        if not isChainStart(i, oa.chain_starts, n=4):
            if oa.n[i] != -1 and oa.n[i-4] != -1:
                pap.addDonor(oa.n[i], oa.n[i-4], -1, [i])

    # pap.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
    pap.setCutoffDistance(1.0)
    # print(count)
    pap.setForceGroup(forceGroup)
    return pap

def pap_term_2(oa, k=0.5*kilocalories_per_mole, dis_i_to_i4=1.2, forceGroup=28, ssweightFileName="ssweight"):
    print("pap_2 term ON")
    k_pap = convert_units(k) * oa.k_awsem
    nres, ca = oa.nres, oa.ca
    # r0 = 2.0 # nm
    r0 = 0.8  # nm
    eta_pap = 70  # nm^-1
    gamma_aph = 1.0
    gamma_ap = 0.4
    gamma_p = 0.4
    if not os.path.exists(ssweightFileName):
        print("No ssweight given, assume all zero")
        ssweight = np.zeros((nres, 2))
    else:
        ssweight = np.loadtxt(ssweightFileName)

    gamma_3 = np.zeros((nres, nres))
    for i in range(nres):
        for j in range(nres):
            resId1 = i
            chain1 = inWhichChain(resId1, oa.chain_ends)
            resId2 = j
            chain2 = inWhichChain(resId2, oa.chain_ends)
            gamma_3[i][j] = get_pap_gamma_P(i, j, chain1, chain2, gamma_p, ssweight)


    constraint_i_and_i4 = f"0.5*(1+tanh({eta_pap}*(distance(a1,a2)-{dis_i_to_i4})))"
    pap_function = f"-{k_pap}*gamma_3(donor_idx,acceptor_idx)\
                        *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
                        *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))\
                        *{constraint_i_and_i4}"
    pap = CustomHbondForce(pap_function)

    # Set PBC. 02082024 Rebekah Added. --- Start
    if oa.periodic:
        pap.setNonbondedMethod(pap.CutoffPeriodic)
        print('\npap_2 is in PBC')
    else:
        pap.setNonbondedMethod(pap.CutoffNonPeriodic)
    # Set PBC. 02082024 Rebekah Added. --- End
        
    pap.addPerDonorParameter("donor_idx")
    pap.addPerAcceptorParameter("acceptor_idx")
    pap.addTabulatedFunction("gamma_3", Discrete2DFunction(nres, nres, gamma_3.T.flatten()))
    # print(oa.n)
    # count = 0;
    for i in range(nres):
        if not isChainEnd(i, oa.chain_ends, n=4):
            pap.addAcceptor(ca[i], ca[i+4], -1, [i])
            # pap.addDonor(ca[i], ca[i+4], -1, [i])
            if oa.n[i] != -1 and oa.n[i+4] != -1:
                pap.addDonor(oa.n[i], oa.n[i+4], -1, [i])

    # pap.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
    pap.setCutoffDistance(1.0)
    # print(count)
    pap.setForceGroup(forceGroup)
    return pap

def get_helical_f(oneLetterCode, inMembrane=False):
    if inMembrane:
        table = {"A": 0.79, "R": 0.62, "N": 0.49, "D": 0.44, "C": 0.76, "Q": 0.61, "E": 0.57, "G": 0.57, "H": 0.63, "I": 0.81,
            "L": 0.81, "K": 0.56, "M": 0.80, "F": 0.76, "P": 0.44, "S": 0.6, "T": 0.67, "W": 0.74, "Y": 0.71, "V": 0.79}
    else:
        table = {"A": 0.77, "R": 0.68, "N": 0.07, "D": 0.15, "C": 0.23, "Q": 0.33, "E": 0.27, "G": 0.0, "H": 0.06, "I": 0.23,
            "L": 0.62, "K": 0.65, "M": 0.5, "F": 0.41, "P": 0.4, "S": 0.35, "T": 0.11, "W": 0.45, "Y": 0.17, "V": 0.14}
    return table[oneLetterCode]

def helical_term(oa, k_helical=4.184, inMembrane=False, forceGroup=29):
    # without density dependency.
    # without z dependency for now.
    k_helical *= oa.k_awsem
    sigma_NO = 0.068
    sigma_HO = 0.076
    r_ON = 0.298
    r_OH = 0.206

    theta_ij = f"exp(-(r_Oi_Nip4-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hip4-{r_OH})^2/(2*{sigma_HO}^2))"
    helical = CustomCompoundBondForce(3, f"-{k_helical}*(fa_i+fa_ip4)*{theta_ij};\
                                        r_Oi_Nip4=distance(p1,p2);r_Oi_Hip4=distance(p1,p3);")
    helical.addPerBondParameter("fa_i")
    helical.addPerBondParameter("fa_ip4")
    for i in range(oa.nres):
        # if not isChainEnd(i, oa.chain_ends, n=4) and oa.res_type[i+4] == "IPR":
        #     print(oa.o[i], oa.n[i+4], oa.h[i+4])
        if not isChainEnd(i, oa.chain_ends, n=4) and oa.res_type[i+4] != "IPR":
            fa_i = get_helical_f(oa.seq[i], inMembrane=inMembrane)
            fa_ip4 = get_helical_f(oa.seq[i+4], inMembrane=inMembrane)
            helical.addBond([oa.o[i], oa.n[i+4], oa.h[i+4]], [fa_i, fa_ip4])

    helical.setForceGroup(forceGroup)
    return helical

def z_dependent_helical_term(oa, k_helical=4.184, membrane_center=0*angstrom, z_m=1.5, forceGroup=29):
    # without density dependency.
    k_helical *= oa.k_awsem
    sigma_NO = 0.068
    sigma_HO = 0.076
    r_ON = 0.298
    r_OH = 0.206
    eta_switching = 10
    membrane_center = membrane_center.value_in_unit(nanometer)   # convert to nm

    alpha_membrane = f"0.5*tanh({eta_switching}*((z4-{membrane_center})+{z_m}))+0.5*tanh({eta_switching}*({z_m}-(z4-{membrane_center})))"
    theta_ij = f"exp(-(r_Oi_Nip4-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hip4-{r_OH})^2/(2*{sigma_HO}^2))"
    helical = CustomCompoundBondForce(4, f"-{k_helical}*{theta_ij}*((fa_i+fa_ip4)*(1-alpha_membrane)+(fa_i_membrane+fa_ip4_membrane)*(alpha_membrane));\
                                        alpha_membrane={alpha_membrane};\
                                        r_Oi_Nip4=distance(p1,p2);r_Oi_Hip4=distance(p1,p3);")
    helical.addPerBondParameter("fa_i")
    helical.addPerBondParameter("fa_ip4")
    helical.addPerBondParameter("fa_i_membrane")
    helical.addPerBondParameter("fa_ip4_membrane")
    for i in range(oa.nres):
        # if not isChainEnd(i, oa.chain_ends, n=4) and oa.res_type[i+4] == "IPR":
        #     print(oa.o[i], oa.n[i+4], oa.h[i+4])
        if not isChainEnd(i, oa.chain_ends, n=4) and oa.res_type[i+4] != "IPR":
            fa_i = get_helical_f(oa.seq[i], inMembrane=False)
            fa_ip4 = get_helical_f(oa.seq[i+4], inMembrane=False)
            fa_i_membrane = get_helical_f(oa.seq[i], inMembrane=True)
            fa_ip4_membrane = get_helical_f(oa.seq[i+4], inMembrane=True)
            helical.addBond([oa.o[i], oa.n[i+4], oa.h[i+4], oa.ca[i]], [fa_i, fa_ip4, fa_i_membrane, fa_ip4_membrane])

    helical.setForceGroup(forceGroup)
    return helical


# def pap_term_1(oa, k_pap=4.184, dis_i_to_i4=-1):
#     print("pap_1 term ON")
#     nres, ca = oa.nres, oa.ca
#     # r0 = 2.0 # nm
#     r0 = 0.8 # nm
#     eta_pap = 70 # nm^-1
#     gamma_aph = 1.0
#     gamma_ap = 0.4
#     gamma_p = 0.4

#     gamma_1 = np.zeros((nres, nres))
#     gamma_2 = np.zeros((nres, nres))
#     for i in range(nres):
#         for j in range(nres):
#             resId1 = i
#             chain1 = inWhichChain(resId1, oa.chain_ends)
#             resId2 = j
#             chain2 = inWhichChain(resId2, oa.chain_ends)
#             gamma_1[i][j] = get_pap_gamma_APH(i, j, chain1, chain2, gamma_aph)
#             gamma_2[i][j] = get_pap_gamma_AP(i, j, chain1, chain2, gamma_ap)

#     pap_function = f"-{k_pap}*(gamma_1(donor_idx,acceptor_idx)+gamma_2(donor_idx,acceptor_idx))\
#                         *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
#                         *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))"

#     # pap_function = f"-{k_pap}*(gamma_1(donor_idx,acceptor_idx))\
#     #                     *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
#     #                     *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))"

#     # pap_function = f"-{k_pap}*distance(a1,d1)"
#     pap = CustomHbondForce(pap_function)
#     pap.addPerDonorParameter("donor_idx")
#     pap.addPerAcceptorParameter("acceptor_idx")
#     pap.addTabulatedFunction("gamma_1", Discrete2DFunction(nres, nres, gamma_1.T.flatten()))
#     pap.addTabulatedFunction("gamma_2", Discrete2DFunction(nres, nres, gamma_2.T.flatten()))
#     # print(ca)
#     # count = 0;
#     i = 0
#     # pap.addAcceptor(ca[0], ca[4], -1, [0])
#     # pap.addAcceptor(ca[20], ca[8], -1, [4])
#     # pap.addDonor(ca[20], ca[0], -1, [4])
#     for i in range(nres):
#         if not isChainEnd(i, oa.chain_ends, n=4):
#             pap.addAcceptor(ca[i], ca[i+4], -1, [i])

#         if i > 13 and not isChainStart(i, oa.chain_starts, n=4):
#             pap.addDonor(oa.n[i], oa.n[i-4], -1, [i])

#     pap.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
#     pap.setCutoffDistance(1.0)
#     # print(count)
#     pap.setForceGroup(26)
#     return pap


# def pap_term_1(oa, k_pap=4.184):
#     print("pap_1 term ON")
#     nres, ca = oa.nres, oa.ca
#     # r0 = 2.0 # nm
#     r0 = 0.8 # nm
#     eta_pap = 70 # nm^-1
#     gamma_aph = 1.0
#     gamma_ap = 0.4
#     gamma_p = 0.4

#     gamma_1 = np.zeros((nres, nres))
#     gamma_2 = np.zeros((nres, nres))
#     for i in range(nres):
#         for j in range(nres):
#             resId1 = i
#             chain1 = inWhichChain(resId1, oa.chain_ends)
#             resId2 = j
#             chain2 = inWhichChain(resId2, oa.chain_ends)
#             gamma_1[i][j] = get_pap_gamma_APH(i, j, chain1, chain2, gamma_aph)
#             gamma_2[i][j] = get_pap_gamma_AP(i, j, chain1, chain2, gamma_ap)

#     pap_function = f"-{k_pap}*(gamma_1(donor_idx,acceptor_idx)+gamma_2(donor_idx,acceptor_idx))\
#                         *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
#                         *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))"

#     pap_function = f"-{k_pap}*(gamma_1(donor_idx,acceptor_idx))\
#                         *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
#                         *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))"

#     pap_function = f"-{k_pap}*distance(a1,d1)"
#     pap = CustomHbondForce(pap_function)
#     pap.addPerDonorParameter("donor_idx")
#     pap.addPerAcceptorParameter("acceptor_idx")
#     pap.addTabulatedFunction("gamma_1", Discrete2DFunction(nres, nres, gamma_1.T.flatten()))
#     pap.addTabulatedFunction("gamma_2", Discrete2DFunction(nres, nres, gamma_2.T.flatten()))
#     # print(ca)
#     # count = 0;
#     i = 0
#     # pap.addAcceptor(ca[0], ca[4], -1, [0])
#     # pap.addAcceptor(ca[20], ca[8], -1, [4])
#     # pap.addDonor(ca[20], ca[0], -1, [4])
#     for i in range(nres):
#         if not isChainEnd(i, oa.chain_ends, n=4):
#             pap.addAcceptor(ca[i], ca[i+4], -1, [i])

#         if i > 13 and not isChainStart(i, oa.chain_starts, n=4):
#             pap.addDonor(ca[i], ca[i-4], -1, [i])

#     pap.setNonbondedMethod(CustomHbondForce.CutoffNonPeriodic)
#     pap.setCutoffDistance(1.0)
#     # print(count)
#     pap.setForceGroup(26)
#     return pap

def beta_lammps_awsemmd(oa,term_number,ssweight,forceGroup,k_beta=4.184):
    """ 
    Function to compute either beta 1, beta 2, or beta 3, as implemented in a particular LAMMPS AWSEM-MD commit,
    https://github.com/adavtyan/awsemmd/tree/cea754f1208fde6332d4d0f1cae3212bf7e8afbb

    Standard usage is forceGroup=23 for term 1, forceGroup=24 for term 2, and forceGroup=25 for term 3.
    """
    print(f"beta_{term_number} term ON")
    #
    # set constants
    nres, n, h, ca, o, res_type = oa.nres, oa.n, oa.h, oa.ca, oa.o, oa.res_type
    r_ON = .298
    sigma_NO = .068
    r_OH = .206
    sigma_HO = .076
    eta_beta_1 = 10.0
    eta_beta_2 = 5.0
    r_HB_c = 1.2
    number_atoms = [7,10,10] # number of atoms participating in each interaction type (beta1, beta2, beta3)
    #
    # load ssweight
    rama_biases = []
    with open(ssweight,'r') as f:
        for line in f:
            if line.strip() == "0.0 1.0":
                rama_biases.append('beta')
            else:
                rama_biases.append("not beta")
    #
    # load parameters
    p_par, p_anti, p_antihb, p_antinhb, p_parhb = read_beta_parameters()
    a = [] # list to help us to convert from amino acid type to the appropriate row/column indices in p_par, p_anti, etc.
    for ii in range(oa.nres):
        a.append(se_map_1_letter[oa.seq[ii]])
    #
    # define energy functions
    #   hydrogen bond geometry
    theta_ij =   f"exp(-(r_Oi_Nj-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hj-{r_OH})^2/(2*{sigma_HO}^2))"
    theta_ji =   f"exp(-(r_Oj_Ni-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oj_Hi-{r_OH})^2/(2*{sigma_HO}^2))"
    theta_jip2 = f"exp(-(r_Oj_Nip2-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oj_Hip2-{r_OH})^2/(2*{sigma_HO}^2))" 
    theta = [theta_ij, f"{theta_ij}*{theta_ji}", f"{theta_ij}*{theta_jip2}"] # [theta for beta1, theta for beta2, theta for beta3]
    distance_definitions = [f"r_Oi_Nj=distance(p1,p2);r_Oi_Hj=distance(p1,p3);\
                                  r_CAim2_CAip2=distance(p4,p5);r_CAjm2_CAjp2=distance(p6,p7);",
                            f"r_Oi_Nj=distance(p1,p2);r_Oi_Hj=distance(p1,p3);\
                                  r_Oj_Ni=distance(p4,p5);r_Oj_Hi=distance(p4,p6);\
                                  r_CAim2_CAip2=distance(p7,p8);r_CAjm2_CAjp2=distance(p9,p10)",
                            f"r_Oi_Nj=distance(p1,p2);r_Oi_Hj=distance(p1,p3);\
                                  r_Oj_Nip2=distance(p4,p5);r_Oj_Hip2=distance(p4,p6);\
                                  r_CAim2_CAip2=distance(p7,p8);r_CAjm2_CAjp2=distance(p9,p10)"]
    #   set up nu as in the lammps version
    #   if we are on an edge or second-to-edge residue, we can't compute nu by the normal method
    #   (the CA i-2 or i+2 does not exist) and we set nu to 1
    #   we do this by constructing a conditional expression for each chain, equal to 1 when we're not either one of the first two or last two residues
    #   and equal to 0 otherwise.
    #   Then, we take the min of 1 and the sum of all these expressions to tell us if we're in the middle (not first or last 2 residues) of ANY chain.
    #   If 1, then the answer is YES; if 0, the answer is NO.
    #   these are then incorporated algebraically into our nu expression to apply the proper computation method
    #      nu_i
    nu_1_bit_list = [f"step(res_index_i-{start_res_index}-1)*step(res_index_i-{start_res_index}-1-1)*step({end_res_index}-res_index_i-1)*step({end_res_index}-1-res_index_i-1)+"\
            for start_res_index,end_res_index in zip(oa.chain_starts,oa.chain_ends)]
    nu_1_bit = "min(1,"
    for condition in nu_1_bit_list:
        nu_1_bit += condition
    nu_1_bit = nu_1_bit[:-1] # get rid of trailing +
    nu_1_bit += ")" # add ) to close the max( opened at the beginning of the statement
    nu_i = f"(0.5*(1+tanh({eta_beta_1}*(r_CAim2_CAip2-{r_HB_c})))*{nu_1_bit}+(1-{nu_1_bit}))"
    #      nu_j
    nu_1_bit_list = [f"step(res_index_j-{start_res_index}-1)*step(res_index_j-{start_res_index}-1-1)*step({end_res_index}-res_index_j-1)*step({end_res_index}-1-res_index_j-1)+"\
            for start_res_index,end_res_index in zip(oa.chain_starts,oa.chain_ends)]
    nu_1_bit = "min(1,"
    for condition in nu_1_bit_list:
        nu_1_bit += condition
    nu_1_bit = nu_1_bit[:-1] # get rid of trailing +
    nu_1_bit += ")" # add ) to close the max( opened at the beginning of the statement
    nu_j = f"(0.5*(1+tanh({eta_beta_2}*(r_CAjm2_CAjp2-{r_HB_c})))*{nu_1_bit}+(1-{nu_1_bit}))"
    #      adjusted nu_i*nu_j
    seqsep_lessthan18 = "(1-step(abs(res_index_i-res_index_j)-18))" # useful conditional to check whether our sequence separation is strictly less than 18
    #         set nu_i*nu_j to 1 if sequence separation is less than 18 and i and j are in the same chain;
    #         otherwise, pass through the unadjusted value nu_i*nu_j
    adjusted_nu = f"(samechain*({seqsep_lessthan18}+(1-{seqsep_lessthan18})*{nu_i}*{nu_j})+(1-samechain)*{nu_i}*{nu_j})" # samechain is a per-bond parameter
    #   this just truncates the potential to 0 for long range
    distance_truncation = "step(-(r_Oi_Nj-.7))"
    #
    # total energy function
    #    we make k_beta a global parameter so we can pass it in as an openmm unit object
    #    note that Lambda is a per bond parameter
    beta_term = f"-k_beta*Lambda*{theta[term_number-1]}*{adjusted_nu}*{distance_truncation};{distance_definitions[term_number-1]}"
    #print(f"beta_term: {beta_term}")
    #
    # set up openmm Force
    Beta = CustomCompoundBondForce(number_atoms[term_number-1],beta_term)
    Beta.addGlobalParameter("k_beta", k_beta)
    Beta.addPerBondParameter("Lambda")
    Beta.addPerBondParameter("res_index_i")
    Beta.addPerBondParameter("res_index_j")
    Beta.addPerBondParameter("samechain")
    for i in range(nres):
        for j in range(nres):
            # compute something needed below
            same_chain_end = [chain_end for chain_end in oa.chain_ends if inSameChain(i,chain_end, oa.chain_starts, oa.chain_ends)]
            assert(len(same_chain_end)) == 1, f"same_chain_end: {same_chain_end}, oa.chain_ends: {oa.chain_ends}, i: {i}, inSameChain(i,chain_end,oa.chain_starts,oa.chain_ends):{inSameChain(1,299,oa.chain_starts,oa.chain_ends)}"
            same_chain_end = same_chain_end[0]
            # the conditionals that guard the entire compute_dssp_hdrgn function in the lammps code
            #    these might have changed over time
            if isChainEnd(i,oa.chain_ends,n=1) or isChainStart(j,oa.chain_starts,n=1) or res_type[j] == "IPR":
                continue
            elif abs(i-j) < 4 and inSameChain(i, j, oa.chain_starts, oa.chain_ends): 
                # the guard conditional actually has a cutoff of <=2 (not <=3) in the lammps code, but the energy is always set to 0 for |i-j|=3
                # so there's no need to add a bond if |i-j|=3
                continue
            # if sequence separation is less than 18, i and j are in the same chain, and both are not designated as beta in ssweight,
            #     then we alway set the energy to 0, so we can just exclude the Bond from the Force
            elif abs(i-j) < 18 and inSameChain(i, j, oa.chain_starts, oa.chain_ends) and (rama_biases[i]=="not beta" or rama_biases[j]=="not beta"):
                continue
            # the lammps code excludes certain pairs of residues from Beta2 but not the others
            elif term_number==2 and (isChainStart(i,oa.chain_starts,n=1) or isChainEnd(j,oa.chain_ends,n=1) or res_type[i]=='IPR'):
                continue 
            elif term_number==3 and (i>same_chain_end-2 or isChainEnd(j,oa.chain_ends,n=1) or res_type[i+2]=="IPR"):
                # res_type[i+2] may not exist, but only if i>same_chain_end-2 is True, so the conditional passes without needing to compute res_type[i+2]
                continue
            # if we've made it this far, we can now set up Bonds
            else:
                # Set variables representing certain atoms in the equations (for example, ca_im2 for CA of residue i-2) 
                # to their index in the structure (for example, ca[i-2]), if it exists; otherwise (for example, the case that i==0),
                # set the variable to the index of an atom that does exist. It doesn't matter which one. The force term knows that it shouldn't
                # contibute to the potential. We just need some atom to exist at the given index to avoid an OpenMM error
                if i<2: 
                    ca_im2 = 0 # we could have chosen the index of any atom in the system
                else:
                    ca_im2 = ca[i-2]
                if j<2:
                    ca_jm2 = 0 # we could have chosen the index of any atom in the system
                else:
                    ca_jm2 = ca[j-2]
                if j>nres-3: # implies j+2>n-1, meaning that ca[j+2] doesn't exist
                    ca_jp2 = 0 # we could have chosen the index of any atom in the system
                else:
                    ca_jp2 = ca[j+2] 
                if i>nres-3: # implies i+2>n-1, meaning that ca[i+2] doesn't exist
                    ca_ip2 = 0 # we could have chosen the index of any atom in the system
                    n_ip2 = 0 # we could have chosen the index of any atom in the system
                    h_ip2 = 0 # we could have chosen the index of any atom in the system
                else:
                    ca_ip2 = ca[i+2]
                    n_ip2 = n[i+2] # needed for beta3 (but not the others)
                    h_ip2 = h[i+2] # needed for beta3 (but not the others)
                if term_number==1:
                    # make list of atoms to be included in Bond
                    bond_atoms = [o[i], n[j], h[j], ca_im2, ca_ip2, ca_jm2, ca_jp2]
                    if -1 in bond_atoms:
                        # missing atoms should be caught by earlier code in the system setup (Structure? AWSEM?), 
                        # so it's probably an issue with something in this function, not the user input
                        raise AssertionError(f"Found index of -1 in list o, n, or h! {bond_atoms}. i: {i}, j: {j}")
                    # assign Lambda, a per-bond parameter that is different for each beta term
                    Lambda = get_lambda_by_index(i, j, 0, oa.chain_starts, oa.chain_ends)
                elif term_number==2:
                    # make list of atoms to be included in Bond
                    bond_atoms = [o[i], n[j], h[j], o[j], n[i], h[i], ca_im2, ca_ip2, ca_jm2, ca_jp2]
                    if -1 in bond_atoms:
                        # missing atoms should be caught by earlier code in the system setup (Structure? AWSEM?), 
                        # so it's probably an issue with something in this function, not the user input
                        raise AssertionError(f"Found index of -1 in list o, n, or h! {bond_atoms}. i: {i}, j: {j}")
                    # assign Lambda, a per-bond parameter that is different for each beta term
                    Lambda = get_Lambda_2(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, oa.chain_starts, oa.chain_ends)
                else:
                    # check that we have a valid term_number argument
                    if not term_number==3:
                        raise ValueError(f"term_number must be 1, 2, or 3, but was {term_number}")
                    # make list of atoms to be included in Bond
                    bond_atoms = [o[i], n[j], h[j], o[j], n_ip2, h_ip2, ca_im2, ca_ip2, ca_jm2, ca_jp2]
                    if -1 in bond_atoms:
                        # missing atoms should be caught by earlier code in the system setup (Structure? AWSEM?), 
                        # so it's probably an issue with something in this function, not the user input
                        raise AssertionError(f"Found index of -1 in list o, n, or h! {bond_atoms}. i: {i}, j: {j}")
                    # assign Lambda, a per-bond parameter that is different for each beta term
                    Lambda = get_Lambda_3(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, oa.chain_starts, oa.chain_ends)
                # assign per-bond parameters
                res_index_i = i
                res_index_j = j
                samechain = int(inSameChain(res_index_i,res_index_j,oa.chain_starts,oa.chain_ends))
                # add Bond with designated atoms and per-bond parameters
                Beta.addBond(bond_atoms, [Lambda, res_index_i, res_index_j, samechain])
    Beta.setForceGroup(forceGroup)
    return Beta

def beta_term_1_old(oa, k_beta=4.184, debug=None, forceGroup=23, ssweight='ssweight'):
    """
    Wrapper that allows us to call hydrogenBondTerms.beta_term_1_old() in forces_setup.py as before.
    Debug is no longer used but is kept as a parameter in the spirit of allowing old arguments
    """
    return beta_lammps_awsemmd(oa, 1, ssweight, forceGroup, k_beta=k_beta)

def beta_term_2_old(oa, k_beta=4.184, debug=None, forceGroup=24, ssweight='ssweight'):
    """
    Wrapper that allows us to call hydrogenBondTerms.beta_term_2_old() in forces_setup.py as before.
    Debug is no longer used but is kept as a parameter in the spirit of allowing old arguments
    """
    return beta_lammps_awsemmd(oa, 2, ssweight, forceGroup, k_beta=k_beta)
    #return beta_term_2_old_reference(oa,forceGroup=24,k_beta=0.5*4.184)

def beta_term_3_old(oa, k_beta=4.184, debug=None, forceGroup=25, ssweight='ssweight'):
    """
    Wrapper that allows us to call hydrogenBondTerms.beta_term_1_old() in forces_setup.py as before.
    Debug is no longer used but is kept as a parameter in the spirit of allowing old arguments
    """
    return beta_lammps_awsemmd(oa, 3, ssweight, forceGroup, k_beta=k_beta)


def pap_term_old(oa, k_pap=4.184, forceGroup=26):
    print("pap term ON")
    nres, ca = oa.nres, oa.ca
    # r0 = 2.0 # nm
    r0 = 0.8 # nm
    eta_pap = 70 # nm^-1
    gamma_aph = 1.0
    gamma_ap = 0.4
    gamma_p = 0.4
    pap_function = f"-k_pap*gamma*0.5*(1+tanh({eta_pap}*({r0}-distance(p1,p2))))*0.5*(1+tanh({eta_pap}*({r0}-distance(p3,p4))))"
    pap = CustomCompoundBondForce(4, pap_function)
    pap.addGlobalParameter("k_pap", k_pap)
    pap.addPerBondParameter("gamma")
    #count = 0;
    for i in range(nres):
        for j in range(nres):
            # anti-parallel hairpin for i from 1 to N-13 and j from i+13 to min(i+16,N)
            # CAi CAj CAi+4 CAj-4
            # 1   2   3     4
            if i <= nres-13 and j >= i+13 and j <= min(i+16,nres):
                pap.addBond([ca[i], ca[j], ca[i+4], ca[j-4]], [gamma_aph])
                #count = count + 1
                #print([ca[i], ca[j], ca[i+4], ca[j-4]], [gamma_aph])
            # anti-parallel for i from 1 to N-17 and j from i+17 to N
            # CAi CAj CAi+4 CAj-4
            # 1   2   3     4
            if i <= nres-17 and j >= i+17 and j <= nres:
                pap.addBond([ca[i], ca[j], ca[i+4], ca[j-4]], [gamma_ap])
                #count = count + 1;
                #print([ca[i], ca[j], ca[i+4], ca[j-4]], [gamma_ap])
            # parallel for i from 1 to N-13 and j from i+9 to N-4
            # CAi CAj CAi+4 CAj+4
            # 1   2   3     4
            if i <= nres-13 and j >= i+9 and j < nres-4:
                #print([i, j, i+4, j+4])
                #print([i, j, i+4, j+4, ca[i], ca[j], ca[i+4], ca[j+4]], [gamma_p])
                pap.addBond([ca[i], ca[j], ca[i+4], ca[j+4]], [gamma_p])
                #count = count + 1;

    # print(count)
    pap.setForceGroup(forceGroup)
    return pap