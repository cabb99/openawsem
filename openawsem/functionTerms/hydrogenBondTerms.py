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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# GLOBALS

se_map_1_letter = {'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                   'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                   'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                   'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

LAMBDA_TABLE = [
    #class1, class2, class3, class4
    [0.00,   1.37,   1.36,   1.17], # lambda_0
    [0.00,   3.49,   3.50,   3.52], # lambda_1
    [0.00,   0.00,   3.47,   3.62], # lambda_2
]

ALPHA_TABLE = [
    #class1, class2, class3, class4
    [0.00,   1.30,   1.30,   1.30],  # alpha_0
    [0.00,   1.32,   1.32,   1.32],  # alpha_1
    [0.00,   1.22,   1.22,   1.22],  # alpha_2
    [0.00,   0.00,   0.33,   0.33],  # alpha_3
    [0.00,   0.00,   1.01,   1.01],  # alpha_4
]


# HELPER FUNCTIONS

def load_ssweight(ssweight_file, nres=None):
    if not os.path.exists(ssweight_file):
        print("No ssweight given, assume all zero")
        ssweight = np.zeros((nres, 2))
    else:
        ssweight = np.loadtxt(ssweight_file)    
    return ssweight

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

def find_chain_index(res: int, chain_starts, chain_ends) -> int:
    """
    Find the index of the chain that contains the residue with index `res`.
    """

    chain_index = [int(chain_start<=res and res<=chain_end) for chain_start,chain_end in zip(chain_starts,chain_ends)]
    assert sum(chain_index) == 1, f"res: {res}, chain_starts: {chain_starts}, chain_ends: {chain_ends}, list: {chain_index}"
    return chain_index.index(1)

def inSameChain(res_i: int,res_j: int,chain_starts,chain_ends) -> bool:
    """
    Return True if residues i and j lie in the same chain; False if exactly one of them 
    is out of any chain. If both are out of chain bounds, raise an AssertionError.
    
    chain_starts and chain_ends are parallel lists of the same length, where each pair
    (chain_starts[k], chain_ends[k]) defines the inclusive index range of chain k.
    """

    max_index = chain_ends[-1]
    def is_out_of_bounds(res: int) -> bool:
        return res < 0 or res > max_index

    if is_out_of_bounds(res_i) and is_out_of_bounds(res_j):
        raise AssertionError(
            f"Both residues are outside chain boundaries: i={res_i}, j={res_j}, "
            f"allowed range=[0…{max_index}]"
        )
    
    if is_out_of_bounds(res_i) or is_out_of_bounds(res_j):
        # If only one of the residues is out of bounds we'll treat them 
        # as if they were in a different chain but it shouldn't really affect anything
        return False
    
    # if we've made it this far, we know that both residues exist
    return find_chain_index(res_i, chain_starts, chain_ends) == find_chain_index(res_j, chain_starts, chain_ends)

def inWhichChain(residueId, chain_ends):
    chain_table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    for i, end_of_chain_resId in enumerate(chain_ends):
        if end_of_chain_resId < residueId:
            pass
        else:
            return chain_table[i]

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

def get_beta_class(i,j,chain_starts, chain_ends):
    """
    Return the beta class of the pair (i, j) based on the sequence separation and chain information.
    """
    same_chain = inSameChain(i,j,chain_starts,chain_ends)
    if not same_chain:
        return 3
    if abs(j-i) >= 4 and abs(j-i) < 18:
        return 1
    elif abs(j-i) >= 18 and abs(j-i) < 45:
        return 2
    elif abs(j-i) >= 45:
        return 3
    else:
        return 0 # need to do this for the regular hydrogen bond terms, while the lammps ones won't reach this else block

def get_lambda_by_index(i, j, lambda_i, chain_starts, chain_ends):
    beta_cls = get_beta_class(i, j, chain_starts, chain_ends)
    return LAMBDA_TABLE[lambda_i][beta_cls]

def get_alpha_by_index(i, j, alpha_i, chain_starts, chain_ends):
    beta_cls = get_beta_class(i, j, chain_starts, chain_ends)
    return ALPHA_TABLE[alpha_i][beta_cls]

def get_pap_gamma_APH(donor_idx, acceptor_idx, chain_i, chain_j, gamma_APH):
    # if chain_i == chain_j and abs(j-i) < 13 or abs(j-i) > 16:
    # if abs(j-i) < 13 or abs(j-i) > 16:
    # if i-j < 13 or i-j > 16:
    # if (donor_idx - acceptor_idx >= 13 and donor_idx - acceptor_idx <= 16) or chain_i != chain_j:
    if (donor_idx - acceptor_idx >= 13 and donor_idx - acceptor_idx <= 16) and chain_i == chain_j:
        return gamma_APH
    else:
        return 0

def get_pap_gamma_AP(donor_idx, acceptor_idx, chain_i, chain_j, gamma_AP, ssweight_file):
    if ssweight_file[donor_idx][1] == 1 and ssweight_file[acceptor_idx][1] == 1:
        additional_scale = 1.5
    else:
        additional_scale = 1.0
    # if (donor_idx - acceptor_idx >= 17):
    if (donor_idx - acceptor_idx >= 17) or chain_i != chain_j:
        return additional_scale * gamma_AP
    else:
        return 0

def get_pap_gamma_P(donor_idx, acceptor_idx, chain_i, chain_j, gamma_P, ssweight_file):
    if ssweight_file[donor_idx][1] == 1 and ssweight_file[acceptor_idx][1] == 1:
        additional_scale = 1.5
    else:
        additional_scale = 1.0
    if (donor_idx - acceptor_idx >= 9) or chain_i != chain_j:
        return additional_scale * gamma_P
    else:
        return 0

def get_Lambda_2(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, chain_starts, chain_ends):
    beta_class = get_beta_class(i, j, chain_starts, chain_ends)
    if beta_class >= 2:
        layer=1
    else:
        layer=0
    Lambda = get_lambda_by_index(i, j, 1, chain_starts, chain_ends)
    Lambda += 0.5*get_alpha_by_index(i, j, 0, chain_starts, chain_ends)*p_antihb[a[i], a[j]][layer]
    Lambda += 0.25*get_alpha_by_index(i, j, 1, chain_starts, chain_ends)*(p_antinhb[a[i+1], a[j-1]][layer] + p_antinhb[a[i-1], a[j+1]][layer])
    Lambda += get_alpha_by_index(i, j, 2, chain_starts, chain_ends)*(p_anti[a[i]] + p_anti[a[j]])
    return Lambda

def get_Lambda_3(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, chain_starts, chain_ends):
    beta_class = get_beta_class(i, j, chain_starts, chain_ends)
    if beta_class >= 2:
        layer=1
    else:
        layer=0
    Lambda = get_lambda_by_index(i, j, 2, chain_starts, chain_ends)
    Lambda += get_alpha_by_index(i, j, 3, chain_starts, chain_ends)*p_parhb[a[i+1], a[j]][layer]
    Lambda += get_alpha_by_index(i, j, 4, chain_starts, chain_ends)*p_par[a[i+1]]
    Lambda += get_alpha_by_index(i, j, 4, chain_starts, chain_ends)*p_par[a[j]]
    return Lambda

def convert_units(k):
    if isinstance(k, float) or isinstance(k, int):
        k = k   # just for backward comptable
    elif isinstance(k, Quantity):
        k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    else:
        print(f"Unknown input, {k}, {type(k)}")
    return k

def get_helical_f(oneLetterCode, inMembrane=False):
    if inMembrane:
        table = {"A": 0.79, "R": 0.62, "N": 0.49, "D": 0.44, "C": 0.76, "Q": 0.61, "E": 0.57, "G": 0.57, "H": 0.63, "I": 0.81,
            "L": 0.81, "K": 0.56, "M": 0.80, "F": 0.76, "P": 0.44, "S": 0.6, "T": 0.67, "W": 0.74, "Y": 0.71, "V": 0.79}
    else:
        table = {"A": 0.77, "R": 0.68, "N": 0.07, "D": 0.15, "C": 0.23, "Q": 0.33, "E": 0.27, "G": 0.0, "H": 0.06, "I": 0.23,
            "L": 0.62, "K": 0.65, "M": 0.5, "F": 0.41, "P": 0.4, "S": 0.35, "T": 0.11, "W": 0.45, "Y": 0.17, "V": 0.14}
    return table[oneLetterCode]


# MAIN API
def beta_term_1(oa, k=0.5*kilocalories_per_mole, forceGroup=27, ssweight_file='ssweight', version='efficiency_optimized', beta_nu_on=True, **kwargs):
    """
    Main API for the pairwise beta-sheet hydrogen bonding term. Defaults to the "efficiency_optimized" version, meaning the
    potential described in the OpenAWSEM paper SI, as corrected in June 2025 (see https://github.com/cabb99/openawsem/issues/52).
    This function strives to be as similar as possible to the lammps implementation, except that the nu_i*nu_j term is removed, 
    as mentioned in the SI of the OpenAWSEM paper. There may be other slight differences arising from the constraints imposed 
    on the functional form by the CustomHbondForce class, but I am not aware of any at the moment. 

    Alternatively, we can use the "lammps_awsemmd" version, which implements the potential from a particular LAMMPS AWSEM-MD commit,
    https://github.com/adavtyan/awsemmd/tree/cea754f1208fde6332d4d0f1cae3212bf7e8afbb, within a tolerance of 0.01 kcal/mol 
    on all tested systems and computational Platforms. The non-hardcoded parameters required for the LAMMPS energy calculation
    may be found in tests/data/test_implementation_of_lammps_hbond_energies/lammps_setup_parameters.   
    It should be noted that not all LAMMPS AWSEM-MD versions are identical.


    The OpenAWSEM paper:
    Lu, W.; Bueno, C.; Schafer, N. P.; Moller, J.; Jin, S.; Chen, X.; Chen, M.; Gu, X.; 
        Davtyan, A.; de Pablo, J. J.; Wolynes, P. G. OpenAWSEM with Open3SPN2: 
        A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations.
        PLoS Comput. Biol. 2021, 17, 2, e1008308.
    
    The LAMMPS AWSEM-MD paper:
    Davtyan, A.; Schafer, N. P.; Zheng, W.; Clementi, C.; Wolynes, P. G.; Papoian, G. A. 
        AWSEM-MD: Protein Structure Prediction Using Coarse-Grained Physical Potentials and Bioinformatically Based Local Structure Biasing. 
        J. Phys. Chem. B 2012, 116, 29, 8494-8503.
    """
    if "ssweight" in kwargs:
        warnings.warn(
            "beta_term_2: `ssweight` is deprecated; use "
            "`ssweight_file` instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        ssweight_file = kwargs.pop("ssweight")
    
    if kwargs:
        # any other unexpected kwargs?
        unexpected = ", ".join(kwargs)
        raise TypeError(f"beta_term_2() got unexpected keyword argument(s): {unexpected}")
    
    print(f"beta_term_1 ({version} version) on")
    if version == 'lammps_awsemmd':
        return _beta_lammps_awsemmd(oa, 1, ssweight_file, forceGroup, k, beta_nu_on)
    elif version == 'efficiency_optimized':
        return _beta_efficiency_optimized(oa, 1, ssweight_file, forceGroup, k)
    else:
        raise ValueError(f"version must be 'efficiency_optimized' or 'lammps_awsemmd', but was {version}")

def beta_term_2(oa, k=0.5*kilocalories_per_mole, forceGroup=27, ssweight_file='ssweight', version='efficiency_optimized', beta_nu_on=True, **kwargs):
    """
    Main API for the antiparallel cooperative beta-sheet hydrogen bonding term. Defaults to the "efficiency_optimized" version, meaning the
    potential described in the OpenAWSEM paper SI, as corrected in June 2025 (see https://github.com/cabb99/openawsem/issues/52).
    This function strives to be as similar as possible to the lammps implementation, except that the nu_i*nu_j term is removed, 
    as mentioned in the SI of the OpenAWSEM paper. There may be other slight differences arising from the constraints imposed 
    on the functional form by the CustomHbondForce class, but I am not aware of any at the moment. 

    Alternatively, we can use the "lammps_awsemmd" version, which implements the potential from a particular LAMMPS AWSEM-MD commit,
    https://github.com/adavtyan/awsemmd/tree/cea754f1208fde6332d4d0f1cae3212bf7e8afbb, within a tolerance of 0.01 kcal/mol 
    on all tested systems and computational Platforms. The non-hardcoded parameters required for the LAMMPS energy calculation
    may be found in tests/data/test_implementation_of_lammps_hbond_energies/lammps_setup_parameters.   
    It should be noted that not all LAMMPS AWSEM-MD versions are identical.

    The OpenAWSEM paper:
    Lu, W.; Bueno, C.; Schafer, N. P.; Moller, J.; Jin, S.; Chen, X.; Chen, M.; Gu, X.; 
        Davtyan, A.; de Pablo, J. J.; Wolynes, P. G. OpenAWSEM with Open3SPN2: 
        A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations.
        PLoS Comput. Biol. 2021, 17, 2, e1008308.
    
    The LAMMPS AWSEM-MD paper:
    Davtyan, A.; Schafer, N. P.; Zheng, W.; Clementi, C.; Wolynes, P. G.; Papoian, G. A. 
        AWSEM-MD: Protein Structure Prediction Using Coarse-Grained Physical Potentials and Bioinformatically Based Local Structure Biasing. 
        J. Phys. Chem. B 2012, 116, 29, 8494-8503.
    """
    if "ssweight" in kwargs:
        warnings.warn(
            "beta_term_2: `ssweight` is deprecated; use "
            "`ssweight_file` instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        ssweight_file = kwargs.pop("ssweight")
    
    if kwargs:
        # any other unexpected kwargs?
        unexpected = ", ".join(kwargs)
        raise TypeError(f"beta_term_2() got unexpected keyword argument(s): {unexpected}")
    
    print(f"beta_term_2 ({version} version) on")
    if version == 'lammps_awsemmd':
        return _beta_lammps_awsemmd(oa, 2, ssweight_file, forceGroup, k, beta_nu_on)
    elif version == 'efficiency_optimized':
        return _beta_efficiency_optimized(oa, 2, ssweight_file, forceGroup, k)
    else:
        raise ValueError(f"version must be 'efficiency_optimized' or 'lammps_awsemmd', but was {version}")

def beta_term_3(oa, k=0.5*kilocalories_per_mole, forceGroup=27, ssweight_file='ssweight', version='efficiency_optimized', beta_nu_on=True, **kwargs):
    """
    Main API for the parallel cooperative beta-sheet hydrogen bonding term. Defaults to the "efficiency_optimized" version, meaning the
    potential described in the OpenAWSEM paper SI, as corrected in June 2025 (see https://github.com/cabb99/openawsem/issues/52).
    This function strives to be as similar as possible to the lammps implementation, except that the nu_i*nu_j term is removed, 
    as mentioned in the SI of the OpenAWSEM paper. There may be other slight differences arising from the constraints imposed 
    on the functional form by the CustomHbondForce class, but I am not aware of any at the moment. 

    Alternatively, we can use the "lammps_awsemmd" version, which implements the potential from a particular LAMMPS AWSEM-MD commit,
    https://github.com/adavtyan/awsemmd/tree/cea754f1208fde6332d4d0f1cae3212bf7e8afbb, within a tolerance of 0.01 kcal/mol 
    on all tested systems and computational Platforms. The non-hardcoded parameters required for the LAMMPS energy calculation
    may be found in tests/data/test_implementation_of_lammps_hbond_energies/lammps_setup_parameters.   
    It should be noted that not all LAMMPS AWSEM-MD versions are identical.

    The OpenAWSEM paper:
    Lu, W.; Bueno, C.; Schafer, N. P.; Moller, J.; Jin, S.; Chen, X.; Chen, M.; Gu, X.; 
        Davtyan, A.; de Pablo, J. J.; Wolynes, P. G. OpenAWSEM with Open3SPN2: 
        A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations.
        PLoS Comput. Biol. 2021, 17, 2, e1008308.
    
    The LAMMPS AWSEM-MD paper:
    Davtyan, A.; Schafer, N. P.; Zheng, W.; Clementi, C.; Wolynes, P. G.; Papoian, G. A. 
        AWSEM-MD: Protein Structure Prediction Using Coarse-Grained Physical Potentials and Bioinformatically Based Local Structure Biasing. 
        J. Phys. Chem. B 2012, 116, 29, 8494-8503.
    """
    if "ssweight" in kwargs:
        warnings.warn(
            "beta_term_3: `ssweight` is deprecated; use "
            "`ssweight_file` instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        ssweight_file = kwargs.pop("ssweight")
    
    if kwargs:
        # any other unexpected kwargs?
        unexpected = ", ".join(kwargs)
        raise TypeError(f"beta_term_3() got unexpected keyword argument(s): {unexpected}")
    
    print(f"beta_term_3 ({version} version) on")
    if version == 'lammps_awsemmd':
        return _beta_lammps_awsemmd(oa, 3, ssweight_file, forceGroup, k, beta_nu_on)
    elif version == 'efficiency_optimized':
        return _beta_efficiency_optimized(oa, 3, ssweight_file, forceGroup, k)
    else:
        raise ValueError(f"version must be 'efficiency_optimized' or 'lammps_awsemmd', but was {version}")

def beta_term_1_old(oa, k_beta=4.184, debug=None, forceGroup=23, ssweight_file='ssweight', beta_nu_on=True):
    """
    Wrapper that allows us to call hydrogenBondTerms.beta_term_1_old() in forces_setup.py as before.
    Debug is no longer used but is kept as a parameter in the spirit of allowing old arguments
    """
    return beta_term_1(oa, k_beta, forceGroup, ssweight_file, 'lammps_awsemmd', beta_nu_on)

def beta_term_2_old(oa, k_beta=4.184, debug=None, forceGroup=24, ssweight_file='ssweight', beta_nu_on=True):
    """
    Wrapper that allows us to call hydrogenBondTerms.beta_term_2_old() in forces_setup.py as before.
    Debug is no longer used but is kept as a parameter in the spirit of allowing old arguments
    """
    return beta_term_2(oa, k_beta, forceGroup, ssweight_file, 'lammps_awsemmd', beta_nu_on)

def beta_term_3_old(oa, k_beta=4.184, debug=None, forceGroup=25, ssweight_file='ssweight', beta_nu_on=True):
    """
    Wrapper that allows us to call hydrogenBondTerms.beta_term_1_old() in forces_setup.py as before.
    Debug is no longer used but is kept as a parameter in the spirit of allowing old arguments
    """
    return beta_term_3(oa, k_beta, forceGroup, ssweight_file, 'lammps_awsemmd', beta_nu_on)

def pap_term_1(oa, k=0.5*kilocalories_per_mole, dis_i_to_i4=1.2, forceGroup=28, ssweight_file="ssweight", 
               version='efficiency_optimized', pap_nu_on=True, **kwargs):
    """
    Main API for the antiparallel cooperative liquid crystal beta-sheet (P_AP) hydrogen bonding term. 
    Defaults to the "efficiency_optimized" version, meaning the potential described in the OpenAWSEM paper,
    although I think that the code does not actually do what the paper says it should (see https://github.com/cabb99/openawsem/issues/60).
    In the absence of an efficient solution (https://github.com/openmm/openmm/issues/2565 not yet implemented), we leave it as is.

    Alternatively, we can use the "lammps_awsemmd" version, which implements the potential from a particular LAMMPS AWSEM-MD commit,
    https://github.com/adavtyan/awsemmd/tree/cea754f1208fde6332d4d0f1cae3212bf7e8afbb, within a tolerance of 0.01 kcal/mol 
    on all tested systems and computational Platforms. The non-hardcoded parameters required for the LAMMPS energy calculation
    may be found in tests/data/test_implementation_of_lammps_hbond_energies/lammps_setup_parameters.   
    It should be noted that not all LAMMPS AWSEM-MD versions are identical.

    I believe using pap_nu_on=False with version='efficiency_optimized' is equivalent to but maybe more efficient than 
    using version='lammps_awsemmd' (pap_nu_on is irrelevant in this case). I'll check this soon.

    The rigorously tested LAMMPS AWSEM-MD potential is implemented as a single term that should be accessed using pap_term_old.

    The OpenAWSEM paper:
    Lu, W.; Bueno, C.; Schafer, N. P.; Moller, J.; Jin, S.; Chen, X.; Chen, M.; Gu, X.; 
        Davtyan, A.; de Pablo, J. J.; Wolynes, P. G. OpenAWSEM with Open3SPN2: 
        A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations.
        PLoS Comput. Biol. 2021, 17, 2, e1008308.
    
    The LAMMPS AWSEM-MD paper:
    Davtyan, A.; Schafer, N. P.; Zheng, W.; Clementi, C.; Wolynes, P. G.; Papoian, G. A. 
        AWSEM-MD: Protein Structure Prediction Using Coarse-Grained Physical Potentials and Bioinformatically Based Local Structure Biasing. 
        J. Phys. Chem. B 2012, 116, 29, 8494-8503.
    """

    if "ssweightFileName" in kwargs:
        warnings.warn(
            "pap_term_1: `ssweightFileName` is deprecated; use "
            "`ssweight_file` instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        ssweight_file = kwargs.pop("ssweightFileName")
    
    if kwargs:
        # any other unexpected kwargs?
        unexpected = ", ".join(kwargs)
        raise TypeError(f"pap_term_1() got unexpected keyword argument(s): {unexpected}")
    
    if version == 'lammps_awsemmd':
        warnings.warn("lammps_awsemmd implements both pap_term_1 and pap_term_2 as a single term, pap_term_old().\
               Calling pap_term_old() instead and assigning to forceGroup 26.",stacklevel=2)
        return pap_term_old(oa, k_pap=k, ssweight_file=ssweight_file)
    elif version == 'efficiency_optimized':
        print(f"pap_term_1 ({version} version) on")
        return _pap_efficiency_optimized(oa, 1, ssweight_file, forceGroup, k, dis_i_to_i4, pap_nu_on)
    else:
        raise ValueError(f"version must be 'efficiency_optimized' or 'lammps_awsemmd', but was {version}")

def pap_term_2(oa, k=0.5*kilocalories_per_mole, dis_i_to_i4=1.2, forceGroup=28, ssweight_file="ssweight", ssweightFileName="ssweight",
                   version='efficiency_optimized', pap_nu_on=True, **kwargs):
    """
    Main API for the parallel cooperative liquid crystal beta-sheet (P_AP) hydrogen bonding term. 
    Defaults to the "efficiency_optimized" version, meaning the potential described in the OpenAWSEM paper,
    although I think that the code does not actually do what the paper says it should (see https://github.com/cabb99/openawsem/issues/60).
    In the absence of an efficient solution (https://github.com/openmm/openmm/issues/2565 not yet implemented), we leave it as is.

    Alternatively, we can use the "lammps_awsemmd" version, which implements the potential from a particular LAMMPS AWSEM-MD commit,
    https://github.com/adavtyan/awsemmd/tree/cea754f1208fde6332d4d0f1cae3212bf7e8afbb, within a tolerance of 0.01 kcal/mol 
    on all tested systems and computational Platforms. The non-hardcoded parameters required for the LAMMPS energy calculation
    may be found in tests/data/test_implementation_of_lammps_hbond_energies/lammps_setup_parameters.   
    It should be noted that not all LAMMPS AWSEM-MD versions are identical.

    I believe using pap_nu_on=False with version='efficiency_optimized' is equivalent to but maybe more efficient than 
    using version='lammps_awsemmd' (pap_nu_on is irrelevant in this case). I'll check this soon.

    The rigorously tested LAMMPS AWSEM-MD potential is implemented as a single term that should be accessed using pap_term_old.

    The OpenAWSEM paper:
    Lu, W.; Bueno, C.; Schafer, N. P.; Moller, J.; Jin, S.; Chen, X.; Chen, M.; Gu, X.; 
        Davtyan, A.; de Pablo, J. J.; Wolynes, P. G. OpenAWSEM with Open3SPN2: 
        A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations.
        PLoS Comput. Biol. 2021, 17, 2, e1008308.
    
    The LAMMPS AWSEM-MD paper:
    Davtyan, A.; Schafer, N. P.; Zheng, W.; Clementi, C.; Wolynes, P. G.; Papoian, G. A. 
        AWSEM-MD: Protein Structure Prediction Using Coarse-Grained Physical Potentials and Bioinformatically Based Local Structure Biasing. 
        J. Phys. Chem. B 2012, 116, 29, 8494-8503.
    """
    if "ssweightFileName" in kwargs:
        warnings.warn(
            "pap_term_2: `ssweightFileName` is deprecated; use "
            "`ssweight_file` instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        ssweight_file = kwargs.pop("ssweightFileName")
    
    if kwargs:
        # any other unexpected kwargs?
        unexpected = ", ".join(kwargs)
        raise TypeError(f"pap_term_2() got unexpected keyword argument(s): {unexpected}")
    

    if version == 'lammps_awsemmd':
        warnings.warn("lammps_awsemmd implements both pap_term_1 and pap_term_2 as a single term, pap_term_old(). Returning dummy Force.",stacklevel=2)
        return CustomExternalForce("0") # force has 0 energy and no particles
    elif version == 'efficiency_optimized':
        print(f"pap_term_2 ({version} version) on")
        return _pap_efficiency_optimized(oa, 2, ssweight_file, forceGroup, k, dis_i_to_i4, pap_nu_on)
    else:
        raise ValueError(f"version must be 'efficiency_optimized' or 'lammps_awsemmd', but was {version}")  

def pap_term_old(oa, k_pap=4.184, forceGroup=26, ssweight_file="ssweight", enable_antiparallel=True, enable_parallel=True):
    """
    Wrapper that allows us to call hydrogenBondTerms.pap_term_old() in forces_setup.py as before.
    """
    return _pap_lammps_awsemmd(oa, ssweight_file, forceGroup, k_pap, enable_antiparallel, enable_parallel)

def helical_term(oa, k_helical=4.184, inMembrane=False, forceGroup=29):
    """
    We can think of this as the "default" or "efficiency-optimized" helical term
    written by Lu et al. around 2020.
    
    This term is not exactly the same as the LAMMPS AWSEM-MD helical term.
    The major difference is that this term doesn't consider the sigma_helix (analogous to
    sigma_wat but possibly with different parameters) of the pair of residues
    involved in the hydrogen bonding interaction (see density_dependent_helical_term
    for explanation).

    Another difference is that, since we don't have a virtual particle for the proline 
    backbone H (which doesn't exist in reality), it can't act as a donor. 
    Here, we simply turn off the potential in the case that proline is in the donor position.

    In LAMMPS, proline in the donor (i+4) position is allowed to participate in an hbond interaction,
    but the default proline weight makes the interaction strongly unfavorable. 
    Having a hydrogen bond with proline in the acceptor (i) position can then be made
    more favorable by the pro_accepter_flag argument and passing a more favorable
    weight than the normal unfavorable weight. For example:
    [Helix]
    0.5
    2.0 -1.0
    7.0 7.0
    3.0
    4
    15.0
    4.5 6.5
    0.77 0.68 0.07 0.15 0.23 0.33 0.27 0.0 0.06 0.23 0.62 0.65 0.50 0.41 -3.0 0.35 0.11 0.45 0.17 0.14
    0 -3.0     <-- 0 is pro_accepter_flag, so proline will get the default weight of -3.0 for both i and i+4 positions
                   and even if we had set it to 1, the second number, -3.0, is the same as the default, so it
                   wouldn't have mattered. But we could change 0 -> 1 and -3.0 -> something else           
    0.76 0.68
    2.1558 2.9862
    """
    # some useful parameters
    k_helical *= oa.k_awsem
    sigma_NO = 0.068
    sigma_HO = 0.076
    r_ON = 0.298 # mingchen/weihua used a slightly different number (0.21558 nm)
    r_OH = 0.206 # mingchen/weihua used a slightly different number (0.29862 nm)

    # theta_ij checks whether our O_i, N_i+4, and H_i+4 are in an H-bond geometry
    theta_ij = f"exp(-(r_Oi_Nip4-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hip4-{r_OH})^2/(2*{sigma_HO}^2))"

    #declare 3-body Force
    helical = CustomCompoundBondForce(3, f"-{k_helical}*(fa_i+fa_ip4)*{theta_ij};\
                                        r_Oi_Nip4=distance(p1,p2);r_Oi_Hip4=distance(p1,p3);")
    
    # add per-amino-acid helical propensities
    helical.addPerBondParameter("fa_i")
    helical.addPerBondParameter("fa_ip4")

    # add a Bond to the Force for each (i, i+4) pair
    for i in range(oa.nres):
        if not isChainEnd(i, oa.chain_ends, n=4) and oa.res_type[i+4] != "IPR":
            fa_i = get_helical_f(oa.seq[i], inMembrane=inMembrane)
            fa_ip4 = get_helical_f(oa.seq[i+4], inMembrane=inMembrane)
            helical.addBond([oa.o[i], oa.n[i+4], oa.h[i+4]], [fa_i, fa_ip4])

    # finalize setup and return Force
    helical.setForceGroup(forceGroup)
    return helical

def density_dependent_helical_term(oa, k_helical=4.184, inMembrane=False, forceGroup=29):
    # we avoid get_helical_f here so that we can reproduce old work
    helical_gamma_i=[0.77,0.68,0.07,0.15,0.23,0.33,0.27,0.0,0.06,0.23,0.62,0.65,0.50,0.41,-3.0,0.35,0.11,0.45,0.17,0.14]
    arnd = 'ARNDCQEGHILKMFPSTWYV'
    """
    WARNING: THIS TERM IS EXPECTED TO BE VERY SLOW AND/OR RAM-INTENSIVE
    FOR ALL BUT THE SMALLEST SYSTEMS.
    SEE density_dependent_helical_term_approximate FOR AN ALTERNATIVE.

    This is the exact density-dependent helical term calculated by lammps commit
    https://github.com/adavtyan/awsemmd/tree/cea754f1208fde6332d4d0f1cae3212bf7e8afbb
    using the following vitural site and helical parameters:
    [ABC]
    0.4831806 0.703282 -0.1864262
    0.4436538 0.2352006 0.3211455
    0.841 0.89296 -0.73389
    [Helix]
    0.5
    2.0 -1.0
    7.0 7.0
    3.0
    4
    15.0
    4.5 6.5
    0.77 0.68 0.07 0.15 0.23 0.33 0.27 0.0 0.06 0.23 0.62 0.65 0.50 0.41 -3.0 0.35 0.11 0.45 0.17 0.14
    0 -3.0
    0.76 0.68
    2.1558 2.9862

    This term is similar to helical_term, except it includes the sigma_helix factor in the energy expression
    and creates a hydrogen pseudoatom on the proline N so it can be treated like the other residues.
    
    sigma_helix is analogous to sigma_water but its default parameters are different. 
    Both sigma_helix and sigma_water assign a pair of residues as either exposed or buried
    based on the rho values of the two residues in the pair. The "sigma_helix factor"
    linearly interpolates between +2 and -1 as sigma_helix goes from 0 (buried residues)
    to 1 (exposed residues), reflecting that helices are somewhat less stable in 
    solvent-exposed environments than in buried environments.

    The modern density-dependent helical potential was introduced in 
    2010 (doi.org/10.1016/j.ymeth.2010.05.006). As noted in this paper, the roots of
    the helical potential go back to 1999 (doi.org/10.1063/1.479101).

    Strangely, the helical conformation actually becomes favorable for proline in a 
    solvent-exposed environment. This is because the sum of the default proline
    helical propensity (-3.0) with any other residue's helical propensity (>= 0)
    is a fairly big negative number which, when multiplied by the default weight
    for the water environment of -1.0, and then by the coefficient of -k_helical,
    gives a negative (favorable) potential. It would probably make more sense to 
    just exclude pairs (i,i+4) with PRO at i+4 from the potential.
    """
    if inMembrane:
        raise NotImplementedError("inMembrane option=True not supported for density_dependent_helical_term")

    # some useful parameters
    k_helical *= oa.k_awsem
    sigma_NO = 0.068
    sigma_HO = 0.076
    r_ON = 0.29862#0.298 # mingchen/weihua used 2.9862, at least for the helical hbond sigma
    r_OH = 0.21558#0.206 # mingchen/weihua used 2.1558, at least for the helical hbond sigma
    # parameters for the sigma involved in the helical hbonds
    helix_eta = 7 # CAREFUL: THIS NUMBER HAS UNITS OF 1/ANGSTROM, SO WE MULTIPLY THE DISTANCES (IN nm) BY 10 (SEE BELOW)
    helix_eta_sigma = 7.0
    helix_rho_0 = 3.0

    # interpolates between favorable and unfavorable weights for burial and exposed pairs, respectively
    sigma_helix_term = '(2-3*sigma_helix)' # equal to (helix_gamma_p*(1-sigma_helix)+helix_gamma_w*sigma_helix), where helix_gamma_p=2, helix_gamma_w=-1

    # characterizes the compatibility of O_i, N_i+4, and H_i+4 with hydrogen bonding
    theta_helix = f"exp(-(distance(p1,p2)-{r_ON})^2/(2*{sigma_NO}^2)-(pointdistance(x1,y1,z1,xh,yh,zh)-{r_OH})^2/(2*{sigma_HO}^2))"

    # Create rho expressions to help us select the needed rho value for each residue.
    #
    #     This expression is the sum of the rho value of each residue multiplied by
    #     a delta function that depends on the PerBondParameter i. Each Bond sets
    #     i to the 0-indexed position of the acceptor or donor residue 
    #     so that the density of that residue can be isolated. 
    rho_i = "(0"
    for i in range(oa.nres):
        rho_i += f"+delta({i}-i)*(0" # multiply delta function by density of residue i
        for j in range(oa.nres): # create string representing density of residue i
            if abs(j-i) >= 2: # get contribution of residue j to the rho of residue i
                r_ij = f'10*distance(p{i+6},p{j+6})'
                rho_i += f"+0.25*(1+tanh({helix_eta}*({r_ij}-4.5)))*(1+tanh({helix_eta}*(6.5-{r_ij})))"
            else:
                continue # we would have violated min seq sep rho of 2 if we included this residue in the rho calculation
        rho_i += ')'
    rho_i += ')'
    rho_ip4 = "(0"
    for i in range(4,oa.nres):
        rho_ip4 += f"+delta({i-4}-i)*(0"
        for j in range(oa.nres):
            if abs(j-i) >= 2:
                r_ip4j = f'10*distance(p{i+6},p{j+6})'
                rho_ip4 += f"+0.25*(1+tanh({helix_eta}*({r_ip4j}-4.5)))*(1+tanh({helix_eta}*(6.5-{r_ip4j})))"
        rho_ip4 += ')'
    rho_ip4 += ')'
 
    # get hydrogen pseudoatom coordinates (this allows us to treat IPR (proline) like the other residues)
    definitions = f';xh=0.841*x4+0.89296*x5-0.73389*x3;yh=0.841*y4+0.89296*y5-0.73389*y3;zh=0.841*z4+0.89296*z5-0.73389*z3\
        ;sigma_helix=0.25*(1-tanh({helix_eta_sigma}*({rho_i}-{helix_rho_0})))*(1-tanh({helix_eta_sigma}*({rho_ip4}-{helix_rho_0})))'
    energy_string = f'-{k_helical}*{sigma_helix_term}*(helical_propensities(i)+helical_propensities(i+4))*{theta_helix}{definitions}' 

    # set up Force
    HB = CustomCompoundBondForce(6+oa.nres-1, energy_string) # see below for how we calculate the number of atoms needed
    HB.addPerBondParameter("i") #                note that we don't use get_helical_f here because it has a different proline parameter
    HB.addTabulatedFunction('helical_propensities', Discrete1DFunction([helical_gamma_i[arnd.index(one_letter_code)] for one_letter_code in oa.seq]))

    # get particle indices and set up each Bond
    # index 1: Oi    (needed to calculate geometric compatibility with H bond)
    # index 2: Nip4  (needed to calculate geometric compatibility with H bond)
    # index 3: Oip3  (needed to calculate position of hip4, used to calculate geometric compatibility with H bond)
    # index 4: CAip3 (needed to calculate position of hip4)
    # index 5: CAip4 (needed to calculate position of hip4)
    # indices 6 to 6+N-1: CB atoms, from residue 1 to N (needed to calculate rho);
    for i in range(oa.nres-4):
        if inSameChain(i, i+4, oa.chain_starts, oa.chain_ends):
            # we should add a bond
            particles = [oa.o[i], oa.n[i+4], oa.o[i+3], oa.ca[i+3], oa.ca[i+4]]
            cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
            particles += cb_fixed
            assert len(particles) == oa.nres+5, particles
            HB.addBond(particles, [i])  # perbondparameter "i" indicates the 0-indexed residue index of the acceptor
    HB.setForceGroup(forceGroup)
    return HB

def density_dependent_helical_term_approximate(oa, k_helical=4.184, inMembrane=False, forceGroup=29):
    # we avoid get_helical_f here so that we can reproduce old work                        -3.0
    helical_gamma_i=[0.77,0.68,0.07,0.15,0.23,0.33,0.27,0.0,0.06,0.23,0.62,0.65,0.50,0.41,0.0,0.35,0.11,0.45,0.17,0.14]
    arnd = 'ARNDCQEGHILKMFPSTWYV'
    # just do this for now until we decide how to implement membrane parameters
    if inMembrane:
        raise NotImplementedError("still figuring out how density-dependent helical term should mix with membrane gammas")
    """ 
    Efficient approximation to density_dependent_helical_term with density dependence.
    The approximation lies in the use of combined attractive and repulsive PAIRWISE interactions
    to give an effectively directional hydrogen bond.
    """
    # the general implementation strategy here is to use a CustomCompoundBondForce
    # to get rho just like we do for the contact term, then add pairwise energy terms
    # to the Force to implement the attractive and repulsive interactions ONLY
    # between residues i and i+4. Since pairs (i,i+4) are excluded from the Beta
    # and P-AP hydrogen bonding terms, there shouldn't be any side effects of 
    # changing the form of the interaction; we just want to get as close as 
    # reasonably possible to the lammps version.

    # some useful parameters
    k_helical *= oa.k_awsem
    sigma_NO = 0.068
    sigma_HO = 0.076
    r_ON = 0.29862#0.298 # mingchen/weihua used 2.9862, at least for the helical hbond sigma
    r_OH = 0.21558#0.206 # mingchen/weihua used 2.1558, at least for the helical hbond sigma
    # parameters for the sigma involved in the helical hbonds
    helix_eta = 7 # CAREFUL: THIS NUMBER HAS UNITS OF 1/ANGSTROM, SO WE MULTIPLY THE DISTANCES (IN nm) BY 10 (SEE BELOW)
    helix_eta_sigma = 7.0
    helix_rho_0 = 3.0

    # set up Force
    HB = CustomGBForce()
    HB.addPerParticleParameter("resId")
    HB.addPerParticleParameter("isCb")
    HB.addPerParticleParameter("isN") 
    HB.addPerParticleParameter("isO")
    HB.addPerParticleParameter("isCa")
    HB.addPerParticleParameter("isCp") # C prime (backbone carbonyl carbon)
    HB.addTabulatedFunction('helical_propensities', Discrete1DFunction([helical_gamma_i[arnd.index(one_letter_code)] for one_letter_code in oa.seq]))
    # HB.addTabulatedFunction('helical_propensities', Discrete1DFunction([get_helical_f(oa.seq[i], inMembrane=inMembrane) for one_letter_code in oa.seq]))
    #     should consider contributes for residue with sequence separation of less that 2 (meaning 1) if 
    #     the two residues are in different chains
    HB.addComputedValue("rho", f"isCb1*isCb2*step(abs(resId1-resId2)-2)*0.25*(1+tanh({helix_eta}*(10*r-4.5)))*(1+tanh({helix_eta}*(6.5-10*r)))", CustomGBForce.ParticlePair)

    # add particles to Force
    #     replace cb with ca for GLY
    cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
    none_cb_fixed = [i for i in range(oa.natoms) if i not in cb_fixed]
    assert len(cb_fixed) == oa.nres, f"Number of atoms in cb_fixed (non-GLY CB and GLY CA atoms), {len(cb_fixed)}, does not match number of residues {oa.nres}."
    for i in range(oa.natoms):
        HB.addParticle([oa.resi[i], int(i in cb_fixed), int(i in oa.n), int(i in oa.o), int(i in oa.ca), int(i in oa.c)]) 

    # define energy
    #definitions =    f';sigma_helix_term=2\
    definitions = f';sigma_helix_term=(2-3*sigma_helix)\
        ;sigma_helix=0.25*(1-tanh({helix_eta_sigma}*(rho1-{helix_rho_0})))*(1-tanh({helix_eta_sigma}*(rho2-{helix_rho_0})))\
        ;propensity=(helical_propensities(resId1)+helical_propensities(resId2))\
        ;LJ=(-isO1*isN2*tanh(0.03*((2/(10*r-3.43))^2))+(isCp1*isCp2+isCp1*isCa2+isCa1*isN2+isN1*isN2)*tanh(0.03*((2/(10*r-2))^2)))'
        #;LJ=(-isO1*isN2*tanh(0.03*((2/(10*r-3.43))^2)))'
        #;LJ=(-isO1*isN2+isCp1*isCp2+isCp1*isCa2)*tanh(0.03*((2/(10*r-3.43))^2))'
        #;LJ=isO1*isN2*(5*((2.0/(10*r-1.43))^(12))-6*((2.0/(10*r-1.43))^(10)))' # converting r from nm to Angstrom to match units of constants
    # no leading negative sign in energy string because LJ term is sort of like a negative helix theta
    energy_string = f"{k_helical}*delta((resId2-resId1)-4)*sigma_helix_term*propensity*LJ{definitions}"
    HB.addEnergyTerm(energy_string, CustomGBForce.ParticlePair)

    # finalize and return Force
    #     cutoff shouldn't matter because i and i+4 should always be close in space
    if oa.periodic_box:
        HB.setNonbondedMethod(HB.CutoffPeriodic)
    else:
        HB.setNonbondedMethod(HB.CutoffNonPeriodic)
    HB.setForceGroup(forceGroup)
    return HB

def sequence_independent_helical_term(oa, k_helical=4.184, inMembrane=False, forceGroup=29):
    """
    Experimental sequence-independent helical term for folding designed sequences
    """
    # see helical_term for helpful comments on parameters and code in this function
    k_helical *= oa.k_awsem
    sigma_NO = 0.068
    sigma_HO = 0.076
    r_ON = 0.298
    r_OH = 0.206

    theta_ij = f"exp(-(r_Oi_Nip4-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hip4-{r_OH})^2/(2*{sigma_HO}^2))"
    helical = CustomCompoundBondForce(3, f"-{k_helical}*(1)*{theta_ij};\
                                        r_Oi_Nip4=distance(p1,p2);r_Oi_Hip4=distance(p1,p3);")
    for i in range(oa.nres):
        if not isChainEnd(i, oa.chain_ends, n=4) and oa.res_type[i+4] != "IPR":
            helical.addBond([oa.o[i], oa.n[i+4], oa.h[i+4]])

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


# MAIN LOGIC FOR THE BETA SHEET AND LIQUID CRYSTAL (P_AP) TERMS
#     For clarity and backwards compatibility, the user must be allowed to access these terms in multiple ways,
#     So we provide those interfaces in the main API, then they all call one of these functions

def _beta_lammps_awsemmd(oa, term_number, ssweight_file, forceGroup, k_beta, beta_nu_on):
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
    rama_biases = load_ssweight(ssweight_file, nres)
    #
    # load parameters
    a = [] # list to help us to convert from amino acid type to the appropriate row/column indices in p_par, p_anti, etc.
    for ii in range(oa.nres):
        a.append(se_map_1_letter[oa.seq[ii]])
    p_par, p_anti, p_antihb, p_antinhb, p_parhb = read_beta_parameters()
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
    if beta_nu_on:
        nu_i = f"(0.5*(1+tanh({eta_beta_1}*(r_CAim2_CAip2-{r_HB_c})))*{nu_1_bit}+(1-{nu_1_bit}))"
    else:
        nu_i = "1"
    #      nu_j
    nu_1_bit_list = [f"step(res_index_j-{start_res_index}-1)*step(res_index_j-{start_res_index}-1-1)*step({end_res_index}-res_index_j-1)*step({end_res_index}-1-res_index_j-1)+"\
            for start_res_index,end_res_index in zip(oa.chain_starts,oa.chain_ends)]
    nu_1_bit = "min(1,"
    for condition in nu_1_bit_list:
        nu_1_bit += condition
    nu_1_bit = nu_1_bit[:-1] # get rid of trailing +
    nu_1_bit += ")" # add ) to close the max( opened at the beginning of the statement
    if beta_nu_on:
        nu_j = f"(0.5*(1+tanh({eta_beta_2}*(r_CAjm2_CAjp2-{r_HB_c})))*{nu_1_bit}+(1-{nu_1_bit}))"
    else:
        nu_j = "1"
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
    #        also, global parameters can be modified during a simulation, but I don't think we want to do that
    #        so just leaving k_beta as a global parameter for back compatibility basically
    #    note that Lambda is a per bond parameter
    beta_term = f"-k_beta*Lambda*{theta[term_number-1]}*{adjusted_nu}*{distance_truncation};{distance_definitions[term_number-1]}"
    print(f"beta_term: {beta_term}")
    #
    # set up openmm Force
    Beta = CustomCompoundBondForce(number_atoms[term_number-1],beta_term)
    if oa.periodic_box:
        Beta.setUsesPeriodicBoundaryConditions(True)
        print(f'\nbeta_term_{term_number} is in PBC')
    else:
        Beta.setUsesPeriodicBoundaryConditions(False)
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
            #     then we always set the energy to 0, so we can just exclude the Bond from the Force
            elif abs(i-j) < 18 and inSameChain(i, j, oa.chain_starts, oa.chain_ends) and (rama_biases[i][1]==0 or rama_biases[j][1]==0):
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

def _inner_loop_eoc(term_number,i,nres,chain_starts,chain_ends,rama_biases,
                      p_par, p_anti, p_antihb, p_antinhb, p_parhb, a):
    """
    Helper function for _beta_efficiency_optimized
    """
    lambda_row = np.zeros(nres)
    for j in range(nres):
        if 4<=abs(i-j)<18 and inSameChain(i,j,chain_starts,chain_ends) and not (rama_biases[i][1] and rama_biases[j][1]): # in same chain, seqsep<18, not both beta
            lambda_row[j] = 0
        else:
            if term_number == 1:
                lambda_row[j] = get_lambda_by_index(i, j, 0, chain_starts, chain_ends)
            elif term_number == 2:
                if isChainEdge(i,chain_starts,chain_ends,n=1) or isChainEdge(j,chain_starts,chain_ends,n=1):
                    continue # i+1 or i-1 or j+1 or j-1 don't exist so we won't be able to get a[i-1] and/or a[i+1] and/or a[j-1] and/or a[j+1]
                                # such groups end up not being added to the potential anyway (see below), but this is needed to get the code to run
                lambda_row[j] = get_Lambda_2(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, chain_starts, chain_ends)
            elif term_number == 3:
                if isChainEnd(i,chain_ends,n=1):
                    continue # i+1 doesn't exist so we won't be able to get a[i+1]
                                # such groups end up not being added to the potential anyway (see below), but this is needed to get the code to run
                lambda_row[j] = get_Lambda_3(i, j, p_par, p_anti, p_antihb, p_antinhb, p_parhb, a, chain_starts, chain_ends)
            else:
                raise ValueError(f"term_number must be 1, 2, or 3, but was {term_number}")
    return i, lambda_row

def _beta_efficiency_optimized(oa, term_number, ssweight_file, forceGroup, k_beta, parallel=True): 
    # parallel argument would only be changed for debugging and dev purposes so I'm not going to add it to the main API
    # set constants
    k_beta = convert_units(k_beta) * oa.k_awsem
    nres, n, h, ca, o, res_type = oa.nres, oa.n, oa.h, oa.ca, oa.o, oa.res_type
    r_ON = .298
    sigma_NO = .068
    r_OH = .206
    sigma_HO = .076
    eta_beta_1 = 10.0
    eta_beta_2 = 5.0
    r_HB_c = 1.2
    #
    # load ssweight
    rama_biases = load_ssweight(ssweight_file, nres)
    #
    # load parameters
    a = [] # list to help us to convert from amino acid type to the appropriate row/column indices in p_par, p_anti, etc.
    for ii in range(oa.nres):
        a.append(se_map_1_letter[oa.seq[ii]])
    p_par, p_anti, p_antihb, p_antinhb, p_parhb = read_beta_parameters()
    #
    # calculate Lambda function depending on term number and zero out for short intrachain sequence separation if not both beta
    lambda_term_number = np.zeros((nres, nres))
    if parallel:
        with ThreadPoolExecutor() as executor:
            futures = (executor.submit(_inner_loop_eoc, term_number, i, nres, oa.chain_starts, oa.chain_ends, rama_biases,
                                                         p_par, p_anti, p_antihb, p_antinhb, p_parhb, a) for i in range(nres))
            for future in as_completed(futures):
                i, lambda_row = future.result()
                lambda_term_number[i, :] = lambda_row
    else:
        for i in range(nres):
            _, lambda_term_number[i,:] = _inner_loop_eoc(term_number, i, nres, oa.chain_starts, oa.chain_ends, rama_biases,
                                                            p_par, p_anti, p_antihb, p_antinhb, p_parhb, a)
    #
    # define energy functions
    theta_ij =   f"exp(-(r_Oi_Nj-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oi_Hj-{r_OH})^2/(2*{sigma_HO}^2))"
    theta_ji =   f"exp(-(r_Oj_Ni-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oj_Hi-{r_OH})^2/(2*{sigma_HO}^2))"
    theta_jip2 = f"exp(-(r_Oj_Nip2-{r_ON})^2/(2*{sigma_NO}^2)-(r_Oj_Hip2-{r_OH})^2/(2*{sigma_HO}^2))"
    if term_number == 1:
        beta_string = f"-{k_beta}*lambda_term_number(res_i,res_j)*theta_ij;\
                            theta_ij={theta_ij};r_Oi_Nj=distance(a1,d1);r_Oi_Hj=distance(a1,d2);"
    elif term_number == 2:
        beta_string = f"-{k_beta}*lambda_term_number(res_i,res_j)*theta_ij*theta_ji;\
                        theta_ij={theta_ij};r_Oi_Nj=distance(a1,d1);r_Oi_Hj=distance(a1,d2);\
                        theta_ji={theta_ji};r_Oj_Ni=distance(d3,a2);r_Oj_Hi=distance(d3,a3);"
    elif term_number == 3:
        beta_string = f"-{k_beta}*lambda_term_number(res_i,res_j)*theta_ij*theta_jip2;\
                        theta_ij={theta_ij};r_Oi_Nj=distance(a1,d1);r_Oi_Hj=distance(a1,d2);\
                        theta_jip2={theta_jip2};r_Oj_Nip2=distance(d3,a2);r_Oj_Hip2=distance(d3,a3);"
    else:
        raise ValueError(f"term_number must be 1, 2, or 3, but was {term_number}")
    #
    # set up OpenMM Force
    Beta = CustomHbondForce(beta_string)
    if oa.periodic_box:
        Beta.setNonbondedMethod(Beta.CutoffPeriodic)
        print(f'\nbeta_term_{term_number} is in PBC')
    else:
        Beta.setNonbondedMethod(Beta.CutoffNonPeriodic)        
    Beta.addPerDonorParameter("res_i")
    Beta.addPerAcceptorParameter("res_j")
    # transpose to convert from our matrix-style indexing to OpenMM's cartesian-style indexing,
    # then fortran (F) order flattening to match OpenMM's expectation.
    # This is equivalent to lambda_term_number.T.T.flatten() and therefore also equivalent
    # to lambda_term_number.flatten() (optionally adding the default argument order='C')
    Beta.addTabulatedFunction("lambda_term_number", Discrete2DFunction(nres, nres, lambda_term_number.T.flatten(order='F')))
    Beta.setCutoffDistance(1.0)
    Beta.setForceGroup(forceGroup)
    #
    # loop to add donor and acceptor groups
    for i in range(nres):
        if term_number==1:
            # note that both conditionals may be true (in fact, we typically expect this)
            #
            # see if we can add this amino acid as an acceptor group (acts as the "i" residue)
            if oa.o[i] != -1 and not isChainEnd(i,oa.chain_ends,n=1):
                Beta.addAcceptor(oa.o[i], -1, -1, [i])
            # see if we can add this amino acid as a donor group (acts as the "j" residue)
            if oa.n[i]!=-1 and oa.h[i]!=-1:
                assert not isChainStart(i,oa.chain_starts,n=1) # n[i] and h[i] shouldn't exist for a start residue
                Beta.addDonor(oa.n[i], oa.h[i], -1, [i])            
        elif term_number==2:
            # to participate in beta2, an amino acid must have both donor and acceptor groups
            if isChainEdge(i,oa.chain_starts,oa.chain_ends,n=1):
                pass
            elif o[i]==-1 or n[i]==-1 or h[i]==-1: 
                pass # we're dealing with a proline or have missing atoms for some reason
            else: 
                Beta.addAcceptor(o[i], n[i], h[i], [i])
                Beta.addDonor(n[i], h[i], o[i], [i])
        elif term_number==3:
            # note that both conditionals may be true (in fact, we typically expect this)
            #
            # see if we can add this amino acid and its neighbor as an acceptor group (the "i and i+2" residues)
            if not isChainEnd(i, oa.chain_ends, n=2):
                if o[i] != -1 and n[i+2] !=-1 and h[i+2] !=-1:
                    Beta.addAcceptor(o[i], n[i+2], h[i+2], [i])
            # see if we can add this amino acid as a donor group (the "j" residue)
            if not isChainEnd(i, oa.chain_ends, n=1):
                if o[i] != -1 and n[i] !=-1 and h[i] !=-1:
                    Beta.addDonor(n[i], h[i], o[i], [i])
        else:
            raise ValueError(f"term_number must be 1, 2, or 3, but was {term_number}")
    return Beta

def _pap_lammps_awsemmd(oa, ssweight_file, forceGroup, k_pap, enable_antiparallel=True, enable_parallel=True):
    print("pap term ON")
    # define constants
    nres, ca = oa.nres, oa.ca
    r0 = 0.8 # nm
    eta_pap = 70 # nm^-1
    gamma_aph = 1.0
    gamma_ap = 0.4
    gamma_p = 0.4
    k_beta_pred_p_ap = 1.5
    rama_biases = load_ssweight(ssweight_file, nres)
    # define energy term
    nu_ij = f"0.5*(1+tanh({eta_pap}*({r0}-distance(p1,p2))))" # distance(p1,p2) is r_CAi_Caj
    other_nu = f"0.5*(1+tanh({eta_pap}*({r0}-distance(p3,p4))))"# distance(p3,p4) is r_CAi+4_CAj+4 (parallel) or r_CAi+4_CAj-4 (antiparallel)
    nu_product = f"{nu_ij}*{other_nu}" # either (nu_ij * nu_i+4,j+4) or (nu_ij * nu_i+4,j-4)
    pap_energy = f"-{k_pap}*K*{nu_product}"
    # initialize Force
    pap = CustomCompoundBondForce(4, pap_energy)
    pap.addPerBondParameter("K")
    if oa.periodic_box:
        pap.setUsesPeriodicBoundaryConditions(True)
        print(f'\npap_term_old is in PBC')
    else:
        pap.setUsesPeriodicBoundaryConditions(False)
    # add Bonds to Force and set per-bond parameters (coefficients)
    for i in range(nres):
        if not inSameChain(i,i+4,oa.chain_starts,oa.chain_ends):
            # Not a valid i
            continue
        for j in range(nres):
            delta = j-i
            intrachain = inSameChain(i,j,oa.chain_starts,oa.chain_ends)
            if enable_antiparallel:
                # check if we may be able to add an antiparallel hydrogen bond
                if not inSameChain(j,j-4,oa.chain_starts,oa.chain_ends):
                    K = 0 # Not a valid j
                elif intrachain and delta < 13: 
                    K = 0 #The pair is too close to be a hairpin bond
                elif intrachain and 13<=delta<17: 
                    K = gamma_aph # The pair is a hairpin, k_beta_pred_p_ap doesn't need to be scaled
                elif delta >= 17 or not intrachain:
                    K = gamma_ap
                    if rama_biases[i][1] == 1 and rama_biases[j][1] == 1:
                        K *= k_beta_pred_p_ap
                else:
                    raise AssertionError("unexpected else block")
                if K: # add the Bond unless its energy would always be 0
                    pap.addBond([ca[i],ca[j],ca[i+4],ca[j-4]], [K])
            if enable_parallel:
                # check if we may be able to add a parallel hydrogen bond
                if not inSameChain(j,j+4,oa.chain_starts,oa.chain_ends):
                    K=0 #Not a valid j
                elif intrachain and delta < 9:
                    K=0 # The pair is too close to be a parallel bond
                elif rama_biases[i][1] == 1 and rama_biases[j][1] == 1:
                    K = gamma_p*k_beta_pred_p_ap
                else:
                    K = gamma_p
                if K: # add the Bond unless its energy would always be 0
                    pap.addBond([ca[i],ca[j],ca[i+4],ca[j+4]], [K])
    # finalize and return Force 
    pap.setForceGroup(forceGroup)
    return pap

def _pap_efficiency_optimized(oa, term_number, ssweight_file, forceGroup, k, dis_i_to_i4, pap_nu_on=True):
    # set constants
    k_pap = convert_units(k) * oa.k_awsem
    nres, ca = oa.nres, oa.ca
    r0 = 0.8  # nm
    eta_pap = 70  # nm^-1
    gamma_aph = 1.0
    gamma_ap = 0.4
    gamma_p = 0.4
    #
    # load ssweight
    ssweight = load_ssweight(ssweight_file, nres)
    #
    # load parameters
    gamma_1 = np.zeros((nres, nres))
    gamma_2 = np.zeros((nres, nres))
    gamma_3 = np.zeros((nres, nres))
    for i in range(nres):
        for j in range(nres):
            resId1 = i
            chain1 = inWhichChain(resId1, oa.chain_ends)
            resId2 = j
            chain2 = inWhichChain(resId2, oa.chain_ends)
            gamma_1[i][j] = get_pap_gamma_APH(i, j, chain1, chain2, gamma_aph)
            gamma_2[i][j] = get_pap_gamma_AP(i, j, chain1, chain2, gamma_ap, ssweight)
            gamma_3[i][j] = get_pap_gamma_P(i, j, chain1, chain2, gamma_p, ssweight)
    #
    # define energy functions
    if pap_nu_on:
        constraint_i_and_i4 = f"0.5*(1+tanh({eta_pap}*(distance(a1,a2)-{dis_i_to_i4})))"
        # note that we will not call addBond when i and i+4 are in different chains or i+4 does not exist,
        # so we don't have to worry about including those conditionals in our definition of constraint_i_and_i4
    else:
        constraint_i_and_i4 = "1"
    if term_number == 1:
        gamma_string = "(gamma_1(donor_idx,acceptor_idx)+gamma_2(donor_idx,acceptor_idx))"
    elif term_number == 2:
        gamma_string = "gamma_3(donor_idx,acceptor_idx)"
    else:
        raise ValueError(f"term_number must be 1 or 2, but was {term_number}")
    pap_function = f"-{k_pap}*{gamma_string}\
                        *0.5*(1+tanh({eta_pap}*({r0}-distance(a1,d1))))\
                        *0.5*(1+tanh({eta_pap}*({r0}-distance(a2,d2))))\
                        *{constraint_i_and_i4}"
    #
    # set up OpenMM Force
    pap = CustomHbondForce(pap_function)
    if oa.periodic_box:
        pap.setNonbondedMethod(pap.CutoffPeriodic)
        print(f'\npap_{term_number} is in PBC')
    else:
        pap.setNonbondedMethod(pap.CutoffNonPeriodic)
    pap.addPerDonorParameter("donor_idx")
    pap.addPerAcceptorParameter("acceptor_idx")
    pap.setCutoffDistance(1.0)
    pap.setForceGroup(forceGroup)
    #     Residue i is the acceptor and residue j is the donor and we index our gammas with (donor_idx,acceptor_idx),
    #     so this is actually the cartesian indexing (as opposed to the matrix indexing) that openmm expects.
    #     However, we still need to flatten our matrix in column-major (Fortran) order
    if term_number == 1:
        pap.addTabulatedFunction("gamma_1", Discrete2DFunction(nres, nres, gamma_1.flatten(order='F')))
        pap.addTabulatedFunction("gamma_2", Discrete2DFunction(nres, nres, gamma_2.flatten(order='F')))
    elif term_number == 2:
        pap.addTabulatedFunction("gamma_3", Discrete2DFunction(nres, nres, gamma_3.flatten(order='F')))
    else:
        raise ValueError(f"term_number must be 1 or 2, but was {term_number}")
    #
    # add donor and acceptor groups
    for i in range(nres):
        if term_number == 1:
            if not isChainEnd(i, oa.chain_ends, n=4):
                pap.addAcceptor(ca[i], ca[i+4], -1, [i])
            if not isChainStart(i, oa.chain_starts, n=4):
                pap.addDonor(oa.ca[i], oa.ca[i-4], -1, [i])
        elif term_number == 2:
            if not isChainEnd(i, oa.chain_ends, n=4):
                pap.addAcceptor(ca[i], ca[i+4], -1, [i])
                pap.addDonor(oa.ca[i], oa.ca[i+4], -1, [i])   
        else:
            raise ValueError(f"term_number must be 1 or 2, but was {term_number}")
    return pap
