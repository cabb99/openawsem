try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except ModuleNotFoundError:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
import numpy as np

def atom_dna(pdbfile="clean.pdb",DNAID=1,DNAlength=100):
    dnagroup=[]
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             stdenv = line.split()
             if stdenv[3][0:1] == "D":
                for j in range(8):
                    if (int(stdenv[5]) == DNAID+j or int(stdenv[5]) == 2*DNAlength-DNAID+1-j):
                       print (line)
                       dnagroup.append(int(stdenv[1])-1)
    print (dnagroup)
    return dnagroup

def atom_protein(self,pdbfile="clean.pdb",flag=1):
    proteingroup=[]
    if flag == 1:
       resID1=[20,21,23,26,27,28,29,99,171,204,205,206,237,268,269,328,330,331,334,335,336,337,338,407,480,481]
    else:
       resID1=[]
    for i in range(len(resID1)):
        proteingroup.append(self.ca[resID1[i]-1])
    print (proteingroup)
    return proteingroup

def group_constraint_by_distance_PD(self, d0=10*angstrom,pdbfile="clean.pdb",proteinID=1,DNAID=1, forceGroup=4, k=10*kilocalorie_per_mole):
    # CustomCentroidBondForce only work with CUDA not OpenCL.
    # only CA, CB, O has mass. so the group have to include those.
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * self.k_awsem
    d0 = d0.value_in_unit(nanometer)   # convert to nm
    constraint = CustomCentroidBondForce(2, f"0.5*{k_constraint}*(distance(g1,g2)-{d0})^2")
    # example group set up group1=[oa.ca[7], oa.cb[7]] use the ca and cb of residue 8.
    group1= atom_protein(self,pdbfile=pdbfile,flag=proteinID)
    group2= atom_dna(pdbfile,DNAID,DNAlength=100) 
    constraint.addGroup(group1)    # group use particle index.
    constraint.addGroup(group2)
    constraint.addBond([0, 1])
    constraint.setForceGroup(forceGroup)
    return constraint

def group_constraint_by_distance_TG(self, d0=10*angstrom,group1file="ProteinGroup.txt",group2file="DnaGroup.txt",forceGroup=4, k=10*kilocalorie_per_mole):
    # CustomCentroidBondForce only work with CUDA not OpenCL.
    # only CA, CB, O has mass. so the group have to include those.
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * self.k_awsem
    d0 = d0.value_in_unit(nanometer)   # convert to nm
    constraint = CustomCentroidBondForce(2, f"0.5*{k_constraint}*(distance(g1,g2)-{d0})^2")
    # example group set up group1=[oa.ca[7], oa.cb[7]] use the ca and cb of residue 8.
    group1a= np.loadtxt(group1file)
    group1=[]
    for i in range(len(group1a)):
        group1.append(int(group1a[i]))
    group2a= np.loadtxt(group2file)
    group2=[]
    for i in range(len(group2a)):
        group2.append(int(group2a[i]))
    constraint.addGroup(group1)    # group use particle index.
    constraint.addGroup(group2)
    constraint.addBond([0, 1])
    constraint.setForceGroup(forceGroup)
    return constraint

def constarint_distfile(self,distfile="PDdist.txt",k=10*kilocalorie_per_mole, forceGroup=5):
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * self.k_awsem
    dist = np.loadtxt(distfile)
    [m,n] = np.shape(dist)
    con = HarmonicBondForce()
    for i in range(m):
        con.addBond(int(dist[i][0]),int(dist[i][1]),float(dist[i][2]),k_constraint/m)
    con.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return con

def constraint_by_distance(oa, res1, res2,  d0=0*angstrom, forceGroup=3, k=1*kilocalorie_per_mole):
    # print(len(oa.ca))
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * oa.k_awsem
    d0 = d0.value_in_unit(nanometer)   # convert to nm
    constraint = CustomBondForce(f"0.5*{k_constraint}*(r-{d0})^2")
    # res1, res2 is 0 index. res1 = 0 means the first residue.
    constraint.addBond(*[oa.ca[res1], oa.ca[res2]])         # you could also do constraint.addBond(oa.ca[res1], oa.ca[res2])
    constraint.setForceGroup(forceGroup)
    return constraint

#def group_constraint_by_distance(oa, d0=0*angstrom, group1=[oa.ca[0], oa.ca[1]], group2=[oa.ca[2], oa.ca[3]], forceGroup=3, k=1*kilocalorie_per_mole):
    # CustomCentroidBondForce only work with CUDA not OpenCL.
    #
    # note added 11 Jun 2025: CustomCentroidBondForce worked for me on OpenCL on my workstation, ws1808
    #
    # only CA, CB, O has mass. so the group have to include those.
#    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
#    k_constraint = k * oa.k_awsem
#    d0 = d0.value_in_unit(nanometer)   # convert to nm
#    constraint = CustomCentroidBondForce(2, f"0.5*{k_constraint}*(distance(g1,g2)-{d0})^2")
    # example group set up group1=[oa.ca[7], oa.cb[7]] use the ca and cb of residue 8.
#    constraint.addGroup(group1)    # group use particle index.
#    constraint.addGroup(group2)
#    constraint.addBond([0, 1])
#    constraint.setForceGroup(forceGroup)
#    return constraint

from simtk.unit import dalton
def group_constraint_by_position(oa, k=1*kilocalorie_per_mole, x0=10, y0=10, z0=10, appliedToResidues=-1, forceGroup=3):
    # x0, y0, z0 is in unit of nm.
    # appliedToResidues can be a list of residue index. for example appliedToResidues=[0, 1], to tether the first two residues.
    # 1 Kcal = 4.184 kJ strength by overall scaling
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * oa.k_awsem
    sum_of_x_coord = CustomExternalForce(f"x*mass")
    sum_of_y_coord = CustomExternalForce(f"y*mass")
    sum_of_z_coord = CustomExternalForce(f"z*mass")

    sum_of_x_coord.addPerParticleParameter("mass")
    sum_of_y_coord.addPerParticleParameter("mass")
    sum_of_z_coord.addPerParticleParameter("mass")

    # print("index for CAs", oa.ca)
    print(f"mass can be retrieved as ", oa.system.getParticleMass(oa.ca[0]))
    total_mass = 0.0
    for i in range(oa.natoms):
        if appliedToResidues == -1:
            mass = oa.system.getParticleMass(i).value_in_unit(dalton)
            sum_of_x_coord.addParticle(i, [mass])
            sum_of_y_coord.addParticle(i, [mass])
            sum_of_z_coord.addParticle(i, [mass])
            total_mass += mass
        elif oa.resi[i] in appliedToResidues:
            mass = oa.system.getParticleMass(i).value_in_unit(dalton)
            sum_of_x_coord.addParticle(i, [mass])
            sum_of_y_coord.addParticle(i, [mass])
            sum_of_z_coord.addParticle(i, [mass])
            total_mass += mass
        # if oa.resi[i] == appliedToResidue:
        #     pulling.addParticle(i)
        # print(oa.resi[i] , oa.seq[oa.resi[i]])
    print(f"total_mass = {total_mass}")
    harmonic = CustomCVForce(f"{k_constraint}*((sum_x/{total_mass}-{x0})^2+(sum_y/{total_mass}-{y0})^2+(sum_z/{total_mass}-{z0})^2)")
    harmonic.addCollectiveVariable("sum_x", sum_of_x_coord)
    harmonic.addCollectiveVariable("sum_y", sum_of_y_coord)
    harmonic.addCollectiveVariable("sum_z", sum_of_z_coord)
    harmonic.setForceGroup(forceGroup)
    return harmonic

def group_constraint_by_position_centroid(oa,k,restraint_coords,restraint_group,forceGroup=30):
    """
    k: kcal/mol, force constant for the restraint, where the potential is (1/2)*k*x^2
    restraint_coords: nm, a list of coordinates [x,y,z] to which the group is restrained
    restraint_group: a list of 0-indexed residue indices of the group to be restrained 
    forceGroup: i think default value of 30 is not used by any other openawsem term, but am not sure
    """
    # initialize Force
    restraint_force = CustomCentroidBondForce(1,f"0.5*{k}*sqrt(((x1-{restraint_coords[0]})^2)+((y1-{restraint_coords[1]})^2)+((z1-{restraint_coords[2]})^2))")
    # get particle indices
    particles = []
    for counter,ca_i in enumerate(oa.ca):
        if ca_i != -1 and counter in restraint_group:
            particles.append(ca_i)
    for counter,cb_i in enumerate(oa.cb):
        if cb_i != -1 and counter in restraint_group:
            particles.append(cb_i)
    for counter,o_i in enumerate(oa.o):
        if o_i != -1 and counter in restraint_group:
            particles.append(o_i)
    # add particles to Force
    restraint_force.addGroup(particles)
    #
    restraint_force.setForceGroup(forceGroup)
    return restraint_force

