from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import pandas as pd
import math
def vector(p1, p2):
    return [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]

def vabs(a):
    return math.sqrt(pow(a[0],2)+pow(a[1],2)+pow(a[2],2))

def calc_dist(p1, p2):
        v=vector(p1,p2)
        return vabs(v)

def read_DNA_config(pdbfile):
    index=[]
    atoms = []
    structure_interactions=[]
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             if line.split()[3] == "DA" or line.split()[3] == "DG" or line.split()[3] == "DT" or line.split()[3] == "DC":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom = [x,y,z]
                index.append(int(float(line.split()[1])-1))
                atoms.append(atom)
    for i in range(len(index)):
        for j in range(i+1,len(index)):
            dist = calc_dist(atoms[i],atoms[j])/10.0
            structure_interaction = [index[i], index[j],dist]
            structure_interactions.append(structure_interaction)
    return structure_interactions

def constrained_groups(oa):
    print ("config on")
    structure_interactions = read_DNA_config("clean.pdb")
    print (structure_interactions)
    con2 = HarmonicBondForce()
    normalization = len(structure_interactions)
    for structure_interaction in structure_interactions:
        con2.addBond(structure_interaction[0],structure_interaction[1],structure_interaction[2], 1000*4.184)
    con2.setForceGroup(9)
    return con2
          
