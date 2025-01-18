from pdbfixer import PDBFixer
from simtk.openmm import app
from simtk.openmm.app import pdbxfile
from simtk.openmm.app import Topology
import math
from openmm.unit import norm

fixer = PDBFixer('/home/fc36/dump/1ag2.cif')
writer = pdbxfile.__dict__['PDBxFile']

awsem_atoms = ["CA", "O", "CB", "C", "H", "N"]
dont_remove_bonds = []
dont_remove_atoms = []

#for bond in fixer.topology._bonds:
#    for atom in bond:
#        if atom.name in awsem_atoms:
#            dont_remove_bonds.append(bond)
for bond in fixer.topology.bonds():
    flag = False
    for atom in bond:
        if atom.name not in awsem_atoms:
            flag = True
    if not flag:
        dont_remove_bonds.append(bond)
#dont_remove_bonds = set(dont_remove_bonds)
# assuming positions are stored in the same order as atoms
dont_remove_positions = []
for counter,atom in enumerate(fixer.topology.atoms()):
    if atom.name in awsem_atoms:
        dont_remove_atoms.append(atom)
        dont_remove_positions.append([10*fixer.positions[counter][0].__dict__['_value'],
                                      10*fixer.positions[counter][1].__dict__['_value'],10*fixer.positions[counter][2].__dict__['_value']])
        print(dont_remove_positions[-1])
        #dont_remove_positions.append(fixer.positions[counter][:])
#dont_remove_atoms = set(dont_remove_atoms)

new_top = Topology()

for atom in dont_remove_atoms:
#for counter in range(fixer.topology._numAtoms):
#    if counter not in [atom.index for atom in dont_remove_atoms]:
#        continue
#    else: 
#        atom = [atom for atom in dont_remove_atoms if atom.index==counter][0]
    #print(atom.residue.chain.id)
    if atom.residue.chain.id not in [chain.id for chain in new_top.chains()]:
        new_top.addChain(atom.residue.chain.id)
    if len(list(new_top.residues())) == 0:
        new_top.addResidue(atom.residue.name,list(new_top.chains())[-1],atom.residue.id)
    elif atom.residue.id not in [residue.id for residue in list(new_top.residues()) if residue.chain.id == atom.residue.chain.id]:
        #new_top.addResidue(atom.residue.name,list(new_top.chains())[list(chain.id for chain in new_top.chains()).index(atom.residue.chain.id)],
        #atom.residue.id) #name string, chain object, id string 
        new_top.addResidue(atom.residue.name,[chain for chain in new_top.chains() if chain.id==atom.residue.chain.id][0],atom.residue.id)
                              
    # we could have repeat residue numbering on different chains
    assert len([residue for residue in new_top.residues() if residue.id==atom.residue.id and residue.chain.id==atom.residue.chain.id]) == 1, [residue for residue in new_top.residues() if residue.id==atom.residue.id]
    new_top.addAtom(atom.name,atom.element,[residue for residue in new_top.residues() if residue.id==atom.residue.id and residue.chain.id==atom.residue.chain.id][0],id=None)
    

    

print(new_top)
#for pos in dont_remove_positions:
#    print(math.isnan(norm(pos)))    #new_positions.append(fixer.positions[])
    #new_positions.append
    # name string, element object, residue object, id (going to skip that here)
    #if counter==10:
    #    print(new_top) 
    #    print(list(new_top.chains()))
    #    exit()
    
    #new_top.addAtom(atom.name,atom.element,atom.residue)
    #print(counter)
writer.writeFile(new_top,dont_remove_positions,open('test_fixer_foo.cif','w'))
#fixer.topology._bonds = [fixer.topology._bonds[bond_index] for bond_index in dont_remove_bonds]
#fixer.positions = [fixer.positions[atom_index] for atom_index in dont_remove_atoms]
#fixer.topology._numAtoms = len(fixer.positions)
#print(fixer.topology.__dict__)
#print(len(fixer.positions))
#try:
#    writer.writeFile(fixer.topology,fixer.positions,open('test_fixer_foo.cif','w'),keepIds=False)
#except ValueError:
#    print(fixer.topology._numAtoms)
#    print(len(list(fixer.topology.atoms())))
#    print(len(fixer.positions))
