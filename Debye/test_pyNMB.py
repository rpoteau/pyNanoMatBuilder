import os
import sys

import visualID as vID
from visualID import  fg, hl, bg

import pyNanoMatBuilder.utils as pNMBu

#### build nanosphere Ru-hcp radius=4nm
from pyNanoMatBuilder import crystalNPs as cyNP


def writexyz(atoms,filename):
    #from ase import Atoms
    element_array=atoms.get_chemical_symbols()
    # extract composition in dict form
    composition={}
    for element in element_array:
        if element in composition:
            composition[element]+=1
        else:
            composition[element]=1
       
    coord=atoms.get_positions()
    natoms=len(element_array)  
    line2write='%d \n'%natoms
    line2write+='%s\n'%str(composition)
    for i in range(natoms):
        line2write+='%s'%str(element_array[i])+'\t %.6f'%float(coord[i,0])+'\t %.6f'%float(coord[i,1])+'\t %.6f'%float(coord[i,2])+'\n'
    with open(filename,'w') as file:
        file.write(line2write)
            
    
cwd0 = '/home-local/ratel-ra/anaconda3_c/envs/py311/lib/python3.11/site-packages/styles/'
vID.init(cwd0)

vID.centerTitle(f"Ru Sphere")
RuNP = cyNP.Crystal("Au",shape='sphere',size=[1])
RuNP = RuNP.makeNP()

#print('type RuNP',type(RuNP))


filepath=os.getcwd()+'/structures-pyNMB/'
xyzfile=filepath+"SphericalAuNP.xyz"

# Write xyz file
writexyz(RuNP,xyzfile)


###### Calcul de I(q)
from debyecalculator import DebyeCalculator
calc = DebyeCalculator(qmin=0.01,qmax=20,qstep=0.01,device='cpu',biso=0)
Q, I = calc.iq(structure_source=xyzfile)

xyz2=filepath+'Sphere_10.0.xyz'
Q2, I2 = calc.iq(structure_source=xyz2)

from matplotlib import pyplot as plt
plt.figure()
plt.loglog(Q,I,label='pyNanoMatBuilder')
plt.loglog(Q2,I2,label='WAXS_toolbox')
plt.legend()
plt.show()