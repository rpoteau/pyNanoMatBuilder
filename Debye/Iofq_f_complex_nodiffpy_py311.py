import numpy as np
#import sys
import numba as nb
from numba import jit,types
import larch
import math
import os
import pandas as pd
from sas.sascalc.calculator import sas_gen
from larch.xray import f0,f1_chantler,f2_chantler

def f_complex(element,q,energy):
    energy=int(energy)
    
    f_0=f0(element,q)
    f1=f1_chantler(element,energy)
    f2=f2_chantler(element,energy)
    f=np.empty_like(q,dtype=complex)
    for i in range(len(q)):
        f[i]=f_0[i]+f1+1j*f2
    return f

@nb.jit(nb.float64[:](nb.float64[:,:]), nopython=True, fastmath=True, parallel=True)
def r_ij(coords):# fonction qui donne la distance euclidienne entre chaque paire d'atomes
    N, dim = np.shape(coords) #N est le nombre d'atomes, on veut une liste de dimension N
    #r=np.zeros(int((N*N-N)/2),dtype=float)
    r= np.zeros(int((N*N-N)/2), dtype=nb.float64)#création du tableau vide
    for i in nb.prange(N): #fonction de numba qui parallélise l'exécution de la boucle et améliore les performances
        for j in range(i+1,N):
            l = int(i * (N - 1) - i * (i + 1) / 2 + j - 1) #expression qui permet de retrouver l'indice correspondant à chaque paire d'atomes unique dans le tableau r
            tmp = 0.0 
            for k in range(dim): #boucle qui parcourt la dim de xyz
                tmp += (coords[i,k]-coords[j,k])**2 #somme des distances
            r[l] = tmp**0.5
    return r #r est un tableau qui stocke les distances entre les paires d'atomes

# We have to keep records of the pair of atoms corresponding to distances
# use of numba not requested (manipulation of strings)
def pair_ij(element_array):
    """
    takes as input array of N elements extracted from xyz file
    generates array of N(N-1)/2 pairs of atoms
    """
    N=len(element_array) # N is the number of atoms
    shape=(int(N*(N-1)/2),2)
    pair_array=np.empty(shape,dtype='U2')
    for i in range(N): 
        for j in range(i+1,N):
            l = int(i * (N - 1) - i * (i + 1) / 2 + j - 1)
            pair_array[l]=[element_array[i],element_array[j]]
    return pair_array

            

#use of numba not possible due to call to f0, f1_chantler, etc... in f_complex
def f_ij(pair_array,r,Q,energy):
    """
    takes as input an array of N(N_1)/2 pairs
    r, Q are provided to specify the shape of the output
    generates an array containing the  product of scattering factors of the 2 elements of a single pair (ordered in the same sequence as r since calculation algorithms of r and pair_array use similar loops)
    """
    # Computing fi*fj for each pair on the whole Q range
    #f_array=np.empty((len(r),len(Q)),dtype=nb.float64)
    f_array=np.empty((len(r),len(Q)),dtype=float)
    i=0
    for pair in pair_array:
        el1=pair[0];el2=pair[1]
        f_el1=f_complex(el1,Q,energy);f_el2=f_complex(el2,Q,energy)
        f_array[i]=abs(f_el1*f_el2)
        i+=1
    return f_array



@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.complex128[:]), nopython=True, fastmath=True, parallel=True)
def Debye_complex_monoat(Q, r, f_array):
    #here f_array is f_complex(element)
    # retrieve N from shape(f_array)
    a=1;b=-1;c=-2*np.shape(f_array)[0]
    delta=b*b-4*a*c
    N=-b+np.sqrt(delta)/(2*a)
    # initialize output
    res = np.zeros(int(len(Q)), dtype = nb.float64)
    #res = np.zeros(int(len(Q)), dtype = float)
    # calculate output
    for i_Q in nb.prange(len(Q)):
        tmp = 0.0
        for i_r in range(len(r)):
            tmp += math.sin(Q[i_Q]*r[i_r])/(Q[i_Q]*r[i_r])
        res[i_Q] = (N + 2*tmp)*abs(f_array[i_Q])**2
    return res



@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:,:]), nopython=True, fastmath=True, parallel=True)
def Debye_complex(Q, r, f_array):
    # we remind that f_array contains 
    # retrieve N from shape(f_array)
    a=1;b=-1;c=-2*np.shape(f_array)[0]
    delta=b*b-4*a*c
    N=-b+np.sqrt(delta)/(2*a)
    # initialize output
    res = np.zeros(int(len(Q)), dtype = nb.float64)
    #res = np.zeros(int(len(Q)), dtype = float)
    # calculate output
    for i_Q in nb.prange(len(Q)):
        tmp = 0.0
        for i_r in range(len(r)):
            tmp += math.sin(Q[i_Q]*r[i_r])/(Q[i_Q]*r[i_r])
        res[i_Q] = (N + 2*tmp)*abs(f_array[i_r][i_Q])
    return res



def xyz_parser(file):
    """
    Takes a path to xyz file as argument
    Returns an array containing element array and xyz coordinates array
    """
    element=np.loadtxt(file,skiprows=2,usecols=(0),dtype='U8')
    x,y,z=np.loadtxt(file, skiprows=2,usecols=(1,2,3), unpack=True)
    xyz_coords = np.column_stack((x, y, z))
    print(len(element))
    return element, xyz_coords

def Iofq(q,energy,file):
    # grab elements from structure file
    structure=xyz_parser(file)
    element_array=structure[0];xyz_coords=structure[1];N=len(element_array) 
    element_type=np.unique(element_array)
    print('element type',element_type)
    r=r_ij(xyz_coords)
    if len(element_type)==1: #Monoatomic case (faster calculations since f_complex is called only once)
        f_array=f_complex(element_type[0],q,energy)
        I=Debye_complex_monoat(q,r,f_array)
        
    
    else :#polyatomic case, f_complex is called twice for each pair (can be improved, but complicates numba implementation since dict types are not implemented in numba)
    
        #Computation of pairs (sames sequence as rij)
        pair_array=pair_ij(element_array)
        f_array=f_ij(pair_array,r,q,energy)
        print('f_array',f_array)
        I= Debye_complex(q,r,f_array)
    return I


# Create sasview_calculator functions for comparison
def convertxyz2pdb(xyzfilepath):
    xyz = pd.read_csv(xyzfilepath, delim_whitespace=True, skiprows=2, names=["atom", "x", "y", "z"])
    print(xyz.head())
    pdb_template = "ATOM  {num:5} {atom:2}        {molnum:4}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp_factor:6.2f}\n"
    pdb_lines = []
    molnum = 1
    num = 1
    H_label = "1"
    occ=1
    temp_factor=0
    for row_index, row in xyz.iterrows():
        atomlabel = row.atom
       
        pdb_lines.append(pdb_template.format(
                             num=num, atom=atomlabel,molnum=molnum, x=row.x, y=row.y, z=row.z,occ=occ,temp_factor=temp_factor))
    
        num += 1
        if num==10000:
                molnum+=1
                num=1        
    
    output_filename = os.path.join(os.path.dirname(xyzfilepath), os.path.basename(xyzfilepath).replace("xyz", "pdb"))
    #print(output_filename)
    with open(output_filename, "w") as f:
        f.writelines(pdb_lines)
    print("pdb file created:", output_filename)
    return output_filename


def run_sasview(qmin,qmax,qstep,pdbfilepath):
        # load pdb file
        pdbloader = sas_gen.PDBReader()
        pdbData = pdbloader.read(str(pdbfilepath))
        
        # create genSAS class and set data
        model = sas_gen.GenSAS()
        model.set_sld_data(pdbData)
        # calculate along given q range
        q_vals = np.linspace(qmin, qmax, int(qmax/qstep))
        output = model.run([q_vals, []])
        
        iq_file=os.path.join(os.path.dirname(pdbfilepath), os.path.basename(pdbfilepath).replace(".pdb", "_sasview.iq"))
        np.savetxt(iq_file,[q_vals,output],delimiter='    ',newline='\n')
        return q_vals,output   
    



#################################  TEST

q=np.linspace(0.001,20.001,20000)
wavelength=1.54
energy=int(12314/wavelength)
structurefile='/home-local/ratel-ra/Documents/BiMAN/code_python/IQ/Zn3P2.xyz'
structurefile='/home-local/ratel-ra/Documents/BiMAN/code_python/IQ/Ag_Icosahedron_4shells.xyz'
os.system('jmol %s'%structurefile)
i=Iofq(q,energy,structurefile)


# Compare with sasview solution

qmin=0.001;qmax=20.001;qstep=0.001
pdbfile=convertxyz2pdb(structurefile)
q1,i1=run_sasview(qmin,qmax,qstep,pdbfile)

from matplotlib import pyplot as plt

plt.figure(1)
plt.loglog(q,i/np.max(i),'-k',label='Mycode')
plt.loglog(q1,i1/np.max(i1),'--r',label='sasview')
plt.legend()
plt.xlabel('Q (1/A)')
plt.ylabel('S(q)')
plt.show()

