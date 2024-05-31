import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
import numpy as np        
from debyecalculator import DebyeCalculator
from pyNanoMatBuilder import platonicNPs as pNP
from scipy.optimize import curve_fit
import os

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
        
# Create function which calculates a pdf for a given particle shape
def calcpdf1(rexp,expfile,element,shape, scalefactor,distance,size0,biso,qdamp=0.014):
    """
    expfile : filepath to experimental pdf. Is used to extract r grid
    element : atom type of the structural model
    distance : interatomic distance (1st peak of the experimental PDF)
    shape : Icosahedron, Decahedron, Cuboctahedron,... shapes avaialble in pyNMB library
    size : size parameter according to shape, as defined in pyNMB
    qdamp : damping parameter used in the pdf calculator
    biso : atomic displacement parameter used in the evaluation of Debye Waller coefficient
    
    """
    print('expfile',expfile)
    print('element',element)
    print('shape',shape)
    print('scalefactor',scalefactor)
    print('distance',distance)
    print('size',round(size0))
    print('biso',biso)
    print('qdamp',qdamp)
    
    # At first, create DebyeCalculator and configure as working on the same r grid as the experimental pdf
    f=open(expfile,'r')
    line=f.readline()
    while line:
        line=f.readline()
        if "rmin" in line:
            rmin=float(line.split('=')[1])
        if "rmax" in line:
            rmax=float(line.split('=')[1])
        if "rstep" in line:
            rstep=float(line.split('=')[1])
        if "qmin" in line:
            qmin=float(line.split(' = ')[1])
        if "qmax" in line:
            qmax=float(line.split(' = ')[1])
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    calc=DebyeCalculator(qmin=qmin,qmax=qmax,qstep=((qmax-qmin)/1000),rmin=rmin,rmax=rmax,rstep=rstep,qdamp=qdamp,device=device,biso=biso)
    
    # Create structure
    strufile='structure.xyz'
    if shape=='Icosahedron':
        pyNMBstru=pNP.regIco(element,distance,nShell=round(size0))
        asestru,_=pyNMBstru.coords()
        writexyz(asestru,strufile)
    """
    implement other shapes here
    
    """
    r,G=calc.gr(structure_source=strufile)
    Gok=np.interp(rexp,r,G)
    os.system('rm %s'%strufile)
       
    return scalefactor*Gok

## Since scipy.optimize.curve_fit does not accpet str type variables, we write this function which takes as input int or float variables
## Note that size can not be defined as an array: as a consquence, these 2 functions calcpdf and custom_calcpdf should be written for cases when shape has 2 or 3 (or more) size parameters

def custom_calcpdf1(rexp,scalefactor,distance,size,biso,qdamp=0.014):
    
    return calcpdf1 (rexp,expfile,element,shape,scalefactor,distance,size,biso,qdamp)

def r2calc(iexp,ith):
        mean=np.mean(iexp)
        TSS=np.sum((iexp-mean)**2)
        RSS=np.sum((iexp-ith)**2)
        return 1-RSS/TSS

# fit Experimental data on Au - Icosahedron 4 shells
nshell=4

# Provide initial parameters

expfile='/home-local/ratel-ra/Documents/PDF_data/ESRF-exp/Au_ico/NS_Ti25C_1000_0001/diffpy/data/NS_Ti25C_1000_0001_0001.gr'
element='Au';shape='Icosahedron';scalefactor=1;distance=2.85;size=nshell;biso=0.1;qdamp=0.016

init_params_pdf=[scalefactor,distance,size,biso,qdamp]

lb=[0,0.95*distance,nshell-1,0,0]
ub=[np.inf,1.05*distance,nshell+1,1,np.inf]

# load data
rexp,gexp=np.loadtxt(expfile,skiprows=27,unpack=True)

# extract region of data to fit
rfitmin=0;rfitmax=25
r2fit=rexp[(rexp>=rfitmin) & (rexp<=rfitmax)]
g2fit=gexp[(rexp>=rfitmin) & (rexp<=rfitmax)]

# Perform optimization
#params, _ =curve_fit(custom_calcpdf1,rexp,gexp,p0=init_params_pdf,bounds=(lb,ub),method='trf')
params, _ =curve_fit(custom_calcpdf1,r2fit,g2fit,p0=init_params_pdf,bounds=(lb,ub),method='trf')
"""
# successive iterations for sequential parameter optimization
for i in range(len(init_params_pdf)):
    init_params_partial=init_params_pdf[:i]
    lb_partial=lb[:i];ub_partial=ub[:i];bounds_partial=(lb_partial,ub_partial)
    ref_params, _ =curve_fit(custom_calcpdf1,rexp,gexp,p0=init_params_partial,bounds=bounds_partial,method='trf')
    
    # update init_params with refined values from init_params_partial
    np.replace(init_params[:i],ref_params)?????
    #for prm in ref_params:
    #    init_params[i]=prm
    

"""

scalefactorf=params[0];distancef=params[1];sizef=params[2];bisof=params[3];
qdampf=params[4]
gfit=custom_calcpdf1(r2fit,scalefactorf, distancef,sizef,bisof,qdampf)

def plot(rexp,gexp,gfit):
    r2=r2calc(gexp,gfit)
    diff = gexp - gfit
    diffzero = (min(gexp)-np.abs(max(diff))) * \
        np.ones_like(gexp)
    
    # Calculate the residual (difference) array and offset it vertically.
    diff = gexp - gfit + diffzero    
    plt.figure()
    #plt.clf()
    # Change some style detials of the plot
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # Create a figure and an axis on which to plot
    fig, ax1 = plt.subplots(1, 1)
    
    # Plot the difference offset line
    ax1.plot(rexp, diffzero, lw=1.0, ls="--", c="black")
    
    # Plot the measured data
    ax1.plot(rexp,gexp,
             ls="None",
             marker="o",
             ms=5,
             mew=0.2,
             mfc="None",
             label="G(r) Data")
    ax1.plot(rexp, diff, lw=1.2, label="G(r) diff")
    ax1.plot(rexp,gfit,'g',label='G(r) calc - R²=%f'%r2)
    ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
    ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
    ax1.tick_params(axis="both",
                    which="major",
                    top=True,
                    right=True)
    
    ax1.set_xlim(rexp[0], rexp[-1])
    ax1.legend(ncol=2)
    plt.tight_layout()
    plt.show()
    
plot(r2fit,g2fit,gfit)



## Fit I(q) curves

def calciq1(expfile,element,shape, scalefactor,background,distance,size0,biso=0.1,qdamp=0.014):
    """
    expfile : filepath to experimental pdf. Is used to extract r grid
    element : atom type of the structural model
    distance : interatomic distance (1st peak of the experimental PDF)
    shape : Icosahedron, Decahedron, Cuboctahedron,... shapes avaialble in pyNMB library
    size : array of 3 parameters [a b c] used to build the desired shape (e.g. [4,0,0] for a 4 shell icosahedron (note that [4] would also be fine)
    qdamp : damping paramters used in the pdf calculator
    biso : atomic displacement parameter used in the evaluation of Debye Waller coefficient
    
    """
    
    # At first, create DebyeCalculator and configure as working on the same r grid as the experimental pdf
    f=open(expfile,'r')
    line=f.readline()
    while line:
        line=f.readline()
        if "rmin" in line:
            rmin=float(line.split('=')[1])
        if "rmax" in line:
            rmax=float(line.split('=')[1])
        if "rstep" in line:
            rstep=float(line.split('=')[1])
        if "qmin" in line:
            qmin=float(line.split(' = ')[1])
        if "qmax" in line:
            qmax=float(line.split(' = ')[1])
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    calc=DebyeCalculator(qmin=qmin,qmax=qmax,qstep=((qmax-qmin)/1000),rmin=rmin,rmax=rmax,rstep=rstep,qdamp=qdamp,device=device,biso=biso)
    
    # Create structure
    strufile='structure.xyz'
    if shape=='Icosahedron':
        pyNMPstru=pNP.regIco(element,distance,nShell=size0)
        asestru,_=pyNMBstru.coords()
        writexyz(asestru,strufile)
    """
    implement other shapes here
    
    """
    q,I=calc.iq(structure_source=strufile)
    os.system('rm %s'%strufile)
    return scalefactor*I + background
