import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
import numpy as np        
from debyecalculator import DebyeCalculator
from pyNanoMatBuilder import platonicNPs as pNP
from scipy.optimize import curve_fit
from ase.io import write
import os
import lmfit
from lmfit import create_params, fit_report, minimize


# Create function which calculates a pdf for a given particle shape
def calcpdf1(params,rexp, expfile, element,shape,delete_tag=True):
    """
    expfile : filepath to experimental pdf. Is used to extract r grid
    element : atom type of the structural model
    distance : interatomic distance (1st peak of the experimental PDF)
    shape : Icosahedron, Decahedron, Cuboctahedron,... shapes avaialble in pyNMB library
    size : size parameter according to shape, as defined in pyNMB
    qdamp : damping parameter used in the pdf calculator
    biso : atomic displacement parameter used in the evaluation of Debye Waller coefficient
    
    """
    
    scalefactor=params['scalefactor'].value
    distance=params['distance'].value
    size=params['size'].value
    biso=params['biso'].value
    qdamp=params['qdamp'].value    
    """
    print('expfile',expfile)
    print('element',element)
    print('shape',shape)
    print('scalefactor',scalefactor)
    print('distance',distance)
    print('size',size)
    print('biso',biso)
    print('qdamp',qdamp)
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
        pyNMBstru=pNP.regIco(element,distance,nShell=round(size),aseView=False)
        write(strufile,pyNMBstru.NP)
        #writexyz(asestru,strufile)
    """
    implement other shapes here
    
    """
    r,G=calc.gr(structure_source=strufile)
    Gok=np.interp(rexp,r,G)
    if delete_tag:
        os.system('rm %s'%strufile)
       
    return scalefactor*Gok


def residuals_pdf(params,rexp,gexp,expfile,element,shape):
    return gexp - calcpdf1(params,rexp, expfile, element,shape)

def r2calc(iexp,ith):
        mean=np.mean(iexp)
        TSS=np.sum((iexp-mean)**2)
        RSS=np.sum((iexp-ith)**2)
        return 1-RSS/TSS

def plotpdf(rexp,gexp,result):
    # residuals
    diff=result.residual
    diffzero = (min(gexp)-np.abs(max(diff))) * \
        np.ones_like(gexp)
    
    #
    chi2=result.chisqr
    # Calculate the residual (difference) array and offset it vertically.
    diff +=diffzero    
    
    #mpl.rcParams.update(mpl.rcParamsDefault)
    
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
    ax1.plot(rexp, diff, lw=1.2, label="G(r) diff - chi²=%.4f"%chi2)
    gfit=calcpdf1(result.params,rexp, expfile, element,shape)
    ax1.plot(rexp,gfit,'g',label='G(r) calc ')
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
    
    
    
# fit Experimental data on Au - Icosahedron 3 shells


# Provide initial parameters

expfile='/home-local/ratel-ra/Documents/PDF_data/ESRF-exp/Au_ico/NS_Ti25C_1000_0001/diffpy/data/NS_Ti25C_1000_0001_0001.gr'


# load data
rexp,gexp=np.loadtxt(expfile,skiprows=27,unpack=True)

# non refinable parameters
element='Au';shape='Icosahedron';
# refinable parameters: initial values
nshell=3
scalefactor=1.65;distance=2.88;size=nshell;biso=0.1;qdamp=0.016

# define parameters for refinement
# Refinement of all params
params=lmfit.create_params(scalefactor={'value':scalefactor,'vary':True,'min':0.5,'max':np.inf},
                           distance={'value':distance,'vary':True,'min':0,'max':np.inf},
                           size={'value':size,'vary':True, 'min':1,'max':np.inf},
                           biso={'value':biso,'vary':True,'min':0,'max':np.inf},
                           qdamp={'value':qdamp,'vary':True,'min':0,'max':np.inf})


# extract region of data to fit
rfitmin=2.4;rfitmax=25
r2fit=rexp[(rexp>=rfitmin) & (rexp<=rfitmax)]
g2fit=gexp[(rexp>=rfitmin) & (rexp<=rfitmax)]



# Start refinement
result=lmfit.minimize(residuals_pdf,params,method='leastsq',args=(r2fit,g2fit,expfile,element,shape))

best_params=result.params
g_refined=calcpdf1(best_params,r2fit, expfile, element,shape,delete_tag=False)

r2=r2calc(g2fit,g_refined)
chi2=result.chisqr

# Write results to file
line2write='Experimental data file: %s'%expfile +'\n'
line2write+='R²=%.6f'%r2
line2write+='--------------------------------\n'
line2write+= str(fit_report(result))

with open('results.txt', "w") as f:
    f.writelines(line2write)

    
plotpdf(r2fit,g2fit,result)



## Fit I(q) curves

# One approach could be to consider a mixture of solvent+sample, with corresponding scalefactors. In such condition the user must supply 2 structural models (sovent model is fixed).
# another approach could be to simply substratc a baseline


def calciq(params,q,solvent_strufile,element,shape):
    """
    solvent_strufile : filepath to structure of solvent (used to compute reference signal)
    distance : interatomic distance (1st peak of the experimental PDF)
    shape : Icosahedron, Decahedron, Cuboctahedron,... shapes avaialble in pyNMB library
    size : array of 3 parameters [a b c] used to build the desired shape (e.g. [4,0,0] for a 4 shell icosahedron (note that [4] would also be fine)
    qdamp : damping paramters used in the pdf calculator
    biso : atomic displacement parameter used in the evaluation of Debye Waller coefficient
    s, s1: scale factors (s: main scale factor, s1 scale factors of individual phases, the reference fraction being 1-s1)
    """
    scalefactor=params['scalefactor'].value
    s1=params['s1'].value
    #s2=params['s2'].value
    distance=params['distance'].value
    size=params['size'].value
    biso=params['biso'].value   
    qdamp=params['qdamp'].value
    
    
    # At first, create DebyeCalculator and configure as working on the same r grid as the experimental pdf
    qmin=min(q);qmax=max(q)
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    calc=DebyeCalculator(qmin=qmin,qmax=qmax,qstep=((qmax-qmin)/len(q)),qdamp=qdamp,device=device,biso=biso)
    
    # Create structure
    strufile='structure.xyz'
    if shape=='Icosahedron':
        pyNMPstru=pNP.regIco(element,distance,nShell=round(size),aseView=False)
        write(strufile,pyNMPstru.NP)
    """
    implement other shapes here
    
    """
    qc,I=calc.iq(structure_source=strufile)
    Iok=np.interp(q,qc,I)
    q,reference=calc.iq(structure_source=solvent_strufile)
    ref_ok=np.interp(q,qc,reference)
    os.system('rm %s'%strufile)
    return scalefactor*(s1*Iok +(1-s1)*ref_ok)

def residuals_iq(params,q,i,solvent_strufile,element,shape):
    
    return i-calciq(params,q,solvent_strufile,element,shape)

def plotiq(qexp,iexp,result):
    # residuals
    diff=result.residual
    diffzero = (min(iexp)-np.abs(max(diff))) * \
        np.ones_like(iexp)
    
    #
    chi2=result.chisqr
    # Calculate the residual (difference) array and offset it vertically.
    diff +=diffzero    
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # Create a figure and an axis on which to plot
    fig, ax1 = plt.subplots(1, 1)
    
    # Plot the difference offset line
    ax1.plot(qexp, diffzero, lw=1.0, ls="--", c="black")
    
    # Plot the measured data
    ax1.plot(qexp,iexp,
             ls="None",
             marker="o",
             ms=5,
             mew=0.2,
             mfc="None",
             label="I(q) Data")
    ax1.plot(qexp, diff, lw=1.2, label="I(q) diff - chi²=%.4f"%chi2)
    ifit=calciq(result.params,qexp,solvent_strufile,element,shape)
    ax1.plot(qexp,ifit,'g',label='I(q) calc ')
    ax1.set_xlabel(r"q ($\mathrm{\AA}⁻1$)")
    ax1.set_ylabel(r"I")
    ax1.tick_params(axis="both",
                    which="major",
                    top=True,
                    right=True)
    
    ax1.set_xlim(qexp[0], qexp[-1])
    ax1.legend(ncol=2)
    plt.tight_layout()
    plt.show()


datafile='/home-local/ratel-ra/Documents/PDF_data/Ezgi/5mM-Au-NPS_hexane.xy'

tthexp,iexp=np.loadtxt(datafile,unpack=True)
qexp=4*np.pi*np.sin(tthexp*np.pi/360)/0.71

# extract region of data to fit
qfitmin=2;qfitmax=max(qexp)
q2fit=qexp[(qexp>=qfitmin) & (qexp<=qfitmax)]
i2fit=iexp[(qexp>=qfitmin) & (qexp<=qfitmax)]

#solvent 
solvent_strufile='/home-local/ratel-ra/Documents/BiMAN/code_python/IQ/hexane.xyz'
scalefactor=100
s1=0.5
best_distance=best_params['distance'].value
best_biso=best_params['biso'].value
best_qdamp=best_params['qdamp'].value  
best_size=best_params['size'].value

params=lmfit.create_params(scalefactor={'value':scalefactor,'vary':True,'min':0,'max':np.inf},
                           s1={'value':s1,'vary':True,'min':0,'max':1},
                           distance={'value':best_distance,'vary':False,'min':0,'max':np.inf},
                           size={'value':best_size,'vary':False, 'min':1,'max':np.inf},
                           biso={'value':best_biso,'vary':False,'min':0,'max':np.inf},
                           qdamp={'value':best_qdamp,'vary':False,'min':0,'max':np.inf})


result=lmfit.minimize(residuals_iq,params,method='leastsq',args=(q2fit,i2fit,solvent_strufile,element,shape))

print(fit_report(result))

plotiq(q2fit,i2fit,result)