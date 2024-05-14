import numpy as np

from larch.xray import f0,f1_chantler,f2_chantler

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


def f_complex(element,q,energy):
    energy=int(energy)
    f_0=f0(element,q)
    f1=f1_chantler(element,energy)
    f2=f2_chantler(element,energy)
    f=np.empty_like(q,dtype=complex)
    for i in range(len(q)):
        f[i]=f_0[i]+f1+1j*f2
    return f

def dbetween2atoms(xyz1,xyz2):
    return np.sqrt((xyz1[0]-xyz2[0])**2+(xyz1[1]-xyz2[1])**2+(xyz1[2]-xyz2[2])**2)

def iofq(q,energy,structurefile):
    
    #from complex_form_factor import _f_complex,_f0,atomicformfactor_nist
    """Calculate I(Q) (X-ray) using the Debye Equation.
    
    I(Q) = 2 sum(i,j) f_i(Q) f_j(Q) sinc(rij Q) exp(-0.5 ssij Q**2)
    (The exponential term is the Debye-Waller factor.)
    
    structurefile   --  path to xyz file. It is assumed that the
            structure is that of an isolated scatterer. Periodic boundary
            conditions are not applied.
    q   --  The q-points to calculate over.
    energy -- expressed in eV, X-ray energy
    """
    
    # The functions we need
    sinc = np.sinc
    exp = np.exp
    pi = np.pi
    # read structure
    element_array,xyz_coords=xyz_parser(structurefile)
    
    Uij=0.005
    # The brute-force calculation is very slow. Thus we optimize a little bit.
    # The precision of distance measurements
    deltad = 1e-6
    dmult = int(1/deltad)
    deltau = deltad**2
    umult = int(1/deltau)
    
    pairdict = {}
    elcount = {}
    n = len(xyz_coords)
    
    for i in range(n):
        eli=element_array[i]
        m = elcount.get(eli, 0)
        # incrmentation du nombre de paires
        elcount[eli] = m + 1        
        for j in range(i + 1, n):

            elj = element_array[j]
    
            # Get the pair
            els = [eli, elj]
            els.sort()
    
            # Get the distance to the desired precision
            d = dbetween2atoms(xyz_coords[i],xyz_coords[j])
            D = int(d*dmult)
    
            # Get the DW factor to the same precision
            ss = 2*Uij
            SS = int(ss*umult)
    
            # Record the multiplicity of this pair
            key = (els[0], els[1], D, SS)
            mult = pairdict.get(key, 0)
            # incrémentation du nbre de paires
            pairdict[key] = mult + 1
    print('elcount',elcount)        
    # First we must cache the scattering factors
    fdict = {}
    for el in elcount:
        fdict[el]=f_complex(str(el),q,energy)

    # Now we can compute I(Q) for the i != j pairs
    y = 0
    x = q * deltad / pi
    for key, mult in pairdict.items():
        eli = key[0]
        elj = key[1]
        fi = fdict[eli]
        fj = fdict[elj]
        D = key[2]
        SS = key[3]
        # Debye Waller factor
        DW = exp(-0.5 * SS * deltau * q**2)
        # Note that numpy's sinc(x) = sin(x*pi)/(x*pi)
        y += np.abs(fi * fj) * mult * sinc(x * D) * DW
    
    # We must multiply by 2 since we only counted j > i pairs.
    y *= 2
    
    # Now we must add in the i == j pairs.
    for el, f in fdict.items():
            y += np.abs(f**2) * elcount[el]
    
    return y        
    