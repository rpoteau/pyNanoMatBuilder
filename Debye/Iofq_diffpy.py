import numpy as np
from diffpy.structure import Structure
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

def iofq(q,energy,structurefile):
    
    #from complex_form_factor import _f_complex,_f0,atomicformfactor_nist
    """Calculate I(Q) (X-ray) using the Debye Equation.
    
    I(Q) = 2 sum(i,j) f_i(Q) f_j(Q) sinc(rij Q) exp(-0.5 ssij Q**2)
    (The exponential term is the Debye-Waller factor.)
    
    S   --  A diffpy.structure.Structure instance. It is assumed that the
            structure is that of an isolated scatterer. Periodic boundary
            conditions are not applied.
    q   --  The q-points to calculate over.
    atom_list
    f_at_list: np.Arrays extracted from form factor calculations
    """
    
    # The functions we need
    sinc = np.sinc
    exp = np.exp
    pi = np.pi
    # read structure
    S = Structure()
    S.read(str(structurefile))
    S.Uisoequiv=0.005
    
    
    # The brute-force calculation is very slow. Thus we optimize a little bit.
    # The precision of distance measurements
    deltad = 1e-6
    dmult = int(1/deltad)
    deltau = deltad**2
    umult = int(1/deltau)
    
    pairdict = {}
    elcount = {}
    n = len(S)
    for i in range(n):
            
    # count the number of each element
            eli = S[i].element
            #f_at_=np.zeros((nb_grp,len(q)),dtype='complex128')
            m = elcount.get(eli, 0)
            elcount[eli] = m + 1
    
            for j in range(i + 1, n):

                    elj = S[j].element
            
                    # Get the pair
                    els = [eli, elj]
                    els.sort()
            
                    # Get the distance to the desired precision
                    d = S.distance(i, j)
                    D = int(d*dmult)
            
                    # Get the DW factor to the same precision
                    ss = S[i].Uisoequiv + S[j].Uisoequiv
                    SS = int(ss*umult)
            
                    # Record the multiplicity of this pair
                    key = (els[0], els[1], D, SS)
                    mult = pairdict.get(key, 0)
                    pairdict[key] = mult + 1

# Now we can calculate IofQ from the pair dictionary. Making the dictionary
# first reduces the amount of calls to sinc and exp we have to make.

# First we must cache the scattering factors
    fdict = {}
    for el in elcount:
        fdict[el]=self.f_complex(str(el),q,energy)

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

    return np.array([q,y])    