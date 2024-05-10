import visualID as vID
from visualID import  fg, hl, bg
import numpy as np

from ase.atoms import Atoms
from ase.io import write
import os

##############################################################################################
######################################## time
from datetime import datetime
import datetime, time
class timer:

    def __init__(self):
        _start_time   = None
        _end_time     = None
        _chrono_start = None
        _chrono_stop  = None

    # Return human delay like 01:14:28 543ms
    # delay can be timedelta or seconds
    def hdelay_ms(self,delay):
        if type(delay) is not datetime.timedelta:
            delay=datetime.timedelta(seconds=delay)
        sec = delay.total_seconds()
        hh = sec // 3600
        mm = (sec // 60) - (hh * 60)
        ss = sec - hh*3600 - mm*60
        ms = (sec - int(sec))*1000
        return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'
    
    def chrono_start(self):
        global _chrono_start, _chrono_stop
        _chrono_start=time.time()
    
    # return delay in seconds or in humain format
    def chrono_stop(self, hdelay=False):
        global _chrono_start, _chrono_stop
        _chrono_stop = time.time()
        sec = _chrono_stop - _chrono_start
        if hdelay : return self.hdelay_ms(sec)
        return sec

    def hdelay_ms(self,delay):
        if type(delay) is not datetime.timedelta:
            delay=datetime.timedelta(seconds=delay)
        sec = delay.total_seconds()
        hh = sec // 3600
        mm = (sec // 60) - (hh * 60)
        ss = sec - hh*3600 - mm*60
        ms = (sec - int(sec))*1000
        return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'
    
    def chrono_show(self):
        print(f'{fg.BLUE}Duration : {self.hdelay_ms(time.time() - _chrono_start)}{fg.OFF}')

##############################################################################################
######################################## coupling with pymatgen in order to find the symmetry
def MolSym(aseobject: Atoms,
           getEquivalentAtoms: bool=False):
    import pymatgen.core as pmg
    from pymatgen.io.ase import AseAtomsAdaptor as aaa
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    
    chrono = timer(); chrono.chrono_start()
    vID.centertxt("Symmetry analysis",bgc='#007a7a',size='14',weight='bold')
    print(f"Currently using the PointGroupAnalyzer class of pymatgen\nThe analyzis can take a while for large compounds")
    print()
    pmgmol = pmg.Molecule(aseobject.get_chemical_symbols(),aseobject.get_positions())
    pga = PointGroupAnalyzer(pmgmol, tolerance=0.6, eigen_tolerance=0.02, matrix_tolerance=0.2)
    pg = pga.get_pointgroup()
    print(f"Point Group: {pg}")
    print(f"Rotational Symmetry Number = {pga.get_rotational_symmetry_number()}")
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    if getEquivalentAtoms:
        return pg, pga.get_equivalent_atoms()
    else:
        return pg, []

##############################################################################################
######################################## Folder pathways
def ciflist(dbFolder='cif_database'):
    import os
    path2cif = os.path.join(pNMB_location(),dbFolder)
    print(os.listdir(path2cif))
        
def pNMB_location():
    import pyNanoMatBuilder, pathlib, os
    path = pathlib.Path(pyNanoMatBuilder.__file__)
    return pathlib.Path(*path.parts[0:-2])

##############################################################################################
######################################## Vectors and distances
def RAB(coord,a,b):
    import numpy as np
    """calculate the interatomic distance between two atoms a and b"""
    r = np.sqrt((coord[a][0]-coord[b][0])**2 + (coord[a][1]-coord[b][1])**2 + (coord[a][2]-coord[b][2])**2)
    return r

def Rbetween2Points(p1,p2):
    import numpy as np
    r = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    return r

def vector(coord,a,b):
    import numpy as np
    v = [coord[b][0]-coord[a][0],coord[b][1]-coord[a][1],coord[b][2]-coord[a][2]]
    v = np.array(v)
    return v

def vectorBetween2Points(p1,p2):
    import numpy as np
    v = [p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]]
    v = np.array(v)
    return v

def coord2xyz(coord):
    import numpy as np
    x = np.array(coord)[:,0]
    y = np.array(coord)[:,1]
    z = np.array(coord)[:,2]
    return x,y,z

def vertex(x, y, z, scale):
    import numpy as np
    """ Return vertex coordinates fixed to the unit sphere """
    length = np.sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x,y,z)]

    
def RadiusSphereAfterV(V):
    import numpy as np
    return (3*V/(4*np.pi))**(1/3)

def centerOfGravity(c: np.ndarray,
                    select=None):
    import numpy as np
    if select is None:
        select = np.array((range(len(c))))
    nselect = len(select)
    xg = 0
    yg = 0
    zg = 0
    for at in select:
        xg += c[at][0]
        yg += c[at][1]
        zg += c[at][2]
    cog = [xg/nselect, yg/nselect, zg/nselect]
    return np.array(cog)

def center2cog(c: np.ndarray):
    import numpy as np
    cog = centerOfGravity(c)
    ccog = []
    for at in c:
        at = at - cog
        ccog.append(at)
    return np.array(ccog)

def normV(V):
    '''
    returns the norm of a vector V, [V0,V1,V2]
    '''
    import numpy as np
    return np.sqrt(V[0]**2+V[1]**2+V[2]**2)

def centerToVertices(coordVertices: np.ndarray,
                     cog: np.ndarray):
    '''
    returns the vector and distance between the center of gravity (cog) and each vertex of a polyhedron
    input:
        - coordVertices = coordinates of the vertices ((nvertices,3) numpy array)
        - cog = center of gravity of the NP
    returns:
        - (cog-nvertices)x3 coordinates of the vectors (np.array)
        - nvertices-cog distances (np.array)
    '''
    import numpy as np
    directions = []
    distances = []
    for v in coordVertices:
        distances.append(Rbetween2Points(v,cog))
        directions.append(v - cog)
    return np.array(directions), np.array(distances)

##############################################################################################
######################################## Fill edges and facets
def MakeFaceCoord(Rnn,f,coord,nAtomsOnFaces,coordFaceAt):
    import numpy as np
    # the idea here is to interpolate between edge atoms of two relevant edges
    # (for example two opposite edges of a squared face)
    # be careful of the vectors orientation of the edges!
    if (len(f) == 3):  #triangular facet
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    if (len(f) == 4):  #square facet 0-1-2-3-4-0
        edge1 = [f[3],f[0]]
        edge2 = [f[2],f[1]]
    if (len(f) == 5):  #pentagonal facet
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    if (len(f) == 6):  #hexagonal facet
        edge1 = [f[0],f[1]]
        edge2 = [f[5],f[4]]
    nAtomsOnEdges = int((RAB(coord,f[1],f[0])+1e-6)/Rnn) - 1
    nIntervalsE = nAtomsOnEdges + 1
    for n in range(nAtomsOnEdges):
        CoordAtomOnEdge1 = coord[edge1[0]]+vector(coord,edge1[0],edge1[1])*(n+1) / nIntervalsE
        CoordAtomOnEdge2 = coord[edge2[0]]+vector(coord,edge2[0],edge2[1])*(n+1) / nIntervalsE
        distBetween2EdgeAtoms = Rbetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)
        nAtomsBetweenEdges = int((distBetween2EdgeAtoms+1e-6)/Rnn) - 1
        nIntervalsF = nAtomsBetweenEdges + 1
        for m in range(nAtomsBetweenEdges):
            coordFaceAt.append(CoordAtomOnEdge1 + vectorBetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)*(m+1) / nIntervalsF)
            nAtomsOnFaces += 1
    return nAtomsOnFaces,coordFaceAt

##############################################################################################
######################################## Momenta of inertia
def moi(model: Atoms):
    import numpy as np
    vID.centertxt("Moments of inertia",bgc='#007a7a',size='14',weight='bold')
    model.moi = model.get_moments_of_inertia() # in amu*angstrom**2
    print(f"Moments of inertia = {model.moi[0]:.2f} {model.moi[1]:.2f} {model.moi[2]:.2f} amu.Å2")
    model.masses = model.get_masses()
    model.M = model.masses.sum()
    model.moiM = model.moi/model.M
    print(f"Moments of inertia / M = {model.moiM[0]:.2f} {model.moiM[1]:.2f} {model.moiM[2]:.2f} amu.Å2")
    model.dim = 2*np.sqrt(5*model.moiM)
    print(f"Size of the ellipsoid = {model.dim[0]*0.1:.2f} {model.dim[1]*0.1:.2f} {model.dim[2]*0.1:.2f} nm")

##############################################################################################
######################################## Geometry optimization
def optimizeEMT(model: Atoms, pathway="./coords/model", fthreshold=0.05):
    from varname import nameof, argname
    import numpy as np
    from ase.io import write
    from ase import Atoms
    from ase.visualize import view
    from ase.calculators.emt import EMT
    chrono = timer(); chrono.chrono_start()
    vID.centerTitle(f"ase EMT calculator & Quasi Newton algorithm for geometry optimization")
    model.calc=EMT()
    model.get_potential_energy()
    from ase.optimize import QuasiNewton
    dyn = QuasiNewton(model, trajectory=pathway+'.opt')
    dyn.run(fmax=fthreshold)
    write(pathway+"_opt.xyz", model)
    print(f"{fg.BLUE}Optimization steps saved in {pathway+'_.opt'} (binary file){fg.OFF}")
    print(f"{fg.RED}Optimized geometry saved in {pathway+'_opt.xyz'}{fg.OFF}")
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    view(model)
    return model

##############################################################################################
######################################## Planes
def point2PlaneDistance(point: np.float64,
                              plane: np.float64):
    import numpy as np
    from numpy.linalg import norm
    distance = abs(np.dot(point,plane[0:3]) + plane[3]) / norm(plane[0:3])
    return distance

def interPlanarSpacing(plane: np.ndarray,
                       unitcell: np.ndarray,
                       CrystalSystem: str='CUB'):
    '''
    - input:
        - plane = numpy array that the contains the [h k l d] parameters of the plane of equation
                hx + ky +lz + d = 0
        - unitcell = numpy array with [a b c alpha beta gamma]
        - CrystalSystem = name of the crystal system, string among:
          ['CUB', 'HEX', 'TRH', 'TET', 'ORC', 'MCL', 'TRI'] = cubic, hexagonal, trigonal-rhombohedral, tetragonal, orthorombic, monoclinic, tricilinic
    returns the interplanar spacing (float value)
    '''
    import sys
    h = plane[0]
    k = plane[1]
    l = plane[2]
    a = unitcell[0]
    match CrystalSystem.upper():
        case 'CUB':
            d2 = a**2 / (h**2+k**2+l**2)
        case 'HEX':
            c = unitcell[2]
            d2inv = (4/3)*(h**2 + k**2 + h*k)/a**2 + l**2/c**2
            d2 = 1/d2inv
        case 'TRH':
            alpha = (np.pi/180) * unitcell[3]
            d2inv = ((h**2 + k**2 + l**2)*np.sin(alpha)**2 + 2*(h*k + k*l + h*l)*(np.cos(alpha)**2-np.cos(alpha)))/(a**2*(1-3*np.cos(alpha)**2+2*np.cos(alpha)**3))
            d2 = 1/d2inv
        case 'TET':
            c = unitcell[2]
            d2inv = (h**2+k**2)/a**2 + l**2/c**2
            d2 = 1/d2inv    
        case 'ORC':
            b = unitcell[1]
            c = unitcell[2]
            d2inv = h**2/a**2 + k**2/b**2 + l**2/c**2
            d2 = 1/d2inv    
        case 'MCL':
            b = unitcell[1]
            c = unitcell[2]
            # beta = np.pi - (np.pi/180) * unitcell[4]
            # d2inv = h**2/(a**2*np.sin(beta)) + k**2/b**2 + l**2/(c**2*np.sin(beta)) + 2*h*l*np.cos(beta)/(a*c*np.sin(beta)**2)
            beta = (np.pi/180) * unitcell[4]
            d2inv = ((h/a)**2 + (k*np.sin(beta)/b)**2 + (l/c)**2 - 2*h*l*np.cos(beta)/(a*c))/np.sin(beta)**2
            d2 = 1/d2inv    
        case 'TRI':
            b = unitcell[1]
            c = unitcell[2]
            alpha = (np.pi/180) * unitcell[3]
            beta = (np.pi/180) * unitcell[4]
            gamma = (np.pi/180) * unitcell[5]
            V = (a*b*c) * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
            astar = b*c*np.sin(alpha)/V
            bstar = a*c*np.sin(beta)/V
            cstar = a*b*np.sin(gamma)/V
            cosalphastar = (np.cos(gamma)*np.cos(beta) - np.cos(alpha))/(np.sin(gamma)*np.sin(beta))
            cosbetastar  = (np.cos(alpha)*np.cos(gamma) - np.cos(beta))/(np.sin(alpha)*np.sin(gamma))
            cosgammastar = (np.cos(beta)*np.cos(alpha) - np.cos(gamma))/(np.sin(beta)*np.sin(alpha))
            d2inv = (h*astar)**2 + (k*bstar)**2 + (l*cstar)**2 + 2*k*l*bstar*cstar*cosalphastar\
                                                               + 2*l*h*cstar*astar*cosbetastar\
                                                               + 2*h*k*astar*bstar*cosgammastar
            d2 = 1/d2inv    
        case _:
            sys.exit(f"{CrystalSystem} crystal system is unknown. Check your data.\n"\
                      "Or do not try to calculate interplanar distances on this system with interPlanarSpacing()")
    d = np.sqrt(d2)
    return d

def planeFittingLSF(coords: np.float64,
                    printErrors: bool=False,
                    printEq: bool=True):
    '''
    least-square fitting of the equation of a plane ux + vy + wz + h = 0
    that passes as close as possible to a set of 3D points
    - input
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - printErrors: if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed 
    - returns a numpy array [u v w h]
    '''
    import numpy as np
    from numpy import linalg as la
    xs = coords[:,0]
    ys = coords[:,1]
    zs = coords[:,2]
    nat = len(xs)
    select=[i for i in range(nat)]
    cog = centerOfGravity(coords, select)
    mat = np.zeros((3,3))
    for i in range(nat):
        mat[0,0]=mat[0,0]+(xs[i]-cog[0])**2
        mat[1,1]=mat[1,1]+(ys[i]-cog[1])**2
        mat[2,2]=mat[2,2]+(zs[i]-cog[2])**2
        mat[0,1]=mat[0,1]+(xs[i]-cog[0])*(ys[i]-cog[1])
        mat[0,2]=mat[0,2]+(xs[i]-cog[0])*(zs[i]-cog[2])
        mat[1,2]=mat[1,2]+(ys[i]-cog[1])*(zs[i]-cog[2])    
    mat[1,0]=mat[0,1]
    mat[2,0]=mat[0,2]
    mat[2,1]=mat[1,2]
    eigenvalues, eigenvectors = la.eig(mat)
    # the eigenvector associated with the smallest eigenvalue is the vector normal to the plane
    # print(eigenvalues)
    # print(eigenvectors)
    indexMinEigenvalue = np.argmin(eigenvalues)
    # print(indexMinEigenvalue)
    # print(la.norm(eigenvectors[:,indexMinEigenvalue]))
    u,v,w = eigenvectors[:,indexMinEigenvalue]
    h = -u*cog[0] - v*cog[1] - w*cog[2]
    if printEq:
        print(f"bare solution: {u:.5f} x + {v:.5f} y + {w:.5f} z + {h:.5f} = 0")
        # print("     or")
        # print(f"bare solution: {-u/w:.5f} x + {-v/w:.5f} y + {-h/w:.5f} = z")
    tmp = coords.copy
    ones = np.ones(nat)
    tmp = np.column_stack((coords,ones))
    fit = np.array([u,v,w,h])
    fit = fit.reshape(4,1)
    errors = tmp@fit
    residual = la.norm(errors)
    if printErrors:
        print(f"errors:")
        print(errors)
        print(f"residual: {residual}")
    return np.array([u,v,w,h])

def convertuvwh2hkld(plane: np.float64,
                     prthkld: bool=True):
    '''
    converts an ux + vy + wz + h = 0 equation of a plane, where u, v, w and h can be real numbers as 
    an hx + ky + lz + d = 0 equation, where h, k, and l are all integer numbers
    - input:
        - [u v w h] numpy array 
        - prthkld: does print the result by default 
    - returns an [h k l d] numpy array 
    '''
    import numpy as np
    from fractions import Fraction
    # apply only on non-zero uvw values
    planeEff = []
    acc = 1e-8
    for x in plane[0:3]: # u,v,w only
        if (np.abs(x) >= acc):
            planeEff.append(x)
    planeEff = np.array(planeEff)
    F = np.array([Fraction(x).limit_denominator() for x in np.abs(planeEff)]) # don't change the signe of hkl
    Fmin = np.min(F)
    hkld = plane/Fmin
    
    if prthkld:
        print(f"hkl solution: {hkld[0]:.5f} x + {hkld[1]:.5f} y + {hkld[2]:.5f} z + {hkld[3]:.5f} = 0")
        # print("     or")
        # print(f"hkl solution: {-hkld[0]/hkld[2]:.5f} x + {-hkld[1]/hkld[2]:.5f} y + {-hkld[3]/hkld[2]:.5f} = z")
    return hkld

def hklPlaneFitting(coords: np.float64,
                   printErrors: bool=False):
    '''
    Context: finding the Miller indices of a plane, if relevant
    Consists in a least-square fitting of the equation of a plane hx + ky + lz + d = 0
    that passes as close as possible to a set of 3D points
    - input
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - printErrors: if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed
    - returns a numpy array [h k l d], where h, k, and l are as close as possible to integers
    '''
    plane = planeFittingLSF(coords,printErrors)
    plane = convertuvwh2hkld(plane)
    return plane

def shortestPoint2PlaneVectorDistance(plane:np.ndarray,
                                      point:np.ndarray):
    '''
    returns the shortest distance, d, from a point X0 to a plane p (projection of X0 on p = P), as well as the PX0 vector 
    - input:
        - plane = [u v w h] definition of the p plane (numpy array)
        - point = [x0 y0 z0] coordinates of the X0 point (numpy array)
    returns the PX0 vector and ||PX0||
    '''
    t = (plane[3] + np.dot(plane[0:3],point))/(plane[0]**2+plane[1]**2+plane[2]**2)
    v = -t*plane[0:3]
    d = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    return v, d

def Pt2planeSignedDistance(plane,point):
    '''
    returns the orthogonal distance of a given point X0 to the plane p in a metric space (projection of X0 on p = P), 
    with the sign determined by whether or not X0 is in the interior of p with respect to the center of gravity [0 0 0]
    - input:
        - plane = [u v w h] definition of the P plane (numpy array)
        - point = [x0 y0 z0] coordinates of the X0 point (numpy array)
    returns the signed modulus ±||PX0||
    '''
    sd = (plane[3] + np.dot(plane[0:3],point))/np.sqrt(plane[0]**2+plane[1]**2+plane[2]**2)
    return sd

def planeAtVertices(coordVertices: np.ndarray,
                    cog: np.ndarray):
    '''
    returns the equation of the plane defined by vectors between the center of gravity (cog) and each vertex of a polyhedron
    and that is located at the vertex
    input:
        - coordVertices = coordinates of the vertices ((nvertices,3) numpy array)
        - cog = center of gravity of the NP
    returns the (cog-nvertices)x3 coordinates of the plane (np.array)
    '''
    import numpy as np
    planes = []
    for vx in coordVertices:
        vector = vx - cog
        d = -np.dot(vx,vector)
        vector = np.append(vector,d)
        planes.append(vector)
    return np.array(planes)

def planeAtPoint(plane: np.ndarray,
                 P0: np.ndarray):
    '''
    given a former [a,b,c,d] plane as input, d is recalculated so that the plane passes through P0,
    a known point on the plane
    input:
        -plane: numpy array [a b c d]
        -P0: numpy array with P0 coordinates [x0 y0 z0]
    returns [a b c -(ax0+by0+cz0)]
    '''
    d = np.dot(plane[0:3],P0)
    planeAtP = plane.copy()
    planeAtP[3] = -d
    return planeAtP

def normalizePlane(p):
    import numpy as np
    '''
    normalizes the [a,b,c,d] coordinates of a plane
    - input: plane [a,b,c,d]
    returns [a/norm,b/norm,c/norm,d/norm] where norm=dsqrt(a**2+b**2+c**2)
    '''
    return p/normV(p[0:3])

##############################################################################################
######################################## coupling with Jmol
def saveCoords_DrawJmol(asemol,prefix,scriptJ=""):
    path2Jmol = '/usr/local/src/jmol-14.32.50'
    fxyz = "./figs/"+prefix+".xyz"
    write(fxyz, asemol)
    # jmolscript = 'cpk 0; wireframe 0.025; script "./figs/script-facettes-3-4RuLight.spt"; facettes34rulight; draw * opaque; color atoms black; set zShadePower 1; set specularPower 80; pngon; write image 1024 1024 ./figs/'
    jmolscript = scriptJ + '; frank off; cpk 0; wireframe 0.05; script "./figs/script-facettes-345PtLight.spt"; facettes345ptlight; draw * opaque;'
    jmolscript = jmolscript + 'set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    jmolcmd="java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz + " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    print(jmolcmd)
    os.system(jmolcmd)

def DrawJmol(mol,prefix,scriptJ=""):
    path2Jmol = '/usr/local/src/jmol-14.32.50'
    fxyz = "./figs/"+mol+".xyz"
    jmolscript = scriptJ + '; frank off; set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    jmolcmd="java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz + " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    print(jmolcmd)
    os.system(jmolcmd)

##############################################################################################
######################################## coordination numbers
def calculateCN(coords,Rmax):
    '''
    returns the coordination number of each atom, where CN is calculated after threshold Rmax
    - input:
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - Rmax: threshold to calculate CN
    returns an array that contains CN for each atom
    '''
    CN = np.zeros(len(coords))
    for i,ci in enumerate(coords):
        for j in range(0,i):
            Rij = Rbetween2Points(ci,coords[j])
            if Rij <= Rmax:
                CN[i]+=1
                CN[j]+=1
    return CN

def delAtomsWithCN(coords: np.ndarray,
                   Rmax: np.float64,
                   targetCN: int=12):
    '''
    identifies atoms that have a coordination number (CN) == targetCN and returns them in an array
    - input:
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - CN: array of integers with the coordination number of each atom
        - targetCN (default=12)
    returns an array that contains the indexes of atoms with CN == targetCN
    '''
    CN = calculateCN(coords,Rmax)
    tabDelAtoms = []
    for i,cn in enumerate(CN):
        if cn == targetCN: tabDelAtoms.append(i)
    tabDelAtoms = np.array(tabDelAtoms)
    return tabDelAtoms

##############################################################################################
######################################## cut above planes
def calculateTruncationPlanesFromVertices(planes, cutFromVertexAt, nAtomsPerEdge, debug=False):
    n = int(round(1/cutFromVertexAt))
    print(f"factor = {cutFromVertexAt:.3f} ▶ {round(nAtomsPerEdge/n)} layer(s) will be removed, starting from each vertex")

    trPlanes = []
    for p in planes:
        pNormalized =normalizePlane(p.copy())
        pNormalized[3] =  pNormalized[3] - pNormalized[3]*cutFromVertexAt
        trPlanes.append(pNormalized)
        if (debug):
            print("normalized original plane = ",normalizePlane(p))
            print("cut plane = ",pNormalized,"... norm = ",normV(pNormalized[0:3]))
            print("signed distance between original plane and origin = ",Pt2planeSignedDistance(p,[0,0,0]))
            print("signed distance between cut plane and origin = ",Pt2planeSignedDistance(pNormalized,[0,0,0]))
            print("pcut/pRef = ",Pt2planeSignedDistance(pNormalized,[0,0,0])\
                                /Pt2planeSignedDistance(p,[0,0,0]))
        print(f"Will remove atoms just above plane "\
              f"{pNormalized[0]:.2f} {pNormalized[1]:.2f} {pNormalized[2]:.2f} d:{pNormalized[3]:.3f}")
    return np.array(trPlanes)    

def truncateAboveEachPlane(planes: np.ndarray,
                           coords,
                           debug=False,
                           delAbove: bool = True):
    '''
    - input: 
        - planes = numpy array with all [u v w d] plane definitions
        - coords = (N,3) numpy array will all coordinates
        - delAbove = if True (default) delete atoms that lie above the planes + eps = 1e-4. Delete atoms below the
                     planes otherwise (use with precaution, could return no atoms as a function of their definition)
        - hkldRef, hkld: for debugging purpose
    - returns the coordinates of the atoms that are below ALL input planes
    '''

    keptAtoms = []

    eps =1e-3
    for p in planes:
        for i,c in enumerate(coords):
            signedDistance = Pt2planeSignedDistance(p,c)
            if delAbove and signedDistance > eps:
                keptAtoms.append(i)
            elif not delAbove and signedDistance < eps:
                keptAtoms.append(i)
        # print(keptAtoms)
        if debug:
            for a in keptAtoms:
                print(f"@{a+1}",end=',')
            print("",end='\n')
    keptAtoms = np.array(keptAtoms)
    keptAtoms = np.unique(keptAtoms)
    # if (debug): plot3D()
    return keptAtoms

##############################################################################################
######################################## symmetry
def reflection(plane,points):
    '''
    applies a mirror-image symmetry operation of an array of points w.r.t. a plane of symmetry
    - input:
        - plane = [u,v,w,d] parameters that define a plane
        - point = (N, 3) array of points
    - returns an (N, 3) array of mirror-image points
    '''
    import numpy as np
    pr = []
    eps = 1.e-4
    for p in points:
        vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane,p)
        if dp2plane >= eps: # otherwise the point belongs to the reflection plane
            # print(dp2plane, vp2plane, p)
            ptmp = p+2*vp2plane
            pr.append(ptmp)
    return np.array(pr)

##############################################################################################
######################################## rotation
def Rx(a):
  return np.matrix([[ 1, 0           , 0   ],
                    [ 0, m.cos(a),-m.sin(a)],
                    [ 0, m.sin(a), m.cos(a)]])
  
def Ry(a):
  return np.matrix([[ m.cos(a), 0, m.sin(a)],
                   [ 0           , 1, 0           ],
                   [-m.sin(a), 0, m.cos(a)]])
  
def Rz(a):
  return np.matrix([[ m.cos(a), -m.sin(a), 0 ],
                   [ m.sin(a), m.cos(a) , 0 ],
                   [ 0           , 0            , 1 ]])

def EulerRotationMatrix(gamma,beta,alpha,order="zyx"):
    """
    - input:
        - gamma: Rot/x (°)
        - beta: Rot/y (°)
        - alpha: Rot/z (°)
        - if (order="zyx"): returns Rz(alpha) * Ry(beta) * Rx(gamma)
    returns a 3x3 Euler matrix as a numpy array
    """
    import math as m
    #REuler is a 3x3 matrix
    R = 1.
    gammarad = gamma*m.pi/180
    betarad = beta*m.pi/180
    alpharad = alpha*m.pi/180
    for i in range(3):
        if order[i] == "x":
            R = R*Rx(gammarad)
        if order[i] == "y":
            R = R*Ry(betarad)
        if order[i] == "z":
            R = R*Rz(alpharad)
    return R

def EulerRotationMol(coord, gamma, beta, alpha, order="zyx"):
    return np.array(EulerRotationMatrix(gamma,beta,alpha,order)@coord.transpose()).transpose()

def RotationMatrixFromAxisAngle(u,angle):
    import math as m
    a = angle*m.pi/180
    ux = u[0]
    uy = u[1]
    uz = u[2]
    return np.matrix([[cos(a)+ux**2*(1-m.cos(a))     , ux*uy*(1-m.cos(a))-uz*m.sin(a), ux*uz*(1-m.cos(a))+uy*m.sin(a)],
                      [uy*ux*(1-m.cos(a))+uz*m.sin(a), cos(a)+uy**2*(1-m.cos(a))     , uy*uz*(1-m.cos(a))-ux*m.sin(a)],
                      [uz*ux*(1-m.cos(a))-uy*m.sin(a), uz*uy*(1-m.cos(a))+ux*m.sin(a), cos(a)+uz**2*(1-m.cos(a))     ]])

