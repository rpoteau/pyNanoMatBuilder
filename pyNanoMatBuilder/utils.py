import visualID as vID
from visualID import  fg, hl, bg

######################################## Folder pathways
def ciflist(dbFolder='cif_database'):
    import os
    path2cif = os.path.join(pNMB_location(),dbFolder)
    print(os.listdir(path2cif))
        
def pNMB_location():
    import pyNanoMatBuilder, pathlib, os
    path = pathlib.Path(pyNanoMatBuilder.__file__)
    return pathlib.Path(*path.parts[0:-2])

######################################## Builder
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

def centerOfGravity(c,select):
    import numpy as np
    nselect = len(select)
    xg = 0
    yg = 0
    zg = 0
    for at in select:
        xg += c[at][0]
        yg += c[at][1]
        zg += c[at][2]
    cog = [xg/nselect, yg/nselect, zg/nselect]
    return cog

######################################## Momenta of inertia
def moi(model):
    import numpy as np
    model.moi = model.get_moments_of_inertia() # in amu*angstrom**2
    print(f"Moments of inertia = {model.moi[0]:.2f} {model.moi[1]:.2f} {model.moi[2]:.2f} amu.Å2")
    model.masses = model.get_masses()
    model.M = model.masses.sum()
    model.moiM = model.moi/model.M
    print(f"Moments of inertia / M = {model.moiM[0]:.2f} {model.moiM[1]:.2f} {model.moiM[2]:.2f} amu.Å2")
    model.dim = 2*np.sqrt(5*model.moiM)
    print(f"Size of the ellipsoid = {model.dim[0]*0.1:.2f} {model.dim[1]*0.1:.2f} {model.dim[2]*0.1:.2f} nm")

######################################## Geometry optimization
def optimizeEMT(model, pathway="./coords/model", fthreshold=0.05):
    from varname import nameof, argname
    import numpy as np
    from ase.io import write
    from ase import Atoms
    from ase.visualize import view
    from ase.calculators.emt import EMT
    vID.centerTitle(f"ase EMT calculator & Quasi Newton algorithm for geometry optimization")
    model.calc=EMT()
    model.get_potential_energy()
    from ase.optimize import QuasiNewton
    dyn = QuasiNewton(model, trajectory=pathway+'.opt')
    dyn.run(fmax=fthreshold)
    write(pathway+"_opt.xyz", model)
    print(f"{fg.BLUE}Optimization steps saved in {pathway+'_.opt'} (binary file){fg.OFF}")
    print(f"{fg.RED}Optimized geometry saved in {pathway+'_opt.xyz'}{fg.OFF}")
    view(model)
