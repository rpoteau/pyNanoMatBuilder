import visualID as vID
from visualID import  fg, hl, bg
import numpy as np
#ASE import
from ase import io
from ase.atoms import Atoms
from ase.io import write
from ase.visualize import view
from ase.io import read

from ase.geometry import cellpar_to_cell
from ase.spacegroup import get_spacegroup
import os

from scipy import linalg

from pyNanoMatBuilder import data
import pathlib
import re

import importlib
from pathlib import Path


#######################################################################
######################################## time
from datetime import datetime
import datetime, time
class timer:
    """
    Timer class to measure elapsed time in seconds and display it 
    in the format hh:mm:ss ms.
    """
    def __init__(self):
        _start_time   = None
        _end_time     = None
        _chrono_start = None
        _chrono_stop  = None

    # delay can be timedelta or seconds
    def hdelay_ms(self,delay):
        """
        Converts a delay into a human-readable format: hh:mm:ss ms.

        Args:
            delay: A timedelta object or a float representing a duration in seconds.
        Return: A formatted string in hh:mm:ss ms.
        """
        if type(delay) is not datetime.timedelta:
            delay=datetime.timedelta(seconds=delay)
        sec = delay.total_seconds()
        hh = sec // 3600
        mm = (sec // 60) - (hh * 60)
        ss = sec - hh*3600 - mm*60
        ms = (sec - int(sec))*1000
        return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'
    
    def chrono_start(self):
        """
        Starts the chrono.
        """
        global _chrono_start, _chrono_stop
        _chrono_start=time.time()
    
    # return delay in seconds or in humain format
    def chrono_stop(self, hdelay=False):
        """
        Stops the chrono and returns the elapsed time.
        """
        global _chrono_start, _chrono_stop
        _chrono_stop = time.time()
        sec = _chrono_stop - _chrono_start
        if hdelay : return self.hdelay_ms(sec)
        return sec

    
    def chrono_show(self):
        """
        Prints the elapsed time.
        """
        print(f'{fg.BLUE}Duration : {self.hdelay_ms(time.time() - _chrono_start)}{fg.OFF}')

#######################################################################
######################################## ase unitcells and symmetry analyzis
def returnUnitcellData(system):
    """
    Function that calculates various unit cell properties from the `system.cif` object 
    and assigns them to attributes within the `system` instance.

    Args:
        An instance of the Crystal class containing CIF file data.
    """
    system.ucUnitcell = system.cif.cell.cellpar()
    system.ucV = cellpar_to_cell(system.ucUnitcell)
    system.ucBL = system.cif.cell.get_bravais_lattice()
    system.ucSG = get_spacegroup(system.cif,symprec=system.aseSymPrec)
    system.ucVolume = system.cif.cell.volume
    system.ucReciprocal = np.array(system.cif.cell.reciprocal())
    system.ucFormula = system.cif.get_chemical_formula()
    system.G = G(system)
    system.Gstar = Gstar(system)

def print_ase_unitcell(system: Atoms):
    '''
    Function that prints unitcell informations : chemical formula, bravais lattice, n° space group, cell parameters, volume, etc.
    Args:
        An instance of the Crystal class
    '''
    unitcell = system.ucUnitcell
    bl = system.ucBL
    formula = system.ucFormula
    volume = system.ucVolume
    sg = system.ucSG
    print(f"Bravais lattice: {bl}")
    print(f"Chemical formula: {formula}")
    print(f"Crystal family = {bl.crystal_family} (lattice system = {bl.lattice_system})")
    print(f"Name = {bl.longname} (Pearson symbol = {bl.pearson_symbol})")
    print(f"Variant names = {bl.variant_names}")
    print()
    print(f"From ase: space group number = {sg.no}      Hermann-Mauguin symbol for the space group = {sg.symbol}")
    print()
    print(f"a: {unitcell[0]:.3f} Å, b: {unitcell[1]:.3f} Å, c: {unitcell[2]:.3f} Å. (c/a = {unitcell[2]/unitcell[0]:.3f})")
    print(f"α: {unitcell[3]:.3f} °, β: {unitcell[4]:.3f} °, γ: {unitcell[5]:.3f} °")
    print()
    print(f"Volume: {volume:.3f} Å3")

def listCifsOfTheDatabase():
    '''
    Displays all filenames  of the database
    '''
    from ase import io
    import pathlib
    import glob
    from ase.spacegroup import get_spacegroup
    import re
    
    path2cifFolder = os.path.join(pyNMB_location(),'cif_database')
    print(f"path to cif database = {path2cifFolder}")
    
    sgITField = "_space_group_IT_number"
    sgHMField = "_symmetry_space_group_name_H-M"
    
    class Crystal:
        pass
        
    for cif in glob.glob(f'{path2cifFolder}/*.cif'):
        path2cifFile = pathlib.Path(cif)
        cifName = pathlib.Path(*path2cifFile.parts[-1:])
        vID.centertxt(f"{cifName}",size=14,weight='bold')
        cifContent = io.read(cif)
        cifFile =  open(cif, 'r')
        cifFileLines = cifFile.readlines()
        re_sgIT = re.compile(sgITField)
        re_sgHM = re.compile(sgHMField)
        for line in cifFileLines: 
            if re_sgIT.search(line): sgIT = ' '.join(line.split()[1:])
            if re_sgHM.search(line): sgHM = ' '.join(line.split()[1:])
        cifFile.close()
        c = Crystal()
        c.cif = cifContent
        c.aseSymPrec = 1e-4
        returnUnitcellData(c)
        print_ase_unitcell(c)
        color="vID.fg.RED"
        print()
        if int(sgIT) == c.ucSG.no:
            print(f"{vID.fg.GREEN}Symmetry in the cif file = {sgIT}   {sgHM}{vID.hl.BOLD} in agreement with the ase symmetry analyzis{vID.fg.OFF}")
        else:
            print(f"{vID.fg.RED}Symmetry in the cif file = {sgIT}   {sgHM}{vID.hl.BOLD} disagrees with the ase symmetry analyzis{vID.fg.OFF}")

def scaleUnitCell(crystal: Atoms,
                  scaleDmin2: float,
                  noOutput: bool=False,
                  ):
    """
    Scales the unit cell of a given crystal structure to match a target nearest-neighbor distance.

    This function expands the unit cell of the input crystal by creating a 2×2×2 supercell, 
    then computes the nearest-neighbor distance and scales the entire structure to match the 
    desired minimum nearest-neighbor distance.

    Args:
    - crystal (Atoms): An ASE Atoms object representing the initial crystal structure.
    - scaleDmin2 (float): The target nearest-neighbor distance (in Å) after scaling.
    - noOutput (bool): If True, suppresses output messages. 

    Returns:
    - None: The function modifies the input `crystal` object in place.

    Notes:
    - The function first generates a 2×2×2 supercell to ensure a representative environment.
    - It computes the minimum nearest-neighbor distance (`Rmin`) using `kDTreeCN`.
    - The structure is then uniformly scaled so that the new nearest-neighbor distance 
      equals `scaleDmin2`.
    - The ASE `set_cell` method is used with `scale_atoms=True` to adjust atomic positions accordingly.
    """
    from ase.build.supercells import make_supercell
    if not noOutput: vID.centertxt(f"Scaling the unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    M = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    x = make_supercell(crystal.cif,M)
    nn,CN,R = kDTreeCN(x,4.0,returnD=True)
    Rmin = min(R[0])
    scale=scaleDmin2/Rmin
    if not noOutput:
        print(f"Unitcell lengths and atomic positions scaled by {scale:.3f} factor")
        print(f"New nearest neighbour distance = {scaleDmin2:.3f} Å")
    ucv = crystal.cif.cell.cellpar()
    ucv[0:3] = ucv[0:3]*scale
    crystal.cif.set_cell(ucv, scale_atoms=True)

#######################################################################
######################################## coupling with pymatgen in order to find the symmetry
def MolSym(aseobject: Atoms,
           getEquivalentAtoms: bool=False,
           noOutput: bool=False,
          ):
    """
    Performs symmetry analysis on a molecular structure using pymatgen's PointGroupAnalyzer.
    This function computes the **point group** of a given ASE `Atoms` object and optionally 
    returns equivalent atoms. 
    Args:
        aseobject: An ASE `Atoms` object representing the molecular structure.
        getEquivalentAtoms: If True, returns the indices of symmetry-equivalent atoms.
        noOutput: If False, details of the files are printed.
    Return:
        A tuple containing:
        - `pg`: The point group symbol as a string.
        - `equivalent_atoms`: A list of equivalent atom indices (if `getEquivalentAtoms=True`), otherwise an empty list.
    """

    import pymatgen.core as pmg
    from pymatgen.io.ase import AseAtomsAdaptor as aaa
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    
    if not noOutput: chrono = timer(); chrono.chrono_start()
    if not noOutput: vID.centertxt("Symmetry analysis",bgc='#007a7a',size='14',weight='bold')
    if not noOutput: print(f"Currently using the PointGroupAnalyzer class of pymatgen\nThe analyzis can take a while for large compounds")
    if not noOutput: print()
    pmgmol = pmg.Molecule(aseobject.get_chemical_symbols(),aseobject.get_positions())
    pga = PointGroupAnalyzer(pmgmol, tolerance=0.6, eigen_tolerance=0.02, matrix_tolerance=0.2)
    pg = pga.get_pointgroup()
    if not noOutput: print(f"Point Group: {pg}")
    if not noOutput: print(f"Rotational Symmetry Number = {pga.get_rotational_symmetry_number()}")
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    aseobject.pg = pg
    if getEquivalentAtoms:
        return pg, pga.get_equivalent_atoms()
    else:
        return pg, []

#######################################################################
######################################## cif files informations
def get_crystal_type(self):
    """
    Find the Bravais lattice based on the space group number.
    Returns:
        str: Bravais lattice
    """  
    spacegroup_number = self.ucSG.no  # space group number
    
    # Bravais lattice based on space group number https://fr.wikipedia.org/wiki/Groupe_d%27espace
    if 195 <= spacegroup_number <= 230:  # Cubic
        if spacegroup_number == 225:
            return 'fcc'
        elif spacegroup_number == 229:
            return 'bcc'
        else:
            return 'cubic'
    elif 168 <= spacegroup_number <= 194:  # Hexagonal
        return 'hcp'
    elif 75 <= spacegroup_number <= 142:  # Tetragonal
        return 'tetragonal'
    elif 16 <= spacegroup_number <= 74:  # Orthorhombic
        return 'orthorhombic'
    elif 3 <= spacegroup_number <= 15:  # Monoclinic
        return 'monoclinic'
    elif 1 <= spacegroup_number <= 2:  # Triclinic
        return 'triclinic'
    else:
        return 'unknown'
   
def FindInterAtomicDist(self) :
    """
    Computes the interatomic distance based on the Bravais lattice (fcc, bcc or hcp only).
    Returns:
        float: Interatomic distance
    """   
    import math
    if self.crystal_type=='fcc':
        d=self.parameters[0]*math.sqrt(2)/2
    
    if self.crystal_type=='bcc' :
        d=self.parameters[0]*math.sqrt(3)/2
    
    if self.crystal_type=='hcp' :
        d_a=self.parameters[0]
        d_c=self.parameters[2]/2
        if d_a>d_c: #if compact
            d=d_c
        if d_c>d_a: #if not compact
            d=d_a

    return d

def extract_cif_info(self,cif_file):
    """
    Extract useful information from a CIF file.
    Args:
        cif_file: CIF file.
    Returns:
        dict: A dictionary containing extracted CIF information:
            - 'cif_path' (Path): Absolute path to the CIF file.
            - 'crystal_type' (str): The crystal type.
            - 'Unitcell_param' (list): Unit cell parameters [a, b, c, α, β, γ].
            - 'ucBL' (str): Bravais lattice type.

    """   
    structure = read(cif_file) # load structure with ase
    
    self.ucUnitcell = self.cif.cell.cellpar() #self.ucUnitcell[0]=a, self.ucUnitcell[1]=b, self.ucUnitcell[2]=c,  self.ucUnitcell[3]= α, etc
    self.parameters=self.cif.cell.lengths() 
    self.ucBL = self.cif.cell.get_bravais_lattice() #HEX, CUB etc
    self.ucSG = get_spacegroup(self.cif,symprec= float(1e-2))
    self.ucFormula = self.cif.get_chemical_formula()
    self.crystal_type = get_crystal_type(self)
    return {
        # 'crystal_name': self.ucFormula,
        'cif_path': cif_file,
        'crystal_type' : self.crystal_type,
        'Unitcell_param' : self.ucUnitcell,
        'ucBL': self.ucBL
        
    }


def load_cif(self, cif_file,noOutput):
    """
    Loads a CIF file and extracts its information if it has not been loaded before.
    Args:
        cif_file: a CIF file.
        noOutput (bool): If False, prints the CIF file path.
    Return:
        dict: Extracted CIF information (from `extract_cif_info`).
    Notes:
    - CIF files are assumed to be stored in the "cif_database" directory.
    - If the file has already been loaded, its information is retrieved from `self.loaded_cifs`. 
        
    """      
    cif_folder = "cif_database"
    path2cif = Path(os.path.join(cif_folder, cif_file)).resolve()
    self.cif = io.read(path2cif)
    if not noOutput :
        print("Absolute path to CIF:", path2cif)
    if not path2cif.exists():
        raise FileNotFoundError(f"File {cif_file} not found.")
    if path2cif not in self.loaded_cifs:
        self.loaded_cifs[path2cif] = extract_cif_info(self,path2cif)
    return self.loaded_cifs[path2cif]



#######################################################################
######################################## Folder pathways
def ciflist(dbFolder=data.pyNMBvar.dbFolder):
    """
    Function that prints the CIF files in the dataset.
    Args:
        dbFolder: The database folder name (default is `data.pyNMBvar.dbFolder`).
    """
 
    import os
    path2cif = os.path.join(pyNMB_location(),dbFolder)
    print(os.listdir(path2cif))
        
def pyNMB_location():
    """
    Returns the root directory of the pyNanoMatBuilder package.
    """
    import pyNanoMatBuilder, pathlib, os
    path = pathlib.Path(pyNanoMatBuilder.__file__)
    return pathlib.Path(*path.parts[0:-2])

#######################################################################
######################################## Coordinates, vectors, etc
def RAB(coord,a,b):
    import numpy as np
    """
    Function that calculates the interatomic distance between two atoms "a" and "b".
    Args:
        coord (array-like): A list or array of 3D coordinates
        a (str): Element (index of the starting point in the `coord` list).
        b (str): Element (index of the starting point in the `coord` list).
    Return:
        The distance r between the two atoms.
    """
    r = np.sqrt((coord[a][0]-coord[b][0])**2 + (coord[a][1]-coord[b][1])**2 + (coord[a][2]-coord[b][2])**2)
    return r

def Rbetween2Points(p1,p2):
    """
    Function that calculates the interatomic distance between two points "p1" and "p2".
    Args:
        p1 (array-like): A list or array of 3D coordinates
        p2 (array-like): A list or array of 3D coordinates
    Return:
        The distance r between the two points.
    """
    import numpy as np
    r = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    return r

def vector(coord,a,b):
    """
    Computes the vector from point `a` to point `b` given a list of coordinates.
    Args:
        coord (array-like): A list or array of 3D coordinates
        a (str): Element (index of the starting point in the `coord` list).
        b (str): Element (index of the starting point in the `coord` list).
    Return:
        A NumPy array representing the vector from `a` to `b`.
    """
    import numpy as np
    v = [coord[b][0]-coord[a][0],coord[b][1]-coord[a][1],coord[b][2]-coord[a][2]]
    v = np.array(v)
    return v

def vectorBetween2Points(p1,p2):
    """
    Computes the vector between two 3D points.
    Args: 
        p1: A list or array of 3D coordinates
        p2: A list or array of 3D coordinates
    Return:
        A NumPy array representing the vector from `p1` to `p2`.
    """
    import numpy as np
    v = [p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]]
    v = np.array(v)
    return v

def coord2xyz(coord):
    """
    Extracts x, y, and z from a list of 3D coordinates.
    Args:
        A list or array of 3D coordinates in the format [[x1, y1, z1], [x2, y2, z2], ...].
    Return:
        Three NumPy arrays containing the x, y, and z coordinates separately.
    """
    import numpy as np
    x = np.array(coord)[:,0]
    y = np.array(coord)[:,1]
    z = np.array(coord)[:,2]
    return x,y,z

def vertex(x, y, z, scale):
    import numpy as np
    """ 
    Returns vertex coordinates fixed to the unit sphere 
    """
    length = np.sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x,y,z)]

def vertexScaled(x, y, z, scale):
    import numpy as np
    """ 
    Returns vertex coordinates multiplied by the scale factor 
    """
    return [i * scale for i in (x,y,z)]

    
def RadiusSphereAfterV(V):
    """
    Computes the radius of a sphere given its volume.
    Args:
        V (float): Volume of the sphere in cubic units.

    Returns:
        float: Radius of the sphere.

    Formula: R = (3V / (4π))^(1/3)
       
    """
    import numpy as np
    return (3*V/(4*np.pi))**(1/3)

def centerOfGravity(c: np.ndarray,
                    select=None):
    """
    Computes the center of gravity (geometric center) of a set of points.

    Args:
        c (np.ndarray): An array of shape (N, 3) representing N atomic positions (x, y, z).
        select (np.ndarray, optional): Indices of selected atoms to include in the calculation.
                                       If None, all atoms are used.
    Returns:
        np.ndarray: A 3-element array representing the center of gravity coordinates (x, y, z).

    Notes:
    - The center of gravity is computed as the average of the selected atomic positions.
    """
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
    """
    Centers a set of atomic coordinates to their center of gravity.
    Args:
        c (np.ndarray): An array of shape (N, 3) representing N atomic positions (x, y, z).
    Returns:
        np.ndarray: A new array of centered coordinates where the center of gravity is at (0,0,0).

    Notes:
    - Uses `centerOfGravity(c)` to compute the center of gravity.
    - Each atomic position is shifted by subtracting the center of gravity.
    """
    import numpy as np
    cog = centerOfGravity(c)
    c2cog = []
    for at in c:
        at = at - cog
        c2cog.append(at)
    return np.array(c2cog)

def normOfV(V):
    '''
    Returns the norm of a vector V.
    Args:
        V (np.ndarray): A 3-element array representing a vector [Vx, Vy, Vz].
    Returns:
        float: The norm of the vector.
    '''
    import numpy as np
    return np.sqrt(V[0]**2+V[1]**2+V[2]**2)

def normV(V):
    '''
    Computes the normalized unit vector of a vector V.
    Args:
        V (np.ndarray): A 3-element array representing a vector [Vx, Vy, Vz].
    Returns:
        np.ndarray: A 3-element array representing the normalized vector.
        
    '''
    import numpy as np
    N = normOfV(V)
    return np.array([V[0]/N,V[1]/N,V[2]/N])

def centerToVertices(coordVertices: np.ndarray,
                     cog: np.ndarray):
    '''
    Computes the vectors and distances between the center of gravity (cog) 
    and each vertex of a polyhedron.
    Args:
        coordVertices (np.ndarray): Array of shape (n_vertices, 3) containing the coordinates of the vertices.
        cog (np.ndarray): A 3-element array representing the center of gravity of the nanoparticle.
    Returns:
        tuple:
            - directions (np.ndarray): Array of shape (n_vertices, 3) containing the vectors from cog to each vertex.
            - distances (np.ndarray): Array of shape (n_vertices,) containing the distances from cog to each vertex.
    '''
    import numpy as np
    directions = []
    distances = []
    for v in coordVertices:
        distances.append(Rbetween2Points(v,cog))
        directions.append(v - cog)
    return np.array(directions), np.array(distances)

#######################################################################
######################################## Fill edges and facets

def MakeFaceCoord(Rnn,f,coord,nAtomsOnFaces,coordFaceAt):
    """
    Interpolates atom positions on a given face of a polyhedron by distributing atoms 
    between two relevant edges.

    Args:
        Rnn (float): Nearest neighbor distance.
        f (list): List of vertex indices defining the face.
        coord (np.ndarray): Array containing the coordinates of all atoms.
        nAtomsOnFaces (int): Counter for the number of atoms placed on faces.
        coordFaceAt (list): List of face atom coordinates to be updated.

    Returns:
        tuple: 
            - nAtomsOnFaces (int): Updated count of face atoms.
            - coordFaceAt (list): Updated list of coordinates of atoms placed on faces.

    Method:
        1. Determines two relevant edges based on the number of vertices in the face.
        2. Interpolates atoms along these edges.
        3. Fills the face by interpolating between interpolated edge atoms.
    """
 
    import numpy as np
    # the idea here is to interpolate between edge atoms of two relevant edges
    # (for example two opposite edges of a squared face)
    # be careful of the vectors orientation of the edges!
    if (len(f) == 3):  #triangular facet
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    elif (len(f) == 4):  #square facet 0-1-2-3-4-0
        edge1 = [f[3],f[0]]
        edge2 = [f[2],f[1]]
    elif (len(f) == 5):  #pentagonal facet #not working
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    elif (len(f) == 6):  #hexagonal facet #not working
        edge1 = [f[0],f[1]]
        edge2 = [f[5],f[4]]
    else:
        raise ValueError("Face type not supported (only 3, 4, 5, or 6 vertices).")
        
    # Determine the number of atoms along the edges
    nAtomsOnEdges = int((RAB(coord,f[1],f[0])+1e-6)/Rnn) - 1
    nIntervalsE = nAtomsOnEdges + 1

    # Interpolate atoms along the edges
    for n in range(nAtomsOnEdges):
        CoordAtomOnEdge1 = coord[edge1[0]]+vector(coord,edge1[0],edge1[1])*(n+1) / nIntervalsE
        CoordAtomOnEdge2 = coord[edge2[0]]+vector(coord,edge2[0],edge2[1])*(n+1) / nIntervalsE

        # Compute distance and interpolate atoms between edge atoms
        distBetween2EdgeAtoms = Rbetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)
        nAtomsBetweenEdges = int((distBetween2EdgeAtoms+1e-6)/Rnn) - 1
        nIntervalsF = nAtomsBetweenEdges + 1
        for m in range(nAtomsBetweenEdges):
            coordFaceAt.append(CoordAtomOnEdge1 + vectorBetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)*(m+1) / nIntervalsF)
            nAtomsOnFaces += 1
    return nAtomsOnFaces,coordFaceAt

#######################################################################
######################################## Momenta of inertia
def moi(model: Atoms,
        noOutput: bool=False,
       ):
    '''
    Get the moments of inertia along the principal axes.
    The three principal moments of inertia are computed from the
    eigenvalues of the symmetric inertial tensor. 
    
    Notes:
        Units of the moments of inertia are amu.angstrom**2.
        Periodic boundary conditions are ignored. 
    '''
    import numpy as np
    if not noOutput: vID.centertxt("Moments of inertia",bgc='#007a7a',size='14',weight='bold')
    model.moi = model.get_moments_of_inertia() # in amu*angstrom**2
    if not noOutput: print(f"Moments of inertia = {model.moi[0]:.2f} {model.moi[1]:.2f} {model.moi[2]:.2f} amu.Å2")
    model.masses = model.get_masses()
    model.M = model.masses.sum()
    model.moiM = model.moi/(model.M)
    moiM=model.moiM
    if not noOutput: print(f"Moments of inertia / M = {model.moiM[0]:.2f} {model.moiM[1]:.2f} {model.moiM[2]:.2f} amu.Å2")
    return moiM
   


#NEW
def get_moments_of_inertia_for_size(self, vectors=False): #from ASE but with mass modification
    '''
    Get the moments of inertia along the principal axes with
    mass normalisation.The three principal moments of inertia are computed from the
    eigenvalues of the symmetric inertial tensor.
    Args:
        vectors (bool, optional): If True, returns both eigenvalues and eigenvectors.
                                  If False, returns only eigenvalues (default: False).
    Returns:
        evals (np.ndarray): Principal moments of inertia (3 values).
        evecs (np.ndarray, optional): Principal axes (3x3 matrix, columns are eigenvectors).

    Notes:
        Periodic boundary conditions are ignored. 
        Units of the moments of inertia are angstrom**2.
    '''
    com = self.get_center_of_mass()
    positions = self.get_positions()
    #number_atoms=len(positions)
    positions -= com  # translate center of mass to origin
    masses = self.get_masses()

    # Initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(self)):
        x, y, z = positions[i]
        m = 1
        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    Itensor = np.array([[I11, I12, I13],
                        [I12, I22, I23],
                        [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(Itensor) #valeurs propes de la matrice 
    if vectors:
        return evals, evecs.transpose()
    else:
        return evals


def moi_size(model: Atoms, # normalized moment of inertia with masses=1
        noOutput: bool=False,
       ):
    '''
    Get the moments of inertia along the principal axes with mass normalization to acces  size informations.
    The three principal moments of inertia are computed from the eigenvalues of the symmetric inertial tensor. 
    
    Note:
        Units of the moments of inertia are angstrom**2.
    '''

    model.moi_size_all = get_moments_of_inertia_for_size(model)
    positions = model.get_positions()
    number_atoms=len(positions)
    model.moi_size = model.moi_size_all/(number_atoms)
    if not noOutput: print(f"Moments of inertia with mass=1/ M = {model.moi_size[0]:.2f} {model.moi_size[1]:.2f} {model.moi_size[2]:.2f} Å2")
    return [model.moi_size[0],model.moi_size[1],model.moi_size[2]]
  


#######################################################################
######################################## Geometry optimization

def optimizeEMT(model: Atoms, saveCoords=True, pathway="./coords/model", fthreshold=0.05):
    """
    Optimize the geometry of an atomic system using the EMT potential 
    and the Quasi-Newton algorithm.

    Args:
        model (ase.Atoms): Atomic system to optimize.
        saveCoords (bool, optional): If True, saves the optimized coordinates. Default is True.
        pathway (str, optional): Path where to save the trajectory and final structure. Default is "./coords/model".
        fthreshold (float, optional): Convergence threshold for forces (in eV/Å). Default is 0.05.

    Returns:
        ase.Atoms: Optimized atomic model.
    """
    # from varname import nameof, argname
    import numpy as np
    from ase.io import write
    from ase import Atoms
    from ase.calculators.emt import EMT
    chrono = timer(); chrono.chrono_start()
    vID.centerTitle(f"ase EMT calculator & Quasi Newton algorithm for geometry optimization")
    model.calc=EMT()
    model.get_potential_energy()
    from ase.optimize import QuasiNewton
    dyn = QuasiNewton(model, trajectory=pathway+'.opt')
    dyn.run(fmax=fthreshold)
    if saveCoords:
        write(pathway+"_opt.xyz", model)
        print(f"{fg.BLUE}Optimization steps saved in {pathway+'_.opt'} (binary file){fg.OFF}")
        print(f"{fg.RED}Optimized geometry saved in {pathway+'_opt.xyz'}{fg.OFF}")
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return model

#######################################################################
######################################## Planes & Directions
def planeFittingLSF(coords: np.float64,
                    printErrors: bool=False,
                    printEq: bool=True):
    '''
    Least-square fitting of the equation of a plane ux + vy + wz + h = 0
    that passes as close as possible to a set of 3D points
    Args:
        - coords (np.ndarray): array with shape (N,3) that contains the 3 coordinates for each of the N points
        - printErrors (bool): if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed 
        - printEq (bool): if True, prints equation.
    Returns:
        numpy array([u v w h])
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
    if printEq: print(f"bare solution: {u:.5f} x + {v:.5f} y + {w:.5f} z + {h:.5f} = 0")
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
    return np.array([u,v,w,h]).real

def convertuvwh2hkld(plane: np.float64,
                     prthkld: bool=True):
    '''
    Converts an ux + vy + wz + h = 0 equation of a plane, where u, v, w and h can be real numbers as 
    an hx + ky + lz + d = 0 equation, where h, k, and l are all integer numbers
    Args:
        - plane (np.ndarray): [u v w h]  
        - prthkld (bool): does print the result by default 
    Returns
        hkld (np.ndarray): [h k l d]
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
                    printEq: bool=True,
                    printErrors: bool=False):
    '''
    Context: finding the Miller indices of a plane, if relevant.
    Consists in a least-square fitting of the equation of a plane hx + ky + lz + d = 0
    that passes as close as possible to a set of 3D points
    Args:
        coords (np.ndarray): array with shape (N,3) that contains the 3 coordinates for each of the N points
        printErrors (bool): if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed
    Returns:
        plane (np.ndarray): [h k l d], where h, k, and l are as close as possible to integers
    '''
    plane = planeFittingLSF(coords,printErrors,printEq)
    plane = convertuvwh2hkld(plane, printEq)
    return plane

def shortestPoint2PlaneVectorDistance(plane:np.ndarray,
                                      point:np.ndarray):
    '''
    Returns the shortest distance, d, from a point X0 to a plane p (projection of X0 on p = P), as well as the PX0 vector 
    Args:
        plane (np.ndarray): [u v w h] definition of the p plane 
        point (np.ndarray): [x0 y0 z0] coordinates of the X0 point 
    Returns:
        v,d (tuple): the PX0 vector and ||PX0||
    '''
    t = (plane[3] + np.dot(plane[0:3],point))/(plane[0]**2+plane[1]**2+plane[2]**2)
    v = -t*plane[0:3]
    d = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    return v, d

def Pt2planeSignedDistance(plane,point):
    '''
    Returns the orthogonal distance of a given point X0 to the plane p in a metric space (projection of X0 on p = P), 
    with the sign determined by whether or not X0 is in the interior of p with respect to the center of gravity [0 0 0]
    Args:
        - plane (numpy array): [u v w h] definition of the P plane 
        - point (numpy array): [x0 y0 z0] coordinates of the X0 point 
    Returns:
        the signed modulus ±||PX0||
    '''
    sd = (plane[3] + np.dot(plane[0:3],point))/np.sqrt(plane[0]**2+plane[1]**2+plane[2]**2)
    return sd

def planeAtVertices(coordVertices: np.ndarray,
                    cog: np.ndarray):
    '''
    Returns the equation of the plane defined by vectors between the center of gravity (cog) and each vertex of a polyhedron
    and that is located at the vertex
    Args:
        coordVertices (np.ndarray): coordinates of the vertices ((nvertices,3) numpy array)
        cog (np.ndarray): center of gravity of the NP
    Returns:
        np.array(plane): the (cog-nvertices)x3 coordinates of the plane 
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
    Given a former [a,b,c,d] plane as input, d is recalculated so that the plane passes through P0,
    a known point on the plane
    Args:
        -plane (np.ndarray): array [a b c d]
        -P0 (np.ndarray): array with P0 coordinates [x0 y0 z0]
    Returns 
        planeAtp: [a b c -(ax0+by0+cz0)]
    '''
    d = np.dot(plane[0:3],P0)
    planeAtP = plane.copy()
    planeAtP[3] = -d
    return planeAtP

def normalizePlane(p):
    import numpy as np
    '''
    Normalizes the [a,b,c,d] coordinates of a plane
    - input: plane [a,b,c,d]
    returns [a/norm,b/norm,c/norm,d/norm] where norm=dsqrt(a**2+b**2+c**2)
    '''
    return p/normOfV(p[0:3])

def point2PlaneDistance(point: np.float64,
                              plane: np.float64):
    """
    Compute the shortest distance between a point and a plane in 3D space.
    Args:
        point (np.ndarray): A 3D point as [x, y, z].
        plane (np.ndarray): A plane defined by [A, B, C, D] where Ax + By + Cz + D = 0.
     Returns:
        float: The shortest distance from the point to the plane.
    """
    import numpy as np
    from numpy.linalg import norm
    distance = abs(np.dot(point,plane[0:3]) + plane[3]) / norm(plane[0:3])
    return distance

def AngleBetweenVV(lineDV,planeNV):
    '''
    Returns the angle, in degrees, between two vectors
    '''
    import numpy as np
    ldv = np.array(lineDV)
    pnv = np.array(planeNV)
    numerator = np.dot(ldv,pnv)
    denominator = normOfV(lineDV)*normOfV(planeNV)
    if denominator == 0:
        alpha = np.NaN
    else:
        alpha = 180*np.arccos(np.clip(numerator/denominator,-1,1))/np.pi
    return alpha

def signedAngleBetweenVV(v1,v2,n):
    '''
    returns, between [0°,360°] the signed angle, in degrees, between two vectors
    n is the normal of the plane formed by the two vectors
    '''
    import numpy as np
    cosTh = np.dot(v1,v2)
    sinTh = np.dot(np.cross(v1,v2),n)
    angle = np.rad2deg(np.arctan2(sinTh,cosTh))
    if angle >= 0: return angle
    else: return 360+angle

def normal2MillerPlane(Crystal,MillerIndexes,printN=True):
    '''
    returns the normal direction to the plane defined by h,k,l Miller indices is defined as [n1 n2 n3] = (hkl) x G*,
    where G* is the reciprocal metric tensor (G* = G-1)

    the convertuvwh2hkld() function applied here converts real plane indexes to integers
    '''
    normal = MillerIndexes@Crystal.Gstar
    normal = np.append(normal,0.0) #trick because convertuvwh2hkld() converts (u v w h) planes
    normalI = convertuvwh2hkld(normal,False)[0:3]
    if printN: 
        print(f"Normal to the ({MillerIndexes[0]:2} {MillerIndexes[1]:2} {MillerIndexes[2]:2}) user-defined plane > [{normal[0]: .3e} {normal[1]: .3e} {normal[2]: .3e}]",\
              f" = [{normalI[0]: .2f} {normalI[1]: .2f} {normalI[2]: .2f}]")
    return normalI

def isPlaneParrallel2Line(v1,v2,tol=1e-5):
    '''
    returns a boolean
    a line direction vector and a plane are parallel if the |angle| between the line and the normal vector of the plane is 90°
    '''
    return np.abs(np.abs(AngleBetweenVV(v1,v2)) - 90) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 270) < tol 

def isPlaneOrthogonal2Line(v1,v2,tol=1e-5):
    '''
    returns a boolean
    a line direction vector and a plane are orthogonal if the |angle| between the line and the normal vector of the plane is 0° or 180°
    '''
    return np.abs(AngleBetweenVV(v1,v2)) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 180) < tol

def areDirectionsOrthogonal(v1,v2,tol=1e-6):
    '''
    returns a boolean
    lines are orthogonal if the |angle| between their direction vector is 90°
    '''
    return np.abs(np.abs(AngleBetweenVV(v1,v2)) - 90) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 270) < tol

def areDirectionsParallel(v1,v2,tol=1e-6):
    '''
    returns a boolean
    lines are orthogonal if the |angle| between their direction vector is 0° or 180°
    '''
    return np.abs(AngleBetweenVV(v1,v2)) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 180) < tol

def returnPlaneParallel2Line(V, shift=[1,0,0], debug = False):
    '''
    returns the [a b c] parameters for a plane parallel to the input direction
    (d must be found separately)

    algorithm:
        - choose any arbitrary vector not parallel to V[i,j,k] such as V[i+1,j,k]
        - calculate the vector perpendicular to both of these, i.e. the cross product
        - this is the normal to the plane, i.e. you directly obtain the equation of the plane ax+by+cz+d = 0, d being indeterminate
        (to find d, it would be necessary to provide an (x0,y0,z0) point that does not belong to the line, hence d = -ax0-by0-cz0)
    '''
    arbV = np.array(V.copy())
    arbV = arbV + np.array(shift)
    plane = np.cross(V,arbV)
    if areDirectionsParallel(V,arbV): sys.exit(f"Error in returnPlaneParallel2Line(): plane {V} is parallel to {arbV}. "\
                                               f"Are you sure of your data?\n(this function wants to return an equation for a plane parallel to the direction V = {V}.\n"\
                                               f" Play with the shift variable - current problematic value = {shift})")
    if debug: print(areDirectionsParallel(V,arbV), V, arbV, "cross product = ",plane)
    return plane

def planeRotation(Crystal: Atoms,
                  refPlane,
                  rotAxis,
                  nRot=6,
                  debug: bool=False,
                  noOutput: bool=False
                 ):
    '''
    returns an array with planes obtained by rotating the reference plane around the input axis
    - input: 
        - Crystal = Crystal object
        - refPlane = plane to rotate
        - nRot = rotation angle is 360°/nRot
        - rotAxis = rotation axis
        - debug = normalized planes are printed
    '''
    pRef = np.array([refPlane])
    aRot = np.array([rotAxis])
    msg = f"Projection of the ({pRef[0][0]: .2f} {pRef[0][1]: .2f} {pRef[0][2]: .2f}) reference truncation plane around the "\
          f"[{rotAxis[0]: .2f}  {rotAxis[1]: .2f}  {rotAxis[2]: .2f}] axis, after projection in the cartesian coordinate system"
    if not noOutput: vID.centertxt(msg,bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    pRefCart = lattice_cart(Crystal,pRef,True,printV=not noOutput)
    rotAxisCart = lattice_cart(Crystal,aRot,True,printV=not noOutput)
    msg = f"{nRot}th order rotation around {rotAxisCart[0][0]: .2f} {rotAxisCart[0][1]: .2f} {rotAxisCart[0][2]: .2f}"\
          f"of the ({pRefCart[0][0]: .2f} {pRefCart[0][1]: .2f} {pRefCart[0][2]: .2f}) truncation plane"
    if not noOutput: vID.centertxt(msg,bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    planesCart = []
    for i in range(0,nRot):
        angle = i*360/nRot
        # print("rot around z    = ",RotationMol(pRefCart[0],angle,'z'))
        x = rotationMolAroundAxis(pRefCart[0],angle,rotAxisCart[0])
        # print("rot around axis = ",x)
        planesCart.append(x)
    if (debug): print(np.array(planesCart))
    if not noOutput: vID.centertxt(f"Just for your knowledge: indexes of the {nRot} normal directions to the truncation planes after projection to the {Crystal.cif.cell.get_bravais_lattice()} unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    planesHCP = lattice_cart(Crystal,np.array(planesCart),False,printV=not noOutput)
    if debug:
        vID.centertxt(f"Normalized HCP planes",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        for i,p in enumerate(planesHCP):
            print(i,normV(p))
        print()
        vID.centertxt(f"Normalized cartesian planes",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    return np.array(planesCart)

def alignV1WithV2_returnR(v1,v2=np.array([0, 0, 1])):
    """
    returns the rotation matrix [rMat] between two vectors using scipy, so that rMat@v1 is aligned with v2
    uses the align_vectors function of scipy.spatial.transform, align_vectors(a, b)
    in which when a single vector is given for a and b, the shortest distance rotation that aligns b to a is returned
    - input:
        - two vectors given as numpy arrays, in the order v1, v2
    - returns the (3,3) rotation matrix that aligns v1 with v2
    """
    from scipy.spatial.transform import Rotation
    import numpy as np
    import sys
    v1 = np.reshape(v1, (1, -1))
    v2 = np.reshape(v2, (1, -1))
    rMat = Rotation.align_vectors(v2, v1)
    rMat = rMat[0].as_matrix()
    v1_rot = rMat@v1[0]
    aligned = np.allclose(v1_rot / np.linalg.norm(v1_rot), v2 / np.linalg.norm(v2))
    if not aligned: sys.exit(f"Was unable to align {v1} with {v2}. Check your data")
    return rMat

def rotateMoltoAlignItWithAxis(coords,axis,targetAxis=np.array([0, 0, 1])):
    '''
    returns coordinates after rotation made to align axis with targetAxis
    - input:
        - coords = natoms x 3 numpy array
        - axis, targetAxis = directions given under the form [u,v,w]
    - returns a (natoms,3) numpy array
    '''
    import numpy as np
    if isinstance(axis, list):
        axis = np.array(axis)
    if isinstance(targetAxis, list):
        targetAxis = np.array(targetAxis)
    rMat = alignV1WithV2_returnR(axis,targetAxis)
    return np.array(rMat@coords.transpose()).transpose()

def setdAsNegative(planes):
    """
    input:
        - array of planes
    returns each initial plane [a b c d] as [-a -b -c -d] if d is positive
    """
    for i,p in enumerate(planes):
        if p[3] > 0:
            p = -p
            planes[i] = p
    return planes

#######################################################################
######################################## cut above planes
def calculateTruncationPlanesFromVertices(planes, cutFromVertexAt, nAtomsPerEdge, debug=False, noOutput=False):
    n = int(round(1/cutFromVertexAt))
    if not noOutput: print(f"factor = {cutFromVertexAt:.3f} ▶ {round(nAtomsPerEdge/n)} layer(s) will be removed, starting from each vertex")

    trPlanes = []
    for p in planes:
        pNormalized =normalizePlane(p.copy())
        pNormalized[3] =  pNormalized[3] - pNormalized[3]*cutFromVertexAt
        trPlanes.append(pNormalized)
        if (debug and not noOutput):
            print("normalized original plane = ",normalizePlane(p))
            print("cut plane = ",pNormalized,"... norm = ",normOfV(pNormalized[0:3]))
            print("signed distance between original plane and origin = ",Pt2planeSignedDistance(p,[0,0,0]))
            print("signed distance between cut plane and origin = ",Pt2planeSignedDistance(pNormalized,[0,0,0]))
            print("pcut/pRef = ",Pt2planeSignedDistance(pNormalized,[0,0,0])\
                                /Pt2planeSignedDistance(p,[0,0,0]))
        if not noOutput: print(f"Will remove atoms just above plane "\
              f"{pNormalized[0]:.2f} {pNormalized[1]:.2f} {pNormalized[2]:.2f} d:{pNormalized[3]:.3f}")
    return np.array(trPlanes)    

def truncateAboveEachPlane(planes: np.ndarray,
                           coords,
                           debug: bool=False,
                           delAbove: bool=True,
                           noOutput: bool=False):
    '''
    - input: 
        - planes = numpy array with all [u v w d] plane definitions
        - coords = (N,3) numpy array will all coordinates
        - delAbove = if True (default) delete atoms that lie above the planes + eps = 1e-4. Delete atoms below the
                     planes otherwise (use with precaution, could return no atoms as a function of their definition)
        - noOutput = do not print any message
    - returns the indexes of the atoms that are above each input planes
    '''

    delAtoms = []

    eps =1e-3
    
    for p in planes:
        for i,c in enumerate(coords):
            signedDistance = Pt2planeSignedDistance(p,c)
            if delAbove and signedDistance > eps:
                delAtoms.append(i)
            elif not delAbove and signedDistance < eps:
                delAtoms.append(i)
        # print(keptAtoms)
        if debug and not noOutput:
            for a in delAtoms:
                print(f"@{a+1}",end=',')
            print("",end='\n')
    delAtoms = np.array(delAtoms)
    delAtoms = np.unique(delAtoms)
    # if (debug): plot3D()
    return delAtoms

def truncateAbovePlanes(planes: np.ndarray,
                        coords: np.ndarray,
                        allP: bool=False,
                        delAbove: bool = True,
                        debug: bool=False,
                        noOutput: bool=False,
                        eps: float=1e-3,
                        depth_max: float=None):
    '''
    - input: 
        - planes = numpy array with all [u v w d] plane definitions
        - coords = (N,3) numpy array will all coordinates
        - allP = deleted atoms must lie simultaneously above ALL individual planes (default: False)
        - delAbove = if True (default) delete atoms that lie above the planes + eps = 1e-3 (default). Delete atoms below the
                     planes otherwise (use with precaution, could return no atoms as a function of their definition)
        - debug: if True (default is False) print atoms that match the allP/delAbove planes conditions
        - noOutput = do not print any message
        - eps: atom-to-plane signed distance threshold (default 1e-3)
    - returns an N-dimension boolean array that tells which atoms above each input planes (allP = False) 
      or above all input planes at the same time (allP=True) (opposite if delAbove is False)
    '''

    import numpy as np
    if not noOutput: vID.centertxt(f"Plane truncation (all planes condition: {allP}, delete above planes: {delAbove}, initial number of atoms = {len(coords)})",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    
    if allP:
        delAtoms = np.ones(len(coords), dtype=bool)
    else:
        delAtoms = np.zeros(len(coords), dtype=bool)
    nOfDeletedAtoms = 0
    for p in planes:
       
        if allP:
            delAtomsP = np.ones(len(coords), dtype=bool)
        else:
            delAtomsP = np.zeros(len(coords), dtype=bool)
        for i,c in enumerate(coords):
            signedDistance = Pt2planeSignedDistance(p,c)
            if delAbove and allP:
                delAtoms[i] = delAtoms[i] and signedDistance > eps
            elif delAbove and not allP:
                delAtoms[i] = delAtoms[i] or signedDistance > eps
                delAtomsP[i] = signedDistance > eps 
            elif not delAbove and allP:
                delAtoms[i] = delAtoms[i] and signedDistance < -eps
            elif not delAbove and not allP:
                delAtoms[i] = delAtoms[i] or signedDistance < -eps 
                delAtomsP[i] = signedDistance < -eps 
        nOfDeletedAtoms = np.count_nonzero(delAtoms)
        nOfDeletedAtomsP = np.count_nonzero(delAtomsP)
        if debug and not allP:
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtomsP} atoms deleted")
            for i,a in enumerate(delAtomsP):
                if a: print(f"@{i+1}",end=',')
            print("",end='\n')
        if debug and allP:
            print("allP is True => deletion of all atoms that simultaneously lie above/below each plane")
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtoms} atoms deleted")
            for i,a in enumerate(delAtoms):
                if a: print(f"@{i+1}",end=',')
            print("",end='\n')
    delAtoms = np.array(delAtoms)
    if not noOutput: 
        if delAbove:
            print(f"{len(coords)-np.count_nonzero(delAtoms)} atoms lie below the plane(s)")
        else:
            print(f"{np.count_nonzero(delAtoms)} atoms lie below the plane(s)")
    return delAtoms

def returnPointsThatLieInPlanes(planes: np.ndarray,
                                coords: np.ndarray,
                                debug: bool=False,
                                threshold: float=1e-3,
                                noOutput: bool=False,
                               ):
    """
    Finds all points (atoms) that lie within the given planes based on a signed distance criterion.

    Args:
        planes (np.ndarray): A 2D array where each row represents a plane equation [a, b, c, d] for the plane ax + by + cz + d = 0.
        coords (np.ndarray): A 2D array where each row is the coordinates of an atom [x, y, z].
        debug (bool, optional): If True, prints additional debugging information. Defaults to False.
        threshold (float, optional): The tolerance for the distance to the plane to consider a point as lying in the plane. Defaults to 1e-3.
        noOutput (bool, optional): If True, suppresses the output messages. Defaults to False.

    Returns:
        np.ndarray: A boolean array where True indicates that the atom lies in one of the planes.
    """
    import numpy as np
    if not noOutput: vID.centertxt(f"Find all points that lie in the given planes",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    AtomsInPlane = np.zeros(len(coords), dtype=bool)
    for p in planes:
        for i,c in enumerate(coords):
            signedDistance = Pt2planeSignedDistance(p,c)
            AtomsInPlane[i] = AtomsInPlane[i] or np.abs(signedDistance) < threshold
        nOfAtomsInPlane = np.count_nonzero(AtomsInPlane)
        if debug:
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfAtomsInPlane} atoms lie in the planes")
            for i,a in enumerate(delAtoms):
                if a: print(f"@{i+1}",end=',')
            print("",end='\n')
    AtomsInPlane = np.array(AtomsInPlane)
    if not noOutput: print(f"{np.count_nonzero(AtomsInPlane)} atoms lie in the plane(s)")
    return AtomsInPlane

def deleteElementsOfAList(t,
                          list2Delete: bool):
    '''
    returns a new list
    input:
        - t: list or table
        - list2Delete = list of booleans. list2Delete[i] = True ==> t[i] is deleted 
    '''
    import numpy as np
    if len(t) != len(list2Delete): sys.exit("the input list and the array of booleans must have the same dimension. Check your code")
    if type(t) == list: 
        tloc = np.array(t.copy())
    else:
        tloc = t.copy()
    tloc = np.delete(tloc,list2Delete,axis=0)
    return list(tloc)

#######################################################################
######################################## coupling with Jmol & DebyeCalculator
def saveCoords_DrawJmol(asemol, prefix, scriptJ="", boundaries=False, noOutput=True):
    from pyNanoMatBuilder import data
    path2Jmol = data.pyNMBvar.path2Jmol
    fxyz = "./figs/"+prefix+".xyz"
    writexyz(fxyz, asemol)
    # jmolscript = 'cpk 0; wireframe 0.025; script "./figs/script-facettes-3-4RuLight.spt"; facettes34rulight; draw * opaque; color atoms black; set zShadePower 1; set specularPower 80; pngon; write image 1024 1024 ./figs/'
    if not boundaries:
        jmolscript = scriptJ + '; frank off; cpk 0; wireframe 0.05; script "./figs/script-facettes-345PtLight.spt"; facettes345ptlight; draw * opaque;'
    else:
        jmolscript = scriptJ + '; frank off; cpk 0; wireframe 0.0; draw * opaque;'
    jmolscript = jmolscript + 'set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    jmolcmd="java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz + " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    if not noOutput: print(jmolcmd)
    os.system(jmolcmd)

def DrawJmol(mol,prefix,scriptJ=""):
    path2Jmol = '/usr/local/src/jmol-14.32.50'
    fxyz = "./figs/"+mol+".xyz"
    jmolscript = scriptJ + '; frank off; set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    jmolcmd="java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz + " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    print(jmolcmd)
    os.system(jmolcmd)




#######################################################################
######################################## Function that writes xyz and cif files



def writexyz(filename: str,
             atoms: Atoms,
             wa: str='w'):
    '''
    Simple xyz writing, with atomic symbols/x/y/z and no other information sometimes misunderstood by some utilities, such as DebyeCalculator.
    '''
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
        line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
    with open(filename,'w') as file:
        file.write(line2write)

def writexyz_generalized_archimedean(self,structure, path, instance_class,number, noOutput:bool=True):
    """
    Function that creates xyz and cif files of Arcimedeans NPs containing their main information 
    in a dictionary and their xyz coordinates.

    Parameters:
        - structure (str): "anatase", "rutile", "alpha", etc.
        - path (str): Path where xyz/cif files will be created.
        - instance_class (object): Instance of the platonic class.
        - number (int): Used to track the size.
        - noOutput (bool): If False, prints details about the files created.

    Notes:
        - Dimensions are in Å, MOI are in amu.Å², normalized MOI in Å².
        - Filenames follow the format: Element_structure_shape_number_0000000.xyz
    """  
   
    # Verify if the path exists
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory '{path}'does not exist.")

    # Extract attributes from the class to write the dictionnary and the name of the file
    if isinstance(instance_class,object):
        number2=0
        crystalStructure=self.crystal_type
        metadata = instance_class.__dict__.copy() # Get all attributes
        element=instance_class.element
        shape= instance_class.shape
        NP=instance_class.NP
        element_array=NP.get_chemical_symbols()

        # Moment of inertia
        MOI=instance_class.moi #amu.angs²
        MOI_normalized=instance_class.moisize  #angs²
        dim_MOI=np.round(np.array(instance_class.dim),3) #angs
        
        # Main dimensions, secondary dimensions and truncature
        if instance_class.shape=='fccTrOh' or  instance_class.shape=='fccTrCube':
            n_atoms_per_edges_1=instance_class.nAtomsPerEdge
            truncated= True
            number_truncated_atoms= int(n_atoms_per_edges_1*0.33) #gives the number of truncated atoms per edges
            radius1=round(instance_class.radiusCircumscribedSphere(),3) #angs
            radius2=round(instance_class.radiusInscribedSphere(),3) #angs
        if instance_class.shape=='fccCubo' :
            n_atoms_per_edges_1=instance_class.nShell+1
            truncated= False
            number_truncated_atoms= None
            radius1=round(instance_class.radiusCircumscribedSphere(),3) #angs
            radius2=round(instance_class.radiusInscribedSphere(),3) #angs 
        if instance_class.shape=='fccTrTd' :
            n_atoms_per_edges_1=instance_class.nAtomsPerEdge
            truncated= True
            number_truncated_atoms= int(n_atoms_per_edges_1*0.33)
            radius1=round(instance_class.radiusCircumscribedSphere,3) #angs
            radius2=round(instance_class.radiusInscribedSphere,3) #angs
        edge_length_1=instance_class.edgeLength()
        main_dim=np.array([2*radius1,2*radius2])

        # Total number of atoms
        number_atoms=int(instance_class.nAtoms)

        # Composition
        composition={}
        for elements in element_array:
            if elements in composition:
                composition[elements]+=1
            else:
                composition[elements]=1
                
        coord=NP.get_positions()
        natoms=len(element_array) 
        
        # Write the xyz file
        filename_xyz= f"{element}_{structure}_{shape}_{'{:07d}'.format(number)}_{'{:07d}'.format(number2)}.xyz"
        full_path = os.path.join(path, filename_xyz)
        if not noOutput :
            print(f' \x1B[3m xyz file created:{full_path} \x1B[0m')
            
        line2write='%d \n'%natoms
        dictionnary = {
            'composition': composition,
            'crystal structure': crystalStructure,
            'shape': shape,
            'MOI': MOI,
            'MOInormalized': MOI_normalized,
            'MOI_dim': dim_MOI,
            'main_dim': main_dim,
            'secondary_dim': { 
                'edges': {
                    'edge_1': {'length': edge_length_1, 'atoms_per_edge': n_atoms_per_edges_1},
                    'edge_2': {'length': None, 'atoms_per_edge': None}}},  # Only one type of edges for platonic NPs
            'truncation': truncated,
            # 'inscribed_sphere_radius': radius2,
            # 'circumscribed_sphere_radius': radius1,
            'number_of_atoms': number_atoms,
            'wulff': False,
            'crystallization': {
                'state': 'crystalline',
                'type': 'monocrystalline',
                'subtype': 'random'
            }
            }
        
        line2write+='%s \n'%dictionnary

        # Write the coordinates
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(full_path,'w') as file:
            file.write(line2write)


        # Write the cif files using the function write from ASE.io
        filename_cif = filename_xyz.replace('.xyz', '.cif')
        new_path = os.path.join(path, filename_cif)
        if not noOutput :
            print(f' \x1B[3m cif file created:{new_path}\x1B[0m')
        write(new_path, instance_class.NP)   
        
    else :
        print('Objet is not a class instance')

def writexyz_generalized_catalan(self,structure, path, instance_class,number, noOutput:bool=True):
    """
    Function that creates xyz and cif files of catalan NPs containing their main informations  
    in a dictionary and their xyz coordinates.

    Parameters:
        - structure (str): "anatase", "rutile", "alpha", etc.
        - path (str): Path where xyz/cif files will be created.
        - instance_class (object): Instance of the platonic class.
        - number (int): Used to track the size.
        - noOutput (bool): If False, prints details about the files created.

    Notes:
        - Dimensions are in Å, MOI are in amu.Å², normalized MOI in Å².
        - Filenames follow the format: Element_structure_shape_number_0000000.xyz
    """  
   
    # Verify if the path exists
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory '{path}'does not exist.")

    # Extract attributes from the class to write the dictionnary and the name of the file
    if isinstance(instance_class,object):
        number2=0
        crystalStructure=self.crystal_type
        metadata = instance_class.__dict__.copy() # Get all attributes
        element=instance_class.element
        shape= instance_class.shape
        NP=instance_class.NP
        element_array=NP.get_chemical_symbols()

        # Moment of inertia
        MOI=instance_class.moi #amu.angs²
        MOI_normalized=instance_class.moisize  #angs²
        dim_MOI=np.round(np.array(instance_class.dim),3) #angs
       
          
        
        # Main and secondary dimensions
        radius1=round(instance_class.radiusCircumscribedSphere,3) #angs
        radius2=round(instance_class.radiusInscribedSphere,3) #angs
        main_dim=np.array([2*radius1,2*radius2])
        n_atoms_per_edges_1=instance_class.nShell+1
        edge_length_1=instance_class.edgeLength()
        truncated= False
        number_truncated_atoms= None

        # Total number of atoms
        number_atoms=int(instance_class.nAtoms)

        # Composition
        composition={}
        for elements in element_array:
            if elements in composition:
                composition[elements]+=1
            else:
                composition[elements]=1

        coord=NP.get_positions()
        natoms=len(element_array) 
        
        # Write the xye file
        filename_xyz= f"{element}_{structure}_{shape}_{'{:07d}'.format(number)}_{'{:07d}'.format(number2)}.xyz"
        full_path = os.path.join(path, filename_xyz)
        if not noOutput :
            print(f' \x1B[3m xyz file created:{full_path} \x1B[0m')
            
        line2write='%d \n'%natoms

        dictionnary = {
            'composition': composition,
            'crystal structure': crystalStructure,
            'shape': shape,
            'MOI': MOI,
            'MOInormalized': MOI_normalized,
            'MOI_dim': dim_MOI,
            'main_dim': main_dim,
            'secondary_dim': { 
                'edges': {
                    'edge_1': {'length': edge_length_1, 'atoms_per_edge': n_atoms_per_edges_1},
                    'edge_2': {'length': None, 'atoms_per_edge': None}}},  # Only one type of edges for catalan NPs
            'truncation': truncated,
            # 'inscribed_sphere_radius': radius2,
            # 'circumscribed_sphere_radius': radius1,
            'number_of_atoms': number_atoms,
            'wulff': False,
            'crystallization': {
                'state': 'crystalline',
                'type': 'monocrystalline',
                'subtype': 'random'
            }
            }
        
        line2write+='%s \n'%dictionnary

        # Write the coordinates
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(full_path,'w') as file:
            file.write(line2write)


        # Write the cif files using the function write from ASE.io
        filename_cif = filename_xyz.replace('.xyz', '.cif')
        new_path = os.path.join(path, filename_cif)
        if not noOutput :
            print(f' \x1B[3m cif file created:{new_path}\x1B[0m')
        write(new_path, instance_class.NP)   
        
    else :
        print('Objet is not a class instance')

def writexyz_generalized_otherNPs(self,structure, path, instance_class,number, noOutput:bool=True):
    """
    Function that creates xyz and cif files of other NPs containing their main informations
    in a dictionary and their xyz coordinates.

    Parameters:
        - structure (str): "anatase", "rutile", "alpha", etc.
        - path (str): Path where xyz/cif files will be created.
        - instance_class (object): Instance of the platonic class.
        - number (int): Used to track the size.
        - noOutput (bool): If False, prints details about the files created.

    Notes:
        - Dimensions are in Å, MOI are in amu.Å², normalized MOI in Å².
        - Filenames follow the format: Element_structure_shape_number_0000000.xyz
    """  
   
    # Verify if the path exists
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory '{path}'does not exist.")

    # Extract attributes from the class to write the dictionnary and the name of the file
    if isinstance(instance_class,object):
        number2=0
        crystalStructure=self.crystal_type
        metadata = instance_class.__dict__.copy() # Get all attributes
        element=instance_class.element
        shape= instance_class.shape
        NP=instance_class.NP
        element_array=NP.get_chemical_symbols()

        # Moment of inertia
        MOI=instance_class.moi #amu.angs²
        MOI_normalized=instance_class.moisize  #angs²
        dim_MOI=np.round(np.array(instance_class.dim),3) #angs
        
        # Main dimensions
        radius1=round(instance_class.radiusCircumscribedSphere,3) #angs
        radius2=round(instance_class.radiusInscribedSphere,3) #angs
        main_dim=np.array([2*radius1,2*radius2])
       
        # Secondary dimensions
        n_atoms_per_edges_1=instance_class.nAtomsPerEdge+1
        edge_length_1=instance_class.Rnn*(instance_class.nAtomsPerEdge)
        n_atoms_per_edges_2=instance_class.nLayer
        edge_length_2=instance_class.interLayerDistance*(instance_class.nLayer-1)
        truncated= False
        number_truncated_atoms= None
 
        # Total number of atoms
        number_atoms=int(instance_class.nAtoms)

        # Composition
        composition={}
        for elements in element_array:
            if elements in composition:
                composition[elements]+=1
            else:
                composition[elements]=1

        coord=NP.get_positions()
        natoms=len(element_array) 
        
        # Write the file
        filename_xyz= f"{element}_{structure}_{shape}_{'{:07d}'.format(number)}_{'{:07d}'.format(number2)}.xyz"
        full_path = os.path.join(path, filename_xyz)
        if not noOutput :
            print(f' \x1B[3m xyz file created:{full_path} \x1B[0m')
            
        line2write='%d \n'%natoms

        dictionnary = {
            'composition': composition,
            'crystal structure': crystalStructure,
            'shape': shape,
            'MOI': MOI,
            'MOInormalized': MOI_normalized,
            'MOI_dim': dim_MOI,
            'main_dim': main_dim,
            'secondary_dim': { 
                'edges': {
                    'edge_1': {'length': edge_length_1, 'atoms_per_edge': n_atoms_per_edges_1},
                    'edge_2': {'length': edge_length_2, 'atoms_per_edge': n_atoms_per_edges_2}}},  # Only one type of edges for catalan NPs
            'truncation': truncated,
            # 'inscribed_sphere_radius': radius2,
            # 'circumscribed_sphere_radius': radius1,
            'number_of_atoms': number_atoms,
            'wulff': False,
            'crystallization': {
                'state': 'crystalline',
                'type': 'monocrystalline',
                'subtype': 'random'
            }
            }
        
        line2write+='%s \n'%dictionnary

        # Write the coordinates
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(full_path,'w') as file:
            file.write(line2write)


        # Write the cif files using the function write from ASE.io
        filename_cif = filename_xyz.replace('.xyz', '.cif')
        new_path = os.path.join(path, filename_cif)
        if not noOutput :
            print(f' \x1B[3m cif file created:{new_path}\x1B[0m')
        write(new_path, instance_class.NP)   
        
    else :
        print('Objet is not a class instance')



def writexyz_generalized_platonic(self,structure, path, instance_class,number, noOutput:bool=True):
    """
    Function that creates xyz and cif files of platonic NPs containing their main informations
    in a dictionary and their xyz coordinates.

    Parameters:
        - structure (str): "anatase", "rutile", "alpha", etc.
        - path (str): Path where xyz/cif files will be created.
        - instance_class (object): Instance of the platonic class.
        - number (int): Used to track the size.
        - noOutput (bool): If False, prints details about the files created.

    Notes:
        - Dimensions are in Å, MOI are in amu.Å², normalized MOI in Å².
        - Filenames follow the format: Element_structure_shape_number_0000000.xyz
    """  
   
    # Verify if the path exists
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory '{path}'does not exist.")

    # Extract attributes from the class to write the dictionnary and the name of the file
    if isinstance(instance_class,object):
        number2=0
        crystalStructure=self.crystal_type
        metadata = instance_class.__dict__.copy() # Get all attributes
        element=instance_class.element
        shape= instance_class.shape
        NP=instance_class.NP
        element_array=NP.get_chemical_symbols()

        # Moment of inertia
        MOI=instance_class.moi #amu.angs²
        MOI_normalized=instance_class.moisize  #angs²
        dim_MOI=np.round(np.array(instance_class.dim),3) #angs
        
        # Main dimensions
        radius1=round(instance_class.radiusCircumscribedSphere(),3) #angs
        radius2=round(instance_class.radiusInscribedSphere(),3) #angs
        main_dim=np.array([2*radius1,2*radius2])
        
        # Secondary dimensions
        if instance_class.shape=='regfccOh' or instance_class.shape=='cube' :
            n_atoms_per_edges_1=instance_class.nOrder+1
        if instance_class.shape=='regIco' or instance_class.shape=='regDD':
            n_atoms_per_edges_1=instance_class.nShell+1
        if instance_class.shape=='regfccTd' :
            n_atoms_per_edges_1=instance_class.nLayer
        edge_length_1=instance_class.edgeLength()
        
        # Total number of atoms
        number_atoms=int(instance_class.nAtoms)
        
        # Composition
        composition={}
        
        for elements in element_array:
            if elements in composition:
                composition[elements]+=1
            else:
                composition[elements]=1

        coord=NP.get_positions()
        natoms=len(element_array) 
        
        # Write the xyz file
        filename_xyz= f"{element}_{structure}_{shape}_{'{:07d}'.format(number)}_{'{:07d}'.format(number2)}.xyz"
        full_path = os.path.join(path, filename_xyz)
        if not noOutput :
            print(f' \x1B[3m xyz file created:{full_path} \x1B[0m')
            
        line2write='%d \n'%natoms

        dictionnary = {
            'composition': composition,
            'crystal structure': crystalStructure,
            'shape': shape,
            'MOI': MOI,
            'MOInormalized': MOI_normalized,
            'MOI_dim': dim_MOI,
            'main_dim': main_dim,
            'secondary_dim': { 
                'edges': {
                    'edge_1': {'length': edge_length_1, 'atoms_per_edge': n_atoms_per_edges_1},
                    'edge_2': {'length': None, 'atoms_per_edge': None}}},  # Only one type of edges for platonic NPs
            'truncation': False,
            # 'inscribed_sphere_radius': radius2,
            # 'circumscribed_sphere_radius': radius1,
            'number_of_atoms': number_atoms,
            'wulff': False,
            'crystallization': {
                'state': 'crystalline',
                'type': 'monocrystalline',
                'subtype': 'random'
            }
            }
        
        line2write+='%s \n'%dictionnary

        # Write the coordinates
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(full_path,'w') as file:
            file.write(line2write)


        # Write the cif files using the function write from ASE.io
        filename_cif = filename_xyz.replace('.xyz', '.cif')
        new_path = os.path.join(path, filename_cif)
        if not noOutput :
            print(f' \x1B[3m cif file created:{new_path}\x1B[0m')
        write(new_path, instance_class.NP)   
        
    else :
        print('Objet is not a class instance')


def writexyz_generalized_crystals(self,structure,path,instance_class_crystals, number,noOutput:bool=True):
    '''
    Function that creates xyz and cif files of crystals NPs containing their main informations
    in a dictionary and their xyz coordinates.
    Parameters:
        - structure (str): "anatase", "rutile", "alpha", etc.
        - path (str): Path where xyz/cif files will be created.
        - instance_class (object): Instance of the platonic class.
        - number (int): Used to track the size.
        - noOutput (bool): If False, prints details about the files created.
    
    Notes:
        - Dimensions are in Å, MOI are in amu.Å², normalized MOI in Å².
        - Filenames follow the format: Element_structure_shape_number_0000000.xyz
            
    '''
    # Verify if the path does exist
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory '{path}'does not exist.")
    
    # Extract attributes from the class to write the dictionnary and the name of the file
    if isinstance(instance_class_crystals,object):
        element=self.cif_name.split()[0]
        if structure== None : #example for NaCl, get the lattice for the name of the file
            structure=self.crystal_type
        
        crystalStructure=self.crystal_type
        number2=0 # indicator for data augmentation in the files names
        metadata = instance_class_crystals.__dict__.copy() # Get all attributes
        shape= instance_class_crystals.shape
        
        # 1) is it a wulff shape ? 2) if it's a wire : get nRot (number of edges of the cross section) and refplanewire (plane rotated to create the wire) 
        nRot=None
        plane_rotated=None
        if "Wulff" in shape: 
            shape=shape.split(':')[1]
            wulff= True
            if "hcpwire" in shape :
                nRot=int(6)
                plane_rotated=np.array(instance_class_crystals.surfacesWulff)
                
        elif "wire" in shape :
            wulff= False
            nRot=instance_class_crystals.nRotWire
            plane_rotated=instance_class_crystals.refPlaneWire
        else : 
            wulff= False
     
        NP=instance_class_crystals.NP
        element_array=NP.get_chemical_symbols()

        # Moment of inertia
        MOI=np.round(instance_class_crystals.moi,3) #amu.angs²
        MOI_normalized=np.round(instance_class_crystals.moisize,3)  #angs²
        dim_MOI=np.round(np.array(instance_class_crystals.dim),3) #angs

        #Truncation:
        if 'tr' in shape:
            truncated= True
 
        else:
            truncated= False

        # Total number of atoms
        number_atoms=int(instance_class_crystals.nAtoms)
        
        # Main dimensions
        radius1=round(instance_class_crystals.radiusCircumscribedSphere,3) #angs
        #diam1=round(radius1*2*0.1,2)
        radius2=round(instance_class_crystals.radiusInscribedSphere,3) #angs
        main_dim=np.array([2*radius1,2*radius2])

        # Composition
        composition={}
 
        for elements in element_array:
            if elements in composition:
                composition[elements]+=1
            else:
                composition[elements]=1

        coord=NP.get_positions()
        natoms=len(element_array) 
        
        # Write the xyz file 
        if nRot==None :
            filename_xyz= f"{element}_{structure}_{shape}_{'{:07d}'.format(number)}_{'{:07d}'.format(number2)}.xyz"
        else :
            filename_xyz= f"{element}_{structure}_{shape}_{nRot}_{'{:07d}'.format(number)}_{'{:07d}'.format(number2)}.xyz"
        #filename_xyz= f"{element}_{structure}_{shape}_{diam1}.xyz"
        full_path = os.path.join(path, filename_xyz)
        if not noOutput :
            print('xyz file created:',full_path)
        line2write='%d \n'%natoms
   
        dictionnary = {
            'composition': composition,
            'crystal structure': crystalStructure,
            'shape': shape,
            'MOI': MOI,
            'MOInormalized': MOI_normalized,
            'MOI_dim': dim_MOI,
            'main_dim': main_dim,
            'secondary_dim': { 
                'edges': {
                    'edge_1': {'length': None, 'atoms_per_edge': None},
                    'edge_2': {'length': None, 'atoms_per_edge': None}}},
            'truncation':  truncated,
            # 'inscribed_sphere_radius': radius2,
            # 'circumscribed_sphere_radius': radius1,
            'number_of_atoms': number_atoms,
            'wulff': wulff,
            'crystallization': {
                'state': 'crystalline',
                'type': 'monocrystalline',
                'subtype': 'random'
            },
            'wire_description': {
                'nRot': nRot,
                'plane_rotated': plane_rotated
            }
            }


        line2write+='%s \n'%dictionnary
        
        # Write the coordinates
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(full_path,'w') as file:
            file.write(line2write)
        
        # Write the cif files using the function write from ASE.io
        filename_cif = filename_xyz.replace('.xyz', '.cif')
        filename_script = filename_xyz.replace('.xyz', '.script')
        new_path = os.path.join(path, filename_cif)
        new_path_script = os.path.join(path, filename_script)

        if not noOutput :
            print('cif file created:',new_path)
            print('script file created:',new_path_script)
        write(new_path, instance_class_crystals.NP) 
        with open(new_path_script, 'w') as f: f.write(instance_class_crystals.jMolCS)
        
    else :
        print('Object is not a class instance')

def writexyz_generalized_johnson(self,structure, path, instance_class,number, noOutput:bool=True):
    """
    Function that creates xyz and cif files of johnson NPs containing their main information
    in a dictionary and their xyz coordinates.

    Parameters:
        - structure (str): "anatase", "rutile", "alpha", etc.
        - path (str): Path where xyz/cif files will be created.
        - instance_class (object): Instance of the platonic class.
        - number (int): Used to track the size.
        - noOutput (bool): If False, prints details about the files created.

    Notes:
        - Dimensions are in Å, MOI are in amu.Å², normalized MOI in Å².
        - Filenames follow the format: Element_structure_shape_number_0000000.xyz
    """  
   
    # Verify if the path exists
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory '{path}'does not exist.")

    # Extract attributes from the class to write the dictionnary and the name of the file
    if isinstance(instance_class,object):
        number2=0
        crystalStructure=self.crystal_type
        metadata = instance_class.__dict__.copy() # Get all attributes
        element=instance_class.element
        shape= instance_class.shape
        NP=instance_class.NP
        element_array=NP.get_chemical_symbols()

        # Moment of inertia
        MOI=instance_class.moi #amu.angs²
        MOI_normalized=instance_class.moisize  #angs²
        dim_MOI= None #angs  
        
        # Main dimensions: radiusCircumscribedSphere and radiusInscribedSphere
        radius1=round(instance_class.radiusCircumscribedSphere,3) #angs
        radius2=round(instance_class.radiusInscribedSphere,3) #angs
        main_dim=np.array([2*radius1,2*radius2])
        composition={}

        # Secondary dimensions and truncation
        if instance_class.shape=='fcctbp' :

            n_atoms_per_edges_1=instance_class.nAtomsPerEdge
            edge_length_1=instance_class.edgeLength()
            n_atoms_per_edges_2=None 
            edge_length_2=None
            truncated= False
            number_truncated_atoms= None
        if instance_class.shape=='epbpyM' :
            # Truncation
            if not instance_class.Marks== int(0):
                truncated= True
                number_truncated_atoms= instance_class.Marks
            else:
                truncated= False
                number_truncated_atoms= None
            
            # n_atoms_per_edges_1=instance_class.sizeP+1-instance_class.Marks # number of atoms per edges of pentagonal section after truncation
            n_atoms_per_edges_1=instance_class.nAtomsPerEdgeOfPC_after_truncation()
            edge_length_1=instance_class.edgeLength_after_truncation('PC')
            # n_atoms_per_edges_2=instance_class.sizeE+1 # number of atoms per edges of elongated part
            n_atoms_per_edges_2=instance_class.nAtomsPerEdgeOfEP
            edge_length_2=instance_class.edgeLength('EP')
   
        # Total number of atoms
        number_atoms=int(instance_class.nAtoms)

        
        for elements in element_array:
            if elements in composition:
                composition[elements]+=1
            else:
                composition[elements]=1

        coord=NP.get_positions()
        natoms=len(element_array) 
        
        # Write the xyz filename
        filename_xyz= f"{element}_{structure}_{shape}_{'{:07d}'.format(number)}_{'{:07d}'.format(number2)}.xyz"
        full_path = os.path.join(path, filename_xyz)
        if not noOutput :
            print(f' \x1B[3m xyz file created:{full_path} \x1B[0m')
            
        line2write='%d \n'%natoms

        dictionnary = {
            'composition': composition,
            'crystal structure': crystalStructure,
            'shape': shape,
            'MOI': MOI,
            'MOInormalized': MOI_normalized,
            'MOI_dim': dim_MOI,
            'main_dim': main_dim,
            'secondary_dim': { 
                'edges': {
                    'edge_1': {'length': edge_length_1, 'atoms_per_edge': n_atoms_per_edges_1},
                    'edge_2': {'length': edge_length_2, 'atoms_per_edge': n_atoms_per_edges_2}}},
            'truncation': truncated,
            # 'inscribed_sphere_radius': radius2,
            # 'circumscribed_sphere_radius': radius1,
            'number_of_atoms': number_atoms,
            'wulff': False,
            'crystallization': {
                'state': 'crystalline',
                'type': 'monocrystalline',
                'subtype': 'twinned'
            }
            }
        
        line2write+='%s \n'%dictionnary

        # Write the coordinates
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(full_path,'w') as file:
            file.write(line2write)


        # Write the cif files using the function write from ASE.io
        filename_cif = filename_xyz.replace('.xyz', '.cif')
        new_path = os.path.join(path, filename_cif)
        if not noOutput :
            print(f' \x1B[3m cif file created:{new_path}\x1B[0m')
        write(new_path, instance_class.NP)   
        
    else :
        print('Objet is not a class instance')


def create_data_csv(path_of_files, path_of_csvfiles, noOutput):
    """
    Function that extracts the dictionaries of specified files and creates new CSV files that contain them.
    Args:
        path_of_files: Path of the directory containing the files.
        path_of_csvfiles: Path of the directory where the CSV file will be created.
        noOutput: If True, suppresses print statements.
    Returns:
        None
    """
    import os
    number_created_files = 0

    # Check if the directories exist 
    if not os.path.isdir(path_of_files):
        raise FileNotFoundError(f"Directory '{path_of_files}' does not exist.")
    if not os.path.isdir(path_of_csvfiles):
        raise FileNotFoundError(f"Directory '{path_of_csvfiles}' does not exist.")

    # Loop through the files in the directory  
    for filename in os.listdir(path_of_files):
        if filename.endswith(".xyz") or filename.endswith(".csv"): 
            if not noOutput:
                print('File used:', filename)
            
            structure_source = os.path.join(path_of_files, filename)
            base_name = os.path.splitext(os.path.basename(structure_source))[0]
            csv_file_name = f"{base_name}_data.csv"
            csv_file = os.path.join(path_of_csvfiles, csv_file_name)
    
            # Write the dictionary in the new CSV file
            with open(csv_file, 'w') as csvfile:
                number_created_files += 1
                with open(structure_source, 'r') as file:
                    lignes = file.readlines()  # Correction ici
                    if len(lignes) >= 2: 
                        line_metadata = lignes[1].strip()
                        csvfile.write(f"{line_metadata}\n")

                        if not noOutput:
                            print(f"\n\033[1mNew file created: {csv_file}\033[0m\n")
                    else:
                        print(f"Error: File {filename} does not have enough lines.")

        else:
            if not noOutput:
                print(f"File format error: {filename}")
    
    print('Total CSV files created:', number_created_files)




def reduceHullFacets(Crystal: Atoms,
                     noOutput: bool=False,
                     tolAngle: float=2.0,
                    ):
    '''
    Reduces the facets of the crystal shape based on convex hull and coplanarity of the faces.
    
    Args:
        Crystal (Atoms): The crystal object containing the planes for the facet reduction.
        feasible_point (np.ndarray): A feasible point for half-space intersection. Default is [0, 0, 0].
        tolAngle (float): Tolerance angle to define coplanarity. Default is 2.0.
        noOutput (bool): If True, suppresses output to the console. Default is False.
        
    Returns:
        tuple: The vertices and reduced faces.
    
    Note:
        previous hull.simplices mut have been saved as Crystal.trPlanes
    '''
    from scipy.spatial import HalfspaceIntersection
    from scipy.spatial import ConvexHull
    import networkx as nx
    import scipy as sp
    
    cog = Crystal.cog
    # feasible_point = np.array([0,0,0])
    feasible_point=cog
    # print('------------------------------------------')
    # print("Crystal.trPlanes in reduceHullFacets")
    # print(Crystal.trPlanes)
    # print('------------------------------------------')
    hs = HalfspaceIntersection(Crystal.trPlanes, feasible_point,qhull_options="Q0")
    # hs = HalfspaceIntersection(Crystal.trPlanes, feasible_point)
    # print("hs.intersections")
    # print(hs.intersections)
    # vertices = hs.intersections + cog
    vertices = hs.intersections 
    hull = ConvexHull(vertices)

    faces = hull.simplices
    neighbours = hull.neighbors
    if not noOutput: vID.centertxt("Boundaries figure",bgc='#007a7a',size='14',weight='bold')
    if not noOutput: vID.centertxt(f"Half space intersection of the planes followed by a convex Hull analyzis",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    if not noOutput: print("Found:")
    if not noOutput: print(f"  - {len(hull.vertices)} convex Hull vertices")
    if not noOutput: print(f"  - {len(hull.simplices)} convex Hull simplices before reduction")
    
    def sortVCW(V,C):
        '''
        Sort the vertices of a planar polygon clockwise
        - input:
            - V = list of vertices of a given polygon
            - C = coordinates of all vertices
        '''
        coords = []
        for v in V: coords.append(C[v])
        cog = np.mean(coords,axis=0)
        radialV = coords-cog
        angle = []
        V = list(V)
        normal = planeFittingLSF(np.array(coords),False,False)
        for i in range(len(radialV)):
            angle.append(signedAngleBetweenVV(radialV[0],radialV[i],normal[0:3]))
        ind = np.argsort(angle)
        Vs = np.array(list(V))
        return Vs[ind]
    
    def isCoplanar(p1,p2,tolAngle=tolAngle):
        '''
        Check if two planes p1 and p2 are coplanar.
        '''
        angle = AngleBetweenVV(p1[0:3],p2[0:3])
        return (abs(angle) < tolAngle or abs(angle-180) <= tolAngle)
        
    def reduceFaces(F,coordsVertices):
        '''
        Function to reduce the number of faces by merging coplanar ones
        '''
        flatten = lambda l: [item for sublist in l for item in sublist]
    
        # create a graph in which nodes represent triangles
        # nodes are connected if the corresponding triangles are adjacent and coplanar
        G = nx.Graph()
        G.add_nodes_from(range(len(F)))
        pList = []
        for i,f in enumerate(F):
            planeDef = []
            for v in f:
                planeDef.append(coordsVertices[v])
            planeDef = np.array(planeDef)
            pList.append(planeFittingLSF(planeDef,printErrors=False,printEq=False))
    
        for i,p1 in enumerate(pList):
            for n in neighbours[i]:
                p2 = pList[n]
                if isCoplanar(p1,p2):
                    G.add_edge(i,n)
        components = list(nx.connected_components(G))
        simplified = [set(flatten(F[index] for index in component)) for component in components]
    
        return simplified
        
    new_faces = reduceFaces(faces,vertices)
    new_facesS = []
    for i,nf in enumerate(new_faces):
        new_facesS.append(sortVCW(nf,vertices).tolist())
    if not noOutput: print(f"  - {len(new_faces)} facets after reduction")
    if not noOutput: print(f"New trPlanes saved in self.trPlanes")
    trPlanes = []
    for i,f in enumerate(new_faces):
        planeDef = []
        for v in f:
            planeDef.append(vertices[v])
        planeDef = np.array(planeDef)
        trPlanes.append(planeFittingLSF(planeDef,printErrors=False,printEq=False))
    Crystal.trPlanes = setdAsNegative(np.array(trPlanes))
    return vertices, new_facesS

def defCrystalShapeForJMol(Crystal: Atoms,
                           noOutput: bool=False,
                          ):
    """
    Generates a Jmol command to visualize the crystal shape based on the facets of the crystal.
    Args:
        Crystal (Atoms): The crystal structure object containing the facets and planes.
        noOutput (bool, optional): If True, suppresses the output of the command. Defaults to False.
    Returns:
        str: The Jmol command string for visualizing the crystal shape.
    """
    
    if Crystal.trPlanes is not None:
        ####################################################### Trying alpha shape algorithm for concave NPs
        # if Crystal.shape=='epbpyM':
        #     vertices, redFacets = Crystal.alpha_vertices, Crystal.alpha_faces
        #     if not noOutput: vID.centertxt("generating the jmol command line to view the crystal shape",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        #     cmd = ""
        #     for i,nf in enumerate(redFacets):
        #         cmd += "draw facet" + str(i) + " polygon "
        #         cmd += '['
        #         for at in nf:
        #             cmd+=f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
        #         cmd+="]; "
        #     cmd += "color $facet* translucent 70 [x828282]" 
        #     cmde = ""
        #     index = 0
        #     for nf in redFacets:
        #         nfcycle = np.append(nf,nf[0])
        #         for i, at in enumerate(nfcycle[:-1]):
        #             cmde += "draw line" + str(index) + " ["
        #             cmde += f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
        #             cmde += f"{{{vertices[nfcycle[i+1]][0]:.4f},{vertices[nfcycle[i+1]][1]:.4f},{vertices[nfcycle[i+1]][2]:.4f}}},"
        #             cmde += "] width 0.2; "
        #             index += 1
        #     cmde += "color $line* [xd6d6d6]; "
        #     cmd = cmde + cmd 
        # else:
        # ############################################################################################################

        vertices, redFacets = reduceHullFacets(Crystal, noOutput=noOutput)
        if not noOutput: vID.centertxt("generating the jmol command line to view the crystal shape",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        cmd = ""
        for i,nf in enumerate(redFacets):
            cmd += "draw facet" + str(i) + " polygon "
            cmd += '['
            for at in nf:
                cmd+=f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
            cmd+="]; "
        cmd += "color $facet* translucent 70 [x828282]" 
        cmde = ""
        index = 0
        for nf in redFacets:
            nfcycle = np.append(nf,nf[0])
            for i, at in enumerate(nfcycle[:-1]):
                cmde += "draw line" + str(index) + " ["
                cmde += f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
                cmde += f"{{{vertices[nfcycle[i+1]][0]:.4f},{vertices[nfcycle[i+1]][1]:.4f},{vertices[nfcycle[i+1]][2]:.4f}}},"
                cmde += "] width 0.2; "
                index += 1
        cmde += "color $line* [xd6d6d6]; "
        cmd = cmde + cmd


    
    else: #sphere, ellipsoid
        cmd = ""
    if not noOutput: print("Jmol command: ",cmd)
    return cmd

def saveCN4JMol(Crystal: Atoms,
                save2: str='CN.dat',
                Rmax: float=3.0,
                noOutput: bool=False,
                ):
    """
    Calculates the coordination number (CN) for a given crystal and generates a Jmol command for visualization.
    
    Args:
        Crystal (Atoms): The crystal structure object.
        save2 (str, optional): The filename to save the coordination numbers. Defaults to 'CN.dat'.
        Rmax (float, optional): The maximum distance for neighbors when calculating CN. Defaults to 3.0.
        noOutput (bool, optional): If set to True, suppresses the output. Defaults to False.
    
    Returns:
        None
    """
    import seaborn as sns
    
    # Calculate the coordination number (CN) using a k-D tree method
    nn,CN = kDTreeCN(Crystal,Rmax,noOutput=noOutput)
    CNmin = np.min(CN)
    CNmax = np.max(CN)
    with open(save2, 'w') as f:
        for cn in CN:
            f.write(str(cn) + "\n")
    if not noOutput:
        uniqueCN = np.unique(CN)
        nColors = len(uniqueCN)
        print(f"CN range = [{CNmin} - {CNmax}]")
        print(f"CN = {uniqueCN}")
        CNMax = 16
        # colors = sns.color_palette("tab20c", CNMax+1) #+1 because 0, just in case of an uncoordinated atom
        colorsFull = [
                  (255, 0, 0), (255, 255, 153), (255, 255, 0), (255, 204, 0),
                  (102, 255, 255), (51, 204, 255), (102, 153, 255), (249, 128, 130), 
                  (153, 255, 204), (0, 204, 153), (0, 134, 101), (0, 102, 102),
                  (51, 51, 255), (102, 51, 0), (0, 51, 102), (77, 77, 77),
                  (0, 0, 0)
                 ]
        colorsFull = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in colorsFull]
        path,file = os.path.split(save2)
        prefix=file.split(".")
        fileColors = "./" + path + "/" + prefix[0] + "colors.png"
        fileColorsFull = "./" + path + "/" + "CN_color_palette.png"
        colorNamesFull = np.array(range(0,CNMax+1))
        print("Full palette:")
        plotPalette(colorsFull,colorNamesFull,savePngAs=fileColorsFull)
        print(f"Palette specific to {prefix[0]}:")
        colors = []
        for c in uniqueCN:
            colors.append(colorsFull[c])
        plotPalette(colors,uniqueCN,savePngAs=fileColors)
        
        # Generate Jmol command for CN visualization
        print(f"{hl.BOLD}Jmol command:{hl.OFF}")
        # command = f"CN=load('{file}'); select all; "
        command = f"{{*}}.valence = load('{file}'); "
        colorScheme = ""
        for c in colorsFull:
            colorScheme = colorScheme + rgb2hex(c) + " "
        command = command + f"color atoms property valence 'colorCN' RANGE 0 {CNMax} ;"
        command = command + "label %2.0[valence]; color label yellow ; font label 24 ; set labeloffset 7 0;"
        print(f"color 'colorCN = {colorScheme}';")
        print(command)

def plotPalette(Pcolors, namePC, angle=0,savePngAs=None):
    '''
    plots a 1D palette colors, with names
    input:
        colors = 1D list with hex colors
        nameC = label for each color
        angle = rotation angle of the text
        saveAs = also saves the palette in a png file (default: None)
    '''
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.palplot(sns.color_palette(Pcolors))
    ax = plt.gca()

    for i, name in enumerate(namePC):
        ax.set_xticks(np.arange(len(namePC)))
        ax.tick_params(length=0)
        ax.set_xticklabels(namePC,weight='bold',size=10,rotation=angle)
    if (savePngAs is not None):
        plt.tight_layout()
        plt.savefig(savePngAs,dpi=600,transparent=True)
    plt.show()
    return

def rgb2hex(c,frac=True):
    """
    Converts an RGB color to its hexadecimal representation. 
    It has an optional frac argument to handle the case where the RGB values are provided as fractions
    (ranging from 0 to 1) or as integers (ranging from 0 to 255).
    """
    if frac:
        r = int(round(c[0]*255))
        g = int(round(c[1]*255))
        b = int(round(c[2]*255))
    else:
        r = c[0]
        g = c[1]
        b = c[2]
    return f"[x{r:02X}{g:02X}{b:02X}]"
#######################################################################
######################################## coordination numbers
def calculateCN(coords,Rmax):
    import os
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

def findNeighbours(coords,Rmax):
    '''
    for all atoms i, returns the list of all atoms j within an arbitrarily determined cutoff distance Rmax from atom i
    - input:
        - coords = numpy array with the N-atoms cartesian coordinates
        - Rmax = cutoff distance
    - returns:
        - list of lists (len(list[i]) = number of nearest neighbours of atom i)
    '''
    vID.centertxt(f"Building a table of nearest neighbours",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    chrono = timer(); chrono.chrono_start()
    nAtoms = len(coords)
    nn = [ [] for _ in range(nAtoms)]
    for i in range(nAtoms):
        for j in range(i):
            if RAB(coords,i,j) < Rmax:
                nn[i].append(j)
                nn[j].append(i)
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return nn

def printNeighbours(nn):
    '''
    prints the list of nearest neighbours of each atom
    - input:
        - nn = nearest neighbours given as a list of list - such as the nn provided by the neighbours() function
    '''
    for i,nni in enumerate(nn):
        print(f"Atom {i:6} has {len(nni):2} NN: {nni}")

def kDTreeCN(crystal: Atoms,
             Rmax: float=2.9,
             returnD: bool=False,
             noOutput: bool=False
            ):
    '''
    returns the nearest neighbour (nn) table, under the form of a list, as well as the number of NN per atom
    input:
        - (N,3) array of coordinates
        - Rmax, the NN threshold
        - distances between NN are returned as well if returnD is True
    '''
    from sklearn.neighbors import KDTree
    import numpy as np
    if noOutput: vID.centertxt(f"Building a table of nearest neighbours",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    if noOutput: chrono = timer(); chrono.chrono_start()
    coords = crystal.get_positions()
    tree = KDTree(coords)
    nn = []
    CN = []
    dNN = []
    for i,c in enumerate(coords):
        if returnD:
            l,d = tree.query_radius([c], r=3.0, return_distance=returnD)
            l = list(l[0])
            d = list(d[0])
        else:
            l =  list(tree.query_radius([c], r=3.0, return_distance=returnD)[0])
        if returnD: dNN.append(d)
        ipos = l.index(i)
        l.remove(i)
        if returnD: del(d[ipos])
        nn.append(l)
        CN.append(len(l))
    if noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    if returnD:
        return nn,CN,dNN
    else:
        return nn,CN

#######################################################################
######################################## symmetry
def reflection(plane,points,doItForAtomsThatLieInTheReflectionPlane=False):
    '''
    applies a mirror-image symmetry operation of an array of points w.r.t. a plane of symmetry
    - input:
        - plane = [u,v,w,d] parameters that define a plane
        - point = (N, 3) array of points
        - doItForAtomsThatLieInTheReflectionPlane = slef-explanatory
    - returns an (N, 3) array of mirror-image points
    '''
    import numpy as np
    pr = []
    eps = 1.e-4
    for p in points:
        vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane,p)
        if dp2plane >= eps and not doItForAtomsThatLieInTheReflectionPlane: # otherwise the point belongs to the reflection plane
            # print(dp2plane, vp2plane, p)
            ptmp = p+2*vp2plane
            pr.append(ptmp)
        else:
            ptmp = p+2*vp2plane
            pr.append(ptmp)
    return np.array(pr)

#######################################################################
######################################## rotation
def Rx(a):
    ''' returns the R/x rotation matrix'''
    import math as m
    return np.matrix([[ 1, 0           , 0   ],
                    [ 0, m.cos(a),-m.sin(a)],
                    [ 0, m.sin(a), m.cos(a)]])
  
def Ry(a):
    ''' returns the R/y rotation matrix'''
    import math as m
    return np.matrix([[ m.cos(a), 0, m.sin(a)],
                   [ 0           , 1, 0           ],
                   [-m.sin(a), 0, m.cos(a)]])
  
def Rz(a):
    ''' returns the R/z rotation matrix'''
    import math as m
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

def RotationMol(coords, angle, axis="z"):
    """
    Performs a rotation of the molecule's coordinates around a specified axis.

    Args:
    coords (numpy.ndarray): Coordinates of the molecule as a matrix (n x 3), where n is the number of atoms.
    angle (float): The angle of rotation in degrees.
    axis (str, optional): The axis around which to perform the rotation. Can be 'x', 'y', or 'z'. Default is 'z'.
    Returns:
        R[0] (numpy.ndarray): The coordinates of the molecule after rotation, as a matrix (n x 3).
    """
    import math as m
    angler = angle*m.pi/180
    if axis == 'x':
        R =  np.array(Rx(angler)@coords.transpose())
    elif axis == 'y':
        R =  np.array(Ry(angler)@coords.transpose())
    elif axis == 'z':
        R =  np.array(Rz(angler)@coords.transpose())
    return R[0]

def EulerRotationMol(coords, gamma, beta, alpha, order="zyx"):
    """
    Performs an Euler rotation of the molecule's coordinates.

    Args:
        coords (numpy.ndarray): Coordinates of the molecule as a matrix (n x 3), where n is the number of atoms.
        gamma (float): Angle gamma in degrees.
        beta (float):  Angle beta in degrees.
        alpha (float): Angle alpha in degrees.
        order (str, optional): The order of the Euler rotations. Default is "zyx".

    Returns:
        numpy.ndarray: The coordinates of the molecule after the Euler rotation, as a matrix (n x 3).
    """
    return np.array(EulerRotationMatrix(gamma,beta,alpha,order)@coords.transpose()).transpose()

def RotationMatrixFromAxisAngle(u,angle):
    """
    Generates a 3x3 rotation matrix from a unit vector representing the axis of rotation and a rotation angle.
    Args:
        u (numpy.ndarray): A unit vector representing the axis of rotation (3 elements).
        angle (float): The angle of rotation in degrees.

    Returns: (numpy.ndarray) A 3x3 rotation matrix.
    """
    import math as m
    a = angle*m.pi/180
    ux = u[0]
    uy = u[1]
    uz = u[2]
    return np.array([[m.cos(a)+ux**2*(1-m.cos(a))   , ux*uy*(1-m.cos(a))-uz*m.sin(a), ux*uz*(1-m.cos(a))+uy*m.sin(a)],
                      [uy*ux*(1-m.cos(a))+uz*m.sin(a), m.cos(a)+uy**2*(1-m.cos(a))   , uy*uz*(1-m.cos(a))-ux*m.sin(a)],
                      [uz*ux*(1-m.cos(a))-uy*m.sin(a), uz*uy*(1-m.cos(a))+ux*m.sin(a), m.cos(a)+uz**2*(1-m.cos(a))   ]])

def rotationMolAroundAxis(coords, angle, axis):
    '''
    Returns coordinates after rotation by a given angle around an [u,v,w] axis
    - input:
        - coords = natoms x 3 numpy array
        - angle = angle of rotation
        - axis = directions given under the form [u,v,w]
    - returns a numpy array
    '''
    normalizedAxis = normV(axis)
    return np.array(RotationMatrixFromAxisAngle(normalizedAxis,angle)@coords.transpose()).transpose()

#######################################################################
######################################## magic numbers
def magicNumbers(cluster,i):
    """
    • Certain collections of atoms are more preferred due to energy minimization and exhibiting
    stable structures and providing unique properties to the materials.
    • These collections of atom providing stable structures to the materials are called MAGIC NUMBER.
    
    This function uses specific formulas based on the type of nanocluster 
    (e.g., 'regfccOh', 'regIco', 'fccCube', etc.) to calculate a magic number based on the index i.
    
    Args:
        cluster (str): The type of nanocluster for which the magic number is calculated. 
                        It can be one of the following values: 
                        'regfccOh', 'regIco', 'regfccTd', 'regDD', 'fccCube', 
                        'bccCube', 'fccCubo', 'fccTrOh', 'fccTrCube', 'bccrDD', 
                        'fccdrDD', 'pbpy'.
        i (int): The index of the nanocluster size. It must be a positive integer greater than zero.

    Returns:
        float: The calculated magic number for the specified nanocluster type and index i.

    """
    match cluster:
        case 'regfccOh':
            mn = np.round((2/3)*i**3 + 2*i**2 + (7/3)*i + 1)
            return mn
        case 'regIco':
            mn = (10*i**3 + 11*i)//3 + 5*i**2 + 1
            return mn
        case 'regfccTd':
            mn = np.round(i**3/6 + i**2 + 11*i/6 + 1)
            return mn
        case 'regDD':
            mn = 10*i**3 + 15*i**2 + 7*i + 1
            return mn
        case 'fccCube':
            mn = 4*i**3 + 6*i*2 + 3*i + 1
            return mn
        case 'bccCube':
            mn = 2*i**3 + 3*i*2 + 3*i
            return mn
        case 'fccCubo':
            mn = np.round((10*i**3 + 11*i)/3 + 5*i**2 + 1)
            return mn
        case 'fccTrOh':
            mn = np.round(16*i**3 + 15*i**2 + 6*i + 1)
            return mn
        case 'fccTrCube':
            mn = np.round(4*i**3 + 6*i**2 + 3*i - 7)
            return mn
        case 'bccrDD':
            mn = 4*i**3 + 6*i**2 + 4*i + 1
            return mn
        case 'fccdrDD':
            mn = 8*i**3 + 6*i**2 + 2*i + 3
            return mn
        case 'pbpy':
            mn = 5*i**3/6 + 5*i**2/2 + 8*i/3 + 1
            return mn
        case _:
            sys.exit(f"The {cluster} nanocluster is unknown")

#######################################################################
######################################## Bravais
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

def lattice_cart(Crystal,vectors,Bravais2cart=True,printV=False):
    '''
    - input:
        - Crystal = Crystal object
        - vectors = vectors to project from the Bravais basis to the cartesian coordinate system (if Bravais2cart is True)
                         or to project from the cartesian coordinate system to the Bravais basis  (if Bravais2cart is False)
        - printV = boolean (default: False), prints the resulting vectors if True
    - returns an array of projected vectors
    '''
    import numpy as np
    unitcell = Crystal.ucUnitcell
    Vuc = Crystal.ucV
    if Bravais2cart:
        Vproj = (vectors@Vuc)
        B = 'B'
        E = 'C'
    else:
        VucInv = np.linalg.inv(Vuc)
        Vproj = (vectors@VucInv)
        B = 'C'
        E = 'B'
    if printV:
        print("Change of basis")
        for i,V in enumerate(vectors):
            Bstr = f"{V[0]: .2f} {V[1]: .2f} {V[2]: .2f}"
            Vp = Vproj[i]
            Estr = f"{Vp[0]: .2f} {Vp[1]: .2f} {Vp[2]: .2f}"
            print(f"({Bstr}){B} > ({Estr}){E}")
    return Vproj 

def G(Crystal): #https://fr.wikibooks.org/wiki/Cristallographie_g%C3%A9om%C3%A9trique/Outils_math%C3%A9matiques_pour_l%27%C3%A9tude_du_r%C3%A9seau_cristallin
    """
    Computes the metric tensor (G) of a crystal's unit cell.
    The metric tensor is calculated based on the unit cell parameters: 
    the lengths of the unit cell vectors (a, b, c) and the angles (alpha, beta, gamma) between them.

    Args:
        Crystal : object
    Returns:
        GG (numpy.ndarray): The 3x3 metric tensor (G) of the unit cell.
    """
    a = Crystal.ucUnitcell[0]
    b = Crystal.ucUnitcell[1]
    c = Crystal.ucUnitcell[2]
    alpha = Crystal.ucUnitcell[3]*np.pi/180.
    beta = Crystal.ucUnitcell[4]*np.pi/180.
    gamma = Crystal.ucUnitcell[5]*np.pi/180.
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    GG = np.array([[      a**2, a * b * cg, a * c * cb],
                  [a * b * cg,       b**2, b * c * ca],
                  [a * c * cb, b * c * ca,       c**2]])
    return GG

def Gstar(Crystal):
    """
    Computes the inverse of the metric tensor (G*) for a crystal's unit cell.

    Args:
        Crystal : object
    Returns:
        Gmat (numpy.ndarray): The 3x3 inverse metric tensor (G*) of the unit cell, represented as a numpy array.
    """
    Gmat = G(Crystal)
    return linalg.inv(Gmat)

#######################################################################
######################################## Misc for plots
def imageNameWithPathway(imgName):
    """
    Constructs the full file path for an image by joining the base directory with the image name.
    Args:
        imgName(str):  The name of the image file.
    Returns:
        imgNameWithPathway (str):The full file path to the image file.
    """
    path2image= os.path.join(pyNMB_location(),'figs')
    imgNameWithPathway = os.path.join(path2image,imgName)
    return imgNameWithPathway

def plotImageInPropFunction(imageFile):
    """
    Plots an image using matplotlib with no axes and a specified size.
    Args:
        imageFile : The path to the image file to be displayed.
    
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image = mpimg.imread(imageFile)
    plt.figure(figsize=(2, 10))
    plt.imshow(image,interpolation='nearest')
    plt.axis('off')
    plt.show()

#######################################################################
######################################## Core/surface identification / Convex Hull analysis
def coreSurface(Crystal: Atoms,
                threshold,
                noOutput=False,
               ):

    """
    Identifies the core and surface atoms of a crystal using Convex Hull analysis.

    Args:  
        Crystal : Atoms
        threshold (float): The threshold used to identify surface atoms.
        noOutput (bool, optional): If set to True, suppresses output during the analysis. Default is False.

    Returns:
        list: A list containing:
            - Hull vertices
            - Hull simplices
            - Hull neighbors
            - Hull equations (planes)
        surfaceAtoms (numpy.ndarray): The atomic positions of the atoms that lie on the surface of the crystal.
    """


    
    from ase.visualize import view
    from scipy.spatial import ConvexHull
    if not noOutput: vID.centertxt("Core/Surface analyzis",bgc='#007a7a',size='14',weight='bold')
    if not noOutput: chrono = timer(); chrono.chrono_start()
    coords = Crystal.NP.get_positions()
    if not noOutput: vID.centertxt(f"Convex Hull analyzis",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    hull = ConvexHull(coords)
    if not noOutput: print("Found:")
    if not noOutput: print(f"  - {len(hull.vertices)} vertices")
    if not noOutput: print(f"  - {len(hull.simplices)} simplices")
    Crystal.trPlanes = hull.equations
    # print("Crystal.trplanes inside coreSurface")
    # print(Crystal.trPlanes)
    # print(np.unique(Crystal.trPlanes, axis=0, return_counts=True))
    # print(Crystal.trPlanes.shape)
    #_ = defCrystalShapeForJMol(Crystal,noOutput=noOutput)
    # print("Crystal.trplanes inside coreSurface after defCrystalShapeForJMol")
    # print(Crystal.trPlanes)
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    if not noOutput: chrono = timer(); chrono.chrono_start()
    surfaceAtoms = returnPointsThatLieInPlanes(Crystal.trPlanes,coords,noOutput=noOutput,threshold=threshold)
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return [hull.vertices,hull.simplices,hull.neighbors,hull.equations],surfaceAtoms

#######################################################################
######################################## basic rdf profile
def rdf(NP: Atoms,
        dr: float=0.05,
        sigma: float=2,
        ncores: int=1,
        noOutput: bool=True
       ):
    '''
    rdf - g(r) - calculator for non-PBC systems
    arguments:
        - NP = ase Atoms object
        - dr = determines the spacing between successive radii over which g(r) is computed. Default: 0.05
        - sigma = standard deviation for Gaussian kernel. Default: 2
        - ncores = number of jobs to schedule for parallel processing (only used by query_ball_point() of scipy.spatial.KDTree). Default: 1
        - noOutput = do not print anything. Default: True

    returns:
        - r and g(r)

    wann know more? Read https://doi.org/10.1021/acs.chemrev.1c00237
    '''
    from ase.atoms import Atoms
    from ase.visualize import view
    from scipy.spatial import KDTree
    from scipy.spatial import distance
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    if not noOutput: vID.centertxt("Basic RDF profile calculation",bgc='#007a7a',size='14',weight='bold')
    com = NP.get_center_of_mass()
    # view(NP)
    coords = NP.get_positions()
    if not noOutput: chrono = timer(); chrono.chrono_start()
    tree = KDTree(coords)
    dist = distance.cdist(coords,[com])
    rMax = np.max(dist)
    dMax = 1.05*2*rMax
    radii = np.arange(dr, dMax, dr)
    if not noOutput: print(f"dMax = {dMax:.2f} (number of points = {len(radii)})")
    g_r = np.zeros(len(radii))
    dist = distance.pdist(coords)
    for ir, r in enumerate(radii):
        for i,c in enumerate(coords):
            neighbours = tree.query_ball_point(c,r,return_length=True,workers=ncores) - tree.query_ball_point(c,r-dr,return_length=True,workers=ncores)
            g_r[ir] += neighbours
    g_r = gaussian_filter1d(g_r,sigma=sigma,mode='nearest')
    g_r = np.divide(g_r,radii)
    peaks, _ = find_peaks(g_r)
    if not noOutput: print(f"First peak found at: {radii[peaks[0]]:.2f} Å. g(r) = {g_r[peaks[0]]:.2f}")
    g_r = g_r/g_r[peaks[0]]
    radii = radii/radii[peaks[0]]
    if not noOutput: print("(Intensity and position of the returned RDF profile normalized w.r.t. this first peak)")
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return radii, g_r, len(radii)

#######################################################################
######################################## simple file management utilities
def createDir(path2,forceDel=False):
    """
    Creates a directory at the specified path. If the directory already exists, it will either be left 
    unchanged or deleted and recreated based on the 'forceDel' argument.

    Args:
        path2 (str): The path where the directory should be created.
        forceDel (bool, optional): If set to True, will delete the existing directory and recreate it.
                                   Default is False.
    
    Returns:
        None
    """
    import os
    import shutil
    if os.path.isdir(path2) and not forceDel:
        print(f"{path2} already exists. No need to recreate it")
    if os.path.isdir(path2) and forceDel:
        print(f"{fg.RED}Previously created {path2} is deleted{fg.OFF}")
        shutil.rmtree(path2)
    if (os.path.isdir(path2) and forceDel) or not os.path.isdir(path2):
        print(f"{fg.BLUE}{path2} is created{fg.OFF}")
        os.mkdir(path2)


# New : for cylinder

def isnt_inside_cylinder(position, radius, radius_squared, half_height): 
    """
    Checks whether a given position is outside a cylinder defined by a radius and half height.
    The cylinder is aligned along the z-axis. If the position lies outside the circular base or 
    beyond the half height of the cylinder, the function returns True. Otherwise, it returns False.

    Args:
        position (tuple or list): The (x, y, z) coordinates of the point to check.
        radius (float): The radius of the cylinder's base.
        radius_squared (float): The square of the radius, for optimization purposes.
        half_height (float): Half the height of the cylinder (from the center along the z-axis).
    
    Returns:
        bool: Returns True if the position is outside the cylinder, False otherwise.
    """
    if abs(position[0])>radius or  abs(position[1])>radius or abs(position[2]) > half_height : #coord défini dans writexyz
        return True
    if position[0]**2+ position[1]**2 > radius_squared :
        return True
    return False



# NEW : MOI size based on shapes 


def MOI_shapes(self, noOutput) :
    """
    Function that computes the size of the NPs based on the principal moments of inertia normalized (eigen values).
    The size are either specific dimensions like the edge lengths or more general descriptors such as the diameter of
    the circumscribed sphere for symmetric NPs.
    
    Args:
        noOutput (bool, optional): If set to True, suppresses output during the analysis. Default is False.
    Returns:
        self.dim (numpy.ndarray): A 3-element array containing the sizes =[d1,d2,d3], with d1>d2>d3.
    Notes:
        The formulas for basic shapes are well-known; however, more complex shapes do not have exact formulas. 
        These are approximated by similar known shapes.
    """
    
    import math

    if self.MOIshape == 'ellipsoid': # https://scienceworld.wolfram.com/physics/MomentofInertiaEllipsoid.html
        self.dim[0] = 2*np.sqrt((5 *self.moisize[1] + 5 * self.moisize[2] - 5 * self.moisize[0]) / 2)
        self.dim[1] = 2*np.sqrt((5 * self.moisize[0] + 5 * self.moisize[2] - 5 * self.moisize[1]) / 2)
        self.dim[2] = 2*np.sqrt((5 * self.moisize[0] + 5 * self.moisize[1] - 5 * self.moisize[2]) / 2)
        if not noOutput:
            print(f"Diameters of the ellipsoid (calculated from MOI) { self.dim[0]* 0.1:.2f}  { self.dim[1] * 0.1:.2f}  { self.dim[2] * 0.1:.2f} nm")
            
    if self.MOIshape == 'sphere' or self.MOIshape=='fccTrOh': #wikipedia
        self.dim[0] = 2 * np.sqrt(5 / 2 * self.moisize[0]) 
        self.dim[1] = 2 * np.sqrt(5 / 2 * self.moisize[0])
        self.dim[2] = 2 * np.sqrt(5 / 2 * self.moisize[0])
        if not noOutput:
            print(f"Diameter of the sphere (calculated from MOI) = {self.dim[0] * 0.1:.2f} nm")
            
    if self.MOIshape == 'cylinder': #wikipedia
        self.dim[0] = np.sqrt(12 * self.moisize[1] - 6 * self.moisize[0]) #longest distance
        self.dim[1] = 2 * np.sqrt(2 * self.moisize[0]) 
        self.dim[2] = 2 * np.sqrt(2 * self.moisize[0])
        if not noOutput:
            print(f"Hieght of cylinder= {self.dim[0] * 0.1:.2f}, diameter of the cylinder= {self.dim[2] * 0.1:.2f} nm")
            
    if self.MOIshape == 'parallepiped': #wikipedia
        self.dim[0] = np.sqrt(6 *(self.moisize[1] + self.moisize[2] - self.moisize[0])) 
        self.dim[1] = np.sqrt(6 * (self.moisize[0] + self.moisize[2] - self.moisize[1])) 
        self.dim[2] = np.sqrt(6 * (self.moisize[0] + self.moisize[1] - self.moisize[2])) 
        if not noOutput:
            print(f"Size of the parallepiped (calculated from MOI)=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm")

    if  self.MOIshape == "wire" :  # wikipedia usual wire and hcp wire of predefinened wulff forms
        # if self.nRotWire==4 :
        #     self.dim[0]=np.sqrt(12*self.moisize[1]-6*self.moisize[0]) #longest distance
        #     self.dim[1]=np.sqrt(6*self.moisize[0])
        #     self.dim[2]=np.sqrt(6*self.moisize[0])
        #     if not noOutput:
        #         print(f"Size of the wire (calculated from MOI)=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm")
        # if self.nRotWire==6 or "hcpwire" in self.shape:
        #     self.dim[0]=np.sqrt(12*self.moisize[1]-6*self.moisize[0]) #longest distance
        #     self.dim[1]=2*np.sqrt(2*self.moisize[0])
        #     self.dim[2]=2*np.sqrt(2*self.moisize[0])
        #     if not noOutput:
        #         print(f"Size of the wire (calculated from MOI)=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm")

        
        R_polygon=math.sqrt((6*self.moisize[0])/(1+2*math.cos(math.pi/self.nRotWire)**2)) # https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/Polygons.pdf 
        
        # Main dimensions : height, circumscribed of the cross section
        self.dim[0]=np.sqrt(12*self.moisize[1]-6*self.moisize[0]) #height
        self.dim[1]=R_polygon  #circumscribed of the cross section
        self.dim[2]=R_polygon  #circumscribed of the cross section
        
        edge= 2*R_polygon*math.sin(math.pi/self.nRotWire) # https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/Polygons.pdf
        if not noOutput:
            print(f"Height of the nanoprism=  {self.dim[0]* 0.1:.2f} nm, radius of the circumscribed cross section= {self.dim[1]* 0.1:.2f} nm and edge of the nanoprism= { edge* 0.1:.2f}  nm")
 
    if self.MOIshape == 'cube' or self.MOIshape == 'fccTrCube' or self.MOIshape == 'fccCubo' : 
        if self.MOIshape == 'cube':
            self.dim[0]=np.sqrt(6*self.moisize[0])*math.sqrt(3)
            self.dim[1]=self.dim[0]
            self.dim[2]=self.dim[0]
        if self.MOIshape == 'fccCubo':
            self.dim[0]=np.sqrt(6*self.moisize[0])*math.sqrt(2)
            self.dim[1]=self.dim[0]
            self.dim[2]=self.dim[0]
        if self.MOIshape == 'fccTrCube':
            self.dim[0]=np.sqrt(6*self.moisize[0])*math.sqrt(1.5)
            self.dim[1]=self.dim[0]
            self.dim[2]=self.dim[0]
        if not noOutput:
            print(f"diameter of the circumscribed sphere of the cube (calculated from MOI)=  { self.dim[0]* 0.1:.2f}  nm")


    if self.MOIshape == 'Oh' or self.MOIshape=='regfccOh' :
        a=np.sqrt(10*self.moisize[0]) #arete https://www.vcalc.com/collection/?uuid=1a8912a2-f145-11e9-8682-bc764e2038f2
        self.dim[0] =2*a*math.sqrt(2)/2 #diameter of the circumscribed sphere https://en.wikipedia.org/wiki/Octahedron
        self.dim[1] = self.dim[0]
        self.dim[2] = self.dim[0]
        if not noOutput:
            print(f"Size of the octahedron (diameter of the circumscribed sphere) (calculated from MOI) :  { self.dim[0]* 0.1:.2f} nm")
            print(f"Edge  of the octahedron:  { a* 0.1:.2f}   nm")


    if self.MOIshape == "hcpsph" :  #https://mathworld.wolfram.com/Spheroid.html
        self.dim[0]= 2 *np.sqrt(5/2*self.moisize[2])#longest distance
        self.dim[1]=2 *np.sqrt(5/2*self.moisize[2])#longest distance
        #self.dim[2]=2 *np.sqrt(5*self.moisize[0]-5/2*self.moisize[2]) not working well

        #calculate c using basic formula
        positions=self.NP.get_positions()
        positions=self.NP.get_positions()
        x_min, y_min, z_min = positions.min(axis=0) #columns
        x_max, y_max, z_max = positions.max(axis=0) 
        self.dim[2]=z_max-z_min
        if not noOutput:
           print(f"Size of the spheroid (calculated from MOI)=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm") 
        

    if self.MOIshape == "bccrDD" or self.MOIshape == "regDD"or self.MOIshape == "fccdrDD" : 
        a=np.sqrt((150*self.moisize[0])/(39*((1+math.sqrt(5))/2)+28))  #arete https://www.vcalc.com/collection/?uuid=1a8912a2-f145-11e9-8682-bc764e2038f2
        self.dim[0] = 2*a*math.cos(36*math.pi/180)*math.sqrt(3) 
        self.dim[1] = self.dim[0]
        self.dim[2] = self.dim[0]
        #https://fr.wikipedia.org/wiki/Dod%C3%A9ca%C3%A8dre_r%C3%A9gulier#:~:text=Les%2020%20%C3%97%206%20%3D%2012,sur%20les%20faces%20du%20poly%C3%A8dre.
        if not noOutput:
            print(f"Size of the dodecahedron (diameter of the circumscribed sphere) (calculated from MOI) :  { self.dim[0]* 0.1:.2f} nm")
            print(f"Edge  of the dodecahedron:  { a* 0.1:.2f}   nm")

    if self.MOIshape == "regIco": 
        a=np.sqrt((10*self.moisize[0])/(((1+math.sqrt(5))/2)**2)) #https://www.vcalc.com/wiki/EmilyB/Moment+of+Inertia+-+Solid+Icosahedron
        self.dim[0] =a*math.sqrt(((1+math.sqrt(5))/2)*math.sqrt(5))#https://fr.wikipedia.org/wiki/Icosa%C3%A8dre
        self.dim[1] = self.dim[0]
        self.dim[2] = self.dim[0]
        if not noOutput:
            print(f"Size of the icosahedron (diameter of the circumscribed sphere) (calculated from MOI) :  { self.dim[0]* 0.1:.2f} nm")
            print(f"Edge  of the icosahedron:  { a* 0.1:.2f}   nm")
 
    if self.MOIshape == 'fcctbp':
        a = np.sqrt(20 * self.moisize[0]) - 5  
        H = 2 * a * np.sqrt(2/3)  # height
        D = np.sqrt(H**2 + (a * np.sqrt(3) / 3)**2)  # circumscribed sphere diameter
    
        self.dim[0] = D #longest dim
        self.dim[1] = D
        self.dim[2] = D
    
        if not noOutput:
            print(f"Size of the bipyramide (diameter circumscribed sphere) : {self.dim[0]*0.1:.2f} {self.dim[1]*0.1:.2f} {self.dim[2]*0.1:.2f} nm")
            print(f"Edge length of the bipyramide: {a*0.1:.2f} nm")
 
    if self.MOIshape== 'Td' or self.MOIshape== 'regfccTd'  :
        a=np.sqrt(20*self.moisize[0])-5 # side length https://www.vcalc.com/collection/?uuid=1a8912a2-f145-11e9-8682-bc764e2038f2, (-5 being a correction)
        self.dim[0] = 2*a*math.sqrt(3/8) # diameter of the circumscribed sphere: https://fr.wikipedia.org/wiki/T%C3%A9tra%C3%A8dre_r%C3%A9gulier
        self.dim[1] = self.dim[0]
        self.dim[2] = self.dim[0]
        if not noOutput:
            print(f"Size of the tetrahedron (diameter of the circumscribed sphere) :  { self.dim[0]* 0.1:.2f}  { self.dim[1] * 0.1:.2f}  { self.dim[2] * 0.1:.2f} nm")
            print(f"Edge  of the tetrahedron:  { a* 0.1:.2f}   nm")

    if  self.MOIshape== 'fccTrTd' :
        a=(np.sqrt(20*self.moisize[0])-5)*0.33 # side length https://www.vcalc.com/collection/?uuid=1a8912a2-f145-11e9-8682-bc764e2038f2, (-5 being a correction and 0.66 with the truncation)
        self.dim[0] = 2*a*math.sqrt(3/8) #diameter of the circumscribed sphere: https://fr.wikipedia.org/wiki/T%C3%A9tra%C3%A8dre_r%C3%A9gulier
        self.dim[1] = self.dim[0]
        self.dim[2] = self.dim[0]
        if not noOutput:
            print(f"Size of the tetrahedron (diameter of the circumscribed sphere) :  { self.dim[0]* 0.1:.2f}  { self.dim[1] * 0.1:.2f}  { self.dim[2] * 0.1:.2f} nm")
            print(f"Edge  of the tetrahedron:  { a* 0.1:.2f}   nm")
            
            

def Inscribed_circumscribed_spheres(self,noOutput):
    """
    Calculates the diameters of the inscribed and circumscribed spheres for the nanoparticle (NP) based on 
    its positions and the plane equations.

    The circumscribed sphere is the smallest sphere that completely encloses the NP, while the inscribed sphere 
    is the largest sphere that fits within the NP.

    Args:
        noOutput (bool, optional): If set to True, suppresses output during the analysis. Default is False.
    
    Returns:
        None: The function updates the object's attributes with the calculated diameters of the spheres.
    
    Notes:
        The circumscribed sphere radius is calculated as the maximum distance from the center of gravity 
        (COG) to the NP positions, and the inscribed sphere radius is calculated as the minimum distance 
        from the NP positions to the planes (based on Hull equations)
    """

    if self.shape=='ellipsoid':
        self.radiusInscribedSphere= min(self.ellipsoid_axes) 
        self.radiusCircumscribedSphere= max(self.ellipsoid_axes) 
    if self.shape=='sphere':
        self.radiusInscribedSphere= self.spheres_axes
        self.radiusCircumscribedSphere= self.spheres_axes
    else:
        distances = np.linalg.norm(self.NP.positions- self.cog, axis=1)
        self.radiusCircumscribedSphere= np.max(distances)
        distances = [
        abs(d) / np.sqrt(a**2 + b**2 + c**2)  
        for a, b, c, d in self.equations
        ]
        self.radiusInscribedSphere= np.min(distances)
        
    if not noOutput: vID.centertxt("Diameters of the inscribed and circumscribed sphere using the Hull equations",bgc='#007a7a',size='14',weight='bold')
    if not noOutput : 
        print(f"diameters of the circumscribed sphere: {self.radiusCircumscribedSphere * 2* 0.1:.2f}  {self.radiusCircumscribedSphere* 2 * 0.1:.2f}  {self.radiusCircumscribedSphere* 2* 0.1:.2f} nm")
        print(f"diameters of the inscribed sphere: { self.radiusInscribedSphere* 2* 0.1:.2f}  {self.radiusInscribedSphere* 2 * 0.1:.2f}  {self.radiusInscribedSphere * 2* 0.1:.2f} nm")

