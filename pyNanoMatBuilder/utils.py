import datetime
import importlib
import os
import pathlib
from pathlib import Path
import re
import time

import numpy as np
from scipy import linalg
import math
import sys
from . import visualID as vID
from .visualID import bg, fg, hl

from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase import io as ase_io
from ase.spacegroup import get_spacegroup
from ase.visualize import view

from pyNanoMatBuilder import data


#######################################################################
######################################## time
from datetime import datetime
import datetime, time
class timer:
    """
    Timer class to measure elapsed time in seconds and display it in hh:mm:ss ms.
    """

    def __init__(self):
        _start_time = None
        _end_time = None
        _chrono_start = None
        _chrono_stop = None

    # delay can be timedelta or seconds
    def hdelay_ms(self, delay):
        """
        Converts a delay into a human-readable format: hh:mm:ss ms.

        Args:
            delay: A timedelta object or a float representing a duration in seconds.
        Return:
            A formatted string in hh:mm:ss ms.
        """
        if type(delay) is not datetime.timedelta:
            delay = datetime.timedelta(seconds=delay)
        sec = delay.total_seconds()
        hh = sec // 3600
        mm = (sec // 60) - (hh * 60)
        ss = sec - hh * 3600 - mm * 60
        ms = (sec - int(sec)) * 1000
        return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'

    def chrono_start(self):
        """
        Starts the chrono.
        """
        global _chrono_start, _chrono_stop
        _chrono_start = time.time()

    # return delay in seconds or in humain format
    def chrono_stop(self, hdelay=False):
        """
        Stops the chrono and returns the elapsed time.
        """
        global _chrono_start, _chrono_stop
        _chrono_stop = time.time()
        sec = _chrono_stop - _chrono_start
        if hdelay:
            return self.hdelay_ms(sec)
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
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # ase analyzis
    system.ucUnitcell = system.cif.cell.cellpar()
    system.ucV = cellpar_to_cell(system.ucUnitcell)
    system.ucBL = system.cif.cell.get_bravais_lattice()
    # system.ucSG = get_spacegroup(system.cif, symprec=system.aseSymPrec) #deprecated
    system.ucVolume = system.cif.cell.volume
    system.ucReciprocal = np.array(system.cif.cell.reciprocal())
    system.ucFormula = system.cif.get_chemical_formula()
    system.G = G(system)
    system.Gstar = Gstar(system)

    # Pymatgen Symmetry Analysis (Replacing ase get_spacegroup)
    # Convert ASE Atoms to Pymatgen Structure
    pmg_struct = AseAtomsAdaptor.get_structure(system.cif)
    # Analyze symmetry
    sga = SpacegroupAnalyzer(pmg_struct, symprec=system.aseSymPrec)
    # Store the Spacegroup Info (Spglib format)
    # This replaces the old system.ucSG
    system.ucSG_symbol = sga.get_space_group_symbol()
    system.ucSG_number = sga.get_space_group_number()
    system.ucCrystalSystem = sga.get_crystal_system()

def print_ase_unitcell(system: Atoms):
    """
    Function that prints unitcell informations : chemical formula, bravais lattice, n° space group, cell parameters, volume, etc.
    
    Args:
        An instance of the Crystal class
    """
    unitcell = system.ucUnitcell
    bl = system.ucBL
    formula = system.ucFormula
    volume = system.ucVolume
    sg_symbol = system.ucSG_symbol
    sg_number = system.ucSG_number
    print(f"Bravais lattice: {bl}")
    print(f"Chemical formula: {formula}")
    print(f"Crystal family = {bl.crystal_family} (lattice system = {bl.lattice_system})")
    print(f"Name = {bl.longname} (Pearson symbol = {bl.pearson_symbol})")
    print(f"Variant names = {bl.variant_names}")
    print()
    print(
        f"From ase: space group number = {sg_number}      "
        f"Hermann-Mauguin symbol for the space group = {sg_symbol}"
    )
    print()
    print(
        f"a: {unitcell[0]:.3f} Å, b: {unitcell[1]:.3f} Å, "
        f"c: {unitcell[2]:.3f} Å. (c/a = {unitcell[2] / unitcell[0]:.3f})"
    )
    print(f"α: {unitcell[3]:.3f} °, β: {unitcell[4]:.3f} °, γ: {unitcell[5]:.3f} °")
    print()
    print(f"Volume: {volume:.3f} Å3")


def listCifsOfTheDatabase():
    """Display all CIF filenames in the database."""
    import pathlib
    import glob
    from ase.spacegroup import get_spacegroup
    import re

    path2cifFolder = os.path.join(pyNMB_location(), 'cif_database')
    print(f"path to cif database = {path2cifFolder}")

    sgITField = "_space_group_IT_number"
    sgHMField = "_symmetry_space_group_name_H-M"

    class Crystal:
        pass

    for cif in glob.glob(f'{path2cifFolder}/*.cif'):
        path2cifFile = pathlib.Path(cif)
        cifName = pathlib.Path(*path2cifFile.parts[-1:])
        vID.centertxt(f"{cifName}", size=14, weight='bold')
        cifContent = ase_io.read(cif)
        cifFile = open(cif, 'r')
        cifFileLines = cifFile.readlines()
        re_sgIT = re.compile(sgITField)
        re_sgHM = re.compile(sgHMField)
        for line in cifFileLines:
            if re_sgIT.search(line):
                sgIT = ' '.join(line.split()[1:])
            if re_sgHM.search(line):
                sgHM = ' '.join(line.split()[1:])
        cifFile.close()
        c = Crystal()
        c.cif = cifContent
        c.aseSymPrec = 1e-4
        returnUnitcellData(c)
        print_ase_unitcell(c)
        color = "vID.fg.RED"
        print()
        if int(sgIT) == c.ucSG.no:
            print(
                f"{vID.fg.GREEN}Symmetry in the cif file = {sgIT}   {sgHM}"
                f"{vID.hl.BOLD} in agreement with the ase symmetry analyzis{vID.fg.OFF}"
            )
        else:
            print(
                f"{vID.fg.RED}Symmetry in the cif file = {sgIT}   {sgHM}"
                f"{vID.hl.BOLD} disagrees with the ase symmetry analyzis{vID.fg.OFF}"
            )

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
    if not noOutput:
        vID.centertxt(
            "Scaling the unitcell",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    M = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    x = make_supercell(crystal.cif, M)
    nn, CN, R = kDTreeCN(x, 4.0, returnD=True)
    Rmin = min(R[0])
    scale = scaleDmin2 / Rmin
    if not noOutput:
        print(f"Unitcell lengths and atomic positions scaled by {scale:.3f} factor")
        print(f"New nearest neighbour distance = {scaleDmin2:.3f} Å")
    ucv = crystal.cif.cell.cellpar()
    ucv[0:3] = ucv[0:3] * scale
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

    if not noOutput:
        chrono = timer()
        chrono.chrono_start()
        vID.centertxt("Symmetry analysis", bgc='#007a7a', size='14', weight='bold')
        print(
            "Currently using the PointGroupAnalyzer class of pymatgen\n"
            "The analyzis can take a while for large compounds"
        )
        print()
    pmgmol = pmg.Molecule(aseobject.get_chemical_symbols(), aseobject.get_positions())
    pga = PointGroupAnalyzer(pmgmol, tolerance=0.6, eigen_tolerance=0.02, matrix_tolerance=0.2)
    pg = pga.get_pointgroup()
    if not noOutput:
        print(f"Point Group: {pg}")
        print(f"Rotational Symmetry Number = {pga.get_rotational_symmetry_number()}")
        chrono.chrono_stop(hdelay=False)
        chrono.chrono_show()
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
    spacegroup_number = self.ucSG_number  # space group number

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


def FindInterAtomicDist(self):
    """
    Computes the interatomic distance based on the Bravais lattice (fcc, bcc or hcp only).

    Returns:
        float: Interatomic distance
    """
    if self.crystal_type == 'fcc':
        d = self.parameters[0] * math.sqrt(2) / 2

    if self.crystal_type == 'bcc':
        d = self.parameters[0] * math.sqrt(3) / 2

    if self.crystal_type == 'hcp':
        d_a = self.parameters[0]
        d_c = self.parameters[2] / 2
        if d_a > d_c:  # if compact
            d = d_c
        if d_c > d_a:  # if not compact
            d = d_a

    return d

def extract_cif_info(self, cif_file):
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
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    # structure = read(cif_file)  # load structure with ase, not used ?
    # self.ucUnitcell[0]=a, self.ucUnitcell[1]=b, self.ucUnitcell[2]=c,
    # self.ucUnitcell[3]= α, etc

    # 1. Basic ASE extractions
    self.ucUnitcell = self.cif.cell.cellpar()
    self.parameters = self.cif.cell.lengths()
    self.ucBL = self.cif.cell.get_bravais_lattice()  # HEX, CUB, etc
    # self.ucSG = get_spacegroup(self.cif, symprec=float(1e-2)) #deprecated. Moved to pymatgen symmetry analyzer
    self.ucFormula = self.cif.get_chemical_formula()

    # Pymatgen Symmetry Analysis
    # Convert ASE Atoms to Pymatgen Structure
    pmg_struct = AseAtomsAdaptor.get_structure(self.cif)
    # Initialize the analyzer (symprec 1e-2 matches the original code)
    sga = SpacegroupAnalyzer(pmg_struct, symprec=1e-2)
    # Get trustworthy symmetry data
    self.ucSG_number = sga.get_space_group_number()
    self.ucSG_symbol = sga.get_space_group_symbol()

    self.crystal_type = get_crystal_type(self)
    return {
        # 'crystal_name': self.ucFormula,
        'cif_path': cif_file,
        'crystal_type': self.crystal_type,
        'Unitcell_param': self.ucUnitcell,
        'ucBL': self.ucBL,
    }


def load_cif(self, cif_file, noOutput):
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
    path2cif = pathlib.Path(os.path.join(cif_folder, cif_file)).resolve()
    self.cif = ase_io.read(path2cif)
    if not noOutput:
        print("Absolute path to CIF:", path2cif)
    if not path2cif.exists():
        raise FileNotFoundError(f"File {cif_file} not found.")
    if path2cif not in self.loaded_cifs:
        self.loaded_cifs[path2cif] = extract_cif_info(self, path2cif)
    return self.loaded_cifs[path2cif]


#######################################################################
######################################## Folder pathways
def ciflist(dbFolder=data.pyNMBvar.dbFolder):
    """
    Function that prints the CIF files in the dataset.
    Args:
        dbFolder: The database folder name (default is `data.pyNMBvar.dbFolder`).
    """
    path2cif = os.path.join(pyNMB_location(), dbFolder)
    print(os.listdir(path2cif))

def pyNMB_location():
    """
    Returns the root directory of the pyNanoMatBuilder package.
    """
    import pathlib
    import pyNanoMatBuilder
    path = pathlib.Path(pyNanoMatBuilder.__file__)
    return pathlib.Path(*path.parts[0:-2])

#######################################################################
######################################## Coordinates, vectors, etc
def RAB(coord, a, b):
    """
    Calculate the distance between two atoms by indices.
    
    Args:
        coord (array-like): Array of 3D coordinates.
        a (int): Index of the first atom.
        b (int): Index of the second atom.
    
    Returns:
        float: Distance between atoms a and b.
    """
    return np.linalg.norm(np.asarray(coord[a]) - np.asarray(coord[b]))


def Rbetween2Points(p1, p2):
    """
    Calculate the distance between two points.
    
    Args:
        p1 (array-like): 3D coordinates of the first point.
        p2 (array-like): 3D coordinates of the second point.
    
    Returns:
        float: Distance between p1 and p2.
    """
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))


def vector(coord, a, b):
    """
    Compute the vector from point a to point b given a list of coordinates.
    
    Args:
        coord (array-like): Array of 3D coordinates.
        a (int): Index of the starting point.
        b (int): Index of the ending point.
    
    Returns:
        np.ndarray: Vector from a to b.
    """
    return np.asarray(coord[b]) - np.asarray(coord[a])


def vectorBetween2Points(p1, p2):
    """
    Compute the vector between two 3D points.
    
    Args:
        p1 (array-like): 3D coordinates of the first point.
        p2 (array-like): 3D coordinates of the second point.
    
    Returns:
        np.ndarray: Vector from p1 to p2.
    """
    return np.asarray(p2) - np.asarray(p1)


def coord2xyz(coord):
    """
    Extracts x, y, and z from a list of 3D coordinates.

    Args:
        A list or array of 3D coordinates in the format [[x1, y1, z1], [x2, y2, z2], ...].

    Returns:
        Three NumPy arrays containing the x, y, and z coordinates separately.
    """
    x = np.array(coord)[:, 0]
    y = np.array(coord)[:, 1]
    z = np.array(coord)[:, 2]
    return x, y, z


def vertex(x, y, z, scale):
    """Return vertex coordinates fixed to the unit sphere."""
    length = np.sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x, y, z)]


def vertexScaled(x, y, z, scale):
    """Return vertex coordinates multiplied by the scale factor."""
    return [i * scale for i in (x, y, z)]


def RadiusSphereAfterV(V):
    """
    Computes the radius of a sphere given its volume.

    Args:
        V (float): Volume of the sphere in cubic units.

    Returns:
        float: Radius of the sphere.

    Formula: R = (3V / (4π))^(1/3)
    """
    return (3 * V / (4 * np.pi)) ** (1 / 3)

# def centerOfGravity(c: np.ndarray,
#                     select=None):
#     """
#     Computes the center of gravity (geometric center) of a set of points.

#     Args:
#         c (np.ndarray): An array of shape (N, 3) representing N atomic positions (x, y, z).
#         select (np.ndarray, optional): Indices of selected atoms to include in the calculation.
#                                        If None, all atoms are used.
#     Returns:
#         np.ndarray: A 3-element array representing the center of gravity coordinates (x, y, z).

#     Notes:
#     - The center of gravity is computed as the average of the selected atomic positions.
#     """
#     import numpy as np
#     if select is None:
#         select = np.array((range(len(c))))
#     nselect = len(select)
#     xg = 0
#     yg = 0
#     zg = 0
#     for at in select:
#         xg += c[at][0]
#         yg += c[at][1]
#         zg += c[at][2]
#     cog = [xg/nselect, yg/nselect, zg/nselect]
#     return np.array(cog)

def centerOfGravity(c: np.ndarray, select=None):
    """Compute center of gravity of selected atoms."""
    c = np.asarray(c)
    if select is None:
        select = np.arange(len(c))
    return np.mean(c[select], axis=0)


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
    c = np.asarray(c)
    cog = centerOfGravity(c)
    return c - cog


def normOfV(V):
    '''
    Returns the norm of a vector V.
    Args:
        V (np.ndarray): A 3-element array representing a vector [Vx, Vy, Vz].
    Returns:
        float: The norm of the vector.
    '''
    V = np.asarray(V)
    return np.linalg.norm(V)


def normV(V):
    '''
    Computes the normalized unit vector of a vector V.
    Args:
        V (np.ndarray): A 3-element array representing a vector [Vx, Vy, Vz].
    Returns:
        np.ndarray: A 3-element array representing the normalized vector.
        
    '''
    V = np.asarray(V)
    N = normOfV(V)
    if N == 0:
        return np.zeros_like(V)
    return V / N


def centerToVertices(coordVertices: np.ndarray,
                     cog: np.ndarray):
    """
    Computes the vectors and distances between the center of gravity (cog) 
    and each vertex of a polyhedron.
    Args:
        coordVertices (np.ndarray): Array of shape (n_vertices, 3) containing the coordinates of the vertices.
        cog (np.ndarray): A 3-element array representing the center of gravity of the nanoparticle.

    Returns:
        tuple:
            - directions (np.ndarray): Array of shape (n_vertices, 3) containing the vectors from cog to each vertex.
            - distances (np.ndarray): Array of shape (n_vertices,) containing the distances from cog to each vertex.
    """
    coordVertices = np.asarray(coordVertices)
    cog = np.asarray(cog)
    distances = Rbetween2Points(coordVertices, cog)  # Vectorisé pour tous les vertices
    directions = coordVertices - cog

    return directions, distances

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
    """
    Get the moments of inertia along the principal axes.

    Notes:
        Units of the moments of inertia are amu.angstrom**2.
        Periodic boundary conditions are ignored.
    """
    if not noOutput:
        vID.centertxt(
            "Moments of inertia", bgc='#007a7a', size='14', weight='bold'
        )
    model.moi = model.get_moments_of_inertia()  # in amu*angstrom**2
    if not noOutput:
        print(
            f"Moments of inertia = {model.moi[0]:.2f} {model.moi[1]:.2f} "
            f"{model.moi[2]:.2f} amu.Å2"
        )
    model.masses = model.get_masses()
    model.M = model.masses.sum()
    model.moiM = model.moi/(model.M)
    moiM=model.moiM
    if not noOutput:
        print(
            f"Moments of inertia / M = {model.moiM[0]:.2f} {model.moiM[1]:.2f} "
            f"{model.moiM[2]:.2f} amu.Å2"
        )
    return moiM
   


#NEW
def get_moments_of_inertia_for_size(self, vectors=False):  # from ASE with mass modification
    """
    Get the moments of inertia along the principal axes with mass normalization.

    Args:
        vectors (bool, optional): If True, returns both eigenvalues and eigenvectors.
            If False, returns only eigenvalues. Defaults to False.

    Returns:
        np.ndarray: Principal moments of inertia (3 values).
        np.ndarray: Principal axes (3x3 matrix) if vectors is True.

    Notes:
        Periodic boundary conditions are ignored.
        Units of the moments of inertia are angstrom**2.
    """
    com = self.get_center_of_mass()
    positions = self.get_positions()
    #number_atoms=len(positions)
    positions -= com  # translate center of mass to origin
    # masses = self.get_masses() # mass normalization is done by setting all masses to 1 in the inertia tensor calculation

    # Initialize elements of the inertial tensor
    I11 = np.sum(positions[:, 1]**2 + positions[:, 2]**2)
    I22 = np.sum(positions[:, 0]**2 + positions[:, 2]**2)
    I33 = np.sum(positions[:, 0]**2 + positions[:, 1]**2)
    I12 = -np.sum(positions[:, 0] * positions[:, 1])
    I13 = -np.sum(positions[:, 0] * positions[:, 2])
    I23 = -np.sum(positions[:, 1] * positions[:, 2])
    Itensor = np.array([[I11, I12, I13],
                        [I12, I22, I23],
                        [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(Itensor)  # valeurs propes de la matrice
    if vectors:
        return evals, evecs.transpose()
    else:
        return evals


def moi_size(model: Atoms,  # normalized moment of inertia with masses=1
             noOutput: bool=False,
            ):
    """
    Get the moments of inertia along the principal axes with mass normalization.

    Notes:
        Units of the moments of inertia are angstrom**2.
    """

    model.moi_size_all = get_moments_of_inertia_for_size(model)
    positions = model.get_positions()
    number_atoms = len(positions)
    model.moi_size = model.moi_size_all/(number_atoms)
    if not noOutput:
        print(
            f"Moments of inertia with mass=1/M = {model.moi_size[0]:.2f} "
            f"{model.moi_size[1]:.2f} {model.moi_size[2]:.2f} Å2"
        )
    return [model.moi_size[0], model.moi_size[1], model.moi_size[2]]
  


#######################################################################
######################################## Geometry optimization
def full_diagnostics(atoms, verbose=True):
    """
    Run basic diagnostics on an ASE Atoms object for EMT calculations.

    Args:
        atoms: ASE Atoms object to diagnose.
        verbose (bool): If True, prints diagnostic details.

    Returns:
        bool: True if no obvious fatal problems are found, else False.
    """
    pos = atoms.get_positions()
    elems = atoms.get_chemical_symbols()
    print("N atoms:", len(atoms))
    # 1) finiteness check
    if not np.isfinite(pos).all():
        bad = np.where(~np.isfinite(pos).any(axis=1))[0]
        print("ERROR: Non-finite positions at indices:", bad)
        print(pos[bad])
        return False

    # 2) coordinate magnitude
    max_coord = np.abs(pos).max()
    if max_coord > 1e6:
        print("WARNING: coordinates very large (>{:.1e})".format(1e6))
    print("Positions range: min", pos.min(axis=0), " max", pos.max(axis=0))

    # 3) cell / pbc check
    cell = atoms.get_cell()
    print("Cell:", cell)
    if np.isnan(cell).any() or np.isinf(cell).any():
        print("ERROR: cell contains non-finite values")
        return False
    print("PBC flags:", atoms.get_pbc())

    # 4) pairwise distances (fast)
    from scipy.spatial.distance import pdist

    if len(atoms) > 1:
        d = pdist(pos)
        dmin = d.min()
        dmean = d.mean()
        print(f"Pairwise dmin={dmin:.4f} Å, dmean={dmean:.4f} Å")
        if dmin < 0.5:
            print("WARNING: very small interatomic distance (<0.5 Å).")
        if dmin < 1e-6:
            print("ERROR: essentially zero distance between atoms (duplicate).")
            return False
    else:
        print("Only one atom present.")

    # 5) neighbor counts (with a safe cutoff)
    # try:
    #     cutoff = 4.0  # Å, adjust if your element has larger NN
    #     nl = NeighborList([cutoff / 2.0] * len(atoms), self_interaction=False, bothways=True)
    #     nl.update(atoms)
    #     counts = np.array([len(nl.get_neighbors(i)[0]) for i in range(len(atoms))])
    #     print(
    #         "Neighbor counts (cutoff {:.1f} Å): min {}, mean {:.2f}, max {}".format(
    #             cutoff, counts.min(), counts.mean(), counts.max()
    #         )
    #     )
    #     sparsely = np.where(counts <= 1)[0]
    #     if len(sparsely) > 0:
    #         print("Atoms with <=1 neighbor (indices):", sparsely)
    # except Exception as e:
    #     print("Neighborlist construction failed:", e)
    #     return False

    # 6) atomic symbols validity
    valid_symbols = set(['Al', 'Cu', 'Ag', 'Au', 'Ni', 'Pd', 'Pt', 'Fe', 'Co'])
    bad_symbols = [i for i, s in enumerate(elems) if s not in valid_symbols]
    if bad_symbols:
        print("WARNING: unusual element symbols at indices:", bad_symbols, [elems[i] for i in bad_symbols])

    print("Diagnostics OK (no obvious fatal problems found).")
    return True


def optimizeEMT(model: Atoms, saveCoords=True, pathway="./coords/model", fthreshold=0.1):
    """
    Optimize the geometry of an atomic system using EMT and Quasi-Newton.

    Args:
        model (ase.Atoms): Atomic system to optimize.
        saveCoords (bool, optional): If True, saves the optimized coordinates.
        pathway (str, optional): Path to save the trajectory and final structure.
        fthreshold (float, optional): Convergence threshold for forces (in eV/Å).

    Returns:
        ase.Atoms: Optimized atomic model.
    """
    # from varname import nameof, argname
    from ase.io import write
    from ase import Atoms
    from ase.calculators.emt import EMT
    chrono = timer()
    chrono.chrono_start()
    vID.centerTitle("ase EMT calculator & Quasi Newton algorithm for geometry optimization")
    full_diagnostics(model, verbose=True)
    model.set_pbc(False)
    model.center(vacuum=5.0)
    model.calc = EMT()
    model.get_potential_energy()
    from ase.optimize import QuasiNewton

    dyn = QuasiNewton(model, trajectory=pathway + '.opt')
    dyn.run(fmax=fthreshold)
    if saveCoords:
        write(pathway + "_opt.xyz", model)
        print(
            f"{fg.BLUE}Optimization steps saved in {pathway + '_.opt'} "
            f"(binary file){fg.OFF}"
        )
        print(
            f"{fg.RED}Optimized geometry saved in {pathway + '_opt.xyz'}{fg.OFF}"
        )
    chrono.chrono_stop(hdelay=False)
    chrono.chrono_show()
    return model

#######################################################################
######################################## Planes & Directions
# def planeFittingLSF(coords: np.float64,
#                     printErrors: bool=False,
#                     printEq: bool=True):
#     """
#     Fit a plane ux + vy + wz + h = 0 to 3D points using least squares.

#     Args:
#         coords (np.ndarray): Array with shape (N, 3) containing point coordinates.
#         printErrors (bool): If True, prints error/residual details.
#         printEq (bool): If True, prints the fitted plane equation.

#     Returns:
#         np.ndarray: Array [u, v, w, h].
#     """
#     from numpy import linalg as la
#     xs = coords[:, 0]
#     ys = coords[:, 1]
#     zs = coords[:, 2]
#     nat = len(xs)
#     cog = centerOfGravity(coords)
#     # mat = np.zeros((3,3))
#     # for i in range(nat):
#     #     mat[0,0]=mat[0,0]+(xs[i]-cog[0])**2
#     #     mat[1,1]=mat[1,1]+(ys[i]-cog[1])**2
#     #     mat[2,2]=mat[2,2]+(zs[i]-cog[2])**2
#     #     mat[0,1]=mat[0,1]+(xs[i]-cog[0])*(ys[i]-cog[1])
#     #     mat[0,2]=mat[0,2]+(xs[i]-cog[0])*(zs[i]-cog[2])
#     #     mat[1,2]=mat[1,2]+(ys[i]-cog[1])*(zs[i]-cog[2])    
#     # mat[1,0]=mat[0,1]
#     # mat[2,0]=mat[0,2]
#     # mat[2,1]=mat[1,2]

#     coords_centered = coords - cog
#     mat = coords_centered.T @ coords_centered

#     eigenvalues, eigenvectors = la.eig(mat)
#     # the eigenvector associated with the smallest eigenvalue is the vector normal to the plane
#     # print(eigenvalues)
#     # print(eigenvectors)
#     indexMinEigenvalue = np.argmin(eigenvalues)
#     # print(indexMinEigenvalue)
#     # print(la.norm(eigenvectors[:,indexMinEigenvalue]))
#     u, v, w = eigenvectors[:, indexMinEigenvalue]
#     h = -u * cog[0] - v * cog[1] - w * cog[2]
#     if printEq:
#         print(
#             f"bare solution: {u:.5f} x + {v:.5f} y + {w:.5f} z + {h:.5f} = 0"
#         )
#     tmp = coords.copy()
#     ones = np.ones(nat)
#     tmp = np.column_stack((coords, ones))
#     fit = np.array([u, v, w, h])
#     fit = fit.reshape(4, 1)
#     errors = tmp @ fit
#     residual = la.norm(errors)
#     if printErrors:
#         print("errors:")
#         print(errors)
#         print(f"residual: {residual}")
#     return np.array([u, v, w, h]).real

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
    tmp = coords.copy()
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
    """
    Convert a real plane ux + vy + wz + h = 0 to integer hkl form.

    Args:
        plane (np.ndarray): Array [u, v, w, h].
        prthkld (bool): If True, prints the hkl equation.

    Returns:
        np.ndarray: Array [h, k, l, d].
    """
    from fractions import Fraction
    # apply only on non-zero uvw values
    planeEff = []
    acc = 1e-8
    for x in plane[0:3]:  # u,v,w only
        if np.abs(x) >= acc:
            planeEff.append(x)
    planeEff = np.array(planeEff)
    F = np.array(
        [Fraction(x).limit_denominator() for x in np.abs(planeEff)]
    )  # don't change the signe of hkl
    Fmin = np.min(F)
    hkld = plane / Fmin

    if prthkld:
        print(
            f"hkl solution: {hkld[0]:.5f} x + {hkld[1]:.5f} y + "
            f"{hkld[2]:.5f} z + {hkld[3]:.5f} = 0"
        )
        # print("     or")
        # print(f"hkl solution: {-hkld[0]/hkld[2]:.5f} x + {-hkld[1]/hkld[2]:.5f} y + {-hkld[3]/hkld[2]:.5f} = z")
    return hkld

def hklPlaneFitting(coords: np.float64,
                    printEq: bool=True,
                    printErrors: bool=False):
    """
    Context: finding the Miller indices of a plane, if relevant.
    Consists in a least-square fitting of the equation of a plane hx + ky + lz + d = 0
    that passes as close as possible to a set of 3D points.

    Args:
        coords (np.ndarray): array with shape (N,3) that contains the 3 coordinates for each of the N points
        printErrors (bool): if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed

    Returns:
        plane (np.ndarray): [h k l d], where h, k, and l are as close as possible to integers
    """
    plane = planeFittingLSF(coords,printErrors,printEq)
    plane = convertuvwh2hkld(plane, printEq)
    return plane

def shortestPoint2PlaneVectorDistance(plane:np.ndarray,
                                      point:np.ndarray):
    """
    Return the shortest distance, d, from a point X0 to a plane p (projection of X0 on p = P), as well as the PX0 vector.

    Args:
        plane (np.ndarray): [u v w h] definition of the p plane 
        point (np.ndarray): [x0 y0 z0] coordinates of the X0 point or (N, 3) array of points.

    Returns:
        v,d (tuple): the PX0 vector and ||PX0||
    """
    point = np.asarray(point)
    norm_squared = np.sum(plane[0:3]**2)
    if point.ndim == 1:

        t = (plane[3] + np.dot(plane[0:3], point)) / norm_squared
        v = -t * plane[0:3]
        d = np.linalg.norm(v)
    
    else:
        # Multiple points (N, 3)
        t = (plane[3] + point @ plane[0:3]) / norm_squared
        v = -t[:, np.newaxis] * plane[0:3]
        d = np.linalg.norm(v, axis=1)

    return v, d


def Pt2planeSignedDistance(plane,point):
    """
    Return the orthogonal distance of a given point X0 to the plane p in a metric space (projection of X0 on p = P), 
    with the sign determined by whether or not X0 is in the interior of p with respect to the center of gravity [0 0 0].

    Args:
        plane (np.ndarray): Array [u, v, w, h] defining the plane.
        point (np.ndarray): Array [x0, y0, z0] or (N, 3) array of points.

    Returns:
        np.ndarray: Signed distance(s) to the plane.
    """
    plane = np.asarray(plane)
    point = np.asarray(point)
    plane_norm = np.linalg.norm(plane[0:3])

    if point.ndim == 1:
        # Single point: use dot product
        sd = (plane[3] + np.dot(plane[0:3], point)) / plane_norm
    else:
        # Multiple points (N, 3): use matrix multiply
        sd = (plane[3] + point @ plane[0:3]) / plane_norm

    return sd

def planeAtVertices(coordVertices: np.ndarray,
                    cog: np.ndarray):
    """
    Returns the equation of the plane defined by vectors between the center of gravity (cog) and each vertex of a polyhedron
    and that is located at the vertex.

    Args:
        coordVertices (np.ndarray): Coordinates of the vertices (n_vertices, 3).
        cog (np.ndarray): Center of gravity of the NP.

    Returns:
        np.array(plane): the (cog-nvertices)x3 coordinates of the plane 
    """
    planes = []
    for vx in coordVertices:
        vector = vx - cog
        d = -np.dot(vx, vector)
        vector = np.append(vector, d)
        planes.append(vector)

    # TO BE TESTED: the following vectorized version of the above loop, should be faster
    # coordVertices = np.asarray(coordVertices)
    # cog = np.asarray(cog)
    # vectors = coordVertices - cog
    # d = -np.einsum('ij,ij->i', coordVertices, vectors)  # Compute d for each vertex
    # planes = np.column_stack((vectors, d))

    return np.array(planes)

def planeAtPoint(plane: np.ndarray,
                 P0: np.ndarray):
    """
    Recalculate plane d so the plane passes through P0.

    Args:
        plane (np.ndarray): Array [a, b, c, d].
        P0 (np.ndarray): Coordinates [x0, y0, z0].

    Returns:
        np.ndarray: Plane parameters [a, b, c, -(ax0+by0+cz0)].
    """
    d = np.dot(plane[0:3], P0)
    planeAtP = plane.copy()
    planeAtP[3] = -d
    return planeAtP


def normalizePlane(p):
    """Normalize plane parameters [a, b, c, d] by the normal norm."""
    return p / normOfV(p[0:3])

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
    from numpy.linalg import norm
    distance = abs(np.dot(point, plane[0:3]) + plane[3]) / norm(plane[0:3])
    return distance

def AngleBetweenVV(lineDV, planeNV):
    """Return the angle, in degrees, between two vectors."""
    ldv = np.array(lineDV)
    pnv = np.array(planeNV)
    numerator = np.dot(ldv, pnv)
    denominator = normOfV(lineDV) * normOfV(planeNV)
    if denominator == 0:
        alpha = np.NaN
    else:
        alpha = 180 * np.arccos(np.clip(numerator / denominator, -1, 1)) / np.pi
    return alpha

def signedAngleBetweenVV(v1, v2, n):
    """
    Return the signed angle between two vectors in degrees in [0, 360].

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        n (np.ndarray): Normal of the plane formed by the two vectors.

    Returns:
        float: Signed angle in degrees.
    """
    cosTh = np.dot(v1, v2)
    sinTh = np.dot(np.cross(v1, v2), n)
    angle = np.rad2deg(np.arctan2(sinTh, cosTh))
    if angle >= 0:
        return angle
    return 360 + angle

def normal2MillerPlane(Crystal,MillerIndexes,printN=True):
    """
    Return the normal direction to the plane defined by h,k,l Miller indices
    defined as [n1 n2 n3] = (hkl) x G*, where G* is the reciprocal metric tensor (G* = G-1).
    convertuvwh2hkld() function applied here converts real plane indexes to integers.

    Args:
        Crystal: Crystal object with G*.
        MillerIndexes (np.ndarray): Miller indices [h, k, l].
        printN (bool): If True, prints the computed normals.

    Returns:
        np.ndarray: Integer-normalized plane normal.
    """
    normal = MillerIndexes @ Crystal.Gstar
    normal = np.append(normal, 0.0)  # convertuvwh2hkld expects (u v w h)
    normalI = convertuvwh2hkld(normal, False)[0:3]
    if printN:
        print(
            f"Normal to the ({MillerIndexes[0]:2} {MillerIndexes[1]:2} "
            f"{MillerIndexes[2]:2}) user-defined plane > "
            f"[{normal[0]: .3e} {normal[1]: .3e} {normal[2]: .3e}]"
            f" = [{normalI[0]: .2f} {normalI[1]: .2f} {normalI[2]: .2f}]"
        )
    return normalI

def isPlaneParrallel2Line(v1,v2,tol=1e-5):
    """
    Return True if line direction and plane normal are parallel.
    A line direction vector and a plane are parallel if $|angle|$
    between the line and the normal vector of the plane is 90°.

    Args:
        v1 (np.ndarray): Line direction vector. 
        v2 (np.ndarray): Plane normal vector.
        tol (float): Tolerance for angle comparison in degrees.

    Returns:
        bool: True if line and plane are parallel, False otherwise.
    """
    return (
        np.abs(np.abs(AngleBetweenVV(v1, v2)) - 90) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 270) < tol
    )

def isPlaneOrthogonal2Line(v1, v2, tol=1e-5):
    """
    Return True if line direction and plane normal are orthogonal.
    A line direction vector and a plane are orthogonal if $|angle|$ 
    between the line and the normal vector of the plane is 0° or 180°.

    Args:
        v1 (np.ndarray): Line direction vector.
        v2 (np.ndarray): Plane normal vector.
        tol (float): Tolerance for angle comparison in degrees.

    Returns:
        bool: True if line and plane are orthogonal, False otherwise.
    """
    return (
        np.abs(AngleBetweenVV(v1, v2)) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 180) < tol
    )

def areDirectionsOrthogonal(v1, v2, tol=1e-6):
    """
    Return True if directions are orthogonal.
    Lines are orthogonal if the $|angle|$ between their direction vector is 90°

    Args:   
        v1 (np.ndarray): First direction vector.
        v2 (np.ndarray): Second direction vector.
        tol (float): Tolerance for angle comparison in degrees.
    
    Returns:
        bool: True if directions are orthogonal, False otherwise.
    """
    return (
        np.abs(np.abs(AngleBetweenVV(v1, v2)) - 90) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 270) < tol
    )

def areDirectionsParallel(v1, v2, tol=1e-6):
    """
    Return True if directions are parallel.
    Lines are parallel if the $|angle|$ between their direction vector is 0° or 180°.

    Args:
        v1 (np.ndarray): First direction vector.
        v2 (np.ndarray): Second direction vector.
        tol (float): Tolerance for angle comparison in degrees. 
    
    Returns:
        bool: True if directions are parallel, False otherwise.
    """
    return (
        np.abs(AngleBetweenVV(v1, v2)) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 180) < tol
    )

def returnPlaneParallel2Line(V, shift=[1,0,0], debug = False):
    """
    Return plane parameters for a plane parallel to a direction vector.

    Args:
        V (np.ndarray): Direction vector.
        shift (list): Shift vector used to build an arbitrary non-parallel vector.
        debug (bool): If True, prints intermediate values.

    Returns:
        np.ndarray: Plane normal [a, b, c]; d must be found separately.

    Method:
    
        - choose any arbitrary vector not parallel to V[i,j,k] such as V[i+1,j,k]
        - calculate the vector perpendicular to both of these, i.e. the cross product
        - this is the normal to the plane, i.e. you directly obtain the equation of the plane ax+by+cz+d = 0, d being indeterminate
          (to find d, it would be necessary to provide an (x0,y0,z0) point that does not belong to the line, hence d = -ax0-by0-cz0)
        
    """
    arbV = np.array(V.copy())
    arbV = arbV + np.array(shift)
    plane = np.cross(V, arbV)
    if areDirectionsParallel(V, arbV):
        sys.exit(
            f"Error in returnPlaneParallel2Line(): plane {V} is parallel to {arbV}. "
            f"Are you sure of your data?\n(this function wants to return an equation for a "
            f"plane parallel to the direction V = {V}.\n"
            f" Play with the shift variable - current problematic value = {shift})"
        )
    if debug:
        print(areDirectionsParallel(V, arbV), V, arbV, "cross product = ", plane)
    return plane

def planeRotation(Crystal: Atoms,
                  refPlane,
                  rotAxis,
                  nRot=6,
                  debug: bool=False,
                  noOutput: bool=False
                 ):
    """
    Return planes obtained by rotating a reference plane around an axis.

    Args:
        Crystal: Crystal object.
        refPlane: Plane to rotate.
        rotAxis: Rotation axis.
        nRot (int): Rotation count, angle is 360°/nRot.
        debug (bool): If True, prints normalized planes.
        noOutput (bool): If True, suppresses output.

    Returns:
        np.ndarray: Rotated planes in cartesian coordinates.
    """
    pRef = np.array([refPlane])
    aRot = np.array([rotAxis])
    msg = (
        f"Projection of the ({pRef[0][0]: .2f} {pRef[0][1]: .2f} {pRef[0][2]: .2f}) "
        f"reference truncation plane around the [{rotAxis[0]: .2f}  {rotAxis[1]: .2f}  "
        f"{rotAxis[2]: .2f}] axis, after projection in the cartesian coordinate system"
    )
    if not noOutput:
        vID.centertxt(msg, bgc='#cbcbcb', size='12', fgc='b', weight='bold')
    pRefCart = lattice_cart(Crystal, pRef, True, printV=not noOutput)
    rotAxisCart = lattice_cart(Crystal, aRot, True, printV=not noOutput)
    msg = (
        f"{nRot}th order rotation around {rotAxisCart[0][0]: .2f} "
        f"{rotAxisCart[0][1]: .2f} {rotAxisCart[0][2]: .2f}"
        f"of the ({pRefCart[0][0]: .2f} {pRefCart[0][1]: .2f} "
        f"{pRefCart[0][2]: .2f}) truncation plane"
    )
    if not noOutput:
        vID.centertxt(msg, bgc='#cbcbcb', size='12', fgc='b', weight='bold')
    planesCart = []
    for i in range(0, nRot):
        angle = i * 360 / nRot
        # print("rot around z    = ",RotationMol(pRefCart[0],angle,'z'))
        x = rotationMolAroundAxis(pRefCart[0], angle, rotAxisCart[0])
        # print("rot around axis = ",x)
        planesCart.append(x)
    if debug:
        print(np.array(planesCart))
    if not noOutput:
        vID.centertxt(
            f"Just for your knowledge: indexes of the {nRot} normal directions to the "
            f"truncation planes after projection to the {Crystal.cif.cell.get_bravais_lattice()} unitcell",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    planesHCP = lattice_cart(Crystal, np.array(planesCart), False, printV=not noOutput)
    if debug:
        vID.centertxt("Normalized HCP planes", bgc='#cbcbcb', size='12', fgc='b', weight='bold')
        for i, p in enumerate(planesHCP):
            print(i, normV(p))
        print()
        vID.centertxt(
            "Normalized cartesian planes", bgc='#cbcbcb', size='12', fgc='b', weight='bold'
        )
    return np.array(planesCart)

def alignV1WithV2_returnR(v1,v2=np.array([0, 0, 1])):
    """
    Return the rotation matrix that aligns v1 with v2.

    Args:
        v1 (np.ndarray): Source vector.
        v2 (np.ndarray): Target vector.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    from scipy.spatial.transform import Rotation
    v1 = np.reshape(v1, (1, -1))
    v2 = np.reshape(v2, (1, -1))
    rMat = Rotation.align_vectors(v2, v1)
    rMat = rMat[0].as_matrix()
    v1_rot = rMat @ v1[0]
    aligned = np.allclose(v1_rot / np.linalg.norm(v1_rot), v2 / np.linalg.norm(v2))
    if not aligned:
        sys.exit(f"Was unable to align {v1} with {v2}. Check your data")
    return rMat

def rotateMoltoAlignItWithAxis(coords, axis, targetAxis=np.array([0, 0, 1])):
    """
    Return coordinates after rotation aligning axis with targetAxis.

    Args:
        coords (np.ndarray): (n_atoms, 3) array of coordinates.
        axis (np.ndarray): Axis direction [u, v, w].
        targetAxis (np.ndarray): Target axis direction.

    Returns:
        np.ndarray: Rotated coordinates (n_atoms, 3).
    """
    if isinstance(axis, list):
        axis = np.array(axis)
    if isinstance(targetAxis, list):
        targetAxis = np.array(targetAxis)
    rMat = alignV1WithV2_returnR(axis, targetAxis)
    return np.array(rMat @ coords.transpose()).transpose()

def setdAsNegative(planes):
    """
    Flip plane signs so that d is negative.

    Args:
        planes (np.ndarray): Array of planes.

    Returns:
        np.ndarray: Updated planes.
    """
    for i,p in enumerate(planes):
        if p[3] > 0:
            p = -p
            planes[i] = p
    return planes

#######################################################################
######################################## cut above planes
def calculateTruncationPlanesFromVertices(planes, cutFromVertexAt, nAtomsPerEdge, debug=False, noOutput=False, 
                                          trTd=False):
    """
    Calculate truncation planes from vertex planes.

    Args:
        planes (np.ndarray): Array of planes [u, v, w, d].
        cutFromVertexAt (float): Fraction of edge length to cut from each vertex.
        nAtomsPerEdge (int): Number of atoms per edge.
        debug (bool): If True, prints debug information.
        noOutput (bool): If True, suppresses output.

    Returns:
        np.ndarray: Truncation planes.
    """
    n = int(round(1/cutFromVertexAt))
    if not noOutput:
        print(
            f"factor = {cutFromVertexAt:.3f} ▶ {round(nAtomsPerEdge / n)} layer(s) "
            "will be removed, starting from each vertex"
        )

    trPlanes = []
    # for the truncated tetrahedron, the substracted number of layers may be under estimated for big NPs
    # let's introduce a small eps proportionnal to the size of the tetrahedron
    if trTd:
        eps = 0.1 # without it, the truncated tetrahedron is not enough truncated (one layer is missing)
    else:
        eps = 0 # works with the other shapes


    for p in planes:
        pNormalized = normalizePlane(p.copy())
        pNormalized[3] = pNormalized[3] - pNormalized[3] * (cutFromVertexAt + eps)
        trPlanes.append(pNormalized)
    if debug and not noOutput:
        print("normalized original plane = ", normalizePlane(p))
        print("cut plane = ", pNormalized, "... norm = ", normOfV(pNormalized[0:3]))
        print(
            "signed distance between original plane and origin = ",
            Pt2planeSignedDistance(p, [0, 0, 0]),
        )
        print(
            "signed distance between cut plane and origin = ",
            Pt2planeSignedDistance(pNormalized, [0, 0, 0]),
        )
        print(
            "pcut/pRef = ",
            Pt2planeSignedDistance(pNormalized, [0, 0, 0])
            / Pt2planeSignedDistance(p, [0, 0, 0]),
        )
    if not noOutput:
        print(
            "Will remove atoms just above plane "
            f"{pNormalized[0]:.2f} {pNormalized[1]:.2f} {pNormalized[2]:.2f} "
            f"d:{pNormalized[3]:.3f}"
        )



       
    return np.array(trPlanes)



def truncateAboveEachPlane(planes: np.ndarray,
                           coords,
                           debug: bool=False,
                           delAbove: bool=True,
                           noOutput: bool=False,
                           eps: float = 1e-3):
    """
    Return atom indices above (or below) each plane.

    Args:
        planes (np.ndarray): Array with plane definitions [u, v, w, d].
        coords (np.ndarray): (N, 3) array of coordinates.
        debug (bool): If True, prints selected atom indices.
        delAbove (bool): If True, deletes atoms above planes + eps; below otherwise.
        noOutput (bool): If True, suppresses output.
        eps (float): Atom-to-plane signed distance threshold.
    Returns:
        np.ndarray: Unique indices of atoms above/below any input plane.
    """
    coords = np.asarray(coords)
    planes = np.asarray(planes)

    
    delAtoms = []
    
    for p in planes:
        # Vectorized: compute signed distances for all atoms at once
        signedDistances = Pt2planeSignedDistance(p, coords)
        
        if delAbove:
            atomsToDelete = np.where(signedDistances > eps)[0]
        else:
            atomsToDelete = np.where(signedDistances < eps)[0]
        
        delAtoms.extend(atomsToDelete)
        
        if debug and not noOutput:
            for a in atomsToDelete:
                print(f"@{a+1}", end=',')
            print("", end='\n')
    
    delAtoms = np.unique(np.array(delAtoms))
    return delAtoms

def truncateAbovePlanes(planes: np.ndarray,
                        coords: np.ndarray,
                        allP: bool=False,
                        delAbove: bool = True,
                        debug: bool=False,
                        noOutput: bool=False,
                        eps: float=1e-3,
                        depth_max: float=None):
    """
    Return a boolean mask of atoms above/below plane(s).

    Args:
        planes (np.ndarray): Array with plane definitions [u, v, w, d].
        coords (np.ndarray): (N, 3) array of coordinates.
        allP (bool): If True, deleted atoms must lie simultaneously above ALL individual plane.
        delAbove (bool): If True, delete atoms above planes (+eps); below otherwise
        (use with precaution, could return no atoms as a function of their definition)
        debug (bool): If True, prints atoms matching the conditions.
        noOutput (bool): If True, suppresses output.
        eps (float): Atom-to-plane signed distance threshold.
        depth_max (float): Reserved for depth-based filtering.

    Returns:
        np.ndarray: Boolean mask for atoms above/below planes.
    """
    coords = np.asarray(coords)
    planes = np.asarray(planes)
    
    if not noOutput: vID.centertxt(f"Plane truncation (all planes condition: {allP}, delete above planes: {delAbove}, initial number of atoms = {len(coords)})",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    
    if allP:
        delAtoms = np.ones(len(coords), dtype=bool)
    else:
        delAtoms = np.zeros(len(coords), dtype=bool)
    
    for p in planes:
        # Vectorized: compute signed distances for all atoms at once
        signedDistances = Pt2planeSignedDistance(p, coords)
        
        if delAbove and allP:
            delAtomsP = signedDistances > eps
            delAtoms = delAtoms & delAtomsP
        elif delAbove and not allP:
            delAtomsP = signedDistances > eps
            delAtoms = delAtoms | delAtomsP
        elif not delAbove and allP:
            delAtomsP = signedDistances < -eps
            delAtoms = delAtoms & delAtomsP
        elif not delAbove and not allP:
            delAtomsP = signedDistances < -eps
            delAtoms = delAtoms | delAtomsP
        
        nOfDeletedAtomsP = np.count_nonzero(delAtomsP)
        nOfDeletedAtoms = np.count_nonzero(delAtoms)
        
        if debug and not allP:
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtomsP} atoms deleted")
            for i in np.where(delAtomsP)[0]:
                print(f"@{i+1}",end=',')
            print("",end='\n')
        if debug and allP:
            print("allP is True => deletion of all atoms that simultaneously lie above/below each plane")
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtoms} atoms deleted")
            for i in np.where(delAtoms)[0]:
                print(f"@{i+1}",end=',')
            print("",end='\n')
    
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
        planes (np.ndarray): A 2D array with plane equations [a, b, c, d].
        coords (np.ndarray): A 2D array of atom coordinates [x, y, z].
        debug (bool, optional): If True, prints additional debugging information.
        threshold (float, optional): Tolerance for distance to consider a point in plane.
        noOutput (bool, optional): If True, suppresses output messages.

    Returns:
        np.ndarray: A boolean array where True indicates that the atom lies in one of the planes.
    """
    coords = np.asarray(coords)
    planes = np.asarray(planes)
    
    if not noOutput:
        vID.centertxt(
            "Find all points that lie in the given planes",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    AtomsInPlane = np.zeros(len(coords), dtype=bool)
    
    for p in planes:
        # Vectorized: compute signed distances for all atoms at once
        signedDistances = Pt2planeSignedDistance(p, coords)
        atomsInThisPlane = np.abs(signedDistances) < threshold
        AtomsInPlane = AtomsInPlane | atomsInThisPlane
        
        nOfAtomsInPlane = np.count_nonzero(AtomsInPlane)
        if debug:
            print(
                f"- plane",
                [f"{x: .2f}" for x in p],
                f"> {nOfAtomsInPlane} atoms lie in the planes",
            )
            for i in np.where(atomsInThisPlane)[0]:
                print(f"@{i+1}", end=',')
            print("", end='\n')

    if not noOutput:
        print(f"{np.count_nonzero(AtomsInPlane)} atoms lie in the plane(s)")
    return AtomsInPlane

def deleteElementsOfAList(t,
                          list2Delete: bool):
    """
    Return a new list with elements deleted based on a boolean mask.

    Args:
        t: List or array-like.
        list2Delete (bool): Boolean mask, list2Delete[i] = True deletes t[i].

    Returns:
        list: Filtered list.
    """

    if len(t) != len(list2Delete):
        sys.exit(
            "the input list and the array of booleans must have the same dimension. "
            "Check your code"
        )
    if type(t) == list:
        tloc = np.array(t.copy())
    else:
        tloc = t.copy()
    tloc = np.delete(tloc, list2Delete, axis=0)
    return list(tloc)

#######################################################################
######################################## coupling with Jmol & DebyeCalculator
def saveCoords_DrawJmol(asemol, prefix, scriptJ="", boundaries=False, noOutput=True):
    """
    Save coordinates and generate a Jmol visualization.

    Args:
        asemol: ASE Atoms object to visualize.
        prefix (str): Filename prefix for output files.
        scriptJ (str): Additional Jmol script commands.
        boundaries (bool): If True, draws boundaries without facets script.
        noOutput (bool): If True, suppresses command output.
    """
    from pyNanoMatBuilder import data
    path2Jmol = data.pyNMBvar.path2Jmol
    fxyz = "./figs/" + prefix + ".xyz"
    writexyz(fxyz, asemol)
    if not boundaries:
        jmolscript = (
            scriptJ + '; frank off; cpk 0; wireframe 0.05; '
            'script "./figs/script-facettes-345PtLight.spt"; '
            'facettes345ptlight; draw * opaque;'
        )
    else:
        jmolscript = scriptJ + '; frank off; cpk 0; wireframe 0.0; draw * opaque;'
    jmolscript = (
        jmolscript +
        'set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; '
        'set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    )
    jmolcmd = (
        "java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz +
        " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    )
    if not noOutput:
        print(jmolcmd)
    os.system(jmolcmd)


def DrawJmol(mol, prefix, scriptJ=""):
    """
    Generate a Jmol visualization from an existing XYZ file.

    Args:
        mol (str): Molecule filename (without extension).
        prefix (str): Output image filename prefix.
        scriptJ (str): Additional Jmol script commands.
    """
    path2Jmol = '/usr/local/src/jmol-14.32.50'
    fxyz = "./figs/" + mol + ".xyz"
    jmolscript = (
        scriptJ + '; frank off; set specularPower 80; set antialiasdisplay; '
        'set background [xf1f2f3]; set zShade ON;set zShadePower 1; '
        'write image pngt 1024 1024 ./figs/'
    )
    jmolcmd = (
        "java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz +
        " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    )
    if not noOutput:
        print(jmolcmd)
    os.system(jmolcmd)


#######################################################################
######################################## Functions that writes xyz, cif, jmol script files

def write(filename: str, atoms, wa='w', **kwargs):
    """
    Unified write function for pyNanoMatBuilder.

    This function serves as a central hub for exporting data. It handles directory 
    creation automatically and routes the data to the appropriate writer based 
    on the file extension.

    Args:
        filename (str): Path to the output file (e.g., 'coords/np.xyz').
        atoms (ase.Atoms or str): The atomic structure to save, or a string 
            containing script content for .script/.spt files.
        wa (str, optional): Write mode. 'w' for overwrite (default) or 'a' 
            for append (useful for multi-frame trajectories).
        **kwargs: Additional arguments passed to the underlying ASE write 
            function (e.g., 'format').

    Note:
        - Automatically creates parent directories if they do not exist.
        - Uses internal 'writexyz' for .xyz files to preserve custom headers.
        - Uses ASE for crystallography formats (.cif, .res, .pdb, etc.).
        - Handles raw text writing for Jmol scripts (.script, .spt).
    """
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    extension = file_path.suffix.lower()

    if extension == ".xyz":
        # Call your custom function for the specific pyNMB header
        writexyz(filename, atoms, wa=wa)
    
    elif extension in [".script", ".spt"]:
        # Handle Jmol scripts (which are strings, not Atoms objects)
        with open(file_path, wa) as f:
            f.write(atoms)
            
    else:
        # For everything else (.cif, .pdb, etc.), use the power of ASE
        # Translate 'wa' for ASE
        # If wa is 'a', then is_append is True.
        is_append = (wa == 'a')
        ase_io.write(filename, atoms, append=is_append, **kwargs)
        
def writexyz(filename: str,
             atoms: Atoms,
             wa: str='w'):
    """
    Simple xyz writing, with atomic symbols/x/y/z and no other information.
    Automatically creates the parent directories if they do not exist.
    Args:
        filename (str): Output filename.
        atoms (Atoms): ASE Atoms object.
        wa (str): Write mode ('w' or 'a').
    """
    from collections import Counter
    element_array = atoms.get_chemical_symbols()
    # extract composition in dict form - optimized with Counter
    composition = dict(Counter(element_array))

    coord = atoms.get_positions()
    natoms = len(element_array)
    line2write = '%d \n' % natoms
    line2write += '%s\n' % str(composition)
    for i in range(natoms):
        line2write += (
            '%s' % str(element_array[i]) +
            '\t %.8f' % float(coord[i, 0]) +
            '\t %.8f' % float(coord[i, 1]) +
            '\t %.8f' % float(coord[i, 2]) + '\n'
        )
        
    # Ensure the directory exists
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, wa) as file:
        file.write(line2write)

def create_data_csv(path_of_files, path_of_csvfiles, noOutput):
    """
    Extract dictionaries from XYZ/CSV files and create new CSV files.

    Args:
        path_of_files: Path of the directory containing the files.
        path_of_csvfiles: Path of the directory where the CSV file will be created.
        noOutput: If True, suppresses print statements.

    Returns:
        None
    """
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
                    lignes = file.readlines()
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
    """
    Reduce crystal facets based on convex hull and coplanarit of facets.

    Args:
        Crystal (Atoms): The crystal object containing the planes for the facet reduction.
        feasible_point (np.ndarray): A feasible point for half-space intersection. Default is [0, 0, 0].
        tolAngle (float): Tolerance angle to define coplanarity. Default is 2.0.
        noOutput (bool): If True, suppresses output to the console. Default is False.
        
    Returns:
        tuple: The vertices and reduced faces.

    Note:
        Previous hull.simplices must have been saved as Crystal.trPlanes.
    """
    from scipy.spatial import HalfspaceIntersection
    from scipy.spatial import ConvexHull
    import networkx as nx
    import scipy as sp

    cog = Crystal.cog
    feasible_point = cog
    hs = HalfspaceIntersection(Crystal.trPlanes, feasible_point)
    vertices = hs.intersections
    hull = ConvexHull(vertices)

    faces = hull.simplices
    neighbours = hull.neighbors
    if not noOutput:
        vID.centertxt("Boundaries figure", bgc='#007a7a', size='14', weight='bold')
        vID.centertxt(
            "Half space intersection of the planes followed by a convex Hull analyzis",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
        print("Found:")
        print(f"  - {len(hull.vertices)} convex Hull vertices")
        print(f"  - {len(hull.simplices)} convex Hull simplices before reduction")

    def sortVCW(V, C):
        """
        Sort vertices of a planar polygon clockwise.

        Args:
            V (list): List of vertex indices.
            C (np.ndarray): Array of vertex coordinates.
        
        Returns:
            list: Sorted vertex indices in clockwise order.
        """
        coords = []
        for v in V : coords.append(C[v])
        cog = np.mean(coords, axis=0)
        radialV = coords-cog
        angle = []
        V = list(V)
        normal = planeFittingLSF(np.array(coords), False, False)
        for i in range(len(radialV)):
            angle.append(signedAngleBetweenVV(radialV[0], radialV[i], normal[0:3]))
        ind = np.argsort(angle)
        Vs = np.array(list(V))
        return Vs[ind]
    

    def isCoplanar(p1, p2, tolAngle=tolAngle):
        """Check if two planes p1 and p2 are coplanar."""
        angle = AngleBetweenVV(p1[0:3], p2[0:3])
        return abs(angle) < tolAngle or abs(angle - 180) <= tolAngle

    def reduceFaces(F, coordsVertices):
        """Reduce the number of faces by merging coplanar ones."""
        flatten = lambda l: [item for sublist in l for item in sublist]

        # create a graph in which nodes represent triangles
        # nodes are connected if the corresponding triangles are adjacent and coplanar
        G = nx.Graph()
        G.add_nodes_from(range(len(F)))
        pList = []
        for i, f in enumerate(F):
            planeDef = []
            for v in f:
                planeDef.append(coordsVertices[v])
            planeDef = np.array(planeDef)
            pList.append(planeFittingLSF(planeDef, printErrors=False, printEq=False))

        for i, p1 in enumerate(pList):
            for n in neighbours[i]:
                p2 = pList[n]
                if isCoplanar(p1, p2):
                    G.add_edge(i, n)
        components = list(nx.connected_components(G))
        simplified = [
            set(flatten(F[index] for index in component)) for component in components
        ]

        return simplified

    new_faces = reduceFaces(faces, vertices)
    new_facesS = []
    for i, nf in enumerate(new_faces):
        new_facesS.append(sortVCW(nf, vertices).tolist())
    if not noOutput:
        print(f"  - {len(new_faces)} facets after reduction")
        print("New trPlanes saved in self.trPlanes")
    trPlanes = []
    for i, f in enumerate(new_faces):
        planeDef = []
        for v in f:
            planeDef.append(vertices[v])
        planeDef = np.array(planeDef)
        trPlanes.append(planeFittingLSF(planeDef, printErrors=False, printEq=False))
    Crystal.trPlanes = setdAsNegative(np.array(trPlanes))
    return vertices, new_facesS

def defCrystalShapeForJMol(Crystal: Atoms,
                           noOutput: bool=True,
                          ):
    """
    Generate a Jmol command to visualize the crystal shape based on the facets of the crystal.

    Args:
        Crystal (Atoms): The crystal structure object containing the facets and planes.
        noOutput (bool): If True, suppresses the output of the command.

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
        if not noOutput:
            vID.centertxt(
                "generating the jmol command line to view the crystal shape",
                bgc='#cbcbcb',
                size='12',
                fgc='b',
                weight='bold',
            )
        cmd = ""
        for i, nf in enumerate(redFacets):
            cmd += "draw facet" + str(i) + " polygon "
            cmd += '['
            for at in nf:
                cmd += f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
            cmd += "]; "
        cmd += "color $facet* translucent 70 [x828282]"
        cmde = ""
        index = 0
        for nf in redFacets:
            nfcycle = np.append(nf, nf[0])
            for i, at in enumerate(nfcycle[:-1]):
                cmde += "draw line" + str(index) + " ["
                cmde += f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
                cmde += (
                    f"{{{vertices[nfcycle[i+1]][0]:.4f},"
                    f"{vertices[nfcycle[i+1]][1]:.4f},"
                    f"{vertices[nfcycle[i+1]][2]:.4f}}},"
                )
                cmde += "] width 0.2; "
                index += 1
        cmde += "color $line* [xd6d6d6]; "
        cmd = cmde + cmd
    else:  # sphere, ellipsoid
        cmd = ""
    if not noOutput:
        print("Jmol command: ", cmd)
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
    nn, CN = kDTreeCN(Crystal, Rmax, noOutput=noOutput)
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
        colorsFull = [
            (255, 0, 0), (255, 255, 153), (255, 255, 0), (255, 204, 0),
            (102, 255, 255), (51, 204, 255), (102, 153, 255), (249, 128, 130),
            (153, 255, 204), (0, 204, 153), (0, 134, 101), (0, 102, 102),
            (51, 51, 255), (102, 51, 0), (0, 51, 102), (77, 77, 77),
            (0, 0, 0)
        ]
        colorsFull = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in colorsFull]
        path, file = os.path.split(save2)
        prefix = file.split(".")
        fileColors = "./" + path + "/" + prefix[0] + "colors.png"
        fileColorsFull = "./" + path + "/" + "CN_color_palette.png"
        colorNamesFull = np.array(range(0, CNMax + 1))
        print("Full palette:")
        plotPalette(colorsFull, colorNamesFull, savePngAs=fileColorsFull)
        print(f"Palette specific to {prefix[0]}:")
        colors = []
        for c in uniqueCN:
            colors.append(colorsFull[c])
        plotPalette(colors, uniqueCN, savePngAs=fileColors)

        # Generate Jmol command for CN visualization
        print(f"{hl.BOLD}Jmol command:{hl.OFF}")
        command = f"{{*}}.valence = load('{file}'); "
        colorScheme = ""
        for c in colorsFull:
            colorScheme = colorScheme + rgb2hex(c) + " "
        command = command + f"color atoms property valence 'colorCN' RANGE 0 {CNMax} ;"
        command = (
            command +
            "label %2.0[valence]; color label yellow ; font label 24 ; set labeloffset 7 0;"
        )
        print(f"color 'colorCN = {colorScheme}';")
        print(command)

def plotPalette(Pcolors, namePC, angle=0, savePngAs=None):
    """
    Plot a 1D palette of colors with names.

    Args:
        Pcolors: 1D list with hex colors.
        namePC: Label for each color.
        angle: Rotation angle of the text.
        savePngAs: Also saves the palette in a png file (default: None).
    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.palplot(sns.color_palette(Pcolors))
    ax = plt.gca()

    for i, name in enumerate(namePC):
        ax.set_xticks(np.arange(len(namePC)))
        ax.tick_params(length=0)
        ax.set_xticklabels(namePC, weight='bold', size=10, rotation=angle)
    if savePngAs is not None:
        plt.tight_layout()
        plt.savefig(savePngAs, dpi=600, transparent=True)
    plt.show()
    return

def rgb2hex(c, frac=True):
    """
    Convert an RGB color to its hexadecimal representation.

    It has an optional frac argument to handle the case where the RGB values
    are provided as fractions (ranging from 0 to 1) or as integers
    (ranging from 0 to 255).

    Args:
        c: RGB color tuple.
        frac (bool): If True, RGB values are fractions (0-1).

    Returns:
        str: Hexadecimal representation.
    """
    if frac:
        r = int(round(c[0] * 255))
        g = int(round(c[1] * 255))
        b = int(round(c[2] * 255))
    else:
        r = c[0]
        g = c[1]
        b = c[2]
    return f"[x{r:02X}{g:02X}{b:02X}]"
#######################################################################
######################################## coordination numbers
def calculateCN(coords,Rmax):
    '''
    Calculate the coordination number (CN) for each atom.

    The coordination number is determined by counting neighbors within 
    a spherical cutoff distance defined by Rmax.

    Args:
        coords (numpy.ndarray): An (N, 3) array containing the Cartesian 
            coordinates for each of the N points.
        r_max (float): The threshold distance (cutoff) used to define 
            a coordination bond.

    Returns:
        numpy.ndarray: An array of length N containing the calculated 
            coordination number for each atom.
    '''
    # CN = np.zeros(len(coords))
    # for i,ci in enumerate(coords):
    #     for j in range(0,i):
    #         Rij = Rbetween2Points(ci,coords[j])
    #         if Rij <= Rmax:
    #             CN[i]+=1
    #             CN[j]+=1
    
    # NEW
    from scipy.spatial.distance import squareform, pdist
    coords = np.asarray(coords)
    # Compute all pairwise distances at once
    dist_matrix = squareform(pdist(coords))
    # Count neighbors within Rmax (excluding self with > 0)
    CN = np.sum((dist_matrix > 0) & (dist_matrix <= Rmax), axis=1)

    return CN

def delAtomsWithCN(coords: np.ndarray,
                   Rmax: np.float64,
                   targetCN: int=12):
    """
    Return the coordination number of each atom.

    CN is calculated after threshold Rmax.

    Args:
        coords: numpy array with shape (N,3) containing 3 coordinates for each of N points.
        Rmax: threshold to calculate CN.

    Returns:
        np.ndarray: Array containing CN for each atom.
    """
    '''
    identifies atoms that have a coordination number (CN) == targetCN and returns them in an array
    - input:
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - CN: array of integers with the coordination number of each atom
        - targetCN (default=12)
    returns an array that contains the indexes of atoms with CN == targetCN
    '''
    CN = calculateCN(coords,Rmax)
    # tabDelAtoms = []
    # for i,cn in enumerate(CN):
    #     if cn == targetCN: tabDelAtoms.append(i)
    # tabDelAtoms = np.array(tabDelAtoms)
    
    tabDelAtoms = np.where(CN == targetCN)[0]
    return tabDelAtoms

def findNeighbours(coords,Rmax):
    """
    Find all atoms j within cutoff distance Rmax from atom i.

    For all atoms i, returns the list of all atoms j within an arbitrarily
    determined cutoff distance Rmax from atom i.

    Args:
        coords: numpy array with the N-atoms cartesian coordinates.
        Rmax: cutoff distance.

    Returns:
        list: List of lists (len(list[i]) = number of nearest neighbours of atom i).
    """
    # vID.centertxt(f"Building a table of nearest neighbours",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    # chrono = timer(); chrono.chrono_start()
    # nAtoms = len(coords)
    # nn = [ [] for _ in range(nAtoms)]
    # for i in range(nAtoms):
    #     for j in range(i):
    #         if RAB(coords,i,j) < Rmax:
    #             nn[i].append(j)
    #             nn[j].append(i)
    
    # chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    # return nn

    from scipy.spatial.distance import squareform, pdist
    vID.centertxt(
        "Building a table of nearest neighbours",
        bgc='#cbcbcb',
        size='12',
        fgc='b',
        weight='bold',
    )
    chrono = timer()
    chrono.chrono_start()
    coords = np.asarray(coords)
    # Compute all pairwise distances at once
    dist_matrix = squareform(pdist(coords))
    # Find neighbors for each atom (excluding self)
    nAtoms = len(coords)
    nn = []
    for i in range(nAtoms):
        neighbors = np.where((dist_matrix[i] > 0) & (dist_matrix[i] < Rmax))[0]
        nn.append(neighbors)
    chrono.chrono_stop(hdelay=False)
    chrono.chrono_show()
    return nn 


def printNeighbours(nn):
    """
    Print the list of nearest neighbours of each atom.

    Args:
        nn: Nearest neighbours given as a list of lists.
    """
    for i, nni in enumerate(nn):
        print(f"Atom {i:6} has {len(nni):2} NN: {nni}")

def kDTreeCN(crystal: Atoms,
             Rmax: float=2.9,
             returnD: bool=False,
             noOutput: bool=False
            ):
    """
    Return the nearest neighbour (nn) table and number of NN per atom.

    Args:
        crystal (Atoms): Crystal structure object.
        Rmax (float): The NN threshold.
        returnD (bool): If True, distances between NN are returned as well.
        noOutput (bool): If True, suppresses output messages.

    Returns:
        tuple: (nn, CN) or (nn, CN, dNN) if returnD is True.
    """
    from sklearn.neighbors import KDTree
    if noOutput:
        vID.centertxt(
            "Building a table of nearest neighbours",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    if noOutput:
        chrono = timer()
        chrono.chrono_start()
    coords = crystal.get_positions()
    tree = KDTree(coords)
    nn = []
    CN = []
    dNN = []
    for i, c in enumerate(coords):
        if returnD:
            l, d = tree.query_radius([c], r=3.0, return_distance=returnD)
            l = list(l[0])
            d = list(d[0])
        else:
            l = list(tree.query_radius([c], r=3.0, return_distance=returnD)[0])
        if returnD:
            dNN.append(d)
        ipos = l.index(i)
        l.remove(i)
        if returnD:
            del(d[ipos])
        nn.append(l)
        CN.append(len(l))
    if noOutput:
        chrono.chrono_stop(hdelay=False)
        chrono.chrono_show()
    if returnD:
        return nn, CN, dNN
    else:
        return nn, CN

#######################################################################
######################################## symmetry
# def reflection(plane, points, dontDoItForAtomsThatLieInTheReflectionPlane=True):
#     """
#     Apply a mirror-image symmetry operation w.r.t. a plane of symmetry.

#     Args:
#         plane: [u,v,w,d] parameters that define a plane.
#         points: (N, 3) array of points.
#         dontDoItForAtomsThatLieInTheReflectionPlane: Self-explanatory.

#     Returns:
#         np.ndarray: (N, 3) array of mirror-image points.
#     """
#     pr = []
#     eps = 1.e-4
#     for p in points:
#         vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane, p)
#         if (
#             (dontDoItForAtomsThatLieInTheReflectionPlane and dp2plane >= eps) or
#             not dontDoItForAtomsThatLieInTheReflectionPlane
#         ):
#             ptmp = p + 2 * vp2plane
#             pr.append(ptmp)
#     return np.array(pr)

def reflection(plane,points,doItForAtomsThatLieInTheReflectionPlane=False):
    '''
    Apply a mirror-image symmetry operation to an array of points.

    Calculates the reflection of each point across a symmetry plane defined by
    the general equation $ax + by + cz + d = 0$.

    Args:
        points (numpy.ndarray): An (N, 3) array of Cartesian coordinates 
            representing the points to be reflected.
        plane (list or numpy.ndarray): The four parameters $[a, b, c, d]$ 
            that define the reflection plane equation.
        include_plane_atoms (bool, optional): If True, points located exactly 
            on the reflection plane are processed. If False, they are 
            skipped. Defaults to True.

    Returns:
        numpy.ndarray: An (N, 3) array containing the coordinates of the 
        reflected mirror-image points.
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

def reflection_tetra(plane,points):
    """
    Simplified reflection function for the helix of tetrahedrons.

    Args:
        plane: [u,v,w,d] parameters that define a plane.
        points: (N, 3) array of points.

    Returns:
        np.ndarray: (N, 3) array of reflected points.
    """
    # pr = []
    # for p in points:
    #     vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane,p)
        
    #     ptmp = p+2*vp2plane
    #     pr.append(ptmp)
    # return np.array(pr)
    points = np.asarray(points)
    vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane, points)
    return points + 2 * vp2plane


#######################################################################
######################################## rotation
def Rx(a):
    """Return the R/x rotation matrix."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ]
    )
  
def Ry(a):
    """Return the R/y rotation matrix."""
    return np.array(
        [
            [np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0, np.cos(a)]
        ]
    )
  
def Rz(a):
    """Return the R/z rotation matrix."""
    return np.array(
        [
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ]
    )

def EulerRotationMatrix(gamma, beta, alpha, order="zyx"):
    """
    Return a 3x3 Euler rotation matrix.

    Args:
        gamma: Rot/x (°).
        beta: Rot/y (°).
        alpha: Rot/z (°).
        order: If (order="zyx"): returns Rz(alpha) * Ry(beta) * Rx(gamma).

    Returns:
        np.ndarray: A 3x3 Euler matrix.
    """
    R = 1.
    gammarad = gamma * np.pi / 180
    betarad = beta * np.pi / 180
    alpharad = alpha * np.pi / 180
    for i in range(3):
        if order[i] == "x":
            R = R * Rx(gammarad)
        if order[i] == "y":
            R = R * Ry(betarad)
        if order[i] == "z":
            R = R * Rz(alpharad)
    return R

def RotationMol(coords, angle, axis="z"):
    """
    Perform a rotation of the molecule's coordinates around a specified axis.

    Args:
        coords (numpy.ndarray): Coordinates of the molecule as a matrix (n x 3).
        angle (float): The angle of rotation in degrees.
        axis (str): The axis around which to perform the rotation ('x', 'y', or 'z').

    Returns:
        numpy.ndarray: R[0] The coordinates of the molecule after rotation, as a matrix (n x 3).
    """
    import math as m
    angler = angle * m.pi / 180
    if axis == 'x':
        R = np.array(Rx(angler) @ coords.transpose())
    elif axis == 'y':
        R = np.array(Ry(angler) @ coords.transpose())
    elif axis == 'z':
        R = np.array(Rz(angler) @ coords.transpose())

    return R


def EulerRotationMol(coords, gamma, beta, alpha, order="zyx"):
    """
    Perform an Euler rotation of the molecule's coordinates.

    Args:
        coords (numpy.ndarray): Coordinates of the molecule as a matrix (n x 3).
        gamma (float): Angle gamma in degrees.
        beta (float):  Angle beta in degrees.
        alpha (float): Angle alpha in degrees.
        order (str): The order of the Euler rotations (default is "zyx").

    Returns:
        numpy.ndarray: The coordinates of the molecule after the Euler rotation, as a matrix (n x 3).
    """
    return np.array(
        EulerRotationMatrix(gamma, beta, alpha, order) @ coords.transpose()
    ).transpose()

def RotationMatrixFromAxisAngle(u, angle):
    """
    Generates a 3x3 rotation matrix from a unit vector representing the axis of rotation and a rotation angle.
    Args:
        u (numpy.ndarray): A unit vector representing the axis of rotation (3 elements).
        angle (float): The angle of rotation in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    a = angle * np.pi / 180
    ux = u[0]
    uy = u[1]
    uz = u[2]
    return np.array(
        [
            [
                np.cos(a) + ux**2 * (1 - np.cos(a)),
                ux * uy * (1 - np.cos(a)) - uz * np.sin(a),
                ux * uz * (1 - np.cos(a)) + uy * np.sin(a)
            ],
            [
                uy * ux * (1 - np.cos(a)) + uz * np.sin(a),
                np.cos(a) + uy**2 * (1 - np.cos(a)),
                uy * uz * (1 - np.cos(a)) - ux * np.sin(a)
            ],
            [
                uz * ux * (1 - np.cos(a)) - uy * np.sin(a),
                uz * uy * (1 - np.cos(a)) + ux * np.sin(a),
                np.cos(a) + uz**2 * (1 - np.cos(a))
            ]
        ]
    )

def rotationMolAroundAxis(coords, angle, axis):
    """
    Return coordinates after rotation by a given angle around an [u,v,w] axis.

    Args:
        coords: natoms x 3 numpy array.
        angle: Angle of rotation.
        axis: Directions given under the form [u,v,w].

    Returns:
        numpy.ndarray: Rotated coordinates.
    """
    normalizedAxis = normV(axis)
    return np.array(
        RotationMatrixFromAxisAngle(normalizedAxis, angle) @ coords.transpose()
    ).transpose()

def rotation_around_axis_through_point(coords, angle_deg, axis, center):
    """
    Rotate coords (n,3) around a given axis (3,) passing through a point `center` (3,)
    by a given angle (in degrees).
    """
    # Mise en radians
    angle_rad = math.radians(angle_deg)
    
    # Normalise l'axe
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    # Translation pour mettre le point 'center' à l'origine
    shifted = coords - center

    # Matrice de rotation (formule de Rodrigues)
    ux, uy, uz = axis
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta),
         ux * uy * (1 - cos_theta) - uz * sin_theta,
         ux * uz * (1 - cos_theta) + uy * sin_theta],
        
        [uy * ux * (1 - cos_theta) + uz * sin_theta,
         cos_theta + uy**2 * (1 - cos_theta),
         uy * uz * (1 - cos_theta) - ux * sin_theta],
        
        [uz * ux * (1 - cos_theta) - uy * sin_theta,
         uz * uy * (1 - cos_theta) + ux * sin_theta,
         cos_theta + uz**2 * (1 - cos_theta)]
    ])

    # Appliquer la rotation
    rotated_shifted = shifted @ R.T

    # Replacer autour de 'center'
    rotated = rotated_shifted + center

    return rotated


#######################################################################
######################################## magic numbers
def magicNumbers(cluster, i):
    """
    Calculate magic numbers for nanocluster stable structures.

    Certain collections of atoms are more preferred due to energy minimization
    and exhibiting stable structures and providing unique properties to the
    materials. These collections of atoms providing stable structures to the
    materials are called MAGIC NUMBERS.

    This function uses specific formulas based on the type of nanocluster
    (e.g., 'regfccOh', 'regIco', 'fccCube', etc.) to calculate a magic number
    based on the index i.

    Args:
        cluster (str): The type of nanocluster for which the magic number is
            calculated. It can be one of the following values:
            'regfccOh', 'regIco', 'regfccTd', 'regDD', 'fccCube',
            'bccCube', 'fccCubo', 'fccTrOh', 'fccTrCube', 'bccrDD',
            'fccdrDD', 'pbpy'.
        i (int): The index of the nanocluster size. It must be a positive
            integer greater than zero.

    Returns:
        float: The calculated magic number for the specified nanocluster type
            and index i.
    """
    match cluster:
        case 'regfccOh':
            mn = np.round((2 / 3) * i**3 + 2 * i**2 + (7 / 3) * i + 1)
            return mn
        case 'regIco':
            mn = (10 * i**3 + 11 * i) // 3 + 5 * i**2 + 1
            return mn
        case 'regfccTd':
            mn = np.round(i**3 / 6 + i**2 + 11 * i / 6 + 1)
            return mn
        case 'regDD':
            mn = 10 * i**3 + 15 * i**2 + 7 * i + 1
            return mn
        case 'fccCube':
            mn = 4 * i**3 + 6 * i ** 2 + 3 * i + 1
            return mn
        case 'bccCube':
            mn = 2 * i**3 + 3 * i ** 2 + 3 * i + 1
            return mn
        case 'fccCubo':
            mn = np.round((10 * i**3 + 11 * i) / 3 + 5 * i**2 + 1)
            return mn
        case 'fccTrOh':
            mn = np.round(16 * i**3 + 15 * i**2 + 6 * i + 1)
            return mn
        case 'fccTrCube':
            mn = np.round(4 * i**3 + 6 * i**2 + 3 * i - 7)
            return mn
        case 'bccrDD':
            mn = 4 * i**3 + 6 * i**2 + 4 * i + 1
            return mn
        case 'fccdrDD':
            mn = 8 * i**3 + 6 * i**2 + 2 * i + 3
            return mn
        case 'pbpy':
            mn = 5 * i**3 / 6 + 5 * i**2 / 2 + 8 * i / 3 + 1
            return mn
        case _:
            sys.exit(f"The {cluster} nanocluster is unknown")

#######################################################################
######################################## Bravais
def interPlanarSpacing(plane: np.ndarray,
                       unitcell: np.ndarray,
                       CrystalSystem: str='CUB'):
    """
    Calculate the interplanar spacing.

    Args:
        plane: numpy array containing the [h k l d] parameters of the plane
            of equation hx + ky + lz + d = 0.
        unitcell: numpy array with [a b c alpha beta gamma].
        CrystalSystem: Name of the crystal system, string among:
            ['CUB', 'HEX', 'TRH', 'TET', 'ORC', 'MCL', 'TRI'] = cubic,
            hexagonal, trigonal-rhombohedral, tetragonal, orthorombic,
            monoclinic, tricilinic.

    Returns:
        float: The interplanar spacing.
    """
    h = plane[0]
    k = plane[1]
    l = plane[2]
    a = unitcell[0]
    match CrystalSystem.upper():
        case 'CUB':
            d2 = a**2 / (h**2 + k**2 + l**2)
        case 'HEX':
            c = unitcell[2]
            d2inv = (4 / 3) * (h**2 + k**2 + h * k) / a**2 + l**2 / c**2
            d2 = 1 / d2inv
        case 'TRH':
            alpha = (np.pi / 180) * unitcell[3]
            d2inv = (
                (h**2 + k**2 + l**2) * np.sin(alpha)**2 +
                2 * (h * k + k * l + h * l) * (np.cos(alpha)**2 - np.cos(alpha))
            ) / (a**2 * (1 - 3 * np.cos(alpha)**2 + 2 * np.cos(alpha)**3))
            d2 = 1 / d2inv
        case 'TET':
            c = unitcell[2]
            d2inv = (h**2 + k**2) / a**2 + l**2 / c**2
            d2 = 1 / d2inv
        case 'ORC':
            b = unitcell[1]
            c = unitcell[2]
            d2inv = h**2 / a**2 + k**2 / b**2 + l**2 / c**2
            d2 = 1 / d2inv
        case 'MCL':
            b = unitcell[1]
            c = unitcell[2]
            beta = (np.pi / 180) * unitcell[4]
            d2inv = (
                (h / a)**2 + (k * np.sin(beta) / b)**2 + (l / c)**2 -
                2 * h * l * np.cos(beta) / (a * c)
            ) / np.sin(beta)**2
            d2 = 1 / d2inv
        case 'TRI':
            b = unitcell[1]
            c = unitcell[2]
            alpha = (np.pi / 180) * unitcell[3]
            beta = (np.pi / 180) * unitcell[4]
            gamma = (np.pi / 180) * unitcell[5]
            V = (a * b * c) * np.sqrt(
                1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
                2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
            astar = b * c * np.sin(alpha) / V
            bstar = a * c * np.sin(beta) / V
            cstar = a * b * np.sin(gamma) / V
            cosalphastar = (
                (np.cos(gamma) * np.cos(beta) - np.cos(alpha)) /
                (np.sin(gamma) * np.sin(beta))
            )
            cosbetastar = (
                (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) /
                (np.sin(alpha) * np.sin(gamma))
            )
            cosgammastar = (
                (np.cos(beta) * np.cos(alpha) - np.cos(gamma)) /
                (np.sin(beta) * np.sin(alpha))
            )
            d2inv = (
                (h * astar)**2 + (k * bstar)**2 + (l * cstar)**2 +
                2 * k * l * bstar * cstar * cosalphastar +
                2 * l * h * cstar * astar * cosbetastar +
                2 * h * k * astar * bstar * cosgammastar
            )
            d2 = 1 / d2inv
        case _:
            sys.exit(
                f"{CrystalSystem} crystal system is unknown. Check your data.\n"
                "Or do not try to calculate interplanar distances on this "
                "system with interPlanarSpacing()"
            )
    d = np.sqrt(d2)
    return d

def lattice_cart(Crystal, vectors, Bravais2cart=True, printV=False):
    """
    Project vectors between Bravais basis and cartesian coordinate system.

    Args:
        Crystal: Crystal object.
        vectors: Vectors to project from the Bravais basis to the cartesian
            coordinate system (if Bravais2cart is True) or to project from
            the cartesian coordinate system to the Bravais basis
            (if Bravais2cart is False).
        Bravais2cart (bool): If True, project from Bravais to cartesian.
        printV (bool): If True, prints the resulting vectors.

    Returns:
        np.ndarray: Array of projected vectors.
    """
    import numpy as np
    unitcell = Crystal.ucUnitcell
    Vuc = Crystal.ucV
    if Bravais2cart:
        Vproj = (vectors @ Vuc)
        B = 'B'
        E = 'C'
    else:
        VucInv = np.linalg.inv(Vuc)
        Vproj = (vectors @ VucInv)
        B = 'C'
        E = 'B'
    if printV:
        print("Change of basis")
        for i, V in enumerate(vectors):
            Bstr = f"{V[0]: .2f} {V[1]: .2f} {V[2]: .2f}"
            Vp = Vproj[i]
            Estr = f"{Vp[0]: .2f} {Vp[1]: .2f} {Vp[2]: .2f}"
            print(f"({Bstr}){B} > ({Estr}){E}")
    return Vproj 

def G(Crystal):
    """
    Compute the metric tensor (G) of a crystal's unit cell.

    The metric tensor is calculated based on the unit cell parameters:
    the lengths of the unit cell vectors (a, b, c) and the angles
    (alpha, beta, gamma) between them.

    Reference:
        https://fr.wikibooks.org/wiki/Cristallographie_g%C3%A9om%C3%A9trique/
        Outils_math%C3%A9matiques_pour_l%27%C3%A9tude_du_r%C3%A9seau_cristallin

    Args:
        Crystal: Crystal object.

    Returns:
        numpy.ndarray: The 3x3 metric tensor (G) of the unit cell.
    """
    a = Crystal.ucUnitcell[0]
    b = Crystal.ucUnitcell[1]
    c = Crystal.ucUnitcell[2]
    alpha = Crystal.ucUnitcell[3] * np.pi / 180.
    beta = Crystal.ucUnitcell[4] * np.pi / 180.
    gamma = Crystal.ucUnitcell[5] * np.pi / 180.
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    GG = np.array(
        [
            [a**2, a * b * cg, a * c * cb],
            [a * b * cg, b**2, b * c * ca],
            [a * c * cb, b * c * ca, c**2]
        ]
    )
    return GG

def Gstar(Crystal):
    """
    Compute the inverse of the metric tensor (G*) for a crystal's unit cell.

    Args:
        Crystal: Crystal object.

    Returns:
        numpy.ndarray: The 3x3 inverse metric tensor (G*) of the unit cell.
    """
    Gmat = G(Crystal)
    return linalg.inv(Gmat)

#######################################################################
######################################## Misc for plots
def imageNameWithPathway(imgName):
    """
    Construct the full file path for an image.

    Constructs the full file path for an image by joining the base directory
    with the image name.

    Args:
        imgName (str): The name of the image file.

    Returns:
        str: The full file path to the image file.
    """
    path2image = os.path.join(pyNMB_location(), 'figs')
    imgNameWithPathway = os.path.join(path2image, imgName)
    return imgNameWithPathway


def plotImageInPropFunction(imageFile):
    """
    Plot an image using matplotlib with no axes and a specified size.

    Args:
        imageFile: The path to the image file to be displayed.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image = mpimg.imread(imageFile)
    plt.figure(figsize=(2, 10))
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()

#######################################################################
######################################## Core/surface identification / Convex Hull analysis
def coreSurface(Crystal: Atoms,
                threshold,
                noOutput=False,
               ):
    """
    Identify the core and surface atoms of a crystal using Convex Hull analysis.

    Args:
        Crystal (Atoms): Crystal structure object.
        threshold (float): The threshold used to identify surface atoms.
        noOutput (bool): If True, suppresses output during the analysis.

    Returns:
        tuple: A tuple containing:
            - list: [Hull vertices, Hull simplices, Hull neighbors, Hull equations]
            - surfaceAtoms (numpy.ndarray): The atomic positions of atoms on the surface.
    """
    from ase.visualize import view
    from scipy.spatial import ConvexHull
    if not noOutput:
        vID.centertxt("Core/Surface analyzis", bgc='#007a7a', size='14', weight='bold')
    if not noOutput:
        chrono = timer()
        chrono.chrono_start()
    coords = Crystal.NP.get_positions()
    if not noOutput:
        vID.centertxt(
            "Convex Hull analyzis",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    hull = ConvexHull(coords)
    if not noOutput:
        print("Found:")
        print(f"  - {len(hull.vertices)} vertices")
        print(f"  - {len(hull.simplices)} simplices")
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
    surfaceAtoms = returnPointsThatLieInPlanes(Crystal.trPlanes, coords, noOutput=noOutput,threshold=threshold)
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return [hull.vertices, hull.simplices, hull.neighbors, hull.equations],surfaceAtoms


#######################################################################
######################################## basic rdf profile
def rdf(NP: Atoms,
        dr: float=0.05,
        sigma: float=2,
        ncores: int=1,
        noOutput: bool=True
       ):
    """
    rdf - g(r) - calculator for non-PBC systems.
    
    Args:
        NP (Atoms): ase Atoms object
        dr (float): determines the spacing between successive radii over which g(r) is computed. Default: 0.05
        sigma (float): standard deviation for Gaussian kernel. Default: 2
        ncores (int): number of jobs to schedule for parallel processing (only used by query_ball_point() of scipy.spatial.KDTree). Default: 1
        noOutput (bool): do not print anything. Default: True

    Returns:
        - r and g(r)

    Wanna know more? Read https://doi.org/10.1021/acs.chemrev.1c00237
    """
    from ase.atoms import Atoms
    from ase.visualize import view
    from scipy.spatial import KDTree
    from scipy.spatial import distance
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    if not noOutput:
        vID.centertxt(
            "Basic RDF profile calculation",
            bgc='#007a7a',
            size='14',
            weight='bold',
        )
    com = NP.get_center_of_mass()
    coords = NP.get_positions()
    if not noOutput:
        chrono = timer()
        chrono.chrono_start()
    tree = KDTree(coords)
    dist = distance.cdist(coords, [com])
    rMax = np.max(dist)
    dMax = 1.05 * 2 * rMax
    radii = np.arange(dr, dMax, dr)
    if not noOutput:
        print(f"dMax = {dMax:.2f} (number of points = {len(radii)})")
    g_r = np.zeros(len(radii))
    dist = distance.pdist(coords)
    for ir, r in enumerate(radii):
        for i, c in enumerate(coords):
            neighbours = (
                tree.query_ball_point(c, r, return_length=True, workers=ncores) -
                tree.query_ball_point(c, r - dr, return_length=True, workers=ncores)
            )
            g_r[ir] += neighbours
    g_r = gaussian_filter1d(g_r, sigma=sigma, mode='nearest')
    g_r = np.divide(g_r, radii)

    # TO BE TESTED: should be faster, see if the results are the same as the previous code
    # g_r = np.zeros(len(radii))
    # for ir, r in enumerate(radii):
    #     neighbors_r = tree.query_ball_tree(tree, r)
    #     neighbors_r_dr = tree.query_ball_tree(tree, max(r-dr, 0))
        
    #     for i in range(len(coords)):
    #         # Nombre de voisins dans la coquille [r-dr, r]
    #         shell_count = len(neighbors_r[i]) - len(neighbors_r_dr[i])
    #         g_r[ir] += shell_count

    peaks, _ = find_peaks(g_r)
    if not noOutput: print(f"First peak found at: {radii[peaks[0]]:.2f} Å. g(r) = {g_r[peaks[0]]:.2f}")
    g_r = g_r/g_r[peaks[0]]
    radii = radii/radii[peaks[0]]
    if not noOutput: print("(Intensity and position of the returned RDF profile normalized w.r.t. this first peak)")
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return radii, g_r, len(radii)

#######################################################################
######################################## simple file management utilities


def createDir(path2, forceDel=False):
    """
    Creates a directory at the specified path.

    If the directory already exists, it will either be left unchanged or
    deleted and recreated based on the 'forceDel' argument.

    Args:
        path2 (str): The path where the directory should be created.
        forceDel (bool, optional): If set to True, will delete the existing
            directory and recreate it. Default is False.

    Returns:
        None
    """
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


# def isnt_inside_cylinder(position, radius, radius_squared, half_height):
#     """
#     Checks whether a given position is outside a cylinder.

#     The cylinder is defined by a radius and half height and aligned along the
#     z-axis. If the position lies outside the circular base or beyond the half
#     height of the cylinder, the function returns True. Otherwise, it returns
#     False.

#     Args:
#         position (tuple or list): The (x, y, z) coordinates of the point to
#             check.
#         radius (float): The radius of the cylinder's base.
#         radius_squared (float): The square of the radius, for optimization
#             purposes.
#         half_height (float): Half the height of the cylinder (from the center
#             along the z-axis).

#     Returns:
#         bool: Returns True if the position is outside the cylinder, False
#             otherwise.
#     """
#     # coord défini dans writexyz
#     if (
#         abs(position[0]) > radius
#         or abs(position[1]) > radius
#         or abs(position[2]) > half_height
#     ):
#         return True
#     if position[0] ** 2 + position[1] ** 2 > radius_squared:
#         return True
#     return False


def Inscribed_circumscribed_spheres(self, noOutput):
    """
    Calculates the diameters of the inscribed and circumscribed spheres for the nanoparticle (NP) based on 
    its positions and the plane equations.

    The circumscribed sphere is the smallest sphere that completely encloses the NP, while the inscribed sphere 
    is the largest sphere that fits within the NP.

    Args:
        noOutput (bool, optional): If set to True, suppresses output during the
            analysis. Default is False.

    Returns:
        None: The function updates the object's attributes with the calculated
            diameters of the spheres.

    Notes:
        The circumscribed sphere radius is calculated as the maximum distance from the center of gravity 
        (COG) to the NP positions, and the inscribed sphere radius is calculated as the minimum distance 
        from the NP positions to the planes (based on Hull equations)
    """
    if self.shape == 'ellipsoid':
        self.radiusInscribedSphere = min(self.sasview_dims)
        self.radiusCircumscribedSphere = max(self.sasview_dims)
    elif self.shape == 'sphere':
        self.radiusInscribedSphere = self.radius
        self.radiusCircumscribedSphere = self.radius
    else:
        distances = np.linalg.norm(self.NP.positions - self.cog, axis=1)
        self.radiusCircumscribedSphere = np.max(distances)
        distances = [
            abs(d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
            for a, b, c, d in self.equations
        ]
        self.radiusInscribedSphere = np.min(distances)

    if not noOutput:
        vID.centertxt(
            "Diameters of the inscribed and circumscribed sphere using the "
            "Hull equations",
            bgc='#007a7a',
            size='14',
            weight='bold'
        )
    if not noOutput:
        print(
            f"diameters of the circumscribed sphere: "
            f"{self.radiusCircumscribedSphere * 2 * 0.1:.2f} nm"
        )
        print(
            f"diameters of the inscribed sphere: "
            f"{self.radiusInscribedSphere * 2 * 0.1:.2f} nm"
        )

    return self.radiusInscribedSphere, self.radiusCircumscribedSphere

#######################################################################
def remove_duplicate_atoms(coordinates, reference_coords, tolerance):
    """
    Robustly removes duplicate atoms from coordinates using adaptive tolerance.
    
    Identifies atoms in 'coordinates' that are within 'tolerance' of any atom
    in 'reference_coords' and removes them. This is ideal for removing atoms
    added by reflection that already exist on shared faces.
    
    Args:
        coordinates (np.ndarray): (N, 3) array of new atoms to filter
        reference_coords (np.ndarray): (M, 3) array of existing atoms (reference)
        tolerance (float): Distance threshold for duplicate detection.
                          Recommended: 0.1 * Rnn for molecular structures.
    
    Returns:
        tuple: (unique_coords, n_removed)
            - unique_coords: (K, 3) filtered coordinates
            - n_removed: (int) number of duplicates removed
    
    Notes:
        - Atoms from 'coordinates' that are far from all atoms in 'reference_coords'
          are considered unique and kept.
        - Efficiency: O(N*M) where N=len(coordinates), M=len(reference_coords)
        - Use this when adding incrementally to existing structure (helix generation)
    
    Example:
        >>> new_atoms = np.array([[0.5, 0, 0], [5, 0, 0]])
        >>> existing = np.array([[0, 0, 0], [1, 0, 0]])
        >>> tol = 0.1
        >>> unique, n_dup = remove_duplicate_atoms(new_atoms, existing, tol)
        >>> print(n_dup)  # 1 (first atom is duplicate)
    """
    coordinates = np.asarray(coordinates, dtype=float)
    reference_coords = np.asarray(reference_coords, dtype=float)
    tolerance = float(tolerance)
    
    if len(coordinates) == 0:
        return coordinates, 0
    
    if len(reference_coords) == 0:
        return coordinates, 0
    
    # Compute distances from each new atom to closest reference atom
    # Using broadcasting: (N, 3) - (1, 3) → (N, 3), then norm → (N,)
    distances_matrix = np.linalg.norm(
        coordinates[:, np.newaxis, :] - reference_coords[np.newaxis, :, :],
        axis=2
    )  # Shape: (N, M)
    
    min_distances = np.min(distances_matrix, axis=1)  # Shape: (N,)
    
    # Keep atoms that are far enough from all reference atoms
    mask_unique = min_distances > tolerance
    unique_coords = coordinates[mask_unique]
    n_removed = np.sum(~mask_unique)
    
    return unique_coords, n_removed
