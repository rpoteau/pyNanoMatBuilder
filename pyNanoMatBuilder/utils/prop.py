import time, datetime
import importlib
import os
import pathlib
from pathlib import Path
import re

import numpy as np
from scipy import linalg
import math
import sys

from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase import io as ase_io
from ase.spacegroup import get_spacegroup
from ase.visualize import view

from importlib import resources

from pyNanoMatBuilder import data
from .core import (pyNMB_location, get_resource_path, timer, RAB, Rbetween2Points,
                   vector, vectorBetween2Points, coord2xyz, vertex, vertexScaled, RadiusSphereAfterV,
                   centerOfGravity, center2cog, normOfV, normV, centerToVertices, Rx, Ry, Rz,
                   EulerRotationMatrix, plotPalette, rgb2hex, clone, deleteElementsOfAList,
                   planeFittingLSF, AngleBetweenVV, signedAngleBetweenVV
                   )
from .core import centertxt, centerTitle, fg, bg, hl, color
from .parallel import njit, prange
from .geometry import coreSurface, setdAsNegative
from .symmetry import MolSym
from .external_pgm import defCrystalShapeForJMol

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
    # centertxt(f"Building a table of nearest neighbours",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
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
    centertxt(
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
        centertxt(
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


######################################## Momenta of inertia
def compute_moi(model: Atoms,
        noOutput: bool=False,
       ):
    """
    Get the moments of inertia along the principal axes.

    Notes:
        Units of the moments of inertia are amu.angstrom**2.
        Periodic boundary conditions are ignored.
    """
    if not noOutput:
        centertxt(
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

def calculate_rg(model: Atoms,
                 mass_weighted=True,
                 noOutput: bool=False,
                ):
    """
    Calculates the radius of gyration for an ASE Atoms object.
    
    Args:
        atoms (ase.Atoms): The system to analyze.
        mass_weighted (bool): If True, weights by atomic mass. 
                              If False, calculates geometric Rg.
        noOutput (bool): If False, prints a formatted summary.
    Returns:
        float: The radius of gyration in nm.
    """
    
    calc_type = "Mass-weighted" if mass_weighted else "Geometric"
    if not noOutput:
        centertxt(
            f"Radius of Gyration ({calc_type})", bgc='#007a7a', size='14', weight='bold'
        )
    
    positions = model.get_positions()

    if mass_weighted:
        masses = model.get_masses()
        com = model.get_center_of_mass()
        sq_dist = np.sum((positions - com)**2, axis=1)
        rg = np.sqrt(np.sum(masses * sq_dist) / np.sum(masses))
    else:
        # Geometric center
        center = np.mean(positions, axis=0)
        sq_dist = np.sum((positions - center)**2, axis=1)
        rg = np.sqrt(np.mean(sq_dist))
        
    rg /= 10 # angstrom to nm

    if not noOutput:
        print(f" Rg = {rg:.1f} nm")

    return rg

def calculate_npr(moi,
                  noOutput: bool=False,
                 ):
    """
    Calculates Normalized Principal Moments of Inertia (NPR).
    
    Returns:
        list: [npr1, npr2] where npr1 = I1/I3 and npr2 = I2/I3.
               Returns (1.0, 1.0) for single atoms to avoid division by zero.
    """
    I1, I2, I3 = np.sort(moi)
    if not noOutput:
        centertxt(
            "Normalized Ratios of Principal Moments of Inertia (NPR)", bgc='#007a7a', size='14', weight='bold'
        )
        
    # 2. Prevent division by zero for single atoms or points
    if I3 > 1e-9:
        npr1 = I1 / I3
        npr2 = I2 / I3
    else:
        # Handle the case of a single atom where all MOIs are 0
        npr1, npr2 = 1.0, 1.0

    if not noOutput:
        # Determination of the dominant shape for the printout
        shape_desc = "Spherical/Symmetric"
        if npr1 < 0.2 and npr2 > 0.8:
            shape_desc = "Linear/Rod-like"
        elif 0.35 < npr1 < 0.65 and 0.35 < npr2 < 0.65:
            shape_desc = "Planar/Disk-like"

        print(f" Principal Moments : I1={I1:.2f}, I2={I2:.2f}, I3={I3:.2f}")
        print(f" NPR1 (I1/I3)      : {npr1:.4f}")
        print(f" NPR2 (I2/I3)      : {npr2:.4f}")
        print(f" Predicted Shape   : {shape_desc}")
    
    return [npr1, npr2]
        
#------------------------------------------------------------------------------------------------------------------------
    
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
            radius of the spheres, in Angströms.

    Notes:
        The circumscribed sphere radius is calculated as the maximum distance from the center of gravity 
        (COG) to the NP positions, and the inscribed sphere radius is calculated as the minimum distance 
        from the NP positions to the planes (based on Hull equations)
    """
    # --- Target Selection ---
    # If the system is optimized, we use NP_opt, otherwise we use the builder's NP
    if self.is_optimized and self.NP_opt is not None:
        target_atoms = self.NP_opt
        target_cog   = getattr(self, 'cog_opt', None)
        target_eq    = getattr(self, 'equations_opt', None)
        status_text = "optimized structure"
    else:
        target_atoms = self.NP
        target_cog   = getattr(self, 'cog', None)
        target_eq    = getattr(self, 'equations', None)
        status_text = "initial structure"

    if target_atoms is None or len(target_atoms) == 0:
        raise ValueError(f"CRITICAL: No atoms found in {status_text} structure. "
                         "Build the particle before calling sphere analysis.")

    if target_cog is None or len(target_cog) != 3:
        raise ValueError(f"CRITICAL: Center of Mass (COG) is missing for {status_text} structure. "
                         "Ensure propPostMake() or set_cog() has been executed.")

    if target_eq is None or len(target_eq) == 0:
        raise ValueError(f"CRITICAL: Hull equations are missing for {status_text} structure. "
                         "Inscribed sphere calculation requires surface plane equations.")
        

    # Inscribed and circumscribed sphere
    distances = np.linalg.norm(target_atoms.positions - target_cog, axis=1)
    self.radiusCircumscribedSphere = np.max(distances)
    distances = [
        abs(d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        for a, b, c, d in target_eq
    ]
    self.radiusInscribedSphere = np.min(distances)

    if not noOutput:
        centertxt(
            f"Diameters of the inscribed and circumscribed sphere using the "
            f"Hull equations ({status_text})",
            bgc='#007a7a',
            size='14',
            weight='bold'
        )
    if not noOutput:
        print(
            f"Diameter of the circumscribed sphere: "
            f"{self.radiusCircumscribedSphere * 2:.2f} Å"
        )
        print(
            f"Diameter of the inscribed sphere: "
            f"{self.radiusInscribedSphere * 2:.2f} Å"
        )


    return self.radiusInscribedSphere, self.radiusCircumscribedSphere


def _update_sasview_dims_from_spheres(self, noOutput):
    """
    Update sasview_dims based on inscribed/circumscribed sphere radii.

    IMPORTANT:
    - Initial structure: sasview_dims set in make*() methods
    - After optimization/peeling: recalculated here using real sphere radii
    
    Only recalculates sasview_dims if the structure was optimized or peeled,
    ensuring that the dimensions reflect the actual geometry of the NP after
    modifications. If the shape attribute is missing, it skips the update 
    and issues a warning.
    The supported shapes for sasview_dims updates are "sphere", "cylinder"/"wire",
    and "regfccOh".
    """

    if not (self.is_optimized or self.is_peeled):
        return  # keep initial dims
    
    if not hasattr(self, 'radiusCircumscribedSphere') or self.radiusCircumscribedSphere is None:
        if not noOutput:
            print("⚠ radiusCircumscribedSphere not calculated, skipping sasview_dims update")
        return
    
    if not hasattr(self, 'shape') or self.shape is None:
        if not noOutput:
            print("⚠ shape attribute missing, skipping sasview_dims update")
        return

    
    # Else: recalculate sasview_dims based on the shape and the sphere radii
    try:
        if self.shape == "sphere":
            self.sasview_dims = [self.radiusCircumscribedSphere]
            self.volume = (4/3) * math.pi * (self.radiusCircumscribedSphere)**3
            if not noOutput :
                print(f"New sasview dimensions of the sphere based on the inscribed/circumscribed sphere diameters:"
                      f"diameter = {self.sasview_dims[0]*2:.2f} Å, "
                      f"volume = {self.volume :.2f} Å³")
       
        elif self.shape in ("cylinder", "wire"):
            self.radius = self.radiusInscribedSphere
            self.length = 2 * np.sqrt(self.radiusCircumscribedSphere**2 - self.radius**2)
            self.volume = math.pi * self.radius**2 * self.length
            self.sasview_dims = [self.radius, self.length]
            if not noOutput :
                print(f"New  sasview dimensions of the cylinder/wire based on the inscribed/circumscribed sphere diameters:"
                      f" diameter = {self.radius*2:.2f} Å,"
                      f" length = {self.length:.2f} Å,"
                      f" volume = {self.volume :.2f} Å³.")
    
        elif self.shape == "regfccOh":
            self.demi_axis = self.radiusCircumscribedSphere
            self.truncature = 1
            self.sasview_dims = [self.demi_axis, self.truncature]
            self.volume = (4/3) * self.demi_axis**3
            if not noOutput :
                print(f"New sasview dimensions of the regfccOh based on the inscribed/circumscribed sphere diameters:"
                      f": demi_axis = {self.demi_axis:.2f} Å, "
                      f"truncature = {self.truncature}, volume = {self.volume :.2f} Å³.")
    
    except AttributeError as e:
        if not noOutput:
            print(f"⚠ Error updating sasview_dims: {e}")

# def get_ellipsoid_analysis(self, noOutput=False):
#         """
#         Perform a Principal Component Analysis (PCA) on the outer envelope to 
#         calculate the best-fitting circumscribed ellipsoid.

#         This method identifies the principal axes of the nanoparticle's surface 
#         by analyzing the covariance matrix of the Convex Hull vertices. The 
#         resulting ellipsoid is scaled such that its major semi-axis matches 
#         the maximum radial distance found in the structure, ensuring a perfect 
#         fit for circumscribed diameter measurements (e.g., 1.63 nm for a 
#         3-shell icosahedron).

#         The analysis automatically selects between 'initial structure' and 
#         'optimized structure' based on the current state of the object.

#         Args:
#             noOutput (bool): If True, suppresses printed summaries and Jmol 
#                 command generation. Defaults to False.

#         Returns:
#             dict: A dictionary containing the following physical properties:
#                 - "status": String indicating which envelope was analyzed.
#                 - "D1", "D2", "D3": Major, intermediate, and minor diameters (Å).
#                 - "volume": Volume of the ellipsoid (Å³).
#                 - "surface": Approximate surface area (Å²) using Knud Thomsen's formula.
#                 - "asphericity": Ratio of D1/D3 (1.0 for a perfect sphere).

#         Raises:
#             ValueError: If fewer than 4 Hull vertices are found, indicating 
#                 that coreSurface() has not been run or the NP is invalid.

#         Notes:
#             - Scaling Logic: The semi-axes are derived from the square root of 
#               the eigenvalues of the covariance matrix. The scale factor is 
#               defined as: scale = max_radius / sqrt(max_eigenvalue).
#             - Surface Area: Calculated using an approximation with p=1.6075, 
#               limiting the maximum relative error to 1.061%.
#             - Visualization: Generates a Jmol-ready command using the 'AXES' 
#               and 'CENTER' keywords for precise orientation.
#         """
#         import numpy as np
#         if not hasattr(self, 'ellipsoid'):
#             self.ellipsoid = {}
            
#         # 1. Select the correct envelope data
#         if self.is_optimized and hasattr(self, 'vertices_opt'):
#             target_atoms = self.NP_opt
#             hull_indices = self.vertices_opt
#             status = "optimized envelope"
#             key = "optimized structure"
#         else:
#             target_atoms = self.NP
#             hull_indices = self.vertices
#             status = "initial envelope"
#             key = "initial structure"
    
#         hull_coords = target_atoms.get_positions()[hull_indices]

#         if hull_coords is None or len(hull_coords) < 4:
#             raise ValueError(f"Insufficient Hull vertices found for {status} ({len(hull_coords)} is < 4). "
#                          "Please run coreSurface() before this analysis.")

#         # 2. PCA on surface atoms
#         if self.is_optimized:
#             surface_mask = getattr(self, 'surfaceAtoms_opt', None)
#         else:
#             surface_mask = getattr(self, 'surfaceAtoms', None)

#         if surface_mask is not None and np.count_nonzero(surface_mask) >= 4:
#             pts = target_atoms.get_positions()[surface_mask]
#         else:
#             pts = np.asarray(hull_coords)

#         # Force center to origin — NP is already centered in pyNMB
#         center = np.zeros(3)
#         pos_c = pts - center

#         S = (pos_c.T @ pos_c) / len(pts)
#         evals, evecs = np.linalg.eigh(S)
#         idx = np.argsort(evals)[::-1]
#         evals = evals[idx]
#         evecs = evecs[:, idx]

#         # 3. Scale — use hull vertices for true extent
#         hull_c = np.asarray(hull_coords) - center
#         max_dist = np.max(np.linalg.norm(hull_c, axis=1))
#         scale_factor = max_dist / np.sqrt(evals[0])
#         # Eigenvalue scaling (correct ratios)
#         a_ev, b_ev, c_ev = scale_factor * np.sqrt(evals)
#         # Independent projection on hull vertices (correct for flat shapes)
#         a_pr = np.max(np.abs(hull_c @ evecs[:, 0]))
#         b_pr = np.max(np.abs(hull_c @ evecs[:, 1]))
#         c_pr = np.max(np.abs(hull_c @ evecs[:, 2]))
#         # Take the maximum of both
#         a = max(a_ev, a_pr)
#         b = max(b_ev, b_pr)
#         c = max(c_ev, c_pr)
        
#         # Safety: if a, b, c are too different from max_dist
#         # (PCA axes don't align with vertices), fall back to max_dist
#         tol = 0.05  # 5% tolerance
#         if abs(a - max_dist) / max_dist > tol or \
#            abs(b - max_dist) / max_dist > tol or \
#            abs(c - max_dist) / max_dist > tol:
#             # Use all atoms projected onto eigenvectors
#             all_pos = target_atoms.get_positions() - center
#             a = np.max(np.abs(all_pos @ evecs[:, 0]))
#             b = np.max(np.abs(all_pos @ evecs[:, 1]))
#             c = np.max(np.abs(all_pos @ evecs[:, 2]))
    
#         # 4. Physical Properties
#         # Volume: (4/3) * pi * a * b * c
#         volume = (4/3) * np.pi * a * b * c
        
#         # Surface Area (Knud Thomsen's formula - approximation error < 1.06%)
#         p = 1.6075
#         surface_area = 4 * np.pi * (
#             ((a*b)**p + (a*c)**p + (b*c)**p) / 3
#         )**(1/p)

#         # 5. Results Dictionary
#         D1, D2, D3 = 2*a, 2*b, 2*c

#         self.ellipsoid[key] = {
#             "status": status,
#             "D1": D1, # Major (A) -> Should match 2 * max_dist
#             "D2": D2, # Intermediate (A)
#             "D3": D3, # Minor (A)
#             "volume": volume,
#             "surface": surface_area,
#             "asphericity": D1 / D3 if c > 0 else 1.0
#         }


#         if not noOutput:
#             results = self.ellipsoid[key]
#             centertxt(f"Hull Ellipsoid Analysis ({status})", 
#                         bgc='#007a7a',
#                         size='14',
#                         weight='bold')
#             print(f"  - Dimensions (Å): {results['D1']:.2f} x {results['D2']:.2f} x {results['D3']:.2f}")
#             print(f"  - Volume: {results['volume']/1000:.2f} nm³")
#             print(f"  - Surface: {results['surface']/100:.2f} nm²")
#             print(f"  - Asphericity: {results['asphericity']:.2f}")
#             print(f"  - Max Radius found: {max_dist/10:.3f} nm")
#             # --- Jmol Command Generation ---
#             # Semi-axes vectors for Jmol
#             v1, v2, v3 = evecs[:,0]*a, evecs[:,1]*b, evecs[:,2]*c
            
#             # THE VALIDATED JMOL COMMAND
#             key_cmd = key.replace(" ", "_")
#             jmol_cmd = (f"ellipsoid ID {key_cmd}_el AXES "
#                         f"{{{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}}} "
#                         f"{{{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}}} "
#                         f"{{{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}}}; "
#                         f"ellipsoid ID {key_cmd}_el CENTER {{{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}}}; "
#                         f"color ${key_cmd}_el [x919191] translucent 0.3;")
            
#             print("\n  [Jmol Command to visualize the ellipsoid]:")
#             print(f"  {jmol_cmd}")

### K E E P
# def get_ellipsoid_analysis(self, noOutput=False, mode='inscribed'):
#         """
#         Perform a Principal Component Analysis (PCA) on the outer envelope to 
#         calculate the best-fitting circumscribed or inscribed ellipsoid.
#         This method identifies the principal axes of the nanoparticle's surface 
#         by analyzing the covariance matrix of the surface atoms or convex Hull vertices. The semi-axes
#         are scaled by projecting the target atoms onto each principal axis.
#         Works correctly for convex shapes (spheres, ellipsoids), symmetric
#         shapes (icosahedra), and non-convex shapes (peeled structures).
#         The analysis automatically selects between 'initial structure' and 
#         'optimized structure' based on the current state of the object.
#         Args:
#             noOutput (bool): If True, suppresses printed summaries and Jmol 
#                 command generation. Defaults to False.
#             mode (str): 'inscribed' (default) — PCA on surface atoms, ellipsoid
#                 fits inside the structure. 'circumscribed' — projection on hull
#                 vertices, ellipsoid contains all atoms. 
#         Returns:
#             dict: A dictionary containing the following physical properties:
#                 - "status": String indicating which geometry was analyzed (optimized or initial).
#                 - "mode": String indicating which ellipsoid has been calculated.
#                 - "D1", "D2", "D3": Major, intermediate, and minor diameters (Å).
#                 - "volume": Volume of the ellipsoid (Å³).
#                 - "surface": Approximate surface area (Å²) using Knud Thomsen's formula.
#                 - "asphericity": Ratio of D1/D3 (1.0 for a perfect sphere).
#         Raises:
#             ValueError: If fewer than 4 surface atoms are found.
#         Notes:
#             - PCA is performed on surface atoms (uniform coverage).
#             - Scaling is done by projecting surface atoms onto each PCA axis.
#             - Surface Area: Knud Thomsen approximation, error < 1.06%.
#             - Visualization: Jmol-ready command using AXES and CENTER keywords.
#         """
#         import numpy as np
#         if not hasattr(self, 'ellipsoid'):
#             self.ellipsoid = {}
            
#         # 1. Select the correct structure
#         if self.is_optimized and hasattr(self, 'vertices_opt'):
#             target_atoms = self.NP_opt
#             hull_indices = self.vertices_opt
#             status = "optimized envelope"
#             key = "optimized structure"
#         else:
#             target_atoms = self.NP
#             hull_indices = self.vertices
#             status = "initial envelope"
#             key = "initial structure"

#         hull_coords = target_atoms.get_positions()[hull_indices]
#         # 2. Select surface atoms
#         if self.is_optimized:
#             surface_mask = getattr(self, 'surfaceAtoms_opt', None)
#         else:
#             surface_mask = getattr(self, 'surfaceAtoms', None)

#         # Check that coreSurface() has been run
#         if surface_mask is None or np.count_nonzero(surface_mask) < 4:
#             raise ValueError(f"No surface atoms found for {status}. "
#                              "Please run coreSurface() before this analysis.")

#         if surface_mask is not None and np.count_nonzero(surface_mask) >= 4:
#             pts = target_atoms.get_positions()[surface_mask]
#         else:
#             pts = np.asarray(hull_coords)

#         # Force center to origin — NP is already centered in pyNMB
#         center = np.zeros(3)
#         pos_c = pts - center

#         S = (pos_c.T @ pos_c) / len(pts)
#         evals, evecs = np.linalg.eigh(S)
#         idx = np.argsort(evals)[::-1]
#         evals = evals[idx]
#         evecs = evecs[:, idx]

#         # 3. Scale — use hull vertices for true extent
#         hull_c = np.asarray(hull_coords) - center
#         max_dist = np.max(np.linalg.norm(hull_c, axis=1))
#         scale_factor = max_dist / np.sqrt(evals[0])
#         # Eigenvalue scaling (correct ratios)
#         a_ev, b_ev, c_ev = scale_factor * np.sqrt(evals)
#         # Independent projection on hull vertices (correct for flat shapes)
#         a_pr = np.max(np.abs(hull_c @ evecs[:, 0]))
#         b_pr = np.max(np.abs(hull_c @ evecs[:, 1]))
#         c_pr = np.max(np.abs(hull_c @ evecs[:, 2]))
#         # Take the maximum of both
#         a = max(a_ev, a_pr)
#         b = max(b_ev, b_pr)
#         c = max(c_ev, c_pr)

#         # 5. Physical Properties
#         volume = (4/3) * np.pi * a * b * c
#         p = 1.6075
#         surface_area = 4 * np.pi * (
#             ((a*b)**p + (a*c)**p + (b*c)**p) / 3
#         )**(1/p)

#         # 6. Results Dictionary
#         D1, D2, D3 = 2*a, 2*b, 2*c
#         self.ellipsoid[key] = {
#             "status": status,
#             "mode": mode,
#             "D1": D1,
#             "D2": D2,
#             "D3": D3,
#             "volume": volume,
#             "surface": surface_area,
#             "asphericity": D1 / D3 if c > 0 else 1.0
#         }

#         if not noOutput:
#             results = self.ellipsoid[key]
#             centertxt(f"Hull {mode} Ellipsoid Analysis ({status})", 
#                       bgc='#007a7a', size='14', weight='bold')
#             print(f"  - Dimensions (Å): {results['D1']:.2f} x {results['D2']:.2f} x {results['D3']:.2f}")
#             print(f"  - Volume: {results['volume']/1000:.2f} nm³")
#             print(f"  - Surface: {results['surface']/100:.2f} nm²")
#             print(f"  - Asphericity: {results['asphericity']:.2f}")
#             print(f"  - Max Radius found: {max_dist/10:.3f} nm")
#             v1, v2, v3 = evecs[:,0]*a, evecs[:,1]*b, evecs[:,2]*c
#             key_cmd = key.replace(" ", "_")
#             jmol_cmd = (f"ellipsoid ID {key_cmd}_el AXES "
#                         f"{{{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}}} "
#                         f"{{{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}}} "
#                         f"{{{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}}}; "
#                         f"ellipsoid ID {key_cmd}_el CENTER "
#                         f"{{{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}}}; "
#                         f"color ${key_cmd}_el [x919191] translucent 0.3;")
#             print("\n  [Jmol Command to visualize the ellipsoid]:")
#             print(f"  {jmol_cmd}")

## tentative avec les plans
# def get_ellipsoid_analysis(self, noOutput=False, mode='inscribed'):
#         """
#         Perform a Principal Component Analysis (PCA) on the outer envelope to 
#         calculate the best-fitting circumscribed or inscribed ellipsoid.
#         This method identifies the principal axes of the nanoparticle's surface 
#         by analyzing the covariance matrix of the surface atoms or convex Hull vertices. The semi-axes
#         are scaled by projecting the target atoms onto each principal axis.
#         Works correctly for convex shapes (spheres, ellipsoids), symmetric
#         shapes (icosahedra), and non-convex shapes (peeled structures).
#         The analysis automatically selects between 'initial structure' and 
#         'optimized structure' based on the current state of the object.
#         Args:
#             noOutput (bool): If True, suppresses printed summaries and Jmol 
#                 command generation. Defaults to False.
#             mode (str): 'inscribed' (default) — PCA on surface atoms, ellipsoid
#                 fits inside the structure. 'circumscribed' — projection on hull
#                 vertices, ellipsoid contains all atoms. 
#         Returns:
#             dict: A dictionary containing the following physical properties:
#                 - "status": String indicating which geometry was analyzed (optimized or initial).
#                 - "mode": String indicating which ellipsoid has been calculated.
#                 - "D1", "D2", "D3": Major, intermediate, and minor diameters (Å).
#                 - "volume": Volume of the ellipsoid (Å³).
#                 - "surface": Approximate surface area (Å²) using Knud Thomsen's formula.
#                 - "asphericity": Ratio of D1/D3 (1.0 for a perfect sphere).
#         Raises:
#             ValueError: If fewer than 4 surface atoms are found.
#         Notes:
#             - PCA is performed on surface atoms (uniform coverage).
#             - Scaling is done by projecting surface atoms onto each PCA axis.
#             - Surface Area: Knud Thomsen approximation, error < 1.06%.
#             - Visualization: Jmol-ready command using AXES and CENTER keywords.
#         """
#         import numpy as np
#         if not hasattr(self, 'ellipsoid'):
#             self.ellipsoid = {}
            
#         # 1. Select the correct structure
#         if self.is_optimized and hasattr(self, 'vertices_opt'):
#             target_atoms = self.NP_opt
#             hull_indices = self.vertices_opt
#             status = "optimized envelope"
#             key = "optimized structure"
#         else:
#             target_atoms = self.NP
#             hull_indices = self.vertices
#             status = "initial envelope"
#             key = "initial structure"

#         hull_coords = target_atoms.get_positions()[hull_indices]
#         # 2. Select surface atoms
#         if self.is_optimized:
#             surface_mask = getattr(self, 'surfaceAtoms_opt', None)
#         else:
#             surface_mask = getattr(self, 'surfaceAtoms', None)

#         # Check that coreSurface() has been run
#         if surface_mask is None or np.count_nonzero(surface_mask) < 4:
#             raise ValueError(f"No surface atoms found for {status}. "
#                              "Please run coreSurface() before this analysis.")

#         if surface_mask is not None and np.count_nonzero(surface_mask) >= 4:
#             pts = target_atoms.get_positions()[surface_mask]
#         else:
#             pts = np.asarray(hull_coords)

#         if mode == 'circumscribed':
#             # Project hull vertices — ellipsoid contains all atoms
#             proj_pts = np.asarray(hull_coords)
#             center = proj_pts.mean(axis=0)
#             proj_pts = proj_pts - center
    
#             # PCA on proj_pts (consistent with scaling)
#             S = (proj_pts.T @ proj_pts) / len(proj_pts)
#             evals, evecs = np.linalg.eigh(S)
#             idx = np.argsort(evals)[::-1]
#             evals = evals[idx]
#             evecs = evecs[:, idx]
    
#             # 3. Scale — use hull vertices for true extent
    
#             max_dist = np.max(np.linalg.norm(proj_pts, axis=1))
#             scale_factor = max_dist / np.sqrt(evals[0])
            
#             # Eigenvalue scaling (correct ratios for symmetric shapes)
#             a_ev, b_ev, c_ev = scale_factor * np.sqrt(evals)
            
#             # Independent projection (correct extent for flat/elongated shapes)
#             a_pr = np.max(np.abs(proj_pts @ evecs[:, 0]))
#             b_pr = np.max(np.abs(proj_pts @ evecs[:, 1]))
#             c_pr = np.max(np.abs(proj_pts @ evecs[:, 2]))
            
#             # Take the maximum of both
#             a = max(a_ev, a_pr)
#             b = max(b_ev, b_pr)
#             c = max(c_ev, c_pr)
            
#         elif mode == 'inscribed':
#             # Use hull face equations — normals weighted by face distance
#             # This finds the axes that best fit the convex hull faces
#             if self.is_optimized:
#                 equations = getattr(self, 'equations_opt', None)
#             else:
#                 equations = getattr(self, 'equations', None)
            
#             # equations: [nx, ny, nz, d] with d = distance from origin
#             normals = equations[:, :3]   # unit normals
#             distances = np.abs(equations[:, 3])  # distances from origin
            
#             # Weighted PCA: weight each normal by its distance
#             weighted_pts = normals * distances[:, np.newaxis]
#             center = np.zeros(3)
#             S = (weighted_pts.T @ weighted_pts) / len(weighted_pts)
#             evals, evecs = np.linalg.eigh(S)
#             idx = np.argsort(evals)[::-1]
#             evals = evals[idx]; evecs = evecs[:, idx]
            
#             # Semi-axes = mean distance projected onto each axis
#             a = np.mean(distances)  # for isotropic shapes
#             # or better: max projection of face centers onto axes
#             face_centers = normals * distances[:, np.newaxis]
#             a = np.max(np.abs(face_centers @ evecs[:, 0]))
#             b = np.max(np.abs(face_centers @ evecs[:, 1]))
#             c = np.max(np.abs(face_centers @ evecs[:, 2]))
            
#         print(f"mode={mode}, n proj_pts={len(proj_pts)}")
#         print(f"max_dist={max_dist:.3f}")
#         print(f"evals={evals}")
#         print(f"a_ev={a_ev:.3f}, b_ev={b_ev:.3f}, c_ev={c_ev:.3f}")
#         print(f"a_pr={a_pr:.3f}, b_pr={b_pr:.3f}, c_pr={c_pr:.3f}")


#         # 5. Physical Properties
#         volume = (4/3) * np.pi * a * b * c
#         p = 1.6075
#         surface_area = 4 * np.pi * (
#             ((a*b)**p + (a*c)**p + (b*c)**p) / 3
#         )**(1/p)

#         # 6. Results Dictionary
#         D1, D2, D3 = 2*a, 2*b, 2*c
#         self.ellipsoid[key] = {
#             "status": status,
#             "mode": mode,
#             "D1": D1,
#             "D2": D2,
#             "D3": D3,
#             "volume": volume,
#             "surface": surface_area,
#             "asphericity": D1 / D3 if c > 0 else 1.0
#         }

#         if not noOutput:
#             results = self.ellipsoid[key]
#             centertxt(f"Hull {mode} Ellipsoid Analysis ({status})", 
#                       bgc='#007a7a', size='14', weight='bold')
#             print(f"  - Dimensions (Å): {results['D1']:.2f} x {results['D2']:.2f} x {results['D3']:.2f}")
#             print(f"  - Volume: {results['volume']/1000:.2f} nm³")
#             print(f"  - Surface: {results['surface']/100:.2f} nm²")
#             print(f"  - Asphericity: {results['asphericity']:.2f}")
#             print(f"  - Max Radius found: {max_dist/10:.3f} nm")
#             v1, v2, v3 = evecs[:,0]*a, evecs[:,1]*b, evecs[:,2]*c
#             key_cmd = key.replace(" ", "_")
#             jmol_cmd = (f"ellipsoid ID {key_cmd}_el AXES "
#                         f"{{{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}}} "
#                         f"{{{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}}} "
#                         f"{{{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}}}; "
#                         f"ellipsoid ID {key_cmd}_el CENTER "
#                         f"{{{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}}}; "
#                         f"color ${key_cmd}_el [x919191] translucent 0.3;")
#             print("\n  [Jmol Command to visualize the ellipsoid]:")
#             print(f"  {jmol_cmd}")

# def get_ellipsoid_analysis(self, noOutput=False, mode='inscribed'):
#         """..."""
#         import numpy as np
#         if not hasattr(self, 'ellipsoid'):
#             self.ellipsoid = {}
            
#         # 1. Select the correct structure
#         if self.is_optimized and hasattr(self, 'vertices_opt'):
#             target_atoms = self.NP_opt
#             hull_indices = self.vertices_opt
#             equations    = getattr(self, 'equations_opt', None)
#             surface_mask = getattr(self, 'surfaceAtoms_opt', None)
#             status = "optimized envelope"
#             key    = "optimized structure"
#         else:
#             target_atoms = self.NP
#             hull_indices = self.vertices
#             equations    = getattr(self, 'equations', None)
#             surface_mask = getattr(self, 'surfaceAtoms', None)
#             status = "initial envelope"
#             key    = "initial structure"

#         # 2. Check that coreSurface() has been run
#         if surface_mask is None or np.count_nonzero(surface_mask) < 4:
#             raise ValueError(f"No surface atoms found for {status}. "
#                              "Please run coreSurface() before this analysis.")

#         # 3. Select projection points depending on mode
#         if mode == 'circumscribed':
#             # Hull vertices — ellipsoid contains all atoms
#             proj_pts = target_atoms.get_positions()[hull_indices]

#         elif mode == 'inscribed':
#             # Hull face equations — normals × distances = face centers
#             # Works for any shape, even 13-atom icosahedra
#             if equations is None:
#                 raise ValueError(f"No hull equations found for {status}. "
#                                  "Please run coreSurface() before this analysis.")
#             normals   = equations[:, :3]
#             distances = np.abs(equations[:, 3])
#             proj_pts  = normals * distances[:, np.newaxis]  # face centers

#         else:
#             raise ValueError(f"Unknown mode '{mode}'. Choose 'inscribed' or 'circumscribed'.")

#         # 4. PCA
#         center  = proj_pts.mean(axis=0)
#         proj_c  = proj_pts - center
#         S       = (proj_c.T @ proj_c) / len(proj_c)
#         evals, evecs = np.linalg.eigh(S)
#         idx     = np.argsort(evals)[::-1]
#         evals   = evals[idx]
#         evecs   = evecs[:, idx]

#         # 5. Scale — combine eigenvalue scaling + independent projection
#         max_dist     = np.max(np.linalg.norm(proj_c, axis=1))
#         scale_factor = max_dist / np.sqrt(evals[0])
#         a_ev, b_ev, c_ev = scale_factor * np.sqrt(evals)
#         a_pr = np.max(np.abs(proj_c @ evecs[:, 0]))
#         b_pr = np.max(np.abs(proj_c @ evecs[:, 1]))
#         c_pr = np.max(np.abs(proj_c @ evecs[:, 2]))
#         a = max(a_ev, a_pr)
#         b = max(b_ev, b_pr)
#         c = max(c_ev, c_pr)

#         # 6. Physical Properties
#         volume = (4/3) * np.pi * a * b * c
#         p = 1.6075
#         surface_area = 4 * np.pi * (
#             ((a*b)**p + (a*c)**p + (b*c)**p) / 3
#         )**(1/p)

#         # 7. Results Dictionary
#         D1, D2, D3 = 2*a, 2*b, 2*c
#         self.ellipsoid[key] = {
#             "status":      status,
#             "mode":        mode,
#             "D1":          D1,
#             "D2":          D2,
#             "D3":          D3,
#             "volume":      volume,
#             "surface":     surface_area,
#             "asphericity": D1 / D3 if c > 0 else 1.0
#         }

#         if not noOutput:
#             results = self.ellipsoid[key]
#             centertxt(f"Ellipsoid Analysis — {mode} ({status})",
#                       bgc='#007a7a', size='14', weight='bold')
#             print(f"  - Dimensions (Å): {results['D1']:.2f} x {results['D2']:.2f} x {results['D3']:.2f}")
#             print(f"  - Volume: {results['volume']/1000:.2f} nm³")
#             print(f"  - Surface: {results['surface']/100:.2f} nm²")
#             print(f"  - Asphericity: {results['asphericity']:.2f}")
#             print(f"  - Max Radius found: {max_dist/10:.3f} nm")
#             v1, v2, v3 = evecs[:,0]*a, evecs[:,1]*b, evecs[:,2]*c
#             key_cmd = key.replace(" ", "_")
#             jmol_cmd = (f"ellipsoid ID {key_cmd}_el AXES "
#                         f"{{{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}}} "
#                         f"{{{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}}} "
#                         f"{{{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}}}; "
#                         f"ellipsoid ID {key_cmd}_el CENTER "
#                         f"{{{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}}}; "
#                         f"color ${key_cmd}_el [x919191] translucent 0.3;")
#             print("\n  [Jmol Command to visualize the ellipsoid]:")
#             print(f"  {jmol_cmd}")
# def get_ellipsoid_analysis(self, noOutput=False, mode='vertices'):
#     """
#     Perform a Principal Component Analysis (PCA) to calculate the best-fitting
#     ellipsoid of a nanoparticle, using three different sets of points.

#     The analysis automatically selects between 'initial structure' and
#     'optimized structure' based on the current state of the object.

#     Args:
#         noOutput (bool): If True, suppresses printed summaries and Jmol
#             command generation. Defaults to False.
#         mode (str): Defines which atoms are used for PCA and scaling:
#             - 'vertices' (default): PCA and scaling on convex hull vertices.
#               Gives the circumscribed ellipsoid — ellipsoid contains all atoms.
#               Verified to exactly match core-to-core dimensions measured in JMol.
#               Recommended for SAXS diameter comparison and for
#               peel_by_shifted_ellipsoid().
#             - 'planes': Weighted PCA on face centers of the ConvexHull built
#               on surface atoms (weighted by face area), scaled by projection of
#               surface atoms excluding hull vertices. Gives a slightly smaller
#               ellipsoid than 'vertices' — useful as a lower bound estimate.
#             - 'all': PCA on all atoms of the structure. Gives an intermediate
#               result between 'surface' and 'vertices'. Useful to evaluate the
#               influence of the atomic density distribution on the ellipsoid axes.

#     Returns:
#         dict: Stored in self.ellipsoid[key] with the following fields:
#             - "status" (str): 'initial envelope' or 'optimized envelope'.
#             - "mode"   (str): the mode used for this calculation.
#             - "D1", "D2", "D3" (float): major, intermediate and minor
#               diameters in Å, sorted in descending order.
#             - "volume"      (float): ellipsoid volume in Å³.
#             - "surface"     (float): ellipsoid surface area in Å²,
#               computed via Knud Thomsen's approximation (error < 1.06%).
#             - "asphericity" (float): D1/D3 ratio (1.0 for a perfect sphere).

#     Raises:
#         ValueError: If fewer than 4 surface atoms are found (coreSurface()
#             has not been run), or if an unknown mode is requested.

#     Notes:
#         - The Jmol command to visualize the ellipsoid is printed when
#           noOutput=False, using the AXES and CENTER keywords.
#         - Results are stored under self.ellipsoid['initial structure'] or
#           self.ellipsoid['optimized structure'] and overwritten on each call.
#         - Use effective_diameter() to get the volume-equivalent diameter in Å.
#     """
#     import numpy as np
#     if not hasattr(self, 'ellipsoid'):
#         self.ellipsoid = {}
        
#     # 1. Select the correct structure
#     if self.is_optimized and hasattr(self, 'vertices_opt'):
#         target_atoms = self.NP_opt
#         hull_indices = self.vertices_opt
#         equations    = getattr(self, 'equations_opt', None)
#         surface_mask = getattr(self, 'surfaceAtoms_opt', None)
#         status = "optimized envelope"
#         key    = "optimized structure"
#     else:
#         target_atoms = self.NP
#         hull_indices = self.vertices
#         equations    = getattr(self, 'equations', None)
#         surface_mask = getattr(self, 'surfaceAtoms', None)
#         status = "initial envelope"
#         key    = "initial structure"

#     # 2. Check that coreSurface() has been run
#     if surface_mask is None or np.count_nonzero(surface_mask) < 4:
#         raise ValueError(f"No surface atoms found for {status}. "
#                          "Please run coreSurface() before this analysis.")

#     # 3. Select projection points depending on mode and do PCA
#     if mode == 'vertices':
#         from scipy.spatial import ConvexHull
#         pos = target_atoms.get_positions()
#         hull = ConvexHull(pos)
#         hull_indices = hull.vertices
#         proj_pts = pos[hull_indices]
#         center  = target_atoms.get_center_of_mass()
#         proj_c  = proj_pts - center
#         S       = (proj_c.T @ proj_c) / len(proj_c)
#         jmol_center = center

#     elif mode == 'all':
#         # PCA on all atoms — gives the overall extent of the structure
#         proj_pts = target_atoms.get_positions()
#         center   = proj_pts.mean(axis=0)
#         jmol_center = center
#         proj_c   = proj_pts - center
#         S        = (proj_c.T @ proj_c) / len(proj_c)            

#     elif mode == 'planes':
#         from scipy.spatial import ConvexHull as _ConvexHull
#         from scipy.spatial import KDTree

#         surface_pts = target_atoms.get_positions()[surface_mask]

#         # --- Isotropy test on surface atoms ---
#         def _inertia_evals(pts):
#             I = np.zeros((3, 3))
#             for p in pts:
#                 I[0,0] += p[1]**2 + p[2]**2
#                 I[1,1] += p[0]**2 + p[2]**2
#                 I[2,2] += p[0]**2 + p[1]**2
#                 I[0,1] -= p[0]*p[1]
#                 I[0,2] -= p[0]*p[2]
#                 I[1,2] -= p[1]*p[2]
#             I[1,0]=I[0,1]; I[2,0]=I[0,2]; I[2,1]=I[1,2]
#             ev = np.linalg.eigvalsh(I)
#             return ev / ev.max()

#         surface_pts_c = surface_pts - surface_pts.mean(axis=0)
#         ev_norm  = _inertia_evals(surface_pts_c)
#         isotropy = ev_norm.min() / ev_norm.max()
#         tol_isotropy = 0.02  # 2% tolerance

#         # --- Weighted PCA on face centers → axes ---
#         hull_sa = _ConvexHull(surface_pts)
#         face_areas, face_centers = [], []
#         for simplex in hull_sa.simplices:
#             p1 = surface_pts[simplex[0]]
#             p2 = surface_pts[simplex[1]]
#             p3 = surface_pts[simplex[2]]
#             area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
#             face_areas.append(area)
#             face_centers.append((p1 + p2 + p3) / 3)
#         face_areas   = np.array(face_areas)
#         face_centers = np.array(face_centers)
#         weights      = face_areas / face_areas.sum()
#         center = (face_centers * weights[:, np.newaxis]).sum(axis=0)
#         jmol_center = center
#         proj_c = face_centers - center
#         S = np.zeros((3, 3))
#         for i, fc in enumerate(proj_c):
#             S += weights[i] * np.outer(fc, fc)

#     else:
#         raise ValueError(f"Unknown mode '{mode}'. Choose 'vertices', 'all' or 'surface'.")

#     evals, evecs = np.linalg.eigh(S)
#     idx     = np.argsort(evals)[::-1]
#     evals   = evals[idx]
#     evecs   = evecs[:, idx]


#     # 5. Scale
#     if mode == 'vertices':
#         # PCA axes from hull vertices, half-axes from ALL atoms
#         # → guaranteed circumscribed ellipsoid even for asymmetric NPs
#         all_proj = (pos - center) @ evecs
#         a = np.max(np.abs(all_proj[:, 0]))
#         b = np.max(np.abs(all_proj[:, 1]))
#         c = np.max(np.abs(all_proj[:, 2]))
#         max_dist = np.max(np.linalg.norm(pos - center, axis=1))

#     elif mode == 'all':
#         max_dist     = np.max(np.linalg.norm(proj_c, axis=1))
#         scale_factor = max_dist / np.sqrt(evals[0])
#         a_ev, b_ev, c_ev = scale_factor * np.sqrt(evals)
#         a_pr = np.max(np.abs(proj_c @ evecs[:, 0]))
#         b_pr = np.max(np.abs(proj_c @ evecs[:, 1]))
#         c_pr = np.max(np.abs(proj_c @ evecs[:, 2]))
#         a = max(a_ev, a_pr)
#         b = max(b_ev, b_pr)
#         c = max(c_ev, c_pr)
#         max_dist = max_dist

#     elif mode == 'planes':
#         if isotropy > 1 - tol_isotropy:
#             # Isotropic — force equal axes = max distance of face centers
#             max_dist = np.max(np.linalg.norm(proj_c, axis=1))
#             a = b = c = max_dist
#         else:
#             # Anisotropic — project face centers onto PCA axes
#             a = np.max(np.abs(proj_c @ evecs[:, 0]))
#             b = np.max(np.abs(proj_c @ evecs[:, 1]))
#             c = np.max(np.abs(proj_c @ evecs[:, 2]))
#             max_dist = np.max(np.linalg.norm(proj_c, axis=1))
        
#     # 6. Physical Properties
#     a, b, c = sorted([a, b, c], reverse=True)
#     volume = (4/3) * np.pi * a * b * c
#     p = 1.6075
#     surface_area = 4 * np.pi * (
#         ((a*b)**p + (a*c)**p + (b*c)**p) / 3
#     )**(1/p)

#     # 7. Results Dictionary
#     D1, D2, D3 = 2*a, 2*b, 2*c
#     self.ellipsoid[key] = {
#         "status":      status,
#         "mode":        mode,
#         "D1":          D1,
#         "D2":          D2,
#         "D3":          D3,
#         "volume":      volume,
#         "surface":     surface_area,
#         "asphericity": D1 / D3 if c > 0 else 1.0
#     }

#     if not noOutput:
#         results = self.ellipsoid[key]
#         centertxt(f"Ellipsoid Analysis — {mode} ({status})",
#                   bgc='#007a7a', size='14', weight='bold')
#         print(f"  - Dimensions (Å): {results['D1']:.2f} x {results['D2']:.2f} x {results['D3']:.2f}")
#         print(f"  - Volume: {results['volume']/1000:.2f} nm³")
#         print(f"  - Surface: {results['surface']/100:.2f} nm²")
#         print(f"  - Asphericity: {results['asphericity']:.2f}")
#         print(f"  - Max Radius found: {max_dist/10:.3f} nm")
#         v1, v2, v3 = evecs[:,0]*a, evecs[:,1]*b, evecs[:,2]*c
#         key_cmd = key.replace(" ", "_")
#         jmol_cmd = (f"ellipsoid ID {key_cmd}_el AXES "
#                     f"{{{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}}} "
#                     f"{{{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}}} "
#                     f"{{{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}}}; "
#                     f"ellipsoid ID {key_cmd}_el CENTER "
#                     f"{{{jmol_center[0]:.3f} {jmol_center[1]:.3f} {jmol_center[2]:.3f}}}; "
#                     f"color ${key_cmd}_el [x919191] translucent 0.3;")
#         print("\n  [Jmol Command to visualize the ellipsoid]:")
#         print(f"  {jmol_cmd}")
def get_ellipsoid_analysis(self, noOutput=False, mode='vertices'):
    """
    Perform a Principal Component Analysis (PCA) to calculate the best-fitting
    ellipsoid of a nanoparticle, using three different sets of points.

    The analysis automatically selects between 'initial structure' and
    'optimized structure' based on the current state of the object.

    Args:
        noOutput (bool): If True, suppresses printed summaries and Jmol
            command generation. Defaults to False.
        mode (str): Defines which atoms are used for PCA and scaling:
            - 'vertices' (default): PCA and scaling on convex hull vertices.
              Gives the circumscribed ellipsoid — ellipsoid contains all atoms.
              Verified to exactly match core-to-core dimensions measured in JMol.
              Recommended for SAXS diameter comparison and for
              peel_by_shifted_ellipsoid().
            - 'planes': Weighted PCA on face centers of the ConvexHull built
              on surface atoms (weighted by face area), scaled by projection of
              surface atoms excluding hull vertices. Gives a slightly smaller
              ellipsoid than 'vertices' — useful as a lower bound estimate.
            - 'all': PCA on all atoms of the structure. Gives an intermediate
              result between 'surface' and 'vertices'. Useful to evaluate the
              influence of the atomic density distribution on the ellipsoid axes.

    Returns:
        dict: Stored in self.ellipsoid[key] with the following fields:
            - "status" (str): 'initial envelope' or 'optimized envelope'.
            - "mode"   (str): the mode used for this calculation.
            - "D1", "D2", "D3" (float): major, intermediate and minor
              diameters in Å, sorted in descending order.
            - "volume"      (float): ellipsoid volume in Å³.
            - "surface"     (float): ellipsoid surface area in Å²,
              computed via Knud Thomsen's approximation (error < 1.06%).
            - "asphericity" (float): D1/D3 ratio (1.0 for a perfect sphere).

    Raises:
        ValueError: If fewer than 4 surface atoms are found (coreSurface()
            has not been run), or if an unknown mode is requested.

    Notes:
        - The Jmol command to visualize the ellipsoid is printed when
          noOutput=False, using the AXES and CENTER keywords.
        - Results are stored under self.ellipsoid['initial structure'] or
          self.ellipsoid['optimized structure'] and overwritten on each call.
        - Use effective_diameter() to get the volume-equivalent diameter in Å.
    """
    import numpy as np
    if not hasattr(self, 'ellipsoid'):
        self.ellipsoid = {}
        
    # 1. Select the correct structure
    if self.is_optimized and hasattr(self, 'vertices_opt'):
        target_atoms = self.NP_opt
        hull_indices = self.vertices_opt
        equations    = getattr(self, 'equations_opt', None)
        surface_mask = getattr(self, 'surfaceAtoms_opt', None)
        status = "optimized envelope"
        key    = "optimized structure"
    else:
        target_atoms = self.NP
        hull_indices = self.vertices
        equations    = getattr(self, 'equations', None)
        surface_mask = getattr(self, 'surfaceAtoms', None)
        status = "initial envelope"
        key    = "initial structure"

    # 2. Check that coreSurface() has been run
    if surface_mask is None or np.count_nonzero(surface_mask) < 4:
        raise ValueError(f"No surface atoms found for {status}. "
                         "Please run coreSurface() before this analysis.")

    # 3. Select projection points depending on mode and do PCA
    if mode == 'vertices':
        # Hull vertices — ellipsoid contains all atoms
        proj_pts = target_atoms.get_positions()[hull_indices]
        center  = proj_pts.mean(axis=0)
        proj_c  = proj_pts - center
        S       = (proj_c.T @ proj_c) / len(proj_c)

    elif mode == 'all':
        # PCA on all atoms — gives the overall extent of the structure
        proj_pts = target_atoms.get_positions()
        center   = proj_pts.mean(axis=0)
        proj_c   = proj_pts - center
        S        = (proj_c.T @ proj_c) / len(proj_c)            

    elif mode == 'planes':
        from scipy.spatial import ConvexHull as _ConvexHull
        from scipy.spatial import KDTree

        surface_pts = target_atoms.get_positions()[surface_mask]

        # --- Isotropy test on surface atoms ---
        def _inertia_evals(pts):
            I = np.zeros((3, 3))
            for p in pts:
                I[0,0] += p[1]**2 + p[2]**2
                I[1,1] += p[0]**2 + p[2]**2
                I[2,2] += p[0]**2 + p[1]**2
                I[0,1] -= p[0]*p[1]
                I[0,2] -= p[0]*p[2]
                I[1,2] -= p[1]*p[2]
            I[1,0]=I[0,1]; I[2,0]=I[0,2]; I[2,1]=I[1,2]
            ev = np.linalg.eigvalsh(I)
            return ev / ev.max()

        surface_pts_c = surface_pts - surface_pts.mean(axis=0)
        ev_norm  = _inertia_evals(surface_pts_c)
        isotropy = ev_norm.min() / ev_norm.max()
        tol_isotropy = 0.02  # 2% tolerance

        # --- Weighted PCA on face centers → axes ---
        hull_sa = _ConvexHull(surface_pts)
        face_areas, face_centers = [], []
        for simplex in hull_sa.simplices:
            p1 = surface_pts[simplex[0]]
            p2 = surface_pts[simplex[1]]
            p3 = surface_pts[simplex[2]]
            area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            face_areas.append(area)
            face_centers.append((p1 + p2 + p3) / 3)
        face_areas   = np.array(face_areas)
        face_centers = np.array(face_centers)
        weights      = face_areas / face_areas.sum()
        center = (face_centers * weights[:, np.newaxis]).sum(axis=0)
        proj_c = face_centers - center
        S = np.zeros((3, 3))
        for i, fc in enumerate(proj_c):
            S += weights[i] * np.outer(fc, fc)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'vertices', 'all' or 'surface'.")
 
    evals, evecs = np.linalg.eigh(S)
    idx     = np.argsort(evals)[::-1]
    evals   = evals[idx]
    evecs   = evecs[:, idx]


    # 5. Scale
    if mode in ('vertices', 'all'):
        # Project hull vertices onto PCA axes
        max_dist     = np.max(np.linalg.norm(proj_c, axis=1))
        scale_factor = max_dist / np.sqrt(evals[0])
        a_ev, b_ev, c_ev = scale_factor * np.sqrt(evals)
        a_pr = np.max(np.abs(proj_c @ evecs[:, 0]))
        b_pr = np.max(np.abs(proj_c @ evecs[:, 1]))
        c_pr = np.max(np.abs(proj_c @ evecs[:, 2]))
        a = max(a_ev, a_pr)
        b = max(b_ev, b_pr)
        c = max(c_ev, c_pr)
        max_dist = max_dist  # already defined

    elif mode == 'planes':
        if isotropy > 1 - tol_isotropy:
            # Isotropic — force equal axes = max distance of face centers
            max_dist = np.max(np.linalg.norm(proj_c, axis=1))
            a = b = c = max_dist
        else:
            # Anisotropic — project face centers onto PCA axes
            a = np.max(np.abs(proj_c @ evecs[:, 0]))
            b = np.max(np.abs(proj_c @ evecs[:, 1]))
            c = np.max(np.abs(proj_c @ evecs[:, 2]))
            max_dist = np.max(np.linalg.norm(proj_c, axis=1))
        
    # 6. Physical Properties
    volume = (4/3) * np.pi * a * b * c
    p = 1.6075
    surface_area = 4 * np.pi * (
        ((a*b)**p + (a*c)**p + (b*c)**p) / 3
    )**(1/p)

    # 7. Results Dictionary
    D1, D2, D3 = 2*a, 2*b, 2*c
    self.ellipsoid[key] = {
        "status":      status,
        "mode":        mode,
        "D1":          D1,
        "D2":          D2,
        "D3":          D3,
        "volume":      volume,
        "surface":     surface_area,
        "asphericity": D1 / D3 if c > 0 else 1.0
    }

    if not noOutput:
        results = self.ellipsoid[key]
        centertxt(f"Ellipsoid Analysis — {mode} ({status})",
                  bgc='#007a7a', size='14', weight='bold')
        print(f"  - Dimensions (Å): {results['D1']:.2f} x {results['D2']:.2f} x {results['D3']:.2f}")
        print(f"  - Volume: {results['volume']/1000:.2f} nm³")
        print(f"  - Surface: {results['surface']/100:.2f} nm²")
        print(f"  - Asphericity: {results['asphericity']:.2f}")
        print(f"  - Max Radius found: {max_dist/10:.3f} nm")
        v1, v2, v3 = evecs[:,0]*a, evecs[:,1]*b, evecs[:,2]*c
        key_cmd = key.replace(" ", "_")
        jmol_cmd = (f"ellipsoid ID {key_cmd}_el AXES "
                    f"{{{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}}} "
                    f"{{{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}}} "
                    f"{{{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}}}; "
                    f"ellipsoid ID {key_cmd}_el CENTER "
                    f"{{{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}}}; "
                    f"color ${key_cmd}_el [x919191] translucent 0.3;")
        print("\n  [Jmol Command to visualize the ellipsoid]:")
        print(f"  {jmol_cmd}")
            
def external_facets_info(self, mode='auto', noOutput=False):
    """
    Compute and display geometric properties of each facet of the NP,
    based on either the Wulff construction planes or the convex hull planes.

    Two labeling modes depending on the selected planes:
    - Wulff mode (trPlanes_Wulff): facets are labeled by Miller indices from
      self.surfacesWulff. Relative surface energies are computed from
      the Wulff distances via the Wulff theorem (e_i ∝ d_i).
    - Hull mode (trPlanes or trPlanes_opt): facets are labeled by their
      Miller indices (if convertible) or Cartesian normal direction.
      Relative energies are computed from distances.

    In all modes, facet areas are computed from the reduced convex hull
    facets (same as used in defCrystalShapeForJMol), and each hull facet
    is matched to its corresponding plane by maximizing the dot product
    between the facet normal and the plane normals. Facets absent from
    the reduced hull are reported as warnings.

    Args:
        mode (str): Which truncation planes to use. Options:
            - 'auto'        : automatically selects trPlanes_Wulff if
                              available, then trPlanes_opt, then trPlanes.
            - 'Wulff'       : use trPlanes_Wulff with Miller index labeling
                              and relative energy computation.
            - 'Slices'      : used trPlanes_Slices
            - 'crystal'     : use trPlanes = convex hull planes computed by
                              coreSurface() (initial structure).
            - 'crystal_opt' : use trPlanes_opt (optimized structure).
            Default is 'auto'.
        noOutput (bool): If True, suppresses all printed output.
            Default is False.

    Returns:
        tuple or None:
            - distances (np.ndarray): orthogonal distance from origin to
              each plane in Å.
            - e_relative (np.ndarray): relative surface energies normalized
              to 1 for the most stable face.
            - facet_areas_per_plane (list of float): area in Å² of each
              facet. 0.0 if absent from the reduced hull.
            Returns None if no truncation planes are available.
    """
    from .geometry import reduceHullFacets

    # --- Select target planes and mode ---
    if mode == 'auto':
        useWulff = hasattr(self, 'trPlanes_Wulff') and self.trPlanes_Wulff is not None
        if useWulff:
            target_planes = self.trPlanes_Wulff
        elif getattr(self, 'is_optimized', False) and getattr(self, 'trPlanes_opt', None) is not None:
            target_planes = self.trPlanes_opt
        elif getattr(self, 'trPlanes', None) is not None and len(self.trPlanes) > 0:
            # Prefer hull planes — always up to date regardless of slicing history
            target_planes = self.trPlanes
        elif getattr(self, 'trPlanes_Slices', None) is not None:
            # Fallback to slicing planes if hull not available
            target_planes = self.trPlanes_Slices
        else:
            target_planes = None
    elif mode == 'Wulff':
        target_planes = getattr(self, 'trPlanes_Wulff', None)
        useWulff = True
    elif mode == 'Slices':
        target_planes = getattr(self, 'trPlanes_Slices', None)
        useWulff = False
    elif mode == 'crystal':
        target_planes = getattr(self, 'trPlanes', None)
        useWulff = False
    elif mode == 'crystal_opt':
        target_planes = getattr(self, 'trPlanes_opt', None)
        useWulff = False
    else:
        target_planes = None
        useWulff = False

    if target_planes is None:
        if not noOutput:
            print(f"{bg.LIGHTYELLOWB}Warning: no truncation planes available "
                  f"(trPlanes_Wulff, trPlanes_opt, trPlanes are all None). "
                  f"Run the NP construction first.{bg.OFF}")
        return None

    planes    = np.array(target_planes)
    distances = np.abs(planes[:, 3])   # |d|, since ||n||=1

    # --- Relative energies (always computed from distances) ---
    d_min = distances.min()
    if d_min < 1e-10:
        # Avoid division by zero — planes through origin have d=0
        # Use second smallest non-zero distance as reference
        nonzero = distances[distances > 1e-10]
        if len(nonzero) > 0:
            d_min = nonzero.min()
        else:
            d_min = 1.0  # fallback — all planes through origin
    e_relative = distances / d_min
    if distances.min() < 1e-10 and not noOutput:
        print(f"  Warning: some planes pass through the origin (d=0) — "
              f"relative energies are computed from the smallest non-zero distance.")

    # --- Facet areas from reduceHullFacets ---
    vertices, redFacets = reduceHullFacets(self, noOutput=True,
                                           useWulff=useWulff)

    # --- One-to-one matching via greedy assignment ---
    facet_normals    = []
    facet_area_list  = []
    facet_areas_per_plane = [0.0] * len(planes)
    matched_planes   = set()

    for fi, nf in enumerate(redFacets):
        pts = np.array([vertices[i] for i in nf])
        area = 0.0
        for i in range(1, len(pts) - 1):
            v1 = pts[i]     - pts[0]
            v2 = pts[i + 1] - pts[0]
            area += 0.5 * np.linalg.norm(np.cross(v1, v2))
        if len(pts) >= 3:
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            fn = np.cross(v1, v2)
            norm = np.linalg.norm(fn)
            if norm > 1e-10:
                facet_normals.append(fn / norm)
                facet_area_list.append(area)

    if len(facet_normals) == 0:
        if not noOutput:
            print(f"{bg.LIGHTYELLOWB}Warning: no valid facet normals found.{bg.OFF}")
        return distances, e_relative, facet_areas_per_plane

    facet_normals = np.array(facet_normals)
    dot_matrix    = np.abs(facet_normals @ planes[:, :3].T)

    # Greedy one-to-one assignment
    assigned_facets = set()
    assigned_planes = set()
    while True:
        mask = np.ones_like(dot_matrix)
        for fi in assigned_facets:
            mask[fi, :] = 0
        for pi in assigned_planes:
            mask[:, pi] = 0
        if mask.sum() == 0:
            break
        best = np.argmax(dot_matrix * mask)
        fi, pi = np.unravel_index(best, dot_matrix.shape)
        if dot_matrix[fi, pi] < 0.9:
            break
        facet_areas_per_plane[pi] += facet_area_list[fi]
        matched_planes.add(pi)
        assigned_facets.add(fi)
        assigned_planes.add(pi)

    # --- Build expanded labels/energies BEFORE display ---
    # (needed both for warnings and for display)
    if useWulff:
        has_energies = (hasattr(self, 'eSurfacesWulff') and
                self.eSurfacesWulff is not None)
        from .symmetry import get_equivalent_miller_indices
        expanded_labels = []
        expanded_energies = []
        expanded_sizes = []
        for j, p in enumerate(self.surfacesWulff):
            if getattr(self, 'symWulff', False):
                sym_p = get_equivalent_miller_indices(self.ucSG_number, p)
                n_sym = len(sym_p)
                expanded_labels.extend(sym_p)
            else:
                n_sym = 1
                expanded_labels.append(p)
            has_energies = (hasattr(self, 'eSurfacesWulff') and
                            self.eSurfacesWulff is not None)
            if has_energies:
                expanded_energies.extend([self.eSurfacesWulff[j]] * n_sym)
            else:
                expanded_sizes.extend([self.sizesWulff[j]] * n_sym)
    
    # --- Warn about absent facets ---
    absent_planes = [i for i in range(len(planes)) if i not in matched_planes]
    if absent_planes and not noOutput:
        for i in absent_planes:
            if useWulff:
                p = expanded_labels[i]
                label = f"({p[0]:2d} {p[1]:2d} {p[2]:2d})"
            else:
                nn = planes[i, :3]
                label = f"[{nn[0]:+.2f} {nn[1]:+.2f} {nn[2]:+.2f}]"
            print(f"  {bg.LIGHTYELLOWB}Warning: plane {label} has no matching "
                  f"facet in the reduced hull — it may be too small or absent "
                  f"from this NP.{bg.OFF}")
    n_absent = len(absent_planes)

    # --- Sort by distance (descending) ---
    sort_idx = np.argsort(facet_areas_per_plane)[::-1]

    # --- Display ---
    if not noOutput:
        # --- Build mode label and attribute name for display ---
        mode_label_map = {
            'Wulff':       ('Wulff',             'trPlanes_Wulff'),
            'Slices':      ('Slices',             'trPlanes_Slices'),
            'crystal':     ('crystal',            'trPlanes'),
            'crystal_opt': ('optimized crystal',  'trPlanes_opt'),
        }

        if mode == 'auto':
            if useWulff:
                mode_label, attr_name = 'Wulff', 'trPlanes_Wulff'
            elif getattr(self, 'is_optimized', False) and \
                 getattr(self, 'trPlanes_opt', None) is not None:
                mode_label, attr_name = 'optimized crystal', 'trPlanes_opt'
            elif getattr(self, 'trPlanes_Slices', None) is not None:
                mode_label, attr_name = 'Slices', 'trPlanes_Slices'
            else:
                mode_label, attr_name = 'crystal', 'trPlanes'
        else:
            mode_label, attr_name = mode_label_map.get(
                mode, (mode, 'unknown'))

        centertxt(
            f"Surface area and relative energies analysis "
            f"— {mode_label} (Attribute: {attr_name})",
            bgc='#007a7a', size='14', weight='bold'
        )

        # Convert Cartesian normals to Miller indices for hull mode
        if not useWulff:
            from .core import round_to_Miller
            miller_indexes = []
            for p in planes[:, :3]: # Fixed 20260517
                try:
                    if hasattr(self, 'ucMatrix') and self.ucMatrix is not None:
                        p_miller = p @ self.ucMatrix.T
                        nonzero = np.abs(p_miller) > 1e-6
                        if nonzero.any():
                            p_miller = p_miller / np.max(np.abs(p_miller[nonzero]))
                    else:
                        p_miller = p  # fallback for non-Crystal objects
                    m, ok = round_to_Miller(p_miller.reshape(1, 3), tol=0.15)
                    miller_indexes.append(m[0] if ok else None)
                except Exception:
                    miller_indexes.append(None)

        # Build header
        if useWulff:
                    
            if has_energies:
                header = (f"{'Plane (hkl)':<25} {'d / nm':>8} {'e_input':>9} "
                          f"{'e_rel':>10} {'Area (nm²)':>15}")
            else:
                header = (f"{'Plane (hkl)':<25} {'d / nm':>8} {'D_input / nm':>9} "
                          f"{'e_rel':>10} {'Area (nm²)':>15}")
        else:
            header = (f"{'Plane (hkl)':<25} {'d / nm':>8} "
                      f"{'e_rel':>10} {'Area (nm²)':>15}")

        print(f"\n{'─'*len(header)}")
        print(header)
        print(f"{'─'*len(header)}")

        for i in sort_idx:
            area_str = (f"{facet_areas_per_plane[i] / 100:.2f}"
                        if facet_areas_per_plane[i] > 0 else "  absent")

            if useWulff:
                p = expanded_labels[i]
                label = f"({p[0]:2d} {p[1]:2d} {p[2]:2d})"
                if has_energies:
                    input_str = f"{expanded_energies[i]:9.3f}"
                else:
                    input_str = f"{expanded_sizes[i]:9.3f}"
                print(f"  {label:<24} {distances[i]/10:8.2f}   "
                  f"{e_relative[i]:8.3f}   {area_str:>12}")
            else:
                m = miller_indexes[i]
                if m is not None:
                    label = f"({int(m[0]):2d} {int(m[1]):2d} {int(m[2]):2d})"
                else:
                    nn = planes[i, :3]
                    label = f"[{nn[0]:+.3f} {nn[1]:+.3f} {nn[2]:+.3f}]"
                print(f"  {label:<24} {distances[i]/10:8.2f}   "
                  f"{e_relative[i]:8.3f}   {area_str:>12}")

        print(f"{'─'*len(header)}")
        if n_absent > 0:
            print(f"  {bg.LIGHTYELLOWB}{n_absent} plane(s) absent from "
                  f"the reduced hull.{bg.OFF}")
        print()

    return distances, e_relative, facet_areas_per_plane
#------------------------------------------------------------------------------------------------------------------------

def propPostMake(self, skipChiralityCalculation, skipSymmetryAnalyzis, skipFacetInfo,
                 thresholdCoreSurface, noOutput, is_optimized=False):
    """
    Compute and store various post-construction
    properties of the nanoparticle.

    This function calculates moments of inertia
    (MOI), determines the nanoparticle shape,
    analyzes symmetry (if required), and identifies
    core and surface atoms.

    Args:
        skipChiralityCalculation (bool): If True, skips the calculation of the opd index
        skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
        skipFacetInfo (bool): if True skips external facet info calculation
        thresholdCoreSurface (float): Threshold
            to distinguish core and surface atoms.
        noOutput (bool): If True, suppresses
            output messages.

    Attributes:
        moi (numpy.ndarray): Moment of inertia tensor.
        moisize (numpy.ndarray): Normalized moments of inertia.
        vertices (numpy.ndarray): Geometric vertices of the nanoparticle.
        simplices (numpy.ndarray): Simplices defining the convex hull.
        neighbors (numpy.ndarray): Neighboring relations between facets.
        equations (numpy.ndarray): Plane equations for the hull faces.
        NPcs (ase.Atoms): Copy of the nanoparticle with surface atoms 
            visually marked.
        NP (ase.Atoms): Original nanoparticle object.
        sasview_dims (tuple, optional): Dimensions for SasView, calculated 
            only if the sasview_dims() method exists.
        NPR (numpy.ndarray): normalized ratios of principal moments of inertia
            $NPR_{1}=I_{1}/I_{3}$ and $NPR_{2}=I_{2}/I_{3}$
        Rg (float): Radius of Gyration, in nm
    """

    # print(f"{skipChiralityCalculation=}")
    # print(f"{skipSymmetryAnalyzis=}")
    # print(f"{skipFacetInfo=}")
    # print(f"{thresholdCoreSurface=}")
    # print(f"{noOutput=}")
    # print(f"{is_optimized=}")

    # Determine the "Target" and the "Suffix"
    # This replaces hardcoded self.NP with a dynamic target
    suffix = "_opt" if is_optimized else ""
    target_np = self.NP_opt if is_optimized else self.NP
    status_label = "optimized structure" if is_optimized else "initial structure"

    if not noOutput:
        centertxt(
            f"Post-Make Calculation of Properties for the {status_label}", bgc='#004848', size='14', weight='bold', fgc="white",
        )

    if target_np is None:
        label = "NP_opt" if is_optimized else "NP"
        # This will stop the entire execution and show a red error block in Jupyter
        raise AttributeError(f"Structure Error: '{label}' is None. "
                             f"Cannot perform propPostMake() on an empty structure.")

    moiNP = compute_moi(target_np, noOutput)
    setattr(self, f"moi{suffix}", moiNP)
    setattr(self, f"moisize{suffix}", np.array(moi_size(target_np, noOutput)))
    setattr(self, f"NPR{suffix}", np.array(calculate_npr(moiNP, noOutput)))
    setattr(self, f"Rg{suffix}", calculate_rg(target_np, mass_weighted=True, noOutput=noOutput))
    if not skipChiralityCalculation:
        current_g0s = compute_opd_index(target_np, noOutput=noOutput)
        setattr(self, f"opd_index{suffix}", current_g0s)

    # Symmetry
    if not skipSymmetryAnalyzis:
        MolSym(target_np, noOutput=noOutput)
        
    # Core/surface
    (
        [v, s, n, e], 
        surfaceAtoms
    ) = coreSurface(self, thresholdCoreSurface, noOutput=noOutput)

    # Map hull properties to self.vertices / self.vertices_opt, etc.
    setattr(self, f"vertices{suffix}", v)
    setattr(self, f"simplices{suffix}", s)
    setattr(self, f"neighbors{suffix}", n)
    setattr(self, f"equations{suffix}", e)
    
    # Core/surface visualization
    # 102 is Nobelium, because it has a nice pinkish color in Jmol
    npcs_copy = target_np.copy()
    npcs_copy.numbers[np.invert(surfaceAtoms)] = 102

    setattr(self, f"NPcs{suffix}", npcs_copy)
    setattr(self, f"surfaceatoms{suffix}", npcs_copy[surfaceAtoms])
    setattr(self, f"surfaceAtoms{suffix}", surfaceAtoms)

    # Surface planes — ensure normals point outward from cog
    if hasattr(self, f'trPlanes{suffix}') and getattr(self, f'trPlanes{suffix}') is not None:
        current_planes = getattr(self, f'trPlanes{suffix}')
        cog = target_np.get_center_of_mass()
        setattr(self, f'trPlanes{suffix}', setdAsNegative(current_planes, cog=cog))

    # Jmol Crystal Shape
    if getattr(self, 'jmolCrystalShape', False) and not getattr(self, 'pbc', False):
        # We assume defCrystalShape... can handle the suffix or we pass use_opt
        cs = defCrystalShapeForJMol(self, noOutput=True)
        setattr(self, f"jMolCS{suffix}", cs)

    # Inscribed and circumscribed spheres
    Inscribed_circumscribed_spheres(self, noOutput)

    # Compute sasview_dims if the NP was optimized and if the method exists in the class definition
    _update_sasview_dims_from_spheres(self, noOutput)

    # Ellipsoid analysis (also calculates inscribed/circumscribed sphere radii
    get_ellipsoid_analysis(self, noOutput) 

    # External facets info
    if not skipFacetInfo and not getattr(self, 'pbc', False):
        if hasattr(self, 'trPlanes_Wulff') and self.trPlanes_Wulff is not None:
            self.external_facets_info(mode='Wulff', noOutput=noOutput)
        if hasattr(self, 'trPlanes_Slices') and self.trPlanes_Slices is not None:
            self.external_facets_info(mode='Slices', noOutput=noOutput)
        if (hasattr(self, 'trPlanes') and self.trPlanes is not None
                and not (hasattr(self, 'trPlanes_Wulff') and self.trPlanes_Wulff is not None)
                and not (hasattr(self, 'trPlanes_Slices') and self.trPlanes_Slices is not None)):
            self.external_facets_info(mode='crystal', noOutput=noOutput)
        if (hasattr(self, 'trPlanes_opt') and self.trPlanes_opt is not None):
            self.external_facets_info(mode='crystal_opt', noOutput=noOutput)

    # Specific print for regfccTd helix
    if (hasattr(self, 'n_Td')
            and self.n_Td > 1
            and not noOutput):
        # Just checking attribute existence
        # to avoid error on other classes
        if (hasattr(self, 'nAtomsAnalytic')
                and hasattr(self, 'nAtoms_helix')):
            print(f"\n{'=' * 60}")
            print(f"Helix Information:")
            print(
                f"  Number of tetrahedrons"
                f" in helix:"
                f" {self.n_Td}"
            )
            print(
                f"  Atoms per single"
                f" tetrahedron:"
                f" {self.nAtomsAnalytic()}"
            )
            print(
                f"  Total atoms in helix:"
                f" {self.nAtoms_helix}"
            )
            print(f"{'=' * 60}\n")

###### chirality index ################################################

@njit(parallel=True)
def _opd_kernel(coords, neighbors_indices, neighbors_offsets):
    """
    JIT-compiled kernel to compute the Osipov–Pickup–Dunmur sum in parallel.
    This function handles the core mathematical loops using machine code.
    """
    n = len(coords)
    g0 = 0.0
    
    # Parallel loop over all atoms using all available threads
    for i in prange(n):
        # Access neighbor list for atom i using offsets
        start_i = neighbors_offsets[i]
        end_i = neighbors_offsets[i+1]
        
        for idx_j in range(start_i, end_i):
            j = neighbors_indices[idx_j]
            if i == j: continue
            
            # Vector and squared distance between i and j
            vij = coords[i] - coords[j]
            rij_sq = np.sum(vij**2)
            
            # Neighbors of j to find the third atom k
            start_j = neighbors_offsets[j]
            end_j = neighbors_offsets[j+1]
            
            for idx_k in range(start_j, end_j):
                k = neighbors_indices[idx_k]
                if k == j or k == i: continue
                
                vjk = coords[j] - coords[k]
                rjk_sq = np.sum(vjk**2)
                
                # Neighbors of k to find the fourth atom l
                start_k = neighbors_offsets[k]
                end_k = neighbors_offsets[k+1]
                
                for idx_l in range(start_k, end_k):
                    l = neighbors_indices[idx_l]
                    if l == k or l == j or l == i: continue
                    
                    vkl = coords[k] - coords[l]
                    rkl_sq = np.sum(vkl**2)
                    
                    # Distance between the first and last atom
                    vil = coords[i] - coords[l]
                    ril = np.sqrt(np.sum(vil**2))

                    # Manual cross product implementation for maximum Numba speed
                    vcross_x = vij[1] * vkl[2] - vij[2] * vkl[1]
                    vcross_y = vij[2] * vkl[0] - vij[0] * vkl[2]
                    vcross_z = vij[0] * vkl[1] - vij[1] * vkl[0]
                    
                    # Dot products for the Osipov-Pickup-Dunmur formula
                    dot_triple = vcross_x * vil[0] + vcross_y * vil[1] + vcross_z * vil[2]
                    dot_ij_jk = vij[0] * vjk[0] + vij[1] * vjk[1] + vij[2] * vjk[2]
                    dot_jk_kl = vjk[0] * vkl[0] + vjk[1] * vkl[1] + vjk[2] * vkl[2]

                    # Denominator based on distances (scaled by r^2 for rij, rjk, rkl)
                    denominator = ril * (rij_sq * rjk_sq * rkl_sq)
                    
                    if denominator > 1e-12:
                        # Thread-safe accumulation (Numba handles reduction automatically)
                        g0 += (dot_triple * dot_ij_jk * dot_jk_kl) / denominator
                        
    return g0

def compute_opd_index(NP:Atoms, cutoff=6.0, noOutput=False):
    """
    High-performance Scaled Osipov–Pickup–Dunmur chirality index computation.
    Article: 10.1080/00268979500100831
    Uses KDTree for neighbor searching and Numba for parallelized math.
    
    Args:
        NP (ase.Atoms): ase object.
        cutoff (float): Radius for neighbor search (in Angstroms).
        
    Returns:
        float: The scaled chirality index G0s.
    """
    from scipy.spatial import KDTree
    coords = NP.get_positions()
    n = len(coords)
    if n < 4:
        return 0.0
        
    # 1. Build spatial index and find neighbors within cutoff
    tree = KDTree(coords)
    adj_list = tree.query_ball_point(coords, r=cutoff)
    
    # 2. Flatten the adjacency list for Numba-compatible array processing
    neighbors_indices = []
    neighbors_offsets = [0]
    for neighbors in adj_list:
        neighbors_indices.extend(neighbors)
        neighbors_offsets.append(len(neighbors_indices))
    
    neighbors_indices = np.array(neighbors_indices, dtype=np.int32)
    neighbors_offsets = np.array(neighbors_offsets, dtype=np.int32)
    
    # 3. Call the parallelized JIT kernel
    G0 = _opd_kernel(coords, neighbors_indices, neighbors_offsets)
    # 4. Apply final scaling factor: (8.0 / N^4) * G0
    G0 *= (8.0 / n**4)
    
    if not noOutput:
        centertxt(
            "Osipov–Pickup–Dunmur chirality index", bgc='#007a7a', size='14', weight='bold'
        )
        # Determine hand for visual feedback
        hand = "Right-Handed" if G0 > 0 else "Left-Handed"
        if abs(G0) < 1e-12: hand = "Achiral"
        
        # Final display line
        print(f" G0 = {G0:.2e} ({hand})")
    return G0

class AtomicRadii:
    """
    Container for atomic radii of a given element.
    All radii are stored in nm.
    """
    def __init__(self, el, ionic_list):
        self.metallic_radius = el.metallic_radius / 100 if el.metallic_radius else None
        self.covalent_radius = el.covalent_radius / 100 if el.covalent_radius else None
        self.vdw_radius      = el.vdw_radius      / 100 if el.vdw_radius      else None
        self.atomic_radius   = el.atomic_radius   / 100 if el.atomic_radius   else None
        self.ionic_radii     = ionic_list

    def get_ionic_radii(self, charge, coordination=None, spin=None):
        """
        Retrieve a specific ionic radius.
        Args:
            charge (int): Ionic charge (e.g. +1, +3).
            coordination (str, optional): Coordination number in Roman numerals
                (e.g. 'VI', 'IV', 'IVSQ'). See print_atomic_radii() for available values.
            spin (str, optional): Spin state ('High Spin', 'Low Spin', etc.).
        Returns:
            float: Ionic radius in nm, or None if not found.
        """
        for r in self.ionic_radii:
            if r['charge'] != charge:
                continue
            if coordination is not None and r['coordination'] != coordination:
                continue
            if spin is not None and r['spin'] != spin:
                continue
            return r['radius_ang']
        return None

    def __repr__(self):
        return (f"AtomicRadii(metallic_radius={self.metallic_radius:.4f} Å, "
                f"covalent_radius={self.covalent_radius:.4f} Å, "
                f"vdw_radius={self.vdw_radius:.4f} Å, "
                f"atomic_radius={self.atomic_radius:.4f} Å)")
        
def print_atomic_radii(element_symbol):
    """
    Print the available atomic radii for a given element using mendeleev.
    Helps the user choose the appropriate radius for the SAXS → core-to-core
    diameter correction: D_core = D_SAXS - 2 * r_atom.

    Args:
        element_symbol (str): Chemical symbol of the element (e.g. 'Ag', 'Au').

    Returns:
        dict: Available radii in nm, including ionic radii as a list of dicts.
    """
    from mendeleev import element as mendeleev_element
    el = mendeleev_element(element_symbol)

    print(f"Atomic radii for {el.name} ({element_symbol})")
    print(f"{'─'*50}")
    radii = {
        'metallic_radius': el.metallic_radius,
        'covalent_radius': el.covalent_radius,
        'vdw_radius'     :      el.vdw_radius,
        'atomic_radius'  :   el.atomic_radius,
    }
    for name, val in radii.items():
        if val is not None:
            print(f"  {name:<20} : {val:>5.0f} pm  =  {val/100:.4f} Å")
        else:
            print(f"  {name:<20} : not available")

    # Ionic radii
    print(f"\n  {'Ionic radii':}")
    print(f"  {'─'*46}")
    print(f"  {'Charge':<8} {'Coord.':<8} {'Spin':<12} {'Radius (pm)':<14} {'Radius (Å)'}")
    print(f"  {'─'*46}")
    ionic_list = []
    for ir in el.ionic_radii:
        spin_str = ir.spin if ir.spin else 'n/a'
        coord_str = str(ir.coordination) if ir.coordination else 'n/a'
        print(f"  {ir.charge:>+6}   {coord_str:<8} {spin_str:<12} "
              f"{ir.ionic_radius:>8.0f} pm   {ir.ionic_radius/100:.4f} Å")
        ionic_list.append({
            'charge':       ir.charge,
            'coordination': ir.coordination,
            'spin':         ir.spin,
            'radius_ang':   ir.ionic_radius / 100,  # pm → Å
        })

    print(f"{'─'*50}")
    print(f"  Recommended for metallic NPs : metallic radius")
    print(f"  Recommended for oxides/salts : ionic radius (choose charge and coordination)")
    print(f"  SAXS correction: D_core = D_SAXS - 2 × r")
    print(f"\n  To retrieve a specific radius (in Å):")
    print(f"    Ag = pyNMBu.print_atomic_radii('{element_symbol}')")
    print(f"    Ag.metallic_radius                                # metallic radius in Å")
    print(f"    Ag.covalent_radius                                # covalent radius in Å")
    print(f"    Ag.vdw_radius                                     # Van der Waals radius in Å")
    print(f"    Ag.atomic_radius                                  # atomic radius in Å")
    print(f"    Ag.get_ionic_radii(charge=+1, coordination='VI')  # ionic radius in Å")

    return AtomicRadii(el, ionic_list)

def effective_diameter(self, structure='optimized', mode='vertices',
                       method='rms', n_feret=2000):
    """
    Returns the effective diameter of the nanoparticle in Å.

    Args:
        structure (str): 'optimized' (default) or 'initial'.
        mode (str): Ellipsoid mode used when method requires the ellipsoid
            analysis — 'vertices' (default), 'all', or 'planes'.
            Ignored when method='feret' or method='rg'.
        method (str): How to compute the scalar diameter:
            - 'feret': Mean Feret diameter, averaged over
              n_feret random orientations. The Feret diameter along a
              direction n is the maximum projected extent of the NP.
              Most geometry-independent and physically meaningful —
              directly comparable to TEM image analysis.
              Works for any shape (sphere, icosahedron, nanorod, etc.).
            - 'rg': D = 2*sqrt(5/3)*Rg, where Rg is computed from the
              actual atomic positions. Consistent with the Guinier
              approximation used in SAXS. Exact for a solid uniform
              sphere; approximate for other shapes.
            - 'rms' (defaulkt): D = 2*sqrt((a²+b²+c²)/3) from ellipsoid semi-axes.
              Consistent with Guinier for a uniform solid ellipsoid.
            - 'volume': D = 2*(abc)^(1/3). Diameter of the sphere with
              the same volume as the ellipsoid.
            - 'arithmetic': D = 2*(a+b+c)/3. Arithmetic mean of the
              ellipsoid semi-axes.
            - 'surface': D based on Knud Thomsen surface area
              approximation (error < 1.06%).
            - 'radius': smallest ellipsoid axis D3 — for nanorods,
              returns the transverse diameter 2R.
            - 'length': largest ellipsoid axis D1 — for nanorods,
              returns the axial length L.
        n_feret (int): Number of random orientations for method='feret'.
            Default is 2000. Use 5000+ for highly asymmetric shapes.

    Returns:
        float: Effective diameter in Å.
    """
    import numpy as np

    # --- method='feret': computed directly from atomic positions ---
    if method == 'feret':
        if structure == 'optimized' and getattr(self, 'NP_opt', None) is not None:
            pos = self.NP_opt.get_positions()
        else:
            pos = self.NP.get_positions()

        # Center positions
        pos = pos - pos.mean(axis=0)

        # Random unit vectors on the sphere (reproducible)
        rng = np.random.default_rng(seed=42)
        xyz = rng.standard_normal((n_feret, 3))
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

        # Max projected extent along each direction
        projections = pos @ xyz.T                              # (n_atoms, n_feret)
        feret = projections.max(axis=0) - projections.min(axis=0)  # (n_feret,)

        return float(feret.mean())   # Å

    # --- method='rg': from atomic Rg ---
    if method == 'rg':
        rg_attr = 'Rg_opt' if (structure == 'optimized' 
                                and getattr(self, 'is_optimized', False)) else 'Rg'
        rg = getattr(self, rg_attr, None)
        if rg is None:
            raise ValueError(f"Rg not available for '{structure}' structure. "
                             f"Run propPostMake() first.")
        return 2 * np.sqrt(5/3) * rg * 10   # nm → Å

    # --- All other methods: require ellipsoid analysis ---
    key = 'optimized structure' if structure == 'optimized' else 'initial structure'
    if key not in self.ellipsoid or self.ellipsoid[key].get('mode') != mode:
        self.get_ellipsoid_analysis(noOutput=True, mode=mode)
    e = self.ellipsoid[key]
    a, b, c = e['D1']/2, e['D2']/2, e['D3']/2

    if method == 'rms':
        # Consistent with Guinier for a uniform solid ellipsoid
        return 2 * np.sqrt((a**2 + b**2 + c**2) / 3)

    elif method == 'volume':
        # Sphere of same volume
        return 2 * (a * b * c) ** (1/3)

    elif method == 'arithmetic':
        # Arithmetic mean of semi-axes
        return 2 * (a + b + c) / 3

    elif method == 'surface':
        # Sphere of same surface area (Knud Thomsen approximation)
        p = 1.6075
        return 2 * (((a*b)**p + (a*c)**p + (b*c)**p) / 3) ** (1 / (2*p))

    elif method == 'radius':
        # Transverse diameter — for nanorods (polydispersity on R)
        return e['D3']

    elif method == 'length':
        # Axial length — for nanorods (polydispersity on L)
        return e['D1']

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            f"'feret', 'rg', 'rms', 'volume', 'arithmetic', 'surface', "
            f"'radius', 'length'."
        )

def compare_effective_diameters(nmb_object, n_feret=2000):
    """
    Compare all effective diameter computation methods for a pyNanoMatBuilder NP.

    Displays a pivot table with all combinations of method × mode × structure,
    in nm. Methods 'feret' and 'rg' are computed directly from atomic positions
    and do not depend on the ellipsoid mode. All other methods ('rms', 'volume',
    'arithmetic', 'surface') are computed from the ellipsoid semi-axes and are
    evaluated for each of the three ellipsoid modes ('vertices', 'all', 'planes').

    Args:
        nmb_object: A pyNanoMatBuilder object with NP and optionally NP_opt.
        n_feret (int): Number of random orientations for method='feret'.
            Default is 1000.

    Returns:

    Example:
        compare_effective_diameters(ico)
        compare_effective_diameters(ico, n_feret=5000)
    """
    import pandas as pd
    import numpy as np

    methods_no_mode = ['feret', 'rg']
    methods_with_mode = ['rms', 'volume', 'arithmetic', 'surface']
    modes = ['vertices', 'all', 'planes']
    structures = ['initial', 'optimized']

    rows = []
    for structure in structures:
        # Skip optimized if NP_opt is not available
        if structure == 'optimized' and getattr(nmb_object, 'NP_opt', None) is None:
            continue

        for method in methods_no_mode:
            try:
                D = nmb_object.effective_diameter(
                    structure=structure, method=method, n_feret=n_feret
                ) / 10  # Å → nm
                rows.append({
                    'method'   : method,
                    'mode'     : '—',
                    'structure': structure,
                    'D (nm)'   : round(D, 4),
                })
            except Exception as e:
                rows.append({
                    'method'   : method,
                    'mode'     : '—',
                    'structure': structure,
                    'D (nm)'   : f"error: {e}",
                })

        for method in methods_with_mode:
            for mode in modes:
                try:
                    D = nmb_object.effective_diameter(
                        structure=structure, mode=mode,
                        method=method, n_feret=n_feret
                    ) / 10  # Å → nm
                    rows.append({
                        'method'   : method,
                        'mode'     : mode,
                        'structure': structure,
                        'D (nm)'   : round(D, 4),
                    })
                except Exception as e:
                    rows.append({
                        'method'   : method,
                        'mode'     : mode,
                        'structure': structure,
                        'D (nm)'   : f"error: {e}",
                    })

    df = pd.DataFrame(rows)

    # Pivot table: rows = (method, mode), columns = structure
    pivot = df.pivot_table(
        index=['method', 'mode'],
        columns='structure',
        values='D (nm)',
        aggfunc='first'
    )

    # Reorder rows: methods_no_mode first, then methods_with_mode × modes
    row_order = (
        [(m, '—') for m in methods_no_mode] +
        [(m, mo) for m in methods_with_mode for mo in modes]
    )
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    pivot.columns.name = None
    pivot.index.names = ['method', 'mode']

    display(pivot)
    return pivot