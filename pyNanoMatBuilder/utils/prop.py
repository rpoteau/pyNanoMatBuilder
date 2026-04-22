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
        centertxt(
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
        if npr1 < 0.1 and npr2 < 0.1:
            shape_desc = "Linear/Rod-like"
        elif npr1 < 0.3 and npr2 > 0.4:
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

def get_ellipsoid_analysis(self, noOutput=False):
        """
        Perform a Principal Component Analysis (PCA) on the outer envelope to 
        calculate the best-fitting circumscribed ellipsoid.

        This method identifies the principal axes of the nanoparticle's surface 
        by analyzing the covariance matrix of the Convex Hull vertices. The 
        resulting ellipsoid is scaled such that its major semi-axis matches 
        the maximum radial distance found in the structure, ensuring a perfect 
        fit for circumscribed diameter measurements (e.g., 1.63 nm for a 
        3-shell icosahedron).

        The analysis automatically selects between 'initial structure' and 
        'optimized structure' based on the current state of the object.

        Args:
            noOutput (bool): If True, suppresses printed summaries and Jmol 
                command generation. Defaults to False.

        Returns:
            dict: A dictionary containing the following physical properties:
                - "status": String indicating which envelope was analyzed.
                - "D1", "D2", "D3": Major, intermediate, and minor diameters (Å).
                - "volume": Volume of the ellipsoid (Å³).
                - "surface": Approximate surface area (Å²) using Knud Thomsen's formula.
                - "asphericity": Ratio of D1/D3 (1.0 for a perfect sphere).

        Raises:
            ValueError: If fewer than 4 Hull vertices are found, indicating 
                that coreSurface() has not been run or the NP is invalid.

        Notes:
            - Scaling Logic: The semi-axes are derived from the square root of 
              the eigenvalues of the covariance matrix. The scale factor is 
              defined as: scale = max_radius / sqrt(max_eigenvalue).
            - Surface Area: Calculated using an approximation with p=1.6075, 
              limiting the maximum relative error to 1.061%.
            - Visualization: Generates a Jmol-ready command using the 'AXES' 
              and 'CENTER' keywords for precise orientation.
        """
        import numpy as np
        if not hasattr(self, 'ellipsoid'):
            self.ellipsoid = {}
            
        # 1. Select the correct envelope data
        if self.is_optimized and hasattr(self, 'vertices_opt'):
            target_atoms = self.NP_opt
            hull_indices = self.vertices_opt
            status = "optimized envelope"
            key = "optimized structure"
        else:
            target_atoms = self.NP
            hull_indices = self.vertices
            status = "initial envelope"
            key = "initial structure"

        
        
        hull_coords = target_atoms.get_positions()[hull_indices]
        pts = np.asarray(hull_coords)
    
        if hull_coords is None or len(hull_coords) < 4:
            raise ValueError(f"Insufficient Hull vertices found for {status} ({len(hull_coords)} is < 4). "
                         "Please run coreSurface() before this analysis.")

        # 2. PCA on the envelope vertices only
        # Center the vertices on their own center of geometry
        center = pts.mean(axis=0)
        pos_c = pts - center
        
        # Compute covariance matrix of the surface points
        S = (pos_c.T @ pos_c) / len(hull_coords)
        # We use eigh to get the vectors (v)
        evals, evecs = np.linalg.eigh(S)
        # Sort both in descending order (Major -> Minor)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx] # Columns are the eigenvectors
        
        # 3. Scaling to match the circumscribed sphere (1.63 nm logic)
        # We ensure the major semi-axis 'a' equals the max distance from center
        max_dist = np.max(np.linalg.norm(pos_c, axis=1))
        scale_factor = max_dist / np.sqrt(evals[0])
        
        # Calculate semi-axes a, b, c
        a, b, c = scale_factor * np.sqrt(evals)
        
        # 4. Physical Properties
        # Volume: (4/3) * pi * a * b * c
        volume = (4/3) * np.pi * a * b * c
        
        # Surface Area (Knud Thomsen's formula - approximation error < 1.06%)
        p = 1.6075
        surface_area = 4 * np.pi * (
            ((a*b)**p + (a*c)**p + (b*c)**p) / 3
        )**(1/p)

        # 5. Results Dictionary
        D1, D2, D3 = 2*a, 2*b, 2*c

        self.ellipsoid[key] = {
            "status": status,
            "D1": D1, # Major (A) -> Should match 2 * max_dist
            "D2": D2, # Intermediate (A)
            "D3": D3, # Minor (A)
            "volume": volume,
            "surface": surface_area,
            "asphericity": D1 / D3 if c > 0 else 1.0
        }


        if not noOutput:
            results = self.ellipsoid[key]
            centertxt(f"Hull Ellipsoid Analysis ({status})", 
                        bgc='#007a7a',
                        size='14',
                        weight='bold')
            print(f"  - Dimensions (Å): {results['D1']:.2f} x {results['D2']:.2f} x {results['D3']:.2f}")
            print(f"  - Volume: {results['volume']/1000:.2f} nm³")
            print(f"  - Surface: {results['surface']/100:.2f} nm²")
            print(f"  - Asphericity: {results['asphericity']:.2f}")
            print(f"  - Max Radius found: {max_dist/10:.3f} nm")
            # --- Jmol Command Generation ---
            # Semi-axes vectors for Jmol
            v1, v2, v3 = evecs[:,0]*a, evecs[:,1]*b, evecs[:,2]*c
            
            # THE VALIDATED JMOL COMMAND
            key_cmd = key.replace(" ", "_")
            jmol_cmd = (f"ellipsoid ID {key_cmd}_el AXES "
                        f"{{{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}}} "
                        f"{{{v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}}} "
                        f"{{{v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}}}; "
                        f"ellipsoid ID {key_cmd}_el CENTER {{{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}}}; "
                        f"color ${key_cmd}_el [x919191] translucent 0.3;")
            
            print("\n  [Jmol Command to visualize the ellipsoid]:")
            print(f"  {jmol_cmd}")

#------------------------------------------------------------------------------------------------------------------------

from .geometry import coreSurface, setdAsNegative
from .symmetry import MolSym
from .external_pgm import defCrystalShapeForJMol

def propPostMake(self, skipChiralityCalculation, skipSymmetryAnalyzis, thresholdCoreSurface, noOutput, is_optimized=False):
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

    moiNP = moi(target_np, noOutput)
    setattr(self, f"moi{suffix}", moiNP)
    setattr(self, f"moisize{suffix}", np.array(moi_size(target_np, noOutput)))
    setattr(self, f"NPR{suffix}", np.array(calculate_npr(moiNP, noOutput)))
    setattr(self, f"Rg{suffix}", calculate_rg(target_np, mass_weighted=True, noOutput=noOutput))
    if not skipChiralityCalculation:
        current_g0s = compute_opd_index(target_np, noOutput=noOutput)
        setattr(self, f"opd_index{suffix}", current_g0s)

    # Core/surface
    if not skipSymmetryAnalyzis:
        MolSym(target_np, noOutput=noOutput)
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

    # Surface planes
    if hasattr(self, f'trPlanes{suffix}') and getattr(self, f'trPlanes{suffix}') is not None:
        current_planes = getattr(self, f'trPlanes{suffix}')
        setattr(self, f'trPlanes{suffix}', setdAsNegative(current_planes))

    # Jmol Crystal Shape
    if getattr(self, 'jmolCrystalShape', False):
        # We assume defCrystalShape... can handle the suffix or we pass use_opt
        cs = defCrystalShapeForJMol(self, noOutput=True)
        setattr(self, f"jMolCS{suffix}", cs)

    # Inscribed and circumscribed spheres
    Inscribed_circumscribed_spheres(self, noOutput)

    # Compute sasview_dims if the NP was optimized and if the method exists in the class definition
    _update_sasview_dims_from_spheres(self, noOutput)

    # Ellipsoid analysis (also calculates inscribed/circumscribed sphere radii
    get_ellipsoid_analysis(self, noOutput) 
        
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