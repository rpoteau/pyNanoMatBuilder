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
