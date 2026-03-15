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
from .crystals import G, Gstar

######################################## ase unitcells and pymatgen symmetry analyzis

def print_spacegroup_info(sg_number):
    """
    Fetch and print crystallographic details for a given space group number.
    """

    from pymatgen.symmetry.groups import SpaceGroup
    
    sg = SpaceGroup.from_int_number(sg_number)
    info = (
        f"--- Details for Space Group {sg_number} ---\n"
        f"Symbol (Name)      : {sg.symbol}\n"
        f"Crystal System     : {sg.crystal_system}\n"
        f"Number of Ops      : {len(sg.symmetry_ops)}"
    )
    return info

def get_equivalent_miller_indices(sg_input, hkl):
    """
    Find all unique Miller indices equivalent to the input (hkl) plane.

    This function mimics ASE's `equivalent_lattice_points`. It can accept 
    either a full pyNanoMatBuilder Crystal system or a standard space group integer.

    Args:
        sg_input (int or object): Either the space group number (e.g., 225) 
            OR a Crystal system instance containing the SpacegroupAnalyzer (system.sga).
        hkl (list or np.array): The Miller indices [h, k, l] of the plane.

    Returns:
        np.ndarray: A 2D array of unique, symmetrically equivalent Miller indices.
    """
    import numpy as np
    from pymatgen.symmetry.groups import SpaceGroup
    
    # 1. Figure out what kind of input we received to get the rotation matrices
    if isinstance(sg_input, int):
        # It's a raw integer from your toy script
        sg = SpaceGroup.from_int_number(sg_input)
        rotations = [op.rotation_matrix for op in sg.symmetry_ops]
        
    elif hasattr(sg_input, 'sga'):
        # It's your Crystal object from the main library
        rotations = [op.rotation_matrix for op in sg_input.sga.get_symmetry_operations()]
        
    else:
        raise ValueError("sg_input must be an integer space group number or a Crystal system object.")
    
    # 2. Apply each rotation to the vector and round to handle floating-point precision
    all_variants = np.array([np.dot(rot, hkl) for rot in rotations])
    all_variants = np.round(all_variants, 8)
    
    # 3. Filter for unique rows
    unique_variants = np.unique(all_variants, axis=0)
    
    return unique_variants

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
        centertxt("Symmetry analysis", bgc='#007a7a', size='14', weight='bold')
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
