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

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.groups import SpaceGroup

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

_DUMMY_LATTICES = {
    "cubic":        Lattice.cubic(1.0),
    "hexagonal":    Lattice.hexagonal(1.0, 1.633),
    "trigonal":     Lattice.hexagonal(1.0, 1.633),
    "tetragonal":   Lattice.tetragonal(1.0, 1.2),
    "orthorhombic": Lattice(np.diag([1.0, 1.2, 1.5])),
    "monoclinic":   Lattice.monoclinic(1.0, 1.2, 1.5, 80.0),
    "triclinic":    Lattice.from_parameters(1.0, 1.2, 1.5, 70.0, 80.0, 85.0),
}

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
    Return all lattice points equivalent to any of the lattice points
    in `hkl` with respect to rotations only (conserving distance to origin).

    Equivalent to ASE's deprecated Spacegroup.equivalent_lattice_points().

    Parameters
    ----------
    sg_input : int or SpacegroupAnalyzer or Structure
        Space group number (int), or a pymatgen SpacegroupAnalyzer,
        or a pymatgen Structure (from which symmetry will be analyzed).
    hkl : array-like, shape (N, 3)
        One or more Miller index triplets, e.g. [[0, 0, 2]].

    Returns
    -------
    np.ndarray, shape (M, 3)
        Unique equivalent lattice points, sorted lexicographically.

    Example
    -------
    >>> get_equivalent_miller_indices(225, [[0, 0, 2]])
    array([[ 0,  0, -2],
           [ 0, -2,  0],
           [-2,  0,  0],
           [ 2,  0,  0],
           [ 0,  2,  0],
           [ 0,  0,  2]])
    >>> get_equivalent_miller_indices(194, [[0, 0, 2]])  # hcp P6_3/mmc
    array([[ 0,  0, -2],
           [ 0,  0,  2]])
    """
    if isinstance(sg_input, int):
        crystal_system = SpaceGroup.from_int_number(sg_input).crystal_system
        lattice = _DUMMY_LATTICES[crystal_system]
        structure = Structure.from_spacegroup(
            sg_input, lattice, ["H"], [[0, 0, 0]]
        )
        analyzer = SpacegroupAnalyzer(structure)
    elif isinstance(sg_input, SpacegroupAnalyzer):
        analyzer = sg_input
    elif isinstance(sg_input, Structure):
        analyzer = SpacegroupAnalyzer(sg_input)
    else:
        raise TypeError(
            f"sg_input must be an int, SpacegroupAnalyzer, or Structure, got {type(sg_input)}"
        )

    # Matrices de rotation (N_sym, 3, 3) en coordonnées fractionnaires
    sym_ops = analyzer.get_symmetry_operations(cartesian=False)
    rot = np.array([op.rotation_matrix for op in sym_ops])  # (N_sym, 3, 3)

    # Application des rotations
    hkl = np.array(hkl, ndmin=2)                            # (N, 3)
    rotated = np.einsum("ni,sij->nsj", hkl, rot)            # (N, N_sym, 3)
    directions = rotated.reshape(-1, 3)                      # (N * N_sym, 3)
    directions = np.round(directions).astype(int)

    # Dédoublonnage + tri lexicographique (même comportement qu'ASE)
    ind = np.lexsort(directions.T)
    directions = directions[ind]
    diff = np.diff(directions, axis=0)
    mask = np.any(diff, axis=1)
    return np.vstack((directions[:-1][mask], directions[-1:]))

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
