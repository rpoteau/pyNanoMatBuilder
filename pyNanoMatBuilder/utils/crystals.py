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
from .prop import kDTreeCN

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
        centertxt(
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

#######################################################################

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

