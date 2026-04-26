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

def spacegroup_to_bravais(sg_number: int, hm_symbol: str = '') -> tuple[str, str]:
    """
    Return the crystal system acronyms compatible with interPlanarSpacing()
    from the international space group number and Hermann-Mauguin symbol.

    Args:
        sg_number (int): International space group number (1-230).
        hm_symbol (str): Hermann-Mauguin symbol (e.g. 'F m -3 m',
                         'R -3 m', 'P 63/m m c'). Only the first
                         character is used. Default is ''.

    Returns:
        tuple[str, str]: 
            - crystal_system (str): Crystal system acronym compatible with
              interPlanarSpacing(), among:
              'TRI', 'MCL', 'ORC', 'TET', 'TRH', 'HEX', 'CUB'
            - bravais_lattice (str): Extended Bravais lattice acronym, among:
              'TRI'           : triclinic
              'MCL'           : simple monoclinic
              'MCLC'          : base-centered monoclinic
              'ORC'           : simple orthorhombic
              'ORCF'          : face-centered orthorhombic
              'ORCI'          : body-centered orthorhombic
              'ORCC'          : base-centered orthorhombic
              'TET'           : simple tetragonal
              'BCT'           : body-centered tetragonal
              'TRH'           : trigonal rhombohedral
              'HEX'           : hexagonal / trigonal hexagonal axes
              'CUB'           : simple cubic
              'FCC'           : face-centered cubic
              'BCC'           : body-centered cubic

    Raises:
        ValueError: If sg_number is not in range 1-230.

    Examples:
        >>> spacegroup_to_bravais(225, 'F m -3 m')
        ('CUB', 'FCC')
        >>> spacegroup_to_bravais(229, 'I m -3 m')
        ('CUB', 'BCC')
        >>> spacegroup_to_bravais(194, 'P 63/m m c')
        ('HEX', 'HEX')
        >>> spacegroup_to_bravais(146, 'R -3 m')
        ('TRH', 'TRH')
        >>> spacegroup_to_bravais(12, 'C 1 2/m 1')
        ('MCL', 'MCLC')
        >>> spacegroup_to_bravais(69, 'F m m m')
        ('ORC', 'ORCF')
    """
    if not 1 <= sg_number <= 230:
        raise ValueError(
            f"Space group number must be between 1 and 230, got {sg_number}.")

    centering = hm_symbol.strip()[0].upper() if hm_symbol else ''

    # --- crystal_system (7 systems, for interPlanarSpacing) ---
    if   sg_number <= 2:    crystal_system = 'TRI'
    elif sg_number <= 15:   crystal_system = 'MCL'
    elif sg_number <= 74:   crystal_system = 'ORC'
    elif sg_number <= 142:  crystal_system = 'TET'
    elif sg_number <= 167:  crystal_system = 'TRH' if centering == 'R' else 'HEX'
    elif sg_number <= 194:  crystal_system = 'HEX'
    else:                   crystal_system = 'CUB'

    # --- bravais_lattice (extended, uses centering letter) ---
    if   sg_number <= 2:    bravais_lattice = 'TRI'
    elif sg_number <= 15:
        bravais_lattice = 'MCLC' if centering in ('A', 'B', 'C') else 'MCL'
    elif sg_number <= 74:
        if   centering == 'F': bravais_lattice = 'ORCF'
        elif centering == 'I': bravais_lattice = 'ORCI'
        elif centering in ('A', 'B', 'C'): bravais_lattice = 'ORCC'
        else:                  bravais_lattice = 'ORC'
    elif sg_number <= 142:
        bravais_lattice = 'BCT' if centering == 'I' else 'TET'
    elif sg_number <= 167:
        bravais_lattice = 'TRH' if centering == 'R' else 'HEX'
    elif sg_number <= 194:  bravais_lattice = 'HEX'
    else:
        if   centering == 'F': bravais_lattice = 'FCC'
        elif centering == 'I': bravais_lattice = 'BCC'
        else:                  bravais_lattice = 'CUB'

    return crystal_system, bravais_lattice

def crystallographic_angle(self,
                           v1,
                           v2,
                           type1: str = 'direction',
                           type2: str = 'direction',
                           noOutput: bool = True,
                          ):
    """
    Compute the angle between two crystallographic objects
    (directions or planes) in any crystal system.

    Args:
        self: A pyNMBcore instance with Gstar and G metric tensors.
        v1 (array-like): First vector [h, k, l] or [u, v, w].
        v2 (array-like): Second vector [h, k, l] or [u, v, w].
        type1 (str): Nature of v1: 'direction' [uvw] or 'plane' (hkl).
                     Default is 'direction'.
        type2 (str): Nature of v2: 'direction' [uvw] or 'plane' (hkl).
                     Default is 'direction'.
        noOutput (bool): If True, suppresses output. Default is True.

    Returns:
        float: Angle in degrees between the two objects.

    Raises:
        ValueError: If type1 or type2 is not 'direction' or 'plane'.

    Note:
        - angle between two directions [u1v1w1] and [u2v2w2]:
          uses the direct metric tensor G
        - angle between two planes (h1k1l1) and (h2k2l2):
          uses the reciprocal metric tensor G*
        - angle between a direction [uvw] and a plane (hkl):
          the plane normal in direct space is computed via G*,
          then the angle between the direction and the normal is
          computed, and 90° is subtracted to get the angle between
          the direction and the plane itself.

    Examples:
        # Angle between two directions in hcp
        NP.crystallographic_angle([0,0,1], [1,0,0],
                                  type1='direction', type2='direction')

        # Angle between two planes in fcc
        NP.crystallographic_angle([1,1,1], [1,0,0],
                                  type1='plane', type2='plane')

        # Angle between direction [110] and plane (115) in fcc
        NP.crystallographic_angle([1,1,0], [1,1,5],
                                  type1='direction', type2='plane')

    Note:
        All vectors are converted to Cartesian coordinates via lattice_cart
        before angle computation, ensuring correctness for all crystal systems
        including non-orthogonal lattices (hexagonal, trigonal, monoclinic,
        triclinic).
        - angle between two directions [u1v1w1] and [u2v2w2]:
          both converted to Cartesian, angle computed directly.
        - angle between two planes (h1k1l1) and (h2k2l2):
          plane normals computed via G* then converted to Cartesian.
        - angle between a direction [uvw] and a plane (hkl):
          90° - angle between direction and plane normal.
    """
    from .crystals import lattice_cart, normV
    import numpy as np

    valid_types = ('direction', 'plane')
    if type1 not in valid_types:
        raise ValueError(f"type1 must be 'direction' or 'plane', got '{type1}'.")
    if type2 not in valid_types:
        raise ValueError(f"type2 must be 'direction' or 'plane', got '{type2}'.")

    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # --- Convert everything to Cartesian ---
    # directions [uvw] → Cartesian via lattice vectors
    # planes (hkl) → Cartesian normal via reciprocal lattice (G*)
    if type1 == 'direction':
        c1 = lattice_cart(self, [v1], Bravais2cart=True, printV=not noOutput)[0]
    else:  # plane → normal in Cartesian
        c1 = v1 @ self.Gstar  # normal to (hkl) in reciprocal space
        c1 = lattice_cart(self, [c1], Bravais2cart=True, printV=not noOutput)[0]

    if type2 == 'direction':
        c2 = lattice_cart(self, [v2], Bravais2cart=True, printV=not noOutput)[0]
    else:  # plane → normal in Cartesian
        c2 = v2 @ self.Gstar
        c2 = lattice_cart(self, [c2], Bravais2cart=True, printV=not noOutput)[0]

    # --- Compute angle between the two Cartesian vectors ---
    c1 = normV(c1)
    c2 = normV(c2)
    cos_angle = np.clip(np.dot(c1, c2), -1, 1)
    angle = np.rad2deg(np.arccos(cos_angle))

    # --- For direction/plane: angle between direction and plane
    # = 90° - angle between direction and plane normal ---
    if (type1 == 'direction' and type2 == 'plane') or \
       (type1 == 'plane'     and type2 == 'direction'):
        angle = 90.0 - angle if angle <= 90.0 else angle - 90.0

    if not noOutput:
        t1 = f"[{int(v1[0])} {int(v1[1])} {int(v1[2])}]" \
             if type1 == 'direction' \
             else f"({int(v1[0])} {int(v1[1])} {int(v1[2])})"
        t2 = f"[{int(v2[0])} {int(v2[1])} {int(v2[2])}]" \
             if type2 == 'direction' \
             else f"({int(v2[0])} {int(v2[1])} {int(v2[2])})"
        print(f"Angle between {type1} {t1} and {type2} {t2} : {angle:.4f}°")

    return angle

def generateSlab(self,
                 hkl,
                 size_a: float = 2.0,
                 size_b: float = 2.0,
                 min_thickness: float = 5.0,
                 vacuum: float = 10.0,
                 backend: str = 'ase',
                 noOutput: bool = None,
                ):
    """
    Generate a crystallographic slab from Miller indices (hkl).

    Args:
        self: A Crystal instance with self.cif and self.crystal defined.
        hkl (array-like): Miller indices [h, k, l] of the surface plane.
        size_a (float): Target slab dimension along a in nm. Default is 2.0.
        size_b (float): Target slab dimension along b in nm. Default is 2.0.
        min_thickness (float): Minimum slab thickness in Å. Default is 5.0.
                                For backend='ase', converted to an estimated
                                number of layers. For backend='pymatgen',
                                passed directly as min_slab_size.
        vacuum (float): Vacuum thickness in Å on each side. Default is 10.0.
        backend (str): Slab generation backend: 'ase' or 'pymatgen'.
                       'pymatgen' gives better control over slab thickness
                       via min_thickness in Å. Default is 'ase'.
        noOutput (bool): If True, suppresses output. Default is self.noOutput.

    Returns:
        pyNMBcore: A shallow copy of the parent Crystal instance with
                   self.NP replaced by the generated slab.
    """
    from ase.build import surface as ase_surface
    import numpy as np
    import copy

    if noOutput is None:
        noOutput = self.noOutput

    # =========================================================
    # --- Backend: ASE ---
    # =========================================================
    if backend == 'ase':
        # Estimate number of layers from min_thickness
        h, k, l = hkl
        hkl_arr = np.array([h, k, l], dtype=float)
        d_hkl = 1.0 / np.sqrt(hkl_arr @ self.Gstar @ hkl_arr)
        n_layers = max(3, int(np.ceil(min_thickness / d_hkl)))

        slab_atoms = ase_surface(self.cif, tuple(hkl), n_layers, vacuum=vacuum)

        # --- Check for oblique cell ---
        cellpar = slab_atoms.cell.cellpar()
        for angle, name in zip(cellpar[3:], ['α', 'β', 'γ']):
            if abs(angle - 90.0) > 5.0:
                if not noOutput:
                    print(f"  {bg.DARKREDB}Warning: ASE cell is significantly "
                          f"oblique ({name}={angle:.1f}°) for plane {hkl}. "
                          f"Consider using backend='pymatgen' for better results "
                          f"with high-index planes.{bg.OFF}")
                break

        # Check if cell is already larger than requested size
        cell = slab_atoms.cell
        a_len = np.linalg.norm(cell[0])
        b_len = np.linalg.norm(cell[1])
        
        # Use ceil to always reach at least the requested size
        na = max(1, int(np.ceil(size_a * 10 / a_len)))
        nb = max(1, int(np.ceil(size_b * 10 / b_len)))

        if not noOutput:
            if a_len * na < size_a * 10 or b_len * nb < size_b * 10:
                print(f"  {bg.LIGHTYELLOWB}Warning: ASE minimal cell "
                      f"({a_len:.2f} × {b_len:.2f} Å) — repeating "
                      f"{na}×{nb} times to reach target size "
                      f"({size_a*10:.2f} × {size_b*10:.2f} Å).{bg.OFF}")

        slab_atoms = slab_atoms.repeat([na, nb, 1])

    # =========================================================
    # --- Backend: pymatgen ---
    # =========================================================
    elif backend == 'pymatgen':
        from pymatgen.core.structure import Structure
        from pymatgen.core.surface import SlabGenerator
        from pymatgen.io.ase import AseAtomsAdaptor

        # Convert ASE cif to pymatgen Structure
        structure = AseAtomsAdaptor.get_structure(self.cif)

        gen = SlabGenerator(
            structure,
            miller_index=hkl,
            min_slab_size=min_thickness,
            min_vacuum_size=vacuum,
            center_slab=True,
            in_unit_planes=False,
            primitive=True,
        )
        slabs = gen.get_slabs()
        if not slabs:
            raise ValueError(f"pymatgen could not generate a slab for {hkl}.")

        # Take the first (most symmetric) slab
        slab_pmg = slabs[0]

        if not noOutput and len(slabs) > 1:
            print(f"  pymatgen generated {len(slabs)} slab terminations — "
                  f"using the first (most symmetric).")

        # Try to get orthogonal cell
        try:
            slab_pmg_ortho = slab_pmg.get_orthogonal_c_slab()
            cell_angles = [slab_pmg_ortho.lattice.alpha,
                           slab_pmg_ortho.lattice.beta,
                           slab_pmg_ortho.lattice.gamma]
            is_orthogonal = all(abs(a - 90.0) < 1.0 for a in cell_angles)
            if is_orthogonal:
                slab_pmg = slab_pmg_ortho
                if not noOutput:
                    print(f"  {bg.LIGHTGREENB}Orthogonal cell obtained via "
                          f"get_orthogonal_c_slab(){bg.OFF}")
            else:
                if not noOutput:
                    print(f"  {bg.LIGHTYELLOWB}Warning: get_orthogonal_c_slab() "
                          f"did not produce a fully orthogonal cell "
                          f"(α={cell_angles[0]:.1f}° β={cell_angles[1]:.1f}° "
                          f"γ={cell_angles[2]:.1f}°) — keeping original.{bg.OFF}")
        except Exception as e:
            if not noOutput:
                print(f"  {bg.LIGHTYELLOWB}Warning: get_orthogonal_c_slab() "
                      f"failed ({e}) — keeping original cell.{bg.OFF}")

        slab_atoms = AseAtomsAdaptor.get_atoms(slab_pmg)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'ase' or 'pymatgen'.")

    # --- Count actual atomic layers along z ---
    pos = slab_atoms.get_positions()
    z_coords = np.sort(np.unique(np.round(pos[:, 2], decimals=1)))
    n_actual_layers = len(z_coords[z_coords < z_coords.max() - vacuum / 2])

    # --- Repeat to match requested size ---
    cell = slab_atoms.cell
    na = max(1, int(np.round(size_a * 10 / np.linalg.norm(cell[0]))))
    nb = max(1, int(np.round(size_b * 10 / np.linalg.norm(cell[1]))))
    slab_atoms = slab_atoms.repeat([na, nb, 1])

    # --- Preserve pbc ---
    slab_atoms.set_pbc(self.pbc)

    # --- Copy self and replace NP ---
    slab_obj = self.__class__.__new__(self.__class__)
    skip_attrs = {'NP', 'NPcs', 'NP_opt', 'NPcs_opt', 'sc', 'cog', 'nAtoms',
                  'trPlanes', 'trPlanes_opt', 'trPlanes_Wulff',
                  'jMolCS', 'jMolCS_opt', 'vertices', 'simplices',
                  'neighbors', 'equations', 'surfaceAtoms', 'surfaceatoms'}
    for attr, val in self.__dict__.items():
        if attr not in skip_attrs:
            try:
                setattr(slab_obj, attr, copy.copy(val))
            except Exception:
                setattr(slab_obj, attr, val)

    slab_obj.NP          = slab_atoms
    slab_obj.shape       = f'slab ({hkl[0]}{hkl[1]}{hkl[2]})'
    slab_obj.jmolCrystalShape = False
    slab_obj.cog         = slab_atoms.get_center_of_mass()
    slab_obj.nAtoms      = len(slab_atoms)
    slab_obj.trPlanes    = None
    slab_obj.trPlanes_Wulff = None
    slab_obj.is_optimized = False

    if not noOutput:
        h, k, l = hkl
        print(f"Crystallographic plane ({h} {k} {l})")
        print(f"  Crystal              : {self.crystal}")
        print(f"  Backend              : {backend}")
        print(f"  Min thickness        : {min_thickness:.1f} Å")
        print(f"  Actual atomic layers : {n_actual_layers}")
        print(f"  Atoms in slab        : {len(slab_atoms)}")
        print(f"  Dimensions           : {np.linalg.norm(slab_atoms.cell[0]):.2f} × "
              f"{np.linalg.norm(slab_atoms.cell[1]):.2f} Å")
        print(f"  Cell angles          : α={slab_atoms.cell.cellpar()[3]:.1f}° "
              f"β={slab_atoms.cell.cellpar()[4]:.1f}° "
              f"γ={slab_atoms.cell.cellpar()[5]:.1f}°")
        area = np.linalg.norm(np.cross(slab_atoms.cell[0], slab_atoms.cell[1]))
        print(f"  Surface area         : {area / 100:.4f} nm²")

    if self.aseView:
        from ase.visualize import view
        view(slab_obj.NP)
    return slab_obj
