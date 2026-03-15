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
    centerTitle("ase EMT calculator & Quasi Newton algorithm for geometry optimization")
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

