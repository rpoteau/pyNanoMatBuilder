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

######################################## local-order descriptors (CNP, Steinhardt)
# Parallelized with Numba following the same CSR (flattened neighbour list)
# pattern as _opd_kernel: the KDTree neighbour search stays in Python, and the
# heavy O(n . neighbours^2) loops run in a @njit(parallel=True) kernel.

def _build_csr_neighbours(coords, Rnn):
    """
    Build a flattened (CSR-style) neighbour list within cutoff Rnn, excluding
    self. Returns Numba-friendly typed arrays.

    Args:
        coords (np.ndarray): (N, 3) atomic positions.
        Rnn (float): neighbour cutoff distance in Angstroms.

    Returns:
        tuple:
            - neighbors_indices (np.ndarray[int32]): concatenated neighbour
              indices of every atom (self excluded).
            - neighbors_offsets (np.ndarray[int32]): (N+1,) offsets so that the
              neighbours of atom i are
              neighbors_indices[offsets[i]:offsets[i+1]].
    """
    import numpy as np
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    adj = tree.query_ball_point(coords, r=Rnn)

    neighbors_indices = []
    neighbors_offsets = [0]
    for i, nb in enumerate(adj):
        nb = [j for j in nb if j != i]          # drop self
        neighbors_indices.extend(nb)
        neighbors_offsets.append(len(neighbors_indices))

    neighbors_indices = np.array(neighbors_indices, dtype=np.int32)
    neighbors_offsets = np.array(neighbors_offsets, dtype=np.int32)
    return neighbors_indices, neighbors_offsets


@njit(parallel=True)
def _cnp_kernel(coords, neighbors_indices, neighbors_offsets):
    """
    JIT-compiled Common Neighbour Parameter kernel (Tsuzuki et al.).

    For each atom i computes Q_i = (1/n_i) * sum_j || sum_k (r_ik + r_jk) ||^2,
    where j are the neighbours of i and k the neighbours common to i and j.
    All loops run in machine code; the outer loop is parallel over atoms.
    """
    n = len(coords)
    Q = np.zeros(n)

    for i in prange(n):
        start_i = neighbors_offsets[i]
        end_i = neighbors_offsets[i + 1]
        n_i = end_i - start_i
        if n_i == 0:
            continue

        acc = 0.0
        for idx_j in range(start_i, end_i):
            j = neighbors_indices[idx_j]
            start_j = neighbors_offsets[j]
            end_j = neighbors_offsets[j + 1]

            # accumulate the vector sum over common neighbours k of i and j
            sx = 0.0; sy = 0.0; sz = 0.0
            # intersect neighbour list of i with that of j by scanning both;
            # the lists are short (first shell) so a nested scan is cheap
            for idx_k in range(start_i, end_i):
                k = neighbors_indices[idx_k]
                if k == j:
                    continue
                # is k also a neighbour of j ?
                is_common = False
                for idx_m in range(start_j, end_j):
                    if neighbors_indices[idx_m] == k:
                        is_common = True
                        break
                if not is_common:
                    continue
                # r_ik + r_jk = (coords[k]-coords[i]) + (coords[k]-coords[j])
                sx += (coords[k, 0] - coords[i, 0]) + (coords[k, 0] - coords[j, 0])
                sy += (coords[k, 1] - coords[i, 1]) + (coords[k, 1] - coords[j, 1])
                sz += (coords[k, 2] - coords[i, 2]) + (coords[k, 2] - coords[j, 2])

            acc += sx * sx + sy * sy + sz * sz

        Q[i] = acc / n_i

    return Q

def common_neighbour_parameter(self, Xnn, noOutput=False, store=True,
                               is_optimized=None):
    """
    Compute the Common Neighbour Parameter (CNP) of Tsuzuki et al. for each atom.

    The CNP is a per-atom scalar that measures the departure of the local
    environment from a centrosymmetric crystalline arrangement. For atom i:

        Q_i = (1 / n_i) * sum_j || sum_k (r_ik + r_jk) ||^2

    where j runs over the n_i neighbours of i (within Rnn), and k runs over
    the neighbours common to both i and j. In a perfect, centrosymmetric
    lattice the bond vectors to the common neighbours cancel pairwise, so
    Q_i -> 0. At surfaces, edges, vertices, stacking faults and twin planes
    the symmetry is broken and Q_i grows. CNP therefore grades the local
    disorder continuously, complementing the binary core/surface split from
    coreSurface() and revealing internal defects (e.g. the twin planes of
    pentatwinned rods) that the convex hull cannot see.

    The heavy computation runs in a Numba-parallelized kernel (_cnp_kernel),
    falling back to pure Python if Numba is unavailable.

    Reference: Tsuzuki, Branicio, Rino, Comput. Phys. Commun. 177 (2007) 518.

    Args:
        self: pyNMBcore instance (must hold self.NP and, if optimized, self.NP_opt).
        Xnn (float): Neighbour cutoff distance in Angstroms. Should sit between
            the first and second coordination shells (typically ~1.2-1.3 times
            the nearest-neighbour distance, Rnn).
        noOutput (bool): If True, suppresses all printed output. Default False.
        store (bool): If True, stores the per-atom array on the object as
            self.cnp (or self.cnp_opt) and the mean as self.cnp_mean
            (or self.cnp_mean_opt). Default True.
        is_optimized (bool or None): Force the target structure. If None,
            uses self.is_optimized to decide between NP and NP_opt.

    Returns:
        numpy.ndarray: (nAtoms,) array of Q_i values, in Angstrom^2.

    Note:
        Q_i has units of length^2. Values scale with Xnn^2, so only compare
        CNP computed with the same cutoff. Calibrate the FCC/{111}/{100}/edge
        signatures on a clean regfccOh before interpreting a sculpted NP.
    """
    import numpy as np

    # --- target structure selection (same convention as the rest of prop.py)
    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    if is_optimized and getattr(self, 'NP_opt', None) is not None:
        target_atoms = self.NP_opt
        status = "optimized structure"
        suffix = "_opt"
    else:
        target_atoms = self.NP
        status = "initial structure"
        suffix = ""

    if not noOutput:
        centertxt("Common Neighbour Parameter (CNP, Tsuzuki et al.)",
                  bgc='#007a7a', size='14', weight='bold')
        chrono = timer(); chrono.chrono_start()

    coords = np.ascontiguousarray(target_atoms.get_positions(), dtype=np.float64)
    n = len(coords)

    nbr_idx, nbr_off = _build_csr_neighbours(coords, Xnn)
    Q = _cnp_kernel(coords, nbr_idx, nbr_off)

    Q_mean = float(Q.mean()) if n else 0.0

    if store:
        setattr(self, f"cnp{suffix}", Q)
        setattr(self, f"cnp_mean{suffix}", Q_mean)
        from .external_pgm import defLocalOrderColorForJMol
        defLocalOrderColorForJMol(self, descriptor='cnp', color='turbo',
                                  is_optimized=is_optimized, noOutput=True)

    if not noOutput:
        print(f" - Source              : {status}")
        print(f" - Cutoff Xnn          : {Xnn:.3f} Å")
        print(f" - Mean CNP <Q>        : {Q_mean:.3f} Å²")
        print(f" - CNP range           : {Q.min():.3f} – {Q.max():.3f} Å²")
        print(f" - Stored as           : self.cnp{suffix}  (per atom),"
              f" self.cnp_mean{suffix}  (scalar)")
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    return Q


@njit(parallel=True)
def _steinhardt_kernel(coords, neighbors_indices, neighbors_offsets,
                       l, plm_table, two_l_plus_1):
    """
    JIT-compiled Steinhardt q_l kernel.

    For each atom, accumulates the complex q_lm coefficients from the bond
    directions to its neighbours, then forms the rotationally invariant
    q_l = sqrt( 4pi/(2l+1) * sum_m |q_lm|^2 ). Spherical harmonics are built
    in place from precomputed associated-Legendre normalisation constants
    (plm_table) and an explicit recurrence, so no SciPy call is needed inside
    the kernel. The outer loop over atoms is parallel.

    Args:
        coords (np.ndarray): (N, 3) positions.
        neighbors_indices, neighbors_offsets: CSR neighbour list.
        l (int): harmonic degree.
        plm_table (np.ndarray): (l+1,) normalisation constants N_lm for m>=0.
        two_l_plus_1 (int): 2l+1.

    Returns:
        np.ndarray: (N,) q_l per atom.
    """
    n = len(coords)
    ql = np.zeros(n)

    for i in prange(n):
        start_i = neighbors_offsets[i]
        end_i = neighbors_offsets[i + 1]
        n_i = end_i - start_i
        if n_i == 0:
            continue

        # real/imag parts of q_lm for m = 0..l (use symmetry for negative m)
        re = np.zeros(l + 1)
        im = np.zeros(l + 1)

        for idx_j in range(start_i, end_i):
            j = neighbors_indices[idx_j]
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            dz = coords[j, 2] - coords[i, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-12:
                continue
            ct = dz / r                      # cos(theta)
            phi = np.arctan2(dy, dx)

            # associated Legendre P_l^m(cos theta) for m = 0..l, standard
            # recurrence (no Condon-Shortley here; folded into plm_table sign)
            pmm = 1.0
            somx2 = np.sqrt(max(0.0, 1.0 - ct * ct))
            # P_m^m
            plm = np.zeros(l + 1)
            # build P_l^m for each m via vertical recurrence
            for m in range(0, l + 1):
                # compute P_m^m
                pmm_val = 1.0
                fact = 1.0
                for _k in range(1, m + 1):
                    pmm_val *= -fact * somx2
                    fact += 2.0
                if m == l:
                    plm[m] = pmm_val
                    continue
                # P_{m+1}^m
                pmmp1 = ct * (2.0 * m + 1.0) * pmm_val
                if m + 1 == l:
                    plm[m] = pmmp1
                    continue
                # upward in degree to reach l
                pll = 0.0
                pm0 = pmm_val
                pm1 = pmmp1
                for ll in range(m + 2, l + 1):
                    pll = ((2.0 * ll - 1.0) * ct * pm1 -
                           (ll + m - 1.0) * pm0) / (ll - m)
                    pm0 = pm1
                    pm1 = pll
                plm[m] = pll

            for m in range(0, l + 1):
                ylm_norm = plm_table[m] * plm[m]
                re[m] += ylm_norm * np.cos(m * phi)
                im[m] += ylm_norm * np.sin(m * phi)

        # average over neighbours and assemble the invariant
        inv_ni = 1.0 / n_i
        s = 0.0
        for m in range(0, l + 1):
            rm = re[m] * inv_ni
            imm = im[m] * inv_ni
            mag2 = rm * rm + imm * imm
            if m == 0:
                s += mag2
            else:
                # negative m contributes the same magnitude (|Y_l,-m| = |Y_lm|)
                s += 2.0 * mag2

        ql[i] = np.sqrt(4.0 * np.pi / two_l_plus_1 * s)

    return ql


def _legendre_norm_table(l):
    """
    Precompute the spherical-harmonic normalisation constants
    N_lm = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! ) for m = 0..l.
    Returned as a float64 array indexed by m.
    """
    import numpy as np
    from math import factorial, pi, sqrt
    tab = np.zeros(l + 1)
    for m in range(l + 1):
        tab[m] = sqrt((2 * l + 1) / (4 * pi) *
                      factorial(l - m) / factorial(l + m))
    return tab


def steinhardt_q(self, Xnn, l=6, noOutput=False, store=True,
                 is_optimized=None):
    """
    Compute the Steinhardt bond-orientational order parameter q_l per atom.

    q_l is a rotationally invariant per-atom scalar built from the spherical
    harmonics of the bond directions to the neighbours of each atom
    (Steinhardt, Nelson, Ronchetti, Phys. Rev. B 28 (1983) 784):

        q_lm(i) = (1 / N_i) * sum_j Y_lm( theta_ij, phi_ij )
        q_l(i)  = sqrt( (4 pi / (2l+1)) * sum_m |q_lm(i)|^2 )

    For l = 6 this discriminates FCC, HCP, BCC and icosahedral local order,
    and flags twin/stacking-fault environments — the standard descriptor for
    metallic nanoparticles. Unlike the 2D hexatic parameter it needs no choice
    of reference plane or axis.

    The heavy computation runs in a Numba-parallelized kernel
    (_steinhardt_kernel) that builds the spherical harmonics in place, so SciPy
    is not needed; it falls back to pure Python if Numba is unavailable.

    Reference values (perfect lattices, first shell):
        FCC q6 ~ 0.575, HCP q6 ~ 0.485, BCC q6 ~ 0.511, ICO q6 ~ 0.663.

    Args:
        self: pyNMBcore instance (self.NP / self.NP_opt).
        Xnn (float): Neighbour cutoff distance in Angstroms. Should sit between
            the first and second coordination shells (typically ~1.2-1.3 times
            the nearest-neighbour distance, Rnn). Same meaning as
            in common_neighbour_parameter.
        l (int): Spherical-harmonic degree. Default 6. l=4 is also useful to
            separate FCC from HCP; pass l=4 in a second call if needed.
        noOutput (bool): If True, suppresses output. Default False.
        store (bool): If True, stores the per-atom array as self.q6 / self.q6_opt
            (named after l, e.g. self.q4) and the mean as self.q6_mean, etc.
            Default True.
        is_optimized (bool or None): Force target structure; None -> self.is_optimized.

    Returns:
        numpy.ndarray: (nAtoms,) array of q_l values (dimensionless, in [0, 1]).

    Note:
        Atoms with no neighbour inside Xnn get q_l = 0. q_l is dimensionless
        and cutoff-robust as long as Xnn captures the first shell only.
    """
    import numpy as np

    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    if is_optimized and getattr(self, 'NP_opt', None) is not None:
        target_atoms = self.NP_opt
        status = "optimized structure"
        suffix = "_opt"
    else:
        target_atoms = self.NP
        status = "initial structure"
        suffix = ""

    if not noOutput:
        centertxt(f"Steinhardt bond-orientational order q{l}",
                  bgc='#007a7a', size='14', weight='bold')
        chrono = timer(); chrono.chrono_start()

    coords = np.ascontiguousarray(target_atoms.get_positions(), dtype=np.float64)
    n = len(coords)

    nbr_idx, nbr_off = _build_csr_neighbours(coords, Xnn)
    plm_table = _legendre_norm_table(l)
    ql = _steinhardt_kernel(coords, nbr_idx, nbr_off,
                            l, plm_table, 2 * l + 1)

    ql_mean = float(ql.mean()) if n else 0.0

    if store:
        setattr(self, f"q{l}{suffix}", ql)
        setattr(self, f"q{l}_mean{suffix}", ql_mean)
        from .external_pgm import defLocalOrderColorForJMol
        defLocalOrderColorForJMol(self, descriptor='q', l=l, color='turbo',
                                  is_optimized=is_optimized, noOutput=True)

    if not noOutput:
        print(f" - Source              : {status}")
        print(f" - Cutoff Xnn          : {Xnn:.3f} Å")
        print(f" - Degree l            : {l}")
        print(f" - Mean q{l} <q{l}>      : {ql_mean:.3f}")
        print(f" - q{l} range           : {ql.min():.3f} – {ql.max():.3f}")
        print(f" - Stored as           : self.q{l}{suffix}  (per atom),"
              f" self.q{l}_mean{suffix}  (scalar)")
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    return ql

def plot_local_order(self, descriptor='cnp', Xnn=None, l=6,
                     is_optimized=None, save_path=None, color='turbo',
                     bins=50, noOutput=False):
    """
    Graphical analysis of a per-atom local-order descriptor (CNP or
    Steinhardt q_l): a histogram of the value distribution next to a
    projected scatter of the atoms coloured by the descriptor.

    The descriptor is read from the object (self.cnp / self.q{l}, with the
    _opt suffix when the optimized structure is targeted). If it is not
    present and a cutoff Xnn is supplied, it is computed on the fly via
    common_neighbour_parameter / steinhardt_q (which also store the per-atom
    array and the matching Jmol colouring command, e.g. self.jMol_cnp).

    Args:
        self: pyNMBcore instance.
        descriptor (str): 'cnp' (default) or 'q' for Steinhardt q_l.
        Xnn (float): Neighbour cutoff in Å used to compute the descriptor if
            it is not already stored. If None and the descriptor is missing,
            a ValueError is raised.
        l (int): Harmonic degree for descriptor='q'. Default 6.
        is_optimized (bool or None): Target structure. None -> self.is_optimized.
        save_path (str): If given, saves the figure (.png or .svg).
        color (str): Matplotlib colormap name. Default 'turbo'.
        bins (int): Number of histogram bins. Default 50.
        noOutput (bool): If True, suppresses the file-saved message and any
            on-the-fly descriptor output. Default is False.

    Returns:
        numpy.ndarray: the (nAtoms,) descriptor array that was plotted.

    Note:
        The scatter is a 2D projection onto the two principal axes of the
        structure (PCA), purely for visual inspection. For the genuine 3D
        view, use the Jmol command stored in self.jMol_{descriptor}
        (e.g. self.jMol_cnp, self.jMol_q6), which writes the descriptor into
        the atomic property and colours by it.
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    use_opt = is_optimized and getattr(self, 'NP_opt', None) is not None
    suffix = "_opt" if use_opt else ""
    target_atoms = self.NP_opt if use_opt else self.NP
    status = "optimized structure" if use_opt else "initial structure"

    # --- resolve attribute name and human-readable label ------------------
    if descriptor == 'cnp':
        attr = f"cnp{suffix}"
        label = "CNP  (Å²)"
        title = "Common Neighbour Parameter"
    elif descriptor == 'q':
        attr = f"q{l}{suffix}"
        label = f"q{l}"
        title = f"Steinhardt q{l}"
    else:
        raise ValueError(f"descriptor must be 'cnp' or 'q', got '{descriptor}'.")

    # --- fetch or compute the descriptor ----------------------------------
    values = getattr(self, attr, None)
    if values is None:
        if Xnn is None:
            raise ValueError(
                f"self.{attr} is not available. Either run the descriptor "
                f"first or pass Xnn to compute it here.")
        if descriptor == 'cnp':
            values = common_neighbour_parameter(self, Xnn, noOutput=True,
                                                store=True,
                                                is_optimized=is_optimized)
        else:
            values = steinhardt_q(self, Xnn, l=l, noOutput=True, store=True,
                                  is_optimized=is_optimized)

    values = np.asarray(values)
    pos = target_atoms.get_positions()

    # --- 2D PCA projection of the atoms for the scatter -------------------
    c = pos.mean(axis=0)
    pc = pos - c
    S = (pc.T @ pc) / len(pc)
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    proj = pc @ evecs[:, :2]          # project on the two largest axes

    # --- figure: histogram (left) + coloured scatter (right) --------------
    fig, (axh, axs) = plt.subplots(1, 2, figsize=(14, 6))

    cmap = matplotlib.colormaps[color]
    vmin, vmax = float(values.min()), float(values.max())
    span = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
    n_h, bins_h, patches = axh.hist(values, bins=bins, edgecolor='black',
                                    alpha=0.9)
    for patch, left in zip(patches, bins_h[:-1]):
        centre = left + (bins_h[1] - bins_h[0]) / 2.0
        patch.set_facecolor(cmap((centre - vmin) / span))
    axh.axvline(values.mean(), color='crimson', linestyle='--', linewidth=2,
                label=f"mean = {values.mean():.3f}")
    axh.set_xlabel(label, fontsize=13, fontweight='bold')
    axh.set_ylabel("Number of atoms", fontsize=13, fontweight='bold')
    axh.set_title(f"{title}: distribution ({status})",
                  fontsize=12, fontweight='bold')
    axh.legend(prop={'weight': 'bold', 'size': 12})
    axh.grid(True, linestyle=':', alpha=0.6)
    for tk in (axh.get_xticklabels() + axh.get_yticklabels()):
        tk.set_fontweight('bold')

    sc = axs.scatter(proj[:, 0] / 10, proj[:, 1] / 10, c=values, cmap=color,
                     s=40, edgecolors='black', linewidth=0.3, alpha=0.9)
    axs.set_xlabel("PC1 (nm)", fontsize=13, fontweight='bold')
    axs.set_ylabel("PC2 (nm)", fontsize=13, fontweight='bold')
    axs.set_title(f"{title}: 2D projection ({status})",
                  fontsize=12, fontweight='bold')
    axs.set_aspect('equal')
    axs.grid(True, linestyle=':', alpha=0.6)
    for tk in (axs.get_xticklabels() + axs.get_yticklabels()):
        tk.set_fontweight('bold')
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(label, size=12, weight='bold')
    for tk in cbar.ax.get_yticklabels():
        tk.set_fontweight('bold')

    plt.tight_layout()
    if save_path:
        if save_path.lower().endswith('.svg'):
            import matplotlib
            matplotlib.rcParams['svg.fonttype'] = 'none'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not noOutput:
            print(f"✅ Plot saved to: {save_path}")
    plt.show()

    # --- Jmol colour-mapping command (stored as attribute) ----------------
    from .external_pgm import defLocalOrderColorForJMol
    defLocalOrderColorForJMol(self, descriptor=descriptor, l=l, color=color,
                              is_optimized=is_optimized, noOutput=noOutput)

def local_order_populations(self, descriptor='cnp', l=6, decimals=2,
                            color='turbo', is_optimized=None, noOutput=False):
    """
    Group atoms by their (rounded) local-order descriptor value and report
    each population, with an index, a colour swatch matching the colormap, and
    the atom count. Each value cluster corresponds to a crystallographically
    equivalent set of atoms (core, twin plane, facet, edge, vertex...). The
    index can be passed to select_by_local_order() to extract a population.

    The descriptor must already be stored (self.cnp / self.q{l}, _opt suffix
    for the optimized structure). Run common_neighbour_parameter() or
    steinhardt_q() first.

    Args:
        descriptor (str): 'cnp' (default) or 'q' for Steinhardt q_l.
        l (int): Harmonic degree for descriptor='q'. Default 6.
        decimals (int): Rounding applied before grouping. Default 2.
        color (str): Matplotlib colormap name, matched to the Jmol scheme.
            Default 'turbo'.
        is_optimized (bool or None): Target structure. None -> self.is_optimized.
        noOutput (bool): If True, suppresses the printed table. Default False.

    Returns:
        tuple: (values, counts) — sorted unique descriptor values (their index
            is their position in this array) and the atom count of each
            population.
    """
    import numpy as np
    import matplotlib

    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    use_opt = is_optimized and getattr(self, 'NP_opt', None) is not None
    suffix = "_opt" if use_opt else ""
    status = "optimized structure" if use_opt else "initial structure"

    if descriptor == 'cnp':
        attr = f"cnp{suffix}"
        label = "CNP (Å²)"
    elif descriptor == 'q':
        attr = f"q{l}{suffix}"
        label = f"q{l}"
    else:
        raise ValueError(f"descriptor must be 'cnp' or 'q', got '{descriptor}'.")

    values = getattr(self, attr, None)
    if values is None:
        raise ValueError(f"self.{attr} not found. Run the descriptor first "
                         f"(common_neighbour_parameter / steinhardt_q).")

    values = np.asarray(values)
    uniq, counts = np.unique(np.round(values, decimals), return_counts=True)
    # remember the grouping so select_by_local_order() can reuse it
    self._local_order_decimals = decimals

    if not noOutput:
        centertxt(f"Local-order populations — {label} ({status})",
                  bgc='#007a7a', size='14', weight='bold')
        cmap = matplotlib.colormaps[color]
        vmin, vmax = float(uniq.min()), float(uniq.max())
        span = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
        total = counts.sum()
        for i, (v, c) in enumerate(zip(uniq, counts)):
            rgb = cmap((v - vmin) / span)
            r, g, b = (int(255 * x) for x in rgb[:3])
            swatch = f"\033[48;2;{r};{g};{b}m    \033[0m"
            print(f" - [{i:2d}] {swatch} {label:>10} {v:8.{decimals}f}  →  "
                  f"{c:6d} atoms ({100 * c / total:5.1f} %)")
        print(f" - {len(uniq)} distinct populations, {total} atoms total")
        hint_l = f", l={l}" if descriptor == 'q' else ""
        print(f" - select with: NP.select_by_local_order(index, "
              f"descriptor='{descriptor}'{hint_l})")

    return uniq, counts

def select_by_local_order(self, indices, descriptor='cnp', l=6,
                          is_optimized=None, noOutput=False):
    """
    Build self.NP_select from the atoms belonging to one or more local-order
    populations, identified by their index in local_order_populations().

    Indices refer to the rounded unique values returned (and printed) by
    local_order_populations() with the SAME descriptor, l and decimals. The
    selection is non-destructive: self.NP is left untouched; the chosen atoms
    are copied into self.NP_select (and the boolean mask into
    self.NP_select_mask).

    Args:
        indices (int or list of int): population index/indices to select,
            as shown by local_order_populations() (e.g. 5 or [0, 5]).
        descriptor (str): 'cnp' (default) or 'q' for Steinhardt q_l.
        l (int): Harmonic degree for descriptor='q'. Default 6.
        is_optimized (bool or None): Target structure. None -> self.is_optimized.
        noOutput (bool): If True, suppresses output. Default False.

    Returns:
        ase.Atoms: the selected sub-structure (also stored as self.NP_select).
    """
    import numpy as np
    from ase import Atoms

    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    use_opt = is_optimized and getattr(self, 'NP_opt', None) is not None
    suffix = "_opt" if use_opt else ""
    target = self.NP_opt if use_opt else self.NP

    # decimals is inherited from the last local_order_populations() call,
    # so the indices passed here match the table the user just read.
    decimals = getattr(self, '_local_order_decimals', None)
    if decimals is None:
        hint_l = f", l={l}" if descriptor == 'q' else ""
        raise ValueError(
            f"local_order_populations() has not been run yet, so population "
            f"indices are undefined. Call "
            f"local_order_populations(descriptor='{descriptor}'{hint_l}) "
            f"first to see the index/value table, then select by index.")

    if descriptor == 'cnp':
        attr = f"cnp{suffix}"
        label = "CNP (Å²)"
    elif descriptor == 'q':
        attr = f"q{l}{suffix}"
        label = f"q{l}"
    else:
        raise ValueError(f"descriptor must be 'cnp' or 'q', got '{descriptor}'.")

    values = getattr(self, attr, None)
    if values is None:
        raise ValueError(f"self.{attr} not found. Run the descriptor first "
                         f"(common_neighbour_parameter / steinhardt_q).")

    values  = np.asarray(values)
    rounded = np.round(values, decimals)
    uniq    = np.unique(rounded)

    if isinstance(indices, (int, np.integer)):
        indices = [int(indices)]
    for i in indices:
        if i < 0 or i >= len(uniq):
            raise ValueError(f"population index {i} out of range "
                             f"(0..{len(uniq) - 1}).")

    wanted_vals = uniq[list(indices)]
    mask = np.isin(rounded, wanted_vals)

    pos  = target.get_positions()
    elem = target.get_chemical_symbols()
    np_select = Atoms(
        symbols=[elem[k] for k in np.where(mask)[0]],
        positions=pos[mask])
    setattr(self, f"NP_select{suffix}", np_select)
    setattr(self, f"NP_select_mask{suffix}", mask)

    if not noOutput:
        centertxt("Selecting atoms by local-order population",
                  bgc='#007a7a', size='14', weight='bold')
        for i in indices:
            print(f" - population [{i}] {label} = {uniq[i]:.{decimals}f}")
        print(f" - {np.count_nonzero(mask)} atoms selected "
              f"out of {len(target)}. Stored as self.NP_select{suffix}.")

def plot_q4q6_map(self, Xnn=None, is_optimized=None, save_path=None,
                  aggregate=True, decimals=3, sc_domain=False, noOutput=False):
    """
    Plot the Steinhardt (q4, q6) map: the five ideal crystalline reference
    points (FCC, BCC, HCP, SC, icosahedral) together with the per-atom data of
    this nanoparticle, so one can see how far each atom departs from an ideal
    local environment.

    The reference points are tabulated literature values (perfect first-shell
    environments; the BCC reference includes the first two shells, which sit
    close together). Because Steinhardt parameters depend slightly on the
    neighbour cutoff, the NP data should be read RELATIVE to the references:
    a core atom sits on its lattice point, surface/twin atoms drift away.

    For crystalline NPs, many atoms share the EXACT same (q4, q6) value and
    would overplot at a single pixel. With aggregate=True (default) atoms are
    grouped by their rounded (q4, q6) value and drawn as one marker per value,
    its area proportional to the atom count and its colour encoding that count
    — so a population holding half the particle is immediately visible. With
    aggregate=False every atom is drawn individually (useful for disordered /
    optimized structures where values form a continuous cloud).

    Tabulated values:
    Pieter Rein ten Wolde, Maria J. Ruiz-Montero, Daan Frenkel (1996)
    Numerical calculation of the rate of crystal nucleation in a Lennard-Jones
    system at moderate undercooling.
    J. Chem. Phys. 104: 9932-9947. https://doi.org/10.1063/1.471721

    Requires self.q4 and self.q6 (and the _opt variants for the optimized
    structure). If absent and Xnn is given, both are computed on the fly.

    Args:
        Xnn (float): Neighbour cutoff in Å, used only if q4/q6 must be
            computed here. Ignored if both are already stored.
        is_optimized (bool or None): Target structure. None -> self.is_optimized.
        save_path (str): If given, saves the figure (.png or .svg).
        aggregate (bool): If True (default), group atoms by rounded (q4, q6)
            value and size each marker by its atom count. If False, draw one
            dot per atom.
        decimals (int): Rounding used to group identical (q4, q6) values when
            aggregate=True. Default 3.
        sc_domain (bool): If True, include the simple-cubic (SC) reference at
            (0.764, 0.354) and widen the x-range to show it. If False (default),
            SC is omitted and the view is tightened on the FCC/BCC/HCP/ICO
            region where metallic NPs live (SC sits far to the right and would
            otherwise compress the useful range).
        noOutput (bool): If True, suppresses the saved-file message. Default False.

    Returns:
        tuple: (q4, q6) per-atom arrays that were plotted.
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    use_opt = is_optimized and getattr(self, 'NP_opt', None) is not None
    suffix = "_opt" if use_opt else ""
    status = "optimized structure" if use_opt else "initial structure"

    # --- fetch or compute q4 and q6 ---------------------------------------
    q4 = getattr(self, f"q4{suffix}", None)
    q6 = getattr(self, f"q6{suffix}", None)
    if q4 is None or q6 is None:
        if Xnn is None:
            raise ValueError("q4/q6 not available. Run steinhardt_q(l=4) and "
                             "steinhardt_q(l=6) first, or pass Xnn to compute "
                             "them here.")
        q4 = steinhardt_q(self, Xnn, l=4, noOutput=True,
                          store=True, is_optimized=is_optimized)
        q6 = steinhardt_q(self, Xnn, l=6, noOutput=True,
                          store=True, is_optimized=is_optimized)
    q4 = np.asarray(q4); q6 = np.asarray(q6)

    # --- ideal reference points (literature, perfect first shell) ---------
    refs = {
        'FCC': (0.191, 0.575, '#d62728'),
        'BCC': (0.036, 0.511, '#1f77b4'),
        'HCP': (0.097, 0.485, '#2ca02c'),
        'ICO': (0.000, 0.663, '#9467bd'),
    }
    if sc_domain:
        refs['SC'] = (0.764, 0.354, '#ff7f0e')
        
    fig, ax = plt.subplots(figsize=(9, 7))

    # --- NP data ----------------------------------------------------------
    if aggregate:
        # group atoms by identical rounded (q4, q6) value
        pairs = np.round(np.column_stack([q4, q6]), decimals)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        # marker area ~ count (size ~ sqrt(count) so area is proportional)
        smin, smax = 30.0, 1200.0
        sizes = smin + (smax - smin) * np.sqrt(counts / counts.max())
        sc = ax.scatter(uniq[:, 0], uniq[:, 1], s=sizes, c=counts,
                        cmap='Blues', alpha=0.65, edgecolors='black',
                        linewidth=0.6, zorder=2)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('atoms per $(q_4, q_6)$ value', size=12, weight='bold')
        for tk in cbar.ax.get_yticklabels():
            tk.set_fontweight('bold')
    else:
        ax.scatter(q4, q6, c='grey', s=18, alpha=0.5, zorder=2,
                   edgecolors='none')

    # --- ideal references as hollow rings (do not hide atom points) -------
    for name, (x, y, col) in refs.items():
        ax.scatter(x, y, s=420, facecolors='none', edgecolors=col,
                   linewidth=2.8, marker='o', zorder=4)
        ax.annotate(name, (x, y), fontsize=12, fontweight='bold',
                    xytext=(11, 7), textcoords='offset points', color=col)

    # --- axis range: tight on the metallic region unless SC is shown ------
    if not sc_domain:
        # frame on FCC/BCC/HCP/ICO + the NP cloud, ignore the far SC corner
        x_hi = max(0.25, float(q4.max()) * 1.1)
        ax.set_xlim(-0.02, x_hi)
        
    ax.set_xlabel('$q_4$', fontsize=15, fontweight='bold')
    ax.set_ylabel('$q_6$', fontsize=15, fontweight='bold')
    ax.set_title(f'Steinhardt $(q_4, q_6)$ map ({status})\n'
                 f'○ ideal references · marker area ∝ atom count',
                 fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    for tk in ax.get_xticklabels() + ax.get_yticklabels():
        tk.set_fontweight('bold')

    plt.tight_layout()
    if save_path:
        if save_path.lower().endswith('.svg'):
            matplotlib.rcParams['svg.fonttype'] = 'none'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not noOutput:
            print(f"✅ Plot saved to: {save_path}")
    plt.show()
    return q4, q6

def compare_q4q6_map(objects, labels=None, Xnn=None, nrows=1, ncols=None,
                     decimals=3, same_count=False, sc_domain=False,
                     is_optimized=None, save_path=None, noOutput=False):
    """
    Small-multiples comparison of the Steinhardt (q4, q6) maps of several
    nanoparticles: one mini-panel per object, all sharing identical axes and
    the same ideal reference rings (FCC, BCC, HCP, ICO, optionally SC), so the
    local-order signatures of different morphologies can be compared side by
    side rather than overplotted on a single crowded chart.

    Each panel uses the aggregated representation of plot_q4q6_map: atoms
    sharing the same (q4, q6) value are drawn as a single marker whose area
    encodes the atom count.

    Reference values: ten Wolde, Ruiz-Montero & Frenkel, J. Chem. Phys. 104,
    9932 (1996).

    Args:
        objects (list): pyNMBcore instances to compare. Each must already hold
            q4/q6 (run steinhardt_q with l=4 and l=6), or Xnn must be given to
            compute them on the fly.
        labels (list of str): Panel titles, one per object. Defaults to the
            objects' `shape` attribute (or "NP {i}").
        Xnn (float): Neighbour cutoff in Å, used only where q4/q6 must be
            computed. Ignored for objects that already store them.
        nrows, ncols (int): Grid layout. If ncols is None it is derived from
            nrows and the number of objects (ceil division).
        decimals (int): Rounding used to group identical (q4, q6) values.
            Default 3.
        same_count (bool): If True, marker sizes use a common scale across all
            panels (absolute counts — a larger NP shows larger markers). If
            False (default), each panel is normalised to its own largest
            population, comparing the *patterns* regardless of NP size.
        sc_domain (bool): If True, include the SC reference and widen the
            x-range; if False (default), omit SC and tighten on the metallic
            region.
        is_optimized (bool or None): Target structure for every object.
            None -> each object's own is_optimized.
        save_path (str): If given, saves the figure (.png or .svg).
        noOutput (bool): If True, suppresses the saved-file message.

    Returns:
        matplotlib.figure.Figure: the small-multiples figure.
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from math import ceil

    n = len(objects)
    if labels is None:
        labels = [getattr(o, 'shape', None) or f"NP {i}"
                  for i, o in enumerate(objects)]
    if ncols is None:
        ncols = ceil(n / nrows)

    # --- references ------------------------------------------------------
    refs = {
        'FCC': (0.191, 0.575, '#d62728'),
        'BCC': (0.036, 0.511, '#1f77b4'),
        'HCP': (0.097, 0.485, '#2ca02c'),
        'ICO': (0.000, 0.663, '#9467bd'),
    }
    if sc_domain:
        refs['SC'] = (0.764, 0.354, '#ff7f0e')

    # --- collect aggregated data for every object ------------------------
    data = []          # list of (uniq_pairs, counts)
    for o in objects:
        opt = getattr(o, 'is_optimized', False) if is_optimized is None \
              else is_optimized
        use_opt = opt and getattr(o, 'NP_opt', None) is not None
        suffix = "_opt" if use_opt else ""
        q4 = getattr(o, f"q4{suffix}", None)
        q6 = getattr(o, f"q6{suffix}", None)
        if q4 is None or q6 is None:
            if Xnn is None:
                raise ValueError(f"q4/q6 missing for '{getattr(o,'shape','NP')}'"
                                 f" — run steinhardt_q(l=4/6) or pass Xnn.")
            q4 = steinhardt_q(o, Xnn, l=4, noOutput=True, store=True,
                              is_optimized=opt)
            q6 = steinhardt_q(o, Xnn, l=6, noOutput=True, store=True,
                              is_optimized=opt)
        pairs = np.round(np.column_stack([np.asarray(q4),
                                          np.asarray(q6)]), decimals)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        data.append((uniq, counts))

    # --- common axis limits ----------------------------------------------
    all_q4 = np.concatenate([d[0][:, 0] for d in data] +
                            [np.array([v[0] for v in refs.values()])])
    all_q6 = np.concatenate([d[0][:, 1] for d in data] +
                            [np.array([v[1] for v in refs.values()])])
    x_lo, x_hi = all_q4.min() - 0.02, all_q4.max() + 0.04
    y_lo, y_hi = all_q6.min() - 0.02, all_q6.max() + 0.04

    # --- marker-size scaling ---------------------------------------------
    smin, smax = 20.0, 900.0
    global_cmax = max(d[1].max() for d in data)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.8 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for k, (ax, (uniq, counts), label) in enumerate(
            zip(axes_flat, data, labels)):
        cmax = global_cmax if same_count else counts.max()
        sizes = smin + (smax - smin) * np.sqrt(counts / cmax)
        sc = ax.scatter(uniq[:, 0], uniq[:, 1], s=sizes, c=counts,
                        cmap='Blues', alpha=0.65, edgecolors='black',
                        linewidth=0.5, zorder=2)
        for name, (x, y, col) in refs.items():
            ax.scatter(x, y, s=240, facecolors='none', edgecolors=col,
                       linewidth=2.0, marker='o', zorder=4)
            ax.annotate(name, (x, y), fontsize=9, fontweight='bold',
                        xytext=(7, 4), textcoords='offset points', color=col)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        if k % ncols == 0:
            ax.set_ylabel('$q_6$', fontsize=12, fontweight='bold')
        if k // ncols == nrows - 1:
            ax.set_xlabel('$q_4$', fontsize=12, fontweight='bold')

        # per-panel colorbar (count scale)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('atoms', size=9, weight='bold')
        cbar.ax.tick_params(labelsize=8)

    # hide any unused panels
    for ax in axes_flat[n:]:
        ax.axis('off')

    scale_note = ("common count scale" if same_count
                  else "per-panel count scale")
    fig.suptitle(f'Steinhardt $(q_4, q_6)$ comparison — {scale_note}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        if save_path.lower().endswith('.svg'):
            matplotlib.rcParams['svg.fonttype'] = 'none'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not noOutput:
            print(f"✅ Plot saved to: {save_path}")
    plt.show()