import time, datetime
import numpy as np

from .core import (pyNMB_location, get_resource_path, timer, RAB, Rbetween2Points,
                   vector, vectorBetween2Points, coord2xyz, vertex, vertexScaled, RadiusSphereAfterV,
                   centerOfGravity, center2cog, normOfV, normV, centerToVertices, Rx, Ry, Rz,
                   EulerRotationMatrix, plotPalette, rgb2hex, clone, deleteElementsOfAList,
                   planeFittingLSF, faces_to_planes, AngleBetweenVV, signedAngleBetweenVV
                   )
from .core import centertxt, centerTitle, fg, bg, hl, color

from numba import njit
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# Numba core
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _falk_langer_core(ref_disp, def_disp, start, count):
    """
    Compute per-atom Falk-Langer affine strain descriptors.

    For each atom i, the best-fit affine deformation gradient F_i is the tensor
    that maps the reference neighbor vectors r0 onto the deformed neighbor vectors r
    in a least-squares sense (Falk and Langer, Phys. Rev. E 57, 7192 (1998)):

        X = sum_k  r_k  (r0_k)^T
        Y = sum_k  r0_k (r0_k)^T
        F = X Y^-1

    The Green-Lagrange strain tensor is eta = 0.5 (F^T F - I). From it the
    volumetric (hydrostatic) and von Mises (deviatoric) invariants are extracted.
    The non-affine residual D2min is returned as a diagnostic of how well the
    affine approximation holds locally.

    Convention for the von Mises strain returned here:

        dev   = eta - (trace(eta) / 3) I
        vm    = sqrt(2 * (dev : dev))      where dev : dev = sum_ab dev_ab^2

    This is one of several normalizations used in the literature (LAMMPS and OVITO
    define their "deviatoric strain" as sqrt(dev : dev / 2), which differs from the
    value here by a constant factor of 2). The volumetric strain trace(eta) and the
    residual D2min follow the standard LAMMPS/OVITO definitions and are directly
    comparable. If exact numerical agreement with OVITO on the deviatoric component
    is ever required, rescale vm accordingly.

    Note on convention: neighbor vectors are treated as column vectors, so that
    F = X Y^-1 matches the OVITO/LAMMPS definition. This avoids a spurious
    transpose when comparing against those tools.

    Args:
        ref_disp (float64[:, 3]): Flattened reference neighbor vectors r0 (origin
            at the central atom), stacked for all atoms.
        def_disp (float64[:, 3]): Flattened deformed neighbor vectors r, in the
            same order as ref_disp.
        start (int64[:]): Start index into ref_disp/def_disp for each atom.
        count (int32[:]): Number of neighbors for each atom.

    Returns:
        vol (float64[:]): Volumetric strain per atom, trace(eta).
        vm (float64[:]): Von Mises (deviatoric) strain per atom.
        d2min (float64[:]): Non-affine squared displacement residual per atom.
        detF (float64[:]): det(F) per atom (local relative volume change is detF - 1).
    """
    n = start.shape[0]
    vol = np.zeros(n, dtype=np.float64)
    vm = np.zeros(n, dtype=np.float64)
    d2min = np.zeros(n, dtype=np.float64)
    detF = np.full(n, np.nan, dtype=np.float64)

    eye = np.eye(3)

    for i in range(n):
        c = count[i]
        # An affine fit in 3D needs at least 3 non-coplanar neighbor vectors.
        if c < 3:
            vol[i] = np.nan
            vm[i] = np.nan
            d2min[i] = np.nan
            continue

        s = start[i]
        X = np.zeros((3, 3))
        Y = np.zeros((3, 3))
        for k in range(c):
            r = def_disp[s + k]
            r0 = ref_disp[s + k]
            for a in range(3):
                for b in range(3):
                    X[a, b] += r[a] * r0[b]
                    Y[a, b] += r0[a] * r0[b]

        # Y can be singular for degenerate (collinear/coplanar) neighborhoods.
        detY = np.linalg.det(Y)
        if abs(detY) < 1e-12:
            vol[i] = np.nan
            vm[i] = np.nan
            d2min[i] = np.nan
            continue

        Yinv = np.linalg.inv(Y)
        F = X @ Yinv
        detF[i] = np.linalg.det(F)

        # Non-affine residual D2min = sum_k |r - F r0|^2 / N
        acc = 0.0
        for k in range(c):
            r = def_disp[s + k]
            r0 = ref_disp[s + k]
            pred = F @ r0
            diff = r - pred
            acc += diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
        d2min[i] = acc / c

        # Green-Lagrange strain tensor eta = 0.5 (F^T F - I)
        eta = 0.5 * (F.T @ F - eye)

        tr = eta[0, 0] + eta[1, 1] + eta[2, 2]
        vol[i] = tr

        # Deviatoric part and von Mises equivalent strain
        dev = eta - (tr / 3.0) * eye
        j2 = 0.0
        for a in range(3):
            for b in range(3):
                j2 += dev[a, b] * dev[a, b]
        vm[i] = np.sqrt(0.5 * j2) * np.sqrt(2.0)  # sqrt(2 * dev:dev / ... ) form

    return vol, vm, d2min, detF

def calculate_atomic_strain(self, cutoff=None, neighbor_source='reference', noOutput: bool=False):
    """
    Compute per-atom affine strain descriptors using the Falk-Langer method.

    The deformed configuration self.NP_opt is compared against the undeformed
    reference self.NP. For each atom i, a best-fit affine deformation gradient F_i
    is determined from the displacements of its neighbors, and the local
    Green-Lagrange strain tensor eta_i = 0.5 (F_i^T F_i - I) is built. This is the
    same quantity OVITO visualizes in its Atomic Strain modifier and that is used,
    for example, in Rahm and Erhart, Nano Lett. 17, 5775 (2017) to map the
    volumetric strain of relaxed metal nanoparticles.

    Physical reading: the volumetric strain captures surface-tension-driven core
    compression as well as the intrinsic geometric strain of twinned motifs
    (icosahedra and decahedra cannot tile space with regular tetrahedra and are
    therefore strained even before relaxation). The von Mises component highlights
    local shear, which concentrates on twin planes.

    Neighbor pairing is done by atom index: both configurations must share the same
    atom ordering, which is guaranteed because geometry relaxation moves atoms but
    never renumbers them. Neighbors are determined on the reference configuration
    (neighbor_source='reference') so that the neighbor set is unambiguous and
    identical in both states.

    Args:
        cutoff (float, optional): Neighbor cutoff radius in Angstrom. If None, it is
            set to a value lying between the first and second neighbor shells, based
            on the reference first-neighbor distance (1.2 * d_nn for fcc-like
            packings). Default is None.
        neighbor_source (str, optional): Configuration on which neighbor lists are
            built, either 'reference' (self.NP, recommended) or 'deformed'
            (self.NP_opt). Default is 'reference'.
        noOutput (bool): If True, suppresses output. Default is True.

    Creates:
        self.strain_vol (numpy.ndarray): Volumetric strain trace(eta) per atom.
        self.strain_vm (numpy.ndarray): Von Mises (deviatoric) strain per atom.
        self.strain_d2min (numpy.ndarray): Non-affine residual D2min per atom.
        self.strain_detF (numpy.ndarray): det(F) per atom (detF - 1 is the local
            relative volume change).

    Returns:
        numpy.ndarray: self.strain_vol, the volumetric strain per atom.

    Raises:
        AttributeError: If self.NP or self.NP_opt is not available (the structure
            has not been optimized yet).
        ValueError: If the two configurations do not have the same number of atoms,
            or if neighbor_source is not 'reference' or 'deformed'.
    """
    if not noOutput:
        chrono = timer()
        chrono.chrono_start()
        centertxt("Atomic strain (Falk-Langer)", bgc='#007a7a', size='14', weight='bold')
        
    if not hasattr(self, 'NP') or self.NP is None:
        raise AttributeError("self.NP (reference configuration) is not available.")
    if not hasattr(self, 'NP_opt') or self.NP_opt is None:
        raise AttributeError(
            "self.NP_opt (deformed configuration) is not available. "
            "Run optimize() before computing atomic strain."
        )

    pos_ref = self.NP.get_positions()
    pos_def = self.NP_opt.get_positions()

    if pos_ref.shape[0] != pos_def.shape[0]:
        raise ValueError(
            "Reference and deformed configurations have different atom counts "
            f"({pos_ref.shape[0]} vs {pos_def.shape[0]})."
        )

    n = pos_ref.shape[0]

    # Choose the cutoff from the reference first-neighbor distance if not given.
    if cutoff is None:
        tree_tmp = cKDTree(pos_ref)
        # nearest non-self distance, queried on a small sample for speed
        sample = min(n, 64)
        dists, _ = tree_tmp.query(pos_ref[:sample], k=2)
        d_nn = np.median(dists[:, 1])
        cutoff = 1.2 * d_nn

    if neighbor_source == 'reference':
        tree = cKDTree(pos_ref)
        query_pos = pos_ref
    elif neighbor_source == 'deformed':
        tree = cKDTree(pos_def)
        query_pos = pos_def
    else:
        raise ValueError("neighbor_source must be 'reference' or 'deformed'.")

    # Build a flat (CSR-like) neighbor structure for Numba.
    neigh_lists = tree.query_ball_point(query_pos, r=cutoff)

    count = np.empty(n, dtype=np.int32)
    for i in range(n):
        # exclude self
        count[i] = len(neigh_lists[i]) - 1

    start = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        start[i] = start[i - 1] + count[i - 1]
    total = int(start[-1] + count[-1]) if n > 0 else 0

    ref_disp = np.empty((total, 3), dtype=np.float64)
    def_disp = np.empty((total, 3), dtype=np.float64)

    for i in range(n):
        s = start[i]
        k = 0
        for j in neigh_lists[i]:
            if j == i:
                continue
            ref_disp[s + k] = pos_ref[j] - pos_ref[i]
            def_disp[s + k] = pos_def[j] - pos_def[i]
            k += 1
    # --- PROBE: verify which array holds the larger (reference) bonds ---
    # i0 = int(np.argmax(count))          # best-coordinated atom = core
    # s0 = int(start[i0])
    # nref = np.linalg.norm(ref_disp[s0])
    # ndef = np.linalg.norm(def_disp[s0])
    # print(f"[probe] core atom {i0}: |ref_disp|={nref:.4f}  |def_disp|={ndef:.4f}")
    # print(f"[probe] expect ref > def (2.90 vs 2.86) if arrays are correct")
    # print(f"[probe] id(pos_ref)={id(pos_ref)}  id(pos_def)={id(pos_def)}")
    # print(f"[probe] ref mean bond={np.linalg.norm(ref_disp,axis=1).mean():.4f}  "
    #       f"def mean bond={np.linalg.norm(def_disp,axis=1).mean():.4f}")

    vol, vm, d2min, detF = _falk_langer_core(ref_disp, def_disp, start, count)

    # sonde temporaire : recalcul avec ref et def échangés
    # vol2, vm2, d2min2, detF2 = _falk_langer_core(def_disp, ref_disp, start, count)
    # print("detF médian normal :", np.median(detF))
    # print("detF médian échangé:", np.median(detF2))

    self.strain_vol = vol
    self.strain_vm = vm
    self.strain_d2min = d2min
    self.strain_detF = detF
    self.strain_cutoff = cutoff

    if not noOutput:
        n_nan = int(np.isnan(vol).sum())
        print(f" - Number of atoms            : {n}")
        print(f" - Neighbor cutoff            : {cutoff:.3f} Å")
        print(f" - Atoms not evaluated (NaN)  : {n_nan}")
        print(
            f" - Volumetric strain trace(η) : mean {np.nanmean(vol):+.4f}, "
            f"min {np.nanmin(vol):+.4f}, max {np.nanmax(vol):+.4f}"
        )
        print(
            f" - Von Mises strain           : mean {np.nanmean(vm):.4f}, "
            f"max {np.nanmax(vm):.4f}"
        )
        print(
            f" - Non-affine residual D2min  : mean {np.nanmean(d2min):.2e}, "
            f"max {np.nanmax(d2min):.2e}"
        )
        chrono.chrono_stop(hdelay=False)
        chrono.chrono_show()

    return self.strain_vol


