"""
Simple pyNanoMatBuilder <-> abTEM coupling.

This module provides a lightweight, single-structure bridge between
pyNanoMatBuilder and abTEM, meant for interactive/tutorial use (see the
``pyNMB-abtem-coupling.ipynb`` notebook):

1. Build ONE nanoparticle with the usual pyNanoMatBuilder API (any shape
   class), keeping in mind that the NP must be smaller than the amorphous
   carbon substrate (5x5 or 10x10 nm laterally).
2. Pass the NP (the pyNanoMatBuilder instance, or directly its ``.NP``
   ``ase.Atoms`` object) to :class:`CreateHRTEMStructure`, which places it on
   a flat area of the relaxed amorphous-carbon substrate shipped with the
   package and (optionally) writes the full NP+substrate XYZ file.
3. Pass the resulting structure to :class:`CreateHRTEMImage`, which runs the
   abTEM multislice pipeline (frozen phonons, CTF at Scherzer defocus,
   partial coherence, Poisson shot noise, detector MTF) and produces a single
   HRTEM image.

Unlike ``TEM_creator.py`` (automatic database generation over many
compounds/shapes/sizes/orientations), 
This module handles exactly one nanoparticle and one pose per call,
with no file-numbering or metadata-CSV machinery. For a full database,
a new module ``TEM_creator.py`` will be uploaded later. 
Don't hesitate to rerun with different arguments to explore other poses.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ase.atoms import Atoms
from ase.io import read, write
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as sciR
from sklearn.decomposition import PCA
from PIL import Image as PILImage

import abtem

from pyNanoMatBuilder import utils as pyNMBu


def _load_substrate(substrate_size: int) -> Atoms:
    """
    Load the relaxed amorphous-carbon substrate shipped with the package.

    Args:
        substrate_size (int): Lateral size of the substrate in nm (5 or 10).
            The corresponding file (aC_relax_5x5.xyz.gz or aC_relax_10x10.xyz.gz)
            is resolved from the pyNanoMatBuilder package resources.

    Returns:
        Atoms: ASE Atoms object of the carbon substrate (no cell attached).
    """
    if substrate_size not in (5, 10):
        raise ValueError(f"substrate_size must be 5 or 10 (nm), got {substrate_size}.")
    substrate_file = f"aC_relax_{substrate_size}x{substrate_size}.xyz.gz"
    gz_path = Path(pyNMBu.get_resource_path('resources/amorphousC', substrate_file))
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].split()[0])
    symbols, positions = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    return Atoms(symbols=symbols, positions=np.asarray(positions))


def _surface_atom_indices(positions, grid_size: float = 2., z_tolerance: float = 6.):
    """
    Identify the topmost (surface) atoms of the substrate with a grid method.

    The xy-plane is discretized in cells of ``grid_size`` Angstroms; in each
    cell the atom with the highest z is a surface candidate, kept only if it
    lies within ``z_tolerance`` Angstroms of the global maximum z (this
    prevents deep atoms under vertical gaps from being tagged as surface).

    Args:
        positions (ndarray): (N, 3) substrate atomic positions in Angstroms.
        grid_size (float): xy-grid cell size in Angstroms.
        z_tolerance (float): Maximum distance below the global z maximum.

    Returns:
        ndarray: Sorted unique indices of the surface atoms.
    """
    xy = positions[:, :2]
    z = positions[:, 2]
    z_global_max = z.max()
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    surface_indices = []
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            in_cell = np.where((xy[:, 0] >= x_bins[i]) & (xy[:, 0] < x_bins[i + 1]) &
                               (xy[:, 1] >= y_bins[j]) & (xy[:, 1] < y_bins[j + 1]))[0]
            if len(in_cell) > 0:
                top = in_cell[np.argmax(z[in_cell])]
                if z[top] >= z_global_max - z_tolerance:
                    surface_indices.append(top)
    return np.unique(surface_indices)


def _flat_clusters(surface_xyz, window: float = 20., min_span: float = 20.,
                   max_z_variation: float = 5., anisotropy_max: float = 1.1):
    """
    Find flat areas of the substrate surface with a sliding xy window.

    A window of surface atoms is accepted as a flat area if it covers a
    minimum lateral extent (``min_span``), is flat enough in z
    (``max_z_variation``) and reasonably isotropic in xy (``anisotropy_max``).

    Args:
        surface_xyz (ndarray): (N, 3) positions of the surface atoms (Angstroms).
        window (float): Lateral size of the sliding window in Angstroms.
        min_span (float): Minimum xy extent of an accepted cluster (Angstroms).
        max_z_variation (float): Maximum z range within a cluster (Angstroms).
        anisotropy_max (float): Maximum std(x)/std(y) (or inverse) ratio.

    Returns:
        list[ndarray]: Coordinates of the atoms of each accepted flat area.
    """
    x_min, y_min = surface_xyz[:, :2].min(axis=0)
    x_max, y_max = surface_xyz[:, :2].max(axis=0)
    accepted = []
    for x0 in np.arange(x_min, x_max, 5.0):
        for y0 in np.arange(y_min, y_max, 10.0):
            mask = ((surface_xyz[:, 0] >= x0) & (surface_xyz[:, 0] < x0 + window) &
                    (surface_xyz[:, 1] >= y0) & (surface_xyz[:, 1] < y0 + window))
            patch = surface_xyz[mask]
            if len(patch) < 2:
                continue
            xy_patch = patch[:, :2]
            if np.max(pdist(xy_patch)) < min_span:
                continue
            if patch[:, 2].max() - patch[:, 2].min() > max_z_variation:
                continue
            std_x, std_y = np.std(xy_patch[:, 0]), np.std(xy_patch[:, 1])
            if max(std_x, std_y) / max(1e-6, min(std_x, std_y)) > anisotropy_max:
                continue
            accepted.append(patch)
    return accepted


def detector_mtf(shape, sampling, c1=-0.6, c2=0.1, c3=1.0, clip=False):
    """
    Build a detector Modulation Transfer Function (MTF) on the FFT grid.

    The MTF describes how faithfully the camera records each spatial frequency:
    a real detector transfers the coarse (low-frequency) content almost
    perfectly and progressively attenuates the fine (high-frequency) details.
    This is modelled by a monotonic rolloff from ``mtf(0) = 1`` (the mean
    intensity, i.e. the image DC, is always preserved) toward the
    high-frequency asymptote ``c1``:

        mtf(q) = (1 - c1) / (1 + (q / (2 * c2 * q_N))**c3) + c1

    with ``q_N = 1 / (2 * sampling)`` the Nyquist frequency of the grid, ``c2``
    the half-scale of the rolloff (as a fraction of Nyquist) and ``c3`` its
    steepness.

    The array is returned in the native (unshifted) ``np.fft.fftfreq`` layout,
    i.e. aligned with ``np.fft.fft2``: it must multiply the FFT of the image
    **directly, without any fftshift**. Applying ``fftshift`` here would move
    the DC response to the array centre while the image FFT keeps its DC in the
    corner, which swaps the low- and high-frequency responses (the historical
    bug this function replaces).

    Args:
        shape (tuple): (ny, nx) image shape.
        sampling (float): Real-space sampling in Angstrom per pixel.
        c1 (float): High-frequency asymptote (detector contrast floor).
        c2 (float): Rolloff half-scale as a fraction of the Nyquist frequency.
        c3 (float): Rolloff exponent (steepness).
        clip (bool): If True, clip the MTF to [0, 1]. A physical MTF is never
            negative, but ``c1 < 0`` makes the formula dip below zero at high
            frequency; clipping enforces strict physicality. Left False by
            default to preserve the exact parametrization of earlier databases.

    Returns:
        ndarray: (ny, nx) MTF array, in fftfreq layout (DC at [0, 0]).
    """
    ny, nx = shape
    q_N = 1.0 / (2.0 * sampling)
    qx = np.fft.fftfreq(nx, d=sampling)
    qy = np.fft.fftfreq(ny, d=sampling)
    qx, qy = np.meshgrid(qx, qy)
    q = np.sqrt(qx**2 + qy**2)
    mtf = (1 - c1) / (1 + (q / (2 * c2 * q_N))**c3) + c1
    if clip:
        mtf = np.clip(mtf, 0.0, 1.0)
    return mtf


def apply_detector_mtf(image, sampling, c1=-0.6, c2=0.1, c3=1.0, clip=False):
    """
    Apply the detector MTF to a real-space image (Fourier-space multiplication).

    The image FFT is multiplied by :func:`detector_mtf` on the aligned fftfreq
    grid and transformed back. Because ``mtf(0) = 1``, the mean intensity of the
    image is preserved. See :func:`detector_mtf` for the parameters.

    Args:
        image (ndarray): (ny, nx) real-space image.
        sampling (float): Real-space sampling in Angstrom per pixel.
        c1, c2, c3 (float): MTF parameters (see :func:`detector_mtf`).
        clip (bool): Clip the MTF to [0, 1] (see :func:`detector_mtf`).

    Returns:
        ndarray: (ny, nx) filtered real-space image.
    """
    mtf = detector_mtf(image.shape, sampling, c1, c2, c3, clip)
    return np.real(np.fft.ifft2(np.fft.fft2(image) * mtf))


class CreateHRTEMStructure:
    """
    Place ONE pyNanoMatBuilder nanoparticle on the amorphous-carbon substrate.

    The NP is built beforehand by the user with the usual pyNanoMatBuilder API
    (its size must be smaller than the substrate); this class only handles the
    placement:

    1. The resting facet is the convex-hull facet closest to the NP center of
       gravity (the most stable orientation: lowest center of gravity).
    2. A flat area of the carbon surface is detected (grid + sliding-window
       method) and the one closest to the substrate center is selected.
    3. The NP is rotated so that its resting facet lies on the carbon flat
       area, shifted laterally above it (and pushed back towards the
       substrate center when its actual xy extent would stick out of the
       borders), and lowered to ``tolerance`` Angstroms above the carbon
       surface. An error is raised only when the NP is genuinely wider than
       the substrate.
    4. Optional in-plane rotation (``angle_xy``) and small tilt (``tilt``)
       mimic the variety of experimental poses.

    Parameters
    ----------
    NP : pyNMBcore instance or ase.Atoms
        The nanoparticle: either a pyNanoMatBuilder shape instance (e.g. the
        result of ``platonicNPs.regIco(...)``, its ``trPlanes`` facets are
        then reused) or directly an ``ase.Atoms`` object (e.g. ``AuNP.NP``,
        the facets are then recomputed from the convex hull).
    substrate_size : int, default 10
        Lateral size of the amorphous-carbon substrate in nm (5 or 10).
    tolerance : float, default 3.
        Distance in Angstroms between the NP and the carbon surface. Too
        small a value creates chemical bonds at the interface.
    angle_xy : float or None, default None
        In-plane rotation of the NP (degrees) around the substrate normal.
        None draws a random angle (reproducible through ``seed``).
    tilt : float, default 0.
        Tilt (degrees) applied around a random horizontal axis, to mimic
        substrate roughness / imperfect facet contact. The vertical clearance
        is recomputed after the tilt. Typical experimental-like values: 5-15.
    output_xyz : str or None, default None
        If given, path of the XYZ file to write (NP + substrate).
    seed : int or None, default None
        Seed of the random generator (random angle_xy and tilt axis).
    noOutput : bool, default True
        If False, print a summary of the placement.

    Attributes
    ----------
    structure : ase.Atoms
        The full structure (carbon substrate + placed NP).
    NP : ase.Atoms
        The placed NP alone (rotated/translated copy of the input).
    substrate : ase.Atoms
        The carbon substrate alone.
    circumsphere_diameter : float
        Diameter of the NP circumscribed sphere in nm.
    angle_xy, tilt, tolerance, substrate_size
        The placement parameters actually used.

    Examples
    --------
    >>> AuNP = pNP.regIco(element='Au', Rnn=2.885, nShell=5, postAnalyzis=True,
    ...                   skipSymmetryAnalyzis=True, aseView=False, noOutput=True)
    >>> struct = CreateHRTEMStructure(AuNP, substrate_size=10, tolerance=3.)
    >>> struct.structure   # ASE Atoms, ready for CreateHRTEMImage
    """

    def __init__(self, NP, substrate_size: int = 10, tolerance: float = 3.,
                 angle_xy: float = None, tilt: float = 0.,
                 output_xyz: str = None, seed: int = None, noOutput: bool = True):
        rng = np.random.default_rng(seed)

        # 1. Resolve the NP input: pyNanoMatBuilder instance or bare ASE Atoms
        if isinstance(NP, Atoms):
            np_atoms = NP.copy()
            planes = None
        elif hasattr(NP, "NP") and isinstance(NP.NP, Atoms):
            np_atoms = NP.NP.copy()
            planes = getattr(NP, "trPlanes", None)
        else:
            raise TypeError("NP must be a pyNanoMatBuilder shape instance or an ase.Atoms "
                            f"object, got {type(NP).__name__}.")
        positions = np_atoms.positions
        if planes is None:
            # Facets recomputed from the convex hull (outward normals, n.x + d <= 0 inside)
            planes = ConvexHull(positions).equations
        planes = np.asarray(planes, dtype=float)

        # 2. NP size (informative; the actual fit test uses the placed NP
        # xy extent, which is smaller than the circumscribed sphere)
        cog = positions.mean(axis=0)
        self.circumsphere_diameter = 2 * np.linalg.norm(positions - cog, axis=1).max() * 0.1  # nm

        # 3. Load the substrate and pick the flat area closest to its center
        substrate = _load_substrate(substrate_size)
        sub_xy = substrate.positions[:, :2]
        x_min, y_min = sub_xy.min(axis=0)
        x_max, y_max = sub_xy.max(axis=0)
        surface_idx = _surface_atom_indices(substrate.positions)
        clusters = _flat_clusters(substrate.positions[surface_idx])
        sub_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        cluster = min(clusters, key=lambda c: np.linalg.norm(c.mean(axis=0)[:2] - sub_center))

        # Plane best fitting the flat carbon area (PCA), normal pointing up
        pca = PCA(n_components=3)
        pca.fit(cluster)
        normal_carbon = pca.components_[-1]
        if normal_carbon[2] < 0:
            normal_carbon = -normal_carbon
        normal_carbon /= np.linalg.norm(normal_carbon)
        center_carbon = cluster.mean(axis=0)

        # 4. Resting facet: the facet closest to the center of gravity
        normals = planes[:, :3]
        heights = np.abs(normals @ cog + planes[:, 3]) / np.linalg.norm(normals, axis=1)
        k = int(np.argmin(heights))
        facet_normal = normals[k].copy()
        if facet_normal @ cog + planes[k, 3] > 0:
            facet_normal = -facet_normal  # force the outward orientation
        # The outward facet normal must point INTO the substrate for the NP
        # to rest on that facet
        pos = pyNMBu.rotateMoltoAlignItWithAxis(positions, axis=facet_normal,
                                                targetAxis=-normal_carbon)

        # 5. In-plane rotation around the substrate normal, applied BEFORE the
        # lateral placement so that the final xy extent is the one fitted on
        # the substrate
        if angle_xy is None:
            angle_xy = float(rng.uniform(0., 360.))
        pos = pyNMBu.rotation_around_axis_through_point(pos, angle_deg=angle_xy,
                                                        axis=normal_carbon,
                                                        center=pos.mean(axis=0))

        # 6. Lateral centering above the flat area
        center_np = pos.mean(axis=0)
        dist_to_plane = (center_np - center_carbon) @ normal_carbon
        pos = pos + (center_carbon - (center_np - dist_to_plane * normal_carbon))

        # 7. If the NP actually sticks out of the substrate, push it back
        # towards the center by the minimal shift; refuse only when the NP
        # is genuinely wider than the substrate minus the border margin
        margin = 5.0  # A, safety margin to the substrate borders
        shift = np.zeros(3)
        for k, (lo, hi) in enumerate(((x_min, x_max), (y_min, y_max))):
            extent = pos[:, k].max() - pos[:, k].min()
            if extent > (hi - lo) - 2 * margin:
                raise ValueError(
                    f"The NP lateral extent ({extent * 0.1:.2f} nm along {'xy'[k]}) exceeds the "
                    f"{substrate_size}x{substrate_size} nm substrate minus the {margin} A border "
                    f"margin: build a smaller NP or use substrate_size=10.")
            if pos[:, k].min() < lo + margin:
                shift[k] = lo + margin - pos[:, k].min()
            elif pos[:, k].max() > hi - margin:
                shift[k] = hi - margin - pos[:, k].max()
        pos = pos + shift
        if not noOutput and np.any(shift != 0.):
            print(f"NP shifted by ({shift[0]:+.1f}, {shift[1]:+.1f}) A towards the substrate "
                  f"center so that it fits within the borders.")

        # 8. Optional tilt around a random horizontal axis
        if tilt != 0.:
            ref = np.array([1., 0., 0.])
            if abs(ref @ normal_carbon) > 0.9:
                ref = np.array([0., 1., 0.])
            axis0 = np.cross(normal_carbon, ref)
            axis0 /= np.linalg.norm(axis0)
            azimuth = float(rng.uniform(0., 360.))
            tilt_axis = sciR.from_rotvec(np.radians(azimuth) * normal_carbon).apply(axis0)
            pos = (sciR.from_rotvec(np.radians(tilt) * tilt_axis)
                   .apply(pos - pos.mean(axis=0)) + pos.mean(axis=0))

        # 9. Vertical clearance, applied last (after every rotation/shift).
        # First relative to the fitted flat-area plane, then lifted until the
        # TRUE minimum NP-substrate distance reaches the tolerance: a NP wider
        # than the flat area can overhang rougher carbon that sticks out above
        # the fitted plane.
        proj = (pos - center_carbon) @ normal_carbon
        pos = pos + (-proj.min() + tolerance) * normal_carbon
        substrate_tree = cKDTree(substrate.positions)
        for _ in range(20):
            d_min = substrate_tree.query(pos)[0].min()
            if d_min >= tolerance - 1e-3:
                break
            pos = pos + (tolerance - d_min) * normal_carbon

        # 10. Assemble the final structure (original chemical symbols are kept,
        # so alloy or core/shell NPs are supported)
        np_atoms.set_positions(pos)
        self.NP = np_atoms
        self.substrate = substrate
        self.structure = substrate + np_atoms
        self.substrate_size = substrate_size
        self.tolerance = tolerance
        self.angle_xy = angle_xy
        self.tilt = tilt
        self.output_xyz = output_xyz

        if output_xyz is not None:
            write(output_xyz, self.structure,
                  comment="Carbon substrate + nanoparticle (pyNanoMatBuilder/abtem_coupling)")

        if not noOutput:
            print(f"NP: {len(np_atoms)} atoms, circumscribed diameter = "
                  f"{self.circumsphere_diameter:.2f} nm")
            print(f"Substrate: {substrate_size}x{substrate_size} nm amorphous carbon "
                  f"({len(substrate)} atoms)")
            print(f"Placement: angle_xy = {angle_xy:.1f} deg, tilt = {tilt:.1f} deg, "
                  f"clearance = {tolerance} A")
            if output_xyz is not None:
                print(f"XYZ file written: {output_xyz}")


class CreateHRTEMImage:
    """
    Simulate ONE HRTEM image of a NP+substrate structure with abTEM.

    The multislice pipeline is the same as in ``TEM_creator.CreateHRTEMImage``
    (frozen phonons with Debye-Waller displacements, plane-wave multislice,
    CTF at Scherzer defocus with astigmatism and temporal coherence, Poisson
    shot noise, detector MTF), but applied to a single structure passed in
    memory or as one XYZ file, without any metadata-CSV requirement.

    Aberration notation follows the Krivanek convention used by abTEM:

    ============  ============================================  ==========
    Symbol        Physical meaning                              Unit
    ============  ============================================  ==========
    Cs (C30)      3rd-order spherical aberration                Angstrom
    C10           Defocus (set to Scherzer by default)          Angstrom
    C12 / phi12   2-fold astigmatism amplitude / azimuth        Angstrom / rad
    focal_spread  Temporal-coherence envelope = Cc * dE / E     Angstrom
    ============  ============================================  ==========

    Parameters
    ----------
    structure : CreateHRTEMStructure, ase.Atoms or str/Path
        The structure to image: a :class:`CreateHRTEMStructure` result, an
        ``ase.Atoms`` object, or the path of an XYZ file.
    substrate_thickness : float, default 20.
        Thickness (Angstroms) of the top slab of the carbon substrate kept
        for the simulation; the rest is cropped to keep the multislice cheap.
    sampling : float, default 0.05
        Real-space sampling in Angstrom per grid point (impacts run time).
    energy : float, default 200e3
        Electron beam energy in eV (200 keV by default).
    phonon_config : int, default 8
        Number of frozen-phonon configurations.
    sigmas : float, default 0.1
        Standard deviation of the phonon displacements in Angstroms.
    slice_thickness : float, default 1.
        Multislice slice thickness in Angstroms.
    Cs_value : float, default -8e-6 * 1e10
        Spherical aberration C30 in Angstroms (negative: corrected microscope).
    C12, phi12 : float, default 0
        2-fold astigmatism amplitude (Angstrom) and azimuth (rad).
    Cc_value : float, default 1.0e-3 * 1e10
        Chromatic aberration coefficient in Angstroms (1 mm).
    semiangle_cutoff_value : int, default 45
        Objective aperture semiangle cutoff in mrad.
    energy_spread : float, default 0.35
        Source energy spread dE in eV; focal_spread = Cc * dE / E.
    c1, c2, c3 : float, defaults -0.6, 0.1, 1.0
        Detector MTF parameters (contrast floor, half-power frequency
        scaling, roll-off exponent). See :func:`detector_mtf`.
    mtf_clip : bool, default False
        Clip the MTF to [0, 1] for strict physicality (a real MTF is never
        negative). Left False to preserve the exact ``c1 < 0`` parametrization
        used in earlier databases. See :func:`detector_mtf`.
    dose_poisson_noise : float, default 1e4
        Electron dose for the Poisson shot noise in e-/Angstrom^2.
    vacuum : float, default 2.
        Vacuum added above/below the structure along z (Angstroms).
    device : str, default 'cpu'
        Computation device: 'cpu' or 'gpu' (requires cupy/CUDA).
    noOutput : bool, default True
        If False, print diagnostic information (defocus, focal spread...).

    Attributes
    ----------
    image : ndarray
        The final HRTEM image, normalized to [0, 1].
    atoms : ase.Atoms
        The cropped structure actually used in the simulation (with cell).
    defocus : float
        Scherzer defocus C10 in Angstroms.
    metadata : dict
        All the simulation parameters (for CSV export through ``save``).

    Examples
    --------
    >>> struct = CreateHRTEMStructure(AuNP, substrate_size=10)
    >>> img = CreateHRTEMImage(struct, device='cpu')
    >>> img.show()
    >>> img.save('Au_hrtem.png', metadata=True)
    """

    def __init__(self, structure, substrate_thickness: float = 20.,
                 sampling: float = 0.05, energy: float = 200e3,
                 phonon_config: int = 8, sigmas: float = 0.1,
                 slice_thickness: float = 1.,
                 Cs_value: float = -8e-6 * 1e10, C12: float = 0., phi12: float = 0.,
                 Cc_value: float = 1.0e-3 * 1e10,
                 semiangle_cutoff_value: int = 45, energy_spread: float = 0.35,
                 c1: float = -0.6, c2: float = 0.1, c3: float = 1.0,
                 mtf_clip: bool = False,
                 dose_poisson_noise: float = 1e4, vacuum: float = 2.,
                 device: str = 'cpu', noOutput: bool = True):
        try:
            abtem.config.set({"device": device, "fft": "fftw"})
        except Exception:
            abtem.config.set({"device": device})

        self.sampling = sampling
        self.energy = energy
        self.phonon_config = phonon_config
        self.sigmas = sigmas
        self.slice_thickness = slice_thickness
        self.Cs_value = Cs_value
        self.C12 = C12
        self.phi12 = phi12
        self.Cc_value = Cc_value
        self.semiangle_cutoff_value = semiangle_cutoff_value
        self.energy_spread = energy_spread
        self.focal_spread = Cc_value * energy_spread / energy
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.mtf_clip = mtf_clip
        self.dose_poisson_noise = dose_poisson_noise
        self.substrate_thickness = substrate_thickness
        self.vacuum = vacuum
        self.device = device
        self.noOutput = noOutput

        # Resolve the input structure
        if isinstance(structure, CreateHRTEMStructure):
            atoms = structure.structure.copy()
        elif isinstance(structure, Atoms):
            atoms = structure.copy()
        else:
            atoms = read(str(structure))

        self.atoms = self._prepare_cell(atoms)
        self._simulate()

    def _prepare_cell(self, atoms: Atoms) -> Atoms:
        """
        Crop the substrate to its top slab and attach a simulation cell.

        Only the top ``substrate_thickness`` Angstroms of the carbon are kept
        (they carry the surface the NP sits on and dominate the substrate
        contrast); the NP atoms are always kept. Positions are shifted to the
        cell origin and vacuum is added along z (the beam direction).
        """
        symbols = np.array(atoms.get_chemical_symbols())
        z = atoms.positions[:, 2]
        if 'C' in symbols:
            z_top_carbon = z[symbols == 'C'].max()
            atoms = atoms[(symbols != 'C') | (z >= z_top_carbon - self.substrate_thickness)]
        pos = atoms.positions - atoms.positions.min(axis=0)
        atoms.set_positions(pos)
        extent = pos.max(axis=0)
        atoms.set_cell([extent[0], extent[1], extent[2]])
        atoms.center(axis=2, vacuum=self.vacuum)
        if not self.noOutput:
            print(f"Simulation cell: {extent[0]:.1f} x {extent[1]:.1f} x "
                  f"{extent[2] + 2 * self.vacuum:.1f} A, {len(atoms)} atoms")
        return atoms

    def _simulate(self):
        """Run the abTEM multislice pipeline and store the final image."""
        # 1. Frozen-phonon potential
        self.frozen_phonons = abtem.FrozenPhonons(self.atoms, self.phonon_config,
                                                  sigmas=self.sigmas)
        self.potential = abtem.Potential(self.frozen_phonons, sampling=self.sampling,
                                         projection="infinite",
                                         slice_thickness=self.slice_thickness)
        # 2. Plane-wave multislice
        self.wave = abtem.PlaneWave(energy=self.energy)
        self.exit_wave = self.wave.multislice(self.potential)
        self.exit_wave.compute()
        # 3. CTF at Scherzer defocus, with astigmatism and temporal coherence
        self.ctf = abtem.CTF(Cs=self.Cs_value, energy=self.energy, defocus="scherzer",
                             semiangle_cutoff=self.semiangle_cutoff_value,
                             C12=self.C12, phi12=self.phi12,
                             focal_spread=self.focal_spread)
        self.defocus = self.ctf.defocus
        if not self.noOutput:
            print(f"defocus (Scherzer) = {self.defocus:.2f} A, "
                  f"focal_spread = {self.focal_spread:.2f} A")
        # 4. Partial coherence: intensity of the phonon ensemble through the CTF
        measurement = self.exit_wave.apply_ctf(self.ctf).intensity().mean(0)
        # 5. Poisson shot noise at the requested dose
        noisy = measurement.poisson_noise(dose_per_area=self.dose_poisson_noise)
        arr = noisy.array
        if hasattr(arr, "get"):  # cupy array (device='gpu') -> numpy
            arr = arr.get()
        # 6. Detector MTF, applied in Fourier space on the aligned fftfreq grid
        # (mean intensity preserved, see apply_detector_mtf).
        image = apply_detector_mtf(arr, self.sampling, self.c1, self.c2, self.c3,
                                   clip=self.mtf_clip)
        # 7. Normalize to [0, 1]
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        i_min, i_max = image.min(), image.max()
        self.image = ((image - i_min) / (i_max - i_min) if i_max > i_min
                      else np.zeros_like(image))

    @property
    def metadata(self) -> dict:
        """All the simulation parameters, ready for CSV export."""
        return {
            "sampling_A": self.sampling,
            "energy_eV": self.energy,
            "phonon_config": self.phonon_config,
            "sigmas_A": self.sigmas,
            "slice_thickness_A": self.slice_thickness,
            "Cs_value_A": self.Cs_value,
            "defocus_C10_A": self.defocus,
            "C12_A": self.C12,
            "phi12_rad": self.phi12,
            "Cc_value_A": self.Cc_value,
            "energy_spread_eV": self.energy_spread,
            "focal_spread_A": self.focal_spread,
            "semiangle_cutoff_mrad": self.semiangle_cutoff_value,
            "mtf_c1": self.c1,
            "mtf_c2": self.c2,
            "mtf_c3": self.c3,
            "mtf_clip": self.mtf_clip,
            "dose_poisson_noise_e-A-2": self.dose_poisson_noise,
            "substrate_thickness_A": self.substrate_thickness,
            "device": self.device,
        }

    def show(self, ax=None, cmap: str = 'gray'):
        """
        Display the HRTEM image with matplotlib.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to draw on; a new figure
                is created when omitted.
            cmap (str): Matplotlib colormap name.

        Returns:
            matplotlib.axes.Axes: The axes containing the image.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        cell = self.atoms.cell.lengths()
        # abTEM arrays are indexed (x, y): transpose for imshow's (row, col)
        ax.imshow(self.image.T, cmap=cmap, origin='lower',
                  extent=[0, cell[0], 0, cell[1]])
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        return ax

    def save(self, filename: str, size: int = None, metadata: bool = False):
        """
        Save the HRTEM image as a grayscale PNG file.

        Args:
            filename (str): Output PNG path.
            size (int, optional): If given, resize the image to size x size
                pixels (Lanczos resampling); the native resolution is kept
                otherwise.
            metadata (bool): If True, also write ``<filename>_metadata.csv``
                with all the simulation parameters.
        """
        image_uint8 = (self.image * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(image_uint8, mode="L")
        if size is not None:
            pil_image = pil_image.resize((size, size),
                                         resample=PILImage.Resampling.LANCZOS)
        pil_image.save(filename)
        if metadata:
            csv_path = str(Path(filename).with_suffix('')) + "_metadata.csv"
            pd.DataFrame([self.metadata]).to_csv(csv_path, index=False)
            if not self.noOutput:
                print(f"Metadata written: {csv_path}")
        if not self.noOutput:
            print(f"Image written: {filename}")
