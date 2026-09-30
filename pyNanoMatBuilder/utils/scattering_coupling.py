"""
Simple pyNanoMatBuilder <-> scattering-library coupling.

Compute the powder X-ray scattering profile I(q) of a nanoparticle built with
pyNanoMatBuilder, using either of two backends:

- :func:`debye_profile`  -- DebyeCalculator (https://github.com/FrederikLizakJohansen/DebyeCalculator)
- :func:`ausaxs_profile` -- pyAUSAXS (https://github.com/AUSAXS/pyAUSAXS)

Both backends require a structure file on disk, so each function writes the NP
to a temporary XYZ file (or to ``xyz_file`` when a path is given) and cleans up
afterwards; the caller never has to handle that.

The two backends have mutually incompatible dependency constraints and
therefore live in **separate environments** (see ``requirements-debye.txt`` /
``requirements-pyausaxs.txt`` and their coupling notebooks). Each function
imports its own backend lazily, so importing this module never fails: the
missing-backend error is raised only when the function is actually called, with
a message pointing to the right environment.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np

from ase.atoms import Atoms

# pyNanoMatBuilder's own writer, not ase.io.write: for .xyz it emits a plain
# XYZ file, whereas ASE defaults to the extended format whose
# "Properties=species:S:1:pos:R:3" comment line pyAUSAXS cannot parse.
from .io import write


def _resolve_atoms(NP) -> Atoms:
    """
    Return the ``ase.Atoms`` object of the nanoparticle.

    Args:
        NP: Either a pyNanoMatBuilder shape instance (its ``.NP`` attribute is
            used) or directly an ``ase.Atoms`` object.

    Returns:
        Atoms: The nanoparticle as an ASE Atoms object.
    """
    if isinstance(NP, Atoms):
        return NP
    if hasattr(NP, "NP") and isinstance(NP.NP, Atoms):
        return NP.NP
    raise TypeError("NP must be a pyNanoMatBuilder shape instance or an ase.Atoms "
                    f"object, got {type(NP).__name__}.")


_DEBYE_SCATTERING = ("iq", "sq", "fq", "gr")


def _apply_form_factors(q, intensity, atoms, noOutput=True):
    """
    Apply the atomic form factor correction to a raw Debye sum.

    A raw Debye sum weights every atom by 1, so the scattering power of the
    element is missing. For a **monoatomic** nanoparticle the correction is
    exact: I(q) = f0(q)^2 * I_raw(q). For a multi-element particle the
    per-pair products f_i(q) f_j(q) cannot be recovered from the raw sum, so
    the correction is skipped and a warning is issued.

    Args:
        q (ndarray): Scattering vector magnitudes in 1/Angstrom.
        intensity (ndarray): Raw Debye intensity.
        atoms (Atoms): The nanoparticle, used to identify the element(s).
        noOutput (bool): If False, print the element used for the correction.

    Returns:
        ndarray: The corrected (or unchanged) intensity.
    """
    symbols = set(atoms.get_chemical_symbols())
    if len(symbols) != 1:
        warnings.warn(
            f"The NP contains several elements ({sorted(symbols)}): the atomic form "
            "factor correction is exact only for a monoatomic particle and has been "
            "skipped. The returned intensity is the raw Debye sum, without form "
            "factors.", stacklevel=3)
        return intensity

    from .compute_f0 import f0_from_Q
    from ..resources.elements_f0_coeff import elements_info

    element = symbols.pop()
    f0 = np.asarray(f0_from_Q(q, element, elements_info), dtype=float)
    if not noOutput:
        print(f"Atomic form factor correction applied for {element} "
              f"(f0 at q={q[0]:.4g} 1/A = {f0[0]:.3f})")
    return intensity * f0**2


def debye_profile(NP, scattering: str = "iq",
                  qmin: float = 0.001, qmax: float = 20.0, qstep: float = 0.001,
                  rmin: float = 0.0, rmax: float = 20.0, rstep: float = 0.01,
                  biso: float = 0.0, device: str = 'cpu',
                  xyz_file: str = None, noOutput: bool = True):
    """
    Compute a scattering function of a nanoparticle with DebyeCalculator.

    DebyeCalculator evaluates the Debye scattering equation including the
    atomic form factors of every element, so the reciprocal-space functions
    need no further correction.

    IMPORTANT - the meaning of the returned x-axis depends on ``scattering``:

    - ``'iq'``, ``'sq'``, ``'fq'`` are **reciprocal-space** functions: the
      x-axis is the scattering vector q (1/Angstrom), sampled on the q grid
      (``qmin``, ``qmax``, ``qstep``). Only ``'iq'`` is positive and spans
      several decades (plot it log-log); ``'sq'`` and ``'fq'`` oscillate and
      take negative values, so they must be plotted on **linear** axes.
    - ``'gr'`` is the **real-space** reduced pair distribution function G(r):
      the x-axis is the distance r (Angstrom), sampled on the r grid
      (``rmin``, ``rmax``, ``rstep``) -- NOT the q grid. G(r) oscillates and
      is signed, so it must be plotted on **linear** axes. The q range still
      matters: it is the Fourier-transform window used to build G(r) (a larger
      ``qmax`` sharpens the peaks, a non-zero ``qmin`` acts like a PDF Qmin cut).

    Args:
        NP: A pyNanoMatBuilder shape instance, or an ``ase.Atoms`` object.
        scattering (str): 'iq' scattering intensity I(q), 'sq' total structure
            function S(q), 'fq' reduced structure function F(q), or 'gr' reduced
            pair distribution function G(r).
        qmin (float): Lower bound of the q range in 1/Angstrom.
        qmax (float): Upper bound of the q range in 1/Angstrom.
        qstep (float): Step of the q grid in 1/Angstrom. Smaller is finer/slower.
        rmin (float): Lower bound of the r range in Angstrom (used by 'gr').
        rmax (float): Upper bound of the r range in Angstrom (used by 'gr').
            Set it larger than the NP diameter to capture the whole particle.
        rstep (float): Step of the r grid in Angstrom (used by 'gr').
        biso (float): Isotropic atomic displacement parameter B in Angstrom^2
            (0 disables the Debye-Waller damping).
        device (str): 'cpu' or 'cuda' (GPU acceleration through PyTorch).
        xyz_file (str, optional): Path of the XYZ file to write and keep. When
            omitted, a temporary file is used and deleted afterwards.
        noOutput (bool): If False, print a short summary.

    Returns:
        tuple[ndarray, ndarray]: ``(x, y)``. For 'iq'/'sq'/'fq', x is q
        (1/Angstrom); for 'gr', x is r (Angstrom). y is the selected function.

    Examples
    --------
    >>> q, iq = debye_profile(AuNP, scattering="iq", qmin=0.001, qmax=20, qstep=0.001)
    >>> r, gr = debye_profile(AuNP, scattering="gr", qmin=1.0, qmax=20, rmax=30)
    """
    if scattering not in _DEBYE_SCATTERING:
        raise ValueError(f"Invalid scattering type '{scattering}': must be one of "
                         f"{_DEBYE_SCATTERING}.")
    try:
        from debyecalculator import DebyeCalculator
    except ImportError as e:
        raise ImportError(
            "debye_profile requires the 'debyecalculator' package, which is not "
            "installed in this environment. Set up the dedicated environment "
            "described in the pyNMB-debyecalculator-coupling.ipynb notebook "
            "(pip install -r requirements-debye.txt), or install the extra with "
            "pip install \"pyNanoMatBuilder[debye]\". Note that debyecalculator "
            "and abTEM/pyAUSAXS cannot share the same environment."
        ) from e

    atoms = _resolve_atoms(NP)
    calc = DebyeCalculator(qmin=qmin, qmax=qmax, qstep=qstep,
                           rmin=rmin, rmax=rmax, rstep=rstep,
                           biso=biso, device=device)

    def _run(path):
        write(path, atoms)
        # calc.iq / calc.sq / calc.fq / calc.gr all take the structure path
        return getattr(calc, scattering)(path)

    if xyz_file is not None:
        x, y = _run(str(xyz_file))
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            x, y = _run(str(Path(tmpdir) / "pynmb_debyecalculator_scattering.xyz"))

    if not noOutput:
        axis = "r (A)" if scattering == "gr" else "q (1/A)"
        print(f"DebyeCalculator [{scattering}]: {len(atoms)} atoms, {len(x)} points, "
              f"x-axis = {axis}, device = {device}")
    return x, y


def ausaxs_profile(NP, qmin: float = 0.001, qmax: float = 20.0, npoints: int = 20000,
                   apply_f0: bool = True, threads: int = None,
                   bin_width: float = 0.1, bin_count: int = 1000,
                   xyz_file: str = None, noOutput: bool = True):
    """
    Compute the scattering profile I(q) of a nanoparticle with pyAUSAXS.

    The raw Debye sum of pyAUSAXS (``debye_raw``) carries no atomic form
    factors, and its higher-level model only knows the light elements
    (H, C, N, O, S) -- every heavier element would be treated as argon, which
    is wrong for the metals usually built with pyNanoMatBuilder. This function
    therefore uses the raw sum and applies pyNanoMatBuilder's own form factors
    (``utils.compute_f0``), which is exact for a monoatomic particle. See
    :func:`_apply_form_factors` for the multi-element case.

    Args:
        NP: A pyNanoMatBuilder shape instance, or an ``ase.Atoms`` object.
        qmin (float): Lower bound of the q range in 1/Angstrom.
        qmax (float): Upper bound of the q range in 1/Angstrom.
        npoints (int): Number of points of the (linear) q grid.
        apply_f0 (bool): Apply the atomic form factor correction (recommended;
            without it the intensity is a bare Debye sum).
        threads (int, optional): Number of threads used by pyAUSAXS. When
            omitted, the pyAUSAXS default is kept.
        bin_width (float): Width of the distance-histogram bins in Angstrom.
        bin_count (int): Number of histogram bins. It must be large enough to
            cover the whole distance range of the NP.
        xyz_file (str, optional): Path of the XYZ file to write and keep. When
            omitted, a temporary file is used and deleted afterwards.
        noOutput (bool): If False, print a short summary.

    Returns:
        tuple[ndarray, ndarray]: ``(q, I)``, the scattering vector magnitudes
        in 1/Angstrom and the corresponding intensities (a.u.).

    Examples
    --------
    >>> q, iq = ausaxs_profile(AuNP, qmin=0.001, qmax=20, npoints=20000)
    """
    try:
        import pyausaxs as ausaxs
    except ImportError as e:
        raise ImportError(
            "ausaxs_profile requires the 'pyausaxs' package, which is not installed "
            "in this environment. Set up the dedicated environment described in the "
            "pyNMB-pyausaxs-coupling.ipynb notebook "
            "(pip install -r requirements-pyausaxs.txt), or install the extra with "
            "pip install \"pyNanoMatBuilder[pyausaxs]\". Note that pyAUSAXS and "
            "debyecalculator/abTEM cannot share the same environment."
        ) from e

    atoms = _resolve_atoms(NP)
    q_in = np.linspace(qmin, qmax, npoints)

    if threads is not None:
        ausaxs.settings.general(threads=threads)
    ausaxs.settings.histogram(weighted_bins=True, bin_width=bin_width,
                              bin_count=bin_count)

    def _run(path):
        # pyAUSAXS reads the structure file (XYZ is supported by read_pdb)
        structure = ausaxs.read_pdb(path)
        molecule = ausaxs.create_molecule(structure)
        return molecule.debye_raw(q_in)

    if xyz_file is not None:
        write(str(xyz_file), atoms)
        q, iq = _run(str(xyz_file))
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pynmb_pyausaxs_scattering.xyz")
            write(path, atoms)
            q, iq = _run(path)

    if apply_f0:
        iq = _apply_form_factors(q, iq, atoms, noOutput=noOutput)
    if not noOutput:
        print(f"pyAUSAXS: {len(atoms)} atoms, {len(q)} q points "
              f"in [{qmin}, {qmax}] 1/A, form factors={'on' if apply_f0 else 'off'}")
    return q, iq
