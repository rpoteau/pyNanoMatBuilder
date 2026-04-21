# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyNanoMatBuilder is a Python library (GPL-3.0) for generating atomic-scale 3D models of nanoparticles from crystal structures and atomically precise polyhedra. It uses ASE `Atoms` objects throughout and integrates with pymatgen for crystallography, Jmol for visualization, and optionally Numba for performance.

## Installation & Build

```bash
# Editable install from source (standard dev setup)
pip install -e .

# With optional extras
pip install -e ".[docs,debye,tem,visu]"

# Build docs locally (requires sphinx)
cd docs && make html
```

There is no test suite currently.

## Architecture

### Class Hierarchy

All NP classes inherit from `pyNMBcore` (pyNMBcore.py), which centralizes shared state:

```
pyNMBcore (base — shared attributes & methods)
├── Crystal          (crystalNPs.py)    — Wulff construction, crystal-based shapes
├── PlatonicNP       (platonicNPs.py)   — regfccOh, regfccTd, regIco
├── ArchimedeanNP    (archimedeanNPs.py)— fccCubo (cuboctahedra)
├── CatalanNP        (catalanNPs.py)    — bccrDD (rhombic dodecahedra)
├── JohnsonNP        (johnsonNPs.py)    — fcctbp, fcctpt
├── OtherNP          (otherNPs.py)      — miscellaneous polyhedra
└── bch              (chiralNPs.py)     — Boerdijk-Coxeter helices
```

### Key State on `pyNMBcore`

- `NP`, `NPcs` — full NP and core/shell `ASE.Atoms` objects
- `surfaceAtoms`, `cog` — surface atom indices and center of gravity
- `vertices`, `simplices`, `equations` — convex hull (scipy)
- `moi`, `Rg`, `NPR` — moments of inertia, radius of gyration, normalized principal moments
- `vol_Hull`, `area_Hull` — hull geometry
- `opd_index`, `chirality` — OPD chirality index and RH/LH/achiral classification
- `*_opt` variants — same properties after ASE geometry relaxation
- `jMolCS` — Jmol visualization script string

### Utility Modules (`pyNanoMatBuilder/utils/`)

| Module | Role |
|---|---|
| `geometry.py` | Polyhedron vertex generation, face filling (`MakeFaceCoord`), hull facet coplanarity reduction |
| `prop.py` | CN calculation, Rg, NPR, OPD chirality index, inscribed/circumscribed spheres |
| `crystals.py` | Metric tensors `G()`/`G_star()`, Miller index conversion, unit cell scaling |
| `io.py` | File read/write, CIF database indexing, resource path management |
| `symmetry.py` | Pymatgen `SpacegroupAnalyzer` wrapper, equivalent Miller plane detection |
| `energy.py` | ASE EMT calculator, geometry optimization |
| `polydispersity.py` | `NanoparticleDistribution` class — Gaussian/LogNormal TEM size distribution analysis |
| `external_pgm.py` | Jmol integration: script generation, `saveCoords_DrawJmol()`, PNG rendering |
| `parallel.py` | Numba JIT fallback — graceful degradation when Numba is unavailable (`HAS_NUMBA` flag) |

### Wulff Construction Data Flow

`data.py` holds `WulffShapes`: a dict mapping crystal systems to Miller planes and surface energies. `Crystal` reads this, queries `symmetry.py` (pymatgen) for equivalent planes, then builds the Wulff shape via `crystalNPs`.

### Resource Files

Built-in CIF files and visualization assets live in `pyNanoMatBuilder/resources/` and are accessed via `importlib.resources` (see `utils/io.py`). Do not use bare relative paths to reach these files.

## Adding a New NP Shape Class

1. Create a new module (e.g., `myNPs.py`) and inherit from `pyNMBcore`.
2. In `__init__`, call `super().__init__()` first, then populate `self.NP` (an `ASE.Atoms` object).
3. Populate the hull/surface attributes by calling the relevant helpers in `utils/prop.py` and `utils/geometry.py`.
4. Export the class from `pyNanoMatBuilder/__init__.py`.

## Numba / Threading

`parallel.py` wraps Numba's `@njit` decorator and exposes `HAS_NUMBA`. Performance-critical kernels (e.g., `prop._opd_kernel`) guard with `if HAS_NUMBA`. Package-level thread control: `set_threads(n)` / `get_threads()` in `__init__.py`.

## Key Dependencies

`ase`, `numpy`, `scipy`, `pandas`, `pymatgen`, `matplotlib`, `plotly`, `py3Dmol`, `ipywidgets`. Python ≥ 3.11 required.

## Code Style

- All docstrings must be written in English
- All inline comments must be written in English
- Follow Google-style or NumPy-style docstrings (consistent with ASE conventions)