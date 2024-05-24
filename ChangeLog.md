**Development of a pre-release version -> date-based versioning**

## 20240524
### added
- new functions in `utils.py`, that will be useful as basic tools for the automatic generation of wires: `returnPlaneParallel2Line()`, `AngleBetweenVV()`, `isPlaneParrallel2Line()`, `isPlaneOrthogonal2Line()`, `areDirectionsOrthogonal()`, `areDirectionsParallel()`

## 20240523
### added
- basic tests on the orthogonality between vectors and planes ([*sandbox notebook*](./SandBox-doNotDelete-dev.ipynb), section *Planes parallel to a line*)

## 20240522
### changed
- *Change of basis* section in the [*sandbox notebook*](./SandBox-doNotDelete-dev.ipynb): `lattice_cart()` validated in the hcp case

### fixed
- `lattice_cart()` function in `utils.py`: now does the right projection (`vectors@Vuc` instead of `Vuc@vectors`, ahah)

## 20240521
### added
- `lattice_cart()` function in `utils.py`: project vectors from/to the Bravais axis system to/from the cartesian coordinate system. The unitcell definition is given under the form of an ase `Atoms` object with periodic boundary conditions (see basic example in the [*sandbox notebook*](./SandBox-doNotDelete-dev.ipynb), *Change of basis* section)

### changed
- `Crystal` class of `CrystalNP.py` module: cif and cifname variables are now variables of the instance created by this class (i.e. `self.cif` and `self.cifname`). Was necessary to directly address some `Crystal` functions, without using the `makeNP(self)` "macro"

## 20240520
### added
- pentagonal bipyramids, Ino and Marks decahedra added on the introductory figure - with the help of the corresponding code of the main notebook
- class names added in the introductory figure
- new `data.py` module, with the `pNMBdata` class, that contains the list of clusters defined in pNMB (so far)
- `magicNumbers` function in `utils.py`,as well as new Magic Numbers section in the `pyNanoMatBuilder` notebook
 
### changed
- `prop` function in the `epbpyM` class

### fixed
- `magicEdgeNumberOfOh2MakeATrOh` in `fccTrOh` class of `archimedeanNPs` module now starts at 3 (previously 9. Oups)
- magic numbers of the `fccdrDD` class were wrong (formula was that of `bccrDD`)

## 20240516-19
### added
- strategy for pentagonal bipyramids, Ino decahedra and Marks decahedra implemented in the `epbpyM` class (elongated - aka Ino - pentagonal bipyramid and Marks decahedra) of `johnsonNPs.py`
- docstring of `utils.truncateAbovePlanes`

## 20240515
### added
- `utils.truncateAbovePlanes`: intends to be a generalization of `utils.truncateAboveEachPlane` (see also the *Decahedron, Ino & Marks decahedra* section in [`SandBox-doNotDelete-dev.ipynb`](./SandBox-doNotDelete-dev.ipynb)). Returns an array of booleans that tells which atoms fulfill the input conditions (above/below each/all input plane(s))
- `deleteElementsOfAList`: delete all items of a list (or array) specified by a list (or array) of booleans (returns a list)

## 20240514 (NRR)
### added
- function to compute I(q) from a xyz file. Needs to be optimized (cython?) and implemented in the library
### deleted
- doublons
- obsolete functions

## 20240512
### added
- strategy for pentagonal bipyramids, Ino decahedra and Marks decahedra evaluated in [`SandBox-doNotDelete-dev.ipynb`](./SandBox-doNotDelete-dev.ipynb)
- truncated cube sub-section in the Archimidean solids section of `pyNanoMatBuilder.ipynb` and in `archimedeanNPs.py`. <span style='color:red'>Development in progress (magic numbers inconsitent with those of Kaatz2019)</span>

### changed
- trigonal bipyramid class (`fcctbp`) and pentagonal bipyramid code in `pyNanoMatBuilder.ipynb` moved to newly created `johnsonNPs.py`

## 20240510
### added
- new fcc truncated octahedron, aka `fccTrOh` class in `archimedeanNPs` module & added in the introductory figure
- pentagonal bipyramid <span style='color:red'>under development</span> in the main notebook

## 20240509
### added
- trigonal platelet added on the introductory figure - with the help of the corresponding code of the main notebook

### changed
- `calculateTruncationPlanesFromVertices`, developed for truncated tetrahedra (under the name `calculateTruncationPlanes`) has been moved to `utils.py`. And from now on the *relative* position of the trucation plane is given as a parameter, `cutFromVertexAt` (for example, `cutFromVertexAt=1/3` -> remove atoms that lie above the plane defined by (edge length)/3 from the vertices)

## 20240508
### added
- new `timer` class in utils.py, now used in `utils.MolSym()` and `utils.optimizeEMT()` functions, in the `coords()` functions of each class as well as in the `makeNP()` function of the `crystalNP` module
- docstring of `utils.truncateAboveEachPlane`
- new fcc trigonal platelet, aka `fcctpt` class in `otherNPs` module

### changed
- `utils.truncateAboveEachPlane`:
    - the `fractionOfEdgeDeleted` and `numberOfAtomsPerEdge` input variables were just given for printing purposes. Removed and now the message must be printed before calling `truncateAboveEachPlane`, if needed (this is for example the case in the `fcctrtd` class)
    - `AtomsAbovePlanes` variable renamed `delAtoms`
    - input simplified to a truncation plane and to the input coordinates (all other parameters were specific to the truncated tetrahedron. Now externalized in the `fcctrtd` class, using the new `calculateTruncationPlanes` function)

### fixed
- `regfccTd` class of the `platonicNPs` module: wrong `interLayerDistance`. It revealed some issues. Now edge length of the reference Td = 1 (instead of 2sqrt(2), and all changes have been made accordingly (mainly coord vertices = f(c = 1/(2*np.sqrt(2))) and scale = `self.radiusCircumscribedSphere()`)
- `fccTrTd` class of the `archimedeanNPs` module: `nAtomsAnalytic(self)` was calculated as a function of the number of edge atoms of the initial tetrahedron instead of the number of edge atoms of the resulting truncated Td
- `fcctbp` class of the `otherNPs` module: `heightOfBiPyramid` was wrong

## 20240507
### added
- new rotation tools in `utils.py`
    - `Rx(a)`, `Ry(a)`, `Rz(a)` return rotation matrices around x, y and z
    - `EulerRotationMatrix()` returns a 3x3 Euler matrix (order of x/y/z rotations can be specified)
    - `RotationMatrixFromAxisAngle()` returns an [axis-angle rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)
