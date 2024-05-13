**Development of a pre-release version -> date-based versioning**

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
### changed
- trigonal platelet added on the introductory figure - with the help of the corresponding code of the main notebook
- `calculateTruncationPlanesFromVertices`, developed for truncated tetrahedra (under the name `calculateTruncationPlanes`) has been moved to `utils.py`. And from now on the *relative* position of the trucation plane is given as a parameter, `cutFromVertexAt` (for example, `cutFromVertexAt=1/3` -> remove atoms that lie above the plane defined by (edge length)/3 from the vertices)

## 20240508
### added
- new `timer` class in utils.py, now used in `utils.MolSym()` and `utils.optimizeEMT()` functions, in the `coords()` functions of each class as well as in the `makeNP()` function of the `crystalNP` module
- docstring of `utils.truncateAboveEachPlane`
- new fcc trigonal platelet, aka `fcctpt` class in `otherNPs` module

### changed
- `utils.truncateAboveEachPlane`:
    - the `fractionOfEdgeDeleted` and `numberOfAtomsPerEdge` input variables were just given for printing purposes. Removed and now the message must be printed before calling `truncateAboveEachPlane`, if needed (this is for example the case in the `fcctrtd` class)
    - `AtomsAbovePlanes` variable renamed `keptAtoms`
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