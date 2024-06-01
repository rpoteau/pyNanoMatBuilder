**Development of a pre-release version -> date-based versioning**

## 20250601
### added
- cif coordinates for TiO2 rutile, TiO2 anatase, NaCl
- in the `Crystal` class, it is now possible to load a cif file that does not belong to the database, using the new `userDefCif` keyword. The `loadExternalCif()` function is called if `userDefCif` is not `None`, *i.e.* it contains the path to a cif file
- `alignV1WithV2_returnR()` and `rotateMoltoAlignItWithAxis()` functions in `utils.py`. See example of application in the sandbox notebook, section *Calculate rotation matrix to align two vectors in 3D space*. Could for example be useful to align wires along the *c* direction

### changed
- changes previously done to the `Crystal` class in the **20250529** version applied to the classes of:
    - `crystalNPs.py`
- `silent` variable renamed as `noOutput`

## 20250531
### changed
- changes previously done to the `Crystal` class in the **20250529** version applied to the classes of:
    - `otherNPs.py`

## 20250530
### added
- because classes of `platonicNPs.py` can be used as generators for other polyhedra (truncated NPs, etc), new keywords are introduced:
    - `silent`: does not print anything
    - `calcPropOnly`: does not calculate the coordinates 

### changed
- changes previously done to the `Crystal` class in the **20250529** version applied to the classes of:
    - `platonicNPs.py`
    - `archimedeanNPs.py`
    - `catalanNPs.py`
    - `johnsonNPs.py`

## 20250529
### added
- new *Convert the images to base64 code* section in the notebook. The intent is to embed images of nanoparticles in the log of the GUI that is under development. Base64 encoding of the `~/figs/*-C.png` files are now available in the same folder (*embedding not implemented yet*)
- in `utils.py`
    - `findNeighbours()` and `printNeighbours()`. <span style='color:red'>for several thousand atoms molecules, this python implementation is time-consuming. Use with caution... or don't be surprised if your builder seems frozen</span>
    - `coreSurface()`: does the convex Hull analysis available in `scipy.spatial`. Returns the `[hull.vertices,hull.simplices,hull.neighbors,hull.equations]`, as well as a `SurfaceOrCoreAtom` array of booleans
    - `returnPointsThatLieInPlanes`
- in the `Crystal` class, used as a test, and before generalization to the other classes
    - a new `propPostMake()` function is now called in `__init__`, to give some specifics properties of the final nano-object, **including a core/surface analyzis** based on the `scipy.spatial` convex Hull analysis. <span style='color:red'>Mind that such strategy does not apply to stepped surfaces</span>... in principle. A possible way to bypass this limitation is to define a threshold, so that all atoms lying "*just below*" the simplices are surface atoms. This is the intent of the new `thresholdCoreSurface` variable 
    - the final call to `propPostMake()` can be skipped if the instance is created with `postAnalyzis=False` (default is `True`)
    - the symmetry analyzis is skipped if `skipSymmetryAnalyzis` is `True` (default is `False`)

### changed
- in the `Crystal` class, used as a test, and before generalization to the other classes
    - `makeNP()` & `prop()` calls are now direcly made when an NP instance is created
    - `vID.centerTitle(general title)` now introduced in the `__init__` call of each class
    - all properties/objects associated are now associated to the created instance, let's call it `object`
        - `object.cif` returns the cif info of the crystal that serves as shape generator as an `ase` `Atoms` object
        - `object.sc` returns the supercell as an `ase` `Atoms` object
        - `object.NP` returns the nano-object as an `ase` `Atoms` object
        - `object.NPcs` returns the nano-object as an `ase` `Atoms` object, with core atoms labeled as "No", the Nobelium element. Why No? Because it has a nice <span style='color:#be1088'>**magenta**</span> color in jmol
      - calls to the ase viewer are made only if a `aseView` variable is `True` (default)
- in `utils.py`, the `vID.centertxt()` style of sub-subprocesses is changed as black fg on grey bg and font size 12 (example: `vID.centertxt(f"Convex Hull analyzis",bgc='#cbcbcb',size='12',fgc='b',weight='bold')`)

## 20250528
### added
- `ImagePathway()` and `plotImageInPropFunction()` in `utils.py`: their purpose is to draw the schematic representation of the NPs, as they appear in the documentation
- `prop()` function in the `Crystal` class of `crystalNPs.py`

### changed
- `ImagePathway()` and `plotImageInPropFunction()` functions are now used in all `__init__` and `prop()` functions of
    - `platonicNPs.py`
    - `crystalNPs.py` (*spheres* and *ellipsoids* only... so far)
    - `archimedeanNPs.py`
    - `catalanNPs.py`
    - `johnsonNPs.py`
    - `otherNPs`
- `crystalNPs`:
    - call to `self.bulk()`, *i.e.* cif loading, moved to `__init__`
    - call to `self.print_unitcell()` moved to `prop()`
- `johnsonNPs.py`: now `nLayer` is the number of layers per trigonal pyramid (*i.e.* the total number of layer is 2xnLayer+the twinning plane)

## 20240527
### changed
- new `threshold` instance variable (option) in the `Crystall class`. It is the atom-to-plane signed distance threshold used under the `eps` name in `utils.truncateAbovePlanes` (default: 1e-3)

## 20240526
### added
- new `makeWire()` function in the `Crystal` class of `CrystalNP.py`, called by `makeNP()`. This is a first version that must be *thoroughly tested* and *optimized*
- new `planeRotation()` function in `utils.py`: returns an array with planes obtained by an nth-order rotation of a reference plane around an input axis
- new `pbc` option in the `Crystal` class (default = False). Specifies whether periodic boundary conditions mut be considered or not. Only valid for *wires*:
    - if `pbc` is False, a finite-size wire is generated by truncating its extremities by planes normal to its main axis, and its length = size[1]
    - an infinite wire is generated otherwise

<span style="color:red">!! **pbc** only works if all atoms lie along the growth, main axis, pass through a unique plane of the unitcell &#128533; (part of the nanowire is not generated) !!</span>   

### changed
- `makeSuperCell` section, supercell calculation defined for wires (extra space is added if length > width)

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
