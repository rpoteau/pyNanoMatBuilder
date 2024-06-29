# Development of a pre-release version -> date-based versioning

## 20240629
### added
- `utils.py`: `noOutput` added in
    - `calculateTruncationPlanesFromVertices()`
    - `truncateAboveEachPlane()`

### changed
- `aseView` set to `False` by default
- `noOutput` and `aseView` chase continued in:
    - `platonicNPs.py`
    - `archimedeanNPs.py`
    - `catalanNPs.py`
    - `johnsonNPs.py`
    - `otherNPs.py`

## 20240628
### added
- `data.py`:
    - new `pyNMBimg.IMGdf` dataframe, that contains an information necessary to define `self.imageFile`
    - `setdAsNegative(planes)`: returns each plane [a b c d] of the `planes` array as [-a -b -c -d] if d is positive
    - `data.WulffShapes`: new `hcpsph2` (Nørskov *et al*, [10.1126/science.1106435](https://dx.doi.org/10.1126/science.1106435))
- `crystalNPs.py`:
    - `self.WulffShape` now defined from `self.shape` in `__init__` instead of `makeWulff()`
    - `self.trPlanes` initialized to `None` in `__init__`

### changed
- graphical documentation updated with a general Wulff structure

### fixed
- `utils.py`:
    - `defCrystalShapeForJMol()` returns a jMol command if `trPlanes` is not `None`
- several functions were recently modified to accept`noOutput` as argument; the `aseView` argument is now taken into account here and there (#12). Also added:
    - `printN = not noOutput` and `printV = not noOutput` as argument to `normal2MillerPlane()` & `lattice_cart()` in `crystalNPs.py` > `makeWire()` and `makeParallelepiped()`
    - `noOutput` added as argument of `planeRotation()`

## 20240626
### changed
- `data.py`:
    - `bccrdd`, `trbccrdd`, `ttrbccrdd`, `cub`, `trcub` and `dicotd` predifined Wulff shapes renamed `bccrDD`, `trbccrDD`, `ttrbccrDD`, `cube`, `trcube` and `dicoTd`
- graphical documentation updated with the pre-defined Wulff structures

## 20240625
### added
- `utils.py`:
    - `saveCoords_DrawJmol()`:
        - `boundaries` option (default: `False`). If Wulff shapes or any other boundary-defined structure (such as wires), facet plots with the time-consuming `./figs/script-facettes-345PtLight.spt` jmol script is useless. Set as `False` to unactivate it, as well as the bonds and atoms drawing, since they are defined in the `jMolCS` script
        - `noOutput=True` option
    - `path2Jmol` defined locally in `saveCoords_DrawJmol()` is now defined in the `pyNMBvar` class of `data.py`

### fixed
*doing and undoing... is is still work?*
- `crystalNPs.py`: in `makeSuperCell()`
```
        sc.translate(-V[0]/2)
        sc.translate(-V[1]/2)
        sc.translate(-V[2]/2)
```
    is back...

## 20240623
### fixed
- `crystalNPs.py`: in `makeSuperCell()`, `sc.center(about=(0.0,0.0,0.0))` replaces:

```
        sc.translate(-V[0]/2)
        sc.translate(-V[1]/2)
        sc.translate(-V[2]/2)
```

    Check in all examples that everything is consistent with this centering

## 20240622
### added
- `utils.py`:
    - `scaleUnitCell()`: scales the unit cell size so that the nearest NN distance is scaled to the `scaleDmin2` input parameter (see `scaleDmin2` variable of the `Crystal` class)
    - `saveCN4JMol()`: saves coordination numbers in a CN.dat file, and print the jmol command to link atom colors and atom CNs
    - `plotPalette()`: plots a 1D palette colors, with names
    - `rgb2hex()`: converts rgb numbers to #hex code, under the form [xAABBCC] - this is for jMol
- `crystalNPs.py`:
    - new `setSymbols2` variable (array). Can be associated to `scaleDmin2` in order to start from a given cif file and change the atom(s) as new ones. The number of chemical symbols must fit the number of atoms of the reference unit cell

### changed
- `crystalNPs.py`:
    - `noOutput` variable now also effective whith `chrono` calls and various print commands in `makeWulff()`
    - new `scaledR` variable is now effective
- `utils.py`:
    - point group returned as `pg` property of `aseobject: Atoms` object
    - `noOutput` variable now also effective whith `chrono` calls and various `print` commands in `MolSym()`, `moi()`, `returnPointsThatLieInPlanes()`, `defCrystalShapeForJMol()`, `reduceHullFacets()`, ` kDTreeCN()`, `truncateAbovePlanes()`
- `pyNanoMatBuilder.ipynb` renamed as `pyNMB-examples.ipynb
- `pNNBu`, `pNMBdata`, `pNMBvar`, `pNMBcif` shortcuts renamed as `pyNMBu`, `pyNMBvar`, `pyNMBdata`, `pyNMBcif`

## 20240619
### added
- `utils.py`:
    - new `kDTreeCN()` function, that returns the list (`nn`) and number (`CN`) of nearest neighbours of each atom; distances are returned as well if returnD is `True` (based on the very efficient `KDtree()` function of scikit-learn)
- `crystalNPs.py`:
    - new `scaleDmin2` variable: if not `None`, all coordinates are scaled so that the nearest neighbour distance in the crystal becomes scaleDmin2 (**under development**)

## 20240609
### added
- `SandBox-doNotDelete-dev.ipynb`:
    - test of the `KDTree` algorithm for the nearest neighbour search

## 20240609
### added
- `pyNanoMatBuilder.ipynb`:
    - *Find all symmetry-equivalent planes* subsection in the *Miscellaneous* section, aka use of the `ase.spacegroup.Spacegroup` tools

## 20240608
### added
- `crystalNPs.py`
  - new `jmolCrystalShape` boolean variable (default: `False`)
- `utils.py`
    - `coreSurface()`
        - receives `Crystal` instance as parameter instead of the coordinates, so that now it is possible to use or return associated properties (`Crystal.trPlanes`, `Crystal.jMolCS`, `Crystal.NP.get_positions()`)
        - now also calls `defCrystalShapeForJMol(Crystal)`

### changed
- `utils.py`
    - `defWulffShapeForJMol()` renamed as `defCrystalShapeForJMol()`
    - reduction of Hull facets (simplices) that was part of `defWulffShapeForJMol()` is now an external `reduceHullFacets()` function
- `crystalNPs.py`
    - `defCrystalShapeForJMol()` is called at the end of `makeNP()` if `jmolCrystalShape` is True and the jmol command is returned as `self.jmolCS`
- `platonicNPs.py`, `cube` class: coordinates centered in [0,0,0] (was made necessary for the crystal shape calculation)
```
        coords = sc.get_positions()
        oldcog = sc.get_center_of_mass()
        coords = coords - oldcog
        sc.set_positions(coords)
```
- `platonicNPs.py`, `archimedeanNPs.py`, `catalanNPs.py`
    - in `propPostMake()`, `self` is passed as an argument to `pNMBu.coreSurface()`, instead of the array of atomic coordinates
    - `self.cog = self.NP.get_center_of_mass()` added at the end of the `coords()` functions
 
### fixed
- `utils.py`
    - in very borderline cases `linalg.eig` returned complex numbers in `planeFittingLSF()`, the complex part being 0j => returned variable is now `np.array([u,v,w,h]).real`

## 20240607
### changed
- `pyNanoMatBuilder.ipynb`:
    - two main parts, namely *Crystal structure-based shapes* and *Magic number clusters and nanoparticles*
 
### added
- `pyNanoMatBuilder.ipynb`:
    - all predefined Wulff crystals are displayed in a *List of the pre-defined Wulff shapes in the `data.WulffShapes.WSdf` pandas dataframe* section, at the end of the notebook
- `crystalNPs.py`
    - in `predefinedParameters4WulffShapes`, a warning is displayed if the expected lattice system of the Wulff shape is not the same as that of the Bravais lattice system of the cif file 

### fixed
- `signedAngleBetweenVV()` function in `utils.py`: was using an arbitrary normal vector. It is now calculated in `defWulffShapeForJMol()` and passed as an argument of `signedAngleBetweenVV()` 

## 20240606
### added
- `data.py`
    - declaration of a `WSdf` pandas dataframe in a new `WulffShapes` class: contains the energy and plane parameters of some remarkable Wulff structures, as well as the lattice system  and Bravais lattice(s) consistent with each Wulff shape
- `crystalNPs.py`
    - `predefinedParameters4WulffShapes()` function in the `Crystal` class, called if `Crystal.shape` contains `Wulff` followed by the ":" separator and a shortcut associated to a given shape (list in the `WulffShapes.WSdf` dataframe - see index of `WSdf`). Reads the `WSdf` pandas dataframe and initializes `self.eSurfacesWulff` and `elf.surfacesWulff` if the user-defined shortcut is found in the `WSdf` dataframe. Otherwise an error message is returned and `pyNanoMatBuilder` stops

### changed
- names of cif files and shortcut top address them are now also stored as a `CIFdf` pandas dataframe in a new `pNMBcif` class, created in `data.py`. Shortcuts are declared as the index of the dataframe, whilst cif filenames are available in the `cif file` column of `CIFdf`. `pyNanoMatBuilder` stops if the shortcut (case-independent) is not found in `CIFdf`

## 20240605
### added
- abstract in README.md
- `defWulffShapeForJMol()` function in `utils.py`: return the jmol command to plot the Wulff shape
- `signedAngleBetweenVV()` function in `utils.py`, introduced for `defWulffShapeForJMol()`: calculates the signe angle between two vectors. Useful to reorder the vertices of a polygon

### changed
- `crystalNPs.py`
    - `normal2MillerPlane()` called before `pNMBu.lattice_cart()` in `crystal.makeWulff()` & in `crystal.makeParallelepiped()`, and before `pNMBu.planeRotation()` in `crystal.makeWire()`

## 20240603
### added
- `utils.py`
    - `normal2MillerPlane()` function: returns the the normal direction (*n*1, *n2*, *n*3) to the plane defined by *h*,*k*,*l* Miller  indices (calculated as [n1 n2 n3] = (hkl) x G*, where G* is the reciprocal metric tensor). (*n*1, *n2*, *n*3) are returned as the closest integers to the float numbers calculated by this equation. This calculation is mandatory for non-orthogonal basis sets

### changed
- `return_unitcell()` function moved from `CrystalNPs.py` to `utils.py` and renamed `returnUnitcellData()`. Now associates unitcell variables to a `Crystal` class instance:
    - Bravais lattice, as `ucBL`
    - space group calculated by `ase`, as `ucSG`
    - unitcell, as `ucUnitcell`
    - a,b,c = f(x,y,z) vectors, as `ucV`
    - volume as, `ucVolume`
    - reciprocal lattice, as `ucReciprocal`
    - chemical formula as, `ucFormula`

## 20240603
### added
- `pNMBvar` dataclass in `data.py`. So far, defines only `dbFolder = 'cif_database'`
- new `symWulff` instance boolean variable of the `Crystal` class (default `True`). If `True` all symmetry operations of the crystal space group are applied to all `surfaceWulff` truncation planes

### changed
- the listing of all cif files contained in the `dbFolder`, with the symmetry properties and unit cell values has been transformed as a `listCifsOfTheDatabase()` function, available in `utils.py`

## 20240602
### added
- in the `Crystal` class:
    - **new `makeWulff()` function**, with its associated variables: `surfacesWulff`, `eSurfacesWulff`, `sizesWulff`. *This is a basic implementation, still a lot of work to do to account for symmetry*
    - `ase.spacegroup.get_spacegroup(ase_Atoms_object,symprec=1e-4)` added in the `bulk()` function
    - `sg` instance object is available, it is a Spacegroup object of ase
    - the space group number and Hermann-Mauguin symbol are printed by `print_unitcell()`
    - new `aseSymPrec` instantiation variable, set to 1e-4 by default
- in `pyNanoMatBuilder.ipynb` notebook: new *List all cif files available in the database* section, that prints basic crystal info, and compares the ase symmetry analyzis with the space group info available in the cif files. A warning is triggered if they differ
- and regarding crystal symmetries, `ase.spacegroup.get_spacegroup(ase_Atoms_object)` tested in the sandbox notebook. Interesting, but does not find the right group for Ru hcp. Can be fixed by setting up `symprec` as 1e-4 instead of the 1e-5 default). Finally it will probably do the job to generate all symmetry equivalent planes of a given crystal

### changed
- on the graphical documentation, a distinction is made between *atomically precise* NPs and NP *shapes*
- `print_unitcell()` of the `Crystal` class renamed as `print_ase_unitcell()` and moved in `utils.py`

## 20240601
### added
- cif coordinates for TiO2 rutile, TiO2 anatase, NaCl
- in the `Crystal` class, it is now possible to load a cif file that does not belong to the database, using the new `userDefCif` keyword. The `loadExternalCif()` function is called if `userDefCif` is not `None`, *i.e.* it contains the path to a cif file
- `alignV1WithV2_returnR()` and `rotateMoltoAlignItWithAxis()` functions in `utils.py`. See example of application in the sandbox notebook, section *Calculate rotation matrix to align two vectors in 3D space*. Could for example be useful to align wires along the *c* direction
- **new `makeParallelepiped()` function in the `Crystal` class** (called by `makeNP()`). Associated keywords:
    - `directionsPPD` variable, used to build new parallelepiped 3D structures. Default is [[0,0,1],[0,1,0],[0,0,1]]
    - completed with `buildPPD`: if `buildPPD="xyz"` (default), `directionsPPD` are applied in the cartesian coordinate system, otherwise if `buildPPD="abc"`, `directionsPPD` are applied in the Bravais coordinate system

### changed
- `noOutput` and `calcPropOnly` variables introduced for Platonic, Archimedean, Catalan and Johnson NPs also introduced in the `Crystal` class of `crystalNPs.py`
- `silent` variable renamed as `noOutput` in all classes
- in `crystalNPs.py`, `direction`, `nRot` and `refPlane` renamed `directionWire`, `nRotWire` and `refPlaneWire`

## 20240531
### changed
- changes previously done to the `Crystal` class in the **20240529** version applied to the classes of:
    - `otherNPs.py`

## 20240530
### added
- because classes of `platonicNPs.py` can be used as generators for other polyhedra (truncated NPs, etc), new keywords are introduced:
    - `silent`: does not print anything
    - `calcPropOnly`: does not calculate the coordinates 

### changed
- changes previously done to the `Crystal` class in the **20240529** version applied to the classes of:
    - `platonicNPs.py`
    - `archimedeanNPs.py`
    - `catalanNPs.py`
    - `johnsonNPs.py`

## 20240529
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

## 20240528
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
- **new `makeWire()` function in the `Crystal` class of `CrystalNP.py`** (called by `makeNP()`). This is a first version that must be *thoroughly tested* and *optimized*. It builds a facetted nanowire along a given direction. Only one facet must be defined, as well as a rotation factor to automatically calculate the other facets. Associated keywords:
    - `direction` is the growth direction of the wire (default=[0,0,1])
    - `refPlane` is the reference facet (default=[1,0,0])
    - `nRot`
    - `pbc` (default = False). Specifies whether periodic boundary conditions mut be considered or not.
        - if `pbc` is False, a finite-size wire is generated by truncating its extremities by planes normal to its main axis, and its length = size[1]
        - an infinite wire is generated otherwise
- new `planeRotation()` function in `utils.py`: returns an array with planes obtained by an nth-order rotation of a reference plane around an input axis

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
- strategy for **pentagonal bipyramids, Ino decahedra and Marks decahedra** implemented in the `epbpyM` class (elongated - aka Ino - pentagonal bipyramid and Marks decahedra) of `johnsonNPs.py`
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
- new **fcc truncated octahedron, aka `fccTrOh` class in `archimedeanNPs` module** & added in the introductory figure
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
- new **fcc trigonal platelet, aka `fcctpt` class in `otherNPs` module**

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

## 202404 - 202405
**Basic architecure of `pyNanoMatBuilder`, overall strategy defined (vertices, edges and planes for basic polyhedra), and first "easy" polyhedra implemented**
