# Development of a pre-release version -> date-based versioning
## 20260309
## changed
Today is commit day ! I just renamed `make_xyz_files_remastered.py` as 
and  `MakeNPsDatabase.py` and `pyNMB-exmaples-sanspbcolonnes.ipynb` as `pyNMB-exmaples.ipynb`. I deleted a bunch of local folders and files. Also I removed UtilsDC.py and the corresponding notebook because it was not that useful ? To discuss ... (I think redemonstrating the use of an other library can be redundant and make it more heavy for nothing). Instead make a small doc?? 

-`TEM_creator.py`: substrate size is automatically found now, also the second index in the name file only changes if the imaging parameters change.

**TO DO AFTER THIS VERSION**:
- fix this error
"/home/sara/Python3/Debye_calc/lib/python3.11/site-packages/ase/io/cif.py:408: UserWarning: crystal system 'tetragonal' is not interpreted for space group Spacegroup(141, setting=1). This may result in wrong setting!
  warnings.warn(
/home/sara/Python3/Debye_calc/lib/python3.11/site-packages/ase/io/cif.py:408: UserWarning: crystal system 'cubic' is not interpreted for space group Spacegroup(229, setting=1). This may result in wrong setting!
  warnings.warn("
- I skipped the making of the files NaCl and TiO2, should add it.
- Fix the last shapes (bccrDD, double ico and pyramid of tetrahedons size measurements, fcctpbp total number of atoms, size measurement when "abc")
- for `TEM_creator.py` allow to pick specific shapes instead of forcing the user to do all of them.

## 20260308
## changed
-`TEM_creator.py`: New doc + big changes in the parameters (some werent even existing) + delete hcp making icosahedron (ive seen weird images). Also changed the final dataframes. Now the metadata should be more intersting and also accurate.

**TO DO**: PEP8 maybe, the doc seems good.

**TO DO**: verify the parameters and effect on the images (quality), verify the position of the NP on the carbon substrate (seems not good for 10x10nm and for 5x5nm always in the same spot, add function to rotate the images maybe?). + HRTEM_params notebook should be reworked probably.
## 20260304
## changed
-`TEM_creator.py` et `HRTEM_class.py`: changing the name of the files (too long) and making a final clean dataframe 

**TO DO**: reg ico can be hcp right now, put the condition of lattice == "fcc" for reg ico. Verify the content of the metadata: espcially the size !!!! (what is size index ? ) Final dataframe works but verify its content ! Make a nice doc for the notebook ! either add the other shapes, or precise that it is Wulff + ico only right now + rename all the notebook and class and path and position in the folder (HRTEM not TEM), and work on HRTEM_params notebook

-`make_xyz_files_remastered.py`: lot of changes, mostly for code redability (avoiding redundency) , optimization + no more writing the npz iq and gr files in the metadata since their ID is already given in "id" = was useless

-`Make_xyz_files-remastered.ipynb`: cleaned the notebook + more documentation + now show how to create the dataframe and import it as csv

**TO DO**: fcctpbp : number of atoms still wrong !


## 20260302
## changed
-`make_xyz_files_remastered.py`: cleaning the module (adding parent class for crystal for example to avoid redundancy). I only did crystals class today. 

-`Make_xyz_files-remastered.ipynb`,`pyNMB-exmaples-sanspbcolonnes.ipynb` : put a new markdown to explain better, 

**TO DO**: verify I didn't give a wrong explanation of nLayer, nShell, and nOrder and verify the content of the sasview files !

-`platonics.py`, `utils.py`: using the old method to construct the fcc and bcc cube, the issue was in the magic numbers formulas !! for both bcc and fcc i ** 2 instead of i * 2 and adding + 1 for the bcc. Correcting the formula in magicNumbers()

-`johnsonNPs.py`: fixed doc + PEP8 + fixed some dimensions and added some new for epbpyM : 
- height is now correct for each cases (i changed the formula from : self.heightOfPyramidF*self.Rnn*(2+self.nAtomsPerEdgeOfPC) to n (self.heightOfPyramidF * self.Rnn * self.nAtomsPerEdgeOfPC * 2 - 2.84, 2.84 being dAu-Au along the height.
- new pentagonal edge length and elongation edge length after Marks truncation
- in the notebook i print all of them (edge before and after truncating)

## 20260227

-`utils.py`: reflection() was changed and RotationMol() also (back to old cause ive made mistakes)

**To do**: fix the johnson class totally (partially fixed ! just observing smaller interatomic distances along the height for big NPs = the hieght is not well measured in all the cases, however the edges are well measured) + make the doc/pep8, the catalans class (bccrDD, romuald is looking for it also), the platonic class and archimedeans based on it (cube). The rest should be working fine, still some optimizations that could be done and minor fixes (jmol script still printing, dims of new shapes = helix and double ico not accurate). 

## 20260226
## changed
-`otherNPs.py`: fixed the printing of the dimensions and updated the documentation (I think it was a bit confusing, made it clearer). Also applied syntax + doc fixes (pep8). Code should be good.

-`johnsonNPs.py`: for fcctbp class: some dimensions were false, I fixed self.heightOfBiPyramid and edgeLength().

**TO DO URGENT**: epbpyM issues, the atoms are on each other ...

-`catalans.py`: syntax + doc + restructuration (main class called CatalanNP() that is the base class for the shapes. 

**TO DO URGENT**: issue with bccrDD ! (weird structure and the number of shells is not respect, one more).
I forced a good number of shells but the structure is still odd.


## 20260225
## changed
-`archimedeans.py`, `utils.py`, : for the function calculateTruncationPlanesFromVertices(), the issue was that the last layer wasn't being removed when truncating the tetrahedron leading to a not stable tetrahedron. This was only happening to the truncated tetrahedron, the others shapes did not have this problem, even though they're all using this function in the same way. To fix it I tried robust methods that did not work, instead I just put an eps of 0.1 for cutAtVertex (quite a big eps) that makes the right truncature for every sizes (at least until 50 nm). So it's not robust, but it seems to be working fine. May need to find a most robust way. 
The documentation and syntax were fixed. Also trCube doesn't work since it's calling the cube class from the platonic module. More optimizations should be done, example using boolean masks ? 

-`platonics.py`: I tried to fix the truncated cube using the same method that the other shapes in the class: total failure.

**TO DO URGENT** : **fix the cubes and actually use the magic numbers structures** (Creating a cube without it is super easy via ASE).

## 20260224
## changed
-`archimedeans.py`: restructuration : I put a main class called ArchimedeanNP() that is a base class for the other classes to put in them commun utilities = makes the code lighter. Also changed nAtomsPerShellAnalytic(): cumulative sum now (optimization), for the truncated tetrahedron I added the inscribed sphere diameter using the general function from utilities.py. Issue: truncated tetrahedron is not truncated well (not 1/3, smaller truncature).


## 20260223
## changed
-`platonics.py`: restructuration : I put a main class called PlatonicNP() that is a base class for the other classes (regfccOh, regIco etc) to put in them commun utilities = makes the code lighter and would be good to do it for all the modules. Functions:  nAtomsPerShellAnalytic(): cumulative sum now (optimization), regIco class: added the possibility to do the double icosahedron. fccregTd(): added the possibility to make an helix of tetrahedrons (and changed the input). Also changed the documentation and syntax (google format + PEP8) **TO DO**: **change the dimensions computations in these cases (radiuses, volume are wrong now). Also see if we keep sasview_dims, maybe new optimizations ? (for coords vectorization)**
-`utils.py`: planeFittingLSF(): added the possibility to use many points (3D array) in entry, instead of one point to allow vectorization.
ISSUE: sometimes the JMOL script is printed even though there is not printing ... still couldn't figure that out


## 20260220
## changed
- `pyNMB-examples-sanspbcolonnes.ipynb`: notebook of examples: clearer explanations. TO DO: verify the english syntax, rename it wihtout sanspbcolonnes, verify if its always calling writexyz and not the old function that is not working
- `crystals.py and utils.py`:documentation changed in general, for example, i put "Target sizes"(the entries of the user) and "Measured sizes" (the final sizes) for less confusions. I also added the google format for the docstring and applied PEP8.
-`crystals.py`: MakeCylinder(): used to not work well, now should be working for any size and growth direction, MakeSpehre(): added the hollow sphere : new parameter = hollow_sphere_diameter (in nm), MakeParallelepiped(): measured sizes are only right when buildPPD = "xyz", need to the same for "abc" !! MakeEllipsoid(): vectorization for delAtom, and in propPostMake() calling defCrystalShapeForJMol() with noOutput = True because too noisy, let's just keep the script (half done since it's still printed sometimes for a reason I ignore)
So for this class, **TO DO URGENT**: dealing with "abc", see if we keep "sasview_dims" 
- `data.py`: added the tetrahedron (surface planes, energies etc) + some cif files 'Ag fcc': 'cod9008459-Ag_fcc.cif', 'CsPbBr3 ortho' : 'CsPbBr3_ortho_14608.cif', 'CsPbBr3 cubic' : 'CsPbBr3_cubic_231023.cif' (maybe gonna delete the last ones)



## 20260219
## changed
-`utils.py`: findNeighbours(), truncateAboveEachPlane(), truncateAbovePlanes(), returnPointsThatLieInPlanes(), Pt2planeSignedDistance(), 
centerofgravity(), calculateCN(), delAtomsWithCN(), sortVCW = all optimization using numpy (vectorization) or scipy, rdf() = using query ball tree instead of query ball point from scipy (in comments right now, not sure about it), planeAtVertices() = using einsum from numpy but nore sure about it too (keep it as comments), findNeighbours() = using pdist and squareform from sicpy(needs to be verified), reflection_tetra(), Rx(), Ry(), Rz() = np instead of math. 
Also the imports of numpy inside the functions were removed, now it's only in the beginning of the module.
writexyz() = using counter for the dict
The documentation and writing were also fixed (PEP8 + google style docstring)

## 20260218
*Notes: A lot of changes have been made and not committed since. Therefore, the following changes will be dense. They will mainly be on the src files and the example notebook + the HRTEM notebook.*
### changed
-`utils.py`: inscribed_circumscribed_spheres (maybe to change again, sasview dims seems weird). Chnages to many functions : 
RAB(), RBetween2points(), vector(), vectorBetween2points(), normofV(), normV(), centerOgGravity(), center2cog(), centerToVertices(), planeFittingLSF() and get_moments_of_inertia_for_size() were changed for **optimization** (basically loops transformed into vectorizations), should still work because of the new checks (if they are np arrays) and be way faster. 
reflection() was modified : new condition with eps
### added
-`utils.py`:
full_diagnostics() a diagnostic function for ASE EMT computations (which can fail in certain cases, try to find out why exactly).
rotation_around_axis_through_point() to be used in the module TEM_creator.py (when creating the structure = NP on the carbon substrate in the HRTEM images).
reflection_tetra() function for the helix of tetrahedrons
### deleted
MOI_shapes(): removed cause not working that well (working for certain shapes but not others, I will work on it locally), writexyz_generalized(name of the class)(): the functions to automatize the creation of files: are now in a new module (make_files.py)

## 20260115
in the pyNMB-examples :To do : make sure to always use pyNmbu.writexyz and not just write (because there is an extra column) 

## 20250310
### changed
For every NPs classes: changed the position of the function defCrystalShapeForJMol(), it is now in  propPostMake() and is done by default (also added the attributes if it was missing).

## 20250307
### changed
Documentation for each NPs classes and their functions

## 20250305
### added
- `Create_otherNPs_database.ipynb`: class that creates nps + cif/xyz files
- `Create_johnson_database.ipynb`: class that creates nps + cif/xyz files
- `Create_catalan_database.ipynb`: class that creates nps + cif/xyz files
- `johnsonNPs.py`: `nAtomsPerEdgeOfPC_after_truncation()` and `edgeLength_after_truncation(whichEdge)` to be specified in dictionnary in the files
### changed
-`utils.py`: all the files writing functions were changed, now the main dimensions are the radius of the insphere and cicumscribed sphere (the MOI dim used to be the main dim but some of them are not accurate), secondary dims are added (specific lengths and the number of atoms they're containing), and the truncature was also added (True or False)
## 20250227
### changed
-`archimedeanNPs.py`: changed MOI formulas, also added self.shape for each classes.

### added
-`Create_archimedean_database.ipynb`: class that creates archimedean nps + cif/xyz files, some conditions are made for the bravais lattice (only fcc). NB: not possible to make multi elements nps and trcubes aren't added for now
-`utils.py`:  `writexyz_generalized_archimedean()`: function that writes archimedean nps cif/xyz files

## 20250225
### changed
-`Create_crystals_database.ipynb`: when creating multiple NPs (and their files), their sizes depend on dhkl, it allows to have the minimum size step while never having twice the same size. The sizes are now chosen using multiple of dhkl, example for a sphere: size=[2*dhkl],[3*dhkl], etc. Also "max_size" was added in order to choose a maximum size when creating the NPs, it corresponds to the diameter of the circumscribed sphere. 
-`Create_platonic_dataset.ipynb` : Same changes adapted for the Platonic Class
`utils.py`:  `writexyz_generalized_platonic()`: function that writes platonic nps cif/xyz files

## 20250219
### changed

-`Create_crystals_database.ipynb`: nRot is now written in the name of the wires files (and the size indicator restart at 0 when nRot changes)
-`platonicNPs` : nShell, nLayer and nOrder restored, they do not mean the same thing !
-`Create_platonic_dataset.ipynb` : was adapted with dynamic variables for the intanciation of the class : nShell, nLayer, nOrder based on the form. Even if nLayer=nOrder or nShell+1, the files size indicator increases the same way no matter the form (ex : Co_fcc_regIco_0000001 = 1 bond by edge,  Co_fcc_regIco_0000002 = 2 ..etc and same for other forms)

## 20250218
### added

-`utils.py`:`create_data_csv()` : creates csv files containing only the dictionnary of the xyz files (or Iq csv files)


## 20250214

### changed
-`platonicNPs` : same name for number of shells = 'nOrdet' for all the classes to ease the generalization !MISTAKE!
### added
-`Create_platonic_dataset.ipynb` : class that creates Platonics nps + cif/xyz files, some conditions are made for the bravais lattice. NB: not possible to make multi elements nps and hollow cubes aren't added for now

## 20250212
### changed
- `Create_crystals_database.ipynb` : sizes can be defined by np.arange(initial,final,step)
- `crystalNPS.py` : all the shapes are now created from a given diameter (used to be diameter for some shapes and radius for other shapes, same descriptor is better : less confusing for user but also easier for creating multiple files)


## 20250211
### added
- 3 classes in `Create_crystals_database.ipynb`, a notebook made to create a database of xyz/cif files of Crystal nps :
    -  `Crystals_ellipsoids_parallepipeds`: finished
    -  `Crystals_spheres`: finished
    -  `Crystals_wires`: ongoing work (how to defined the planes of the wire depending on the element used)
-  `utils.py` : functions `get_crystal_type` (to find the Bravais lattice based on the n° space group), `extract_cif_info`(cif name and crystal type), `load_cif` for the 3 classes in `Create_crystals_database.ipynb`
### changed
- `crystalNPS.py` : in `loadExternalCif(self)`: a condition was added (if hasattr(self, 'cif'):return) to not load the structures from the cif files twice if they were already loaded in the classes `Crystals_ellipsoids_parallepipeds`, `Crystals_spheres` or `Crystals_wires` (gain of time)

## 20250207
### changed
-  `crystalNPS.py` : issue with MOI fixed for wulff forms that aren't predifined
### added
- `utils.py` : `writexyz_generalized_crystals` : noOutput added
- `utilsDC.py` : `create_iqfiles_from_xyzfiles` :  noOutput added 
- `Create_crystals_database` : `class PredifinedWulffFiles` :  noOutput added
Usefull when we will create a lot of files


## 20250204
### changed
- `utils.py` : new function `Inscribed_circumscribed_spheres(self,noOutput)` that calculates the inscribed/circumscribed spheres radius that can be call for each classes (instead of re writing it everytime)
- `johnsonNPS` : calling the function `Inscribed_circumscribed_spheres(self,noOutput)` 
- `UtilsDC.ipynb` : change of the functions of the graphs (mistakes were made)


## 20250203
### added 
- `utils.py` : new size/MOI formula for the wires that work for any nRot and both wires from crystals and predefined wulff forms, gives the height of the wire and the edge of the pentagon
- `size_test_MOI_real.ipynb`: verification of the sizes (height and edge) given by the new formula


## 20250131
### added 
- `cif_to_xyz.ipynb` : possibility to create  files (xyz,cif,iq) of a specific form given by the user from cif files in data.py 

## 20250127
### added 
The idea is to calculate the size from the MOI for each nps in the crystals class taking into consideration the truncated ones that need to be approximated. Also, using a function in utils to do it.

- `utlis.py`: def MOI_shapes(class_shape) function that calculates the size from the MOI (used to be in crystals)
- `data.py` : in class WulffShapes : 'MOI for size' : truncated shapes are approximated as shapes with knowns formulas for the MOI/size
- `crystalNPS.py` :  in `predefinedParameters4WulffShapes` : self.MOIshape=data.WulffShapes.WSdf['MOI for size'].loc[self.WulffShape] and in `propPostMake()` : calling the function self.pyNMBu.MOI_shapes()
- `utils.py` : writexyz_generalized_crystals to create the files for predefined wulff shapes in a good format

## 20250111
- `platonicNPs.py`: the call to `pyNMBu.defCrystalShapeForJMol` is now made in `propPostMake()` (moved from `__init__`, it was obviously wrong)
- `utils.py`: in the function `reduceHullFacets()`, `HalfspaceIntersection()` <as called with the option `qhull_options="QJ"` (joggles each input coordinate), which was not working in most cases

## 20240106
### added 
-  `crystalNPs.py` : in def prop(self,noOutput): if "Wulff" in self.shape : compute the inscribed sphere and circumscribed sphere (for pre deffined wulff form)
- `cif_to_xyz.py` : create all the  files (xyz,cif,iq) of all the wulff forms from cif files in data.py

## 20241220
### added 
- in every class of NPS (for every shapes except hollow shapes): MOI mass normalized (m of each atoms=1) : self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))`
-  `crystalNPs.py`:  main dimensions calculated from MOI normalized (in an array [d1,d2,d3] with d1>d2>d3) for : sphere, ellipsoid, cylinder, wire (for nRot=4 and nRot=6), parallepiped 
-  `platonicNPs.py`:  main dimensions calculated from MOI normalized for : regfccOh regIco  regfccTd regDD cube (not working with hollow cube)
### changed
-  `utils.py`:
      - `writexyz_generalized()`: updated function that writes xyz files (with dictionnaries and good file names with numerotation), inputs : path and instance of a class. Example of use in 2.2.6  `pyNMB-examples-sanspbcolonnes` with the use of loops
        
### changed
-  `platonicNPs.py`:
      - `class regfccOh` : `radiusCircumscribedSphere(self)` error in the formula, changed and working now
    
## 20241218
### added
-  `utils.py`:
      - `get_moments_of_inertia_for_size(self, vectors=False)` : Get the moments of inertia along the principal axes with
    mass normalisation. Units of the moments of inertia are angstrom**2
      - `moi_size(model: Atoms,noOutput: bool=False,)` : Returns the 3 moments of inertia along the principal axes with mass normalization to get acces to size informations
### changed
-  `propPostMake(self,skipSymmetryAnalyzis, thresholdCoreSurface, noOutput)`: the size of the crystals are calculated using the pNMBu.moi_size() function

## 20241211
### added
-  `utils.py`:
      - `writexyz_generalized()`: function that writes xyz files (with dictionnaries and good file names), inputs : path and instance of a class.
        Possibility to create multiple files if the instanciation of the class is in a loop (loop on the size of atoms for example), cf `pyNMB-examples-sanspbcolonnes`.
### changed
-  `utilsDC.py`:
      - `create_iqfiles_from_xyzfiles(self, path_of_xyzfiles, path_of_csvfiles)`: dictionnaries are added in the csv files


## 20241206
### added
- `utilsDC.py`:
    - imports for Debye calculations
    - functions that plot I=f(q), S=f(q), F=f(q), G=f(r)
    - functions that create multiple csv files containing scattering data calculated from xyx/cif files/ ASE object
  
- `DebyeTest.ipynb` : tests the new utilities from utilsDC.py

- `coords_test_debye` : new repository containing xyz and cif files to use as examples in DebyeTest.ipynb

- `csv_files` : new repository that will contain csv files created by user when they will try DebyeTest.ipynb



## 20240630
### added
- `utils.py`:
    - very basic rdf calculator `rdf()` for finite size compounds, *i.e.* without PBC
    - `createDir()` utility
- `SandBox-doNotDelete-dev.ipynb`: **RDF profiles** section, made to create rdf profiles for a machine learning tutorial, which will soon be added to the [pyPhysChem github repository](https://github.com/rpoteau/pyPhysChem)

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
