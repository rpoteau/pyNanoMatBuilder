import sys, os, pathlib
import re
import math
import numpy as np
import pandas as pd
import pyNanoMatBuilder.utils as pyNMBu
from pyNanoMatBuilder import data
from ase.build import bulk
from ase import io
from ase.visualize import view
from ase.build.supercells import make_supercell
from ase.geometry import cellpar_to_cell
from visualID import  fg, hl, bg
import visualID as vID

class Crystal:
    """
    A class for generating XYZ and CIF files of crystalline nanoparticles (NPs) 
    of various shapes and sizes, based on user-defined compounds (either by 
    name, e.g., "Fe bcc", or from a CIF file). The supported nanoparticle 
    shapes include:

    - Spheres
    - Ellipsoids
    - Parallelepipeds
    - Wires with different cross-sections
    - Wulff constructions: cube, octahedron, cuboctahedron, dodecahedron, 
      spheroids, and their truncated versions

    Key Features:
    - Allows to choose the NP size and shape.
    - Supports Wulff construction with customizable surface energies.
    - Enables the creation of wires with defined orientations and cross-sections.
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The symmetry analysis can be skipped to speed up computations.
    - Periodic boundary conditions (PBC) can be enabled if needed.
    - Customizable precision thresholds for structural analysis.
    """

    def __init__(self,
                 crystal: str='Au',
                 scaleDmin2: float=None,
                 setSymbols2: np.ndarray=None,
                 userDefCif: str=None,
                 shape: str='sphere',
                 MOIshape=None,
                 size: float=[2,2,2],
                 directionsPPD: np.ndarray=np.array([[1,0,0],[0,1,0],[0,0,1]]),
                 buildPPD: str='xyz',
                 directionWire: float=[0,0,1],
                 #NEW CYLINDER
                 directionCylinder: float=[0,0,1],
                 refPlaneWire: float=[1,0,0],
                 nRotWire: int=6,
                 surfacesWulff: np.ndarray=None,
                 eSurfacesWulff: np.ndarray=None,
                 sizesWulff: np.ndarray=None,
                 symWulff: bool = True,
                 jmolCrystalShape: bool=True,
                 aseSymPrec: float=1e-4,
                 pbc: bool=False,
                 threshold: float=1e-3,
                 dbFolder: str=data.pyNMBvar.dbFolder,
                 postAnalyzis: bool=True,
                 aseView: bool= False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis: bool = False,
                 noOutput: bool = False,
                 calcPropOnly: bool = False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            userDefCif (str, optional): Path to a user-defined CIF file.
            shape (str): Shape of the nanoparticle (NP). Options:
                - 'sphere'
                - 'ellipsoid'
                - 'parallelepiped'
                - 'wire'
                - 'Wulff: ...' (various Wulff constructions)
            size (list): List defining the NP size. The required number of elements depends on the shape:
                - Sphere: [diameter]
                - Ellipsoid / Parallelepiped: [size_x, size_y, size_z]
                - Wire: [cross-section diameter, length]
            directionsPPD (np.ndarray): Array defining the three directions of the parallelepiped.
            buildPPD (str): Specifies the coordinate system used to build the NP:
                - "xyz" for Cartesian coordinates.
                - "abc" for the lattice parameter system.
            directionWire (list): Growth direction of the wire (e.g., [0,0,1]).
            directionCylinder (list): Growth direction of the cylinder (e.g., [0,0,1]).
            refPlaneWire (list): Miller indices of the reference plane, which is rotated around "directionWire" to generate the wire.
            nRotWire (int): Number of sides in the wire's cross-section (i.e., number of rotations of refPlaneWire around "directionWire").
            surfacesWulff (np.ndarray, optional): Array of Miller indices defining the surfaces used in Wulff constructions (e.g., [[1,1,1], [0,0,1]]).
            eSurfacesWulff (np.ndarray, optional): Array of surface energies corresponding to the surfaces in Wulff constructions.
            sizesWulff (np.ndarray): Array defining the size of the Wulff construction (e.g., distance between truncated planes, equivalent to NP diameter).
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            aseSymPrec (float): Precision threshold for ASE symmetry calculations (default: 1e-4).
            pbc (bool): If True, applies periodic boundary conditions (PBC).
            threshold (float): Precision threshold for plane truncation (distance threshold for keeping atoms).
            dbFolder (str): Path to the database folder containing CIF files and other information (e.g., for Wulff constructions).
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.
     
        """
        self.dbFolder = dbFolder #database folder that contains cif files
        self.crystal = crystal # see list with the pyNMBu.ciflist() command
        self.shape = shape.strip(' ') # 'sphere', 'ellipsoid', 'cube', 'wire', 'Wulff', 'cylinder'
        self.MOIshape=MOIshape
        self.size = size
        self.directionsPPD = directionsPPD
        self.buildPPD = buildPPD
        self.directionWire = directionWire
        #NEW for cylinder
        self.directionCylinder = directionCylinder
        self.refPlaneWire = refPlaneWire
        self.nRotWire = nRotWire
        self.surfacesWulff = surfacesWulff
        self.eSurfacesWulff = eSurfacesWulff
        self.sizesWulff = sizesWulff
        self.symWulff = symWulff
        self.jmolCrystalShape = jmolCrystalShape
        self.aseSymPrec = aseSymPrec
        self.pbc = pbc
        self.threshold = threshold
        self.nAtoms = 0
        self.cif = None
        self.cifname = None
        self.userDefCif = userDefCif
        self.trPlanes = None
        self.dim=[0,0,0] # main dimensions for files 
        if "Wulff" in self.shape :
            self.WulffShape = self.shape.split(":")
            if len(self.WulffShape) == 2:
                self.WulffShape = self.WulffShape[1].lstrip(' ') #removes 'Wulff:'
                self.shape="Wulff: " + self.WulffShape #normalizes the name of the Wulff shape
            else:
                self.WulffShape = None

        if self.userDefCif is not None: self.loadExternalCif()
        
        # if self.shape in data.pyNMBimg.IMGdf.index:
        #     self.imageFile = pyNMBu.imageNameWithPathway(data.pyNMBimg.IMGdf["png file"].loc[self.shape])
        # else:
        #     sys.exit("Shape {self.shape} is unknown")
        if not noOutput: vID.centerTitle(f"{self.crystal} {self.shape}")

        self.bulk(noOutput)
        if scaleDmin2 is not None: pyNMBu.scaleUnitCell(self,scaleDmin2,noOutput=noOutput)
        if setSymbols2 is not None: self.cif.set_chemical_symbols(setSymbols2)
        if aseView: view(self.cif)
         
        
        if not calcPropOnly:
            self.makeNP(noOutput)
            if aseView: 
                view(self.sc)
                view(self.NP)
            if postAnalyzis:
                self.prop(noOutput)
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
        # self.NP.center() #must be done before the surface calculation, since the Hull analysis assulmes that the structure is centered in [0,0,0]
          
    def __str__(self):
        return(f"Crystal = {self.crystal} {self.shape}")

    def loadExternalCif(self):
        """
        Load an external CIF file and extract the crystal name.
        This method checks if a CIF file is already loaded to avoid redundant loading.
        The method extracts the crystal name from the CIF file using predefined CIF tags.
        Raises:
            SystemExit: If the CIF file is not found.


        """
        if hasattr(self, 'cif'):  
            return  #Check if `self.cif` already exists to prevent reloading.
        else :
            self.cif = io.read(self.userDefCif)
        path2extCif = pathlib.Path(self.userDefCif)
        if not path2extCif.exists():
            sys.exit(f"file {self.userDefCif} not found. Check the file name or its location")
        cifFile =  open(self.userDefCif, 'r')
        #Search for specific CIF tags that contain the crystal name:
        name1 = "_chemical_name_systematic"
        name2 = "_chemical_formula_sum"
        name3 = "_chemical_formula_moiety"
        cifFileLines = cifFile.readlines()
        re_name_systematic = re.compile(name1)
        re_name_sum = re.compile(name2)
        re_name_moiety = re.compile(name3)
        crystal1 = None
        crystal2 = None
        crystal3 = None
        for line in cifFileLines:
            if re_name_systematic.search(line):
                parts = line.split()
                crystal1 = ' '.join(parts[1:])
            if re_name_sum.search(line):
                parts = line.split()
                crystal2 = ' '.join(parts[1:])
            if re_name_moiety.search(line):
                parts = line.split()
                crystal3 = ' '.join(parts[1:])
        cifFile.close()
        if crystal1 is not None:
            self.crystal = crystal1
        elif crystal3 is not None:
            self.crystal = crystal3
        elif crystal2 is not None:
            self.crystal = crystal2
        else: self.crystal = "unknown"



            
    def bulk(self, noOutput):
        """
        Retrieve bulk parameters for the crystal structure.
        This function uses either:
        - a CIF file provided by the user (`self.userDefCif`).
        - a CIF file stored in the internal database (`pyNanoMatBuilder`).
        Args:
            noOutput (bool): If False, details are printed.
    
        Raises:
            SystemExit: If the crystal is not found in the database and no external CIF file is provided.
        """
        if self.userDefCif is None:
            path2cif = os.path.join(pyNMBu.pyNMB_location(),self.dbFolder) #If no external CIF given, searches for a matching CIF file in the internal database.
            # if not noOutput: vID.centertxt(f"List of stored cif files in pyNanoMatBuilder ('{self.dbFolder}' folder)",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
            # if not noOutput: display(data.pyNMBcif.CIFdf)
            if  self.crystal.upper() in data.pyNMBcif.CIFdf.index.str.upper():
                dftmp = pd.DataFrame(index=data.pyNMBcif.CIFdf.index.copy())
                data.pyNMBcif.CIFdf.index = data.pyNMBcif.CIFdf.index.str.upper()
                self.cifname = data.pyNMBcif.CIFdf["cif file"].loc[self.crystal.upper()]
                data.pyNMBcif.CIFdf.index = dftmp.index.copy() # revert. Ugly, but I do not know how to manage it differently
            else:
                if noOutput: display(data.pyNMBcif.CIFdf) #because the databse has not been displayed just above
                sys.exit(f"{fg.RED}{bg.LIGHTREDB}The database does not contain bulk parameters for the '{self.crystal}' crystal.\n"\
                         f"Please provide a cif file{fg.OFF}")
            self.cif = io.read(os.path.join(path2cif,self.cifname))
        else:
            self.cif = io.read(self.userDefCif)
            path2extCif = pathlib.Path(self.userDefCif)
            self.cifname = pathlib.Path(*path2extCif.parts[-1:])

        
        pyNMBu.returnUnitcellData(self)
        if not noOutput: print(f"cif parameters for {self.crystal} found in {self.cifname}")
        return 

    def makeSuperCell(self,noOutput):
        """
        Creates a supercell based on the defined nanoparticle shape and size.
        Args:
            noOutput (bool): If False, details are printed.
        
        The function determines the appropriate supercell dimensions based on the particle shape and size,
        then constructs and translates the supercell to be centered at the origin.
        """
        
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        if not noOutput: vID.centertxt(f"Making a multiple cell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        extendSizeByFactor = 1.06 # Small extension factor to ensure sufficient cell size

        # Determine supercell dimensions based on the nanoparticle shape
        if (self.shape == 'sphere'):
            sphereRadius = self.size[0]/2
            Ma = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'ellipsoid' or self.shape == 'supercell' or self.shape == 'parallepiped'):
            Ma = int(np.round(extendSizeByFactor * self.size[0]*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * self.size[1]*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * self.size[2]*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'wire'):
            maxDim = np.max(self.size)*10*1.5
            Ma = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[2]))
        #NEW CYLINDER
        elif (self.shape == 'cylinder'):
            cylinderRadius = self.size[0]
            half_height=self.size[1]
            Ma = int(np.round(extendSizeByFactor * cylinderRadius*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * cylinderRadius*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * half_height*10/self.cif.cell.lengths()[2]))
        elif ('Wulff' in self.shape):
            if np.argmax(self.sizesWulff) == 1:
                maxDim = self.sizesWulff[0]*10*1.5
                print('maxDim',maxDim)
            else:
                maxDim = np.max(self.sizesWulff)*10*1.5
            # print(f"{maxDim = }")
            Ma = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[2]))
            
        # Define minimum supercell size (at least 20 Å per direction)
        Ma1nm =  int(np.round(20/self.cif.cell.lengths()[0]))
        Mb1nm =  int(np.round(20/self.cif.cell.lengths()[1]))
        Mc1nm =  int(np.round(20/self.cif.cell.lengths()[2]))
        Ma1nm = min(Ma1nm,Ma)
        Mb1nm = min(Mb1nm,Mb)
        Mc1nm = min(Mc1nm,Mc)
        if not noOutput: print(f"First making a {Ma1nm}x{Mb1nm}x{Mc1nm} supercell")
        
        # Generate the initial supercell
        M1nm = [[Ma1nm, 0, 0], [0, Mb1nm, 0], [0, 0, Mc1nm]]
        sc1nm=make_supercell(self.cif,M1nm)
        
        # Scale up the supercell size (find the nearest even numbers)
        Ma = Ma/Ma1nm
        Mb = Mb/Mb1nm
        Mc = Mc/Mc1nm
        Ma = math.ceil(Ma / 2.) * 2
        Mb = math.ceil(Mb / 2.) * 2
        Mc = math.ceil(Mc / 2.) * 2
        if not noOutput: print(f"Making a {Ma}x{Mb}x{Mc} supercell of the supercell")
        if not noOutput: print(f"       = {Ma*Ma1nm}x{Mb*Mb1nm}x{Mc*Mc1nm} supercell")
        M = [[Ma, 0, 0], [0, Mb, 0], [0, 0, Mc]]
        sc=make_supercell(sc1nm,M)
        # print(cif.cell.cellpar())
        # print(cellpar_to_cell(cif.cell.cellpar()))
        # print(sc.cell.cellpar())
        # print(cellpar_to_cell(sc.cell.cellpar()))
        V = cellpar_to_cell(sc.cell.cellpar())
        
        # Center the supercell at the origin
        if not noOutput: print(f"Center of Mass:", [f"{c:.2f}" for c in sc.get_center_of_mass()]," Å")
        if not noOutput: print("Now translating the supercell to O")
        #sc.center(about=(0.0,0.0,0.0))
        sc.translate(-V[0]/2)
        sc.translate(-V[1]/2)
        sc.translate(-V[2]/2)
        if not noOutput: print(f"Center of Mass after translation of the supercell: {sc.get_center_of_mass()} Å")
            
        # Store the final supercell and print atom count
        self.sc = sc.copy()
        nAtoms=len(self.sc.get_positions())
        if not noOutput: print(f"Total number of atoms = {nAtoms}")
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
    def makeSphere(self,noOutput):
        """
        Create a spherical nanoparticle (NP) by removing atoms outside the defined radius (calculated from the diameter in nm).  
        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput: vID.centertxt(f"Removing atoms to make a sphere",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        
        # Get the center of mass of the structure
        com = self.sc.get_center_of_mass()
        
        # Compute sphere radius from the provided diameter
        sphereRadius = self.size[0]/2
        
        # Identify atoms to delete (atoms with a distance greater than the radius)
        delAtom = []
        for atomCoord in self.sc.positions:
            delAtom.extend(pyNMBu.Rbetween2Points(com,atomCoord)/10 > [sphereRadius])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        # Real sizes
        positions = self.NP.get_positions()
        mins = np.min(positions, axis=0)  #  Min coordinates
        maxs = np.max(positions, axis=0)  # Max coordinates
        a_real = (maxs[0]-mins[0])/2  # Semi-axis a               In theory a=b=c but in practice not really

        
        self.spheres_axes = a_real 
        print('self.spheres_axes', self.spheres_axes)
    def makeEllipsoid(self,noOutput):
        """
        Create an ellipsoidal nanoparticle (NP) by removing atoms outside the defined ellipsoid.
        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput: vID.centertxt(f"Removing atoms to make an ellipsoid",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()

        # Get the center of mass of the structure
        com = self.sc.get_center_of_mass()
        
        # Convert the provided size from a diameter in nm to a radius in Ångström for each axis
        size = (np.array(self.size)/2)*10 
        
        def outside(coord,com,size):
            """
            Calculate if an atom is outside the ellipsoid.
            Args:
                coord (array-like): Atom position coordinates.
                com (array-like): Center of mass coordinates.
                size (array-like): Half sizes of the ellipsoid along each axis.
    
            Returns:
                bool: True if the atom is outside the ellipsoid, False otherwise.
            """
            return (coord[0]-com[0])**2/(size[0])**2+(coord[1]-com[1])**2/(size[1])**2+(coord[2]-com[2])**2/(size[2])**2

        # Identify atoms to delete (atoms outside the ellipsoid)
        delAtom = []
        for atom in self.sc.positions:
            delAtom.extend([outside(atom,com,size) > 1])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        # Real sizes
        positions = self.NP.get_positions()
        mins = np.min(positions, axis=0)  #  Min coordinates
        maxs = np.max(positions, axis=0)  # Max coordinates
        a_real = (maxs[0]-mins[0])/2  # Semi-axis a
        b_real = (maxs[1]-mins[1])/2  # Semi-axis b
        c_real = (maxs[2]-mins[2])/2  # Semi-axis c
        
        self.ellipsoid_axes = [a_real, b_real, c_real]  

    def makeWire(self,noOutput):
        """
        Create a nanowire by truncating atoms based on a defined reference plane and a growth direction.
        Process:
        - If no reference plane is provided, it calculates a default reference plane parallel to the wire's growth direction.
        - It computes the normal of the reference plane and the planes of rotation for the wire cross-section.
        - The wire is defined by a radius and length along the growth direction.
        - The atoms outside the wire's shape are removed, and the resulting wire is moved to the center of the unit cell.
    
        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput: vID.centertxt(f"Removing atoms to make a wire",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()

        # Calculate reference plane if not provided by the user
        if self.refPlaneWire is None: self.refPlaneWire = pyNMBu.returnPlaneParallel2Line(self.directionWire,[1,0,0],debug=False)
            
        # Get the normal of the reference plane    
        normal = pyNMBu.normal2MillerPlane(self,self.refPlaneWire,printN=not noOutput)
        
        # Generate the planes for rotation based on the normal and the wire growth direction
        trPlanes = pyNMBu.planeRotation(self,normal,self.directionWire,self.nRotWire,debug=False,noOutput=noOutput)

        # Normalize each rotated plane
        for i,p in enumerate(trPlanes):
            trPlanes[i] = pyNMBu.normV(p)
            
        # Define the wire's radius (scaled from size input)   
        radius = 10*self.size[0]/2
        tradius = np.full((self.nRotWire,1),-radius)
        trPlanes = np.append(trPlanes,tradius,axis=1)

        # If periodic boundary conditions aren't used
        if not self.pbc:
            halfLength = 10*self.size[1]/2 # Half the length of the wire
            ptop = np.append(pyNMBu.normV(self.directionWire),-halfLength)  # Top plane of the wire
            pbottom = np.append(-pyNMBu.normV(self.directionWire),-halfLength) # Bottom plane of the wire
            # Add the top and bottom planes to the list of planes
            trPlanes = np.append(trPlanes,ptop)
            trPlanes = np.append(trPlanes,pbottom)
            trPlanes = np.reshape(trPlanes,(self.nRotWire+2,4))

        # Identify atoms that lie above the defined planes (and should be removed)
        AtomsAbovePlanes = pyNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,eps=self.threshold,noOutput=noOutput)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        self.trPlanes = trPlanes
        if not noOutput: vID.centertxt(f"Nanowire moved to the center of the unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        # self.NP.center()
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
        

    def makeCylinder(self,noOutput): #in def init, the two entries are specified : size=[diameter,length] and directionCylinder=[h,k,l]
        """
        Create a cylindrical nanoparticle (NP) by removing atoms outside the defined radius (calculated from the diameter in nm).  
        Args:
            noOutput (bool): If False, details are printed.
      
        """
    
        if not noOutput: vID.centertxt(f"Removing atoms to make a cylinder",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        
        # Define the cylinder's dimensions (scaled from size input)  
        radius = 10*self.size[0]/2 
        radius_squared=radius**2
        half_height = 10 * self.size[1] / 2
        axis=np.array(self.directionCylinder)
        com = self.sc.get_center_of_mass()
        #delAtom = []
        
        # Identify atoms to delete (atoms outside the cylinder)
        delAtom = [i for i, pos in enumerate(self.sc.positions) if pyNMBu.isnt_inside_cylinder(pos,radius, radius_squared, half_height)]
        
        # Rotate the coordinates to th [0,0,1] orientation
        self.sc.positions= pyNMBu.rotateMoltoAlignItWithAxis(self.sc.positions,axis,targetAxis=np.array([0, 0, 1]))
        self.NP = self.sc.copy()
        del self.NP[delAtom]
    
        if not noOutput: vID.centertxt(f"Nanocylinder moved to the center of the unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        # self.NP.center()
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
       
      
    def makeParallelepiped(self,noOutput):
        """
        Creates a parallelepiped-shaped nanoparticle (NP) by truncating atoms based on specified directions.
        Process:
        - If the `buildPPD` attribute is set to "xyz", the specified directions are used directly. 
        Otherwise, the normal vectors for the directions are calculated, and a lattice transformation is applied.
        - The parallelepiped shape is defined by six planes, and atoms outside the shape are removed.
    
        Args:
            noOutput (bool): If False, details are printed.

        """
        if not noOutput: vID.centertxt(f"Removing atoms to make a parallelepiped",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        
        
        if self.buildPPD == "xyz":
            trPlanes = self.directionsPPD  
        
        # If not 'xyz', calculate the normal vectors for each direction
        else:
            normal = []
            for d in self.directionsPPD:
                normal.append(pyNMBu.normal2MillerPlane(self,d,printN=not noOutput))
            trPlanes = pyNMBu.lattice_cart(self,normal,Bravais2cart=True,printV=not noOutput) # Project from the Bravais basis to the cartesian coordinate system

        # Normalize each of the planes
        for i,p in enumerate(trPlanes): trPlanes[i] = pyNMBu.normV(p)
            
       
        # Define the parallepiped's dimensions (scaled from size input) 
        size = -np.array(self.size)*10/2 #nm!
        size = np.append(size,size,axis=0)
        
        # Define the 6 planes to truncate
        # [-a/2 direction, a/2 direction], [-b/2 direction, b/2 direction], [-c/2 direction, c/2 direction]
        trPlanes = np.append(trPlanes,-trPlanes,axis=0)
        trPlanes = np.append(trPlanes,size.reshape(6,1),axis=1)
        
        # Identify atoms that lie outside the defined parallelepiped 
        AtomsAbovePlanes = pyNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,eps=self.threshold,debug=False,noOutput=noOutput)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        #self.NP.center()
        self.trPlanes = trPlanes
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
  
    def makeWulff(self,noOutput):
        """
        Calculate truncation distances for Wulff shapes.
    
        This function determines the truncation planes of a nanoparticle based on 
        the provided surfaces and their corresponding energies. It then removes 
        atoms that are above these planes, effectively truncating the nanoparticle.
    
        Parameters:
        - noOutput (bool): If True, suppresses output messages.
    
        Attributes Updated:
        - self.trPlanes (numpy array): Stores the normalized truncation planes and 
          their respective truncation distances.
        - self.NP (ASE Atoms object): The truncated nanoparticle.
        
        """
        if not noOutput: vID.centertxt(f"Calculating truncation distances",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        trPlanes = [] # List to store truncation planes
        if self.eSurfacesWulff is None: sizes = []
        if self.eSurfacesWulff is not None: 
            sizes = []
            eSurf = []

        # Loop over surface planes to compute truncation
        for i,p in enumerate(self.surfacesWulff):
            if self.symWulff:
                symP = self.ucSG.equivalent_lattice_points(p)
                normal = []
                for sp in symP:
                    normal.append(pyNMBu.normal2MillerPlane(self,sp,printN=not noOutput))
                trPlanes += list(normal)
                if self.eSurfacesWulff is None: sizes.append(len(symP)*[self.sizesWulff[i]])
                if self.eSurfacesWulff is not None: eSurf += (len(symP)*[self.eSurfacesWulff[i]])
            else:
                trPlanes.append(pyNMBu.normal2MillerPlane(self,p,printN=not noOutput))
                if self.eSurfacesWulff is None: sizes.append(self.sizesWulff[i])
                if self.eSurfacesWulff is not None: eSurf.append(self.eSurfacesWulff[i])
        trPlanes = np.array(trPlanes)
        # Convert Miller indices to Cartesian coordinates
        trPlanes = pyNMBu.lattice_cart(self,trPlanes,Bravais2cart=True,printV=not noOutput)
        # Normalize the vectors
        for i,p in enumerate(trPlanes): trPlanes[i] = pyNMBu.normV(p)
        # print("inside makeWulff, just after trPlanes normalization. ",trPlanes.tolist())
        if self.eSurfacesWulff is None: 
            sizes = -np.array(sizes)*10/2
            trPlanes = np.append(trPlanes,sizes.reshape(len(trPlanes),1),axis=1)
        
        # If energy surfaces are used for truncation
        else:
            mostStableE = min(eSurf) # Find the most stable energy surface
            for i, e in enumerate(eSurf):
                sizes.append(-self.sizesWulff[0]*10*e/2/mostStableE)
            sizes = np.array(sizes)
            trPlanes = np.append(trPlanes,sizes.reshape(len(trPlanes),1),axis=1)
        # print("inside makeWulff, just after trPlanes size calculation. ",trPlanes)
        # print("sizes = ",sizes)

        # Remove atoms above the truncation planes
        AtomsAbovePlanes = pyNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,allP=False,\
                                                     eps=self.threshold,debug=False,noOutput=noOutput)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        # self.NP.center()
        self.trPlanes = trPlanes
        #print(' self.trPlanes', self.trPlanes)
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    def makeNP(self,noOutput):
        """
        Generate a nanoparticle (NP) of a specified shape.
    
        This function constructs different types of nanoparticles based on the user-defined shape
        and size. It supports spheres, ellipsoids, parallelepipeds, wires, cylinders, and Wulff 
        constructions. 
    
        Parameters:
        - noOutput (bool): If True, suppresses output messages.
    
        Attributes Updated:
        - self.NP (ASE Atoms object): The created nanoparticle.
        - self.nAtoms (int): The number of atoms of the nanoparticle.
        - self.cog (array): The center of mass of the nanoparticle.
        - self.trPlanes (numpy array): Truncation planes.
        - self.jMolCS (object): JMol visualization object (if enabled).
    
        Notes:
        - Calls specific shape generation methods depending on the requested nanoparticle type.
        - Uses a supercell as a starting point for most shapes.
        - Checks validity for Wulff construction parameters.
        """
        import os
        if not noOutput: vID.centertxt("Builder",bgc='#007a7a',size='14',weight='bold')
        # Default size if not provided
        
        if self.size is None: 
            self.length = [2,2,2]
            if not noOutput: print(f"length parameter set up as = {self.size} nm")
                
        # Construct different nanoparticle shapes        
        if self.shape == "sphere":
            if not noOutput: print(f"Sphere radius = {self.size[0]/2} nm")
            self.makeSuperCell(noOutput)
            self.makeSphere(noOutput)
        elif self.shape == "ellipsoid":
            if not noOutput: print(f"Ellipsoid radii = {self.size[0]/2} {self.size[1]/2} {self.size[2]/2} nm")
            self.makeSuperCell(noOutput)
            self.makeEllipsoid(noOutput)
        elif self.shape == "parallepiped":
            if not noOutput: print(f"Parallepiped side length = {self.size} nm, directions = {list(self.directionsPPD)}")
            self.makeSuperCell(noOutput)
            self.makeParallelepiped(noOutput)
        elif self.shape == "supercell":
            if not noOutput: print(f"Supercell side length = {self.size} nm")
            if len(self.size) != 3: sys.exit("Please enter lengths along a,b and c axis, i.e. size=[l_a,l_b,l_c]")
            self.makeSuperCell(noOutput)
        elif self.shape == "wire":
            if not noOutput: print(f"Wire in the {self.directionWire} directionWire. Length x width = {self.size[1]} x {self.size[0]} nm")
            if not noOutput: print(f"Reference plane = {self.refPlaneWire}, {self.nRotWire}-th order rotation around {self.directionWire}")
            if not pyNMBu.isPlaneParrallel2Line(self.refPlaneWire, self.directionWire):
                print(f"{bg.DARKREDB}Warning! The reference truncation plane is not parallel to {self.directionWire}. Are you sure?{fg.OFF}")
                suggestedPlane = pyNMBu.returnPlaneParallel2Line(self.directionWire)
                print(f"Among other possibilities, you can try {suggestedPlane}")            
            else:
                if not noOutput: print(f"{bg.LIGHTGREENB}The reference truncation plane is parallel to {self.directionWire}{fg.OFF}")
            self.makeSuperCell(noOutput)
            self.makeWire(noOutput)

        #NEW CYLINDER
        elif self.shape == "cylinder":  
            if not noOutput: print(f"Cylinder in the {self.directionCylinder} directionCylinder. Length x width = {self.size[1]} x {self.size[0]} nm")
            self.makeSuperCell(noOutput)
            self.makeCylinder(noOutput)
        elif "Wulff" in self.shape :
            if self.WulffShape is not None:
                self.predefinedParameters4WulffShapes(noOutput)
            # Ensure necessary parameters are defined
            if self.surfacesWulff == None:
                sys.exit("Wulff construction requested, but no planes were given. Define them with the 'surfacesWulff' variable")
            if self.eSurfacesWulff == None and self.sizesWulff == None: 
                sys.exit("Either 'eSurfacesWulff' or 'sizesWulff' variables must be set up")
            if len(self.surfacesWulff) != len(self.eSurfacesWulff) and len(self.surfacesWulff) != len(self.sizesWulff):
                sys.exit("'surfacesWulff' and ('eSurfacesWulff' or 'sizesWulff') lists have different dimensions")
            self.makeSuperCell(noOutput)
            self.makeWulff(noOutput)
            
        # Final attributes update    
        self.nAtoms=len(self.NP.get_positions())
        # self.NP.center(about=(0.0,0.0,0.0))
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)

        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")

    def predefinedParameters4WulffShapes(self,noOutput):
        """
        Assign pre-defined parameters for Wulff shapes.
    
        This function retrieves pre-defined properties (such as truncation planes, symmetry, 
        and moments of inertia) for Wulff shapes from the `WulffShapes.WSdf` dataset. 
        It also ensures that the selected shape is compatible with the Bravais lattice 
        of the crystal structure.
    
        Parameters:
        - noOutput (bool): If True, suppresses output messages.
    
        Attributes Updated:
        - self.eSurfacesWulff (list): Relative surface energies for the Wulff shape.
        - self.surfacesWulff (list): Miller indices of the truncation planes.
        - self.symWulff (bool): Whether symmetry is applied to the Wulff shape.
        - self.MOIshape (str): Moment of inertia shape identifier for size estimation.
    
        Notes:
        - If the lattice system does not match the expected one, a warning is displayed.
        """

        # if not noOutput: vID.centertxt("List of pre-defined Wulff shapes in pyNanoMatBuilder",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        # if not noOutput: display(data.WulffShapes.WSdf)
        if self.WulffShape in data.WulffShapes.WSdf.index:
            self.eSurfacesWulff = data.WulffShapes.WSdf["relative energies"].loc[self.WulffShape]
            self.surfacesWulff = data.WulffShapes.WSdf["planes"].loc[self.WulffShape]
            self.symWulff = data.WulffShapes.WSdf["apply symmetry"].loc[self.WulffShape]
            self.MOIshape=data.WulffShapes.WSdf['MOI for size'].loc[self.WulffShape]
            if not noOutput: 
                print(f"{hl.BOLD}Selected shape{hl.OFF}")
                display(data.WulffShapes.WSdf.loc[self.WulffShape])
                if data.WulffShapes.WSdf["lattice system"].loc[self.WulffShape] != self.ucBL.lattice_system:
                    print(f"{bg.DARKREDB}The expected lattice system of this Wulff shape is ",
                          f"{data.WulffShapes.WSdf['lattice system'].loc[self.WulffShape]}.\n",
                          f"It does not correspond to the {self.ucBL.lattice_system} cif lattice. The Wulff pre-defined [h,k,l] indexes are meaningless.\n",
                          f"{fg.RED}{bg.LIGHTREDB}I hope you know what you are doing{bg.OFF}")
                else:
                    print(f"{bg.LIGHTGREENB}The lattice system of this Wulff shape ",
                          f"({data.WulffShapes.WSdf['lattice system'].loc[self.WulffShape]}) matches with ",
                          f"the {self.ucBL.lattice_system} cif lattice{bg.OFF}")
        else:
            display(data.WulffShapes.WSdf)
            sys.exit(f"{fg.RED}{bg.LIGHTREDB}The {self.WulffShape} Wulff shape is not a pre-defined shortcut "\
                    "in the WSdf dataframe (see data.py and the index column of the table just above){fg.OFF}")
  
   
    def prop(self,noOutput):
        """
        Display unit cell and nanoparticle properties.
    
        Parameters:
        - noOutput (bool): If True, suppresses output messages.
    
        Notes:
        - Calls utility functions to print ASE unit cell properties.
        """
        #pyNMBu.plotImageInPropFunction(self.imageFile)
        if not noOutput : #added
            vID.centertxt("Unit cell properties",bgc='#007a7a',size='14',weight='bold')
            pyNMBu.print_ase_unitcell(self)
            vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
            print(self)
           

    def propPostMake(self,skipSymmetryAnalyzis, thresholdCoreSurface, noOutput): 
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), the inscribed and cicumscribed spheres, 
        analyzes symmetry (if required), and identifies core and surface atoms.
    
        Parameters:
        - skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
        - thresholdCoreSurface (float): Threshold to distinguish core and surface atoms.
        - noOutput (bool): If True, suppresses output messages.
    
        Attributes Updated:
        - self.moi (array): Moment of inertia tensor.
        - self.moisize (array): Normalized moments of inertia.
        - self.MOIshape (str): Shape identifier used for size calculations.
        - self.vertices, self.simplices, self.neighbors, self.equations (arrays): 
          Geometric properties of the nanoparticle.
        - self.NPcs (Atoms object): Copy of the nanoparticle with surface atoms visually marked.
        - self.NP (Atoms object): Original nanoparticle.
        """
        # size from MOI
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        if not "Wulff" in self.shape :
            self.MOIshape=self.shape  
        if not self.MOIshape==None :
            pyNMBu.MOI_shapes(self, noOutput)
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms = pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        # print('self.equations',self.equations)
        
        # Generate JMol visualization if enabled
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        #inscribed sphere and circumscribed
        pyNMBu.Inscribed_circumscribed_spheres(self,noOutput)
 