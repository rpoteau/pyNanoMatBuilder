from visualID import fg, hl, bg
import visualID as vID

import sys
import numpy as np
import pyNanoMatBuilder.utils as pyNMBu

import ase
from ase.build import bulk
from ase import io
from ase.visualize import view

###########################################################################################################
class CatalanNP:
    """Base class for all Catalan nanoparticles providing common functionality."""

    def propPostMake(self, skipSymmetryAnalyzis, thresholdCoreSurface, noOutput):
        """Compute and store various post-construction properties of the nanoparticle.

        This function calculates moments of inertia (MOI), the inscribed and 
        circumscribed sphere diameters,analyzes symmetry,
        generates a JMOL script, and identifies core and surface atoms.

        Args:
            skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
            thresholdCoreSurface (float): Threshold to distinguish core and surface atoms.
            noOutput (bool): If True, suppresses output messages.
    
        Attributes Updated:
             moi (array): Moment of inertia tensor.
             moisize (array): Normalized moments of inertia.
             vertices, simplices, neighbors, equations (arrays):
                  Geometric properties of the nanoparticle.
             NPcs (Atoms object): Copy of the nanoparticle with surface atoms visually marked.
             NP (Atoms object): Original nanoparticle.
        """
        
        self.moi = pyNMBu.moi(self.NP, noOutput)
        self.moisize = np.array(pyNMBu.moi_size(self.NP, noOutput))  # MOI mass normalized (m of each atoms=1)

        if not skipSymmetryAnalyzis:
            pyNMBu.MolSym(self.NP, noOutput=noOutput)

        [self.vertices, self.simplices, self.neighbors, self.equations], surfaceAtoms = \
            pyNMBu.coreSurface(self, thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102  # Nobelium, because it has a nice pinkish color in jmol
        self.surfaceatoms = self.NPcs[surfaceAtoms]

        pyNMBu.Inscribed_circumscribed_spheres(self,noOutput)

        if hasattr(self, 'jmolCrystalShape') and self.jmolCrystalShape:
            self.jMolCS = pyNMBu.defCrystalShapeForJMol(self, noOutput=True)  # do not print the jmol script



class bccrDD(CatalanNP):
    """
    A class for generating XYZ and CIF files of rhombic dodecahedral nanoparticles (NPs) 
    of various sizes, based on user-defined compounds (either by 
    name, e.g., "Fe", "Au", etc). 

    Key Features:
    - Allows to choose the NP size.
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The symmetry analysis can be skipped to speed up computations.
    - Customizable precision thresholds for structural analysis.
    """
    nFaces = 12
    nEdges = 24
    nVertices = 14
    edgeLengthF = 1
    radiusCSF3rdOrderV = 1
    radiusCSF4thOrderV = 2*np.sqrt(3)/3
#    interShellF = np.sqrt(2*(1-1/np.sqrt(5)))
    radiusISF = np.sqrt(6) / 3
    radiusMSF = 2*np.sqrt(2) / 3
    interShellF3 = 1/radiusCSF3rdOrderV
    interShellF4 = 1/radiusCSF4thOrderV
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nShell: int=1,
                 postAnalyzis: bool=True,
                 aseView: bool=False,
                 thresholdCoreSurface: float=1.,
                 skipSymmetryAnalyzis: bool=False,
                 jmolCrystalShape: bool=True,
                 noOutput: bool=False,
                 calcPropOnly: bool=False,
                ):

        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nShell (int): Number of shells (e.g., `nShell=1` means 2 atoms per edge).
            postAnalyzis (bool): If True, prints additional NP information.
                Examples: cell parameters, moments of inertia,
                inscribed/circumscribed sphere diameters, etc.
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation.
                This is the distance threshold used to decide which atoms are retained.
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.shape (str): Shape 'bccrDD'
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerShell (list): Number of atoms in each shell.
            self.interShellDistance3 (float): Distance between shells at order 3.
            self.interShellDistance4 (float): Distance between shells at order 4.
            self.cog (np.array): Center of gravity of the NP.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.imageFile (str): Path to a reference image.
            

        """
        
        self.element = element
        self.shape='bccrDD'
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance3 = self.Rnn / self.interShellF3
        self.interShellDistance4 = self.Rnn / self.interShellF4
        self.cog = np.array([0., 0., 0.])
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("bccrdd-C.png")
        
        if not noOutput: vID.centerTitle(f"{nShell} shells bcc rhombic dodecahedron ")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Bcc rhombic dodecahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of an bcc rhombic dodecahedron of size i"""
        return 4*i**3 + 6*i**2 + 4*i + 1
    
    def nAtomsPerShellAnalytic(self):
        """
        Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer, 
        computes the number of atoms for the given shell, and subtracts 
        the cumulative sum of the previous shells to get the number of new 
        atoms in the current shell.

        Returns:
            list: A list where each element represents the number of atoms 
                  in a specific shell.
        """
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni-Sum)
        return n
    
    def nAtomsPerShellCumulativeAnalytic(self):
        """
        Computes the cumulative number of atoms up to each shell.

        This function returns the total number of atoms present in the 
        nanoparticle for each shell layer, building up cumulatively.

        Returns:
            list: A list where each element represents the total number of 
                  atoms present up to that shell.
        """
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n
    
    def nAtomsAnalytic(self):
        """
        Computes the total number of atoms in the nanoparticle.
        """
        n = self.nAtomsF(self.nShell)
        return n
    
    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .

        The edge length is determined based on the interatomic distance (Rnn) 
        and the number of shells (`nShell`).
        """
        return self.Rnn*self.nShell

    def radiusCircumscribedSphere3(self):
        """
        Computes the radius of the circumscribed sphere at 3th order of the nanoparticle in Å.
        """
        return self.radiusCSF3rdOrderV*self.edgeLength()

    def radiusCircumscribedSphere4(self):
        """
        Computes the radius of the circumscribed sphere at 4th order of the nanoparticle in Å.
        """
        return self.radiusCSF4thOrderV*self.edgeLength()

    def radiusMidSphere(self):
        """
        Computes the radius of the midsphere of the nanoparticle in Å.
        The midsphere is a sphere that touches the edges of the nanoparticle.
        """
        return self.radiusMSF*self.edgeLength()

    def radiusInscribedSphere(self):
        """
        Computes the radius of the inscribed sphere of the nanoparticle in Å  .
        """
        return self.radiusISF*self.edgeLength()

    def area(self):
        """
        Computes the surface area of the nanoparticle in square Ångströms.
        """
        el = self.edgeLength()
        return 8 * np.sqrt(2) * el**2
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return 16 * np.sqrt(3)/9 * el**3
    
    def MakeVertices(self,i):
        """
        Generates the coordinates of the vertices, edges, and faces 
        for the ith shell of the dodecahedral nanoparticle.
        Args:
            - i (int): Index of the shell
        Returns:
            - CoordVertices (np.ndarray): The 14 vertex coordinates of the ith shell of an icosahedron
            - edges (np.ndarray): Indexes of the 24 edges
            - faces (np.ndarray): Indexes of the 12 faces 
        """
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell = {self.nShell}")
        else:
            scale3 = self.interShellDistance3 * i
            scale4 = self.interShellDistance4 * i
            CoordVertices = [ pyNMBu.vertex( 1, 1, 1, scale3),\
                              pyNMBu.vertex( 1,-1, 1, scale3),\
                              pyNMBu.vertex(-1, 1, 1, scale3),\
                              pyNMBu.vertex(-1,-1, 1, scale3),\
                              pyNMBu.vertex( 1, 1,-1, scale3),\
                              pyNMBu.vertex( 1,-1,-1, scale3),\
                              pyNMBu.vertex(-1, 1,-1, scale3),\
                              pyNMBu.vertex(-1,-1,-1, scale3),\
                              pyNMBu.vertex( 2, 0, 0, scale4),\
                              pyNMBu.vertex(-2, 0, 0, scale4),\
                              pyNMBu.vertex( 0, 2, 0, scale4),\
                              pyNMBu.vertex( 0,-2, 0, scale4),\
                              pyNMBu.vertex( 0, 0, 2, scale4),\
                              pyNMBu.vertex( 0, 0,-2, scale4)]
            edges = [
                (1, 0), (2, 0), (3, 1), (3, 2), (4, 0), (5, 1), (5, 4), (6, 2),
                (6, 4), (7, 3), (7, 5), (7, 6), (8, 0), (8, 1), (8, 4), (8, 5),
                (9, 2), (9, 3), (9, 6), (9, 7), (10, 0), (10, 2), (10, 4), (10, 6),
                (11, 1), (11, 3), (11, 5), (11, 7), (12, 0), (12, 1), (12, 2),
                (12, 3), (13, 4), (13, 5), (13, 6), (13, 7),
            ]
            faces = [
                (1, 0, 8), (1, 0, 12), (2, 0, 10), (2, 0, 12), (3, 1, 11), (3, 1, 12),
                (3, 2, 9), (3, 2, 12), (4, 0, 8), (4, 0, 10), (5, 1, 8), (5, 1, 11),
                (5, 4, 8), (5, 4, 13), (6, 2, 9), (6, 2, 10), (6, 4, 10), (6, 4, 13),
                (7, 3, 9), (7, 3, 11), (7, 5, 11), (7, 5, 13), (7, 6, 9), (7, 6, 13),
            ]
            
            CoordVertices = np.array(CoordVertices)
            edges = np.array(edges)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        """
        Generates atomic coordinates for a tetrahedral nanoparticle.

        Args:
            noOutput (bool): If False, displays progress and timing information.
        """
        
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        # central atom = "1st shell"
        c = [[0., 0., 0.]]
        self.nAtoms = 1
        self.nAtomsPerShell = [0]
        self.nAtomsPerShell[0] = 1
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []

        for i in range(1,self.nShell+1):
            
            # Generate vertex atoms
            nAtoms0 = self.nAtoms
            cshell, E, F = self.MakeVertices(i)
            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

            # Generate edge atoms
            nAtoms0 = self.nAtoms
            Rvv = pyNMBu.RAB(cshell,E[0,0],E[0,1]) #distance between two vertex atoms

            # Old : issue with nshell > 6
            nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
            #print("nAtomsOnEdges at beg= ", nAtomsOnEdges)

            ## New: forcing the right number of atoms
            # nAtomsOnEdges = i - 1
            # print("nAtomsOnEdges after= ", nAtomsOnEdges)

            nIntervals = nAtomsOnEdges + 1
            # print("nAtomsOnEdges = ",nAtomsOnEdges)
            coordEdgeAt = []
            for n in range(nAtomsOnEdges):
                for e in E:
                    a = e[0]
                    b = e[1]
                    coordEdgeAt.append(cshell[a]+pyNMBu.vector(cshell,a,b)*(n+1) / nIntervals)
            self.nAtomsPerShell[i] += nAtomsOnEdges * len(E) # number of edges x nAtomsOnEdges
            self.nAtoms += nAtomsOnEdges * len(E)
            c.extend(coordEdgeAt)
            indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
            
            # Generate facet atoms
            coordFaceAt = []
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            for f in F:
                nAtomsOnFaces,coordFaceAt = pyNMBu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces,coordFaceAt)
            self.nAtomsPerShell[i] += nAtomsOnFaces
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput: print(self.nAtomsPerShell)
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)

        self.cog = pyNMBu.centerOfGravity(c)

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()

        # print("Debugg prints")
        # print("nAtomsOnEdges = ",nAtomsOnEdges)
        # print("nAtomsOnFaces = ",nAtomsOnFaces)
        # print("nAtomsPerShell = ", self.nAtomsPerShell )
    
    def prop(self):
        """
        Display unit cell and nanoparticle properties.
        """
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"intershell factor 3 = {self.interShellF3:.2f}")
        print(f"intershell factor 4 = {self.interShellF4:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance for 3rd order vertices = {self.interShellDistance3:.2f} Å")
        print(f"intershell distance for 4th order vertices = {self.interShellDistance4:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pyNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(
            f"radius of the circumscribed sphere passing through the six 4th order "
            f"vertices = {self.radiusCircumscribedSphere4()*0.1:.2f} nm"
        )
        print(
            f"radius of the circumscribed sphere passing through the eight 3rd "
            f"order vertices = {self.radiusCircumscribedSphere3()*0.1:.2f} nm"
        )
        print(f"radius of the midsphere = {self.radiusMidSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: ")
        print(f"coordinates of the center of gravity = {self.cog}")


###########################################################################################################
class fccdrDD(CatalanNP):
    """
    A class for generating XYZ and CIF files of dihedral rhombic dodecahedral nanoparticles (NPs) 
    of various sizes, based on user-defined compounds (either by 
    name, e.g., "Fe", "Au", etc). 

    Key Features:
    - Allows to choose the NP size.
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The symmetry analysis can be skipped to speed up computations.
    - Customizable precision thresholds for structural analysis.
    """

    # Geometric properties of fccdrDD
    nFaces = 12
    nEdges = 32
    nVertices = 14
    edgeLengthF = 1
    radiusCSF = 1
    radiusCSFTB = np.sqrt(2)
    # radiusISF = np.sqrt(6) / 3
    # radiusMSF = 2*np.sqrt(2) / 3
    interShellF = 1/radiusCSF
    interShellFTB = 1/radiusCSFTB
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nShell: int=1,
                 postAnalyzis: bool=True,
                 aseView: bool=False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis: bool=False,
                 jmolCrystalShape: bool=True,
                 noOutput: bool=False,
                 calcPropOnly: bool=False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nShell (int): Number of shells (e.g., `nShell=1` means 2 atoms per edge).
            postAnalyzis (bool): If True, prints additional NP information.
                Examples: cell parameters, moments of inertia,
                inscribed/circumscribed sphere diameters, etc.
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation.
                This is the distance threshold used to decide which atoms are retained.
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.shape (str): Shape 'fccdrDD'
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerShell (list): Number of atoms in each shell.
            self.interShellDistance (float): Distance between shells.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.cog (np.array): Center of gravity of the NP.
            self.imageFile (str): Path to a reference image.

        """
        self.element = element
        self.shape='fccdrDD'
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.interShellDistanceTB = self.Rnn / self.interShellFTB
        self.cog = np.array([0., 0., 0.])
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("fccrdd-C.png")
        if not noOutput: vID.centerTitle(f"{nShell} shells fcc rhombic dodecahedron")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Dihedral rhombic dodecahedron (drDD) with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ Returns the number of atoms of an bcc rhombic dodecahedron of size i"""
        return 8*i**3 + 6*i**2 + 2*i + 3
    
    def nAtomsPerShellAnalytic(self):
        """
        Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer, 
        computes the number of atoms for the given shell, and subtracts 
        the cumulative sum of the previous shells to get the number of new 
        atoms in the current shell.

        Returns:
            list: A list where each element represents the number of atoms 
                  in a specific shell.
        """
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni-Sum)
        return n
    
    def nAtomsPerShellCumulativeAnalytic(self):
        """
        Computes the cumulative number of atoms up to each shell.

        This function returns the total number of atoms present in the 
        nanoparticle for each shell layer, building up cumulatively.

        Returns:
            list: A list where each element represents the total number of 
                  atoms present up to that shell.
        """
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n
    
    def nAtomsAnalytic(self):
        """
        Computes the total number of atoms in the nanoparticle.
        """
        n = self.nAtomsF(self.nShell)
        return n
    
    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .

        The edge length is determined based on the interatomic distance (Rnn) 
        and the number of shells (`nShell`).
        """
        return self.Rnn*self.nShell

    # def radiusCircumscribedSphere3(self):
    #     return self.radiusCSF3rdOrderV*self.edgeLength()

    # def radiusCircumscribedSphere4(self):
    #     return self.radiusCSF4thOrderV*self.edgeLength()

    # def radiusMidSphere(self):
    #     return self.radiusMSF*self.edgeLength()

    # def radiusInscribedSphere(self):
    #     return self.radiusISF*self.edgeLength()

    def area(self):
        """
        Computes the surface area of the nanoparticle in square Ångströms.
        """
        el = self.edgeLength()
        return 8 * np.sqrt(2) * el**2
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return 16 * np.sqrt(3)/9 * el**3
    
    def MakeVertices(self,i):
        """
        Generates the coordinates of the vertices, edges, and faces 
        for the ith shell of the dodecahedral nanoparticle.
        Args:
            - i (int): index of the shell
        Returns:
            - CoordVertices (np.ndarray): the 14 vertex coordinates of the ith shell of a dihedral rhombic dodecahedron
            - edges (np.ndarray): indexes of the 24 edges
            - faces (np.ndarray): indexes of the 12 faces 
        """

        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell = {self.nShell}")
        else:
            # Define vertex positions based on dodecahedral geometry
            scale = self.interShellDistance * i
            scaleTB = self.interShellDistanceTB * i
            CoordVertices = [ pyNMBu.vertex( 0, 0, 2, scaleTB),\
                              pyNMBu.vertex( 0, 0,-2, scaleTB),\
                              pyNMBu.vertex( 1, 1, 0, scale),\
                              pyNMBu.vertex( 1,-1, 0, scale),\
                              pyNMBu.vertex(-1, 1, 0, scale),\
                              pyNMBu.vertex(-1,-1, 0, scale),\
                              pyNMBu.vertex( 1, 0, 1, scale),\
                              pyNMBu.vertex( 1, 0,-1, scale),\
                              pyNMBu.vertex(-1, 0, 1, scale),\
                              pyNMBu.vertex(-1, 0,-1, scale),\
                              pyNMBu.vertex( 0, 1, 1, scale),\
                              pyNMBu.vertex( 0, 1,-1, scale),\
                              pyNMBu.vertex( 0,-1, 1, scale),\
                              pyNMBu.vertex( 0,-1,-1, scale)]
            edges = [
                (6, 0), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 0), (8, 4),
                (8, 5), (9, 1), (9, 4), (9, 5), (10, 0), (10, 2), (10, 4), (10, 6),
                (10, 8), (11, 1), (11, 2), (11, 4), (11, 7), (11, 9), (12, 0),
                (12, 3), (12, 5), (12, 6), (12, 8), (13, 1), (13, 3), (13, 5),
                (13, 7), (13, 9),
            ]
            faces3 = [
                (6, 0, 10), (6, 0, 12), (6, 2, 10), (6, 3, 12), (7, 1, 11),
                (7, 1, 13), (7, 2, 11), (7, 3, 13), (8, 0, 10), (8, 0, 12),
                (8, 4, 10), (8, 5, 12), (9, 1, 11), (9, 1, 13), (9, 4, 11),
                (9, 5, 13),
            ]
            faces4 = [( 2, 6, 3, 7), ( 4, 8, 5, 9), ( 2, 10, 4, 11), ( 3, 12, 5, 13)]
            CoordVertices = np.array(CoordVertices)
            edges = np.array(edges)
            faces3 = np.array(faces3)
            faces4 = np.array(faces4)
        return CoordVertices, edges, faces3, faces4

    def coords(self,noOutput):
        """
        Generates atomic coordinates for a dodecahedral nanoparticle.

        Args:
            noOutput (bool): If False, displays progress and timing information.

        Steps:
            - Generates vertex atoms.
            - Calculates and places edge atoms along the edges.
            - Generates facet atoms to fill in faces.
            - Stores final atomic positions in an ASE Atoms object.

        Returns:
            None (updates class attributes).
        """
        
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        # central atom = "1st shell"
        c = [[0., 0., 0.]]
        self.nAtoms = 1
        self.nAtomsPerShell = [0]
        self.nAtomsPerShell[0] = 1
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        for i in range(1,self.nShell+1):
            
            # Generate vertex atoms 
            nAtoms0 = self.nAtoms
            cshell, E, F3, F4 = self.MakeVertices(i)
            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

            # Generate edge atoms
            nAtoms0 = self.nAtoms
            Rvv = pyNMBu.RAB(cshell,E[0,0],E[0,1]) #distance between two vertex atoms
            nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
            nIntervals = nAtomsOnEdges + 1
            # print("nAtomsOnEdges = ",nAtomsOnEdges)
            coordEdgeAt = []
            for n in range(nAtomsOnEdges):
                for e in E:
                    a = e[0]
                    b = e[1]
                    coordEdgeAt.append(cshell[a]+pyNMBu.vector(cshell,a,b)*(n+1) / nIntervals)
            self.nAtomsPerShell[i] += nAtomsOnEdges * len(E) # number of edges x nAtomsOnEdges
            self.nAtoms += nAtomsOnEdges * len(E)
            c.extend(coordEdgeAt)
            indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
            
            # Generate triangle facet atoms
            coordFaceAt = []
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            for f in F3:
                nAtomsOnFaces,coordFaceAt = pyNMBu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces,coordFaceAt)
            self.nAtomsPerShell[i] += nAtomsOnFaces
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))
            
            # Generate  square facet atoms
            coordFaceAt = []
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            for f in F4:
                nAtomsOnFaces,coordFaceAt = pyNMBu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces,coordFaceAt)
            self.nAtomsPerShell[i] += nAtomsOnFaces
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput: print(self.nAtomsPerShell)
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)
            
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()
    
    def prop(self):
        """
        Display unit cell and nanoparticle properties.
        """
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"intershell factor for tob and bottom vertices = {self.interShellFTB:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"intershell distance for top and bottom vertices = {self.interShellDistanceTB:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pyNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        # print(f"radius of the circumscribed sphere passing through the six 4th order vertices = {self.radiusCircumscribedSphere4()*0.1:.2f} nm")
        # print(f"radius of the circumscribed sphere passing through the eight 3rd order vertices = {self.radiusCircumscribedSphere3()*0.1:.2f} nm")
        # print(f"radius of the midsphere = {self.radiusMidSphere()*0.1:.2f} nm")
        # print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: ")
        print("Comment: It can be seen as a cuboctahedron with square pyramids augmented on the top and bottom")
        print(f"coordinates of the center of gravity = {self.cog}")
