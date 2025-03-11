from visualID import  fg, hl, bg
import visualID as vID

import sys
import numpy as np
import pyNanoMatBuilder.utils as pyNMBu
import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view
from ase.cluster.cubic import FaceCenteredCubic
import os

###########################################################################################################
class regfccOh:
    """
    A class for generating XYZ and CIF files of regular fcc octahedral nanoparticles (NPs) 
    of various sizes, based on user-defined compounds (either by 
    name, e.g., "Fe", "Au", etc). 

    Key Features:
    - Allows to choose the NP size.
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The `nOrder` parameter determines the level of imbrication
    - The symmetry analysis can be skipped to speed up computations.
    - Customizable precision thresholds for structural analysis.
    """

    # Geometric properties of regfccOh
    nFaces = 8 # Number of triangular faces
    nEdges = 12
    nVertices = 6
    edgeLengthF = 1 # length of an edge
    radiusCSF = edgeLengthF * np.sqrt(2)/2 #Centroid to vertex distance = Radius of circumsphere
    radiusISF = edgeLengthF * np.sqrt(6)/6 #Radius of insphere that is tangent to faces
    radiusMSF = edgeLengthF / 2 #Radius of midsphere that is tangent to edges
    dihedralAngle = np.rad2deg(np.arccos(-1/3)) # Angle between two adjacent triangular faces
    interShellF = 1/radiusCSF
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float = 2.7,
                 nOrder: int = 1,
                 shape: str='regfccOh',
                 postAnalyzis: bool=True,
                 aseView: bool=False,
                 thresholdCoreSurface: float=1.,
                 skipSymmetryAnalyzis: bool=False,
                 jmolCrystalShape: bool=True,
                 noOutput: bool= False,
                 calcPropOnly: bool=False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nOrder (int): Determines the level of imbrication = the number of atomic layers along an edge (e.g., `nOrder=1` means 2 atoms per edge).
            shape (str): Shape 'regfccOh'
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerLayer (list): Number of atoms in each atomic layer.
            self.nAtomsPerEdge (int): Number of atoms per edge.
            self.interLayerDistance (float): Distance between atomic layers.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.cog (np.array): Center of gravity of the NP.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        self.element = element
        self.shape = shape
        self.Rnn = Rnn
        self.nOrder = nOrder
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.nAtomsPerEdge = self.nOrder+1
        self.interLayerDistance = self.Rnn/self.interShellF
        self.jmolCrystalShape = jmolCrystalShape
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pyNMBu.imageNameWithPathway("fccOh-C.png")
        self.trPlanes = None
        if not noOutput: vID.centerTitle(f"{nOrder}th order regular fcc Octahedron")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Regular octahedron of order {self.nOrder} (i.e. {self.nOrder+1} atoms lie on an edge) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of an octahedron of size i"""
        return round((2/3)*i**3 + 2*i**2 + (7/3)*i + 1)
    
    def nAtomsPerShellAnalytic(self):
        """
        Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer (from 1 to `nOrder`), 
        computes the number of atoms for the given shell, and subtracts 
        the cumulative sum of the previous shells to get the number of new 
        atoms in the current shell.

        Returns:
            list: A list where each element represents the number of atoms 
                  in a specific shell.
        """
        n = []
        Sum = 0
        for i in range(1,self.nOrder+1):
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
        for i in range(1,self.nOrder+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n    

    def nAtomsAnalytic(self):
        """
        Computes the total number of atoms in the nanoparticle.
        """
        n = self.nAtomsF(self.nOrder)
        return n
    
    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .

        The edge length is determined based on the interatomic distance (Rnn) 
        and the number of atomic layers (`nOrder`).
        """
        return self.Rnn*self.nOrder #Angs

    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength() #angs

    def radiusInscribedSphere(self):
        """
        Computes the radius of the inscribed sphere of the nanoparticle in Å  .
        """
        return self.radiusISF*self.edgeLength()

    def radiusMidSphere(self):
        """
        Computes the radius of the midsphere of the nanoparticle in Å.
        The midsphere is a sphere that touches the edges of the nanoparticle.
        """
        return self.radiusMSF*self.edgeLength()

    def area(self):
        """
        Computes the surface area of the nanoparticle in square Ångströms.
        """
        el = self.edgeLength()
        return el**2*np.sqrt(3)
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return np.sqrt(2)*el**3/3 

    def MakeVertices(self,i):
        """
        Generates the coordinates of the vertices, edges, and faces 
        for the ith shell of an octahedral nanoparticle.
        Args:
            - i (int): Index of the shell layer.
        Returns:
            - CoordVertices (np.ndarray): the 6 vertex coordinates of the ith shell of an octahedron
            - edges (np.ndarray): indexes of the 30 edges
            - faces (np.ndarray): indexes of the 20 faces 
        """
        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.] # Central atom at the origin
            edges = []
            faces = []
            
        elif (i > self.nOrder):
            sys.exit(f"regfccOh.MakeVertices(i) is called with i = {i} > nOrder = {self.nOrder}")
            
        else:
            scale = self.interLayerDistance * i
            # Define vertex positions based on octahedral geometry
            CoordVertices = [ pyNMBu.vertex(-1, 0, 0, scale),\
                              pyNMBu.vertex( 1, 0, 0, scale),\
                              pyNMBu.vertex( 0,-1, 0, scale),\
                              pyNMBu.vertex( 0, 1, 0, scale),\
                              pyNMBu.vertex( 0, 0,-1, scale),\
                              pyNMBu.vertex( 0, 0, 1, scale)]
            edges = [( 2, 0), ( 2, 1), ( 3, 0), ( 3, 1), ( 4, 0), ( 4, 1), ( 4, 2), ( 4, 3), ( 5, 0), ( 5, 1), ( 5, 2), ( 5, 3)]
            faces = [( 2, 0, 4), ( 2, 0, 5), ( 2, 1, 4), ( 2, 1, 5), ( 3, 0, 4), ( 3, 0, 5), ( 3, 1, 4), ( 3, 1, 5)]
            CoordVertices = np.array(CoordVertices)
            edges = np.array(edges)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        """
        Generates atomic coordinates for an octahedral nanoparticle.

        Args:
            noOutput (bool): If False, displays progress and timing information.

        Steps:
            - Generates vertex atoms.
            - Calculates and places edge atoms along the edges.
            - Generates facet atoms to fill in faces.
            - Adds core atoms layer by layer.
            - Stores final atomic positions in an ASE Atoms object.

        Returns:
            None (updates class attributes).
        """
        
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pyNMBu.timer(); chrono.chrono_start()
        c = [] # List of atomic coordinates
        # print(self.nAtomsPerLayer)
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        indexCoreAtoms = []

        #  Generate vertex atoms 
        nAtoms0 = 0
        self.nAtoms += self.nVertices
        cVertices, E, F = self.MakeVertices(self.nOrder)
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

        # Generate edge atoms
        nAtoms0 = self.nAtoms
        Rvv = pyNMBu.RAB(cVertices,E[0,0],E[0,1]) # Distance between two vertex atoms
        nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
        nIntervals = nAtomsOnEdges + 1
        #print("nAtomsOnEdges = ",nAtomsOnEdges)
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E: # Loop over all edges
                a = e[0]
                b = e[1]
                coordEdgeAt.append(cVertices[a]+pyNMBu.vector(cVertices,a,b)*(n+1) / nIntervals) # Compute interpolated positions along the edge
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
        # print(indexEdgeAtoms)
        
        # Generate facet atoms
        coordFaceAt = []
        nAtomsOnFaces = 0
        nAtoms0 = self.nAtoms
        for f in F:
            nAtomsOnFaces,coordFaceAt = pyNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsOnFaces,coordFaceAt)
        self.nAtoms += nAtomsOnFaces
        c.extend(coordFaceAt)
        indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        # Generate core atoms
        # Layer by layer strategy, starting from bottom to top when identified, just use MakeFaceCoord and define, for each layer, the four atoms on the edge as a facet
        coordCoreAt = []
        nAtomsInCore = 0
        nAtoms0 = self.nAtoms
        # first apply it to atoms 0, 1, 2, 3
        # f = [a,b,c,d] must be given in the order a--b
        #                                          |  |
        #                                          d--c
        f = np.array([0,3,1,2])
        nAtomsInCore,coordCoreAt = pyNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
        # don't ask... it is the algorithm to find the indexes of the square
        # corners of each layer along z

        # Helper functions to define atomic layers
        def layerup(ilayer,f):
            return 12*ilayer+f-2
        def layerdown(ilayer,f):
            return 12*ilayer+f+2

        # Loop to generate multiple layers in the core
        for i in range(1,nAtomsOnEdges+1): 
            f = layerup(i,np.array([0,3,1,2]))
            # print(i,"  fup",f)
            nAtomsInCore,coordCoreAt = pyNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
            f = layerdown(i,np.array([0,3,1,2]))
            # print(i,"fdown",f)
            nAtomsInCore,coordCoreAt = pyNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
            
        self.nAtoms += nAtomsInCore
        c.extend(coordCoreAt)
        indexCoreAtoms.extend(range(nAtoms0,self.nAtoms))

        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")
        # Store results in an ASE Atoms object 
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)

        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
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
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"interlayer distance = {self.interLayerDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"radius after volume = {pyNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print(f"dihedral angle = {self.dihedralAngle:.1f}°")
        # print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("intermediate magic numbers = ",self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: cube")
        print("Indexes of vertex atoms = [0,1,2,3,4,5] by construction")
        print(f"coordinates of the center of gravity = {self.cog}")
        return

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface, noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), determines the nanoparticle shape, 
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
        
        import math
        self.dim=[0,0,0]
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape
        pyNMBu.MOI_shapes(self, noOutput)
        
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        # [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
        #     pyNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)

###########################################################################################################
class regIco:
    """
    A class for generating XYZ and CIF files of regular icosahedral nanoparticles (NPs) 
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
     # Geometric properties of regIco
    nFaces = 20 
    nEdges = 30
    nVertices = 12
    phi = (1 + np.sqrt(5))/2 # Golden ratio 
    edgeLengthF = 1  # length of an edge
    radiusCSF = np.sqrt(10 + 2*np.sqrt(5))/4 # Radius of circumsphere
    interShellF = 1/radiusCSF 
#    interShellF = np.sqrt(2*(1-1/np.sqrt(5)))
    radiusISF = np.sqrt(3) * (3 + np.sqrt(5))/12  #Radius of insphere
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nShell: int=1,
                 shape: str='regIco',
                 postAnalyzis=True,
                 aseView: bool=False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 jmolCrystalShape: bool=True,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nShell (int):Nulber of shells = the number of atomic layers along an edge (e.g., `nOrder=1` means 2 atoms per edge).
            shape (str): Shape 'regIco'
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerShell (list): Number of atoms in each shell.
            self.interShellDistance (float): Distance between atomic shells.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        self.element=element
        self.shape= shape
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("ico-C.png")
        self.trPlanes = None
        if not noOutput: vID.centerTitle(f"{nShell} shells icosahedron")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self): 
        return(f"Regular icosahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of an icosahedron of size i"""
        return (10*i**3 + 11*i)//3 + 5*i**2 + 1
    
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

    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength()

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
        return 5 * el**2 * np.sqrt(3)
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return 5 * el**3 * (3 + np.sqrt(5))/12
    
    def MakeVertices(self,i):
        """
        Generates the coordinates of the vertices, edges, and faces 
        for the ith shell of an octahedral nanoparticle.
        Args:
            - i (int): Index of the shell
        Returns:
            - CoordVertices(np.ndarray): the 12 vertex coordinates of the ith shell of an icosahedron
            - edges (np.ndarray): indexes of the 30 edges
            - faces(np.ndarray): indexes of the 20 faces 
        """
        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell = {self.nShell}")
        else:
            # Define vertex positions based on icosahedral geometry
            phi = self.phi
            scale = self.interShellDistance * i
            CoordVertices = [ pyNMBu.vertex(-1, phi, 0, scale),\
                              pyNMBu.vertex( 1, phi, 0, scale),\
                              pyNMBu.vertex(-1, -phi, 0, scale),\
                              pyNMBu.vertex( 1, -phi, 0, scale),\
                              pyNMBu.vertex(0, -1, phi, scale),\
                              pyNMBu.vertex(0, 1, phi, scale),\
                              pyNMBu.vertex(0, -1, -phi, scale),\
                              pyNMBu.vertex(0, 1, -phi, scale),\
                              pyNMBu.vertex( phi, 0, -1, scale),\
                              pyNMBu.vertex( phi, 0, 1, scale),\
                              pyNMBu.vertex(-phi, 0, -1, scale),\
                              pyNMBu.vertex(-phi, 0, 1, scale) ]
            edges = [( 1, 0), ( 3, 2), ( 4, 2), ( 4, 3), ( 5, 0), ( 5, 1), ( 5, 4), ( 6, 2), ( 6, 3), ( 7, 0),\
                  ( 7, 1), ( 7, 6), ( 8, 1), ( 8, 3), ( 8, 6), ( 8, 7), ( 9, 1), ( 9, 3), ( 9, 4), ( 9, 5),\
                  ( 9, 8), (10, 0), (10, 2), (10, 6), (10, 7), (11, 0), (11, 2), (11, 4), (11, 5), (11,10),]
            faces = [(7,0,1),(7,1,8),(7,8,6),(7,6,10),(7,10,0),\
                     (0,11,5),(0,5,1),(1,5,9),(1,8,9),(8,9,3),(8,3,6),(6,3,2),(6,10,2),(10,2,11),(10,0,11),\
                     (4,2,3),(4,3,9),(4,9,5),(4,5,11),(4,11,2)]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        """
        Generates atomic coordinates for an icosahedral nanoparticle.

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
        chrono = pyNMBu.timer(); chrono.chrono_start()
        # central atom = "1st shell"
        c = [[0., 0., 0.]]
        self.nAtoms = 1
        self.nAtomsPerShell = [0]
        self.nAtomsPerShell[0] = 1
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        
        for i in range(1,self.nShell+1):
         #  Generate vertex atoms 
            nAtoms0 = self.nAtoms
            cshell, E, F = self.MakeVertices(i)
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
                for e in E: # Loop over all edges
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
            
        # print(indexVertexAtoms)
        # print(indexEdgeAtoms)
        # print(indexFaceAtoms)
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP=aseObject
        self.cog = self.NP.get_center_of_mass()
    
    def prop(self): #
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
        print("phi = ",self.phi)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pyNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: dodecahedron")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface, noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), determines the nanoparticle shape, 
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
        import math
        self.dim=[0,0,0]
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape
        pyNMBu.MOI_shapes(self, noOutput)


        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)


###########################################################################################################
class regfccTd:
    """
    A class for generating XYZ and CIF files of regular fcc tetrahedral nanoparticles (NPs) 
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
    # Geometric properties of regfccTd
    nFaces = 4
    nEdges = 6
    nVertices = 4
    edgeLengthF = 1 # length of an edge
    heightOfPyramidF = edgeLengthF * np.sqrt(2/3)
    radiusCSF = edgeLengthF * np.sqrt(3/8) # Centroid to vertex distance = Radius of circumsphere
    radiusISF = edgeLengthF/np.sqrt(24) # Radius of insphere that is tangent to faces
    radiusMSF = edgeLengthF/np.sqrt(8) # Radius of midsphere that is tangent to edges
    fveAngle = np.rad2deg(np.arccos(1/np.sqrt(3))) # Face-vertex-edge angle
    fefAngle = np.rad2deg(np.arccos(1/3)) # Face-edge-face angle
    vcvAngle = np.rad2deg(np.arccos(-1/3)) # Vertex-Center-Vertex angle
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayer: int=1,
                 shape: str='regfccTd',
                 postAnalyzis=True,
                 aseView: bool=False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 jmolCrystalShape: bool=True,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nLayer (int): Number of layers, also equals to the number of atoms per edge (e.g., `nOrder=2` means 2 atoms per edge).
            shape (str): Shape 'regfccOh'
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerLayer (list): Number of atoms in each atomic layer.
            self.nAtomsPerEdge (int): Number of atoms per edge.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.cog (np.array): Center of gravity of the NP.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        
        self.element = element
        self.shape=shape
        self.Rnn = Rnn
        self.nLayer = nLayer
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.nAtomsPerEdge = self.nLayer
        self.jmolCrystalShape = jmolCrystalShape
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pyNMBu.imageNameWithPathway("fccTd-C.png")
        self.NP = None
        self.trPlanes = None
        if not noOutput: vID.centerTitle(f"fcc tetrahedron: {nLayer} atoms/edge = number of layers")
          
        if not noOutput: self.prop()

        if not calcPropOnly:
           self.coords(noOutput)
           if aseView: view(self.NP)
           if postAnalyzis:
               self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput=noOutput)
               if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Regular tetrahedron with {self.nLayer} layer(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of a tetrahedron of size i """
        return round(i**3/6 + i**2 + 11*i/6 + 1)
    
    def nAtomsPerLayerAnalytic(self):
        """
        Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each layer, 
        computes the number of atoms for the given layer, and subtracts 
        the cumulative sum of the previous shells to get the number of new 
        atoms in the current layer.

        Returns:
            list: A list where each element represents the number of atoms 
                  in a specific layer.
        """
        n = []
        Sum = 0
        for i in range(self.nLayer):
            Sum = sum(n)
            ni = int(self.nAtomsF(i))
            n.append(ni-Sum)
            # print(i,ni,Sum,n)
        return n
    
    def nAtomsAnalytic(self):
        """
        Computes the total number of atoms in the nanoparticle.
        """
        n = self.nAtomsF(self.nLayer-1)
        return n
    
    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .

        The edge length is determined based on the interatomic distance (Rnn) 
        and the number of atomic layers (`nLayer`).
        """
        return self.Rnn*(self.nLayer-1)

    def heightOfPyramid(self):
        """
        Computes the length of the height of the pyramid in Å .
        """
        return self.heightOfPyramidF*self.edgeLength()
    
    def interLayerDistance(self):
        """
        Computes the distance between the layers in Å .
        """
        return self.heightOfPyramid()/(self.nLayer-1)
    
    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        """
        Computes the radius of the inscribed sphere of the nanoparticle in Å  .
        """
        return self.radiusISF*self.edgeLength()

    def radiusMidSphere(self):
        """
        Computes the radius of the midsphere of the nanoparticle in Å.
        The midsphere is a sphere that touches the edges of the nanoparticle.
        """
        return self.radiusMSF*self.edgeLength()

    def area(self):
        """
        Computes the surface area of the nanoparticle in square Ångströms.
        """
        el = self.edgeLength()
        return el**2*np.sqrt(3)
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return el**3/(6*np.sqrt(2)) 

    def MakeVertices(self,nL):
        """
        Generates the coordinates of the vertices, edges, and faces 
        for the ith shell of an tetrahedral nanoparticle.
        Args:
            - nL = number of layers = number of atoms per edge
        returns:
            - CoordVertices (np.ndarray): the 4 vertex coordinates of a tetrahedron
            - edges (np.ndarray): indexes of the 6 edges
            - faces (np.ndarray): indexes of the 4 faces 
        """
        if (nL > self.nLayer):
            sys.exit(f"regTd.MakeVertices(nL) is called with nL = {nL} > nLayer = {self.nLayer}")
        else:
            scale = self.radiusCircumscribedSphere()
            c = 1/(2*np.sqrt(2)) # edge length 1
            # Define vertex positions based on tetrahedral geometry
            CoordVertices = [pyNMBu.vertex(c, c, c, scale),\
                             pyNMBu.vertex(c, -c, -c, scale),\
                             pyNMBu.vertex(-c, c, -c, scale),\
                             pyNMBu.vertex(-c, -c, c, scale)]
            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            faces = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        """
        Generates atomic coordinates for a tetrahedral nanoparticle.

        Args:
            noOutput (bool): If False, displays progress and timing information.

        Steps:
            - Generates vertex atoms.
            - Calculates and places edge atoms along the edges.
            - Generates facet atoms to fill in faces.
            - Adds core atoms layer by layer.
            - Stores final atomic positions in an ASE Atoms object.

        Returns:
            None (updates class attributes).
        """
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pyNMBu.timer(); chrono.chrono_start()
        c = [] # List of atomic coordinates
        # print(self.nAtomsPerLayer)
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        indexCoreAtoms = []

        # Generate vertex atoms
        nAtoms0 = 0
        self.nAtoms += self.nVertices
        cVertices, E, F = self.MakeVertices(self.nLayer-1)
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

        # Generate edge atoms
        nAtoms0 = self.nAtoms
        Rvv = pyNMBu.RAB(cVertices,E[0,0],E[0,1]) #distance between two vertex atoms
        nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
        nIntervals = nAtomsOnEdges + 1
        #print("nAtomsOnEdges = ",nAtomsOnEdges)
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E: # Loop over all edges
                a = e[0]
                b = e[1]
                coordEdgeAt.append(cVertices[a]+pyNMBu.vector(cVertices,a,b)*(n+1) / nIntervals)
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
        self.nAtomsPerEdge = nAtomsOnEdges  + 2 #2 vertices
        # print(indexEdgeAtoms)
        
        # Generate facet atoms
        coordFaceAt = []
        nAtomsOnFaces = 0
        nAtoms0 = self.nAtoms
        for f in F:
            nAtomsOnFaces,coordFaceAt = pyNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsOnFaces,coordFaceAt)
        self.nAtoms += nAtomsOnFaces
        c.extend(coordFaceAt)
        indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        # Generate core atoms
        # Layer by layer strategy, using atoms on edges [0-1],[0-2],[0-3] when identified, just use MakeFaceCoord and define, for each layer, the three atoms on the edge as a facet
        # just start from 4th layer
        coordCoreAt = []
        nAtomsInCore = 0
        nAtoms0 = self.nAtoms
        for ilayer in range(4,self.nLayer+1):
            FirstAtom = 4 + (ilayer-2)*6
            f = np.array([FirstAtom, FirstAtom+1, FirstAtom+2])
            # print("layer ",ilayer,f)
            nAtomsInCore,coordCoreAt = pyNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
        self.nAtoms += nAtomsInCore
        c.extend(coordCoreAt)
        indexCoreAtoms.extend(range(nAtoms0,self.nAtoms))

        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput: print(self.nAtomsPerLayer)
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)

        self.cog = pyNMBu.centerOfGravity(c)
        
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
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
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"inter-layer distance = {self.interLayerDistance():.2f} Å")
        print(f"height of pyramid = {self.heightOfPyramid()*0.1:.2f} nm")
        print(f"radius after volume = {pyNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"radius of the midsphere that is tangent to edges = {self.radiusMidSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print(f"face-vertex-edge angle = {self.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle = {self.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond) angle = {self.vcvAngle:.1f}°")
        print("number of atoms per layer = ",self.nAtomsPerLayerAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: tetrahedron")
        print("Indexes of vertex atoms = [0,1,2,3] by construction")
        print(f"coordinates of the center of gravity = {self.cog}")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface,noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), determines the nanoparticle shape, 
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
        # find the size using the MOI mass normalized 
        import math
        self.dim=[0,0,0]
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape
        pyNMBu.MOI_shapes(self, noOutput)
        
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)

###########################################################################################################
class regDD:
    """
    A class for generating XYZ and CIF files of regular dodecahedral nanoparticles (NPs) 
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

    # Geometric properties of regDD
    nFaces = 12
    nEdges = 30
    nVertices = 20
    phi = (1 + np.sqrt(5))/2 # golden ratio
    edgeLengthF = 1
    radiusCSF = np.sqrt(3) * (1 + np.sqrt(5))/4 # Radius of circumsphere
    interShellF = 1/radiusCSF
    radiusISF = np.sqrt((5/2) + (11/10)*np.sqrt(5))/2 # Radius of insphere that is tangent to faces
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nShell: int=1,
                 shape: str='regDD',
                 postAnalyzis=True,
                 aseView: bool=False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 jmolCrystalShape: bool=True,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nShell (int): Number of shells (e.g., `nShell=1` means 2 atoms per edge).
            shape (str): Shape 'regDD'
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerShell (list): Number of atoms in each shell.
            self.interShellDistance (float): Distance between shells.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        self.element = element
        self.shape= shape
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("rDD-C.png")
        self.trPlanes = None
        if not noOutput: vID.centerTitle(f"{nShell} shells regular dodecahedron")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput=noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Regular dodecahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of a dodecahedron of size i"""
        return 10*i**3 + 15*i**2 + 7*i + 1
    
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

    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength()

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
        return 3 * el**2 * np.sqrt(25 + 10*np.sqrt(5))
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return (15 + 7*np.sqrt(5)) * el**2/4 

    def MakeVertices(self,i):
        """
        Generates the coordinates of the vertices, edges, and faces 
        for the ith shell of a dodecahedral nanoparticle.
        Args:
            - i (int): Index of the shell
        Returns:
            - CoordVertices (np.ndarray): the 20 vertex coordinates of the ith shell of a dodecahedron
            - edges (np.ndarray): indexes of the 30 edges
            - faces (np.ndarray): indexes of the 12 faces 
        """
        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell= {self.nShell}")
        else:
            # Define vertex positions based on dodecahedral geometry
            phi = self.phi
            scale = self.interShellDistance * i
            CoordVertices = [pyNMBu.vertex(1, 1, 1, scale),\
                             pyNMBu.vertex(-1, 1, 1, scale),\
                             pyNMBu.vertex(1, -1, 1, scale),\
                             pyNMBu.vertex(1, 1, -1, scale),\
                             pyNMBu.vertex(-1, -1, 1, scale),\
                             pyNMBu.vertex(-1, 1, -1, scale),\
                             pyNMBu.vertex(1, -1, -1, scale),\
                             pyNMBu.vertex(-1, -1, -1, scale),\
                             pyNMBu.vertex(0, phi, 1/phi, scale),\
                             pyNMBu.vertex(0, -phi, 1/phi, scale),\
                             pyNMBu.vertex(0, phi, -1/phi, scale),\
                             pyNMBu.vertex(0, -phi, -1/phi, scale),\
                             pyNMBu.vertex(1/phi, 0, phi, scale),\
                             pyNMBu.vertex(-1/phi, 0, phi, scale),\
                             pyNMBu.vertex(1/phi, 0, -phi, scale),\
                             pyNMBu.vertex(-1/phi, 0, -phi, scale),\
                             pyNMBu.vertex(phi, 1/phi, 0, scale),\
                             pyNMBu.vertex(phi, -1/phi, 0, scale),\
                             pyNMBu.vertex(-phi, 1/phi, 0, scale),\
                             pyNMBu.vertex(-phi, -1/phi, 0, scale)]
            edges = [(8,0), (8,1), (9,2), (9,4), (10,3), (10,5), (10,8), (11,6), (11,7),\
                     (11,9), (12,0), (12,2), (13,1), (13,4), (13,12), (14,3), (14,6), (15,5),\
                     (15,7), (15,14), (16,0), (16,3), (17,2), (17,6), (17,16), (18,1), (18,5),\
                     (19,4), (19,7), (19,18)]
            faces = [(0,8,10,3,16),(0,12,13,1,8),(8,1,18,5,10),(10,5,15,14,3),(3,14,6,17,16),(16,17,2,12,0),\
                     (4,9,11,7,19),(4,13,12,2,9),(9,2,17,6,11),(11,6,14,15,7),(7,15,5,18,19),(19,18,1,13,4)]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

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
        chrono = pyNMBu.timer(); chrono.chrono_start()
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
            nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
            nIntervals = nAtomsOnEdges + 1
            #print("nAtomsOnEdges = ",nAtomsOnEdges)
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
            #print(c)
            
            # Generate facet atoms
            # Center of each pentagonal facet
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            coordFaceAt = []
            for f in F:
                nAtomsOnFaces += 1
                coordCenterFace = pyNMBu.centerOfGravity(cshell,f)
                #print("coordCenterFace",coordCenterFace)
                self.nAtomsPerShell[i] += 1
                coordFaceAt.append(coordCenterFace)
                # atoms from the center of each pentagonal facet to each of its apex 
                nAtomsOnInternalRadius = i-1
                nIntervals = nAtomsOnInternalRadius+1
                # print(f)
                for indexApex,apex in enumerate(f):
                    if (indexApex == len(f)-1):
                        indexApexPlus1 = 0
                    else:
                        indexApexPlus1 = indexApex+1
                    apexPlus1 = f[indexApexPlus1]
                    for n in range(nAtomsOnInternalRadius):
                        nAtomsOnFaces += 1
                        coordAtomOnApex = coordCenterFace+pyNMBu.vectorBetween2Points(coordCenterFace,cshell[apex])*(n+1) / nIntervals
                        coordAtomOnApexPlus1 = coordCenterFace+pyNMBu.vectorBetween2Points(coordCenterFace,cshell[apexPlus1])*(n+1) / nIntervals
                        coordFaceAt.append(coordAtomOnApex)
                        RbetweenRadialAtoms = pyNMBu.Rbetween2Points(coordAtomOnApex,coordAtomOnApexPlus1)
                        nAtomsBetweenRadialAtoms = int((RbetweenRadialAtoms+1e-6) / self.Rnn)-1
                        nIntervalsRadial = nAtomsBetweenRadialAtoms + 1
                        for k in range(nAtomsBetweenRadialAtoms):
                            nAtomsOnFaces += 1
                            coordFaceAt.append(coordAtomOnApex+pyNMBu.vectorBetween2Points(coordAtomOnApex,coordAtomOnApexPlus1)*(k+1) / nIntervalsRadial)
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput: print(self.nAtomsPerShell)
        # Store results in an ASE Atoms object 
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)
                
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
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
        print("phi = ",self.phi)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pyNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: icosahedron")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface,noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), determines the nanoparticle shape, 
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
        import math
        self.dim=[0,0,0]
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape
        pyNMBu.MOI_shapes(self, noOutput)
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)
        

###########################################################################################################
class cube:
    """
    A class for generating XYZ and CIF files of cubic nanoparticles (NPs) 
    of various sizes, based on user-defined compounds (either by 
    name, e.g., "Fe", "Au", etc). 

    Key Features:
    - Allows to choose the NP size.
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The `nOrder` parameter determines the level of imbrication
    - The symmetry analysis can be skipped to speed up computations.
    - Customizable precision thresholds for structural analysis.
    """
    nFaces = 6
    nEdges = 12
    nVertices = 8
    edgeLengthFfcc = np.sqrt(2)
    edgeLengthFbcc = 2/np.sqrt(3)
    radiusCSF = np.sqrt(3)/2
    radiusISF = 1/2
  
    def __init__(self,
                 crystalStructure='fcc',
                 element='Au',
                 Rnn: float=2.7, 
                 nOrder: int=1, 
                 size: int= 0, 
                 shape: str='cube',
                 postAnalyzis=True,
                 aseView: bool=False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 jmolCrystalShape: bool=True,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nOrder (int): Determines the level of imbrication = the number of atomic layers along an edge (e.g., `nOrder=1` means 2 atoms per edge).
            size (float): Size of the cube in nm.
            shape (str): Shape 'cube'.
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerShell (list): Number of atoms in each shell.
            self.nAtomsPerEdge (int): Number of atoms per edge.
            self.interLayerDistance (float): Distance between atomic layers.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.cog (np.array): Center of gravity of the NP.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        self.crystalStructure = crystalStructure
        self.element = element
        self.shape= shape
        self.Rnn = Rnn
        self.nOrder = nOrder
        self.size= size*10 # in angs
        self.nAtomsPerEdge = nOrder+1
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.jmolCrystalShape = jmolCrystalShape
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pyNMBu.imageNameWithPathway("cube-C.png")
        self.trPlanes = None


        
        if not noOutput: vID.centerTitle(f"{nOrder}x{nOrder}x{nOrder} {self.crystalStructure} cube")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput=noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"{self.nOrder}x{self.nOrder}x{self.nOrder} fcc cube with Rnn = {self.Rnn}")
    
    def nAtomsfccF(self,i):
        """ returns the number of atoms of an fcc cube of size i x i x i"""
        return 4*i**3 + 6*i*2 + 3*i + 1

    def nAtomsbccF(self,i):
        """ returns the number of atoms of a bcc cube of size i x i x i"""
        return 2*i**3 + 3*i*2 + 3*i
    
    def nAtomsPerShellAnalytic(self):
        """
        Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer (from 1 to `nOrder`), 
        computes the number of atoms for the given shell, and subtracts 
        the cumulative sum of the previous shells to get the number of new 
        atoms in the current shell.

        Returns:
            list: A list where each element represents the number of atoms 
                  in a specific shell.
        """
        n = []
        Sum = 0
        for i in range(self.nOrder+1):
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
        for i in range(self.nOrder+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n
    
    def nAtomsfccAnalytic(self):
        """
        Computes the total number of atoms in the fcc nanoparticle.
        """
        n = self.nAtomsfccF(self.nOrder)
        return n
        
    def nAtomsbccAnalytic(self):
        """
        Computes the total number of atoms in the bcc nanoparticle.
        """
        n = self.nAtomsbccF(self.nOrder)
        return n
        
    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .

        The edge length is determined based on the interatomic distance (Rnn), the number of atomic layers (`nOrder`) and the crystalStructure (fcc or bcc).
        """
        if self.crystalStructure == 'fcc':
            return self.Rnn*self.edgeLengthFfcc*self.nOrder
        elif self.crystalStructure == 'bcc':
            return self.Rnn*self.edgeLengthFbcc*self.nOrder
        
    def latticeConstant(self):
        """
        Computes the lattice constant length of the nanoparticle in Å, based on the interatomic distance (Rnn) and the crystalStructure (fcc or bcc).
        """
        if self.crystalStructure == 'fcc':
            return self.Rnn*self.edgeLengthFfcc
        elif self.crystalStructure == 'bcc':
            return self.Rnn*self.edgeLengthFbcc
        
    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        """
        Computes the radius of the inscribed sphere of the nanoparticle in Å.
        """
        return self.radiusISF*self.edgeLength()

    def area(self):
        """
        Computes the surface area of the nanoparticle in square Ångströms.
        """
        el = self.edgeLength()
        return 6 * el**2

    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return el**3

    # def coords(self):
    #     print(f"Making a {self.nOrder}x{self.nOrder}x{self.nOrder} fcc cube")
    #     surfaces = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    #     layers = [self.nOrder, self.nOrder, self.nOrder]
    #     fcc = FaceCenteredCubic(self.element, surfaces, layers, latticeconstant=self.latticeConstant())
    #     natoms = len(fcc.positions)
    #     self.nAtoms=natoms
    #     return fcc

    def coords(self,noOutput):
        """
        Generates atomic coordinates for a cubic nanoparticle.

        Args:
            noOutput (bool): If False, displays progress and timing information.

        Steps:
            - Generates vertex atoms.
            - Calculates and places edge atoms along the edges.
            - Generates facet atoms to fill in faces.
            - Adds core atoms layer by layer.
            - Stores final atomic positions in an ASE Atoms object.

        Returns:
            None (updates class attributes).
        """
        

        
        #crystalline structure
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pyNMBu.timer(); chrono.chrono_start()
        if self.crystalStructure == 'fcc':
            cube = bulk(self.element, 'fcc', a=self.latticeConstant(), cubic=True)
        elif self.crystalStructure == 'bcc':
            cube = bulk(self.element, 'bcc', a=self.latticeConstant(), cubic=True)
            
        # Creating supercell depending the entries of user : size of the cube in Angs or nOrder (number of cells)
        if self.size==0 : #if not size given
            if not noOutput: print(f"Now making a {self.nOrder}x{self.nOrder}x{self.nOrder} fcc supercell...")
            M = [[self.nOrder, 0, 0], [0, self.nOrder, 0], [0, 0, self.nOrder]]
            sc=make_supercell(cube, M)
        else : 
            self.n_cells = int(self.size/ self.latticeConstant())
            M = [[self.n_cells, 0, 0], [0, self.n_cells, 0], [0, 0, self.n_cells]]
            sc=make_supercell(cube, M)
       
        # Adding the last layers
        if not noOutput: print(f"... and adding the upper layers")
        sc = cut(sc,extend=1.05)
        natoms = len(sc.positions)
        self.nAtoms=natoms
        self.cog = pyNMBu.centerOfGravity(sc.get_positions())
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        coordsNP = sc.get_positions()
        oldcog = sc.get_center_of_mass()
        coordsNP = coordsNP - oldcog
        sc.set_positions(coordsNP)
        self.NP = sc
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
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"lattice constant = {self.latticeConstant():.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pyNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        # print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        # print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        if self.crystalStructure == 'fcc':
            print("total number of atoms = ",self.nAtomsfccAnalytic())
        elif self.crystalStructure == 'bcc':
            print("total number of atoms = ",self.nAtomsbccAnalytic())
        print("Dual polyhedron: octahedron")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface, noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), determines the nanoparticle shape, 
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
        import math
        self.dim=[0,0,0]
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape
        pyNMBu.MOI_shapes(self, noOutput)        
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)

# NEW

class hollow_shapes:
    #cube
    """
    A class for generating XYZ and CIF files of hollow cubic nanoparticles (NPs)
    with customizable sizes and compositions. Users can define the composition
    by specifying element names (e.g., "Fe", "Au") and provide a "cube" class
    instance from this module to construct the nanoparticle structure.

    Key Features:
    - Allows to choose the cube size and the size of its hollow
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The symmetry analysis can be skipped to speed up computations.
    - Customizable precision thresholds for structural analysis.

    """
    def __init__(self,
                 full_cube,
                 hollow_size: int=0,#Angs?
                 postAnalyzis=True,
                 aseView: bool=False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 jmolCrystalShape: bool=True,
                 noOutput = False,
                 calcPropOnly= False
                ):
        """
        Initialize the class with all necessary parameters.
        
        Args:
            full_cube (class instance): Instance of the class "cube" of the module "pNP".
            hollow_size (float): Size of the hollow in Å.
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            
            self.nAtoms (int): Number of atoms in the NP.
            self.cog (np.array): Center of gravity of the NP.

        """
        if not isinstance(full_cube, cube):
            raise TypeError("full_cube must be an instance of the Class Cube")
        self.full_cube= full_cube
        self.hollow_size = hollow_size 
        self.nAtoms = 0
        self.edgeLength= self.full_cube.edgeLength()
        self.nAtomsPerEdge= self.full_cube.nAtomsPerEdge
        self.cog = np.array([0., 0., 0.])
        if not calcPropOnly:
            self.create_hollow(noOutput)
            
            # if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput=noOutput)
            # #     if aseView: view(self.NPcs)
    
    def __str__(self):
        if self.full_cube.size==0 :
            return(f" cube with order of{self.full_cube.nOrder}, with hollow thickness of {self.hollow_size} Angstrom and with Rnn = {self.full_cube.Rnn}")
        else :
             return(f" fcc cube with size {self.full_cube.size} Angstrom with hollow thickness of {self.hollow_size} Angstrom and with Rnn = {self.full_cube.Rnn}")
    
    def create_hollow(self,noOutput) :
        '''
        Function that creates the cube hollow. 
        The hollow is created using planes that defines the hollow [h k l d] with d= +/- size of the hollow/2.
        
        Args:
            noOutput (bool): If False, prints details about the NP structure.
        
        '''
        
        if not noOutput:   
            print(f"Number of atoms on an edge = {self.nAtomsPerEdge}")
            print(f"Edge length = {round(self.edgeLength*0.1,3)} nm")
            print(f"Creating a hollow of {self.hollow_size} nm ")

        half_inner_cube_size=10*self.hollow_size/2
        self.NP=self.full_cube.NP.copy()
        print("Number of atoms in the cube before creating the hollow =", len(self.NP))
        full_positions = self.full_cube.NP.get_positions()

    
        # Generate the 6 planes that define the hollow (cube)
        planes_with_dist = np.array([
            [0, 0, 1, -half_inner_cube_size],
            [0, 0, -1, -half_inner_cube_size],  
            [0, 1, 0, -half_inner_cube_size],    
            [0, -1, 0, -half_inner_cube_size],  
            [1, 0, 0, -half_inner_cube_size],   
            [-1, 0, 0, -half_inner_cube_size]
        ])   
        
        delAbove= False # delete atoms above/under the 6 planes
        # delete atoms above/under the 6 planes
        # for plane in planes_with_directions:
        current_positions = self.NP.get_positions()
        #     print(f"Plan used: {plane}, delAbove={delAbove}")

        # Generate the truncation 
        AtomsUnderPlanes = pyNMBu.truncateAbovePlanes(
            planes=planes_with_dist,
            coords=current_positions,
            allP= True,
            delAbove=delAbove,
            debug=False,        
            noOutput=False,
            eps= 0.001, # threshold distance
            )
        del self.NP[AtomsUnderPlanes] 
        self.nAtoms=len(self.NP)
        if not noOutput:
            print(f"Number of atoms in the final hollow cube : {self.nAtoms}")
            
        
    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface, noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), determines the nanoparticle shape, 
        analyzes symmetry (if required), and identifies core and surface atoms.
    
        Parameters:
        - skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
        - thresholdCoreSurface (float): Threshold to distinguish core and surface atoms.
        - noOutput (bool): If True, suppresses output messages.
    
        Attributes Updated:
        - self.moi (array): Moment of inertia tensor.
        - self.moisize (array): Normalized moments of inertia.
        - self.vertices, self.simplices, self.neighbors, self.equations (arrays): 
          Geometric properties of the nanoparticle.
        - self.NPcs (Atoms object): Copy of the nanoparticle with surface atoms visually marked.
        - self.NP (Atoms object): Original nanoparticle.
        """
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.dim=[0,0,0]
        print(self.moi)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))   # MOI mass normalized (m of each atoms=1)
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol     
        
        # if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        #if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)
        
            
