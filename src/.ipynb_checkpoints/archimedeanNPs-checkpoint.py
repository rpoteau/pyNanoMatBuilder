import visualID as vID
from visualID import  fg, hl, bg

import sys
import numpy as np
import pyNanoMatBuilder.utils as pyNMBu
import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view

from pyNanoMatBuilder import platonicNPs as pNP

###########################################################################################################
class fccCubo:
    """
    A class for generating XYZ and CIF files of fcc cuboctahedral nanoparticles (NPs) 
    of various sizes, based on user-defined compounds (either by 
    name, e.g., "Fe", "Au", etc). 

    Key Features:
    - Allows to choose the NP size.
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The `nShell` parameter determines the level of imbrication
    - The symmetry analysis can be skipped to speed up computations.
    - Customizable precision thresholds for structural analysis.
    """
    nFaces = 14
    nEdges = 24
    nVertices = 12
    edgeLengthF = 1
    radiusCSF = 1
    interShellF = 1/radiusCSF
    radiusISF = 3/4
  
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
            nShell (int): Determines the level of imbrication = the number of atomic layers along an edge (e.g., `nShell=1` means 2 atoms per edge).
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.shape (str): Shape 'fccCubo'
            self.nAtoms (int): Number of atoms in the NP.
            self.nAtomsPerShell (list): Number of atoms in each shell.
            self.interShellDistance (float): Distance between shells.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.
            self.dim (array): Dimensions calculated from MOI.
        """

        self.element = element
        self.shape='fccCubo'
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("cubo-C.png")
        self.trPlanes = None
        self.dim=[0,0,0]
        if not noOutput: vID.centerTitle(f"{nShell} shells cuboctahedron")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Cuboctahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of a cuboctahedron of size i"""
        return round((10*i**3 + 11*i)/3 + 5*i**2 + 1)
    
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
        return (6 + 2*np.sqrt(3)) * el**2 
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return (5 * np.sqrt(2)/3) * el**3
    
    def MakeVertices(self,i):
        """
        Generates the coordinates of the vertices, edges, and faces 
        for the ith shell of the nanoparticle.
        Args:
            - i (int): index of the shell
        Returns:
            - CoordVertices = the 12 vertex coordinates of the ith shell of an icosahedron
            - edges (np.ndarray): indexes of the 30 edges
            - faces (np.ndarray): indexes of the 20 faces 
        """
        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell = {self.nShell}")
        else:            
            scale = self.interShellDistance * i
            # Define vertex positions based on cuboctahedral geometry
            CoordVertices = [ pyNMBu.vertex(-1, 1, 0, scale),\
                              pyNMBu.vertex( 1, 1, 0, scale),\
                              pyNMBu.vertex(-1,-1, 0, scale),\
                              pyNMBu.vertex( 1,-1, 0, scale),\
                              pyNMBu.vertex( 0,-1, 1, scale),\
                              pyNMBu.vertex( 0, 1, 1, scale),\
                              pyNMBu.vertex( 0,-1,-1, scale),\
                              pyNMBu.vertex( 0, 1,-1, scale),\
                              pyNMBu.vertex( 1, 0,-1, scale),\
                              pyNMBu.vertex( 1, 0, 1, scale),\
                              pyNMBu.vertex(-1, 0,-1, scale),\
                              pyNMBu.vertex(-1, 0, 1, scale) ]
            CoordVertices = np.array(CoordVertices)
            edges = [( 4, 2), ( 4, 3), ( 5, 0), ( 5, 1), ( 6, 2), ( 6, 3), ( 7, 0), ( 7, 1),\
                     ( 8, 1), ( 8, 3), ( 8, 6), ( 8, 7), ( 9, 1), ( 9, 3), ( 9, 4), ( 9, 5),\
                     ( 10, 0), ( 10, 2), ( 10, 6), ( 10, 7), ( 11, 0), ( 11, 2), ( 11, 4), ( 11, 5)]
            faces3 = [( 4, 2, 11), ( 4, 3, 9), ( 5, 0, 11), ( 5, 1, 9), ( 6, 2, 10), ( 6, 3, 8), ( 7, 0, 10), ( 7, 1, 8)]
            faces4 = [( 2, 4, 3, 6), ( 0, 5, 1, 7), ( 1, 8, 3, 9), ( 6, 8, 7, 10), ( 4, 9, 5, 11), ( 0, 10, 2, 11)]
            edges = np.array(edges)
            faces3 = np.array(faces3)
            faces4 = np.array(faces4)
            # print("len = ",len(CoordVertices))
            # for i in range(len(CoordVertices)):
            #     print("i, CoordV ",i,CoordVertices[i])
        return CoordVertices, edges, faces3, faces4

    def coords(self,noOutput):
        """
        Generate the coordinates of a cuboctahedral nanoparticle.
        
        Args:
        - noOutput (bool): If True, suppresses output messages.
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
        indexFace3Atoms = []
        indexFace4Atoms = []
        
        #  Generate vertex atoms 
        for i in range(1,self.nShell+1):
            # vertices
            nAtoms0 = self.nAtoms
            cshell, E, F3, F4 = self.MakeVertices(i)
            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

            # Generate edge atoms
            nAtoms0 = self.nAtoms
            # print("nAtoms0 = ",nAtoms0)
            Rvv = pyNMBu.RAB(cshell,E[0,0],E[0,1]) #distance between two vertex atoms
            nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
            nIntervals = nAtomsOnEdges + 1
            # print("nAtomsOnEdges = ",nAtomsOnEdges,"  len(E) = ",len(E))
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
            
            # Generate triangular facet atoms
            coordFace3At = []
            nAtomsOnFaces3 = 0
            nAtoms0 = self.nAtoms
            for f in F3:
                nAtomsOnFaces3,coordFace3At = pyNMBu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces3,coordFace3At)
            self.nAtomsPerShell[i] += nAtomsOnFaces3
            self.nAtoms += nAtomsOnFaces3
            c.extend(coordFace3At)
            indexFace3Atoms.extend(range(nAtoms0,self.nAtoms))

            # Generate square facet atoms
            coordFace4At = []
            nAtomsOnFaces4 = 0
            nAtoms0 = self.nAtoms
            for f in F4:
                nAtomsOnFaces4,coordFace4At = pyNMBu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces4,coordFace4At)
            self.nAtomsPerShell[i] += nAtomsOnFaces4
            self.nAtoms += nAtomsOnFaces4
            c.extend(coordFace4At)
            indexFace4Atoms.extend(range(nAtoms0,self.nAtoms))

        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput: print(self.nAtomsPerShell)
        # Store results in an ASE Atoms object
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)
            
        # print(indexVertexAtoms)
        # print(indexEdgeAtoms)
        # print(indexFaceAtoms)
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        
    
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
        print("Dual polyhedron: rhombic dodecahedron")

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
        self.moi=pyNMBu.moi(self.NP, noOutput=noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape 
        pyNMBu.MOI_shapes(self, noOutput)
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)
###########################################################################################################
class fccTrTd:
    """
    A class for generating XYZ and CIF files of fcc truncated tetrahedral nanoparticles (NPs) 
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
    nFaces3 = 4
    nFaces6 = 4
    nEdges = 18
    nVertices = 12
    edgeLengthF = 1 # length of an edge
    radiusCSF = edgeLengthF * (1/4)* np.sqrt(22) #Centroid to vertex distance = Radius of circumsphere
    radiusMSF = edgeLengthF * (3/4) * np.sqrt(2) #Radius of midsphere that is tangent to edges
    cutFromVertexAt = 1/3
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayer: int=1,
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
            nLayer (int): Number of layers, also equals to the number of atoms per edge (e.g., `nLayer=2` means 2 atoms per edge).
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.shape (str): Shape 'fccTrTd'
            self.nAtoms (int): Number of atoms in the NP.
            self.dim (array): Dimensions calculated from MOI.
            self.cog (np.array): Center of gravity of the NP.
            self.Tdprop (class instance): Class "regfccTd" of the module "pNP" (creates regular fcc tetrahedron)
            self.interLayerDistance (float): Distance between two layers in Å.
            self.nAtomsPerEdge (int): Number of atoms per edge.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        self.element = element
        self.shape='fccTrTd'
        self.Rnn = Rnn
        self.nAtoms = 0
        self.dim=[0,0,0]
        self.cog = np.array([0., 0., 0.])
        self.nLayer = int(nLayer-1)
        self.Tdprop = pNP.regfccTd(self.element,self.Rnn,self.nLayer+1,noOutput=True,calcPropOnly=True) 
        self.interLayerDistance = self.Tdprop.interLayerDistance()
        isTrTd,self.nAtomsPerEdge = self.NumberOfTdEdgeAtomsValid4ATrTd()
        if not isTrTd:
            listOfPossiblenLayers = self.magicEdgeNumberOfTd2MakeATrTd(int(1.2*nLayer))
            nearest_nL = min(listOfPossiblenLayers, key = lambda x: abs(x-(self.nLayer+1)))
            sys.exit(f"This number of layers cannot yield a perfect truncated tetrahedron.\n"\
                     f"The closest possible nLayer value is {nearest_nL}.\n"\
                     f"Try again."\
                     f"Any doubt about the valid nLayers values? Call the archimedeanNP.magicEdgeNumberOfTd2MakeATrTd(N) "\
                     f"to see all possible values between 1 and N")
        else:
            self.nAtomsPerEdge = int(self.nAtomsPerEdge)
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("trTd-C.png")
        self.trPlanes = None
        if not noOutput: vID.centerTitle(f"Truncated fcc Td, made from a {nLayer} layers Td")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)

    def __str__(self):
        return(f"Truncated tetrahedron based on a {self.nLayer+1} layer(s) Td and Rnn = {self.Rnn}")
   
    def nAtomsF(self,i):
        """
        Returns a factor to compute the number of atoms of a nanoparticle of size i.
        """
        return round((i+1)*(23*i**2 + 19*i + 6)/6)

    def nAtomsAnalytic(self):
        """
        Computes the total number of atoms in the nanoparticle.
        """
        n = self.nAtomsF(self.nAtomsPerEdge-1)
        return n

    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .

        The edge length of the truncated tetrahedron equals the regular tetrahedron's edge length divided by 3.
        """
        # a truncated Td is constructed by truncating all 4 vertices of a regular tetrahedron
        # at one third of the original edge length, hence the remaining edge length is Td.edgeLength/3
        return self.Tdprop.edgeLength()/3
        #return self.edgeLength-int(self.nAtomsPerEdge/2)


    
    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength()

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
        return el**2*7*np.sqrt(3)
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return el**3*(23/12)*np.sqrt(2)

    def magicEdgeNumberOfTd2MakeATrTd(self, index: int):
        '''
        Returns the number of edge atoms of the tetrahedron that will lead to perfect
        trucated tetrahedron with all edges of equal atomic length.
        '''
        import numpy as np
        N = []
        for i in range(1,index+1):
            N.append(3*i-2)
        return np.array(N)
    
    def NumberOfTdEdgeAtomsValid4ATrTd(self):
        """
        Calculate the number of edge atoms in a truncated tetrahedral nanoparticle
        and check if it is a valid integer.
        
        Returns:
        - (bool, float): A tuple where the first element is True if the number is an integer,
          and the second element is the computed number of edge atoms.
        """
        import numpy as np
        N = self.nLayer+1
        nTrTd = N - 2*(N-1)*self.cutFromVertexAt
        return nTrTd.is_integer(), nTrTd

    # def calculateTruncationPlanes(self, planes, nAtomsPerEdge, debug=False):
    #     cutFromVertexAt = (12*n-16)/(12*n)
    #     print(f"factor = {cutFromVertexAt:.3f} ▶ {round(nAtomsPerEdge/n)} layer(s) will be removed, starting from each vertex")

    #     trPlanes = []
    #     for p in planes:
    #         pNormalized = p.copy()
    #         p[3] = p[3]*cutFromVertexAt
    #         trPlanes.append(p)
    #         hkld = pyNMBu.convertuvwh2hkld(p,False)
    #         if (debug):
    #             print("original plane = ",pNormalized,"... norm = ",pyNMBu.normV(pNormalized[0:3]))
    #             print("cut plane = ",p,"... norm = ",pyNMBu.normV(p[0:3]))
    #             hkldRef = pyNMBu.convertuvwh2hkld(pNormalized,False)
    #             print("hkld[3]*factor = ",p[3])
    #             print("signed distance between original hkld and origin = ",pyNMBu.Pt2planeSignedDistance(hkldRef,[0,0,0]))
    #             print("signed distance between cut plane and origin = ",pyNMBu.Pt2planeSignedDistance(hkld,[0,0,0]))
    #             print("pcut/pRef = ",pyNMBu.Pt2planeSignedDistance(hkld,[0,0,0])/pyNMBu.Pt2planeSignedDistance(hkldRef,[0,0,0]))
    #         print(f"Will remove atoms just above plane {hkld[0]:.2f} {hkld[1]:.2f} {hkld[2]:.2f} d:{hkld[3]:.3f}")
    #     return np.array(trPlanes)

    def coords(self,noOutput):
        """
        Generate the coordinates of a truncated tetrahedral nanoparticle.
        
        Args:
        - noOutput (bool): If True, suppresses output messages.
        """
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pyNMBu.timer(); chrono.chrono_start()
        if not noOutput: vID.centertxt("Generation of the coordinates of the tetrahedron",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        
        # Generate a regular fcc tetrahedron using the class "regfccTd" of the "pNP" module
        Td = pNP.regfccTd(self.element,self.Rnn,self.nLayer+1,postAnalyzis=False,noOutput=True)
        self.NP0 = Td.NP.copy()
        aseTd = Td.NP
        if not noOutput: vID.centertxt("Removing atoms ",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: print('First searching for the coordinates of the vertices (atoms 1-4) and of the cog')
        coordVertices = aseTd.get_positions()[0:4]
        if not noOutput: print("Now calculating the coordinates of the planes orthogonal the the cog-vertex directions")
        planesAtVertices = pyNMBu.planeAtVertices(coordVertices, Td.cog)
        
        # Generate the truncature: trTd = truncation all 4 vertices of a regular tetrahedron at one third of the original edge length
        trPlanes = pyNMBu.calculateTruncationPlanesFromVertices(planesAtVertices,self.cutFromVertexAt,\
                                                                Td.nAtomsPerEdge-1,noOutput=noOutput)
        #AtomsAbovePlanes = pyNMBu.truncateAboveEachPlane(trPlanes,aseTd.get_positions())
        AtomsAbovePlanes = pyNMBu.truncateAbovePlanes(trPlanes,aseTd.get_positions(),noOutput=noOutput)
        
        aseTrTd = aseTd.copy() 
        del aseTrTd[AtomsAbovePlanes]
        nAtoms = aseTrTd.get_global_number_of_atoms()
        if not noOutput: print(f"Total number of atoms = {nAtoms}")
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseTrTd
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        

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
        print("number of triangular faces = ",self.nFaces3)
        print("number of hexagonal faces = ",self.nFaces6)
        print(f"truncation all 4 vertices of a regular tetrahedron at {self.cutFromVertexAt:.3f}a_Td from the vertices of the Td")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"inter layer distance = {self.interLayerDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("dual polyhedron: triakis tetrahedron")
        print(f"coordinates of the center of gravity = {self.cog}")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface, noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.
    
        This function calculates moments of inertia (MOI), the inscribed and circumscribed spheres, 
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
        self.moi=pyNMBu.moi(self.NP, noOutput=noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape 
        pyNMBu.MOI_shapes(self, noOutput)
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        pyNMBu.Inscribed_circumscribed_spheres(self,noOutput)
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)




###########################################################################################################
class fccTrOh:
    """
    A class for generating XYZ and CIF files of fcc truncated octahedral nanoparticles (NPs) 
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
    nFaces4 = 6
    nFaces6 = 8
    nEdges = 36
    nVertices = 24
    edgeLengthF = 1 # length of an edge
    radiusCSF = edgeLengthF * (1/2)* np.sqrt(10) # Centroid to vertex distance = Radius of circumsphere
    radiusMSF = edgeLengthF * (3/2)              # Radius of midsphere that is tangent to edges
    radiusISF = edgeLengthF * (9/20) * np.sqrt(10) # inradius
    cutFromVertexAt = 1/3
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nOrder: int=1,
                 postAnalyzis: bool=True,
                 aseView: bool=False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 jmolCrystalShape: bool=True,
                 noOutput: bool=False,
                 calcPropOnly: bool=False,
                ):
        """
        Initialize the class with all necessary parameters.

        Args:
            element: Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nOrder (int): Determines the level of imbrication = the number of atomic layers along an edge (e.g., `nOrder=1` means 2 atoms per edge).
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.Ohprop (class instance): class "regfccOh" of the module "pNP"
            self.self.cog (np.array): Center of gravity of the NP.
            self.nAtoms (int): Number of atoms in the NP.
            self.dim (array): Dimensions calculated from MOI.
            self.nAtomsPerEdge (int): Number of atoms per edge.
            self.interLayerDistance (float): Distance between atomic layers.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.cog (np.array): Center of gravity of the NP.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        self.Ohprop = pNP.regfccOh(element,Rnn,nOrder,noOutput=True,calcPropOnly=True)
        self.element = element
        self.shape='fccTrOh'
        self.Rnn = Rnn
        self.nAtoms = 0
        self.dim=[0,0,0]
        self.interLayerDistance = self.Ohprop.interLayerDistance
        self.cog = np.array([0., 0., 0.])
        self.nOrder = nOrder
        isTrOh,self.nAtomsPerEdge = self.NumberOfOhEdgeAtomsValid4ATrOh()
        if not isTrOh:
            listOfPossiblenLayers = self.magicEdgeNumberOfOh2MakeATrOh(int(1.2*nOrder))
            nearest_nL = min(listOfPossiblenLayers, key = lambda x: abs(x-(self.nOrder)))
            sys.exit(f"This order cannot yield a perfect truncated octahedron.\n"\
                     f"The closest possible nOrder value is {nearest_nL}.\n"\
                     f"Try again."\
                     f"Any doubt about the valid nOrder values? Call the archimedeanNP.magicEdgeNumberOfOh2MakeATrOh(N) "\
                     f"to see all possible values between 1 and N")
        else:
            self.nAtomsPerEdge = int(self.nAtomsPerEdge)
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("trOh-C.png")
        self.trPlanes = None
        if not noOutput: vID.centerTitle(f"Truncated fcc octahedron, made from a {nOrder}th order Oh")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Truncated octahedron based on a {self.nOrder} order Oh (i.e. {self.nOrder+1} atoms lie on an edge) and Rnn = {self.Rnn}")
   
    def nAtomsF(self,i):
        """ Returns a factor to compute the number of atoms of a nanoparticle of size i """
        return round(16*i**3 + 15*i**2 + 6*i + 1)

    def nAtomsAnalytic(self):
        """
        Computes the total number of atoms in the nanoparticle.
        """
        n = self.nAtomsF(self.nAtomsPerEdge-1)
        return n

    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .
        The edge length of the truncated octahedron equals the regular octahedron's edge length divided by 3.
        """
        # a truncated Oh is constructed by truncating all 6 vertices of a regular octahedron
        # at one third of the original edge length, hence the remaining edge length is Oh.edgeLength/3
        return self.Ohprop.edgeLength()/3

    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength()

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
        return el**2*(6 + 12*np.sqrt(3))
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return self.Ohprop.volume() - 6*(el**3*np.sqrt(2)/6)

    def magicEdgeNumberOfOh2MakeATrOh(self, index: int):
        '''
        Returns the number of edge atoms of the octahedron that will lead to perfect
        trucated tetrahedra with all edges of equal atomic length
        '''
        import numpy as np
        N = []
        for i in range(1,index+1):
            N.append(3*i)
        return np.array(N)
    
    def NumberOfOhEdgeAtomsValid4ATrOh(self):
        """
        Calculate the number of edge atoms in a truncated octahedral nanoparticle
        and check if it is a valid integer.
        
        Returns:
        - (bool, float): A tuple where the first element is True if the number is an integer,
          and the second element is the computed number of edge atoms.
        """
        import numpy as np
        N = self.nOrder+1
        nTrOh = N - 2*(N-1)/3
        return nTrOh.is_integer(), nTrOh

    def coords(self,noOutput):
        """
        Generate the coordinates of a truncated tetrahedral nanoparticle.
        
        Args:
        - noOutput (bool): If True, suppresses output messages.
        """
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pyNMBu.timer(); chrono.chrono_start()
        if not noOutput: vID.centertxt("Generation of the coordinates of the octahedron",bgc='#cbcbcb',size='12',fgc='b',weight='bold')

        # Generate a regular fcc octahedron using the class "regfccOh" of the module "pNP"
        Oh = pNP.regfccOh(self.element,self.Rnn,self.nOrder,postAnalyzis=False,noOutput=True)
        self.NP0 = Oh.NP.copy()
        aseOh = Oh.NP
        if not noOutput: vID.centertxt("Removing atoms ",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: print('First searching for the coordinates of the vertices (atoms 1-6) and of the cog')
        coordVertices = aseOh.get_positions()[0:6]
        if not noOutput: print("Now calculating the coordinates of the planes orthogonal the the cog-vertex directions")
        planesAtVertices = pyNMBu.planeAtVertices(coordVertices, Oh.cog)
        
        # Generate truncation
        #trOh = truncation all 6 vertices of a regular octahedron at one third of the original edge length
        trPlanes = pyNMBu.calculateTruncationPlanesFromVertices(planesAtVertices,self.cutFromVertexAt,\
                                                                Oh.nAtomsPerEdge,noOutput=noOutput)
        AtomsAbovePlanes = pyNMBu.truncateAboveEachPlane(trPlanes,aseOh.get_positions())
        
        del aseOh[AtomsAbovePlanes]
        nAtoms = aseOh.get_global_number_of_atoms()
        if not noOutput: print(f"Total number of atoms = {nAtoms}")
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseOh
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        

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
        print("number of square faces = ",self.nFaces4)
        print("number of hexagonal faces = ",self.nFaces6)
        print(f"truncation all 6 vertices of a regular octahedron at {self.cutFromVertexAt:.3f}a_Oh from the vertices of the Oh")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"inter layer distance = {self.interLayerDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the medium sphere = {self.radiusMidSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("dual polyhedron: triakis hexahedron")
        print(f"coordinates of the center of gravity = {self.cog}")

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
        self.moi=pyNMBu.moi(self.NP, noOutput=noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape 
        pyNMBu.MOI_shapes(self, noOutput)
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)
###########################################################################################################
class fccTrCube:
    """
    A class for generating XYZ and CIF files of fcc truncated cubic nanoparticles (NPs) 
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
    nFaces3 = 8
    nFaces8 = 6
    nEdges = 36
    nVertices = 24
    edgeLengthF = 1 # length of an edge
    radiusCSF = edgeLengthF * (1/2) * np.sqrt(7+4*np.sqrt(2)) # Centroid to vertex distance = Radius of circumsphere
    radiusMSF = edgeLengthF * (1/2) * (2+np.sqrt(2))             # Radius of midsphere that is tangent to edges
    radiusISF = edgeLengthF * (1/17) * (5+2*np.sqrt(2)) * np.sqrt(7+4*np.sqrt(2))# inradius
    cutFromVertexAt = 0.15
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nOrder: int=1,
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
            nOrder (int): Determines the level of imbrication = the number of atomic layers along an edge (e.g., `nOrder=1` means 2 atoms per edge).
            shape (str): Shape 'cube'.
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters, moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.   
            
        Attributes:
            self.shape (str): Shape 'fccTrCube'.
            self.cubeProp (class instance): Class "cube" of the module "pNP".
            self.nAtoms (int): Number of atoms in the NP.
            self.dim (array): Dimensions calculated from MOI.
            self.nAtomsPerEdge (int): Number of atoms per edge.
            self.interLayerDistance (float): Distance between atomic layers.
            self.jmolCrystalShape (bool): Flag for JMol visualization.
            self.cog (np.array): Center of gravity of the NP.
            self.imageFile (str): Path to a reference image.
            self.trPlanes (array): Truncation plane equations.

        """
        self.cubeProp = pNP.cube('fcc',element,Rnn,nOrder,noOutput=True,calcPropOnly=True)
        self.element = element
        self.shape='fccTrCube'
        self.Rnn = Rnn
        self.nAtoms = 0
        self.dim=[0,0,0]
        self.cog = np.array([0., 0., 0.])
        self.nOrder = nOrder
        if not noOutput: vID.centerTitle(f"fcc truncated cube")
        self.jmolCrystalShape = jmolCrystalShape
        self.trPlanes = None
        self.imageFile = pyNMBu.imageNameWithPathway("trOh-C.png")
        isTrCube,self.nAtomsPerEdge = self.NumberOfCubeEdgeAtomsValid4ATrCube()
        if not isTrCube:
            listOfPossiblenLayers = self.magicEdgeNumberOfCube2MakeATrCube(int(1.2*nOrder))
            nearest_nL = min(listOfPossiblenLayers, key = lambda x: abs(x-(self.nOrder)))
            sys.exit(f"This order cannot yield a perfect truncated octahedron.\n"\
                     f"The closest possible nOrder value is {nearest_nL}.\n"\
                     f"Try again."\
                     f"Any doubt about the valid nOrder values? Call the archimedeanNP.magicEdgeNumberOfCube2MakeATrCube(N) "\
                     f"to see all possible values between 1 and N")
        else:
            self.nAtomsPerEdge = int(self.nAtomsPerEdge)
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Truncated cube based on a {self.nOrder}x{self.nOrder}x{self.nOrder} cube (i.e. {self.nOrder+1} atoms lie on an edge) and Rnn = {self.Rnn}")
   
    def nAtomsF(self,i):
        """ Returns a factor to compute the number of atoms of a nanoparticle of size i """
        return round(4*i**3 + 6*i**2 + 3*i - 7)

    def nAtomsAnalytic(self):
        """
        Computes the total number of atoms of the nanoparticle.
        """
        n = self.nAtomsF(self.nAtomsPerEdge-1)
        return n

    def edgeLength(self):
        """
        Computes the edge length of the nanoparticle in Å .
        The edge length of the truncated cube equals the regular cube's edge length divided by 3.
        """
        # a truncated Oh is constructed by truncating all 6 vertices of a regular octahedron
        # at one third of the original edge length, hence the remaining edge length is Oh.edgeLength/3
        return self.cubeProp.edgeLength()/3

    def radiusCircumscribedSphere(self):
        """
        Computes the radius of the circumscribed sphere of the nanoparticle in Å.
        """
        return self.radiusCSF*self.edgeLength()

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
        return el**2 * 2 * (6 + 6*np.sqrt(2) + np.sqrt(3))
    
    def volume(self):
        """
        Computes the volume of the nanoparticle in cubic Ångströms.
        """
        el = self.edgeLength()
        return el**3 * (1/3) * (21 + 14*np.sqrt(2))

    def magicEdgeNumberOfCube2MakeATrCube(self, index: int):
        '''
        Returns the number of edge atoms of the octahedron that will lead to perfect
        trucated tetrahedra with all edges of equal atomic length.
        '''
        import numpy as np
        N = []
        for i in range(3,index+1):
            N.append(3*i)
        return np.array(N)
    
    def NumberOfCubeEdgeAtomsValid4ATrCube(self):
        """
        Calculate the number of edge atoms in a truncated fcccubic nanoparticle
        and check if it is a valid integer.
        
        Returns:
        - (bool, float): A tuple where the first element is True if the number is an integer,
          and the second element is the computed number of edge atoms.
        """
        import numpy as np
        N = self.nOrder+1
        nTrCube = N - 2*(N-1)/3
        return nTrCube.is_integer(), nTrCube

    def coords(self,noOutput):
        """
        Generate the coordinates of a truncated fcc cubic nanoparticle.
        
        Args:
        - noOutput (bool): If True, suppresses output messages.
        """
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        def findVertices(c):
            eps = 1.e-4
            max = np.max(c)
            indexV = (np.where((np.abs(np.abs(c[:,0]) - max) < eps) &\
                     (np.abs(np.abs(c[:,1]) - max) < eps) & \
                     (np.abs(np.abs(c[:,2]) - max) < eps)))
            coordV = c[indexV]
            return indexV, coordV
        chrono = pyNMBu.timer(); chrono.chrono_start()
        if not noOutput: vID.centertxt("Generation of the coordinates of the cube",bgc='#cbcbcb',size='12',fgc='b',weight='bold')

        # Generate a regular fcc cube using the class "cube" of the module "pNP".
        aseCube = pNP.cube('fcc',self.element,self.Rnn,self.nOrder,noOutput=True,postAnalyzis=False).NP
        if not noOutput: vID.centertxt("Cube moved to origin",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        c2cog = pyNMBu.center2cog(aseCube.get_positions())
        aseCube.set_positions(c2cog)
        self.NP0 = aseCube.copy()
        if not noOutput: vID.centertxt("Removing atoms ",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: print('First searching for the coordinates of the vertices and of the cog')
        indexV, coordVertices = findVertices(aseCube.get_positions())
        if not noOutput: print("Vertices = atoms ",indexV)
        if not noOutput: print("Now calculating the coordinates of the planes orthogonal the the cog-vertex directions")
        planesAtVertices = pyNMBu.planeAtVertices(coordVertices, self.cog)

        # Generate the truncation
        #trCube = truncation all 6 vertices of a regular octahedron at one third of the original edge length
        trPlanes = pyNMBu.calculateTruncationPlanesFromVertices(planesAtVertices,self.cutFromVertexAt,\
                                                                self.cubeProp.nAtomsPerEdge,noOutput=noOutput)
        AtomsAbovePlanes = pyNMBu.truncateAboveEachPlane(trPlanes,aseCube.get_positions())
        
        aseTrCube = aseCube.copy()
        del aseTrCube[AtomsAbovePlanes]
        nAtoms = aseTrCube.get_global_number_of_atoms()
        self.nAtoms=nAtoms
        if not noOutput: print(f"Total number of atoms = {nAtoms}")
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseTrCube
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        

    def prop(self):
        """
        Display unit cell and nanoparticle properties.
        """
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        # pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of trigonal faces = ",self.nFaces3)
        print("number of octogonal faces = ",self.nFaces8)
        print(f"truncation all 6 vertices of a regular octahedron at {self.cutFromVertexAt:.3f}a_cube from the vertices of the cube")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the medium sphere = {self.radiusMidSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        # print("total number of atoms = ",self.nAtomsAnalytic())
        print("dual polyhedron: Triakis octahedron")
        print(f"coordinates of the center of gravity = {self.cog}")

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
        self.moi=pyNMBu.moi(self.NP, noOutput=noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)
        self.MOIshape=self.shape 
        pyNMBu.MOI_shapes(self, noOutput)
        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)