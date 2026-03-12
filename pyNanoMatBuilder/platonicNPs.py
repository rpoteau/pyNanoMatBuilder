# External dependencies
import os
import sys
import numpy as np
import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view
from ase.cluster.cubic import FaceCenteredCubic

# Internal Relative Imports
from .visualID import fg, hl, bg
from . import visualID as vID
from . import utils as pyNMBu

###############################################################################
class PlatonicNP:
    """Base class for all Platonic nanoparticles providing common functionality."""

    def propPostMake(self, skipSymmetryAnalyzis,
                     thresholdCoreSurface, noOutput):
        """
        Compute and store various post-construction
        properties of the nanoparticle.

        This function calculates moments of inertia
        (MOI), determines the nanoparticle shape,
        analyzes symmetry (if required), and identifies
        core and surface atoms.

        Args:
            skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
            thresholdCoreSurface (float): Threshold
                to distinguish core and surface atoms.
            noOutput (bool): If True, suppresses
                output messages.
    
        Attributes:
            moi (numpy.ndarray): Moment of inertia tensor.
            moisize (numpy.ndarray): Normalized moments of inertia.
            vertices (numpy.ndarray): Geometric vertices of the nanoparticle.
            simplices (numpy.ndarray): Simplices defining the convex hull.
            neighbors (numpy.ndarray): Neighboring relations between facets.
            equations (numpy.ndarray): Plane equations for the hull faces.
            NPcs (ase.Atoms): Copy of the nanoparticle with surface atoms 
                visually marked.
            NP (ase.Atoms): Original nanoparticle object.
            sasview_dims (tuple, optional): Dimensions for SasView, calculated 
                only if the sasview_dims() method exists.
        """
        
        self.moi = pyNMBu.moi(self.NP, noOutput)
        # MOI mass normalized (m of each atoms=1)
        self.moisize = np.array(
            pyNMBu.moi_size(self.NP, noOutput)
        )
        
        # Specific print for hollow_shapes in
        # original code, maybe generic?
        if isinstance(self, hollow_shapes):
            print(self.moi)

        if not skipSymmetryAnalyzis:
            pyNMBu.MolSym(self.NP, noOutput=noOutput)

        (
            [self.vertices, self.simplices,
             self.neighbors, self.equations],
            surfaceAtoms
        ) = pyNMBu.coreSurface(
            self, thresholdCoreSurface,
            noOutput=noOutput
        )

        self.NPcs = self.NP.copy()
        # Nobelium, because it has a nice pinkish
        # color in jmol
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102
        self.surfaceatoms = self.NPcs[surfaceAtoms]

        # Specific update for hollow_shapes
        # in original code
        if isinstance(self, hollow_shapes):
            self.cog = self.NP.get_center_of_mass()

        if hasattr(self, 'trPlanes') and self.trPlanes is not None:
            self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)

        if (hasattr(self, 'jmolCrystalShape')
                and self.jmolCrystalShape):
            # do not print the jmol script
            self.jMolCS = pyNMBu.defCrystalShapeForJMol(
                self, noOutput=True
            )
        
        # Specific print for regfccTd helix
        if (hasattr(self, 'n_tetrahedrons')
                and self.n_tetrahedrons > 1
                and not noOutput):
            # Just checking attribute existence
            # to avoid error on other classes
            if (hasattr(self, 'nAtomsAnalytic')
                    and hasattr(self, 'nAtoms_helix')):
                print(f"\n{'=' * 60}")
                print(f"Helix Information:")
                print(
                    f"  Number of tetrahedrons"
                    f" in helix:"
                    f" {self.n_tetrahedrons}"
                )
                print(
                    f"  Atoms per single"
                    f" tetrahedron:"
                    f" {self.nAtomsAnalytic()}"
                )
                print(
                    f"  Total atoms in helix:"
                    f" {self.nAtoms_helix}"
                )
                print(f"{'=' * 60}\n")


        # Specific fix for regfccOh which has a
        # sasview_dims method that returns dimensions
        # and overwrites itself. We check if the method
        # exists in the class definition.
        if (hasattr(self, 'sasview_dims')
                and callable(
                    getattr(self, 'sasview_dims'))):
            # Check if it's still a method
            # (hasn't been overwritten yet)
            try:
                self.sasview_dims = self.sasview_dims()
                if not noOutput:
                    print(f"{'=' * 60}\n")
                    print(
                        f"SasView dimensions (for"
                        f" comparaison purposes when"
                        f" comparing to SasView"
                        f" models):"
                    )
                    print(
                        f"  t = {self.sasview_dims[1]}"
                        ", t being the truncature"
                        " that is defined by the"
                        " ratio d(truncated_demi"
                        "_axis)/d(demi_axis)"
                    )
                    print(
                        f"  a = {self.sasview_dims[0]}"
                        " Angs, a being the"
                        " demi_axis being the"
                        " distance from the center"
                        " of the octahedron to a"
                        " vertice (in Å)):"
                        f" {self.sasview_dims}"
                    )
                    print(f"{'=' * 60}\n")
            except TypeError:
                # If it's not callable, it might have
                # already been overwritten or is a
                # property
                pass

###############################################################################
class regfccOh(PlatonicNP):
    """A class for generating XYZ and CIF files
    of regular fcc octahedral nanoparticles (NPs).

    Generates NPs of various sizes, based on user-defined compounds (either by
    name, e.g., "Fe", "Au", etc).

    Key Features:
        - Allows to choose the NP size.
        - Can analyze the structure in detail, including symmetry and properties.
        - Offers options for core/surface differentiation based on a threshold.
        - Generates outputs in XYZ and CIF formats for visualization and simulations.
        - Provides compatibility with jMol for 3D visualization.

    Additional Notes:
        - The `nOrder` parameter determines the level of imbrication.
        - The symmetry analysis can be skipped to speed up computations.
        - Customizable precision thresholds for structural analysis.
    """

    # Geometric properties of regfccOh
    nFaces = 8  # Number of triangular faces
    nEdges = 12
    nVertices = 6
    edgeLengthF = 1  # length of an edge
    # Centroid to vertex distance
    # = Radius of circumsphere
    radiusCSF = edgeLengthF * np.sqrt(2) / 2
    # Radius of insphere tangent to faces
    radiusISF = edgeLengthF * np.sqrt(6) / 6
    # Radius of midsphere tangent to edges
    radiusMSF = edgeLengthF / 2
    # Angle between two adjacent triangular faces
    dihedralAngle = np.rad2deg(np.arccos(-1 / 3))
    interShellF = 1 / radiusCSF

    def __init__(self,
                 element: str = 'Au',
                 Rnn: float = 2.7,
                 nOrder: int = 1,
                 shape: str = 'regfccOh',
                 postAnalyzis: bool = True,
                 aseView: bool = False,
                 thresholdCoreSurface: float = 1.,
                 skipSymmetryAnalyzis: bool = False,
                 jmolCrystalShape: bool = True,
                 noOutput: bool = False,
                 calcPropOnly: bool = False,
                 ):
        """Initialize the class with all necessary parameters.

        Args:
            element (str): Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nOrder (int): Determines the level of imbrication = the number
                of atomic layers along an edge (e.g., ``nOrder=1`` means
                2 atoms per edge).
            shape (str): Shape 'regfccOh'.
            postAnalyzis (bool): If True, prints additional NP information
                (e.g., cell parameters, moments of inertia,
                inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface
                differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic
                structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script
                for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure
                of the NP.

        Attributes:
            nAtoms (int): Number of atoms in the NP.
            nAtomsPerLayer (list): Number of atoms in each atomic layer.
            nAtomsPerEdge (int): Number of atoms per edge.
            interLayerDistance (float): Distance between atomic layers.
            jmolCrystalShape (bool): Flag for JMol visualization.
            cog (np.array): Center of gravity of the NP.
            imageFile (str): Path to a reference image.
            trPlanes (array): Truncation plane equations.
        """
        self.element = element
        self.shape = shape
        self.Rnn = Rnn
        self.nOrder = nOrder
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.nAtomsPerEdge = self.nOrder + 1
        self.interLayerDistance = self.Rnn / self.interShellF
        self.jmolCrystalShape = jmolCrystalShape
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pyNMBu.imageNameWithPathway("fccOh-C.png")
        self.trPlanes = None
        if not noOutput:
            vID.centerTitle(f"{nOrder}th order regular fcc Octahedron")

        if not noOutput:
            self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput)
                if aseView:
                    view(self.NPcs)

    def __str__(self):
        """Returns a string representation of the object."""
        return (
            f"Regular octahedron of order {self.nOrder} (i.e. {self.nOrder + 1}"
            f" atoms lie on an edge) and Rnn = {self.Rnn}"
        )

    def nAtomsF(self, i):
        """Returns the number of atoms of an octahedron of size i.

        Args:
            i (int): The order or size parameter.

        Returns:
            int: The calculated number of atoms.
        """
        return round((2 / 3) * i**3 + 2 * i**2 + (7 / 3) * i + 1)

    def nAtomsPerShellAnalytic(self):
        """Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer (from 1 to `nOrder`),
        computes the number of atoms for the given shell, and subtracts
        the cumulative sum of the previous shells to get the number of new
        atoms in the current shell.

        Returns:
            list: A list where each element represents the number of atoms
                  in a specific shell.
        """
        n = []
        current_sum = 0
        for i in range(1, self.nOrder + 1):
            ni = self.nAtomsF(i)  # natoms in the whole octahedron of order i
            n_shell = ni - current_sum
            n.append(n_shell)
            current_sum += n_shell  # Update running sum
        return n

    def nAtomsPerShellCumulativeAnalytic(self):
        """Computes the cumulative number of atoms up to each shell.

        This function returns the total number of atoms present in the
        nanoparticle for each shell layer, building up cumulatively.

        Returns:
            list: A list where each element represents the total number of
                  atoms present up to that shell.
        """
        n = []
        Sum = 0
        for i in range(1, self.nOrder + 1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n

    def nAtomsAnalytic(self):
        """Computes the total number of atoms in the nanoparticle.

        Returns:
            int: Total number of atoms.
        """
        n = self.nAtomsF(self.nOrder)
        return n

    def edgeLength(self):
        """Computes the edge length of the nanoparticle in Å.

        The edge length is determined based on the interatomic distance (Rnn)
        and the number of atomic layers (`nOrder`).

        Returns:
            float: The edge length in Å.
        """
        return self.Rnn * self.nOrder  # Angs

    def radiusCircumscribedSphere(self):
        """Computes the radius of the circumscribed sphere of the nanoparticle in Å.

        Returns:
            float: Radius.
        """
        return self.radiusCSF * self.edgeLength()  # angs

    def radiusInscribedSphere(self):
        """Computes the radius of the inscribed sphere of the nanoparticle in Å.

        Returns:
            float: Radius.
        """
        return self.radiusISF * self.edgeLength()

    def radiusMidSphere(self):
        """Computes the radius of the midsphere of the nanoparticle in Å.

        The midsphere is a sphere that touches the edges of the nanoparticle.

        Returns:
            float: Radius.
        """
        return self.radiusMSF * self.edgeLength()

    def area(self):
        """Computes the surface area of the nanoparticle in square Ångströms.

        Returns:
            float: Surface area.
        """
        el = self.edgeLength()
        return el**2 * np.sqrt(3)

    def volume(self):
        """Computes the volume of the nanoparticle in cubic Ångströms.

        Returns:
            float: Volume.
        """
        el = self.edgeLength()
        return np.sqrt(2) * el**3 / 3

    def MakeVertices(self, i):
        """
        Generates the coordinates of the vertices, edges, and faces
        for the ith shell of an octahedral nanoparticle.

        Args:
            i (int): Index of the shell layer.

        Returns:
            tuple:
                - CoordVertices (np.ndarray): the 6 vertex coordinates
                    of the ith shell of an octahedron
                - edges (np.ndarray): indexes of the 30 edges
                - faces (np.ndarray): indexes of the 20 faces
                
        """
        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.]  # Central atom at the origin
            edges = []
            faces = []

        elif (i > self.nOrder):
            sys.exit(
                f"regfccOh.MakeVertices(i) is called"
                f" with i = {i} > nOrder = {self.nOrder}"
            )

        else:
            scale = self.interLayerDistance * i
            # Define vertex positions based on octahedral geometry
            CoordVertices = [pyNMBu.vertex(-1, 0, 0, scale),
                             pyNMBu.vertex(1, 0, 0, scale),
                             pyNMBu.vertex(0, -1, 0, scale),
                             pyNMBu.vertex(0, 1, 0, scale),
                             pyNMBu.vertex(0, 0, -1, scale),
                             pyNMBu.vertex(0, 0, 1, scale)]
            edges = [
                (2, 0), (2, 1), (3, 0), (3, 1),
                (4, 0), (4, 1), (4, 2), (4, 3),
                (5, 0), (5, 1), (5, 2), (5, 3)
            ]
            faces = [
                (2, 0, 4), (2, 0, 5),
                (2, 1, 4), (2, 1, 5),
                (3, 0, 4), (3, 0, 5),
                (3, 1, 4), (3, 1, 5)
            ]
            CoordVertices = np.array(CoordVertices)
            edges = np.array(edges)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self, noOutput):
        """Generates atomic coordinates for an octahedral nanoparticle.

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

        if not noOutput:
            vID.centertxt("Generation of coordinates",
                          bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        c = []  # List of atomic coordinates
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
        indexVertexAtoms.extend(range(nAtoms0, self.nAtoms))

        # Generate edge atoms
        nAtoms0 = self.nAtoms
        # Distance between two vertex atoms
        Rvv = pyNMBu.RAB(cVertices, E[0, 0], E[0, 1])
        nAtomsOnEdges = int((Rvv + 1e-6) / self.Rnn) - 1
        nIntervals = nAtomsOnEdges + 1
        # print("nAtomsOnEdges = ",nAtomsOnEdges)
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E:  # Loop over all edges
                a = e[0]
                b = e[1]
                # Compute interpolated positions
                coordEdgeAt.append(
                    cVertices[a] + pyNMBu.vector(cVertices, a, b)
                    * (n + 1) / nIntervals)
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        indexEdgeAtoms.extend(range(nAtoms0, self.nAtoms))
        # print(indexEdgeAtoms)

        # Generate facet atoms
        coordFaceAt = []
        nAtomsOnFaces = 0
        nAtoms0 = self.nAtoms
        for f in F:
            nAtomsOnFaces, coordFaceAt = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsOnFaces, coordFaceAt)
        self.nAtoms += nAtomsOnFaces
        c.extend(coordFaceAt)
        indexFaceAtoms.extend(range(nAtoms0, self.nAtoms))

        # Generate core atoms
        # Layer by layer strategy, starting from bottom to top when
        # identified, just use MakeFaceCoord and define, for each
        # layer, the four atoms on the edge as a facet
        coordCoreAt = []
        nAtomsInCore = 0
        nAtoms0 = self.nAtoms
        # first apply it to atoms 0, 1, 2, 3
        # f = [a,b,c,d] must be given in the order a--b
        #                                          |  |
        #                                          d--c
        f = np.array([0, 3, 1, 2])
        nAtomsInCore, coordCoreAt = pyNMBu.MakeFaceCoord(
            self.Rnn, f, c, nAtomsInCore, coordCoreAt)
        # don't ask... it is the algorithm to find the indexes
        # of the square corners of each layer along z

        # Helper functions to define atomic layers
        def layerup(ilayer, f):
            return 12 * ilayer + f - 2

        def layerdown(ilayer, f):
            return 12 * ilayer + f + 2

        # Loop to generate multiple layers in the core
        for i in range(1, nAtomsOnEdges + 1):
            f = layerup(i, np.array([0, 3, 1, 2]))
            # print(i,"  fup",f)
            nAtomsInCore, coordCoreAt = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsInCore, coordCoreAt)
            f = layerdown(i, np.array([0, 3, 1, 2]))
            # print(i,"fdown",f)
            nAtomsInCore, coordCoreAt = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsInCore, coordCoreAt)

        self.nAtoms += nAtomsInCore
        c.extend(coordCoreAt)
        indexCoreAtoms.extend(range(nAtoms0, self.nAtoms))

        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")
        # Store results in an ASE Atoms object
        aseObject = ase.Atoms(self.element * self.nAtoms, positions=c)

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()

    def sasview_dims(self):
        """Converts the dimensions for SasView use.

        In pyNanoMatBuilder, the truncature is defined by the ratio d(truncated_edge)/d(edge).
        In Sasview, the truncature is defined by the ratio d(truncated_demi_axis)/d(demi_axis),
        the demi_axis being the distance from the center of the octahedron to a vertice (in Å).

        Returns:
            tuple: (demi_axis, truncature_ratio)
        """
        # Full demi axis
        positions = self.NP.get_positions()
        # Find the distance between two points (x,y,zmax) and (x,y,zmin)
        zmax = max(positions[:, 2])
        zmin = min(positions[:, 2])
        demi_axis = (zmax - zmin) / 2

        truncature_ratio = 1

        return demi_axis, truncature_ratio

    def prop(self):
        """Display unit cell and nanoparticle properties.
        """
        vID.centertxt("Properties", bgc='#007a7a', size='14', weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ", self.element)
        print("number of vertices = ", self.nVertices)
        print("number of edges = ", self.nEdges)
        print("number of faces = ", self.nFaces)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"interlayer distance = {self.interLayerDistance:.2f} Å")
        print(f"edge length = {self.edgeLength() * 0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"radius after volume = "
              f"{pyNMBu.RadiusSphereAfterV(self.volume() * 1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = "
              f"{self.radiusCircumscribedSphere() * 0.1:.2f} nm")
        print(f"radius of the inscribed sphere = "
              f"{self.radiusInscribedSphere() * 0.1:.2f} nm")
        print(f"area = {self.area() * 1e-2:.1f} nm2")
        print(f"volume = {self.volume() * 1e-3:.1f} nm3")
        print(f"dihedral angle = {self.dihedralAngle:.1f}°")
        # print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print(
            "intermediate magic numbers = ",
            self.nAtomsPerShellCumulativeAnalytic()
        )
        print("total number of atoms = ", self.nAtomsAnalytic())
        print("Dual polyhedron: cube")
        print("Indexes of vertex atoms = [0,1,2,3,4,5] by construction")
        print(f"coordinates of the center of gravity = {self.cog}")
        return



###########################################################################################################
class regIco(PlatonicNP):
    """A class for generating XYZ and CIF files of regular icosahedral nanoparticles (NPs).

    Generates NPs of various sizes, based on user-defined compounds (either by
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
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    edgeLengthF = 1  # length of an edge
    radiusCSF = np.sqrt(10 + 2 * np.sqrt(5)) / 4  # Radius of circumsphere
    interShellF = 1 / radiusCSF
    # interShellF = np.sqrt(2*(1-1/np.sqrt(5)))
    radiusISF = np.sqrt(3) * (3 + np.sqrt(5)) / 12  # Radius of insphere

    def __init__(self,
                 element: str = 'Au',
                 Rnn: float = 2.7,
                 nShell: int = 1,
                 shape: str = 'regIco',
                 double_ico: bool = False,
                 postAnalyzis: bool = True,
                 aseView: bool = False,
                 thresholdCoreSurface=1.,
                 skipSymmetryAnalyzis=False,
                 jmolCrystalShape: bool = True,
                 noOutput=False,
                 calcPropOnly=False,
                 ):
        """Initialize the class with all necessary parameters.

        Args:
            element (str): Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nShell (int): Number of shells = the number of atomic layers along an edge
               (e.g., `nOrder=1` means 2 atoms per edge).
            shape (str): Shape 'regIco'.
            double_ico (bool): If True, generates a double icosahedron.
            postAnalyzis (bool): If True, prints additional NP information (e.g., cell parameters,
                moments of inertia, inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface differentiation
                (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure of the NP.

        Attributes:
            nAtoms (int): Number of atoms in the NP.
            nAtomsPerShell (list): Number of atoms in each shell.
            interShellDistance (float): Distance between atomic shells.
            jmolCrystalShape (bool): Flag for JMol visualization.
            imageFile (str): Path to a reference image.
            trPlanes (array): Truncation plane equations.
        """
        self.element = element
        self.shape = shape
        self.Rnn = Rnn
        self.nShell = nShell
        self.double_ico = double_ico
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("ico-C.png")
        self.trPlanes = None
        if not noOutput:
            vID.centerTitle(f"{nShell} shells icosahedron")

        if not noOutput:
            self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput)
                if aseView:
                    view(self.NPcs)

    def __str__(self):
        """Returns a string representation of the object."""
        return f"Regular icosahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}"

    def nAtomsF(self, i):
        """Returns the number of atoms of an icosahedron of size i.

        Args:
            i (int): The shell number or size parameter.

        Returns:
            int: The calculated number of atoms.
        """
        return (10 * i**3 + 11 * i) // 3 + 5 * i**2 + 1

    def nAtomsPerShellAnalytic(self):
        """Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer, computes the number
        of atoms for the given shell, and subtracts the cumulative sum of
        the previous shells to get the number of new atoms in the current
        shell.

        Returns:
            list: A list where each element represents the number of atoms
                  in a specific shell.
        """
        n = []
        current_sum = 0
        for i in range(self.nShell + 1):
            ni = self.nAtomsF(i)
            n_shell = ni - current_sum
            n.append(n_shell)
            current_sum += n_shell
        return n

    def nAtomsPerShellCumulativeAnalytic(self):
        """Computes the cumulative number of atoms up to each shell.

        This function returns the total number of atoms present in the
        nanoparticle for each shell layer, building up cumulatively.

        Returns:
            list: A list where each element represents the total number of
                  atoms present up to that shell.
        """
        n = []
        Sum = 0
        for i in range(self.nShell + 1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n

    def nAtomsAnalytic(self):
        """Computes the total number of atoms in the nanoparticle.

        Returns:
            int: Total number of atoms.
        """
        n = self.nAtomsF(self.nShell)
        return n

    def edgeLength(self):
        """Computes the edge length of the nanoparticle in Å.

        The edge length is determined based on the interatomic distance (Rnn)
        and the number of shells (`nShell`).

        Returns:
            float: Edge length in Å.
        """
        return self.Rnn * self.nShell

    def radiusCircumscribedSphere(self):
        """Computes the radius of the circumscribed sphere of the nanoparticle in Å.

        Returns:
            float: Radius.
        """
        return self.radiusCSF * self.edgeLength()

    def radiusInscribedSphere(self):
        """Computes the radius of the inscribed sphere of the nanoparticle in Å.

        Returns:
            float: Radius.
        """
        return self.radiusISF * self.edgeLength()

    def area(self):
        """Computes the surface area of the nanoparticle in square Ångströms.

        Returns:
            float: Surface area.
        """
        el = self.edgeLength()
        return 5 * el**2 * np.sqrt(3)

    def volume(self):
        """Computes the volume of the nanoparticle in cubic Ångströms.

        Returns:
            float: Volume.
        """
        el = self.edgeLength()
        return 5 * el**3 * (3 + np.sqrt(5)) / 12

    def MakeVertices(self, i):
        """Generates the coordinates of the vertices, edges, and faces
        for the ith shell of an icosahedral nanoparticle.

        Args:
           i (int): Index of the shell.

        Returns:
            tuple:
                - CoordVertices (np.ndarray): the 12 vertex coordinates of the ith shell of an icosahedron
                - edges (np.ndarray): indexes of the 30 edges
                - faces (np.ndarray): indexes of the 20 faces
                
        """
        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(
                f"icoreg.MakeVertices(i) is called"
                f" with i = {i} > nShell = {self.nShell}"
            )
        else:
            # Define vertex positions based on icosahedral geometry
            phi = self.phi
            scale = self.interShellDistance * i
            CoordVertices = [pyNMBu.vertex(-1, phi, 0, scale),
                             pyNMBu.vertex(1, phi, 0, scale),
                             pyNMBu.vertex(-1, -phi, 0, scale),
                             pyNMBu.vertex(1, -phi, 0, scale),
                             pyNMBu.vertex(0, -1, phi, scale),
                             pyNMBu.vertex(0, 1, phi, scale),
                             pyNMBu.vertex(0, -1, -phi, scale),
                             pyNMBu.vertex(0, 1, -phi, scale),
                             pyNMBu.vertex(phi, 0, -1, scale),
                             pyNMBu.vertex(phi, 0, 1, scale),
                             pyNMBu.vertex(-phi, 0, -1, scale),
                             pyNMBu.vertex(-phi, 0, 1, scale)]
            edges = [
                (1, 0), (3, 2), (4, 2), (4, 3), (5, 0),
                (5, 1), (5, 4), (6, 2), (6, 3), (7, 0),
                (7, 1), (7, 6), (8, 1), (8, 3), (8, 6),
                (8, 7), (9, 1), (9, 3), (9, 4), (9, 5),
                (9, 8), (10, 0), (10, 2), (10, 6), (10, 7),
                (11, 0), (11, 2), (11, 4), (11, 5), (11, 10),
            ]
            faces = [
                (7, 0, 1), (7, 1, 8), (7, 8, 6),
                (7, 6, 10), (7, 10, 0),
                (0, 11, 5), (0, 5, 1), (1, 5, 9),
                (1, 8, 9), (8, 9, 3), (8, 3, 6),
                (6, 3, 2), (6, 10, 2), (10, 2, 11),
                (10, 0, 11),
                (4, 2, 3), (4, 3, 9), (4, 9, 5),
                (4, 5, 11), (4, 11, 2),
            ]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def edges_to_planes(self, coords):
        """Converts a list of edges in planes equations [u, v, w, d].

        Args:
            coords (np.ndarray): Atoms coordinates.

        Returns:
            planes (list of np.ndarray): List of planes equations [u, v, w, d].
        """
        planes = []
        # for face in faces:
        #     # Get the coordinates of the three vertices of the face
        #     points = np.array([coords[face[0]], coords[face[1]], coords[face[2]]])

        points = np.array([coords[0], coords[1], coords[9], coords[11]])
        # Find plane equation using planeFittingLSF
        plane = pyNMBu.planeFittingLSF(points, printEq=False, printErrors=False)

        return plane

    def edges_to_planes2(self, coords):
        """Converts a list of edges in planes equations [u, v, w, d].

        Args:
            coords (np.ndarray): Atoms coordinates.

        Returns:
            planes (list of np.ndarray): List of planes equations [u, v, w, d].
        """
        planes = []
        # for face in faces:
        #     # Get the coordinates of the three vertices of the face
        #     points = np.array([coords[face[0]], coords[face[1]], coords[face[2]]])

        points = np.array([coords[2], coords[3], coords[7], coords[8]])
        # Find plane equation using planeFittingLSF
        plane = pyNMBu.planeFittingLSF(points, printEq=False, printErrors=False)

        return plane

    def coords(self, noOutput):
        """Generates atomic coordinates for an icosahedral nanoparticle.

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
        if not noOutput:
            vID.centertxt("Generation of coordinates",
                          bgc='#007a7a', size='14',
                          weight='bold')
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

        for i in range(1, self.nShell + 1):
            # Generate vertex atoms
            nAtoms0 = self.nAtoms
            cshell, E, F = self.MakeVertices(i)

            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0, self.nAtoms))

            # Generate edge atoms
            nAtoms0 = self.nAtoms
            Rvv = pyNMBu.RAB(cshell, E[0, 0], E[0, 1])  # distance between two vertex atoms
            nAtomsOnEdges = int((Rvv + 1e-6) / self.Rnn) - 1
            nIntervals = nAtomsOnEdges + 1
            # print("nAtomsOnEdges = ",nAtomsOnEdges)
            coordEdgeAt = []
            for n in range(nAtomsOnEdges):
                for e in E:  # Loop over all edges
                    a = e[0]
                    b = e[1]
                    coordEdgeAt.append(
                        cshell[a]
                        + pyNMBu.vector(cshell, a, b)
                        * (n + 1) / nIntervals)
            self.nAtomsPerShell[i] += nAtomsOnEdges * len(E)  # number of edges x nAtomsOnEdges
            self.nAtoms += nAtomsOnEdges * len(E)
            c.extend(coordEdgeAt)
            indexEdgeAtoms.extend(range(nAtoms0, self.nAtoms))

            # Generate facet atoms
            coordFaceAt = []
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            for f in F:
                nAtomsOnFaces, coordFaceAt = pyNMBu.MakeFaceCoord(
                    self.Rnn, f, cshell,
                    nAtomsOnFaces, coordFaceAt)
            self.nAtomsPerShell[i] += nAtomsOnFaces
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0, self.nAtoms))

            if self.double_ico and i == self.nShell:
                print('Nombre atomes avant troncature',
                      self.nAtoms)

                # 1. Truncate above the plane
                # Use the chosen edges to create the plane of truncation and reflection
                plane = self.edges_to_planes(cshell)
                print('Plane 1 used of truncation and reflection', plane)
                AtomsAboveAllPlanes = pyNMBu.truncateAbovePlanes(
                    np.array([plane]), c, allP=False,
                    delAbove=False, debug=False,
                    noOutput=noOutput, eps=0.1)
                c = pyNMBu.deleteElementsOfAList(
                    c, AtomsAboveAllPlanes)
                self.nAtoms = len(c)
                print('Nombre atomes après 1ere troncature', self.nAtoms)

                # 2. Reflect above the same plane to create the double icosahedron
                double_ico = pyNMBu.reflection(plane, c)
                c = np.vstack((c, double_ico))
                self.nAtoms = len(c)

        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput:
            print(self.nAtomsPerShell)
        aseObject = ase.Atoms(self.element * self.nAtoms, positions=c)
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()

    def prop(self):
        """Display unit cell and nanoparticle properties.
        """
        vID.centertxt("Properties", bgc='#007a7a', size='14', weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ", self.element)
        print("number of vertices = ", self.nVertices)
        print("number of edges = ", self.nEdges)
        print("number of faces = ", self.nFaces)
        print("phi = ", self.phi)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"edge length = {self.edgeLength() * 0.1:.2f} nm")
        print(f"radius after volume = "
              f"{pyNMBu.RadiusSphereAfterV(self.volume() * 1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = "
              f"{self.radiusCircumscribedSphere() * 0.1:.2f} nm")
        print(f"radius of the inscribed sphere = "
              f"{self.radiusInscribedSphere() * 0.1:.2f} nm")
        print(f"area = {self.area() * 1e-2:.1f} nm2")
        print(f"volume = {self.volume() * 1e-3:.1f} nm3")
        print("number of atoms per shell = ",
              self.nAtomsPerShellAnalytic())
        print("cumulative number of atoms per shell = ",
              self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ", self.nAtomsAnalytic())
        print("Dual polyhedron: dodecahedron")




###########################################################################################################
class regfccTd(PlatonicNP):
    """A class for generating XYZ and CIF files of regular fcc tetrahedral nanoparticles (NPs).

    Generates NPs of various sizes based on user-defined compounds (either by
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
    edgeLengthF = 1  # length of an edge
    heightOfPyramidF = edgeLengthF * np.sqrt(2 / 3)
    # Centroid to vertex distance = Radius of circumsphere
    radiusCSF = edgeLengthF * np.sqrt(3 / 8)
    # Radius of insphere that is tangent to faces
    radiusISF = edgeLengthF / np.sqrt(24)
    # Radius of midsphere that is tangent to edges
    radiusMSF = edgeLengthF / np.sqrt(8)
    fveAngle = np.rad2deg(np.arccos(1 / np.sqrt(3)))  # Face-vertex-edge angle
    fefAngle = np.rad2deg(np.arccos(1 / 3))  # Face-edge-face angle
    vcvAngle = np.rad2deg(np.arccos(-1 / 3))  # Vertex-Center-Vertex angle
  
    def __init__(self,
                 element: str = 'Au',
                 Rnn: float = 2.7,
                 nLayer: int = 1,
                 shape: str = 'regfccTd',
                 n_tetrahedrons: int = 1,
                 postAnalyzis=True,
                 aseView: bool = False,
                 thresholdCoreSurface=1.,
                 skipSymmetryAnalyzis=False,
                 jmolCrystalShape: bool = True,
                 noOutput=False,
                 calcPropOnly=False,
                 ):
        """Initialize the class with all necessary parameters.

        Args:
            element (str): Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nLayer (int): Number of layers, also equals to the number
                of atoms per edge (e.g., ``nOrder=2`` means 2 atoms
                per edge).
            shape (str): Shape 'regfccTd'.
            n_tetrahedrons (int): The number of tetrahedrons in the
                optional helix.
            postAnalyzis (bool): If True, prints additional NP
                information (e.g., cell parameters, moments of inertia,
                inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP
                using ASE.
            thresholdCoreSurface (float): Precision threshold for
                core/surface differentiation (distance threshold for
                retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic
                structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script
                for visualization.
            noOutput (bool): If False, prints details about the NP
                structure.
            calcPropOnly (bool): If False, generates the atomic
                structure of the NP.

        Attributes:
            nAtoms (int): Number of atoms in the NP.
            nAtomsPerLayer (list): Number of atoms in each atomic layer.
            nAtomsPerEdge (int): Number of atoms per edge.
            jmolCrystalShape (bool): Flag for JMol visualization.
            cog (np.array): Center of gravity of the NP.
            imageFile (str): Path to a reference image.
            trPlanes (array): Truncation plane equations.

        """
        self.element = element
        self.shape = shape
        self.Rnn = Rnn
        self.nLayer = nLayer
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.nAtomsPerEdge = self.nLayer
        self.jmolCrystalShape = jmolCrystalShape
        self.cog = np.array([0., 0., 0.])
        self.n_tetrahedrons = n_tetrahedrons
        self.nAtoms_helix = 0  # Initialize to 0, will be computed in generate_tetrahelix()
        self.imageFile = pyNMBu.imageNameWithPathway("fccTd-C.png")
        self.NP = None
        self.trPlanes = None
        if not noOutput:
            vID.centerTitle(f"fcc tetrahedron: {nLayer} atoms/edge = number of layers")
          
        if not noOutput:
            self.prop()

        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput=noOutput)
                if aseView:
                    view(self.NPcs)
          
    def __str__(self):
        return f"Regular tetrahedron with {self.nLayer} layer(s) and Rnn = {self.Rnn}"
    
    def nAtomsF(self, i):
        """Returns the number of atoms of a tetrahedron of size i."""
        return round(i**3 / 6 + i**2 + 11 * i / 6 + 1)
    
    def nAtomsPerLayerAnalytic(self):
        """Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each layer, computes the number of
        atoms for the given layer, and subtracts the cumulative sum of
        the previous shells to get the number of new atoms in the
        current layer.

        Returns:
            list: A list where each element represents the number of atoms 
                  in a specific layer.
        """
        n = []
        current_sum = 0
        for i in range(self.nLayer):
            ni = int(self.nAtomsF(i))
            n_shell = ni - current_sum
            n.append(n_shell)
            current_sum += n_shell
            # print(i,ni,Sum,n)
        return n
    
    def nAtomsAnalytic(self):
        """Computes the total number of atoms in the nanoparticle."""
        n = self.nAtomsF(self.nLayer - 1)
        return n
    
    def edgeLength(self):
        """Computes the edge length of the nanoparticle in Å.

        The edge length is determined based on the interatomic distance (Rnn) 
        and the number of atomic layers (`nLayer`).
        """
        return self.Rnn * (self.nLayer - 1)

    def heightOfPyramid(self):
        """Computes the length of the height of the pyramid in Å."""
        return self.heightOfPyramidF * self.edgeLength()
    
    def interLayerDistance(self):
        """Computes the distance between the layers in Å."""
        return self.heightOfPyramid() / (self.nLayer - 1)
    
    def radiusCircumscribedSphere(self):
        """Computes the radius of the circumscribed sphere of the nanoparticle in Å."""
        return self.radiusCSF * self.edgeLength()

    def radiusInscribedSphere(self):
        """Computes the radius of the inscribed sphere of the nanoparticle in Å."""
        return self.radiusISF * self.edgeLength()

    def radiusMidSphere(self):
        """Computes the radius of the midsphere of the nanoparticle in Å.
        
        The midsphere is a sphere that touches the edges of the nanoparticle.
        """
        return self.radiusMSF * self.edgeLength()

    def area(self):
        """Computes the surface area of the nanoparticle in square Ångströms."""
        el = self.edgeLength()
        return el**2 * np.sqrt(3)
    
    def volume(self):
        """Computes the volume of the nanoparticle in cubic Ångströms."""
        el = self.edgeLength()
        return el**3 / (6 * np.sqrt(2)) 

    def MakeVertices(self, nL):
        """
        Generates the coordinates of the vertices, edges, and faces
        for the ith shell of a tetrahedral nanoparticle.

        Args:
            nL (int): number of layers = number of atoms per edge.

        Returns:
            - CoordVertices (np.ndarray): the 4 vertex coordinates of a tetrahedron.
            - edges (np.ndarray): indexes of the 6 edges.
            - faces (np.ndarray): indexes of the 4 faces.
            
        """
        if (nL > self.nLayer):
            sys.exit(
                f"regTd.MakeVertices(nL) is called"
                f" with nL = {nL} > nLayer = {self.nLayer}"
            )
        else:
            scale = self.radiusCircumscribedSphere()
            c = 1 / (2 * np.sqrt(2))  # edge length 1
            # Define vertex positions based on tetrahedral geometry
            CoordVertices = [pyNMBu.vertex(c, c, c, scale),
                             pyNMBu.vertex(c, -c, -c, scale),
                             pyNMBu.vertex(-c, c, -c, scale),
                             pyNMBu.vertex(-c, -c, c, scale)]
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    ####################################### Helice of tetrahedrons
    def faces_to_planes(self, faces, coords):
        """Converts a list of faces in planes equations [u, v, w, d].

        Args:
            faces (list of tuples): List of faces [(i, j, k)] with
                the vertices indexes.
            coords (np.ndarray): Atoms coordinates.

        Returns:
            planes (list of np.ndarray): List of planes equations
                [u, v, w, d].
        """
        planes = []

        for face in faces:
            # Get the coordinates of the three vertices of the face
            points = np.array([coords[face[0]],
                               coords[face[1]],
                               coords[face[2]]])

            # Find plane equation using planeFittingLSF
            plane = pyNMBu.planeFittingLSF(
                points, printEq=False, printErrors=False)
            planes.append(plane)

        return np.array(planes)

    def generate_tetrahelix(self, c, n_tetrahedrons, nAtoms,
                             debug=False):
        """Generates a Boerdijk-Coxeter helix made of tetrahedrons.

        Possibility to give the number of tetrahedrons.
        The function uses reflection to create one tetrahedron
        after the other one.
        TO DO: don't count twice the atoms of the reflected face!

        Args:
            c: Atoms coordinates.
            n_tetrahedrons (int): Number of tetrahedrons in the helix.
            debug (bool): Prints debug information.

        Returns:
            c (list): List of the helix coordinates.
        """
        faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)] 
        new_tetra = c
        tetras_list = [c]

        for i in range(1, n_tetrahedrons):
            # Choisir la bonne face pour la réflexion
            last_planes = self.faces_to_planes(faces, new_tetra)
            
            face_index = i % 4  # Alterne entre 0, 1, 2, 3
            last_face = last_planes[face_index]  # Récupérer la face correspondante
            
            # Appliquer une réflexion uniquement
            new_tetra = pyNMBu.reflection_tetra(last_face, new_tetra)
            if debug:
                print('Coordinates of the new tetrahedron', new_tetra)
            
            # TO DO: don't count twice the atoms of the reflected face ! 

            # 4. Ajouter le tétraèdre à la liste
            tetras_list.append(new_tetra)
            self.nAtoms += len(new_tetra)

        # Vectorization: stack all tetrahedrons at once
        c = np.vstack(tetras_list)

        self.nAtoms_helix = (self.nAtoms
            - (self.nLayer * (1 + self.nLayer)
               // 2 * (self.n_tetrahedrons - 1)))
        
        if debug:
            print("Final shape of the coordinates", np.shape(c))
            print("Number of atoms in the helix:", self.nAtoms_helix)
        return c, self.nAtoms_helix
    
    def coords(self, noOutput):
        """Generates atomic coordinates for a tetrahedral nanoparticle.

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
        if not noOutput:
            vID.centertxt("Generation of coordinates",
                          bgc='#007a7a', size='14',
                          weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        c = []  # List of atomic coordinates
        # print(self.nAtomsPerLayer)
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        indexCoreAtoms = []

        # Generate vertex atoms
        nAtoms0 = 0
        self.nAtoms += self.nVertices
        cVertices, E, F = self.MakeVertices(self.nLayer - 1)
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0, self.nAtoms))

        # Generate edge atoms
        nAtoms0 = self.nAtoms
        # distance between two vertex atoms
        Rvv = pyNMBu.RAB(cVertices, E[0, 0], E[0, 1])
        nAtomsOnEdges = int((Rvv + 1e-6) / self.Rnn) - 1
        nIntervals = nAtomsOnEdges + 1
        # print("nAtomsOnEdges = ",nAtomsOnEdges)
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E:  # Loop over all edges
                a = e[0]
                b = e[1]
                coordEdgeAt.append(
                    cVertices[a]
                    + pyNMBu.vector(cVertices, a, b)
                    * (n + 1) / nIntervals)
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        indexEdgeAtoms.extend(range(nAtoms0, self.nAtoms))
        self.nAtomsPerEdge = nAtomsOnEdges + 2  # 2 vertices
        # print(indexEdgeAtoms)
        
        # Generate facet atoms
        coordFaceAt = []
        nAtomsOnFaces = 0
        nAtoms0 = self.nAtoms
        for f in F:
            nAtomsOnFaces, coordFaceAt = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsOnFaces, coordFaceAt)

        self.nAtoms += nAtomsOnFaces
        c.extend(coordFaceAt)
        indexFaceAtoms.extend(range(nAtoms0, self.nAtoms))

        # Generate core atoms
        # Layer by layer strategy, using atoms on edges
        # [0-1],[0-2],[0-3] when identified, just use MakeFaceCoord
        # and define, for each layer, the three atoms on the edge
        # as a facet. Just start from 4th layer.
        coordCoreAt = []
        nAtomsInCore = 0
        nAtoms0 = self.nAtoms
        for ilayer in range(4, self.nLayer + 1):
            FirstAtom = 4 + (ilayer - 2) * 6
            f = np.array([FirstAtom, FirstAtom + 1, FirstAtom + 2])
            # print("layer ",ilayer,f)
            nAtomsInCore, coordCoreAt = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsInCore, coordCoreAt)
        self.nAtoms += nAtomsInCore
        c.extend(coordCoreAt)
        indexCoreAtoms.extend(range(nAtoms0, self.nAtoms))

        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput:
            print(self.nAtomsPerLayer)
        # aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)
        
        ########################### helix ########################################################
        if self.n_tetrahedrons > 1: 
            c, self.nAtoms_helix = self.generate_tetrahelix(
                c, n_tetrahedrons=self.n_tetrahedrons,
                nAtoms=self.nAtoms, debug=False)
            # self.cog = pyNMBu.centerOfGravity(c)
        else:
            # Single tetrahedron: nAtoms_helix equals nAtoms
            self.nAtoms_helix = self.nAtoms
             
        self.nAtoms = len(c)
        self.cog = pyNMBu.centerOfGravity(c)
       ############################################################################################
        # print("Shape de c:", np.shape(c))
        # print("self.nAtoms:", self.nAtoms)

        aseObject = ase.Atoms(self.element * self.nAtoms, positions=c)
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()
    
    def prop(self):
        """Display unit cell and nanoparticle properties."""
        vID.centertxt("Properties", bgc='#007a7a', size='14', weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ", self.element)
        print("number of vertices = ", self.nVertices)
        print("number of edges = ", self.nEdges)
        print("number of faces = ", self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength() * 0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"inter-layer distance = {self.interLayerDistance():.2f} Å")
        print(f"height of pyramid = {self.heightOfPyramid() * 0.1:.2f} nm")
        print(f"radius after volume = "
              f"{pyNMBu.RadiusSphereAfterV(self.volume() * 1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = "
              f"{self.radiusCircumscribedSphere() * 0.1:.2f} nm")
        print(f"radius of the inscribed sphere = "
              f"{self.radiusInscribedSphere() * 0.1:.2f} nm")
        print(f"radius of the midsphere tangent to edges = "
              f"{self.radiusMidSphere() * 0.1:.2f} nm")
        print(f"area = {self.area() * 1e-2:.1f} nm2")
        print(f"volume = {self.volume() * 1e-3:.1f} nm3")
        print(f"face-vertex-edge angle = {self.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle = "
              f"{self.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond) angle = "
              f"{self.vcvAngle:.1f}°")
        print("number of atoms per layer = ",
              self.nAtomsPerLayerAnalytic())
        # Note: nAtoms_helix will be computed after coords()

        print("Dual polyhedron: tetrahedron")
        print("Indexes of vertex atoms = [0,1,2,3] by construction")
        print(f"coordinates of the center of gravity = {self.cog}")



###########################################################################################################
class regDD(PlatonicNP):
    """A class for generating XYZ and CIF files of regular dodecahedral nanoparticles (NPs).

    Generates NPs of various sizes based on user-defined compounds (either by
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
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    edgeLengthF = 1
    radiusCSF = np.sqrt(3) * (1 + np.sqrt(5)) / 4  # Radius of circumsphere
    interShellF = 1 / radiusCSF
    # Radius of insphere that is tangent to faces
    radiusISF = np.sqrt((5 / 2) + (11 / 10) * np.sqrt(5)) / 2

    def __init__(self,
                 element: str = 'Au',
                 Rnn: float = 2.7,
                 nShell: int = 1,
                 shape: str = 'regDD',
                 postAnalyzis=True,
                 aseView: bool = False,
                 thresholdCoreSurface=1.,
                 skipSymmetryAnalyzis=False,
                 jmolCrystalShape: bool = True,
                 noOutput=False,
                 calcPropOnly=False,
                 ):
        """Initialize the class with all necessary parameters.

        Args:
            element (str): Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nShell (int): Number of shells (e.g., ``nShell=1`` means
                2 atoms per edge).
            shape (str): Shape 'regDD'.
            postAnalyzis (bool): If True, prints additional NP
                information (e.g., cell parameters, moments of inertia,
                inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP
                using ASE.
            thresholdCoreSurface (float): Precision threshold for
                core/surface differentiation (distance threshold for
                retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic
                structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script
                for visualization.
            noOutput (bool): If False, prints details about the NP
                structure.
            calcPropOnly (bool): If False, generates the atomic
                structure of the NP.

        Attributes:
            nAtoms (int): Number of atoms in the NP.
            nAtomsPerShell (list): Number of atoms in each shell.
            interShellDistance (float): Distance between shells.
            jmolCrystalShape (bool): Flag for JMol visualization.
            imageFile (str): Path to a reference image.
            trPlanes (array): Truncation plane equations.

        """
        self.element = element
        self.shape = shape
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.jmolCrystalShape = jmolCrystalShape
        self.imageFile = pyNMBu.imageNameWithPathway("rDD-C.png")
        self.trPlanes = None
        if not noOutput:
            vID.centerTitle(f"{nShell} shells regular dodecahedron")
          
        if not noOutput:
            self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput=noOutput)
                if aseView:
                    view(self.NPcs)
          
    def __str__(self):
        return f"Regular dodecahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}"
    
    def nAtomsF(self, i):
        """Returns the number of atoms of a dodecahedron of size i."""
        return 10 * i**3 + 15 * i**2 + 7 * i + 1
    
    def nAtomsPerShellAnalytic(self):
        """Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer, computes the number
        of atoms for the given shell, and subtracts the cumulative sum of
        the previous shells to get the number of new atoms in the
        current shell.

        Returns:
            list: A list where each element represents the number of
                atoms in a specific shell.
        """
        n = []
        current_sum = 0
        for i in range(self.nShell + 1):
            ni = self.nAtomsF(i)
            n_shell = ni - current_sum
            n.append(n_shell)
            current_sum += n_shell
        return n

    def nAtomsAnalytic(self):
        """Computes the total number of atoms in the nanoparticle."""
        n = self.nAtomsF(self.nShell)
        return n

    def edgeLength(self):
        """Computes the edge length of the nanoparticle in Å.

        The edge length is determined based on the interatomic distance
        (Rnn) and the number of shells (``nShell``).
        """
        return self.Rnn * self.nShell

    def radiusCircumscribedSphere(self):
        """Computes the radius of the circumscribed sphere of the nanoparticle in Å."""
        return self.radiusCSF * self.edgeLength()

    def radiusInscribedSphere(self):
        """Computes the radius of the inscribed sphere of the nanoparticle in Å."""
        return self.radiusISF * self.edgeLength()

    def area(self):
        """Computes the surface area of the nanoparticle in square Ångströms."""
        el = self.edgeLength()
        return 3 * el**2 * np.sqrt(25 + 10 * np.sqrt(5))
    
    def volume(self):
        """Computes the volume of the nanoparticle in cubic Ångströms."""
        el = self.edgeLength()
        return (15 + 7 * np.sqrt(5)) * el**2 / 4 

    def MakeVertices(self, i):
        """Generates the coordinates of the vertices, edges, and faces
        for the ith shell of a dodecahedral nanoparticle.

        Args:
            i (int): Index of the shell.

        Returns:
            - CoordVertices (np.ndarray): the 20 vertex coordinates of the ith shell of a dodecahedron.
            - edges (np.ndarray): indexes of the 30 edges.
            - faces (np.ndarray): indexes of the 12 faces.
            
        """
        # If `i == 0`, the function returns a single central vertex
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(
                f"icoreg.MakeVertices(i) is called"
                f" with i = {i} > nShell= {self.nShell}"
            )
        else:
            # Define vertex positions based on dodecahedral geometry
            phi = self.phi
            scale = self.interShellDistance * i
            CoordVertices = [pyNMBu.vertex(1, 1, 1, scale),
                             pyNMBu.vertex(-1, 1, 1, scale),
                             pyNMBu.vertex(1, -1, 1, scale),
                             pyNMBu.vertex(1, 1, -1, scale),
                             pyNMBu.vertex(-1, -1, 1, scale),
                             pyNMBu.vertex(-1, 1, -1, scale),
                             pyNMBu.vertex(1, -1, -1, scale),
                             pyNMBu.vertex(-1, -1, -1, scale),
                             pyNMBu.vertex(0, phi, 1 / phi, scale),
                             pyNMBu.vertex(0, -phi, 1 / phi, scale),
                             pyNMBu.vertex(0, phi, -1 / phi, scale),
                             pyNMBu.vertex(0, -phi, -1 / phi, scale),
                             pyNMBu.vertex(1 / phi, 0, phi, scale),
                             pyNMBu.vertex(-1 / phi, 0, phi, scale),
                             pyNMBu.vertex(1 / phi, 0, -phi, scale),
                             pyNMBu.vertex(-1 / phi, 0, -phi, scale),
                             pyNMBu.vertex(phi, 1 / phi, 0, scale),
                             pyNMBu.vertex(phi, -1 / phi, 0, scale),
                             pyNMBu.vertex(-phi, 1 / phi, 0, scale),
                             pyNMBu.vertex(-phi, -1 / phi, 0, scale)]
            edges = [
                (8, 0), (8, 1), (9, 2), (9, 4), (10, 3),
                (10, 5), (10, 8), (11, 6), (11, 7), (11, 9),
                (12, 0), (12, 2), (13, 1), (13, 4), (13, 12),
                (14, 3), (14, 6), (15, 5), (15, 7), (15, 14),
                (16, 0), (16, 3), (17, 2), (17, 6), (17, 16),
                (18, 1), (18, 5), (19, 4), (19, 7), (19, 18),
            ]
            faces = [
                (0, 8, 10, 3, 16), (0, 12, 13, 1, 8),
                (8, 1, 18, 5, 10), (10, 5, 15, 14, 3),
                (3, 14, 6, 17, 16), (16, 17, 2, 12, 0),
                (4, 9, 11, 7, 19), (4, 13, 12, 2, 9),
                (9, 2, 17, 6, 11), (11, 6, 14, 15, 7),
                (7, 15, 5, 18, 19), (19, 18, 1, 13, 4),
            ]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self, noOutput):
        """Generates atomic coordinates for a dodecahedral nanoparticle.

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
        if not noOutput:
            vID.centertxt("Generation of coordinates",
                          bgc='#007a7a', size='14',
                          weight='bold')
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
        for i in range(1, self.nShell + 1):

            # Generate vertex atoms
            nAtoms0 = self.nAtoms
            cshell, E, F = self.MakeVertices(i)
            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0, self.nAtoms))

            # Generate edge atoms
            nAtoms0 = self.nAtoms
            # distance between two vertex atoms
            Rvv = pyNMBu.RAB(cshell, E[0, 0], E[0, 1])
            nAtomsOnEdges = int((Rvv + 1e-6) / self.Rnn) - 1
            nIntervals = nAtomsOnEdges + 1
            # print("nAtomsOnEdges = ",nAtomsOnEdges)
            coordEdgeAt = []
            for n in range(nAtomsOnEdges):
                for e in E:
                    a = e[0]
                    b = e[1]
                    coordEdgeAt.append(
                        cshell[a]
                        + pyNMBu.vector(cshell, a, b)
                        * (n + 1) / nIntervals)
            self.nAtomsPerShell[i] += nAtomsOnEdges * len(E)
            self.nAtoms += nAtomsOnEdges * len(E)
            c.extend(coordEdgeAt)
            indexEdgeAtoms.extend(range(nAtoms0, self.nAtoms))

            # Generate facet atoms
            # Center of each pentagonal facet
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            coordFaceAt = []
            for f in F:
                nAtomsOnFaces += 1
                coordCenterFace = pyNMBu.centerOfGravity(cshell, f)
                # print("coordCenterFace",coordCenterFace)
                self.nAtomsPerShell[i] += 1
                coordFaceAt.append(coordCenterFace)
                # atoms from the center of each pentagonal facet to each of its apex 
                nAtomsOnInternalRadius = i - 1
                nIntervals = nAtomsOnInternalRadius + 1
                # print(f)
                for indexApex, apex in enumerate(f):
                    if (indexApex == len(f) - 1):
                        indexApexPlus1 = 0
                    else:
                        indexApexPlus1 = indexApex + 1
                    apexPlus1 = f[indexApexPlus1]
                    for n in range(nAtomsOnInternalRadius):
                        nAtomsOnFaces += 1
                        coordAtomOnApex = (
                            coordCenterFace
                            + pyNMBu.vectorBetween2Points(
                                coordCenterFace, cshell[apex])
                            * (n + 1) / nIntervals)
                        coordAtomOnApexPlus1 = (
                            coordCenterFace
                            + pyNMBu.vectorBetween2Points(
                                coordCenterFace, cshell[apexPlus1])
                            * (n + 1) / nIntervals)
                        coordFaceAt.append(coordAtomOnApex)
                        RbetweenRadialAtoms = (
                            pyNMBu.Rbetween2Points(
                                coordAtomOnApex,
                                coordAtomOnApexPlus1))
                        nAtomsBetweenRadialAtoms = int(
                            (RbetweenRadialAtoms + 1e-6)
                            / self.Rnn) - 1
                        nIntervalsRadial = (
                            nAtomsBetweenRadialAtoms + 1)
                        for k in range(nAtomsBetweenRadialAtoms):
                            nAtomsOnFaces += 1
                            coordFaceAt.append(
                                coordAtomOnApex
                                + pyNMBu.vectorBetween2Points(
                                    coordAtomOnApex,
                                    coordAtomOnApexPlus1)
                                * (k + 1) / nIntervalsRadial)
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0, self.nAtoms))

        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput:
            print(self.nAtomsPerShell)
        # Store results in an ASE Atoms object
        aseObject = ase.Atoms(
            self.element * self.nAtoms, positions=c)

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()

    def prop(self):
        """Display unit cell and nanoparticle properties."""
        vID.centertxt("Properties", bgc='#007a7a', size='14', weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ", self.element)
        print("number of vertices = ", self.nVertices)
        print("number of edges = ", self.nEdges)
        print("number of faces = ", self.nFaces)
        print("phi = ", self.phi)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"edge length = {self.edgeLength() * 0.1:.2f} nm")
        print(f"radius after volume = "
              f"{pyNMBu.RadiusSphereAfterV(self.volume() * 1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = "
              f"{self.radiusCircumscribedSphere() * 0.1:.2f} nm")
        print(f"radius of the inscribed sphere = "
              f"{self.radiusInscribedSphere() * 0.1:.2f} nm")
        print(f"area = {self.area() * 1e-2:.1f} nm2")
        print(f"volume = {self.volume() * 1e-3:.1f} nm3")
        print("number of atoms per shell = ",
              self.nAtomsPerShellAnalytic())
        print("total number of atoms = ", self.nAtomsAnalytic())
        print("Dual polyhedron: icosahedron")



###########################################################################################################
class cube(PlatonicNP):
    """A class for generating XYZ and CIF files of cubic nanoparticles (NPs).

    Generates NPs of various sizes based on user-defined compounds (either by
    name, e.g., "Fe", "Au", etc).

    Key Features:
        - Allows to choose the NP size.
        - Can analyze the structure in detail, including symmetry and properties.
        - Offers options for core/surface differentiation based on a threshold.
        - Generates outputs in XYZ and CIF formats for visualization and simulations.
        - Provides compatibility with jMol for 3D visualization.

    Additional Notes:
        - The `nOrder` parameter determines the level of imbrication.
        - The symmetry analysis can be skipped to speed up computations.
        - Customizable precision thresholds for structural analysis.
    """
    nFaces = 6
    nEdges = 12
    nVertices = 8
    edgeLengthFfcc = np.sqrt(2)
    edgeLengthFbcc = 2 / np.sqrt(3)
    radiusCSF = np.sqrt(3) / 2
    radiusISF = 1 / 2

    def __init__(self,
                 crystalStructure='fcc',
                 element='Au',
                 Rnn: float = 2.7,
                 nOrder: int = 1,
                 size: int = 0,
                 shape: str = 'cube',
                 postAnalyzis=True,
                 aseView: bool = False,
                 thresholdCoreSurface=1.,
                 skipSymmetryAnalyzis=False,
                 jmolCrystalShape: bool = True,
                 noOutput=False,
                 calcPropOnly=False,
                 ):
        """Initialize the class with all necessary parameters.

        Args:
            crystalStructure (str): The crystal structure of the NP (e.g., 'fcc', 'bcc').
            element (str): Chemical element of the NP (e.g., "Au", "Fe").
            Rnn (float): Nearest neighbor interatomic distance in Å.
            nOrder (int): Determines the level of imbrication = the
                number of atomic layers along an edge (e.g.,
                ``nOrder=1`` means 2 atoms per edge).
            size (float): Size of the cube in nm.
            shape (str): Shape of the nanoparticle. Defaults to 'cube'.
            postAnalyzis (bool): If True, prints additional NP
                information (e.g., cell parameters, moments of inertia,
                inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization
                of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold
                for core/surface differentiation (distance
                threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs
                an atomic structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a
                JMOL script for visualization.
            noOutput (bool): If False, prints details about
                the NP structure.
            calcPropOnly (bool): If False, generates the
                atomic structure of the NP.

        Attributes:
            nAtoms (int): Number of atoms in the NP.
            nAtomsPerShell (list): Number of atoms in each shell.
            nAtomsPerEdge (int): Number of atoms per edge.
            interLayerDistance (float): Distance between atomic layers.
            jmolCrystalShape (bool): Flag for JMol visualization.
            cog (np.array): Center of gravity of the NP.
            imageFile (str): Path to a reference image.
            trPlanes (array): Truncation plane equations.
        """
        self.crystalStructure = crystalStructure
        self.element = element
        self.shape = shape
        self.Rnn = Rnn
        self.nOrder = nOrder
        self.size = size * 10  # in angs
        self.nAtomsPerEdge = nOrder + 1
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.jmolCrystalShape = jmolCrystalShape
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pyNMBu.imageNameWithPathway("cube-C.png")
        self.trPlanes = None

        if not noOutput:
            vID.centerTitle(f"{nOrder}x{nOrder}x{nOrder} {self.crystalStructure} cube")

        if not noOutput:
            self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput=noOutput)
                if aseView:
                    view(self.NPcs)

    def __str__(self):
        """Returns a string representation of the object."""
        return (
            f"{self.nOrder}x{self.nOrder}x{self.nOrder}"
            f" {self.crystalStructure} cube with Rnn = {self.Rnn}"
        )

    def nAtomsfccF(self, i):
        """Returns the number of atoms of an fcc cube of size i x i x i.

        Args:
            i (int): The size parameter corresponding to nOrder.

        Returns:
            int: The calculated number of atoms.
        """
        return 4 * i**3 + 6 * i**2 + 3 * i + 1

    def nAtomsbccF(self, i):
        """Returns the number of atoms of a bcc cube of size i x i x i.

        Args:
            i (int): The size parameter corresponding to nOrder.

        Returns:
            int: The calculated number of atoms.
        """
        return 2 * i**3 + 3 * i**2 + 3 * i + 1

    def nAtomsPerShellAnalytic(self):
        """Computes the number of atoms per shell in an ordered nanoparticle.

        The function iterates over each shell layer (from 1 to `nOrder`),
        computes the number of atoms for the given shell, and subtracts
        the cumulative sum of the previous shells to get the number of new
        atoms in the current shell.

        Returns:
            list: A list where each element represents the number of atoms
                in a specific shell.
        """
        n = []
        current_sum = 0
        for i in range(1, self.nOrder + 1):
            ni = self.nAtomsF(i)  # natoms in the whole octahedron of order i
            n_shell = ni - current_sum
            n.append(n_shell)
            current_sum += n_shell  # Update running sum
        return n

    def nAtomsPerShellCumulativeAnalytic(self):
        """Computes the cumulative number of atoms up to each shell.

        This function returns the total number of atoms present in the
        nanoparticle for each shell layer, building up cumulatively.

        Returns:
            list: A list where each element represents the total number of
                atoms present up to that shell.
        """
        n = []
        Sum = 0
        for i in range(self.nOrder + 1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n

    def nAtomsfccAnalytic(self):
        """Computes the total number of atoms in the fcc nanoparticle.

        Returns:
            int: Total number of atoms.
        """
        n = self.nAtomsfccF(self.nOrder)
        return n

    def nAtomsbccAnalytic(self):
        """Computes the total number of atoms in the bcc nanoparticle.

        Returns:
            int: Total number of atoms.
        """
        n = self.nAtomsbccF(self.nOrder)
        return n

    def edgeLength(self):
        """Computes the edge length of the nanoparticle in Å.

        The edge length is determined based on the interatomic distance
        (Rnn), the number of atomic layers (``nOrder``) and the
        crystalStructure (fcc or bcc).

        Returns:
            float: The edge length in Å.
        """
        if self.crystalStructure == 'fcc':
            return self.Rnn * self.edgeLengthFfcc * self.nOrder
        elif self.crystalStructure == 'bcc':
            return self.Rnn * self.edgeLengthFbcc * self.nOrder

    def latticeConstant(self):
        """Computes the lattice constant of the nanoparticle in Å.

        Based on the interatomic distance (Rnn) and the
        crystalStructure (fcc or bcc).

        Returns:
            float: The lattice constant in Å.
        """
        if self.crystalStructure == 'fcc':
            return self.Rnn * self.edgeLengthFfcc
        elif self.crystalStructure == 'bcc':
            return self.Rnn * self.edgeLengthFbcc

    def radiusCircumscribedSphere(self):
        """Computes the radius of the circumscribed sphere of the nanoparticle in Å.

        Returns:
            float: Radius.
        """
        return self.radiusCSF * self.edgeLength()

    def radiusInscribedSphere(self):
        """Computes the radius of the inscribed sphere of the nanoparticle in Å.

        Returns:
            float: Radius.
        """
        return self.radiusISF * self.edgeLength()

    def area(self):
        """Computes the surface area of the nanoparticle in square Ångströms.

        Returns:
            float: Surface area.
        """
        el = self.edgeLength()
        return 6 * el**2

    def volume(self):
        """Computes the volume of the nanoparticle in cubic Ångströms.

        Returns:
            float: Volume.
        """
        el = self.edgeLength()
        return el**3

    def coords(self, noOutput):
        """Generates atomic coordinates for a cubic nanoparticle.

        Args:
            noOutput (bool): If False, displays progress and timing
                information.

        Steps:
            - Generates vertex atoms.
            - Calculates and places edge atoms along the edges.
            - Generates facet atoms to fill in faces.
            - Adds core atoms layer by layer.
            - Stores final atomic positions in an ASE Atoms object.

        Returns:
            None (updates class attributes).
        """
        # crystalline structure
        if not noOutput:
            vID.centertxt("Generation of coordinates",
                          bgc='#007a7a', size='14',
                          weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        if self.crystalStructure == 'fcc':
            cube = bulk(self.element, 'fcc',
                        a=self.latticeConstant(), cubic=True)
        elif self.crystalStructure == 'bcc':
            cube = bulk(self.element, 'bcc',
                        a=self.latticeConstant(), cubic=True)

        # Creating supercell depending the entries of user:
        # size of the cube in Angs or nOrder (number of cells)
        if self.size == 0:  # if not size given
            if not noOutput:
                print(f"Now making a {self.nOrder}x{self.nOrder}"
                      f"x{self.nOrder} fcc supercell...")
            M = [[self.nOrder, 0, 0],
                 [0, self.nOrder, 0],
                 [0, 0, self.nOrder]]
            sc = make_supercell(cube, M)
        else:
            self.n_cells = int(self.size / self.latticeConstant())
            M = [[self.n_cells, 0, 0],
                 [0, self.n_cells, 0],
                 [0, 0, self.n_cells]]
            sc = make_supercell(cube, M)

        # Adding the last layers
        if not noOutput:
            print("... and adding the upper layers")
        sc = cut(sc, extend=1.05)
        natoms = len(sc.positions)
        self.nAtoms = natoms
        self.cog = pyNMBu.centerOfGravity(sc.get_positions())
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        coordsNP = sc.get_positions()
        oldcog = sc.get_center_of_mass()
        coordsNP = coordsNP - oldcog
        sc.set_positions(coordsNP)
        self.NP = sc
        self.cog = self.NP.get_center_of_mass()

    def prop(self):
        """Display unit cell and nanoparticle properties.
        """
        vID.centertxt("Properties", bgc='#007a7a', size='14', weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ", self.element)
        print("number of vertices = ", self.nVertices)
        print("number of edges = ", self.nEdges)
        print("number of faces = ", self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"lattice constant = {self.latticeConstant():.2f} Å")
        print(f"edge length = {self.edgeLength() * 0.1:.2f} nm")
        print(f"radius after volume = "
              f"{pyNMBu.RadiusSphereAfterV(self.volume() * 1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = "
              f"{self.radiusCircumscribedSphere() * 0.1:.2f} nm")
        print(f"radius of the inscribed sphere = "
              f"{self.radiusInscribedSphere() * 0.1:.2f} nm")
        print(f"area = {self.area() * 1e-2:.1f} nm2")
        print(f"volume = {self.volume() * 1e-3:.1f} nm3")
        # print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        # print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        if self.crystalStructure == 'fcc':
            print("total number of atoms = ", self.nAtomsfccAnalytic())
        elif self.crystalStructure == 'bcc':
            print("total number of atoms = ", self.nAtomsbccAnalytic())
        print("Dual polyhedron: octahedron")


class hollow_shapes(PlatonicNP):
    """A class for generating XYZ and CIF files
    of hollow cubic nanoparticles (NPs).

    Creates NPs with customizable sizes and
    compositions. Users can define the composition
    by specifying element names (e.g., "Fe", "Au")
    and provide a "cube" class instance from this
    module to construct the nanoparticle structure.

    Key Features:
        - Allows to choose the cube size and the
            size of its hollow.
        - Can analyze the structure in detail,
            including symmetry and properties.
        - Offers options for core/surface
            differentiation based on a threshold.
        - Generates outputs in XYZ and CIF formats
            for visualization and simulations.
        - Provides compatibility with jMol for 3D
            visualization.

    Additional Notes:
        - The symmetry analysis can be skipped to speed up computations.
        - Customizable precision thresholds for structural analysis.
    """

    def __init__(self,
                 full_cube,
                 nOrder_hollow: int = 0,  # Angs?
                 postAnalyzis=True,
                 aseView: bool = False,
                 thresholdCoreSurface=1.,
                 skipSymmetryAnalyzis=False,
                 jmolCrystalShape: bool = True,
                 noOutput=False,
                 calcPropOnly=False
                 ):
        """Initialize the class with all necessary parameters.

        Args:
            full_cube (cube): Instance of the
                class "cube" of the module "pNP".
            nOrder_hollow (int): Size of the hollow
                in nOrder (number of atomic layers
                along an edge).
            postAnalyzis (bool): If True, prints
                additional NP information (e.g.,
                cell parameters, moments of inertia,
                inscribed/circumscribed sphere
                diameters, etc.).
            aseView (bool): If True, enables
                visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision
                threshold for core/surface
                differentiation (distance threshold
                for retaining atoms).
            skipSymmetryAnalyzis (bool): If False,
                performs an atomic structure analysis
                using pymatgen.
            jmolCrystalShape (bool): If True,
                generates a JMOL script for
                visualization.
            noOutput (bool): If False, prints
                details about the NP structure.
            calcPropOnly (bool): If False, generates
                the atomic structure of the NP.

        Attributes:
            nAtoms (int): Number of atoms in the NP.
            cog (np.array): Center of gravity of
                the NP.
        """
        if not isinstance(full_cube, cube):
            raise TypeError("full_cube must be an instance of the Class Cube")
        self.full_cube = full_cube
        self.nOrder_hollow = nOrder_hollow
        self.nAtoms = 0
        self.edgeLength = self.full_cube.edgeLength()
        self.nAtomsPerEdge = self.full_cube.nAtomsPerEdge
        self.cog = np.array([0., 0., 0.])
        self.jmolCrystalShape = jmolCrystalShape
        if not calcPropOnly:
            self.create_hollow(noOutput)

            # if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(
                    skipSymmetryAnalyzis,
                    thresholdCoreSurface,
                    noOutput=noOutput
                )
            # #     if aseView: view(self.NPcs)

    def __str__(self):
        """Return a string representation of the object."""
        return (
            f"Cube with order of "
            f"{self.full_cube.nOrder}, with hollow "
            f"thickness of order "
            f"{self.nOrder_hollow} and with "
            f"Rnn = {self.full_cube.Rnn}"
        )

    def create_hollow(self, noOutput):
        """Function that creates the cube hollow.

        The hollow is created using planes that
        defines the hollow [h k l d] with
        d= +/- size of the hollow/2.
        Update: the hollow size is given in nOrder
        of atomic layers.

        Args:
            noOutput (bool): If False, prints details about the NP structure.
        """
        if not noOutput:
            print(f"Number of atoms on an edge = {self.nAtomsPerEdge}")
            print(f"Edge length = {round(self.edgeLength * 0.1, 3)} nm")
            print(f"Creating a hollow of nOrder = "
                  f"{self.nOrder_hollow}.")

        # Find half_inner_cube_size in Angs based
        # on the number of atomic layers (nOrder)

        # Length of one cell (one order)
        order_length = (
            self.edgeLength / self.full_cube.nOrder
        )
        inner_cube_size = self.nOrder_hollow * order_length

        # Add a tolerance of order_length/4 so the cut planes lie
        # halfway between atomic layers (spaced a/2 in fcc/bcc).
        # Without this, boundary atoms sit exactly ON the planes
        # and are excluded by eps, giving 0 removed atoms for
        # small hollow sizes.
        half_inner_cube_size = inner_cube_size / 2 + order_length / 4
        self.NP = self.full_cube.NP.copy()
        print("Number of atoms in the cube before "
              "creating the hollow =",
              len(self.NP))
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

        delAbove = False  # delete atoms above/under the 6 planes
        current_positions = self.NP.get_positions()
        #     print(f"Plan used: {plane}, delAbove={delAbove}")

        # Generate the truncation
        AtomsUnderPlanes = pyNMBu.truncateAbovePlanes(
            planes=planes_with_dist,
            coords=current_positions,
            allP=True,
            delAbove=delAbove,
            debug=False,
            noOutput=False,
            eps=0.001,  # threshold distance
        )
        del self.NP[AtomsUnderPlanes]
        self.nAtoms = len(self.NP)
        if not noOutput:
            print(f"Number of atoms in the final hollow cube : {self.nAtoms}")
