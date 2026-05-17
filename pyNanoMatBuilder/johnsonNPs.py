# External dependencies
import sys
import numpy as np

import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view
import matplotlib.pyplot as plt

# Internal Relative Imports
from . import visualID as vID
from . import data
from . import utils as pyNMBu
from . import platonicNPs as pNP
from .utils import hl, fg, bg
from .pyNMBcore import pyNMBcore
from .platonicNPs import regfccOh

###########################################################
class JohnsonNP(pyNMBcore):
    """Base class for all Johson nanoparticles providing common functionality."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class fcctbp(JohnsonNP):
    """
    A class for generating fcc trigonal bipyramidal (fcctbp)
    nanoparticles (NPs) with a user-defined number of atomic
    layers and interatomic distances.

    Key Features:
    
    - Generates a trigonal bipyramid shape with customizable
      atomic layers.
    - Computes structural properties like inter-layer distance,
      angles, and number of atoms.
    - Provides options for core/surface differentiation and
      symmetry analysis.
    - Supports visualization via ASE and Jmol for 3D structure
      representation.

    Additional Notes:
    
    - The `nLayerTd` parameter determines the number of layers
      in one pyramid.
    - The `Rnn` parameter controls the nearest neighbor
      interatomic distance.
    - Symmetry analysis can be skipped to speed up computations.
    - Visualization can be enabled for both the initial structure
      and post-processing stages.
      
    """

    nFaces = 6
    nEdges = 9
    nVertices = 5
    edgeLengthF = 2
    heightOfPyramidF = edgeLengthF * np.sqrt(2 / 3)
    heightOfBiPyramidF = 2 * heightOfPyramidF

    def __init__(self, element: str = 'Au',
                 Rnn: float = 2.7,
                 nLayerTd: int = 1,
                 **kwargs
                ):

        """
        Initializes an fcc trigonal bipyramid (fcctbp) with the given parameters.

        Args:
            element (str): Chemical element used to
                construct the NP (e.g., "Au").
            Rnn (float): Nearest neighbor distance (in Å).
            nLayerTd (int): Number of atomic layers in
                each pyramid (default: 1).
            postAnalyzis (bool): If True, prints additional
                NP information (e.g., cell parameters,
                moments of inertia, inscribed/circumscribed
                sphere diameters, etc.).
            aseView (bool): If True, enables visualization
                of the NP using ASE.
            thresholdCoreSurface (float): Threshold for
                core/surface classification (default: 1.0).
            skipSymmetryAnalyzis (bool): If False, performs
                symmetry analysis (default: False).
            jmolCrystalShape (bool): If True, generates a
                JMOL-compatible crystal shape (default: True).
            noOutput (bool): If True, suppresses text output (default: False).
            calcPropOnly (bool): If True, only calculates
                properties without generating structure
                (default: False).

        Attributes:
            nAtoms (int): Total number of atoms in the NP.
            nAtomsPerLayer (list): Number of atoms in each atomic layer.
            nAtomsPerEdge (int): Number of atoms per edge of the bipyramid.
            interLayerDistance (float): Distance between atomic layers.
            fveAngle (float): Face-vertex-edge angle.
            fefAngle (float): Face-edge-face angle.
            vcvAngle (float): Vertex-center-vertex angle.
            heightOfBiPyramid (float): Height of the bipyramid.
            imageFile (str): Path to the image representing the structure.
            cog (np.array): Center of gravity of the NP.
            trPlanes (np.array): Truncation plane equations.
        """
        
        super().__init__(**kwargs)
        self.element = element
        self.shape = 'fcctbp'
        self.Rnn = Rnn
        self.nLayerTd = int(nLayerTd)
        self.Tdprop = pNP.regfccTd(
            self.element, self.Rnn, self.nLayerTd,
            noOutput=True, calcPropOnly=True)
        self.nLayer = 2 * self.nLayerTd - 1
        self.nAtomsPerLayer = []
        self.interLayerDistance = self.Tdprop.interLayerDistance()
        self.nAtomsPerEdge = self.nLayerTd + 1
        self.fveAngle = self.Tdprop.fveAngle
        self.fefAngle = self.Tdprop.fefAngle
        self.vcvAngle = self.Tdprop.vcvAngle
        self.heightOfBiPyramid = (
            2 * self.Tdprop.heightOfPyramid()
            + 2 * (self.Rnn * np.sqrt(2 / 3)))
        self.imageFile = pyNMBu.imageNameWithPathway("tbp-C.png")
        noOutput = self.noOutput

        if not noOutput:
            pyNMBu.centerTitle(
                f"fcc trigonal bipyramid with"
                f" {nLayerTd} shells per pyramid")

        if not noOutput:
            self.prop()
        if not self.calcPropOnly:
            self.coords(noOutput)
            if self.aseView:
                view(self.NP)
            if self.postAnalyzis:
                self.propPostMake(
                    self.skipChiralityCalculation,
                    self.skipSymmetryAnalyzis,
                    self.skipFacetInfo,
                    self.thresholdCoreSurface, noOutput)
                if self.aseView:
                    view(self.NPcs)

    def __str__(self):
        return (f"Regular fcc double tetrahedron of"
                f" {self.element} with {self.nLayer + 1}"
                f" layer(s) and Rnn = {self.Rnn}")

    def edgeLength(self):
        """Computes the edge length of the pyramids in Å
        using the class regfccTd of the Platonic module.
        """
        return self.Tdprop.edgeLength() + self.Rnn

    def coords(self, noOutput):
        """Generates the coordinates of the fcc trigonal
        bipyramid.

        First creates a regular fcc tetrahedron and then
        apply a mirror reflection to create the bipyramid.
        Calculates and centers the nanoparticle's
        coordinates and finally computes the total number
        of atoms and the center of mass.

        Args:
            noOutput (bool): Whether to suppress output
                messages.
        """
        if not noOutput:
            pyNMBu.centertxt(
                "Generation of coordinates",
                bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        if not noOutput:
            pyNMBu.centertxt(
                "Generation of the coordinates"
                " of the tetrahedron",
                bgc='#cbcbcb', size='12',
                fgc='b', weight='bold')

        # Create regular fcc tetrahedron
        Td = pNP.regfccTd(
            self.element, self.Rnn, self.nLayerTd + 1,
            postAnalyzis=False, noOutput=True)
        aseTd = Td.NP
        self.NP0 = aseTd.copy()

        # Get the positions of the atoms in the tetrahedron
        c = aseTd.get_positions()
        if not noOutput:
            pyNMBu.centertxt(
                "Applying mirror reflection w.r.t."
                " facet defined by atoms (0,1,2) ",
                bgc='#cbcbcb', size='12',
                fgc='b', weight='bold')

        # Define the mirror plane
        mirrorPlane = [0, 1, 2]
        cMirrorPlane = []
        for at in mirrorPlane:
            cMirrorPlane.append(aseTd.get_positions()[at])
        cMirrorPlane = np.array(cMirrorPlane)
        mirrorPlane = pyNMBu.planeFittingLSF(
            cMirrorPlane, False, False)
        pyNMBu.convertuvwh2hkld(
            mirrorPlane, prthkld=not noOutput)

        # Apply the reflection operation on the tetrahedral
        # structure using the mirror plane
        cr = pyNMBu.reflection(
            mirrorPlane, aseTd.get_positions())
        nMirroredAtoms = len(cr)
        aseMirror = ase.Atoms(
            self.element * nMirroredAtoms, positions=cr)
        aseObject = aseTd + aseMirror

        # Calculate the center of mass (CoG)
        c = pyNMBu.center2cog(aseObject.get_positions())

        # Get the total number of atoms
        nAtoms = aseObject.get_global_number_of_atoms()

        aseObject = ase.Atoms(
            self.element * nAtoms, positions=c)
        if not noOutput:
            print(f"Total number of atoms = {nAtoms}")
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()
        self.nAtoms = nAtoms
        
    def prop(self):
        """
        Display unit cell and nanoparticle properties.
        """
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ", self.element)
        print("number of vertices = ", self.nVertices)
        print("number of edges = ", self.nEdges)
        print("number of faces = ", self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength() * 0.1:.2f} nm")
        print(f"inter-layer distance"
              f" = {self.interLayerDistance:.2f} Å")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"height of bipyramid"
              f" = {self.heightOfBiPyramid * 0.1:.2f} nm")
        print(f"area"
              f" = {6 * self.Tdprop.area() / 4 * 1e-2:.1f} nm2")
        print(f"volume"
              f" = {2 * self.Tdprop.volume() * 1e-3:.1f} nm3")
        print(f"face-vertex-edge angle"
              f" = {self.Tdprop.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle"
              f" = {self.Tdprop.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond)"
              f" angle = {self.Tdprop.vcvAngle:.1f}°")
        # print("number of atoms per layer = ",self.Tdprop.nAtomsPerLayerAnalytic())
        # print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: triangular prism")
        print("Indexes of vertex atoms = [0,1,2,3] by construction")
        print(f"coordinates of the center of gravity = {self.cog}")

###########################################################
class epbpyM(JohnsonNP):
    """A class for generating pentagonal bipyramidal
    (epbpyM) nanoparticles (NPs) with a user-defined size.

    Important note: the size must be carefully defined.
    It is represented as a three-element array, where:

    - The size of the pentagon (number of bonds per
      pentagonal edge),
    - The size of the elongated part (number of bonds
      per elongated edge),
    - The number of truncated atoms at the vertices
      (the number of truncated atoms per pentagonal
      edge is doubled since there are two vertices
      per edge).

    Key Features:

    - Generates a trigonal bipyramid shape with
      customizable atomic layers.
    - Computes structural properties like inter-layer
      distance, angles, and number of atoms.
    - Provides options for core/surface differentiation
      and symmetry analysis.
    - Supports visualization via ASE and Jmol for 3D
      structure representation.

    Additional Notes:

    - The user must ensure that the number of truncated
      atoms per edge is smaller than the number of atoms
      on the edges. An example of sizes is given.
    - Symmetry analysis can be skipped to speed up
      computations.
    - Visualization can be enabled for both the initial
      structure and post-processing stages.
    """
    nFaces3 = 10
    nFaces4 = 5
    nEdgesPbpy = 15
    nEdgesEpbpy = 20
    nVerticesPbpy = 7
    nVerticesEpbpy = 12
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    edgeLengthF = 1
    heightOfPyramidF = (
        np.sqrt((5 - np.sqrt(5)) / 10) * edgeLengthF
    )
    # interCompactPlanesF: ratio between the {111} interplanar spacing and Rnn
    # for the pentagonal bipyramid geometry. Equals phi/2 where phi = (1+sqrt(5))/2
    # is the golden ratio, a direct consequence of the 5-fold icosahedral symmetry.
    # Note: this differs slightly from the ideal fcc value sqrt(2/3) ≈ 0.816
    # because the pentagonal bipyramid is an intrinsically strained structure
    # (disclination angle ~7.35°) — the 5-fold symmetry cannot tile space exactly.
    interCompactPlanesF = (1 + np.sqrt(5)) / 4
    magicFactorF = (
        2 * heightOfPyramidF / edgeLengthF
    )

    def __init__(self,
                 element: str = 'Au',
                Rnn: float = 2.7,
                sizeP: int = 1,
                sizeE: int = 0,
                Marks: int = 0,
                Multiples_index_plan: bool = False,
                Hollow: bool = False,
                **kwargs
                ):
        """Initialize the class with the necessary parameters.

        Args:
            element (str): Chemical element
                (e.g., 'Au', 'Fe').
            Rnn (float): Nearest neighbor interatomic
                distance in Å.
            sizeP (int): Number of bonds per pentagonal
                edge, defining the size of the pentagon.
            sizeE (int): Number of bonds per elongated
                edge, defining the size of the elongated
                part.
            Marks (int): Number of truncated atoms at
                the vertices.
            postAnalyzis (bool): If True, performs
                post-analysis on the generated NP.
            aseView (bool): If True, visualizes the
                nanoparticle structure using ASE.
            thresholdCoreSurface (float): Precision
                threshold for core/surface differentiation.
            skipSymmetryAnalyzis (bool): If False,
                performs symmetry analysis on the structure.
            jmolCrystalShape (bool): If True, generates a
                Jmol script for visualization.
            noOutput (bool): If True, suppresses output.
            calcPropOnly (bool): If True, only calculates
                properties without generating the NP.

        Attributes:
            element (str): The chemical element.
            shape (str): The shape ('epbpyM').
            Rnn (float): Nearest neighbor interatomic
                distance.
            sizeP (int): Number of bonds per pentagonal
                edge.
            sizeE (int): Number of bonds per elongated
                edge.
            Marks (int): Number of truncated atoms.
            nAtoms (int): Total number of atoms.
            nAtomsPerPentagonalCap (int): Atoms in the
                pentagonal caps.
            nAtomsPerElongatedPart (int): Atoms in the
                elongated part.
            nAtomsPerEdgeOfPC (int): Atoms per edge of
                the pentagonal cap (before truncation).
            nAtomsPerEdgeOfEP (int): Atoms per edge of
                the elongated part.
            jmolCrystalShape (bool): Whether to generate
                a Jmol script for visualization.
            cog (np.array): Center of gravity (CoG).
            interCompactPlanesDistance (float): Distance
                between compact planes.
            imageFile (str): Path to the image file.
        """
        super().__init__(**kwargs)
        self.element = element
        self.shape = 'epbpyM'
        self.Rnn = Rnn
        self.sizeP = sizeP
        self.sizeE = sizeE
        self.Marks = Marks
        # For the creation of (11n) bipyramids
        self.Multiples_index_plan = Multiples_index_plan
        self.Hollow = Hollow
        self.nAtomsPerPentagonalCap = 0
        self.nAtomsPerElongatedPart = 0
        self.nAtomsPerEdgeOfPC = self.sizeP + 1
        self.nAtomsPerEdgeOfEP = self.sizeE + 1
        self.interCompactPlanesDistance = self.interCompactPlanesF * self.Rnn
        noOutput = self.noOutput
        if self.Marks == 0 and self.sizeE == 0:
            # pentagonal bpy
            self.imageFile = pyNMBu.imageNameWithPathway(
                "pbpy-C.png")
        elif self.Marks != 0 and self.sizeE == 0:
            # Marks decahedron
            self.imageFile = pyNMBu.imageNameWithPathway(
                "MarksD-C.png")
        elif self.Marks == 0 and self.sizeE != 0:
            # Ino decahedron
            self.imageFile = pyNMBu.imageNameWithPathway(
                "InoD-C.png")
        else:
            # Elongated Marks decahedron
            self.imageFile = pyNMBu.imageNameWithPathway(
                "MarksD-C.png")
        if not noOutput:
            pyNMBu.centerTitle(
                f"Pentagonal bipyramid with {sizeP}"
                f" atoms/edge, a x{sizeE} elongation"
                f" (Ino) and a x{Marks} edge"
                f" truncation (Marks)")

        if not noOutput:
            self.prop()
        if not self.calcPropOnly:
            self.coords(noOutput)
            if self.aseView:
                view(self.NP)
            if self.postAnalyzis:
                self.propPostMake(
                    self.skipChiralityCalculation,
                    self.skipSymmetryAnalyzis,
                    self.skipFacetInfo,
                    self.thresholdCoreSurface,
                    noOutput)
                if self.aseView:
                    view(self.NPcs)

    def __str__(self):
        if self.Marks == 0:
            msg = (f"Pentagonal pyramid with {self.sizeP + 1}"
                   f" atoms per edge on the pentagonal cap,"
                   f" {self.sizeE} layer(s) in the elongated"
                   f" part, no Marks truncation and"
                   f" Rnn = {self.Rnn}")
        else:
            msg = (f"Pentagonal pyramid with {self.sizeP + 1}"
                   f" atoms per edge on the pentagonal cap,"
                   f" {self.sizeE} layer(s) in the elongated"
                   f" part, a Marks truncation by"
                   f" {self.Marks} atom(s) and"
                   f" Rnn = {self.Rnn}")
        return(msg)

    def edgeLength(self, whichEdge):
        """Compute the edge length of pentagon and the
        elongated part in Å using Rnn and
        nAtomsPerEdgeOfPC / nAtomsPerEdgeOfEP.
        """
        if whichEdge == 'PC':  # pentagonal cap
            return self.Rnn * (self.nAtomsPerEdgeOfPC - 1)
        elif whichEdge == 'EP':  # elongated part
            return (self.Rnn * self.magicFactorF
                    * (self.nAtomsPerEdgeOfEP - 1))

    # NEW
    def nAtomsPerEdgeOfPC_after_truncation(self):
        """Compute the number of atoms per edge of the
        pentagon after truncation, using
        nAtomsPerEdgeOfPC and Marks.
        """
        return self.nAtomsPerEdgeOfPC - 2 * self.Marks
    
    def edgeLength_after_truncation(self, whichEdge):
        """Compute the edge length of pentagon and the
        elongated part after truncation in Å, using Rnn,
        nAtomsPerEdgeOfPC / nAtomsPerEdgeOfEP and Marks.
        """
        if whichEdge == 'PC':  # pentagonal cap
            nAfter = (
                self.nAtomsPerEdgeOfPC_after_truncation()
            )
            if nAfter < 2:
                return self.Rnn
            else:
                return (self.edgeLength('PC')
                        - 2 * self.Marks * self.Rnn)
        elif whichEdge == 'EP':  # elongated part
            return self.edgeLength('EP') 
        


    def area(self):
        """Compute the surface area in square Å."""
        el = self.edgeLength('PC')
        return el**2 * (5 * np.sqrt(3) / 2)
    
    def volume(self):
        """Compute the volume in cubic Å."""
        el = self.edgeLength('PC')
        return el**3 * (5 + np.sqrt(5)) / 12

    def heightOfPyramid(self):
        """Compute the height of the pyramid in Å."""
        print("heightOfPyramidF = ", self.heightOfPyramidF)
        print("natomsPerEdgeOfPC = ", self.nAtomsPerEdgeOfPC)
        return (self.heightOfPyramidF * self.Rnn
                * (self.nAtomsPerEdgeOfPC)) * 2 - 2.84 #
                # 2.84 Angs is the dAu-Au along the height

    def MakeVertices(self):
        """Generate the coordinates of the vertices,
        edges, and faces of a pentagonal bipyramid.

        Returns:
            CoordVertices (np.ndarray): the 7 vertex
                coordinates of a pentagonal dipyramid
            edges (np.ndarray): indexes of the 2x5
                "vertical" edges of the pentagonal cap
                of an elongated pentagonal bipyramid
            faces (np.ndarray): indexes of the 10
                triangular faces
        """
        phi = self.phi
        scale = self.Rnn / 2
        sP = self.sizeP
        sE = self.sizeE
        mF = self.magicFactorF

        CoordVertices = [
            pyNMBu.vertexScaled(
                0, 0,
                sE * mF
                + sP * np.sqrt((10 - 2 * np.sqrt(5))
                               / 5),
                scale),
            pyNMBu.vertexScaled(
                sP * np.sqrt(
                    (10 + 2 * np.sqrt(5)) / 5),
                0, sE * mF, scale),
            pyNMBu.vertexScaled(
                sP * np.sqrt(
                    (5 - np.sqrt(5)) / 10),
                sP * phi, sE * mF, scale),
            pyNMBu.vertexScaled(
                sP * -np.sqrt(
                    (5 + 2 * np.sqrt(5)) / 5),
                sP, sE * mF, scale),
            pyNMBu.vertexScaled(
                sP * -np.sqrt(
                    (5 + 2 * np.sqrt(5)) / 5),
                -sP, sE * mF, scale),
            pyNMBu.vertexScaled(
                sP * np.sqrt(
                    (5 - np.sqrt(5)) / 10),
                -sP * phi, sE * mF, scale),
        ]

        edgesPentagonalCap = [
            (0, 1), (0, 2), (0, 3),
            (0, 4), (0, 5),
            (1, 2), (2, 3), (3, 4),
            (4, 5), (5, 1)]
        faces3 = [
            (0, 1, 2), (0, 2, 3), (0, 3, 4),
            (0, 4, 5), (0, 5, 1)]

        CoordVertices = np.array(CoordVertices)
        edgesPentagonalCap = np.array(edgesPentagonalCap)
        faces3 = np.array(faces3)
        return CoordVertices, edgesPentagonalCap, faces3

    def truncationPlaneTuples4MarksDecahedron(
            self, refPlaneAtoms, debug=False):
        """Calculate the truncation planes for the Marks
        decahedron nanoparticle shape. It defines planes
        that truncate the vertices of the decahedron.

        Args:
            refPlaneAtoms (list): A list of three atoms'
                coordinates that define a reference plane.
                The list should contain:
                - summit atom (topmost point)
                - apex atom (the second point)
                - origin atom (used as reference)
            debug (bool): flag to enable or disable
                debugging information (default is False).

        Returns:
            planes (array-like): An array of 5 planes
                used for truncation, each defined by a
                normal vector and an offset.
            indicesOfTruncationPlanes (list): A list of
                tuples, each containing two indices that
                define the truncation plane pairs.
        """

        # Fit a plane through the three reference atoms (summit, apex, origin)
        pRef = pyNMBu.planeFittingLSF(
            refPlaneAtoms,
            printEq=False,
            printErrors=False)
        pRef = pRef[0:3]
        O = refPlaneAtoms[2]
        apexC = refPlaneAtoms[1]

        # Calculate the distance between planes based on inter-atomic spacing
        interPlanarDistance = self.interCompactPlanesDistance
        d = -interPlanarDistance * (self.nAtomsPerEdgeOfPC - self.Marks - 1)
        # Initialize the first plane using the fitted plane
        # and calculated distance
        plane0 = np.append(pRef, [d])
        planes = [plane0]
        # Generate the next 4 planes by rotating the base plane
        # around the z-axis by 72°
        for i in range(1, 5):
            angle = i * 72
            x = pyNMBu.RotationMol(pRef, angle, 'z')
            x = np.append(x, [d])
            planes.append(x)
            norm = pyNMBu.normV(x)
            if debug:
                print("angle = ", angle, "  plane = ", x,
                      "   norm = ", norm)
        planes = np.array(planes)

        # Define the indices of the pairs of truncation plane
        indices = [0, 1, 2, 3, 4, 0, 1]
        indicesOfTruncationPlanes = []

        # Generate the pairs of truncation planes using the indices
        for i in range(0, 5):
            tuple = (indices[i], indices[i + 2])
            indicesOfTruncationPlanes.append(tuple)
        if debug:
            print("\nIndices of couples of truncation planes:\n",
                  indicesOfTruncationPlanes)
        # Return the planes and their respective index pairs
        return planes, indicesOfTruncationPlanes


    def coords(self, noOutput):
        """Generate coordinates of all atoms in the
        nanoparticle structure, including vertices,
        edges, faces, and internal atoms. Applies
        truncation and symmetry operations where
        necessary.

        Args:
            noOutput (bool): if True, suppresses output
                messages during execution.

        Returns:
            None. Updates class attributes directly.
        """
        if not noOutput:
            pyNMBu.centertxt(
                "Generation of coordinates",
                bgc='#007a7a', size='14',
                weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        c = []  # List of atom coordinates
        # print(self.nAtomsPerLayer)
        indexVertexAtoms = []
        indexEdgePCAtoms = []
        indexEdgeEPAtoms = []
        indexFace3Atoms = []
        indexCoreAtoms = []

        # --- Vertices ---
        # Initialize atom count and generate the 6 vertices of the
        # pentagonal bipyramid (5 equatorial + 1 apical).
        # MakeVertices() also returns edge and face connectivity.
        nAtoms0 = 0
        self.nAtoms = 6
        cVertices, E, F3 = self.MakeVertices()
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0, self.nAtoms))

        # --- Edge atoms ---
        # Interpolate atoms along each edge between two vertices.
        # The number of interpolated atoms per edge is determined by
        # the vertex-vertex distance divided by Rnn, minus the 2 endpoint
        # vertices which are already in c.
        nAtoms0 = self.nAtoms
        Rvv = pyNMBu.RAB(cVertices, E[0, 0], E[0, 1])  # dist vertex-vertex
        nAtomsOnEdges = int((Rvv + 1e-6) / self.Rnn) - 1
        nIntervals = nAtomsOnEdges + 1
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E:
                a = e[0]
                b = e[1]
                tmp = (cVertices[a]
                       + pyNMBu.vector(cVertices, a, b)
                       * (n + 1) / nIntervals)
                coordEdgeAt.append(tmp)
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        # CAtoms.extend(range(nAtoms0,self.nAtoms))
        self.nAtomsPerEdgeOfPC = nAtomsOnEdges + 2  # 2 vertices

        # --- Triangular face atoms ---
        # Fill each triangular face of the upper pyramid with atoms
        # by interpolating between edge atoms using MakeFaceCoord.
        coordFace3At = []
        nAtomsOnFaces3 = 0
        nAtoms0 = self.nAtoms
        for f in F3:
            nAtomsOnFaces3, coordFace3At = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsOnFaces3, coordFace3At)
        self.nAtoms += nAtomsOnFaces3
        c.extend(coordFace3At)
        indexFace3Atoms.extend(range(nAtoms0, self.nAtoms))

        # --- Marks truncation ---
        # If Marks > 0, truncate the vertical edges of the upper pyramid
        # using pairs of planes (truncationPlaneTuples4MarksDecahedron).
        # Each pair of planes removes atoms above both planes simultaneously,
        # creating the characteristic re-entrant {111} facets of the
        # Marks decahedron.
        if self.Marks != 0:
            pMarks, indexCouples = (
                self.truncationPlaneTuples4MarksDecahedron(
                    np.array([c[0], c[1], [0, 0, 0]])))
            for ic in indexCouples:  # truncation planes
                p0 = pMarks[ic[0]].copy()
                p1 = pMarks[ic[1]].copy()
                p1[0:3] = -p1[0:3]  # change sign, see scheme in Sandbox
                planes = [p0, p1]
                AtomsAboveAllPlanes = pyNMBu.truncateAbovePlanes(
                    planes, c, allP=True, delAbove=True,
                    debug=False, noOutput=noOutput)
                c = pyNMBu.deleteElementsOfAList(c, AtomsAboveAllPlanes)
            self.nAtoms = len(c)

        # --- Mirror reflection ---
        # Reflect the upper half (pyramid + elongated part) with respect
        # to the equatorial plane z=0 to generate the lower half.
        # The reflection function avoids duplicating atoms that lie exactly
        # in the mirror plane (z=0).
        symPlane = np.array([0, 0, 1, 0])
        ReflectionAtoms = pyNMBu.reflection(symPlane, c, True)
        c.extend(ReflectionAtoms)
        self.nAtoms += len(ReflectionAtoms)

        # --- Core atoms ---
        # Fill the interior of the bipyramid by interpolating between each
        # surface atom in the upper half and its mirror counterpart in the
        # lower half. The number of interpolated atoms is proportional to
        # the mirror distance divided by Rnn, corrected by magicFactorF
        # (the ratio between the inter-plane spacing and Rnn for this
        # crystal geometry).
        nAtomsHalfPy = int(self.nAtoms / 2)
        coordCoreAt = []
        for i in range(nAtomsHalfPy):
            Rvv = pyNMBu.RAB(c, i, i + nAtomsHalfPy)  # dist mirror atoms
            nAtomsInCore = (
                int((Rvv + 1e-6) / self.Rnn * self.magicFactorF) - 1)
            nIntervals = nAtomsInCore + 1
            for n in range(nAtomsInCore):
                tmp = (c[i]
                       + pyNMBu.vector(c, i, i + nAtomsHalfPy)
                       * (n + 1) / nIntervals)
                coordCoreAt.append(tmp)
        self.nAtoms += len(coordCoreAt)
        c.extend(coordCoreAt)

        # if self.sizeE = 0, it's a pentagonal bipyramid or a
        # Marks decahedron without side faces
        # given the doItForAtomsThatLieInTheReflectionPlane trick
        # in pyNMBu.reflection, it is necessary to remove duplicates
        if self.sizeE == 0:
            c = np.unique(np.array(c), axis=0)
            self.nAtoms = len(c)

        # --- Store results ---
        aseObject = ase.Atoms(self.element * self.nAtoms, positions=c)

        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()

    def prop(self):
        """Display unit cell and nanoparticle properties."""
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ", self.element)
        if self.sizeE == 0 and self.Marks == 0:
            print("number of vertices = ", self.nVerticesPbpy)
            print("number of edges = ", self.nEdgesPbpy)
            print("number of faces = ", self.nFaces3)
        elif self.sizeE != 0 and self.Marks == 0:
            print("number of vertices = ", self.nVerticesEpbpy)
            print("number of edges = ", self.nEdgesEpbpy)
            print("number of faces = ",
                  self.nFaces3 + self.nFaces4)
        print(f"magic factor = {self.magicFactorF:.3f}"
              f" (ratio between the height of the vertical"
              f" interatomic distance and the"
              f" nearest-neighbour distance)")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")

        if self.Marks != 0: 
            print("")
            print(f"Dimensions before Marks truncation:")
        print(f"edge length of the pentagonal cap"
              f" = {self.edgeLength('PC') * 0.1:.2f} nm")
        print(f"edge length of the elongated part"
              f" = {self.edgeLength('EP') * 0.1:.2f} nm")
        print(f"inter compact planes factor"
              f" = {self.interCompactPlanesF:.3f}")
        print(f"inter compact planes distance"
              f" = {self.interCompactPlanesDistance:.2f} Å")
        # print(f"inter-layer distance = "
        #       f"{self.interLayerDistance:.2f} Å")
        print(f"number of atoms per edge on the pentagonal"
              f" cap = {self.nAtomsPerEdgeOfPC}")
        if self.Marks == 0:
            structure = "bipyramid"
        else:
            structure = "Marks decahedron"
            print("")
            print(f"Dimensions after truncation by {self.Marks} atom(s) at the vertices:")
            print(f"number of atoms per edge of the pentagonal cap after"
                  f" truncation = {self.nAtomsPerEdgeOfPC_after_truncation()}")
            print(f"edge length of the pentagonal cap after"
                  f" truncation = {self.edgeLength_after_truncation('PC') * 0.1:.2f} nm")
            print(f"edge length of the elongated part after"
                  f" truncation = {self.edgeLength_after_truncation('EP') * 0.1:.2f} nm")
        if self.sizeE == 0:
            print(f"height of the {structure}"
                  f" = {self.heightOfPyramid():.2f} Å")
            if self.Marks == 0:
                print(f"area = {self.area() * 1e-2:.1f} nm2")
                print(f"volume = {self.volume() * 1e-3:.1f} nm3")
        elif self.sizeE != 0:
            h = (self.heightOfPyramid()
                 + self.edgeLength('EP'))
            print(f"height of the elongated {structure}"
                  f" = {h * 0.1:.2f} nm")
        # print("number of atoms per layer = ",
        #       self.Td.nAtomsPerLayerAnalytic())
        # print("total number of atoms = ",
        #       self.nAtomsAnalytic())
        if self.sizeE == 0 and self.Marks == 0:
            print("Dual polyhedron: triangular prism")
        elif self.sizeE != 0 and self.Marks == 0:
            print("Dual polyhedron: pentagonal bifrustum")
        print(f"coordinates of the center of gravity = {self.cog}")
    

class eOhM(JohnsonNP):
    """
    A class for generating elongated fcc octahedral (eOhM) nanoparticles
    with optional Marks-like truncation of the vertical edges.

    Inherits from regfccOh and extends it with:

    - Elongation along [001] (Ino-like) via sizeE parameter, where each
      sizeE unit corresponds to one fcc unit cell along [001].
    - Marks truncation of vertical <110> edges via Marks parameter.

    The octahedron has 8 {111} faces. Elongation adds 4 {100} square faces
    between the two equatorial crowns. Marks truncation adds {110} facets
    on the vertical edges.

    Key Features:

    - Generates an elongated octahedron with customizable atomic layers.
    - Computes structural properties like inter-plane distance, edge length,
      area, volume, and total height.
    - Provides options for core/surface differentiation and symmetry analysis.
    - Supports visualization via ASE and Jmol for 3D structure representation.

    Additional Notes:

    - The user must ensure that the number of truncated atoms on the
      vertical edges is smaller than the number of atoms on the edges.
    - Symmetry analysis can be skipped to speed up computations.
    - Marks truncation is not yet implemented (TODO).
    """

    # --- Topological invariants ---
    nEdges        = 8    # edges of regular square pyramid
    nFaces3       = 4    # triangular {111} faces of regular square pyramid
    nFaces4       = 1    # no square faces in regular square pyramid
    nVertices     = 5    # vertices of regular square pyramid
    nEdges_Oh     = 12   # edges of regular Oh
    nFaces3_Oh    = 4    # triangular {111} faces of regular Oh
    nFaces4_Oh    = 8    # no square faces in regular Oh
    nVertices_Oh  = 6    # vertices of regular Oh
    nEdges_eOh    = 20   # edges of elongated octahedron
    nFaces3_eOh   = 8    # triangular {111} faces of elongated octahedron
    nFaces4_eOh   = 4    # square {100} faces in elongated octahedron
    nVertices_eOh = 10   # vertices of elongated octahedron

    # --- Geometric factors (class attributes) ---
    edgeLengthF             = 1
    intershellF             = np.sqrt(2) / edgeLengthF
    halfHeightF             = 1.0 / np.sqrt(2)
    interCompactPlanesF_111 = np.sqrt(2/3)  # √(2/3) ≈ 0.816
    interCompactPlanesF_110 = 1.0
    interCompactPlanesF_100 = np.sqrt(2)
    interCompactPlanesF_001 = np.sqrt(2)

    def __init__(self,
                 element: str = 'Au',
                 Rnn: float = 2.88,
                 nOrder: int = 1,
                 wire_length: int = 0,
                 wire_width: int = None,
                 Marks_110: int = 0,
                 Marks_100: int = 0,
                 Marks_on_wire_only: bool=False,
                 **kwargs):
        """Initialize the class with the necessary parameters.

        Args:
            element (str): Chemical element (e.g., 'Au', 'Fe').
            Rnn (float): Nearest-neighbor interatomic distance in Å.
                Default is 2.88 (Au fcc).
            nOrder (int): Number of atomic layers along an edge of the
                octahedron (e.g., nOrder=1 means 2 atoms per edge).
                Default is 1.
            wire_length (int): Number of fcc unit cells along [001] separating
                the two equatorial crowns. Default is 0 (regular octahedron).
            wire_width : width of the wire. If None, will be initialized as nOrder.
                wire_width < nOrder will make a double arrow
            Marks (int): Number of truncated atoms on vertical <110> edges
                . Default is 0.
            Marks_on_wire_only (bool): Apply the Marks-like truncation on the
                wire only (Default: False).
            postAnalyzis (bool): If True, performs post-analysis on the
                generated NP.
            aseView (bool): If True, visualizes the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for
                core/surface differentiation.
            skipSymmetryAnalyzis (bool): If False, performs symmetry analysis.
            jmolCrystalShape (bool): If True, generates a Jmol script.
            noOutput (bool): If True, suppresses output.
            calcPropOnly (bool): If True, only calculates properties without
                generating the NP.

        Attributes:
            element (str): The chemical element.
            shape (str): The shape ('eOhM').
            Rnn (float): Nearest-neighbor distance.
            nOrder (int): Atomic layers along edge.
            wire_length (int): Elongation in fcc (1,0,0) unit cells.
            Marks (int): Number of truncated atoms.
            nAtoms (int): Total number of atoms.
            interCompactPlanesDistance_111 (float): d{111} in Å.
            interCompactPlanesDistance_100 (float): d{100} in Å.
            cog (np.array): Center of gravity.
            imageFile (str): Path to the reference image.
        """
        super().__init__(**kwargs)
        self.element = element
        self.shape = 'eOhM'
        self.Rnn = Rnn
        self.nOrder = nOrder
        self.wire_length = wire_length
        if wire_width is None:
            self.wire_width = nOrder
        else:
            self.wire_width = wire_width
        self.Marks_100 = Marks_100
        self.Marks_110 = Marks_110
        self.Marks_on_wire_only = Marks_on_wire_only
        self.interCompactPlanesDistance_111 = self.interCompactPlanesF_111 * Rnn
        self.interCompactPlanesDistance_110 = self.interCompactPlanesF_110 * Rnn
        self.interCompactPlanesDistance_100 = self.interCompactPlanesF_100 * Rnn
        self.interCompactPlanesDistance_001 = self.interCompactPlanesF_001 * Rnn
        self.interLayerDistance = Rnn / np.sqrt(2)  # = a/2 = d{001}

        # Image file selection
        has_marks = (self.Marks_110 != 0 or self.Marks_100 != 0)
        if not has_marks and self.wire_length == 0:
            self.imageFile = pyNMBu.imageNameWithPathway("fccOh-C.png")
        elif not has_marks and self.wire_length != 0:
            self.imageFile = pyNMBu.imageNameWithPathway("fccOh-C.png")  # à remplacer
        else:
            self.imageFile = pyNMBu.imageNameWithPathway("fccOh-C.png")  # à remplacer

        noOutput = self.noOutput

        # Now run eOhM-specific pipeline
        # (ignoring parent calcPropOnly=True forced above)
        if not noOutput:
            has_marks = (self.Marks_110 != 0 or self.Marks_100 != 0)
            marks_info = (f"Marks_110={self.Marks_110}, Marks_100={self.Marks_100}"
                          f"{' (wire only)' if self.Marks_on_wire_only else ''}"
                          if has_marks else "no Marks truncation")
            pyNMBu.centerTitle(
                f"Elongated fcc Octahedron with {nOrder} atoms/edge,"
                f" a x{wire_length} elongation along [001] (Ino-like),"
                f" {marks_info}")
            self.prop()

        if not kwargs.get('calcPropOnly', False):  # respect user-provided value
            self.coords(noOutput)
            if self.aseView:
                view(self.NP)
            if self.postAnalyzis:
                self.propPostMake(
                    self.skipChiralityCalculation,
                    self.skipSymmetryAnalyzis,
                    self.skipFacetInfo,
                    self.thresholdCoreSurface,
                    noOutput)
                if self.aseView:
                    view(self.NPcs)

    def __str__(self):
        has_marks = (self.Marks_110 != 0 or self.Marks_100 != 0)
        marks_str = (f"Marks_110={self.Marks_110}, Marks_100={self.Marks_100}"
                     f"{' (wire only)' if self.Marks_on_wire_only else ''}")

        if not has_marks and self.wire_length == 0:
            return (f"Regular fcc Octahedron with {self.nOrder + 1}"
                    f" atoms per edge, no Marks truncation,"
                    f" no elongation, Rnn = {self.Rnn} Å")
        elif not has_marks and self.wire_length != 0:
            return (f"Elongated fcc Octahedron with {self.nOrder + 1}"
                    f" atoms per edge, {self.wire_length} fcc unit cell(s)"
                    f" elongation along [001] (Ino-like),"
                    f" no Marks truncation, Rnn = {self.Rnn} Å")
        elif has_marks and self.wire_length == 0:
            return (f"Regular fcc Octahedron with {self.nOrder + 1}"
                    f" atoms per edge, {marks_str}, Rnn = {self.Rnn} Å")
        else:
            return (f"Elongated fcc Octahedron with {self.nOrder + 1}"
                    f" atoms per edge, {self.wire_length} fcc unit cell(s)"
                    f" elongation along [001], {marks_str}, Rnn = {self.Rnn} Å")

    # --- Geometric properties (instance methods) ---

    def edgeLength(self):
        """Compute the octahedral edge length in Å.

        Octahedral edges lie along <110> in fcc, so their length equals
        nOrder * Rnn.

        Returns:
            float: Edge length in Å.
        """
        return self.Rnn * self.nOrder

    def halfHeightPyramid(self):
        """Compute the half-height of one pyramidal cap along [001] in Å.

        For a regular octahedron of edge length L, the half-height equals
        L / sqrt(2).

        Returns:
            float: Half-height of one cap in Å.
        """
        return self.halfHeightF * self.edgeLength()

    def elongationLength(self):
        """Compute the total length of the elongated part along [001] in Å.

        Each elongation unit corresponds to one fcc unit cell along [001],
        i.e. a = Rnn * sqrt(2). The two crowns are separated by wire_length * a.

        Returns:
            float: Elongation length in Å.
        """
        return self.wire_length * self.Rnn * np.sqrt(2)

    def totalHeight(self):
        """Compute the total height of the eOhM nanoparticle along [001] in Å.

        Equals twice the pyramidal cap half-height plus the elongation length.

        Returns:
            float: Total height in Å.
        """
        return 2 * self.halfHeightPyramid() + self.elongationLength()

    def area(self):
        """Compute the surface area of the eOhM nanoparticle in square Å.

        Contributions:
        - 8 equilateral triangular {111} faces of edge length L.
        - 4 rectangular {100} faces of dimensions L × elongationLength
          (only when wire_length > 0).

        Returns:
            float: Surface area in Å².
        """
        L = self.edgeLength()
        area_111 = 8 * (np.sqrt(3) / 4) * L**2     # 8 equilateral triangles
        area_100 = 4 * L * self.elongationLength() # 4 rectangles (0 if wire_length=0)
        return area_111 + area_100

    def volume(self):
        """Compute the volume of the eOhM nanoparticle in cubic Å.

        Contributions:
        - Regular octahedron of edge length L: V = sqrt(2)/3 * L³.
        - Square prism of section L² and height elongationLength
          (only when wire_length > 0).

        Returns:
            float: Volume in Å³.
        """
        L = self.edgeLength()
        vol_oct   = np.sqrt(2) / 3 * L**3
        vol_prism = L**2 * self.elongationLength()
        return vol_oct + vol_prism

    def edge_atom_index(self, n, k, nEdges):
        """
        Return the index in the coordinate list c of the atom located
        at level n on edge E[k].
    
        Edge atoms in c are stored sequentially:
            for n in range(nAtomsOnEdges):
                for k in range(nEdges):
                    append atom on edge E[k] at level n
    
        Therefore the atom on edge E[k] at level n is at index:
            nVertices + n * nEdges + k
    
        Args:
            n (int): Level along the edge (0 = closest to start vertex,
                     nAtomsOnEdges-1 = closest to end vertex).
            k (int): Edge index in the edge list E.
            nEdges (int): Total number of edges (= len(E)).
    
        Returns:
            int: Index in the coordinate list c.
        """
        return self.nVertices + n * nEdges + k
            
    def MakeVertices_SquarePyramid(self, i):
        """
        Generates the coordinates of the vertices, edges, and faces
        for the ith shell of a square pyramid nanoparticle.

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
                             pyNMBu.vertex(0, 0, 1, scale)]
            edges = [
                (2, 0), (2, 1), (3, 0), (3, 1),
                (4, 0), (4, 1), (4, 2), (4, 3)
            ]
            faces3 = [
                (2, 0, 4),
                (2, 1, 4),
                (3, 0, 4),
                (3, 1, 4),
            ]
            face4 = [0, 3, 1, 2]
            CoordVertices = np.array(CoordVertices)
            edges = np.array(edges)
            faces3 = np.array(faces3)
            face4 = np.array(face4)
        return CoordVertices, edges, faces3, face4

    # def truncationPlanes4MarksOctahedron(self, debug=False):
    #     """
    #     Calculate the 4 {110} truncation planes for the Marks-like
    #     octahedron, cutting the 4 vertical corners of the wire section.
    #     Planes are rotated by 90° around [001].

    #     Args:
    #         debug (bool): If True, prints debug information.

    #     Returns:
    #         planes (np.ndarray): Array of 4 truncation planes [h,k,l,d].
    #         indicesOfTruncationPlanes (list): List of 4 pairs of plane
    #             indices used for truncation.
    #     """
    #     # Reference plane normal: [110] direction
    #     # The 4 vertical corners of the wire square are at:
    #     # [+scale, +scale, z], [-scale, +scale, z],
    #     # [-scale, -scale, z], [+scale, -scale, z]
    #     # where scale = wire_width * Rnn / sqrt(2) / sqrt(2) = wire_width * Rnn / 2
        
    #     # Normal to the first {110} truncation plane
    #     pRef = np.array([1., 1., 0.])
    #     pRef = pRef / np.linalg.norm(pRef)

    #     # Offset: distance from origin to the plane
    #     # = wire_width * Rnn/2 - Marks * d{110}
    #     # d{110} = Rnn (interCompactPlanesF_110 = 1.0)
    #     scale = self.wire_width * self.Rnn / np.sqrt(2)
    #     d = -(scale / np.sqrt(2) - self.Marks * self.interCompactPlanesDistance_110)

    #     # Generate 4 planes by rotating 90° around z-axis
    #     plane0 = np.append(pRef, [d])
    #     planes = [plane0]
    #     for i in range(1, 4):
    #         angle = i * 90
    #         x = pyNMBu.RotationMol(pRef, angle, 'z')
    #         x = np.append(x, [d])
    #         planes.append(x)
    #         if debug:
    #             print(f"angle={angle}°  plane={x}  norm={pyNMBu.normV(x)}")
    #     planes = np.array(planes)

    #     # Adjacent plane pairs for truncation (each corner cut by 2 planes)
    #     indices = [0, 1, 2, 3, 0, 1]
    #     indicesOfTruncationPlanes = [(indices[i], indices[i+1])
    #                                  for i in range(4)]
    #     if debug:
    #         print("Truncation plane pairs:", indicesOfTruncationPlanes)

    #     return planes, indicesOfTruncationPlanes

    def truncationPlanes4MarksOctahedron(self, debug=False):
        """
        Calculate the 8 truncation planes for the Marks-like octahedron:
        - 4 {110} planes cutting the vertical corners of the wire
        - 4 {100} planes cutting the vertical faces of the wire
        Both sets are rotated by 90° around [001].

        Args:
            debug (bool): If True, prints debug information.

        Returns:
            planes (np.ndarray): Array of 8 truncation planes [h,k,l,d].
            indicesOfTruncationPlanes (list): List of plane index pairs
                used for truncation.
        """
        # --- {110} planes (cutting corners) ---
        pRef_110 = np.array([1., 1., 0.]) / np.sqrt(2)
        scale = self.wire_width * self.Rnn / np.sqrt(2)
        d_110 = -(scale / np.sqrt(2)
                  - self.Marks_110 * self.interCompactPlanesDistance_110)
        # print("scale 110:", scale)
        # print("d_110:", d_110)
        # print("wire_width:", self.wire_width)

        planes_110 = []
        for i in range(4):
            angle = i * 90
            x = pyNMBu.RotationMol(pRef_110, angle, 'z')
            x = np.append(x, [d_110])
            planes_110.append(x)
            if debug:
                print(f"{{110}} angle={angle}°  plane={x}")

        # --- {100} planes (cutting faces) ---
        pRef_100 = np.array([1., 0., 0.])
        d_100 = -(scale - self.Marks_100 * self.interCompactPlanesDistance_100 / 2)
        # print("scale 100:", scale)
        # print("d_100:", d_100)
        # print("wire_width:", self.wire_width)
        # print("max x of wire atoms should be ~", scale)
        
        planes_100 = []
        for i in range(4):
            angle = i * 90
            x = pyNMBu.RotationMol(pRef_100, angle, 'z')
            x = np.append(x, [d_100])
            planes_100.append(x)
            if debug:
                print(f"{{100}} angle={angle}°  plane={x}")

        planes = np.array(planes_110 + planes_100)

        # Each corner is cut by its two neighboring {110} planes (indices 0-3)
        # Each face is cut by one {100} plane (indices 4-7)
        # Apply {110} pairs first, then {100} independently
        indicesOfTruncationPlanes_110 = [(i, (i+1) % 4) for i in range(4)]
        indicesOfTruncationPlanes_100 = [(i+4,) for i in range(4)]

        return planes, indicesOfTruncationPlanes_110, indicesOfTruncationPlanes_100
        
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
            pyNMBu.centertxt("Generation of coordinates",
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
        cVertices, E, F3, F4 = self.MakeVertices_SquarePyramid(self.nOrder)
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0, self.nAtoms))

        # Generate edge atoms
        nAtoms0 = self.nAtoms
        # Distance between two vertex atoms
        Rvv = pyNMBu.RAB(cVertices, E[0, 0], E[0, 1])
        nAtomsOnEdges = int((Rvv + 1e-6) / self.Rnn) - 1
        nIntervals = nAtomsOnEdges + 1
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

        # Generate facet atoms
        coordFaceAt = []
        nAtomsOnFaces = 0
        nAtoms0 = self.nAtoms
        for f in F3:
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
        nAtomsInCore, coordCoreAt = pyNMBu.MakeFaceCoord(
            self.Rnn, F4, c, nAtomsInCore, coordCoreAt)
        # don't ask... it is the algorithm to find the indexes
        # of the square corners of each layer along z

            
        for n in range(nAtomsOnEdges):
            f = np.array([
                self.edge_atom_index(n, 4, len(E)),  # apex-edge to vertex 0
                self.edge_atom_index(n, 7, len(E)),  # apex-edge to vertex 3
                self.edge_atom_index(n, 5, len(E)),  # apex-edge to vertex 1
                self.edge_atom_index(n, 6, len(E)),  # apex-edge to vertex 2
            ])
            nAtomsInCore, coordCoreAt = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsInCore, coordCoreAt)

        self.nAtoms += nAtomsInCore
        c.extend(coordCoreAt)
        indexCoreAtoms.extend(range(nAtoms0, self.nAtoms))

        c_arr = np.array(c)

        # shift by wire_length*self.interCompactPlanesDistance_001/2
        # then  creates the corners of the layers
        # and for each layer use MakeFaceCoord()

        def Fwire(zF4, size):
            
            coordAt = []
            nAtoms = 0

            scale = self.Rnn * size / np.sqrt(2)
            F4 = np.array([
                [-scale  ,        0, zF4],   # a
                [       0,    scale, zF4],   # b
                [ scale  ,        0, zF4],   # c
                [       0,   -scale, zF4],   # d
            ])

            coordAt.extend(F4.tolist())
            nAtoms += 4

            nIntervals  = size - 1
            for (i, j) in [(0,1), (1,2), (2,3), (3,0)]:
                for n in range(nIntervals):
                    t = (n + 1) / size
                    p = F4[i] + t * (F4[j] - F4[i])
                    coordAt.append(p.tolist())
            nAtoms += nIntervals * 4

            return coordAt, nAtoms, F4

        if self.wire_length != 0:
            c = np.array(c)  # shape (nAtoms, 3)
            d001 = self.interCompactPlanesDistance_001 / 2
            c[:,2] += self.wire_length*d001
            c = c.tolist()

            coordWire = []
            nAtomsWire = 0

            # Generate base inner and outer layers at z=0 (coordinates only)
            # z will be updated in the loop
            F4_coordAt_inner, F4_nAtoms_inner, F4_inner = Fwire(0, self.wire_width - 1)
            F4_coordAt_outer, F4_nAtoms_outer, F4_outer = Fwire(0, self.wire_width)

            # Determine starting layer type based on parity
            # of wire_width and nOrder (mandatory for arrows !!!)
            start_with_inner = (self.wire_width % 2 == self.nOrder % 2)

            for iLayer in range(self.wire_length):
                z_layer = (self.wire_length - (iLayer + 1)) * d001
                
                if start_with_inner:
                    is_inner = (iLayer % 2 == 0)
                else:
                    is_inner = (iLayer % 2 == 1)

                if is_inner:
                    coordAt = [[p[0], p[1], z_layer]
                               for p in F4_coordAt_inner]
                    F4_nAtoms = F4_nAtoms_inner
                else:
                    coordAt = [[p[0], p[1], z_layer]
                               for p in F4_coordAt_outer]
                    F4_nAtoms = F4_nAtoms_outer

                nAtomsWire0 = nAtomsWire
                coordWire.extend(coordAt)
                nAtomsWire += F4_nAtoms
                F4_index = list(range(nAtomsWire0, nAtomsWire0 + 4))
                nAtomsWire, coordWire = pyNMBu.MakeFaceCoord(
                    self.Rnn, F4_index, coordWire, nAtomsWire, coordWire)

            c.extend(coordWire)
            self.nAtoms += nAtomsWire
            
        # --- Marks truncation on full structure ---
        # --- Helper function to apply truncation planes ---
        # def apply_truncation(coord_list):
        #     pMarks, ic_110, ic_100 = self.truncationPlanes4MarksOctahedron(debug=True)
        #     print("ic_100:", ic_100)
        #     print("planes_100:")
        #     for ic in ic_100:
        #         print(f"  index {ic[0]}: {pMarks[ic[0]]}")

        #     # Apply {110} truncation (pairs of adjacent planes)
        #     for ic in ic_110:
        #         p0 = pMarks[ic[0]].copy()
        #         p1 = pMarks[ic[1]].copy()
        #         p1[0:3] = -p1[0:3]
        #         planes = [p0, p1]
        #         AtomsAbove = pyNMBu.truncateAbovePlanes(
        #             planes, coord_list, allP=True, delAbove=True,
        #             debug=False, noOutput=noOutput)
        #         coord_list = pyNMBu.deleteElementsOfAList(coord_list, AtomsAbove)

        #     # Apply {100} truncation (single planes)
        #     for ic in ic_100:
        #         p0 = pMarks[ic[0]].copy()
        #         planes = [p0]
        #         AtomsAbove = pyNMBu.truncateAbovePlanes(
        #             planes, coord_list, allP=True, delAbove=True,
        #             debug=False, noOutput=noOutput)
        #         coord_list = pyNMBu.deleteElementsOfAList(coord_list, AtomsAbove)

        #     return coord_list
        def apply_truncation(coord_list):
            pMarks, ic_110, ic_100 = self.truncationPlanes4MarksOctahedron()

            # Each corner is cut by one {110} plane AND one {100} plane
            for i in range(4):
                p_110_a = pMarks[ic_110[i][0]].copy()
                p_110_b = pMarks[ic_110[i][1]].copy()
                p_110_b[0:3] = -p_110_b[0:3]
                p_100 = pMarks[ic_100[i][0]].copy()
                if self.Marks_110 != 0 and self.Marks_100 != 0:
                    planes = [p_110_a, p_110_b, p_100]
                elif self.Marks_110 != 0 and self.Marks_100 == 0:
                    planes = [p_110_a, p_110_b]
                elif self.Marks_110 == 0 and self.Marks_100 != 0:
                    planes = [p_100]
                AtomsAbove = pyNMBu.truncateAbovePlanes(
                    planes, coord_list, allP=True, delAbove=True,
                    debug=False, noOutput=noOutput)
                coord_list = pyNMBu.deleteElementsOfAList(coord_list, AtomsAbove)

            return coord_list
        # def apply_truncation(coord_list):
        #     pMarks, ic_110, ic_100 = self.truncationPlanes4MarksOctahedron()
            
        #     # Corner 0: +x+y → {110}[0], {100}[0](+x), {100}[1](+y)
        #     # Corner 1: -x+y → {110}[1], {100}[1](+y), {100}[2](-x)
        #     # Corner 2: -x-y → {110}[2], {100}[2](-x), {100}[3](-y)
        #     # Corner 3: +x-y → {110}[3], {100}[3](-y), {100}[0](+x)
        #     corner_planes = [
        #         (0, 0, 1),  # {110}[0], {100}[0], {100}[1]
        #         (1, 1, 2),  # {110}[1], {100}[1], {100}[2]
        #         (2, 2, 3),  # {110}[2], {100}[2], {100}[3]
        #         (3, 3, 0),  # {110}[3], {100}[3], {100}[0]
        #     ]
        #     for (i110, i100a, i100b) in corner_planes:
        #         planes = [
        #             pMarks[ic_110[i110][0]].copy(),
        #             pMarks[ic_100[i100a][0]].copy(),
        #             pMarks[ic_100[i100b][0]].copy(),
        #         ]
        #         AtomsAbove = pyNMBu.truncateAbovePlanes(
        #             planes, coord_list, allP=True, delAbove=True,
        #             debug=False, noOutput=noOutput)
        #         coord_list = pyNMBu.deleteElementsOfAList(coord_list, AtomsAbove)
        #     return coord_list
        # --- Marks truncation on full structure ---
        if (self.Marks_110 != 0 or self.Marks_100 != 0) and not self.Marks_on_wire_only:
            c = apply_truncation(c)
            self.nAtoms = len(c)

        # --- Marks truncation on wire only ---
        elif (self.Marks_110 != 0 or self.Marks_100 != 0) and self.Marks_on_wire_only:
            z_min = 0.0
            z_max = self.wire_length * self.interLayerDistance
            c_wire   = [p for p in c if z_min <= p[2] < z_max]
            c_nowire = [p for p in c if not (z_min <= p[2] < z_max)]

            c_wire = apply_truncation(c_wire)

            c = c_nowire + c_wire
            self.nAtoms = len(c)
            
            c = c_nowire + c_wire
            self.nAtoms = len(c)

        # --- Mirror reflection ---
        symPlane = np.array([0, 0, 1, 0])
        ReflectionAtoms = pyNMBu.reflection(symPlane, c, True)
        c.extend(ReflectionAtoms)
        self.nAtoms += len(ReflectionAtoms)
        
        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")
        # Store results in an ASE Atoms object
        aseObject = ase.Atoms(self.element * self.nAtoms, positions=c)

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        self.NP = aseObject
        self.cog = self.NP.get_center_of_mass()

    def prop(self):
        """Display geometric properties of the nanoparticle."""
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print(f"element                            = {self.element}")
        if self.wire_length == 0:
            print(f"number of vertices                 = {self.nVertices_Oh}")
            print(f"number of edges                    = {self.nEdges_Oh}")
            print(f"number of faces ({self.nFaces3_Oh} {{111}})         = {self.nFaces3_Oh}")
        else:
            print(f"number of vertices                 = {self.nVertices_eOh}")
            print(f"number of edges                    = {self.nEdges_eOh}")
            print(f"number of faces ({self.nFaces3_eOh} {{111}} + {self.nFaces4_eOh} {{100}})  "
                  f"= {self.nFaces3_eOh + self.nFaces4_eOh}")
        print(f"nearest-neighbour distance         = {self.Rnn:.3f} Å")
        print(f"number of atoms per edge           = {self.nOrder + 1}")
        print(f"edge length                        = {self.edgeLength() * 0.1:.3f} nm")
        print(f"half-height of pyramidal cap       = {self.halfHeightPyramid() * 0.1:.3f} nm")
        print(f"d{{111}} / Rnn                       = {self.interCompactPlanesF_111:.4f}")
        print(f"d{{110}} / Rnn                       = {self.interCompactPlanesF_110:.4f}")
        print(f"d{{100}} / Rnn                       = {self.interCompactPlanesF_100:.4f}")
        print(f"d{{111}}                             = {self.interCompactPlanesDistance_111:.3f} Å")
        print(f"d{{110}}                             = {self.interCompactPlanesDistance_110:.3f} Å")
        print(f"d{{100}}                             = {self.interCompactPlanesDistance_100:.3f} Å")
        if self.wire_length > 0:
            print(f"elongation (wire_length = {self.wire_length} fcc cells)  "
                  f"= {self.elongationLength() * 0.1:.3f} nm")
        print(f"total height along [001]           = {self.totalHeight() * 0.1:.3f} nm")
        print(f"area                               = {self.area() * 1e-2:.2f} nm²")
        print(f"volume                             = {self.volume() * 1e-3:.2f} nm³")
        if self.Marks_110 != 0 or self.Marks_100 != 0:
            print(f"Marks_110 truncation               = {self.Marks_110} atom(s) on {{110}} corners")
            print(f"Marks_100 truncation               = {self.Marks_100} atom(s) on {{100}} faces")
            if self.Marks_on_wire_only:
                print(f"Marks truncation applied           = wire only")
            else:
                print(f"Marks truncation applied           = full structure")
        if self.wire_length == 0:
            print("Dual polyhedron: cube")
        print(f"coordinates of the center of gravity = {self.cog}")
