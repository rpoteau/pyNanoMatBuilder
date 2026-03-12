# External dependencies
import sys
import numpy as np

import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view
import matplotlib.pyplot as plt

# Internal Relative Imports
from .visualID import fg, hl, bg
from . import visualID as vID
from . import data
from . import utils as pyNMBu
from . import platonicNPs as pNP

###########################################################
class JohnsonNP:
    """Base class for all Johson nanoparticles providing common functionality."""

    def propPostMake(self, skipSymmetryAnalyzis, thresholdCoreSurface, noOutput):
        """
        Compute and store various post-construction properties of the nanoparticle.

        This function calculates moments of inertia (MOI), the inscribed and 
        circumscribed sphere diameters,analyzes symmetry, generates a JMOL
        script, and identifies core and surface atoms.

        Args:
            skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
            thresholdCoreSurface (float): Threshold to distinguish core and surface atoms.
            noOutput (bool): If True, suppresses output messages.
    
        Attributes Updated:
            moi (numpy.ndarray): Moment of inertia tensor.
            moisize (numpy.ndarray): Normalized moments of inertia.
            vertices (numpy.ndarray): Geometric vertices of the nanoparticle.
            simplices (numpy.ndarray): Simplices defining the convex hull.
            neighbors (numpy.ndarray): Neighboring relations between facets.
            equations (numpy.ndarray): Plane equations for the hull faces.
            NPcs (ase.Atoms): Copy of the nanoparticle with surface atoms visually marked.
            NP (ase.Atoms): Original nanoparticle object.
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

    def __init__(self, element: str = 'Au', Rnn: float = 2.7,
                 nLayerTd: int = 1, postAnalyzis: bool = True,
                 aseView: bool = False,
                 thresholdCoreSurface: float = 1.,
                 skipSymmetryAnalyzis: bool = False,
                 jmolCrystalShape: bool = True,
                 noOutput: bool = False,
                 calcPropOnly: bool = False):

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
        
        self.element = element
        self.shape = 'fcctbp'
        self.Rnn = Rnn
        self.nLayerTd = int(nLayerTd)
        self.Tdprop = pNP.regfccTd(
            self.element, self.Rnn, self.nLayerTd,
            noOutput=True, calcPropOnly=True)
        self.nLayer = 2 * self.nLayerTd - 1
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.interLayerDistance = self.Tdprop.interLayerDistance()
        self.nAtomsPerEdge = self.nLayerTd + 1
        self.cog = np.array([0., 0., 0.])
        self.fveAngle = self.Tdprop.fveAngle
        self.fefAngle = self.Tdprop.fefAngle
        self.vcvAngle = self.Tdprop.vcvAngle
        self.heightOfBiPyramid = (
            2 * self.Tdprop.heightOfPyramid()
            + 2 * (self.Rnn * np.sqrt(2 / 3)))
        self.imageFile = pyNMBu.imageNameWithPathway("tbp-C.png")
        self.jmolCrystalShape = jmolCrystalShape

        if not noOutput:
            vID.centerTitle(
                f"fcc trigonal bipyramid with"
                f" {nLayerTd} shells per pyramid")

        if not noOutput:
            self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(
                    skipSymmetryAnalyzis,
                    thresholdCoreSurface, noOutput)
                if aseView:
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
            vID.centertxt(
                "Generation of coordinates",
                bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        if not noOutput:
            vID.centertxt(
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
            vID.centertxt(
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
    interCompactPlanesF = (1 + np.sqrt(5)) / 4
    magicFactorF = (
        2 * heightOfPyramidF / edgeLengthF
    )

    def __init__(
            self,
            element: str = 'Au',
            Rnn: float = 2.7,
            sizeP: int = 1,
            sizeE: int = 0,
            Marks: int = 0,
            Multiples_index_plan: bool = False,
            Hollow: bool = False,
            postAnalyzis: bool = True,
            aseView: bool = False,
            thresholdCoreSurface=1.,
            skipSymmetryAnalyzis: bool = False,
            jmolCrystalShape: bool = True,
            noOutput: bool = False,
            calcPropOnly: bool = False,
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
        self.element = element
        self.shape = 'epbpyM'
        self.Rnn = Rnn
        self.sizeP = sizeP
        self.sizeE = sizeE
        self.Marks = Marks
        # For the creation of (11n) bipyramids
        self.Multiples_index_plan = Multiples_index_plan
        self.Hollow = Hollow
        self.nAtoms = 0
        self.nAtomsPerPentagonalCap = 0
        self.nAtomsPerElongatedPart = 0
        self.nAtomsPerEdgeOfPC = self.sizeP + 1
        self.nAtomsPerEdgeOfEP = self.sizeE + 1
        self.jmolCrystalShape = jmolCrystalShape
        self.cog = np.array([0., 0., 0.])
        self.interCompactPlanesDistance = self.interCompactPlanesF * self.Rnn
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
            vID.centerTitle(
                f"Pentagonal bipyramid with {sizeP}"
                f" atoms/edge, a x{sizeE} elongation"
                f" (Ino) and a x{Marks} edge"
                f" truncation (Marks)")

        if not noOutput:
            self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(
                    skipSymmetryAnalyzis,
                    thresholdCoreSurface,
                    noOutput)
                if aseView:
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
            vID.centertxt(
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

        # Generate vertices
        nAtoms0 = 0
        self.nAtoms = 6
        cVertices, E, F3 = self.MakeVertices()
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0, self.nAtoms))

        # Generate atoms on the edges
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
        coordFace3At = []
        nAtomsOnFaces3 = 0
        nAtoms0 = self.nAtoms
        for f in F3:
            nAtomsOnFaces3, coordFace3At = pyNMBu.MakeFaceCoord(
                self.Rnn, f, c, nAtomsOnFaces3, coordFace3At)
        self.nAtoms += nAtomsOnFaces3
        c.extend(coordFace3At)
        indexFace3Atoms.extend(range(nAtoms0, self.nAtoms))

   
        # Apply truncation if the Marks decahedron is being used
        # or if (11n) planes are wanted for the pyramids
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

        # Reflection of the upper pyramid w.r.t. the (0,0,1) plane
        symPlane = np.array([0, 0, 1, 0])
        ReflectionAtoms = pyNMBu.reflection(symPlane, c, True)
        c.extend(ReflectionAtoms)
        self.nAtoms += len(ReflectionAtoms)

        # Generate internal atoms (core atoms)
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

    