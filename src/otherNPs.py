import visualID as vID
from visualID import fg, hl, bg

import sys
import numpy as np
import pyNanoMatBuilder.utils as pyNMBu

from ase.visualize import view


from pyNanoMatBuilder import platonicNPs as pNP
from pyNanoMatBuilder import johnsonNPs as jNP

###########################################################################################################
class fcctpt:
    """
    This class generates and manages the properties of an fcc triangular platelet, which is based on 
    a trigonal bipyramid structure. It provides methods to calculate and manipulate atomic coordinates, 
    compute the properties of the nanoparticle, and generate visualizations.
    """
    nFaces = 8
    nEdges = 9
    nVertices = 9
    edgeLengthF = 2

    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayerTd: int=1, 
                 nLayer: int = 3, 
                 postAnalyzis: bool=True,
                 aseView: bool=False,
                 thresholdCoreSurface: float=1.,
                 skipSymmetryAnalyzis: bool=False,
                 jmolCrystalShape: bool=True,
                 noOutput: bool=False,
                 calcPropOnly: bool=False,
                ):
             
        """
        
        Initializes the fcc triangular platelet nanoparticle.

        Args:
            element (str): The element of the nanoparticle (default 'Au').
            Rnn (float): The nearest neighbor distance (default 2.7 Å).
            nLayerTd (int): Number of layers (bonds per edge) of the two
                tetrahedrons used to create the platelet. This value will
                define the edge length of the final platelet; the number of
                layers of the platelet will be ``nLayerTd + 1``.
            nLayer (int): The number of layers kept for each tetrahedron
                (not counting the twin plane). This value will define the
                thickness of the platelet; the number of layers in the final
                platelet will be ``nLayer * 2``.
            postAnalyzis (bool): If True, prints additional NP information.
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for
                core/surface differentiation.
            skipSymmetryAnalyzis (bool): If False, performs an atomic
                structure analysis using pymatgen.
            jmolCrystalShape (bool): If True, generates a JMOL script for
                visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If True, enables visualization of the NP
                using ASE.

        Attributes:
            nFaces (int): The number of faces of the triangular platelet.
            nEdges (int): The number of edges of the triangular platelet.
            nVertices (int): The number of vertices of the triangular platelet.
            edgeLengthF (float): A factor for edge length.
            element (str): The chemical element for the nanoparticle.
            shape (str): The shape of the nanoparticle ('fcctpt').
            Rnn (float): The nearest neighbor distance in Angstroms.
            nLayerTd (int): Number of layers (bonds per edge) of the two
                tetrahedrons used to create the platelet.
            nLayer (int): The number of layers kept for each tetrahedron.
            nAtoms (int): Number of atoms in the nanoparticle.
            interLayerDistance (float): Distance between layers.
            nAtomsPerEdge (int): Number of atoms per edge at the twin boundary.
            cog (np.array): Center of gravity of the nanoparticle.
            dim (list): Dimensions of the nanoparticle.
            jmolCrystalShape (bool): Flag to indicate whether to display the
                crystal shape in Jmol.
            imageFile (str): Path to the image file of the nanoparticle shape.
        """
        self.element = element
        self.shape = 'fcctpt'
        self.Rnn = Rnn
        self.nLayerTd = int(nLayerTd)
        self.tbpprop = jNP.fcctbp(self.element, self.Rnn, self.nLayerTd, noOutput=True, calcPropOnly=True)
        self.nLayertbp = 2 * self.nLayerTd - 1
        self.nLayer = nLayer * 2 + 1  # total number of layers
        self.nAtoms = 0
        self.interLayerDistance = self.tbpprop.interLayerDistance
        self.nAtomsPerEdge = self.nLayerTd + 1
        self.cog = np.array([0.0, 0.0, 0.0])
        self.dim = [0, 0, 0]
        self.jmolCrystalShape = jmolCrystalShape
        if self.nLayer > self.nLayertbp:
            sys.exit(f"Number of layers of the triangular platelet ({self.nLayer}) cannot be > to the total number of layers of the trigonal bipyramid {self.nLayertbp}")
        self.imageFile = pyNMBu.imageNameWithPathway("tpt-C.png")
        if not noOutput:
            vID.centerTitle(
                f"fcc triangular platelet with {nLayer*2+1} remaining shells, made from a trigonal bipyramid with {nLayerTd} shells per pyramid"
            )

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView:
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput)
                if aseView:
                    view(self.NPcs)

    def __str__(self):
        return (f"Truncated fcc double tetrahedron with {self.nLayer} layer(s) and Rnn = {self.Rnn}")

    def edgeLength(self):
        """
        Computes the edge length of the triangular platelet in Å..
        """
        return self.tbpprop.edgeLength()

    def coords(self,noOutput):
        """
        Generates the coordinates of the triangular platelet nanoparticle.

        This method creates the atomic positions by truncating a trigonal bipyramid
        and adjusting the coordinates based on the twin and truncation planes.

        Args:
            noOutput (bool): Whether to suppress output messages.
        """
        if not noOutput:
            vID.centertxt("Generation of coordinates", bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()

        
        # Create the trigonal bipyramid nanoparticle
        if not noOutput:
            vID.centertxt(
                "Generation of the coordinates of the trigonal bipyramid, based on the fcc tetrahedron",
                bgc='#cbcbcb',
                size='12',
                fgc='b',
                weight='bold',
            )
        tbp = jNP.fcctbp(self.element, self.Rnn, self.nLayerTd + 1, postAnalyzis=False, noOutput=True)
        self.NP0 = tbp.NP.copy()
        asetpt = tbp.NP
        nAtoms = asetpt.get_global_number_of_atoms()
        # cog = pyNMBu.centerOfGravity(asetbp.get_positions())
        # print("cog = ",cog)

        # Truncate the trigonal bipyramid based on twin plane and truncation planes
        if not noOutput: vID.centertxt("Truncation of the trigonal bipyramid",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput:
            print("Now calculating the coordinates of the twin plane (defined by atoms 0, 1, 2)")
        coordTwPVertices = asetpt.get_positions()[[0, 1, 2]]
        twinningPlane = pyNMBu.hklPlaneFitting(coordTwPVertices,printEq=not noOutput)
        twinningPlane = pyNMBu.normalizePlane(twinningPlane)
        
        if not noOutput:
            print("Now calculating the coordinates of the truncation planes")
        truncationPlane1 = twinningPlane.copy()
        truncationPlane1[3] = -(self.nLayer-1)*self.interLayerDistance/2
        if not noOutput:
            print(
                f"signed distance between truncation plane 1 and origin = {pyNMBu.Pt2planeSignedDistance(truncationPlane1, [0, 0, 0]):.2f}"
            )
        truncationPlane2 = -twinningPlane.copy()
        truncationPlane2[3] = -(self.nLayer - 1) * self.interLayerDistance / 2
        if not noOutput:
            print(
                f"signed distance between truncation plane 2 and origin = {pyNMBu.Pt2planeSignedDistance(truncationPlane1, [0, 0, 0]):.2f}"
            )
        
        trPlanes = [truncationPlane1, truncationPlane2]
        AtomsAbovePlanes = pyNMBu.truncateAboveEachPlane(
            trPlanes, asetpt.get_positions(), noOutput=noOutput
        )
        del asetpt[AtomsAbovePlanes]
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        nAtoms = asetpt.get_global_number_of_atoms()
        self.nAtoms = nAtoms
        if not noOutput:
            print(f"Total number of atoms = {nAtoms}")
        self.NP = asetpt
        self.cog = self.NP.get_center_of_mass()

    def prop(self):
        """
        Display unit cell and nanoparticle properties.
        """
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length of the tetrahedrons used to create the double truncated tetrahedron = {self.edgeLength()*0.1:.2f} nm")
        print(f"longest edge length of the platelet (at twin boundary) = {(self.edgeLength()+self.Rnn)*0.1:.2f} nm")
        print(f"number of atoms per edge at the twin boundary = {self.nAtomsPerEdge + 1}")
        print(f"inter-layer distance = {self.interLayerDistance:.2f} Å")
        print(f"height of the platelet = {self.interLayerDistance*(self.nLayer-1)*0.1:.2f} nm")
        # print(f"area = {6*self.Td.area()/4*1e-2:.1f} nm2")
        # print(f"volume = {2*self.Td.volume()*1e-3:.1f} nm3")
        print(f"face-vertex-edge angle in Td = {self.tbpprop.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle in Td = {self.tbpprop.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond) angle in Td = {self.tbpprop.vcvAngle:.1f}°")
        print(f"coordinates of the center of gravity = {self.cog}")

    def propPostMake(self, skipSymmetryAnalyzis, thresholdCoreSurface, noOutput):
        """Compute and store post-construction nanoparticle properties.

        This function calculates moments of inertia (MOI), the inscribed and
        circumscribed spheres, determines the nanoparticle shape, analyzes
        symmetry (if required), and identifies core and surface atoms.

        Args:
            skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
            thresholdCoreSurface (float): Threshold to distinguish core and
                surface atoms.
            noOutput (bool): If True, suppresses output messages.

        Attributes Updated:
            self.moi (array): Moment of inertia tensor.
            self.moisize (array): Normalized moments of inertia.
            self.vertices, self.simplices, self.neighbors, self.equations
                (arrays): Geometric properties of the nanoparticle.
            self.NPcs (Atoms): Copy of the nanoparticle with surface atoms
                visually marked.
            self.NP (Atoms): Original nanoparticle.
        """
        self.moi = pyNMBu.moi(self.NP, noOutput=noOutput)
        self.moisize = np.array(pyNMBu.moi_size(self.NP, noOutput))  # MOI mass normalized (m of each atoms=1)

        if not skipSymmetryAnalyzis:
            pyNMBu.MolSym(self.NP, noOutput=noOutput)

        [
            self.vertices,
            self.simplices,
            self.neighbors,
            self.equations,
        ], surfaceAtoms = pyNMBu.coreSurface(self, thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102  # Nobelium (visual color in jmol)
        self.surfaceatoms = self.NPcs[surfaceAtoms]

        pyNMBu.Inscribed_circumscribed_spheres(self, noOutput)

        if self.jmolCrystalShape:
            self.jMolCS = pyNMBu.defCrystalShapeForJMol(self, noOutput)