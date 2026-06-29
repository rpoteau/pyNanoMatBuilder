# External dependencies
import sys
import numpy as np
import time

from ase.visualize import view
import ase

# Internal Relative Imports
from . import visualID as vID
from . import data
from . import utils as pyNMBu
from . import platonicNPs as pNP
from . import johnsonNPs as jNP
from .utils import hl, fg, bg
from .pyNMBcore import pyNMBcore
from .utils.geometry import z_height_nm

###########################################################################################################
class fcctpt(pyNMBcore):
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
                 **kwargs
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
        super().__init__(**kwargs)
        self.element = element
        self.shape = 'fcctpt'
        self.Rnn = Rnn
        self.nLayerTd = int(nLayerTd)
        self.tbpprop = jNP.fcctbp(self.element, self.Rnn, self.nLayerTd, noOutput=True, calcPropOnly=True)
        self.nLayertbp = 2 * self.nLayerTd - 1
        self.nLayer = nLayer * 2 + 1  # total number of layers
        self.interLayerDistance = self.tbpprop.interLayerDistance
        self.nAtomsPerEdge = self.nLayerTd + 1
        self.dim = [0, 0, 0]
        if self.nLayer > self.nLayertbp:
            sys.exit(f"Number of layers of the triangular platelet ({self.nLayer}) cannot be > to the total number of layers of the trigonal bipyramid {self.nLayertbp}")
        self.imageFile = pyNMBu.imageNameWithPathway("tpt-C.png")
        noOutput = self.noOutput
        if not noOutput:
            pyNMBu.centerTitle(
                f"fcc triangular platelet with {nLayer*2+1} remaining shells, made from a trigonal bipyramid with {nLayerTd} shells per pyramid"
            )

        if not noOutput: self.prop()
        if not self.calcPropOnly:
            self.coords(noOutput)
            if self.aseView:
                view(self.NP)
            if self.postAnalyzis:
                self.propPostMake(self.skipChiralityCalculation, self.skipSymmetryAnalyzis,
                                  self.skipFacetInfo, self.thresholdCoreSurface, noOutput)
                if self.aseView:
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
            pyNMBu.centertxt("Generation of coordinates", bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()

        
        # Create the trigonal bipyramid nanoparticle
        if not noOutput:
            pyNMBu.centertxt(
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
        if not noOutput: pyNMBu.centertxt("Truncation of the trigonal bipyramid",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
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
        pyNMBu.centertxt("Properties", bgc='#007a7a', size='14', weight='bold')
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

    # def propPostMake(self, self.skipChiralityCalculation, skipSymmetryAnalyzis, self.thresholdCoreSurface, noOutput):
    #     """
    #     Compute and store post-construction nanoparticle properties.

    #     This function calculates moments of inertia (MOI), the inscribed and
    #     circumscribed spheres, determines the nanoparticle shape, analyzes
    #     symmetry (if required), and identifies core and surface atoms.

    #     Args:
    #         skipSymmetryAnalyzis (bool): If True, skips symmetry analysis.
    #         thresholdCoreSurface (float): Threshold to distinguish core and surface atoms.
    #         noOutput (bool): If True, suppresses output messages.

    #     Attributes:
    #         moi (numpy.ndarray): Moment of inertia tensor.
    #         moisize (numpy.ndarray): Normalized moments of inertia.
    #         vertices (numpy.ndarray): Geometric vertices of the nanoparticle.
    #         simplices (numpy.ndarray): Simplices defining the convex hull.
    #         neighbors (numpy.ndarray): Neighboring relations between facets.
    #         equations (numpy.ndarray): Plane equations for the hull faces.
    #         NPcs (ase.Atoms): Copy of the nanoparticle with surface atoms visually marked.
    #         NP (ase.Atoms): Original nanoparticle object.
    #     """
    #     self.moi = pyNMBu.moi(self.NP, noOutput=noOutput)
    #     self.moisize = np.array(pyNMBu.moi_size(self.NP, noOutput))  # MOI mass normalized (m of each atoms=1)

    #     if not skipSymmetryAnalyzis:
    #         pyNMBu.MolSym(self.NP, noOutput=noOutput)

    #     [
    #         self.vertices,
    #         self.simplices,
    #         self.neighbors,
    #         self.equations,
    #     ], surfaceAtoms = pyNMBu.coreSurface(self, self.thresholdCoreSurface, noOutput=noOutput)
    #     self.NPcs = self.NP.copy()
    #     self.NPcs.numbers[np.invert(surfaceAtoms)] = 102  # Nobelium (visual color in jmol)
    #     self.surfaceatoms = self.NPcs[surfaceAtoms]

    #     pyNMBu.Inscribed_circumscribed_spheres(self, noOutput)

    #     if self.jmolCrystalShape:
    #         self.jMolCS = pyNMBu.defCrystalShapeForJMol(self, noOutput)

###########################################################################################################
class pentaPrism(pyNMBcore):
    """
    Bare pentagonal prism: the {100}-faceted shaft of a pentatwinned
    decahedron, generated directly (no caps are built or removed).

    Adapted from ASE's Decahedron construction: five sheared FCC segments
    are stacked around the 5-fold axis (closing the 7.356° disclination
    deficit), but the conical cap term (h - n) is replaced by a constant
    axial height, yielding a straight pentagonal cylinder with vertical
    {100} side facets and flat terminations. Intended as a starting body to
    be faceted into bipyramid models; the caps then emerge from the chosen
    truncation rather than being pre-built.

    The user specifies physical dimensions (diameter, height) and the
    nearest-neighbour distance Rnn; these are mapped to the integer ring and
    layer counts of the underlying construction by picking the counts whose
    realised dimensions best match the request.

    Args:
        element (str): chemical element (e.g. 'Au').
        Rnn (float): nearest-neighbour distance in Å.
        diameter (float): target circumscribed diameter of the pentagonal
            section, in nm.
        height (float): target height along the 5-fold axis, in nm.
    """

    nFaces4 = 5    # {100} side facets
    nFaces5 = 2    # flat pentagonal terminations
    nVertices = 10
    nEdges = 15

    def __init__(self,
                 element: str = 'Au',
                 Rnn: float = 2.937,
                 diameter: float = 3.0,
                 height: float = 5.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.element = element
        self.shape = 'pentaPrism'
        self.Rnn = Rnn
        self.diameter = diameter
        self.height = height

        # Underlying lattice spacings (ASE convention)
        self._b = Rnn                       # axial spacing (a/sqrt2)
        self._a = Rnn * np.sqrt(3) / 2.0    # radial spacing between rings

        # Map physical dimensions -> integer counts (p, nz)
        self.p, self.n_pairs = self._solve_counts(diameter, height)

        self.imageFile = pyNMBu.imageNameWithPathway("underConstruction-C.png")  # placeholder
        noOutput = self.noOutput
        if not noOutput:
            pyNMBu.centerTitle(
                f"Bare pentagonal prism — target Ø{diameter:.2f} nm × {height:.2f} nm"
                f"  (p={self.p}, n_pairs={self.n_pairs})")
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
        return (f"Bare pentagonal prism of {self.element}, "
                f"Ø~{self.diameter:.2f} nm × {self.height:.2f} nm, Rnn = {self.Rnn} Å")

    def _build(self, p, n_pairs):
        """Build the symmetric bare pentagonal prism (vectorised).

        FCC ABAB stacking along the axis comes from the parity of the ring
        index n: even-n rings sit on "full" layers at z = k*b, odd-n rings on
        "interleaved" layers at z = k*b + b/2 between them. The full layers
        span k in [-n_pairs, +n_pairs], the interleaved ones only
        [-n_pairs, +n_pairs-1], so both ends terminate on the same full layer
        (symmetric, no narrow-vs-wide cap).

        Each ring's in-plane (x, y) sites are built by interpolating along the
        five pentagon edges; the cartesian product of those sites with the
        ring's z-levels is assembled with NumPy (no Python-level per-atom
        loop), which is ~300× faster than appending atom by atom for large
        prisms.

        Args:
            p (int): number of pentagonal rings (section size).
            n_pairs (int): number of full layers on each side of the
                mid-plane. Total height ~ 2*n_pairs*b.

        Returns an (N, 3) array centred on the origin."""
        
        b, a = self._b, self._a
        t = 2.0 * np.pi / 5.0
        verts = a * np.array(
            [[np.cos(np.pi/2 + t*k), np.sin(np.pi/2 + t*k), 0.]
             for k in range(5)])

        full_z = np.arange(-n_pairs, n_pairs + 1) * b
        int_z = np.arange(-n_pairs, n_pairs) * b + b / 2.0

        # (xy_sites, z_levels) for each ring; central ring n=0 sits on the axis
        ring_data = [(np.zeros((1, 2)), full_z)]
        for n in range(1, p):
            zs = full_z if n % 2 == 0 else int_z
            edges = []
            for m in range(5):
                v1, v2 = verts[m-1, :2], verts[m, :2]
                i = np.arange(n)[:, None]
                edges.append((n - i) * v1 + i * v2)   # (n, 2)
            ring_data.append((np.vstack(edges), zs))   # (5n, 2)

        # cartesian product of xy sites with z levels, per ring
        blocks = []
        for xy, zs in ring_data:
            nxy, nz = len(xy), len(zs)
            block = np.empty((nxy * nz, 3))
            block[:, 0] = np.repeat(xy[:, 0], nz)
            block[:, 1] = np.repeat(xy[:, 1], nz)
            block[:, 2] = np.tile(zs, nxy)
            blocks.append(block)

        c = np.vstack(blocks)
        return c - c.mean(axis=0)
    
    def _solve_counts(self, diameter_nm, height_nm):
        """Pick integer (p, n_pairs) whose realised dimensions best match
        the request."""
        b, a = self._b, self._a
        n_pairs = max(1, int(round(height_nm * 10 / (2 * b))))
        target_r = diameter_nm * 10 / 2.0
        best_p, best_err = 2, np.inf
        for p in range(2, int(target_r / a) + 4):
            test = self._build(p, 1)
            r = np.hypot(test[:, 0], test[:, 1]).max()
            if abs(r - target_r) < best_err:
                best_err, best_p = abs(r - target_r), p
        return best_p, n_pairs

    def coords(self, noOutput):
        """Generate the prism coordinates."""
        if not noOutput:
            pyNMBu.centertxt("Generation of coordinates",
                             bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()

        c = self._build(self.p, self.n_pairs)
        c -= c.mean(axis=0)
        self.NP = ase.Atoms(symbols=[self.element] * len(c), positions=c)
        self.nAtoms = len(c)
        self.cog = self.NP.get_center_of_mass()

        if not noOutput:
            r = np.hypot(c[:, 0], c[:, 1]).max()
            h = c[:, 2].max() - c[:, 2].min()
            print(f"Realised diameter = {2*r/10:.3f} nm, "
                  f"height = {h/10:.3f} nm")
            print(f"Total number of atoms = {self.nAtoms}")
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()

    def prop(self):
        """Display nanoparticle properties."""
        print()
        print(self)
        print("element =", self.element)
        print(f"nearest neighbour distance = {self.Rnn:.3f} Å")
        print(f"target diameter = {self.diameter:.3f} nm")
        print(f"target height   = {self.height:.3f} nm")
        print(f"ring count p = {self.p}, full layers per half-height = {self.n_pairs}")
        print("5 vertical {100} side facets, flat pentagonal terminations")

###########################################################################################################
class ptnr(pyNMBcore):
    """
    ptnr, aka Pentatwinned nano-rods (bipyramid, walled bipyramid, double cone,
    rod, ellipsoid, capsule) built from a bare pentagonal prism
    (pentaPrism) and shaped by CSG operations.

    The morphology is chosen with the `shape` argument. For each shape the
    class sizes the required starting prism from the target dimensions, builds
    it internally (composition — a pentaPrism instance), applies the relevant
    rotation / slicing / clipping, and exposes the result as its own self.NP,
    following the usual pyNMBcore contract.

    Morphology-specific parameters default to None (= "not specified"); each
    shape applies its own default for the ones it uses, warns about the ones
    it ignores, and raises if a required one is missing.

    Args:
        element (str): chemical element (e.g. 'Au').
        Rnn (float): nearest-neighbour distance in Å.
        shape (str): one of 'bipyramid', 'walled_bipyramid', 'double_cone',
            'rod', 'ellipsoid', 'capsule'.
        diameter (float): circumscribed waist/section diameter, in nm.
        angle_deg (float): facet tilt from the equatorial plane, in degrees
            ('bipyramid', 'walled_bipyramid'). Default 15.793°.
        h_wall (float): vertical-wall height in nm ('walled_bipyramid',
            required).
        height (float): total length in nm ('double_cone', 'rod',
            'ellipsoid'). If None, derived from the shape.
        cut_nm (float): tip truncation depth in nm ('bipyramid',
            'walled_bipyramid'); 0 = sharp tips.
        bevel (bool): for shape='rod', bevel the 5 vertical edges
            (pentagon → decagon). Default False.
        round_tips (bool): for shape='rod', round both ends with
            hemispherical caps. Default False.
        tip_sphere_radius_nm (float): rounded-tip cap radius ('double_cone').
        edge_factor (float): beveling distance as a fraction of the
            circumscribed radius ('rod' with bevel=True). Default 0.9.

    Example:
        bipy = oNP.ptnr(element="Au", Rnn=2.937,
                        shape="bipyramid", diameter=10.0,
                        oOutput=False)
    """

    _VALID_SHAPES = ('bipyramid', 'walled_bipyramid', 'double_cone',
                     'rod', 'ellipsoid', 'capsule')

    # default value each shape uses when a parameter is left unspecified
    _DEFAULTS = {
        'angle_deg':            15.793169048263962,
        'h_wall':               10.0,
        'cut_nm':               0.0,
        'tip_sphere_radius_nm': 2.0,
        'edge_factor':          0.9,
        'bevel':                False,
        'round_tips':           False,
        # 'height' is derived per-shape when None, handled in the builders
    }

    # parameters each shape actually uses
    _USED = {
        'bipyramid':        {'diameter', 'angle_deg', 'cut_nm'},
        'walled_bipyramid': {'diameter', 'angle_deg', 'h_wall', 'cut_nm'},
        'double_cone':      {'diameter', 'height', 'tip_sphere_radius_nm'},
        'rod':              {'diameter', 'height', 'bevel', 'round_tips', 'edge_factor'},
        'ellipsoid':        {'diameter', 'height'},
        'capsule':          {'diameter', 'height'},
    }
    # parameters strictly required (no sensible default) per shape
    _REQUIRED = {
        'walled_bipyramid': {'h_wall'},
    }

    # image file per shape (shown in prop())
    _IMAGES = {
        'bipyramid':        'pbipy-C.png',
        'walled_bipyramid': 'pbipyW-C.png',
        'double_cone':      'dcone-C.png',
        'ellipsoid':        'ellipsoidPT-C.png',
        'rod':              'uptnr-C.png',
        'capsule':          'capsule-C.png',
    }

    def __init__(self,
                 element: str = 'Au',
                 Rnn: float = 2.937,
                 shape: str = 'bipyramid',
                 diameter: float = 10.0,
                 angle_deg: float = None,
                 h_wall: float = None,
                 height: float = None,
                 cut_nm: float = None,
                 tip_sphere_radius_nm: float = None,
                 edge_factor: float = None,
                 bevel: bool = None,
                 round_tips: bool = None,
                 **kwargs):
        super().__init__(**kwargs)
        if shape not in self._VALID_SHAPES:
            raise ValueError(f"Unknown shape '{shape}'. "
                             f"Choose from {self._VALID_SHAPES}.")
        self.element = element
        self.shape = shape
        self.Rnn = Rnn
        self.diameter = diameter
        # store raw (None = not specified by the user)
        self.angle_deg = angle_deg
        self.h_wall = h_wall
        self.height = height
        self.cut_nm = cut_nm
        self.tip_sphere_radius_nm = tip_sphere_radius_nm
        self.edge_factor = edge_factor
        self.bevel = bevel
        self.round_tips = round_tips

        # pedagogical parameter check (uses the None sentinels)
        self._check_params()

        self.imageFile = pyNMBu.imageNameWithPathway(
            self._IMAGES.get(self.shape, "underConstruction-C.png"))
        noOutput = self.noOutput
        if not noOutput:
            pyNMBu.centerTitle(
                f"Pentatwinned object — shape='{shape}', "
                f"Ø{diameter:.2f} nm, {element}")
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
        return (f"Pentatwinned {self.shape} of {self.element}, "
                f"Ø~{self.diameter:.2f} nm, Rnn = {self.Rnn} Å")

    # ----- parameter validation --------------------------------------------
    def _check_params(self):
        """Pedagogical validation: warn about parameters the chosen shape
        ignores, and raise if a required one is missing. A parameter is
        considered 'specified' when it is not None."""
        all_morpho = {'angle_deg', 'h_wall', 'height', 'cut_nm',
                      'tip_sphere_radius_nm', 'edge_factor', 'bevel', 'round_tips'}
        specified = {p for p in all_morpho if getattr(self, p) is not None}
        used = self._USED[self.shape]

        # 1. Warn about specified-but-irrelevant parameters
        for p in sorted(specified - used):
            shapes = ', '.join(s for s in self._VALID_SHAPES
                               if p in self._USED[s])
            print(f"{bg.DARKREDB}Note: parameter '{p}' is not used by "
                  f"shape='{self.shape}' and will be ignored "
                  f"(it applies to: {shapes}).{bg.OFF}")

        # 2. Raise if a required parameter is missing
        missing = self._REQUIRED.get(self.shape, set()) - specified
        if missing:
            raise ValueError(
                f"shape='{self.shape}' requires {sorted(missing)} "
                f"to be specified explicitly.")

    def _get(self, param):
        """Return the user value if specified, else the shape's default."""
        val = getattr(self, param)
        return val if val is not None else self._DEFAULTS.get(param)

    # ----- geometry helpers specific to pentatwinned objects ---------------
    @staticmethod
    def height_for_sharp_tip(diameter_nm, angle_deg):
        """
        Starting-prism height for a sharp-tipped bipyramid of given waist
        diameter: L = diameter / tan(angle). The aspect ratio L/diameter =
        1/tan(angle) is fixed by the facet angle alone (≈3.54 for 15.793°).
        """
        return diameter_nm / np.tan(np.radians(angle_deg))

    @staticmethod
    def bipyramid_height(diameter_nm, angle_deg):
        """
        Real tip-to-tip height of the faceted bipyramid:
        H = diameter · cos(36°) / tan(angle).

        Differs from height_for_sharp_tip by cos(36°): after the 18° rotation
        the inclined facets meet the {100} FACES (at the apothem
        diameter/2·cos36°), not the vertices, so they converge lower. This is
        the height measured on the final object.
        """
        return diameter_nm * np.cos(np.pi / 5) / np.tan(np.radians(angle_deg))

    @staticmethod
    def waist_distance_for_tilted_facet(prism, angle_deg, start_vec=(1, 0, 0),
                                        target_diameter_nm=None,
                                        z_tol=0.5, rad_tol=0.3):
        """
        Orthogonal distance, along the tilted facet normal, of the cutting
        plane that passes through a real equatorial atom row of `prism`.
        Anchoring on an actual row (not a theoretical value) keeps the {115}
        steps regular. Returns the distance in nm for
        applySlicing(distance_unit='nm').
        """
        pos = prism.NP.get_positions() - prism.NP.get_center_of_mass()
        eq = pos[np.abs(pos[:, 2]) < z_tol]
        if len(eq) == 0:
            raise ValueError("No equatorial atoms found; increase z_tol.")
        u = np.array(start_vec, float)
        u[2] = 0.0
        u /= np.linalg.norm(u)
        proj = eq @ u
        perp = np.linalg.norm(eq[:, :2] - np.outer(proj, u[:2]), axis=1)
        on_axis = eq[(perp < rad_tol) & (proj > 0)]
        if len(on_axis) == 0:
            raise ValueError("No equatorial atom along start_vec; raise "
                             "rad_tol or check the rotation applied.")
        rows = np.unique(np.round(on_axis @ u, 3))
        if target_diameter_nm is None:
            R = rows.max()
        else:
            R = rows[np.argmin(np.abs(rows - target_diameter_nm * 10 / 2))]
        return (R * np.cos(np.radians(angle_deg))) / 10.0

    # ----- coordinate generation (dispatch) --------------------------------
    def coords(self, noOutput):
        """Build the starting prism for the chosen shape, shape it, and store
        the result as self.NP."""
        if not noOutput:
            pyNMBu.centertxt("Generation of coordinates",
                             bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()

        builder = {
            'bipyramid':        self._build_bipyramid,
            'walled_bipyramid': self._build_walled_bipyramid,
            'double_cone':      self._build_double_cone,
            'ellipsoid':        self._build_ellipsoid,
            'rod':              self._build_rod,
            'capsule':          self._build_capsule,
        }[self.shape]
        prism = builder(noOutput)

        # adopt the shaped prism's atoms as our own
        self.NP = prism.NP
        self.nAtoms = len(self.NP)
        self.cog = self.NP.get_center_of_mass()

        if not noOutput:
            print(f"Realised height = {z_height_nm(self):.3f} nm")
            print(f"Total number of atoms = {self.nAtoms}")
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()

    # ----- per-shape internal builders -------------------------------------
    def _new_prism(self, diameter, height, noOutput):
        """Build a bare pentaPrism with our material, no post-analysis
        (we analyse the final shaped object, not the intermediate prism)."""
        return pentaPrism(element=self.element, Rnn=self.Rnn,
                          diameter=diameter, height=height,
                          skipSymmetryAnalyzis=True,
                          postAnalyzis=False, noOutput=noOutput)

    def _build_bipyramid(self, noOutput):
        angle = self._get('angle_deg')
        cut = self._get('cut_nm')
        H = self.height_for_sharp_tip(self.diameter, angle)
        prism = self._new_prism(self.diameter, H, noOutput)
        prism.apply_rotation(angle_deg=18, axis=[0, 0, 1], axis_def='cart',
                             noOutput=noOutput, postAnalyzis=False)
        distance = self.waist_distance_for_tilted_facet(
            prism, angle_deg=angle, start_vec=[1, 0, 0],
            target_diameter_nm=None)
        planes = [
            {'angle':  angle, 'startVec': [1, 0, 0], 'nRot': 5,
             'rotAxis': [0, 0, 1], 'distance': distance,
             'delete': 'above', 'modeP': 'OR'},
            {'angle': -angle, 'startVec': [1, 0, 0], 'nRot': 5,
             'rotAxis': [0, 0, 1], 'distance': distance,
             'delete': 'above', 'modeP': 'OR'},
        ]
        if cut > 0:
            H_bipy = self.bipyramid_height(self.diameter, angle)
            planes += [
                {'normal': [0, 0,  1], 'nRot': 1,
                 'distance': H_bipy/2 - cut, 'delete': 'above'},
                {'normal': [0, 0, -1], 'nRot': 1,
                 'distance': H_bipy/2 - cut, 'delete': 'above'},
            ]
        prism.applySlicing(planes=planes, mode='OR', distance_unit='nm',
                           recenter=False, noOutput=noOutput)
        prism.apply_rotation(angle_deg=-18, axis=[0, 0, 1], axis_def='cart',
                     noOutput=noOutput, postAnalyzis=False)
        return prism

    def _build_walled_bipyramid(self, noOutput):
        angle = self._get('angle_deg')
        h_wall = self._get('h_wall')
        cut = self._get('cut_nm')
        a = np.radians(angle)
        R = self.diameter / 2
        apothem = R * np.cos(np.pi / 5)
        H_tot = h_wall + 2 * R / np.tan(a)
        d_incl = apothem * np.cos(a) + (h_wall / 2) * np.sin(a)
        prism = self._new_prism(self.diameter, H_tot, noOutput)
        prism.apply_rotation(angle_deg=18, axis=[0, 0, 1], axis_def='cart',
                             noOutput=noOutput, postAnalyzis=False)
        prism.applySlicing(
            planes=[
                {'angle':  angle, 'startVec': [1, 0, 0], 'nRot': 5,
                 'rotAxis': [0, 0, 1], 'distance': d_incl,
                 'delete': 'above', 'modeP': 'OR'},
                {'angle': -angle, 'startVec': [1, 0, 0], 'nRot': 5,
                 'rotAxis': [0, 0, 1], 'distance': d_incl,
                 'delete': 'above', 'modeP': 'OR'},
            ],
            mode='OR', distance_unit='nm', recenter=False, noOutput=noOutput)
        if cut > 0:
            H_real = z_height_nm(prism)
            prism.applySlicing(
                planes=[
                    {'normal': [0, 0,  1], 'nRot': 1,
                     'distance': H_real/2 - cut, 'delete': 'above'},
                    {'normal': [0, 0, -1], 'nRot': 1,
                     'distance': H_real/2 - cut, 'delete': 'above'},
                ],
                mode='OR', distance_unit='nm', recenter=False, noOutput=noOutput)
        
        prism.apply_rotation(angle_deg=-18, axis=[0, 0, 1], axis_def='cart',
                             noOutput=noOutput, postAnalyzis=False)
        return prism

    def _build_double_cone(self, noOutput):
        from scipy.optimize import brentq
        tip_r = self._get('tip_sphere_radius_nm')
        H = self.height if self.height is not None \
            else self.height_for_sharp_tip(self.diameter, self._get('angle_deg'))

        margin_nm = 2.0                      # oversize the prism (no edge artefacts)
        R = self.diameter / 2                # base radius of the cone (nm)
        Hhalf = H / 2.0

        # Solve for the virtual apex height L so that the SPHERE POLE (the real
        # tip after rounding) sits at Hhalf. With half-angle a = atan(R/L):
        #   t_pole = L - tip_r/sin(a) + tip_r,  sin(a) = R/sqrt(R²+L²)
        if tip_r > 0:
            if tip_r >= R:
                raise ValueError(f"tip_sphere_radius_nm ({tip_r}) must be "
                                 f"smaller than the base radius ({R} nm).")
            def pole_minus_target(L):
                sin_a = R / np.sqrt(R**2 + L**2)
                return (L - tip_r / sin_a + tip_r) - Hhalf
            L = brentq(pole_minus_target, 1e-3, 10 * Hhalf)
        else:
            L = Hhalf                        # sharp tip: apex is the pole

        prism = self._new_prism(self.diameter + 2 * margin_nm,
                                H + 2 * margin_nm, noOutput)
        # prism.clip_to_cone(
        #     base_center_nm=[0, 0, 0], base_radius_nm=R,
        #     apex_nm=[0, 0, L], tip_sphere_radius_nm=tip_r,
        #     keep='inside', recenter=False, noOutput=noOutput, postAnalyzis=False)
        # prism.align_to_plane(axis=[0, 0, 1], target=0.0,
        #                      noOutput=noOutput, postAnalyzis=False)
        # prism.replicate_by_reflection([0, 0, 1, 0], plane_def='cart',
        #                               noOutput=noOutput, postAnalyzis=False)
        # Build the double cone with two successive clips on the SAME object.
        # Each clip shapes its own half-space (keep_opposite_side=True leaves the
        # other half untouched), so no copy, no union, no waist deduplication —
        # and the lattice stays exactly that of _new_prism (aligned with the
        # bipyramid, no stacking fault at the waist).
        # Top cone: apex up; cleans z >= 0, leaves z < 0 intact.
        prism.clip_to_cone(
            base_center_nm=[0, 0, 0], base_radius_nm=R,
            apex_nm=[0, 0, L], tip_sphere_radius_nm=tip_r,
            keep='inside', keep_opposite_side=True,
            recenter=False, noOutput=noOutput, postAnalyzis=False)
        # Bottom cone: apex down; its apex side is now z <= 0, cleans it, leaves
        # the already-built upper half-cone (z > 0) intact.
        prism.clip_to_cone(
            base_center_nm=[0, 0, 0], base_radius_nm=R,
            apex_nm=[0, 0, -L], tip_sphere_radius_nm=tip_r,
            keep='inside', keep_opposite_side=True,
            recenter=False, noOutput=noOutput, postAnalyzis=False)
        return prism

    def _build_rod(self, noOutput):
        bevel = self._get('bevel')
        round_tips = self._get('round_tips')
        edge_factor = self._get('edge_factor')
        H = self.height if self.height is not None else 3 * self.diameter

        # if beveling, oversize the starting prism so the beveled section
        # reaches the target diameter: prism_diameter = D_target / edge_factor
        if bevel:
            cos36 = np.cos(np.pi / 5)   # 0.809
            if edge_factor < cos36:
                print(f"{bg.DARKREDB}Warning: edge_factor ({edge_factor:.3f}) "
                      f"is below cos(36°) = {cos36:.3f}. The bevels will bite "
                      f"into the {{100}} faces instead of only the vertices, "
                      f"and the final diameter will not reach the target. "
                      f"Use edge_factor >= {cos36:.3f}.{bg.OFF}")
            # oversize the starting prism so the beveled section reaches the
            # target diameter: prism_diameter = D_target / edge_factor
            prism_diam = self.diameter / edge_factor
        else:
            prism_diam = self.diameter
        prism = self._new_prism(prism_diam, H, noOutput)

        if bevel:
            # bevel the vertices down to the target radius
            edge_distance = self.diameter / 2
            prism.applySlicing(
                planes=[
                    {'angle': 0, 'startAngle': 18, 'refVec': [1, 0, 0],
                     'rotAxis': [0, 0, 1], 'nRot': 5, 'distance': edge_distance,
                     'delete': 'above', 'modeP': 'OR'},
                ],
                mode='OR', distance_unit='nm', recenter=False, noOutput=noOutput)

        if round_tips:
            prism.propPostMake(
                skipChiralityCalculation=True, skipSymmetryAnalyzis=True,
                skipFacetInfo=True, thresholdCoreSurface=3.0,
                noOutput=True, is_optimized=False)
            key = 'initial structure'
            cap_diam = (prism.ellipsoid[key]['D2']
                        + prism.ellipsoid[key]['D3']) / 20.0
            prism.round_tip_in_direction(direction=[0, 0, 1], diameter_nm=cap_diam,
                                         axis_def='cart', noOutput=noOutput,
                                         postAnalyzis=False)
            prism.round_tip_in_direction(direction=[0, 0, -1], diameter_nm=cap_diam,
                                         axis_def='cart', noOutput=noOutput,
                                         postAnalyzis=False)
        return prism

    def _build_ellipsoid(self, noOutput):
        margin_nm = 2.0   # prism slightly oversized so the ellipsoid cuts
                          # well inside the material (no surface-grazing artefacts)
        H = self.height if self.height is not None else 3 * self.diameter
        prism = self._new_prism(self.diameter + 2 * margin_nm,
                                H + 2 * margin_nm, noOutput)
        prism.clip_to_ellipsoid(
            diameters_nm=[self.diameter, self.diameter, H],
            noOutput=noOutput, postAnalyzis=False)
        return prism

    def _build_capsule(self, noOutput):
        H = self.height if self.height is not None else 3 * self.diameter
        if H < self.diameter:
            raise ValueError(f"capsule height ({H} nm) must be >= diameter "
                             f"({self.diameter} nm): the hemispherical caps "
                             f"would otherwise overlap.")
        margin_nm = 2.0
        # prism oversized radially and axially, then clipped to a clean cylinder
        prism = self._new_prism(self.diameter + 2 * margin_nm, H, noOutput)
        prism.clip_to_cylinder(diameter_nm=self.diameter, axis=[0, 0, 1],
                               noOutput=noOutput, postAnalyzis=False)
        # round both ends with hemispherical caps of the same radius (tangent,
        # smooth join) -> capsule of total height H
        prism.round_tip_in_direction(direction=[0, 0, 1], diameter_nm=self.diameter,
                                     axis_def='cart', noOutput=noOutput, postAnalyzis=False)
        prism.round_tip_in_direction(direction=[0, 0, -1], diameter_nm=self.diameter,
                                     axis_def='cart', noOutput=noOutput, postAnalyzis=False)
        return prism

    # ----- properties ------------------------------------------------------
    def prop(self):
        """Display nanoparticle properties."""
        print()
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile)
        print("element =", self.element)
        print(f"shape = {self.shape}")
        print(f"nearest neighbour distance = {self.Rnn:.3f} Å")
        print(f"target diameter = {self.diameter:.3f} nm")
        if self.shape in ('bipyramid', 'walled_bipyramid'):
            print(f"facet angle = {self._get('angle_deg'):.3f}°")
        if self.shape == 'walled_bipyramid':
            print(f"wall height = {self._get('h_wall'):.3f} nm")
        if self.shape in ('bipyramid', 'walled_bipyramid') and self._get('cut_nm') > 0:
            print(f"tip truncation = {self._get('cut_nm'):.3f} nm")
        if self.shape == 'rod':
            print(f"bevel = {self._get('bevel')}, round_tips = {self._get('round_tips')}")
        if self.shape in ('double_cone',) and self._get('tip_sphere_radius_nm'):
            print(f"tip sphere radius = {self._get('tip_sphere_radius_nm'):.3f} nm")

###########################################################################################################
class hbpy(pyNMBcore):
    """
    hbpy — hexagonal bipyramid / cone (single crystal).

    A single-crystal hexagonal bipyramid (or single cone) bounded by six
    {10-1l} pyramidal facets, built from a parallelepiped Crystal block faceted
    via applySlicing. Composition, not inheritance: an internal Crystal block is
    built in 'pppd' shape, sculpted by the six tilted facets (plus optional base
    truncation), and its atoms are adopted as self.NP, following the usual
    pyNMBcore contract (same scheme as ptnr with pentaPrism).

    Unlike fcc penta-twinned objects, a hexagonal pyramid is monocrystalline:
    the hcp lattice already carries the six-fold symmetry around [0001], so the
    six {10-1l} facets close without any twin plane.

    The base-to-apex height is DERIVED from the facet indices and the section
    diameter: the six facets are anchored on the equatorial atom row of the
    block (so the {10-1l} steps stay regular), and the apex follows from the
    facet/base dihedral angle, computed from the reciprocal metric tensor. The
    user may additionally specify cut_nm, the apex-to-truncation distance, to
    turn the pyramid into a frustum (truncated pyramid).

    For double=True the full bipyramid is built in place by two facet groups
    (apex up and apex down) in a single applySlicing call: the atoms are never
    reflected or translated, so the lattice stays registered with the source
    block and the waist is a single clean shared layer (no stacking fault).

    Args:
        crystal (str): compound name resolved by crystalNPs.Crystal
            (e.g. 'Co hcp'). Any non-cubic lattice is supported.
        facet (list of int): pyramidal facet indices (3-index [h, k, l],
            e.g. [1, 0, 1] for {10-11}, [1, 0, 3] for {10-13}).
            Default [1, 0, 1].
        diameter (float): circumscribed (vertex-to-vertex) diameter of the
            hexagonal section, in nm. Default 8.0.
        double (bool): True builds the full bipyramid (apex up and down),
            False a single cone (apex up, flat base). Default True.
        cut_nm (float): apex-to-truncation distance in nm; 0 keeps a sharp
            tip, > 0 truncates each apex into a flat {0001} facet (frustum).
            Default 0.0.
        normal_def (str): coordinate system for the facet indices, 'hkl' or
            'cart'. When None (default), 'hkl' is used for non-cubic lattices
            and 'cart' for cubic ones.
        axis_on_atom (bool): placement of the six-fold rotation axis relative
            to the hcp lattice. hcp has no atomic site on its own 6-fold axis:
            the axis passes through the centre of an atom hexagon, a/sqrt(3)
            from the nearest column. When False (default) the facets rotate
            about that interstitial axis and the apex closes on a centred
            triangle of three atoms. When True the block is shifted in-plane so
            a real atomic column sits on the axis, and the six facets converge
            on a single apex atom; the base is then centred on the column
            rather than on the section centroid. Default False.
        symmetric_facets (bool): how the six {10-1l} facets are cut. The 6_3
            screw axis of hcp splits them into two families that, with a single
            cut distance, terminate at different atomic levels (one looks less
            shaved than the other). When False (default) this true-crystal
            asymmetry is preserved. When True the two families are cut with
            distances offset by the lattice registry shift, so all six facets
            are shaved to the same level (visually symmetric pyramid).
            Default False.
        shave_family_A (bool): with symmetric_facets=True, selects which screw
            family is shaved by one atomic row. False (default) shaves class B
            (azimuth 90/210/330), keeping class A the larger; True shaves
            class A (azimuth 30/150/270), keeping class B the larger. Ignored
            when symmetric_facets=False. Default False.
        regularize_base (bool): with symmetric_facets=True, force a regular
            base hexagon by calibrating each screw family separately so both
            reach the same circumscribed radius. The default one-row shave
            (shave_family_A) only equalises the two families when their natural
            offset is about one atomic row, which holds for shallow facets such
            as {10-11} but not for steeper ones such as {10-13}, where the base
            stays visibly irregular. With regularize_base=True both families are
            bisection-calibrated to the target radius, giving six equal-length
            base edges at any facet index l. shave_family_A is then unused.
            Ignored when symmetric_facets=False. Default False.
    Example:
        from pyNanoMatBuilder import otherNPs as oNP
        bp = oNP.hbpy(crystal='Co hcp', facet=[1, 0, 3],
                      diameter=8.0, double=True, cut_nm=0.0,
                      thresholdCoreSurface=0.5, skipSymmetryAnalyzis=True,
                      noOutput=False)
    """
    def __init__(self,
                 crystal: str = 'Co hcp',
                 facet=None,
                 diameter: float = 8.0,
                 double: bool = True,
                 cut_nm: float = 0.0,
                 normal_def: str = None,
                 axis_on_atom: bool = False,
                 symmetric_facets: bool = False,
                 shave_family_A: bool = False,
                 regularize_base: bool = False,
                 **kwargs):

        super().__init__(**kwargs)
        self.crystalName = crystal
        self.shape = 'hbpy'
        self.facet = list(facet) if facet is not None else [1, 0, 1]
        self.diameter = diameter
        self.double = double
        self.cut_nm = cut_nm
        self.normal_def_user = normal_def
        self.axis_on_atom = axis_on_atom
        self.symmetric_facets = symmetric_facets
        self.shave_family_A = shave_family_A
        self.regularize_base = regularize_base

        self.imageFile = pyNMBu.imageNameWithPathway("underConstruction-C.png")
        noOutput = self.noOutput
        if not noOutput:
            pyNMBu.centerTitle(
                f"Hexagonal {'bipyramid' if double else 'cone'} — "
                f"{crystal}, facet {{{self.facet[0]}{self.facet[1]}"
                f"{-(self.facet[0]+self.facet[1])}{self.facet[2]}}}, "
                f"Ø{diameter:.2f} nm")

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
        if not noOutput:
            self.prop()

    def __str__(self):
        kind = 'bipyramid' if self.double else 'cone'
        return (f"Hexagonal {kind} of {self.crystalName}, "
                f"Ø~{self.diameter:.2f} nm, facet {self.facet}")

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def _auto_normal_def(crystal_obj, user_value):
        """Resolve the coordinate system for the facet indices: honour the
        user value if given, else 'hkl' for non-cubic lattices and 'cart'
        for cubic ones (where a plane (hkl) and its normal coincide).
        Falls back to 'hkl' (the always-correct choice) if the lattice system
        cannot be read."""
        if user_value is not None:
            return user_value
        bravais = getattr(crystal_obj, 'ucBL', None)
        system = getattr(bravais, 'lattice_system', None)
        return 'cart' if system == 'cubic' else 'hkl'

    def _facet_dihedral_deg(self, block):
        """Dihedral angle between the {10-1l} facet and the basal (0001)
        plane, via crystallographic_angle (reciprocal metric tensor)."""
        return block.crystallographic_angle(
            [0, 0, 1], self.facet,
            type1='plane', type2='plane', noOutput=True)

    @staticmethod
    def _rot_z(vec, deg):
        """Rotate a 3-vector by `deg` around +z."""
        t = np.radians(deg)
        ct, st = np.cos(t), np.sin(t)
        R = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        return R @ np.asarray(vec, dtype=float)

    def _facet_normal_cart(self, block, facet, ndef):
        """Reference facet normal (Cartesian) for the given facet indices,
        built the same way applySlicing's 'normal' branch does."""
        if ndef == 'hkl':
            from .utils.crystals import plane_to_direction
            n = plane_to_direction(block, facet, cartesian=True)
        else:
            n = np.array(facet, dtype=float)
        return n / np.linalg.norm(n)

    # ----- axis alignment --------------------------------------------------
    def _align_axis_on_column(self, block, noOutput=True):
        """Translate the block in-plane so a real atomic column sits on the
        rotation axis (0, 0).

        hcp has no lattice site on its own 6-fold axis: the axis passes through
        the centre of an atom hexagon, a/sqrt(3) from the nearest column. By
        default the facet planes rotate about that interstitial axis, so the
        apex closes on a triangle of three atoms. Shifting the block so a column
        lands on (0, 0) makes the six facets converge on a single apex atom.

        After this shift the axis is (0, 0) for everything that follows (facet
        distance calibration and applySlicing), so all radii are measured from
        the axis, not from the centre of mass."""
        pos = block.NP.get_positions()
        com = block.NP.get_center_of_mass()
        c_ang = block.cif.cell.lengths()[2]
        eq = pos[np.abs(pos[:, 2] - com[2]) < c_ang / 2.0]
        if len(eq) == 0:
            raise ValueError("No equatorial atoms found to locate a column.")
        # nearest column to the current axis (= COM in-plane)
        d = np.hypot(eq[:, 0] - com[0], eq[:, 1] - com[1])
        nearest = eq[np.argmin(d)]
        shift_x, shift_y = nearest[0], nearest[1]
        block.NP.positions[:, 0] -= shift_x
        block.NP.positions[:, 1] -= shift_y
        if not noOutput:
            print(f"axis aligned on atomic column: block shifted by "
                  f"({-shift_x:+.4f}, {-shift_y:+.4f}) A "
                  f"(column was at {np.hypot(shift_x, shift_y):.4f} A from axis)")


    # ----- facet distance calibration --------------------------------------
    def _calibrate_distance(self, block, normals, eq, eq_r, R_target):
        """Bisect the orthogonal facet distance so the circumscribed radius
        realised by the given set of facet `normals` equals R_target (A).
        Returns the distance in A. Used both for the full nRot=6 group and,
        in symmetric mode, for each screw family (class A, class B) separately
        so both reach the same realised radius despite the square block's
        azimuthal anisotropy."""
        def realised_radius(d_ang):
            keep = np.ones(len(eq), dtype=bool)
            for nn in normals:
                keep &= (eq @ nn <= d_ang)
            kept_r = eq_r[keep]
            return kept_r.max() if len(kept_r) else 0.0

        nh = np.hypot(normals[0][0], normals[0][1])
        d_lo, d_hi = 0.1, nh * R_target * 2.0
        for _ in range(40):
            d_mid = 0.5 * (d_lo + d_hi)
            if realised_radius(d_mid) < R_target:
                d_lo = d_mid
            else:
                d_hi = d_mid
        return 0.5 * (d_lo + d_hi)

    def _equatorial_slab(self, block):
        """Return (eq, eq_r): equatorial-slab atom positions and their radial
        distance from the rotation axis (0, 0)."""
        pos = block.NP.get_positions()
        z0 = 0.0 if self.axis_on_atom else block.NP.get_center_of_mass()[2]
        c_ang = block.cif.cell.lengths()[2]
        eq = pos[np.abs(pos[:, 2] - z0) < c_ang / 2.0]
        if len(eq) == 0:
            raise ValueError("No equatorial atoms found to calibrate the facet "
                             "distance; check the block height.")
        eq_r = np.hypot(eq[:, 0], eq[:, 1])
        return eq, eq_r

    def _facet_distance_nm(self, block, dihedral_deg, ndef, noOutput=True):
        """Orthogonal distance (nm) for the six {10-1l} facets (nRot=6 group),
        calibrated by bisection so the realised circumscribed diameter matches
        self.diameter. Geometry-agnostic analogue of
        ptnr.waist_distance_for_tilted_facet. See _calibrate_distance."""
        n0 = self._facet_normal_cart(block, self.facet, ndef)
        normals = [self._rot_z(n0, k * 60.0) for k in range(6)]
        eq, eq_r = self._equatorial_slab(block)
        R_target = self.diameter / 2.0 * 10.0
        d_ang = self._calibrate_distance(block, normals, eq, eq_r, R_target)
        if not noOutput:
            print(f"facet distance calibrated: d = {d_ang/10:.3f} nm")
        return d_ang / 10.0

    def _facet_row_spacing_nm(self, block, ndef, facet):
        """Inter-row spacing (nm) along the facet normal: the distance between
        consecutive {10-1l} atomic planes. Used to shave one screw family by
        exactly one atomic row relative to the other."""
        n = self._facet_normal_cart(block, facet, ndef)
        pos = block.NP.get_positions()
        pos = pos - pos.mean(axis=0)
        proj = np.unique(np.round(pos @ n, 3))
        return float(np.median(np.diff(proj))) / 10.0

    def _facet_groups(self, block, ndef, d_facet, sign=+1, noOutput=True):
        """Build the facet plane group(s) for one apex direction.

        sign = +1 builds the upward facets, sign = -1 the downward facets
        (l index negated).

        symmetric_facets=False (default): a single nRot=6 group at d_facet,
        preserving the true hcp asymmetry (the two screw families terminate at
        different atomic levels).

        symmetric_facets=True: the six facets are split into the two screw
        families (class A at the reference azimuths, class B pre-rotated by 60
        deg and passed as Cartesian, since applySlicing's 'normal' branch
        ignores startAngle). One family keeps the calibrated distance d_facet;
        the OTHER is pulled inward by one inter-row spacing so it is shaved by
        a single atomic row, giving the regular alternation. shave_family_A
        selects which family is the shaved one:
          - False (default): class B is shaved (class A stays the larger one),
          - True: class A is shaved (class B stays the larger one).
        """
        facet = [self.facet[0], self.facet[1], sign * self.facet[2]]

        if not self.symmetric_facets:
            return [
                {'normal': facet, 'normal_def': ndef, 'distance': d_facet,
                 'nRot': 6, 'rotAxis': [0, 0, 1],
                 'delete': 'above', 'modeP': 'OR'},
            ]

        nA = self._facet_normal_cart(block, facet, ndef)
        nB = self._rot_z(nA, 60.0)

        if getattr(self, 'regularize_base', False):
            # Per-family calibration: bring BOTH screw families to the same
            # realised radius, for a regular base hexagon at any facet index l.
            # Use this for steep facets (e.g. {10-13}) where a one-row shave
            # cannot equalise the two families. shave_family_A is not used here.
            eq, eq_r = self._equatorial_slab(block)
            R_target = self.diameter / 2.0 * 10.0
            normals_A = [self._rot_z(nA, k * 120.0) for k in range(3)]
            normals_B = [self._rot_z(nB, k * 120.0) for k in range(3)]
            dA = self._calibrate_distance(block, normals_A, eq, eq_r, R_target) / 10.0
            dB = self._calibrate_distance(block, normals_B, eq, eq_r, R_target) / 10.0
            if not noOutput:
                print(f"symmetric facets (per-family calibration): "
                      f"dA = {dA:.4f} nm, dB = {dB:.4f} nm "
                      f"(both reaching R = {R_target/10:.3f} nm)")
        else:
            # Original behaviour: shave one family by exactly one atomic row.
            # Correct and important for {10-11}; shave_family_A selects which.
            row = self._facet_row_spacing_nm(block, ndef, facet)
            if self.shave_family_A:
                dA, dB = d_facet - row, d_facet
            else:
                dA, dB = d_facet, d_facet - row
            if not noOutput:
                shaved = 'A' if self.shave_family_A else 'B'
                print(f"symmetric facets: dA = {dA:.4f} nm, dB = {dB:.4f} nm "
                      f"(class {shaved} shaved by one row = {row:.4f} nm)")

        if not noOutput:
            shaved = 'A' if self.shave_family_A else 'B'
            print(f"symmetric facets: dA = {dA:.4f} nm, dB = {dB:.4f} nm "
                  f"(class {shaved} shaved by one row = {row:.4f} nm)")

        return [
            {'normal': nA.tolist(), 'normal_def': 'cart', 'distance': dA,
             'nRot': 3, 'rotAxis': [0, 0, 1],
             'delete': 'above', 'modeP': 'OR'},
            {'normal': nB.tolist(), 'normal_def': 'cart', 'distance': dB,
             'nRot': 3, 'rotAxis': [0, 0, 1],
             'delete': 'above', 'modeP': 'OR'},
        ]
        
    # ----- coordinate generation -------------------------------------------
    def coords(self, noOutput):
        """Build the parallelepiped block, facet it into the hexagonal
        (bi)pyramid, and store the result as self.NP.

        The block is sized exactly from the derived apex height: it is centred
        on z = 0 and the apex sits at +Hhalf (and -Hhalf for the bipyramid),
        so the block half-height must reach Hhalf in both cases. A small fixed
        safety of one lattice spacing (c in z, a in section), read from the
        unit cell, keeps the apex atom row off the block boundary.

        When axis_on_atom is True, the block is shifted in-plane so a real
        atomic column lands on the rotation axis (0, 0) before faceting, so the
        six facets converge on a single apex atom instead of the interstitial
        triangle. The facet distance is then calibrated against that same axis,
        keeping axis alignment and diameter calibration consistent."""
        if not noOutput:
            pyNMBu.centertxt("Generation of coordinates",
                             bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()

        from . import crystalNPs as cyNP

        Rcs = self.diameter / 2.0

        # 1) Tiny probe block: read the lattice and the facet/base dihedral.
        probe = cyNP.Crystal(
            self.crystalName, shape='pppd', size=[1.0, 1.0, 1.0],
            skipSymmetryAnalyzis=True, postAnalyzis=False,
            jmolCrystalShape=False, aseView=False, noOutput=True)
        ndef = self._auto_normal_def(probe, self.normal_def_user)
        dihedral = self._facet_dihedral_deg(probe)

        # lattice spacings (nm) for the auto safety margin
        a_nm = probe.cif.cell.lengths()[0] / 10.0
        c_nm = probe.cif.cell.lengths()[2] / 10.0

        # 2) Derived apex height and EXACT block sizing.
        Hhalf = Rcs * np.tan(np.radians(dihedral))
        block_h = 2.0 * Hhalf + 2.0 * c_nm
        side = self.diameter + 2.0 * a_nm

        if not noOutput:
            pyNMBu.centertxt("Building the parallelepiped block",
                             bgc='#cbcbcb', size='12', fgc='b', weight='bold')
            print(f"facet/base dihedral angle = {dihedral:.2f} deg")
            print(f"derived apex height Hhalf = {Hhalf:.2f} nm")
            print(f"block size = {side:.2f} x {side:.2f} x {block_h:.2f} nm "
                  f"(auto, safety = 1 c-spacing = {c_nm:.3f} nm)")
            print(f"normal_def resolved to '{ndef}' "
                  f"(lattice system: {probe.ucBL.lattice_system})")

        block = cyNP.Crystal(
            self.crystalName, shape='pppd', size=[side, side, block_h],
            skipSymmetryAnalyzis=True, postAnalyzis=False,
            jmolCrystalShape=False, aseView=False, noOutput=noOutput)

        # 3) Optionally align the rotation axis on a real atomic column FIRST.
        if self.axis_on_atom:
            self._align_axis_on_column(block, noOutput=noOutput)

        # 4) Calibrate the facet distance against the rotation axis (0, 0).
        d_facet = self._facet_distance_nm(block, dihedral, ndef,
                                          noOutput=noOutput)

        # 5) Facet plane groups (symmetric or true-crystal, cone or bipyramid)
        planes = self._facet_groups(block, ndef, d_facet, sign=+1, noOutput=noOutput)
        if self.double:
            planes += self._facet_groups(block, ndef, d_facet, sign=-1, noOutput=noOutput)
        else:
            planes.append(
                {'normal': [0, 0, -1], 'normal_def': 'cart',
                 'distance': 0.0, 'nRot': 1, 'rotAxis': [0, 0, 1],
                 'delete': 'above', 'modeP': 'OR'})

        # 6) Apply the facet groups (and base) first, WITHOUT truncation, so the
        #    realised apex height can be measured. cut_nm must be referenced on
        #    the REAL apex, not the theoretical Hhalf: the discrete lattice has
        #    no atom exactly at the ideal tip, so the realised apex sits a
        #    fraction of c below Hhalf, and cutting from Hhalf would remove too
        #    little.
        block.applySlicing(planes=planes, mode='OR', distance_unit='nm',
                           recenter=False, noOutput=noOutput,
                           postAnalyzis=False)

        # 7) Optional apex truncation (frustum), referenced on the realised apex.
        if self.cut_nm and self.cut_nm > 0:
            z = block.NP.get_positions()[:, 2]
            cut_planes = []
            # top apex: realised top is z.max(); cut cut_nm below it
            z_apex_top = z.max()
            z_cut_top = (z_apex_top - self.cut_nm * 10.0) / 10.0   # nm
            if z_cut_top * 10.0 <= z.min() + 1e-6:
                raise ValueError(
                    f"cut_nm ({self.cut_nm} nm) exceeds the realised apex "
                    f"height; nothing would remain.")
            cut_planes.append(
                {'normal': [0, 0, 1], 'normal_def': 'cart',
                 'distance': z_cut_top, 'nRot': 1, 'rotAxis': [0, 0, 1],
                 'delete': 'above', 'modeP': 'OR'})
            if self.double:
                # bottom apex: realised bottom is z.min(); cut cut_nm above it
                z_apex_bot = z.min()
                z_cut_bot = (z_apex_bot + self.cut_nm * 10.0) / 10.0  # nm
                cut_planes.append(
                    {'normal': [0, 0, -1], 'normal_def': 'cart',
                     'distance': -z_cut_bot, 'nRot': 1, 'rotAxis': [0, 0, 1],
                     'delete': 'above', 'modeP': 'OR'})
            block.applySlicing(planes=cut_planes, mode='OR', distance_unit='nm',
                               recenter=False, noOutput=noOutput,
                               postAnalyzis=False)

        # 8) Adopt the shaped block and expose its slicing artefacts.
        self.NP = block.NP
        self.NP.positions -= self.NP.get_center_of_mass()
        self.nAtoms = len(self.NP)
        self.cog = self.NP.get_center_of_mass()
        self.facet_dihedral_deg = dihedral
        self.jMolSlices = getattr(block, 'jMolSlices', None)
        self.trPlanes_Slices = getattr(block, 'trPlanes_Slices', None)
        self._block = block

        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()

    # ----- properties ------------------------------------------------------
    def prop(self):
        """Display nanoparticle properties."""
        print()
        print(self)
        print("element / compound =", self.crystalName)
        print(f"shape = hexagonal {'bipyramid' if self.double else 'cone'}")
        h, k, l = self.facet[0], self.facet[1], self.facet[2]
        print(f"pyramidal facet = {{{h} {k} {-(h+k)} {l}}}")
        print(f"target circumscribed section diameter = {self.diameter:.3f} nm")
        if self.NP is not None and len(self.NP):
            pos = self.NP.get_positions()
            pos = pos - pos.mean(axis=0)
            # realised height along z
            height_nm = (pos[:, 2].max() - pos[:, 2].min()) / 10.0
            # realised circumscribed diameter: twice the largest radial distance
            # from the axis, wherever it occurs (the base for a cone, the waist
            # for a bipyramid). Taking the global max avoids picking a slab at
            # the wrong height (the vertical midpoint of a cone is already
            # narrow and would halve the value).
            r = np.hypot(pos[:, 0], pos[:, 1])
            diam_real_nm = 2.0 * r.max() / 10.0
            print(f"realised circumscribed section diameter = {diam_real_nm:.3f} nm")
            print(f"realised height (z) = {height_nm:.3f} nm")
            # truncation facet diameter, if the apex was cut into a {0001} flat
            if self.cut_nm and self.cut_nm > 0:
                z = pos[:, 2]
                c_ang = (self._block.cif.cell.lengths()[2]
                         if hasattr(self, '_block') else 4.0)
                top = pos[np.abs(z - z.max()) < c_ang]
                r_top = np.hypot(top[:, 0], top[:, 1])
                diam_top_nm = 2.0 * r_top.max() / 10.0
                print(f"realised top truncation diameter = {diam_top_nm:.3f} nm")
        if hasattr(self, 'facet_dihedral_deg'):
            print(f"facet/base dihedral angle = {self.facet_dihedral_deg:.2f}°")
        if self.cut_nm and self.cut_nm > 0:
            print(f"apex truncation (height) = {self.cut_nm:.3f} nm")