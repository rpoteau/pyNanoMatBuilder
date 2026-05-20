import numpy as np

class pyNMBcore:
    def __init__(self,
                 postAnalyzis: bool = True,
                 aseView: bool = False,
                 thresholdCoreSurface: float = 1.,
                 skipChiralityCalculation: bool=True,
                 skipSymmetryAnalyzis: bool = False,
                 skipFacetInfo: bool=True,
                 jmolCrystalShape: bool = True,
                 noOutput: bool = False,
                 calcPropOnly: bool = False,
                ):
        """
        Args:
            postAnalyzis (bool): If True, prints additional NP information
                (e.g., cell parameters, moments of inertia,
                inscribed/circumscribed sphere diameters, etc.).
            aseView (bool): If True, enables visualization of the NP using ASE.
            thresholdCoreSurface (float): Precision threshold for core/surface
                differentiation (distance threshold for retaining atoms).
            skipSymmetryAnalyzis (bool): If False, performs an atomic
                structure analysis using pymatgen.
            skipFacetInfo (bool): If True, skips the automatic computation of
                external facet areas and relative energies in propPostMake.
                Useful for shapes without flat facets (spheres, ellipsoids) or
                for large NPs where the computation is slow. Default is False.
                The analysis can always be run manually afterwards:
                    NP.external_facets_info(mode='auto', noOutput=False)
            jmolCrystalShape (bool): If True, generates a JMOL script
                for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure
                of the NP.      
        """
        self.postAnalyzis = postAnalyzis
        self.aseView = aseView
        self.thresholdCoreSurface = thresholdCoreSurface
        self.skipChiralityCalculation = skipChiralityCalculation
        self.skipSymmetryAnalyzis = skipSymmetryAnalyzis
        self.skipFacetInfo = skipFacetInfo
        self.jmolCrystalShape = jmolCrystalShape
        self.noOutput = noOutput
        self.calcPropOnly = calcPropOnly
        
        self.nAtoms = 0

        self.NP = None
        self.NPcs = None
        self.jMolCS = None
        self.vertices = None
        self.simplices = None
        self.neighbors = None
        self.equations = None
        self.surfaceatoms = None
        self.surfaceAtoms = None
        self.cog = np.array([0., 0., 0.])
        self.trPlanes = None
        self.moi = None
        self.NPR = None
        self.Rg = None
        self.vol_Hull = None
        self.area_Hull = None
        self.opd_index = None
        self.shape = None

        self.NP_opt = None
        self.NPcs_opt = None
        self.jMolCS_opt = None
        self.vertices_opt = None
        self.simplices_opt = None
        self.neighbors_opt = None
        self.equations_opt = None
        self.surfaceatoms_opt = None
        self.surfaceAtoms_opt = None
        self.cog_opt = []
        self.trPlanes_opt = None
        self.is_optimized = False
        self.is_peeled = False
        self.moi_opt = None
        self.NPR_opt = None
        self.Rg_opt = None
        self.vol_Hull_opt = None
        self.area_Hull_opt = None
        self.opd_opt = None
        
        self.ellipsoid = {} #two keys: "initial structure" or "optimized structure"

        self.chirality = "achiral"

        self.trPlanes_Wulff = None
        self.trPlanes_Slices = None
        self.WulffShape = None
        self.jMolSlices = None
        
        self.G = None
        self.Gstar = None
        self.ucMatrix = None
        
    def optimize(self, calculator='EMT', optimizer='QN', fthreshold=0.1, noOutput=False):
        """
        Optimize the NP geometry using an ASE calculator.
        See utils/energy.optimize for full documentation.

        Args:
            calculator (str): ASE calculator to use (default: 'EMT').
            optimizer (str): ASE optimizer to use (default: 'QN').
            fthreshold (float): Force convergence threshold in eV/Å (default: 0.1).
            noOutput (bool):if True, do not print the properties of the final geometry (default is False).
        """
        from .utils.energy import optimize
        return optimize(self, calculator, optimizer, fthreshold, noOutput)
    
    def _update_sasview_dims_from_spheres(self, noOutput = None):
        """
        Update SasView dimensions from inscribed/circumscribed sphere radii.
        See utils/prop._update_sasview_dims_from_spheres for full documentation.

        Args:
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.prop import _update_sasview_dims_from_spheres
        return _update_sasview_dims_from_spheres(self, noOutput)

    def Inscribed_circumscribed_spheres(self, noOutput=None):
        """
        Compute the inscribed and circumscribed sphere diameters of the NP.
        See utils/prop.Inscribed_circumscribed_spheres for full documentation.

        Args:
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.prop import Inscribed_circumscribed_spheres
        if noOutput is None: noOutput = self.noOutput    
        return Inscribed_circumscribed_spheres(self, noOutput)

    def get_ellipsoid_analysis(self, noOutput=None):
        """
        Fit an ellipsoid to the NP and compute its principal axes and dimensions.
        See utils/prop.get_ellipsoid_analysis for full documentation.

        Args:
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.prop import get_ellipsoid_analysis
        if noOutput is None: noOutput = self.noOutput    
        return get_ellipsoid_analysis(self, noOutput)

    def peel_by_coordination(self, threshold_peeling=6, Rmax=2.9, noOutput=None):
        """
        Remove surface atoms with coordination number below threshold_peeling.
        See utils/geometry.peel_by_coordination for full documentation.

        Args:
            threshold_peeling (int): Minimum coordination number to keep an atom
                                     (default: 6).
            Rmax (float): Cutoff distance in Å for neighbor search (default: 2.9).
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.geometry import peel_by_coordination
        if noOutput is None: noOutput = self.noOutput            
        return peel_by_coordination(self, threshold_peeling, Rmax, noOutput)

    def peel_by_shifted_ellipsoid(self, shift_dist=2.5, noOutput=None):
        """
        Truncate the NP using a shape-adaptive ellipsoidal envelope shifted
        in a random direction, simulating asymmetric growth or dissolution.
        See utils/geometry.peel_by_shifted_ellipsoid for full documentation.

        Args:
            shift_dist (float): Shift distance in Å, approximately one atomic
                                 layer (default: 2.5).
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.geometry import peel_by_shifted_ellipsoid
        if noOutput is None: noOutput = self.noOutput            
        return peel_by_shifted_ellipsoid(self, shift_dist, noOutput)

    def _flush_stale_data(self, shape_update=None):
        """
        Reset all stale derived attributes after a geometry modification.
        See utils/core._flush_stale_data for full documentation.

        Args:
            shape_update (str, optional): Tag appended to self.shape to record
                                          the modification (e.g. '_Twist').
        """
        from .utils.core import _flush_stale_data
        return _flush_stale_data(self, shape_update)

    def propPostMake(self, skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
                     skipFacetInfo=None,
                     thresholdCoreSurface=None, noOutput=None, is_optimized=None):
        """
        Compute post-construction properties: MOI, NPR, Rg, core/surface,
        convex hull, inscribed/circumscribed spheres, ellipsoid, and JMol script.
        See utils/prop.propPostMake for full documentation.

        Args:
            skipChiralityCalculation (bool): If True, skips OPD chirality index
                                             (default: self.skipChiralityCalculation).
            skipSymmetryAnalyzis (bool): If True, skips pymatgen symmetry analysis
                                         (default: self.skipSymmetryAnalyzis).
            skipFacetInfo (bool): If True, skips the automatic computation of 
                                  external facet areas and relative energies
            thresholdCoreSurface (float): Distance threshold for core/surface
                                          differentiation (default: self.thresholdCoreSurface).
            noOutput (bool): If True, suppresses output (default: self.noOutput).
            is_optimized (bool): If True, targets NP_opt (default: self.is_optimized).
        """
        from .utils.prop import propPostMake
        if skipChiralityCalculation is None: skipChiralityCalculation = self.skipChiralityCalculation
        if skipSymmetryAnalyzis is None: skipSymmetryAnalyzis = self.skipSymmetryAnalyzis
        if skipFacetInfo is None: skipFacetInfo = self.skipFacetInfo
        if thresholdCoreSurface is None: thresholdCoreSurface = self.thresholdCoreSurface
        if noOutput is None: noOutput = self.noOutput
        if is_optimized is None: is_optimized = self.is_optimized
        return propPostMake(self, skipChiralityCalculation, skipSymmetryAnalyzis,
                            skipFacetInfo,
                            thresholdCoreSurface, noOutput, is_optimized)

    def plot_npr_triangle(self=None, is_optimized: bool = None, save_path: str = None, 
                      external_data: dict = None, color_by: str = 'Rg', color: str = 'viridis'):
        """
        Plot the NPR triangle (Rod/Sphere/Disk) for shape classification.
        See utils/geometry.plot_npr_triangle for full documentation.

        Args:
            is_optimized (bool): If True, uses optimized structure data
                                  (default: self.is_optimized).
            save_path (str, optional): Path to save the figure (SVG or PNG).
            external_data (dict, optional): Population data for multi-NP plots,
                                            with keys 'NPR', 'Rg', and 'shapes'.
            color_by (str): Coloring scheme: 'Rg' or 'shapes' (default: 'Rg').
            color (str): Matplotlib colormap name (default: 'viridis').
        """
        from .utils.geometry import plot_npr_triangle

        if self is not None:
            if is_optimized is None: is_optimized = getattr(self, 'is_optimized', False)

        plot_npr_triangle(
            self, 
            is_optimized=is_optimized, 
            save_path=save_path, 
            external_data=external_data, 
            color_by=color_by, 
            color=color
        )

    def applyTwist(self, axis=[0,0,1], axis_def='hkl', rate: float = 1.0, 
                 profile: str = 'linear', custom_profile=None,
                 pitch: float = None, helix_radius: float = None,
                 chirality: str = 'RH',
                 noOutput: bool = None):
        """
        Apply a Twist to the NP along a given axis.
        See utils/geometry.applyTwist for full documentation.
    
        Args:
            axis (array-like): Twist axis in crystallographic [h, k, l] or
                               Cartesian [x, y, z] coordinates depending on axis_def.
            axis_def (str): Coordinate system of axis: 'hkl' (default) or 'cart'.
            rate (float): Twist rate in degrees per Å (linear, helical), peak
                          amplitude in degrees (sinusoidal, gaussian), or scaling
                          factor (custom). Not used for 'helix'. Default is 1.0.
            profile (str): Twist profile: 'linear', 'sinusoidal', 'gaussian',
                           'helical', 'helix', or 'custom'. Default is 'linear'.
            custom_profile (callable, optional): User-defined function f(z, L) -> float,
                           required when profile='custom'.
            pitch (float, optional): Helix pitch in Å/turn, required when
                           profile='helical' or 'helix'.
            helix_radius (float, optional): Radius of the helical path in Å,
                           required when profile='helix'.
            chirality (str): Handedness of the Twist or helix: 'RH' (Right-Handed,
                         default) or 'LH' (Left-Handed, mirror image).
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.geometry import applyTwist
        if noOutput is None:
            noOutput = self.noOutput
        applyTwist(self, axis, axis_def, rate, profile, custom_profile, pitch, helix_radius, chirality, noOutput)

    def defHelixShapeForJMol(self, n_rings=50, n_sides=12, noOutput=True):
        """
        Generate a Jmol command to visualize the helical envelope as a triangulated tube.
        See utils/external_pgm.defHelixShapeForJMol for full documentation.
    
        Args:
            n_rings (int): Number of rings along the helix (default: 50).
            n_sides (int): Number of vertices per ring (default: 12).
            noOutput (bool): If True, suppresses output. Default is True.
        """
        from .utils.external_pgm import defHelixShapeForJMol
        return defHelixShapeForJMol(self, n_rings, n_sides, noOutput)

    def crystallographic_angle(self, v1, v2,
                            type1: str = 'direction',
                            type2: str = 'direction',
                            noOutput: bool = None):
        """
        Compute the angle between two crystallographic objects
        (directions or planes) in any crystal system.
    
        Args:
            v1 (array-like): First vector [h, k, l] or [u, v, w].
            v2 (array-like): Second vector [h, k, l] or [u, v, w].
            type1 (str): 'direction' or 'plane'. Default is 'direction'.
            type2 (str): 'direction' or 'plane'. Default is 'direction'.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
    
        Returns:
            float: Angle in degrees.
        """
        from .utils.crystals import crystallographic_angle
        if noOutput is None:
            noOutput = self.noOutput
        return crystallographic_angle(self, v1, v2, type1, type2, noOutput)

    def generateSlab(self,
                     hkl,
                     size_a: float = 2.0,
                     size_b: float = 2.0,
                     min_thickness: float = 5.0,
                     n_layers: int = None,
                     vacuum: float = 10.0,
                     backend: str = 'ase',
                     primitive: bool=False,
                     noOutput: bool = None):
        """
        Generate a crystallographic slab from Miller indices (hkl).
        See utils/crystals.generateSlab for full documentation.
    
        Args:
            hkl (array-like): Miller indices [h, k, l].
            size_a (float): Slab dimension along a in nm. Default is 2.0.
            size_b (float): Slab dimension along b in nm. Default is 2.0.
            min_thickness (float): Minimum slab thickness in Å. Default is 5.0.
            n_layers (integer): Directly specifies the number of layers. Default is None.
            vacuum (float): Vacuum thickness in Å. Default is 10.0.
            backend (str): 'ase' or 'pymatgen'. Default is 'ase'.
            primitive (bool): pymatgen backend only. If True, uses the primitive surface cell. Default is False.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
    
        Returns:
            pyNMBcore: A pyNMBcore instance wrapping the generated slab.
        """
        from .utils.crystals import generateSlab
        if noOutput is None:
            noOutput = self.noOutput
        return generateSlab(self, hkl, size_a, size_b, min_thickness,
                            n_layers, vacuum, backend, primitive, noOutput)
    
    def defSlabShapeForJMol(self, hkl, offset: float = 1.5, noOutput: bool = None):
        """
        Generate a Jmol command to visualize the slab surface as a polygon.
        See utils/external_pgm.defSlabShapeForJMol for full documentation.

        Args:
            hkl (array-like): Miller indices [h, k, l] of the plane, used for labeling.
            offset (float): Vertical offset in Å above the topmost atomic layer.
                            Default is 1.5 Å.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.

        Returns:
            str: Jmol command string.
        """
        from .utils.external_pgm import defSlabShapeForJMol
        if noOutput is None:
            noOutput = self.noOutput
        return defSlabShapeForJMol(self, hkl, offset, noOutput)

    def interPlanarSpacing(self, hkl, noOutput=None):
        """
        Compute the interplanar spacing d(hkl) for this crystal.
        Only available for Crystal instances loaded from a CIF file.
    
        Args:
            hkl (array-like): Miller indices [h, k, l].
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
    
        Returns:
            float: Interplanar spacing in Å, or None if crystallographic
                   metadata is not available (e.g. geometric NP classes).
        """
        from .utils.crystals import interPlanarSpacing, spacegroup_to_bravais
        if noOutput is None:
            noOutput = self.noOutput
    
        if not hasattr(self, 'ucSG_number') or not hasattr(self, 'ucUnitcell'):
            if not noOutput:
                print(f"interPlanarSpacing: crystallographic metadata not available "
                      f"for {self.__class__.__name__} — method only supported for "
                      f"Crystal instances loaded from a CIF file.")
            return None
    
        crystal_system, bravais_lattice = spacegroup_to_bravais(self.ucSG_number, self.ucSG_symbol)
        d = interPlanarSpacing(np.array(hkl), self.ucUnitcell, crystal_system)
        if not noOutput:
            print(f"d({hkl}) = {d:.4f} Å  [{crystal_system}]")
        return d
        
    def external_facets_info(self, mode='auto', noOutput=None):
        """
        Compute and display geometric properties of each facet.
    
        Args:
            mode (str): Which planes to use. Options:
                - 'auto'        : automatically selects Wulff if available,
                                  then trPlanes_opt, then trPlanes.
                - 'Wulff'       : use trPlanes_Wulff + Miller indices labeling.
                - 'Slices'      : use trPlanes_Slices
                - 'crystal'     : use trPlanes (initial structure).
                - 'crystal_opt' : use trPlanes_opt (optimized structure).
                Default is 'auto'.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
    
        Returns:
            tuple: (distances, e_relative, facet_areas_per_plane) or None.
        """
        from .utils.prop import external_facets_info
        if noOutput is None:
            noOutput = self.noOutput
    
        valid_modes = ('auto', 'Wulff', 'Slices', 'crystal', 'crystal_opt')
        if mode not in valid_modes:
            print(f"{bg.LIGHTYELLOWB}Warning: unknown mode '{mode}'. "
                  f"Valid options are: {valid_modes}. "
                  f"Falling back to 'auto'.{bg.OFF}")
            mode = 'auto'
    
        return external_facets_info(self, mode=mode, noOutput=noOutput)

    def applySlicing(self, planes, distance_unit='nm', mode='OR', recenter: bool = True, noOutput=None):
        """
        Apply one or more truncation plane groups to self.NP, with optional
        rotational symmetry generation and logical combination of groups.
        Works on any pyNMBcore object.

        See utils.csg.applySlicing for full documentation.

        Args:
            planes (list of dict): List of plane group definitions. Each dict
                must contain either 'normal' or 'angle', plus 'distance' and
                'side'. Optional keys: 'normal_def', 'nRot', 'rotAxis', 'modeP'.
            distance_unit (str): Unit for all distances. 'nm' (default) or
                'Angstrom'.
            mode (str): Logical combination of plane groups.
                'OR'  — atom removed if condemned by ANY group (default).
                'AND' — atom removed only if condemned by ALL groups.
            recenter (bool): If True, recenters self.NP on its center of mass
                after slicing (default).
            noOutput (bool): If True, suppresses output. Default is self.noOutput.

        Returns:
            None. Updates self.NP, self.nAtoms, self.cog,
            self.trPlanes_Slices in place. self.trPlanes_Wulff is set to None.
        """
        from .utils.csg import applySlicing
        if noOutput is None:
            noOutput = self.noOutput
        return applySlicing(self, planes,
                            distance_unit=distance_unit,
                            mode=mode,
                            recenter=recenter,
                            noOutput=noOutput)

    def cut_by(self, NP_B, cogB=None, rotB=None, mode='hull',
              threshold=0.8, skipSymmetryAnalyzis=None,
              thresholdCoreSurface=None, noOutput=None):
        """
        Remove from self.NP the atoms inside NP_B (hollow cavity).
        See utils.csg.cut_by for full documentation.
    
        Args:
            NP_B: pyNMBcore object defining the cavity shape.
            cogB (list): Center of mass position for B in nm.
            rotB: rotation operation applied to B (axis+angle or list of rotations).
                (see detailed docstring of utils.csg.minus())
            mode (str): 'hull' (convex hull) or 'atoms' (distance-based).
            threshold (float): Distance threshold in units of Rnn (mode='atoms').
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.csg import cut_by
        # --- Use defaults from self if not provided ---
        if noOutput is None:
            noOutput = self.noOutput
        if skipSymmetryAnalyzis is None:
            skipSymmetryAnalyzis = getattr(self, 'skipSymmetryAnalyzis', True)
        if thresholdCoreSurface is None:
            thresholdCoreSurface = getattr(self, 'thresholdCoreSurface', 1.0)
        return cut_by(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                      threshold=threshold,
                      skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                      thresholdCoreSurface=thresholdCoreSurface,
                      noOutput=noOutput)
        
    def union_with(self, NP_B, cogB=None, rotB=None, mode='hull',
              threshold=0.8, skipSymmetryAnalyzis=None,
              thresholdCoreSurface=None, noOutput=None):
        """
        Add NP_B to self.NP, removing overlapping atoms.
        See utils.csg.union_with for full documentation.
    
        Args:
            NP_B: pyNMBcore object to add.
            cogB (list): Center of mass position for B in nm.
            rotB: Rotation for B (axis+angle or list of rotations).
                (see detailed docstring of utils.csg.plus())
            mode (str): 'hull' (convex hull) or 'atoms' (distance-based).
            threshold (float): Overlap threshold in units of Rnn.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.csg import union_with
        # --- Use defaults from self if not provided ---
        if noOutput is None:
            noOutput = self.noOutput
        if skipSymmetryAnalyzis is None:
            skipSymmetryAnalyzis = getattr(self, 'skipSymmetryAnalyzis', True)
        if thresholdCoreSurface is None:
            thresholdCoreSurface = getattr(self, 'thresholdCoreSurface', 1.0)
        return union_with(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                          threshold=threshold,
                          skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                          thresholdCoreSurface=thresholdCoreSurface,
                          noOutput=noOutput)

    def intersect_with(self, NP_B, cogB=None, rotB=None,mode='hull',
              threshold=0.8, skipSymmetryAnalyzis=None,
              thresholdCoreSurface=None, noOutput=None):
        """
        Keep in self.NP only the atoms inside NP_B.
        See utils.csg.intersect_with for full documentation.
    
        Args:
            NP_B: pyNMBcore object defining the intersection region.
            cogB (list): Center of mass position for B in nm.
            rotB: Rotation for B (axis+angle or list of rotations).
            mode (str): 'hull' (convex hull) or 'atoms' (distance-based).
            threshold (float): Distance threshold in units of Rnn (mode='atoms').
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.csg import intersect_with
        # --- Use defaults from self if not provided ---
        if noOutput is None:
            noOutput = self.noOutput
        if skipSymmetryAnalyzis is None:
            skipSymmetryAnalyzis = getattr(self, 'skipSymmetryAnalyzis', True)
        if thresholdCoreSurface is None:
            thresholdCoreSurface = getattr(self, 'thresholdCoreSurface', 1.0)
        return intersect_with(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                              threshold=threshold,
                              skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                              thresholdCoreSurface=thresholdCoreSurface,
                              noOutput=noOutput)

    def flush_inlay_with(self, NP_B, cogB=None, rotB=None, mode='hull',
                         threshold=0.8, skipSymmetryAnalyzis=None,
                         thresholdCoreSurface=None, noOutput=None):
        """
        Add to self.NP the part of NP_B that overlaps with self.NP.
        See utils.csg.flush_inlay_with for full documentation.
    
        Args:
            NP_B: pyNMBcore object to partially merge.
            cogB (list): Center of mass position for B in nm.
            rotB: Rotation for B (axis+angle or list of rotations).
            mode (str): 'hull' (convex hull) or 'atoms' (distance-based).
            threshold (float): Distance threshold in units of Rnn.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.csg import flush_inlay_with
        # --- Use defaults from self if not provided ---
        if noOutput is None:
            noOutput = self.noOutput
        if skipSymmetryAnalyzis is None:
            skipSymmetryAnalyzis = getattr(self, 'skipSymmetryAnalyzis', True)
        if thresholdCoreSurface is None:
            thresholdCoreSurface = getattr(self, 'thresholdCoreSurface', 1.0)
        return flush_inlay_with(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                                threshold=threshold,
                                skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                thresholdCoreSurface=thresholdCoreSurface,
                                noOutput=noOutput)

    def copy(self):
        "Create and return a deep copy of any pyNanoMatBuilder system"
        from .utils.core import clone
        return clone(self)

    def effective_diameter(self, structure=None, mode='vertices'):
        """Returns the volume-equivalent diameter from the ellipsoid analysis, in nm."""
        key = 'optimized structure' if structure == 'optimized' else 'initial structure'
        from .utils.prop import effective_diameter
        return effective_diameter(self, structure, mode)

    def get_ellipsoid_analysis(self, noOutput=False, mode='vertices'):
        """
        Perform a Principal Component Analysis (PCA) on the outer envelope to 
        calculate the best-fitting ellipsoid calculated after all atoms (mode='all'),
        surface atoms (mode='surface') or Hull vertices (mode='vertices').
        Returns:
            dict: A dictionary containing the following physical properties:
                - "status": String indicating which geometry was analyzed (optimized or initial).
                - "mode": String indicating which ellipsoid has been calculated.
                - "D1", "D2", "D3": Major, intermediate, and minor diameters (Å).
                - "volume": Volume of the ellipsoid (Å³).
                - "surface": Approximate surface area (Å²) using Knud Thomsen's formula.
                - "asphericity": Ratio of D1/D3 (1.0 for a perfect sphere).
        """
        from .utils.prop import get_ellipsoid_analysis
        return  get_ellipsoid_analysis(self, noOutput, mode)

######################################### load external file
    @classmethod
    def from_file(cls, file_path, **kwargs):
        """
        Create a pyNMB object from any structural file format supported by ASE 
        (.xyz, .cif, .pdb, etc.) and run the analysis pipeline.
        
        Args:
            file_path (str): Path to the structural file.
            **kwargs: Arguments for propPostMake and internal settings.
        """
        from ase.io import read as ase_read
        from pathlib import Path
        # 1. Instantiate the object without calling __init__
        instance = cls.__new__(cls)
        
        # 2. Universal read with ASE
        # ASE will automatically detect the format (CIF, PDB, XYZ, etc.)
        try:
            instance.NP = ase_read(file_path)
        except Exception as e:
            print(f"Error: ASE could not read the file {file_path}. {e}")
            return None
        
        # 3. Basic metadata setup
        symbols = instance.NP.get_chemical_symbols()
        instance.element = symbols[0] if symbols else "X"
        
        # 4. Cleanup and synchronize data structures
        instance.skipChiralityCalculation = kwargs.get('skipChiralityCalculation', True)
        instance.skipSymmetryAnalyzis = kwargs.get('skipSymmetryAnalyzis', False)
        instance.skipFacetInfo = kwargs.get('skipFacetInfo', False)
        instance.thresholdCoreSurface = kwargs.get('thresholdCoreSurface', 1.0)
        instance.aseView = kwargs.get('aseView', False)
        instance.jmolCrystalShape = kwargs.get('jmolCrystalShape', True)
        instance.noOutput = kwargs.get('noOutput', False)
        instance.calcPropOnly = kwargs.get('calcPropOnly', False)
        instance._flush_stale_data(shape_update=f"loaded_{Path(file_path).suffix[1:]}")
        instance.shape=""
        
        # 5. Operational settings
        instance.noOutput = kwargs.get('noOutput', False)
        
        # 6. Property analysis (OPD, MOI, etc.)
        instance.propPostMake(
            skipChiralityCalculation=instance.skipChiralityCalculation,
            skipSymmetryAnalyzis=instance.skipSymmetryAnalyzis,
            skipFacetInfo=instance.skipFacetInfo,
            thresholdCoreSurface=instance.thresholdCoreSurface,
            noOutput=instance.noOutput,
            is_optimized=False,
        )
        
        return instance

    @classmethod
    def from_slab(cls, slab, crystal_name="unknown", **kwargs):
        """
        Create a pyNMBcore instance from an ASE slab object.
        """
        instance = cls.__new__(cls)
        instance.NP = slab.copy()
        instance.element = slab.get_chemical_symbols()[0]
        instance.crystal = crystal_name
        instance.shape = 'slab'
        # standard kwargs
        instance.skipChiralityCalculation = kwargs.get('skipChiralityCalculation', True)
        instance.skipSymmetryAnalyzis = kwargs.get('skipSymmetryAnalyzis', True)
        instance.thresholdCoreSurface = kwargs.get('thresholdCoreSurface', 1.0)
        instance.aseView = kwargs.get('aseView', False)
        instance.noOutput = kwargs.get('noOutput', False)
        instance.jmolCrystalShape = False  # no trPlanes for a slab. Jmol plane will be calculated separately with defSlabShapeForJMol
        instance._flush_stale_data(shape_update='_slab')
        instance.propPostMake(
            skipSymmetryAnalyzis=instance.skipSymmetryAnalyzis,
            thresholdCoreSurface=instance.thresholdCoreSurface,
            noOutput=instance.noOutput,
            is_optimized=False,
        )
        return instance