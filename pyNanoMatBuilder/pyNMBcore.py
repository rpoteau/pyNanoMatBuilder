import numpy as np
from .utils.core import centertxt, centerTitle, fg, bg, hl, color

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
        self.cnp = None
        self.cnp_mean = None
        self.q4 = None
        self.q4_mean = None
        self.q6 = None
        self.q6_mean = None
        self.jMol_cnp = None
        self.jMol_q4 = None
        self.jMol_q6 = None
        self.NP_select = None
        self.NP_select_mask = None

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
        self.cnp_opt = None
        self.cnp_mean_opt = None
        self.q4_opt = None
        self.q4_mean_opt = None
        self.q6_opt = None
        self.q6_mean_opt = None
        self.jMol_cnp_opt = None
        self.jMol_q4_opt = None
        self.jMol_q6_opt = None
        self.NP_select_opt = None
        self.NP_select_mask_opt = None
        
        self.ellipsoid = {} #two keys: "initial structure" or "optimized structure"

        self.chirality = "achiral"

        self._local_order_decimals = None

        self.trPlanes_Wulff = None
        self.trPlanes_Slices = None
        self.WulffShape = None
        self.jMolSlices = None

        self.NP_preview = None
        self.jMolCarvePreview = None
        self.jMolStellationPreview = None
        
        self.G = None
        self.Gstar = None
        self.ucMatrix = None
        
    def optimize(self, calculator='EMT', optimizer='QN', fthreshold=0.1,
                 traj_file=None, xyz_file=None,
                 eam_potential=None, eam_form=None, 
                 noOutput=False):
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
        return optimize(self, calculator, optimizer, fthreshold, traj_file, xyz_file, eam_potential, eam_form, noOutput)
    
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

    def peel_by_coordination(self, threshold_peeling=6, Rmax=2.9, noOutput=None,
                             postAnalyzis=None, skipChiralityCalculation=None,skipSymmetryAnalyzis=None,
                             skipFacetInfo=None, thresholdCoreSurface=None):
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
        return peel_by_coordination(self, threshold_peeling, Rmax, noOutput,
                                    postAnalyzis, skipChiralityCalculation, skipSymmetryAnalyzis,
                                    skipFacetInfo, thresholdCoreSurface)

    def peel_by_shifted_ellipsoid(self, shift_dist=2.5, shift_direction=None,
                                   axis_def='hkl', noOutput=None,
                                   postAnalyzis=None,
                                   skipChiralityCalculation=None,
                                   skipSymmetryAnalyzis=None,
                                   skipFacetInfo=None,
                                   thresholdCoreSurface=None):
        """Truncate self.NP using a shape-adaptive shifted ellipsoid envelope.
        See utils.geometry.peel_by_shifted_ellipsoid."""
        from .utils.geometry import peel_by_shifted_ellipsoid
        if noOutput is None: noOutput = self.noOutput
        return peel_by_shifted_ellipsoid(self, shift_dist=shift_dist,
                                         shift_direction=shift_direction,
                                         axis_def=axis_def,
                                         noOutput=noOutput,
                                         postAnalyzis=postAnalyzis,
                                         skipChiralityCalculation=skipChiralityCalculation,
                                         skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                         skipFacetInfo=skipFacetInfo,
                                         thresholdCoreSurface=thresholdCoreSurface)

    def align_to_plane(self, axis=(0, 0, 1), target=0.0, tol=0.1, noOutput=None,
                       postAnalyzis=None,
                       skipChiralityCalculation=None,
                       skipSymmetryAnalyzis=None,
                       skipFacetInfo=None,
                       thresholdCoreSurface=None):
        """
        Translate the structure so its lowest atomic plane along `axis` lands
        at `target`. See utils.geometry.align_to_plane for full documentation.
        """
        from .utils.geometry import align_to_plane
        if noOutput is None: noOutput = self.noOutput
        return align_to_plane(self, axis=axis, target=target, tol=tol,
                             noOutput=noOutput,
                             postAnalyzis=postAnalyzis,
                             skipChiralityCalculation=skipChiralityCalculation,
                             skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                             skipFacetInfo=skipFacetInfo,
                             thresholdCoreSurface=thresholdCoreSurface)
    
    def remove_plane(self, direction, axis_def='hkl', tol=0.5,
                     noOutput=None,
                     postAnalyzis=None,
                     skipChiralityCalculation=None,
                     skipSymmetryAnalyzis=None,
                     skipFacetInfo=None,
                     thresholdCoreSurface=None):
        """
        Remove the outermost atomic plane in a given direction.
        See utils.geometry.remove_plane for full documentation.
 
        Args:
            direction (array-like): Plane normal as Miller indices [h, k, l]
                (axis_def='hkl') or Cartesian [x, y, z] (axis_def='cart').
            axis_def (str): 'hkl' (default) or 'cart'.
            tol (float): Tolerance in Å to identify atoms in the outermost
                plane. Default is 0.5 Å.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
        """
        from .utils.geometry import remove_plane
        if noOutput is None: noOutput = self.noOutput
        return remove_plane(self, direction, axis_def=axis_def, tol=tol,
                            noOutput=noOutput,
                            postAnalyzis=postAnalyzis,
                            skipChiralityCalculation=skipChiralityCalculation,
                            skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                            skipFacetInfo=skipFacetInfo,
                            thresholdCoreSurface=thresholdCoreSurface)  
        
    def round_tip_in_direction(self, direction, diameter_nm, axis_def='hkl',
                               use_axis_center=True, noOutput=None,
                               postAnalyzis=None,
                               skipChiralityCalculation=None,
                               skipSymmetryAnalyzis=None,
                               skipFacetInfo=None,
                               thresholdCoreSurface=None):
        """
        Round one tip with a hemispherical cap tangent to the outermost atom
        in a given direction. See utils.geometry.round_tip_in_direction for
        full documentation.
        """
        from .utils.geometry import round_tip_in_direction
        if noOutput is None: noOutput = self.noOutput
        return round_tip_in_direction(self, direction, diameter_nm,
                                     axis_def=axis_def,
                                     use_axis_center=use_axis_center,
                                     noOutput=noOutput,
                                     postAnalyzis=postAnalyzis,
                                     skipChiralityCalculation=skipChiralityCalculation,
                                     skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                     skipFacetInfo=skipFacetInfo,
                                     thresholdCoreSurface=thresholdCoreSurface)
    
    def clip_to_sphere(self, radius_nm, noOutput=None,
                           postAnalyzis=None,
                           skipChiralityCalculation=None,
                           skipSymmetryAnalyzis=None,
                           skipFacetInfo=None,
                           thresholdCoreSurface=None):
            """
            Keep only atoms within a sphere of given radius from the center of mass.
            See utils.geometry.clip_to_sphere for full documentation.
            """
            from .utils.geometry import clip_to_sphere
            if noOutput is None: noOutput = self.noOutput
            return clip_to_sphere(self, radius_nm,
                                  noOutput=noOutput,
                                  postAnalyzis=postAnalyzis,
                                  skipChiralityCalculation=skipChiralityCalculation,
                                  skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                  skipFacetInfo=skipFacetInfo,
                                  thresholdCoreSurface=thresholdCoreSurface)

    def clip_to_ellipsoid(self, diameters_nm, noOutput=None,
                          postAnalyzis=None,
                          skipChiralityCalculation=None,
                          skipSymmetryAnalyzis=None,
                          skipFacetInfo=None,
                          thresholdCoreSurface=None):
        """
        Keep only atoms within an axis-aligned ellipsoid centred on the
        center of mass. See utils.geometry.clip_to_ellipsoid for full
        documentation.
        """
        from .utils.geometry import clip_to_ellipsoid
        if noOutput is None: noOutput = self.noOutput
        return clip_to_ellipsoid(self, diameters_nm,
                                 noOutput=noOutput,
                                 postAnalyzis=postAnalyzis,
                                 skipChiralityCalculation=skipChiralityCalculation,
                                 skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                 skipFacetInfo=skipFacetInfo,
                                 thresholdCoreSurface=thresholdCoreSurface)

    def clip_to_cylinder(self, diameter_nm, axis=(0, 0, 1), noOutput=None,
                         postAnalyzis=None,
                         skipChiralityCalculation=None,
                         skipSymmetryAnalyzis=None,
                         skipFacetInfo=None,
                         thresholdCoreSurface=None):
        """
        Keep only atoms within a cylinder of given diameter around an axis
        through the centre of mass. See utils.geometry.clip_to_cylinder for
        full documentation.
        """
        from .utils.geometry import clip_to_cylinder
        if noOutput is None: noOutput = self.noOutput
        return clip_to_cylinder(self, diameter_nm, axis=axis,
                               noOutput=noOutput,
                               postAnalyzis=postAnalyzis,
                               skipChiralityCalculation=skipChiralityCalculation,
                               skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                               skipFacetInfo=skipFacetInfo,
                               thresholdCoreSurface=thresholdCoreSurface)
    
    def clip_to_cone(self, base_center_nm, apex_nm, base_radius_nm=None,
                     apex_angle_deg=None, tip_sphere_radius_nm=0.0, 
                     keep='inside',  keep_opposite_side=False,
                     recenter=True, noOutput=None,
                     postAnalyzis=None,
                     skipChiralityCalculation=None,
                     skipSymmetryAnalyzis=None,
                     skipFacetInfo=None,
                     thresholdCoreSurface=None):
        """
        Clip atoms against a single general cone (base disc + apex).
        See utils.geometry.clip_to_cone for full documentation.
        """
        from .utils.geometry import clip_to_cone
        if noOutput is None: noOutput = self.noOutput
        return clip_to_cone(self, base_center_nm, apex_nm, base_radius_nm, apex_angle_deg,
                            tip_sphere_radius_nm,
                            keep=keep, keep_opposite_side=keep_opposite_side,
                            recenter=recenter,
                            noOutput=noOutput,
                            postAnalyzis=postAnalyzis,
                            skipChiralityCalculation=skipChiralityCalculation,
                            skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                            skipFacetInfo=skipFacetInfo,
                            thresholdCoreSurface=thresholdCoreSurface)
    
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
                     thresholdCoreSurface=None, noOutput=None, is_optimized=None, coreSurfaceMethod=None):
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
            coreSurfaceMethod (str): 'hull' (default), 'cnp', or 'combined'.
                Criterion for the surfaceAtoms mask (default: self.coreSurfaceMethod).
        """
        from .utils.prop import propPostMake
        if skipChiralityCalculation is None: skipChiralityCalculation = self.skipChiralityCalculation
        if skipSymmetryAnalyzis is None: skipSymmetryAnalyzis = self.skipSymmetryAnalyzis
        if skipFacetInfo is None: skipFacetInfo = self.skipFacetInfo
        if thresholdCoreSurface is None: thresholdCoreSurface = self.thresholdCoreSurface
        if noOutput is None: noOutput = self.noOutput
        if is_optimized is None: is_optimized = self.is_optimized
        if coreSurfaceMethod is None: coreSurfaceMethod = 'combined'
        return propPostMake(self, skipChiralityCalculation, skipSymmetryAnalyzis,
                            skipFacetInfo,
                            thresholdCoreSurface, noOutput, is_optimized, coreSurfaceMethod)

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
                 depth_nm: float = 0.0,
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
            depth_nm (float): Thickness (nm) of the twisted surface cap,
                measured inward from the outermost plane along the
                axis. Only the cap is twisted; the core stays fixed,
                with the angle growing from 0 at the cap boundary to
                full value at the surface. depth_nm = 0.0 (default)
                = no limit (whole object twisted). Small positive
                values give a per-facet surface twist.
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
        applyTwist(self, axis, axis_def, rate, depth_nm, profile, custom_profile, pitch, helix_radius, chirality, noOutput)

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

    def angles_between_planes(self, hkl_list, noOutput=False):
        """Compute all pairwise dihedral angles between a list of crystallographic planes."""
        from .utils.crystals import angles_between_planes
        return angles_between_planes(self, hkl_list, noOutput=noOutput)
   
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

    def applySlicing(self, planes, distance_unit='nm', mode='OR', recenter: bool = True,
                     noOutput=None, postAnalyzis=None, skipChiralityCalculation=None,
                     skipSymmetryAnalyzis=None, skipFacetInfo=None, thresholdCoreSurface=None):
        """
        Apply one or more truncation plane groups to self.NP, with optional
        rotational symmetry generation and logical combination of groups.
        Works on any pyNMBcore object. 
    
        See utils.csg.applySlicing for full documentation.
        """
        from .utils.csg import applySlicing
        if noOutput is None:
            noOutput = self.noOutput
        return applySlicing(self, planes,
                            distance_unit=distance_unit,
                            mode=mode,
                            recenter=recenter,
                            noOutput=noOutput,
                            postAnalyzis=postAnalyzis,
                            skipChiralityCalculation=skipChiralityCalculation,
                            skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                            skipFacetInfo=skipFacetInfo,
                            thresholdCoreSurface=thresholdCoreSurface)

    def cut_by(self, NP_B, cogB=None, rotB=None, mode='hull', threshold=0.8, recenter=True,
               noOutput=None, postAnalyzis=None, skipChiralityCalculation=None,
               skipSymmetryAnalyzis=None, skipFacetInfo=None, thresholdCoreSurface=None):
        """Cut self.NP by NP_B — keeps atoms of A outside B. See utils.csg.cut_by for full documentation."""
        from .utils.csg import cut_by
        if noOutput is None: noOutput = self.noOutput
        return cut_by(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                      threshold=threshold, recenter=recenter, noOutput=noOutput,
                      postAnalyzis=postAnalyzis,
                      skipChiralityCalculation=skipChiralityCalculation,
                      skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                      skipFacetInfo=skipFacetInfo,
                      thresholdCoreSurface=thresholdCoreSurface)


    def union_with(self, NP_B, cogB=None, rotB=None, mode='hull', threshold=0.8, recenter=True,
                   noOutput=None, postAnalyzis=None, skipChiralityCalculation=None,
                   skipSymmetryAnalyzis=None, skipFacetInfo=None, thresholdCoreSurface=None):
        """Union of self.NP and NP_B — keeps all atoms of A and B. See utils.csg.union_with for full documentation."""
        from .utils.csg import union_with
        if noOutput is None: noOutput = self.noOutput
        return union_with(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                          threshold=threshold, recenter=recenter, noOutput=noOutput,
                          postAnalyzis=postAnalyzis,
                          skipChiralityCalculation=skipChiralityCalculation,
                          skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                          skipFacetInfo=skipFacetInfo,
                          thresholdCoreSurface=thresholdCoreSurface)
    
    
    def intersect_with(self, NP_B, cogB=None, rotB=None, mode='hull', threshold=0.8, recenter=True,
                       noOutput=None, postAnalyzis=None, skipChiralityCalculation=None,
                       skipSymmetryAnalyzis=None, skipFacetInfo=None, thresholdCoreSurface=None):
        """Intersection of self.NP and NP_B — keeps atoms inside both. See utils.csg.intersect_with for full documentation."""
        from .utils.csg import intersect_with
        if noOutput is None: noOutput = self.noOutput
        return intersect_with(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                              threshold=threshold, recenter=recenter, noOutput=noOutput,
                              postAnalyzis=postAnalyzis,
                              skipChiralityCalculation=skipChiralityCalculation,
                              skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                              skipFacetInfo=skipFacetInfo,
                              thresholdCoreSurface=thresholdCoreSurface)
    
    
    def flush_inlay_with(self, NP_B, cogB=None, rotB=None, mode='hull', threshold=0.8, recenter=True,
                         noOutput=None, postAnalyzis=None, skipChiralityCalculation=None,
                         skipSymmetryAnalyzis=None, skipFacetInfo=None, thresholdCoreSurface=None):
        """Flush inlay of NP_B into self.NP. See utils.csg.flush_inlay_with for full documentation."""
        from .utils.csg import flush_inlay_with
        if noOutput is None: noOutput = self.noOutput
        return flush_inlay_with(self, NP_B, cogB=cogB, rotB=rotB, mode=mode,
                                threshold=threshold, recenter=recenter, noOutput=noOutput,
                                postAnalyzis=postAnalyzis,
                                skipChiralityCalculation=skipChiralityCalculation,
                                skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                skipFacetInfo=skipFacetInfo,
                                thresholdCoreSurface=thresholdCoreSurface)

    def systematic_carve_by(self, NP_B, carve_axis=None, axis_through='vertex',
                            lead='apex', depth_nm=None, scale=1.0, phase_deg=0.0,
                            mode='hull', threshold=0.8,
                            preview=False, recenter=True,
                            noOutput=None, postAnalyzis=None,
                            skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
                            skipFacetInfo=None, thresholdCoreSurface=None):
        """Carve every face of self.NP with the pattern object NP_B (emporte-pièce).
        See utils.csg.systematic_carve_by for full documentation."""
        from .utils.csg import systematic_carve_by
        if noOutput is None: noOutput = self.noOutput
        return systematic_carve_by(self, NP_B, carve_axis=carve_axis,
                                   axis_through=axis_through, lead=lead,
                                   depth_nm=depth_nm, scale=scale, phase_deg=phase_deg,
                                   mode=mode, threshold=threshold,
                                   preview=preview, recenter=recenter, noOutput=noOutput,
                                   postAnalyzis=postAnalyzis,
                                   skipChiralityCalculation=skipChiralityCalculation,
                                   skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                   skipFacetInfo=skipFacetInfo,
                                   thresholdCoreSurface=thresholdCoreSurface)

    def systematic_stellate_by(self, NP_B, carve_axis=None, axis_through='vertex',
                               lead='base', depth_nm=None, seat_on_face=True,
                               scale=1.0, phase_deg=0.0, mode='hull', threshold=0.8,
                               preview=False, recenter=True,
                               noOutput=None, postAnalyzis=None,
                               skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
                               skipFacetInfo=None, thresholdCoreSurface=None):
        """Raise a relief on every face of self.NP by adding NP_B (stellation). See utils.csg.systematic_stellate_by."""
        from .utils.csg import systematic_stellate_by
        if noOutput is None: noOutput = self.noOutput
        return systematic_stellate_by(self, NP_B, carve_axis=carve_axis,
                                      axis_through=axis_through, lead=lead,
                                      depth_nm=depth_nm, seat_on_face=seat_on_face,
                                      scale=scale, phase_deg=phase_deg, mode=mode,
                                      threshold=threshold, preview=preview, recenter=recenter,
                                      noOutput=noOutput, postAnalyzis=postAnalyzis,
                                      skipChiralityCalculation=skipChiralityCalculation,
                                      skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                      skipFacetInfo=skipFacetInfo,
                                      thresholdCoreSurface=thresholdCoreSurface)
    
    def copy(self):
        "Create and return a deep copy of any pyNanoMatBuilder system"
        from .utils.core import clone
        return clone(self)

    def effective_diameter(self, structure=None, mode='vertices',
                           method='rms', n_feret=2000):
        """
        Returns the effective diameter of the nanoparticle in Å.
        See utils/prop.effective_diameter for full documentation.
    
        Args:
            structure (str): 'optimized' (default) or 'initial'.
            mode (str): Ellipsoid mode — 'vertices' (default), 'all', or 'planes'.
                Ignored when method='feret' or method='rg'.
            method (str): Diameter computation method. Options:
                'feret', 'rg', 'rms' (default), 'volume', 'arithmetic',
                'surface', 'radius', 'length'.
                See utils/prop.effective_diameter for details.
            n_feret (int): Number of random orientations for method='feret'.
                Default is 2000.
        """
        from .utils.prop import effective_diameter
        if structure is None:
            structure = 'optimized' if self.is_optimized else 'initial'
        return effective_diameter(self, structure, mode, method, n_feret)

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

    def apply_rotation(self, angle_deg, axis, center=None, axis_def='hkl',
                       noOutput=True, postAnalyzis=None,
                       skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
                       skipFacetInfo=None, thresholdCoreSurface=None):
        """Rotate self.NP around an axis through a center point. See utils.geometry.apply_rotation."""
        from .utils.geometry import apply_rotation
        if noOutput is None: noOutput = self.noOutput
        return apply_rotation(self, angle_deg, axis, center=center,
                              axis_def=axis_def, noOutput=noOutput,
                              postAnalyzis=postAnalyzis,
                              skipChiralityCalculation=skipChiralityCalculation,
                              skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                              skipFacetInfo=skipFacetInfo,
                              thresholdCoreSurface=thresholdCoreSurface)
    
    def apply_reflection(self, plane, plane_def='hkl', noOutput=True,
                         postAnalyzis=None, skipChiralityCalculation=None,
                         skipSymmetryAnalyzis=None, skipFacetInfo=None,
                         thresholdCoreSurface=None):
        """Reflect self.NP across a plane. See utils.geometry.apply_reflection."""
        from .utils.geometry import apply_reflection
        if noOutput is None: noOutput = self.noOutput
        return apply_reflection(self, plane, plane_def=plane_def, noOutput=noOutput,
                                postAnalyzis=postAnalyzis,
                                skipChiralityCalculation=skipChiralityCalculation,
                                skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                skipFacetInfo=skipFacetInfo,
                                thresholdCoreSurface=thresholdCoreSurface)
    def apply_translation(self, vector, vector_def='cart', units='nm',
                          noOutput=None, postAnalyzis=None,
                          skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
                          skipFacetInfo=None, thresholdCoreSurface=None):
        """
        Translate self.NP by a vector (Cartesian or crystallographic direction).
        Also updates all truncation planes. See utils.geometry.apply_translation
        for full documentation.
        """
        from .utils.geometry import apply_translation
        if noOutput is None:
            noOutput = self.noOutput
        apply_translation(self, vector, vector_def=vector_def, units=units,
                          noOutput=noOutput, postAnalyzis=postAnalyzis,
                          skipChiralityCalculation=skipChiralityCalculation,
                          skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                          skipFacetInfo=skipFacetInfo,
                          thresholdCoreSurface=thresholdCoreSurface)
        
    def rotate_to_align(self, axis, target_axis=[0,0,1], axis_def='hkl',
                        noOutput=True, postAnalyzis=None,
                        skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
                        skipFacetInfo=None, thresholdCoreSurface=None):
        """Rotate self.NP to align an axis with a target direction. See utils.geometry.rotate_to_align."""
        from .utils.geometry import rotate_to_align
        if noOutput is None: noOutput = self.noOutput
        return rotate_to_align(self, axis, target_axis=target_axis,
                               axis_def=axis_def, noOutput=noOutput,
                               postAnalyzis=postAnalyzis,
                               skipChiralityCalculation=skipChiralityCalculation,
                               skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                               skipFacetInfo=skipFacetInfo,
                               thresholdCoreSurface=thresholdCoreSurface)

    def replicate_by_rotation(self, n_copies, axis, center=None, axis_def='hkl',
                              noOutput=None, postAnalyzis=None,
                              skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
                              skipFacetInfo=None, thresholdCoreSurface=None):
        """Duplicate self.NP n_copies times by rotation and merge. See utils.geometry.replicate_by_rotation."""
        from .utils.geometry import replicate_by_rotation
        if noOutput is None: noOutput = self.noOutput
        return replicate_by_rotation(self, n_copies, axis, center=center,
                                     axis_def=axis_def, noOutput=noOutput,
                                     postAnalyzis=postAnalyzis,
                                     skipChiralityCalculation=skipChiralityCalculation,
                                     skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                     skipFacetInfo=skipFacetInfo,
                                     thresholdCoreSurface=thresholdCoreSurface)
    
    def replicate_by_reflection(self, plane, plane_def='hkl', eps=1e-2, noOutput=None,
                                 postAnalyzis=None, skipChiralityCalculation=None,
                                 skipSymmetryAnalyzis=None, skipFacetInfo=None,
                                 thresholdCoreSurface=None):
        """Duplicate self.NP by reflection across a plane and merge. See utils.geometry.replicate_by_reflection."""
        from .utils.geometry import replicate_by_reflection
        if noOutput is None: noOutput = self.noOutput
        return replicate_by_reflection(self, plane, plane_def=plane_def, eps=eps, noOutput=noOutput,
                                       postAnalyzis=postAnalyzis,
                                       skipChiralityCalculation=skipChiralityCalculation,
                                       skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                                       skipFacetInfo=skipFacetInfo,
                                       thresholdCoreSurface=thresholdCoreSurface)

    def center(self, noOutput=None, postAnalyzis=None,
               skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
               skipFacetInfo=None, thresholdCoreSurface=None):
        """Recenter self.NP on its center of mass. See utils.geometry.center."""
        from .utils.geometry import center
        if noOutput is None:
            noOutput = self.noOutput
        center(self, noOutput=noOutput, postAnalyzis=postAnalyzis,
               skipChiralityCalculation=skipChiralityCalculation,
               skipSymmetryAnalyzis=skipSymmetryAnalyzis,
               skipFacetInfo=skipFacetInfo,
               thresholdCoreSurface=thresholdCoreSurface)

    def remove_duplicates(self, tol=0.1, noOutput=None):
        """
        Remove duplicate atoms from self.NP — atoms closer than tol Å are
        considered duplicates. See utils.geometry.remove_duplicates.
    
        Args:
            tol (float): Distance threshold in Å. Default is 0.1 Å.
            noOutput (bool): If True, suppresses output. Default is self.noOutput.
    
        Returns:
            None. Updates self.NP in place.
        """
        from .utils.geometry import remove_duplicates
        if noOutput is None: noOutput = self.noOutput
        return remove_duplicates(self, tol=tol, noOutput=noOutput)

    def delete(self, elements=None, indices=None, ranges=None, mode='delete',
               recenter=None, noOutput=None, postAnalyzis=None,
               skipChiralityCalculation=None, skipSymmetryAnalyzis=None,
               skipFacetInfo=None, thresholdCoreSurface=None):
        """Remove or keep atoms by element, 1-based index list, and/or Jmol
        range string. See utils.edit.delete for full documentation."""
        from .utils.geometry import delete
        if noOutput is None: noOutput = self.noOutput
        if recenter is None: recenter = True
        return delete(self, elements=elements, indices=indices, ranges=ranges,
                      mode=mode, recenter=recenter, noOutput=noOutput,
                      postAnalyzis=postAnalyzis,
                      skipChiralityCalculation=skipChiralityCalculation,
                      skipSymmetryAnalyzis=skipSymmetryAnalyzis,
                      skipFacetInfo=skipFacetInfo,
                      thresholdCoreSurface=thresholdCoreSurface)

######################################### load external file
    @classmethod
    def from_file(cls, file_path, index=-1, **kwargs):
        """
        Create a pyNMB object from any structural file format supported by ASE 
        (.xyz, .cif, .pdb, etc.) and run the analysis pipeline.
        
        Args:
            file_path (str): Path to the structural file.
            index (int, optional): Frame selector for multi-frame files
                (default -1, i.e. the last frame). Passed through to the
                pyNMB reader.
            **kwargs: Arguments for propPostMake and internal settings.
        """
        from .utils.io import read as pynmb_read
        from pathlib import Path
        # 1. Instantiate the object without calling __init__
        instance = cls.__new__(cls)

        # 2. Universal read via the pyNMB I/O layer
        # Routes to ASE under the hood (auto-detects CIF, PDB, XYZ, etc.)
        # and recovers the pyNMB composition header for .xyz files.
        try:
            instance.NP = pynmb_read(file_path, index=index)
        except Exception as e:
            print(f"Error: could not read the file {file_path}. {e}")
            return None

        # Guard against a multi-frame selection slipping through (e.g. index=':')
        if isinstance(instance.NP, list):
            print(f"Error: {file_path} resolved to multiple frames. "
                  f"Pass a single integer index (got {index!r}).")
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

    # ------------------------------------------------------------------ #
    #  Local-order descriptors (CNP, Steinhardt q_l)                      #
    #  Thin wrappers — on-demand analysis, NOT wired into propPostMake.   #
    # ------------------------------------------------------------------ #
    def common_neighbour_parameter(self, Xnn, noOutput=None,
                                   store=True, is_optimized=None):
        """Per-atom Common Neighbour Parameter (CNP). See utils.prop.common_neighbour_parameter."""
        from .utils.local_descriptors import common_neighbour_parameter
        if noOutput is None:
            noOutput = self.noOutput
        return common_neighbour_parameter(self, Xnn, noOutput=noOutput,
                                          store=store, is_optimized=is_optimized)

    def steinhardt_q(self, Xnn, l=6, noOutput=None, store=True,
                     is_optimized=None):
        """Per-atom Steinhardt bond-orientational order q_l. See utils.prop.steinhardt_q."""
        from .utils.local_descriptors import steinhardt_q
        if noOutput is None:
            noOutput = self.noOutput
        return steinhardt_q(self, Xnn, l=l, noOutput=noOutput,
                            store=store, is_optimized=is_optimized)

    def plot_local_order(self, descriptor='cnp', Xnn=None, l=6,
                         is_optimized=None, save_path=None, color='turbo',
                         bins=50, noOutput=None):
        """Histogram + per-atom colour map of a local-order descriptor. See utils.prop.plot_local_order."""
        from .utils.local_descriptors import plot_local_order
        if noOutput is None:
            noOutput = self.noOutput
        return plot_local_order(self, descriptor=descriptor, Xnn=Xnn, l=l,
                               is_optimized=is_optimized, save_path=save_path,
                               color=color, bins=bins, noOutput=noOutput)

    def defLocalOrderColorForJMol(self, descriptor='cnp', l=6, color='turbo',
                                  is_optimized=None, noOutput=None):
        """Jmol command colouring atoms by a local-order descriptor. See utils.external_pgm.defLocalOrderColorForJMol."""
        from .utils.external_pgm import defLocalOrderColorForJMol
        if noOutput is None:
            noOutput = self.noOutput
        return defLocalOrderColorForJMol(self, descriptor=descriptor, l=l,
                                         color=color, is_optimized=is_optimized,
                                         noOutput=noOutput)

    def local_order_populations(self, descriptor='cnp', l=6, decimals=1,
                                color='turbo', is_optimized=None, noOutput=None):
        """Group atoms by local-order descriptor value into indexed, coloured populations. See utils.prop.local_order_populations."""
        from .utils.local_descriptors import local_order_populations
        if noOutput is None:
            noOutput = self.noOutput
        return local_order_populations(self, descriptor=descriptor, l=l,
                                       decimals=decimals, color=color,
                                       is_optimized=is_optimized, noOutput=noOutput)

    def select_by_local_order(self, indices, descriptor='cnp', l=6,
                              is_optimized=None, noOutput=None):
        """Build self.NP_select from one or more local-order populations by index. See utils.prop.select_by_local_order."""
        from .utils.local_descriptors import select_by_local_order
        if noOutput is None:
            noOutput = self.noOutput
        return select_by_local_order(self, indices, descriptor=descriptor, l=l,
                                     is_optimized=is_optimized, noOutput=noOutput)

    def plot_q4q6_map(self, Xnn=None, is_optimized=None, save_path=None,
                      aggregate=True, decimals=3, sc_domain=False, noOutput=None):
        """Plot the Steinhardt (q4,q6) map: ideal references + per-atom data. See utils.prop.plot_q4q6_map."""
        from .utils.local_descriptors import plot_q4q6_map
        if noOutput is None:
            noOutput = self.noOutput
        return plot_q4q6_map(self, Xnn=Xnn, is_optimized=is_optimized,
                             save_path=save_path, aggregate=aggregate,
                             decimals=decimals, sc_domain=sc_domain, noOutput=noOutput)

    def interface_distance_histogram(self, elemA, elemB, Rnn,
                                     overlap_frac=0.85, contact_frac=1.2,
                                     bins=60, is_optimized=None,
                                     save_path=None, noOutput=None):
        """Histogram of cross-species nearest-neighbour distances at a bimetallic interface. See utils.local_descriptors.interface_distance_histogram."""
        from .utils.csg import interface_distance_histogram
        if noOutput is None:
            noOutput = self.noOutput
        return interface_distance_histogram(self, elemA, elemB, Rnn,
                                            overlap_frac=overlap_frac,
                                            contact_frac=contact_frac,
                                            bins=bins, is_optimized=is_optimized,
                                            save_path=save_path, noOutput=noOutput)