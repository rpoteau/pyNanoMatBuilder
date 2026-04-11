import numpy as np

class pyNMBcore:
    def __init__(self,
                 postAnalyzis: bool = True,
                 aseView: bool = False,
                 thresholdCoreSurface: float = 1.,
                 skipSymmetryAnalyzis: bool = False,
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
            jmolCrystalShape (bool): If True, generates a JMOL script
                for visualization.
            noOutput (bool): If False, prints details about the NP structure.
            calcPropOnly (bool): If False, generates the atomic structure
                of the NP.      
        """
        self.postAnalyzis = postAnalyzis
        self.aseView = aseView
        self.thresholdCoreSurface = thresholdCoreSurface
        self.skipSymmetryAnalyzis = skipSymmetryAnalyzis
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
        
        self.ellipsoid = {} #two keys: "initial structure" or "optimized structure"


    def optimize(self, calculator='EMT', optimizer='QN', fthreshold=0.1):
        from .utils.energy import optimize
        return optimize(self, calculator, optimizer, fthreshold)
    
    def _update_sasview_dims_from_spheres(self, noOutput = None):
        from .utils.prop import _update_sasview_dims_from_spheres
        return _update_sasview_dims_from_spheres(self, noOutput)

    def Inscribed_circumscribed_spheres(self, noOutput=None):
        from .utils.prop import Inscribed_circumscribed_spheres
        if noOutput is None: noOutput = self.noOutput    
        return Inscribed_circumscribed_spheres(self, noOutput)

    def get_ellipsoid_analysis(self, noOutput=None):
        from .utils.prop import get_ellipsoid_analysis
        if noOutput is None: noOutput = self.noOutput    
        return get_ellipsoid_analysis(self, noOutput)

    def peel_by_coordination(self, threshold_peeling=6, Rmax=2.9, noOutput=None):
        from .utils.geometry import peel_by_coordination
        if noOutput is None: noOutput = self.noOutput            
        return peel_by_coordination(self, threshold_peeling, Rmax, noOutput)

    def peel_by_shifted_ellipsoid(self, shift_dist=2.5, noOutput=None):
        from .utils.geometry import peel_by_shifted_ellipsoid
        if noOutput is None: noOutput = self.noOutput            
        return peel_by_shifted_ellipsoid(self, shift_dist, noOutput)

    def _flush_stale_data(self, shape_update=None):
        from .utils.core import _flush_stale_data
        return _flush_stale_data(self, shape_update)

    def propPostMake(self, skipSymmetryAnalyzis=None, thresholdCoreSurface=None, noOutput=None, is_optimized=None):
        from .utils.prop import propPostMake
        if skipSymmetryAnalyzis is None: skipSymmetryAnalyzis = self.skipSymmetryAnalyzis
        if thresholdCoreSurface is None: thresholdCoreSurface = self.thresholdCoreSurface
        if noOutput is None: noOutput = self.noOutput
        if is_optimized is None: is_optimized = self.is_optimized
        return propPostMake(self, skipSymmetryAnalyzis, thresholdCoreSurface, noOutput, is_optimized)

    def plot_npr_triangle(self=None, is_optimized: bool = None, save_path: str = None, 
                      external_data: dict = None, color_by: str = 'Rg', color: str = 'viridis'):
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
        