import numpy as np

class pyNMBcore:
    def __init__(self,
                 postAnalyzis: bool = True,
                 aseView: bool = False,
                 thresholdCoreSurface: float = 1.,
                 skipChiralityCalculation: bool=True,
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
        self.skipChiralityCalculation = skipChiralityCalculation
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
        self.WulffShape = None

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
        instance.thresholdCoreSurface = kwargs.get('thresholdCoreSurface', 1.0)
        instance.aseView = kwargs.get('aseView', False)
        instance.jmolCrystalShape = kwargs.get('jmolCrystalShape', True)
        instance.noOutput = kwargs.get('noOutput', False)
        instance.calcPropOnly = kwargs.get('calcPropOnly', False)
        instance._flush_stale_data(shape_update=f"loaded_{Path(file_path).suffix[1:]}")
        
        # 5. Operational settings
        instance.noOutput = kwargs.get('noOutput', False)
        
        # 6. Property analysis (OPD, MOI, etc.)
        instance.propPostMake(
            skipSymmetryAnalyzis=instance.skipSymmetryAnalyzis,
            thresholdCoreSurface=instance.thresholdCoreSurface,
            noOutput=instance.noOutput,
            is_optimized=False,
        )
        
        return instance