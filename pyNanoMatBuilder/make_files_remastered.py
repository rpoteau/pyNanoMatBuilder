    # External dependencies
import os
import json

from ase.io import read
from ase.io import write

import debyecalculator

import numpy as np
import pandas as pd

# Internal Relative Imports
from .visualID import fg, hl, bg
from . import visualID as vID

from . import crystalNPs as cyNP
from . import platonicNPs as pNP
from . import archimedeanNPs as aNP
from . import catalanNPs as cNP
from . import johnsonNPs as jNP
from . import otherNPs as oNP

from . import utils as pyNMBu
from . import data

################################################ NP Database Base ###


class NPFiles:
    """Base class for all NP file generators.

    Provides common initialization and a unified
    writexyz_generalized method that writes XYZ, CIF,
    script, CSV, and optional NPZ files.

    Subclasses override hook methods to customize
    shape-specific behavior (element source, truncation,
    SASView output, etc.).
    """

    # not working well, to be revised
    UNSUPPORTED_ELEMENTS = frozenset({'TiO2', 'NaCl'}) 

    def __init__(
        self, path, cif_data, sizes,
        max_size=50.0,
        optimize_structure=False,
        instance_Debye=None,
        create_iq=False,
        create_gr=False,
        noOutput=True
    ):
        self.path = path
        self.cif_data = cif_data
        self.loaded_cifs = {}
        self.sizes = sizes
        self.max_size = max_size
        self.optimize_structure = optimize_structure
        self.instance_Debye = instance_Debye
        self.create_iq = create_iq
        self.create_gr = create_gr
        self.noOutput = noOutput

    # ------ Hook methods (override in subclasses) ------

    @staticmethod
    def _safe_radius(instance, attr='radiusCircumscribedSphere'):
        """Return radius value whether it's a property or method."""
        r = getattr(instance, attr)
        return r() if callable(r) else r

    def _get_element(self, instance_class):
        """Return the chemical element symbol."""
        return instance_class.element

    def _get_shape_metadata(self, instance_class):
        """Return dict: shape, wulff, nRot,
        plane_rotated, length, radius."""
        return {
            'shape': instance_class.shape,
            'wulff': False,
            'nRot': None,
            'plane_rotated': None,
            'length': None,
            'radius': None,
        }

    def _get_truncation_info(self, instance_class, shape):
        """Return (truncated, number_truncated_atoms)."""
        return False, None

    def _get_crystallization_type(self, instance_class):
        """Return the crystallization type string."""
        return 'monocrystalline'

    def _write_sasview(
        self, instance_class, id_file, natoms, Z, shape
    ):
        """Write SASView CSV if applicable. No-op by default."""
        pass

    # ------ Unified writexyz ------

    def writexyz_generalized(
        self, structure, instance_class, number
    ):
        """Write XYZ, CIF, script, CSV, and optional NPZ.

        Parameters
        ----------
        structure : str
            Crystal structure name.
        instance_class : object
            NP shape instance.
        number : int
            Index for filenames.
        """
        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                f"Directory '{self.path}' does not exist."
            )

        # Basic metadata
        element = self._get_element(instance_class)
        if structure is None:
            structure = self.crystal_type
        crystalStructure = self.crystal_type
        state = 'crystalline'
        typ = self._get_crystallization_type(instance_class)
        number2 = 0

        # Shape metadata (via hook)
        shape_meta = self._get_shape_metadata(instance_class)
        shape = shape_meta['shape']
        wulff = shape_meta['wulff']
        nRot = shape_meta['nRot']
        plane_rotated = shape_meta['plane_rotated']
        length = shape_meta['length']
        radius = shape_meta['radius']

        # Geometry
        NP = instance_class.NP
        Z = np.mean(NP.get_atomic_numbers())
        element_array = NP.get_chemical_symbols()
        coord = NP.get_positions()
        natoms = len(element_array)

        # Moment of inertia
        MOI = np.round(instance_class.moi, 3)
        MOI_normalized = np.round(instance_class.moisize, 3)

        # Radius (handle property vs method via callable)
        circumradius = round(
            self._safe_radius(instance_class, 'radiusCircumscribedSphere'), 3
        )
        inradius = round(
            self._safe_radius(instance_class, 'radiusInscribedSphere'), 3
        )

        # Truncation (via hook)
        truncated, _ = self._get_truncation_info(
            instance_class, shape
        )

        # Composition
        composition = {}
        for elem in element_array:
            composition[elem] = composition.get(elem, 0) + 1

        # File names
        id_file = (
            f"{element}_{structure}_{shape}"
            f"_{number:07d}_{number2:07d}"
        )
        filename_xyz = f"{id_file}.xyz"
        filename_cif = f"{id_file}.cif"
        filename_script = f"{id_file}.script"
        filename_csv = f"{id_file}_metadata.csv"
        filename_npz_iq = f"{id_file}_iq.npz"
        filename_npz_gr = f"{id_file}_gr.npz"
        # Paths
        path_xyz = os.path.join(self.path, filename_xyz)
        path_cif = os.path.join(self.path, filename_cif)
        path_script = os.path.join(self.path, filename_script)
        path_csv = os.path.join(self.path, filename_csv)
        path_npz_iq = os.path.join(self.path, filename_npz_iq)
        path_npz_gr = os.path.join(self.path, filename_npz_gr)

        if not self.noOutput:
            print(f"[XYZ] Writing file: {path_xyz}")

        # Metadata dictionary
        metadata_dict = {
            "id": id_file,
            "composition": composition,
            "crystal_structure": crystalStructure,
            "shape": shape,
            "MOI_amuA2": list(MOI),
            "MOI_normalized_A2": list(MOI_normalized),
            "circumradius_A": circumradius,
            "inradius_A": inradius,
            "truncation": truncated,
            "number_of_atoms": natoms,
            "wulff": wulff,
            "crystallization": {"state": state, "type": typ},
            "wire_description": {
                "nRot": nRot,
                "plane_rotated": plane_rotated,
                "length": length, "radius": radius
            }
        }

        # Write XYZ file
        with open(path_xyz, 'w') as f:
            f.write(f"{natoms}\n")
            f.write(json.dumps(metadata_dict) + "\n")
            for i in range(natoms):
                f.write(
                    f"{element_array[i]}\t"
                    f"{coord[i,0]:.8f}\t"
                    f"{coord[i,1]:.8f}\t"
                    f"{coord[i,2]:.8f}\n"
                )

        # EMT optimization for certain elements
        if (element in
                ["Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt"]
                and self.optimize_structure):
            print(
                "Elements in the list of ASE, optimizing "
                "the structure using EMT potential"
            )
            pyNMBu.optimizeEMT(
                instance_class.NP,
                pathway=os.path.join(self.path, id_file)
            )
            path_xyz = os.path.join(
                self.path, id_file + '_opt.xyz'
            )
            NP = read(path_xyz)

        # Write CIF file
        if not self.noOutput:
            print(f"[CIF] Writing file: {path_cif}")
        write(path_cif, NP)

        # Write Jmol script
        with open(path_script, 'w') as f:
            f.write(instance_class.jMolCS)

        # Optional I(q)
        if self.create_iq:
            q, iq = self.instance_Debye.iq(path_xyz)
            if len(q) != len(iq) or len(q) == 0:
                print(
                    f"[WARNING] Invalid q/iq for {id_file}"
                )
                return
            idx_split = np.argmin(np.abs(q - 1.6))
            iq_saxs = iq[:idx_split]
            iq_waxs = iq[idx_split:]
            np.savez(
                path_npz_iq, q=q,
                iq_saxs=iq_saxs, iq_waxs=iq_waxs
            )
            if not self.noOutput:
                print(
                    f"[NPZ] Saved I(q) data: {path_npz_iq}"
                )

        # Optional G(r)
        if self.create_gr:
            r, gr = self.instance_Debye.gr(path_xyz)
            if len(r) != len(gr) or len(r) == 0:
                print(
                    f"[WARNING] Invalid g(r) for {id_file}"
                )
                return
            np.savez(path_npz_gr, r=r, gr=gr)
            if not self.noOutput:
                print(
                    f"[NPZ] Saved g(r) data: {path_npz_gr}"
                )

        # Flat metadata for CSV
        metadata_flat = {
            "id": id_file,
            "element": element,
            "crystal_structure": crystalStructure,
            "shape": shape,
            "MOI_1_amuA2": MOI[0],
            "MOI_2_amuA2": MOI[1],
            "MOI_3_amuA2": MOI[2],
            "MOInorm_1_A2": MOI_normalized[0],
            "MOInorm_2_A2": MOI_normalized[1],
            "MOInorm_3_A2": MOI_normalized[2],
            "circumradius_A": circumradius,
            "inradius_A": inradius,
            "truncated": truncated,
            "number_of_atoms": natoms,
            "wulff": wulff,
            "state": state, "type": typ,
            "wire_nRot": nRot,
            "wire_plane_rotated": str(plane_rotated),
            "wire_length": length,
            "wire_radius": radius,
        }

        df = pd.DataFrame([metadata_flat])
        df.to_csv(path_csv, index=False)
        if not self.noOutput:
            print(f"[CSV] Saved metadata: {path_csv}")

        # SASView (shape-specific, via hook)
        self._write_sasview(
            instance_class, id_file, natoms, Z, shape
        )


################################################ Crystal Database ###


class Crystal_Files(NPFiles):
    """Base class for crystal NP file generators.

    Provides CIF loading and crystal-specific hooks
    shared by all crystal NP file generators
    (Wulff, spheres, ellipsoids, wires, etc.).
    """

    def _load_cif_and_get_dhkl(
        self, cif_name, cif_file, direction=None
    ):
        """Load CIF data and compute d_hkl.

        Parameters
        ----------
        cif_name : str
            Name of the CIF compound.
        cif_file : str
            Path to the CIF file.
        direction : list, optional
            Miller indices. Defaults to [0, 0, 1].

        Returns
        -------
        tuple
            (cif_info, d_hkl, structure)
        """
        if direction is None:
            direction = [0, 0, 1]
        self.cif_name = cif_name
        cif_info = pyNMBu.load_cif(
            self, cif_file, self.noOutput
        )
        crystal_system_name = (
            self.ucBL.__class__.__name__
        )
        d_hkl = (
            pyNMBu.interPlanarSpacing(
                direction, self.ucUnitcell,
                crystal_system_name
            ) * 0.1
        )
        structure = (
            cif_name.split()[1]
            if len(cif_name.split()) == 2
            else None
        )
        return cif_info, d_hkl, structure

    def create_shapes(self):
        """Generate NP files. Override in subclasses."""
        raise NotImplementedError(
            "Subclasses must implement create_shapes()."
        )

    # --- CrystalFiles hook overrides ---

    def _get_element(self, instance_class):
        """Return element from CIF name."""
        return self.cif_name.split()[0]

    def _get_shape_metadata(self, instance_class):
        """Detect Wulff/wire/cylinder shapes."""
        shape = instance_class.shape
        wulff = False
        nRot = None
        plane_rotated = None
        length = None
        radius = None

        if "Wulff" in shape:
            shape = shape.split(':')[1].strip()
            wulff = True
            if "hcpwire" in shape:
                nRot = 6
                plane_rotated = (
                    instance_class.surfacesWulff
                )
                shape = "wire6"
                length = instance_class.length
                radius = instance_class.radius

        elif "wire" in shape:
            nRot = instance_class.nRotWire
            shape = f"wire{nRot}"
            plane_rotated = instance_class.refPlaneWire
            length = instance_class.length
            radius = instance_class.radius

        elif "cylinder" in shape:
            length = instance_class.length
            radius = instance_class.radius

        return {
            'shape': shape, 'wulff': wulff,
            'nRot': nRot,
            'plane_rotated': plane_rotated,
            'length': length, 'radius': radius,
        }

    def _get_truncation_info(self, instance_class, shape):
        """Truncation based on shape name."""
        return 'tr' in shape, None

    def _write_sasview(
        self, instance_class, id_file, natoms, Z, shape
    ):
        """Write SASView CSV for crystal NP shapes."""
        if shape == "sphere":
            sv_file = f"{id_file}_metadata_sasview.csv"
            sv_path = os.path.join(self.path, sv_file)
            d = {
                "shape": "sphere",
                "radius_A": round(
                    float(instance_class.sasview_dims[0]), 3
                ),
                "normalization_debye": round(
                    2 * (natoms ** 2) * (Z ** 2), 3
                ),
                "normalization_sasview": round(
                    instance_class.volume * 1e-4, 3
                )
            }
            pd.DataFrame([d]).to_csv(sv_path, index=False)
            if not self.noOutput:
                print(f"[SASVIEW] Saved: {sv_path}")

        if shape == "parallelepiped":
            sv_file = f"{id_file}_metadata_sasview.csv"
            sv_path = os.path.join(self.path, sv_file)
            d = {
                "shape": "parallelepiped",
                "length_a_A": round(
                    float(instance_class.sasview_dims[0]), 3
                ),
                "length_b_A": round(
                    float(instance_class.sasview_dims[1]), 3
                ),
                "length_c_A": round(
                    float(instance_class.sasview_dims[2]), 3
                ),
                "normalization_debye": round(
                    2 * (natoms ** 2) * (Z ** 2), 3
                ),
                "normalization_sasview": round(
                    instance_class.volume * 1e-4, 3
                )
            }
            pd.DataFrame([d]).to_csv(sv_path, index=False)
            if not self.noOutput:
                print(f"[SASVIEW] Saved: {sv_path}")

        if shape == "ellipsoid":
            sv_file = f"{id_file}_metadata_sasview.csv"
            sv_path = os.path.join(self.path, sv_file)
            d = {
                "shape": "ellipsoid",
                "radius_polar_A": round(
                    float(instance_class.sasview_dims[1]), 3
                ),
                "radius_equatorial_A": round(
                    float(instance_class.sasview_dims[0]), 3
                ),
                "normalization_debye": round(
                    2 * (natoms ** 2) * (Z ** 2), 3
                ),
                "normalization_sasview": round(
                    instance_class.volume * 1e-4, 3
                )
            }
            pd.DataFrame([d]).to_csv(sv_path, index=False)
            if not self.noOutput:
                print(f"[SASVIEW] Saved: {sv_path}")

        if shape == "cylinder":
            sv_file = f"{id_file}_metadata_sasview.csv"
            sv_path = os.path.join(self.path, sv_file)
            d = {
                "shape": "cylinder",
                "radius_A": round(
                    float(instance_class.sasview_dims[0]), 3
                ),
                "length_A": round(
                    float(instance_class.sasview_dims[1]), 3
                ),
                "normalization_debye": round(
                    2 * (natoms ** 2) * (Z ** 2), 3
                ),
                "normalization_sasview": round(
                    instance_class.volume * 1e-4, 3
                )
            }
            pd.DataFrame([d]).to_csv(sv_path, index=False)
            if not self.noOutput:
                print(f"[SASVIEW] Saved: {sv_path}")

    # Backward compat: old crystal call sites use this name
    writexyz_generalized_crystals = (
        NPFiles.writexyz_generalized
    )


class PredefinedWulffFiles(Crystal_Files):
    """
    A class for generating XYZ and CIF files of predefined
    Wulff shape nanoparticles (NPs) from a dataset of compounds (CIF dataset).
    This process enables the creation of a well-structured database optimized
    for machine learning applications, ensuring consistency in format 
    and data representation.
    """

    def __init__(self, path, cif_data, wulff_shapes, sizes, form: str = None, max_size: float = 50, optimize_structure: bool = False,
                 instance_Debye=None, create_iq: bool = False, create_gr: bool = False, noOutput: bool = True):
        """
        Initialize the nanoparticle generator.

        Parameters
        ----------
        path : str
            Path to save generated files.
            
        cif_data : pd.DataFrame
            CIF files with structure data.
            
        wulff_shapes : pd.DataFrame
            Table describing Wulff shapes and associated Bravais lattices.
            
        sizes : list
            List of size ratios (e.g., [2, 3] gives sizes [2*dhkl], [3*dhkl]).
            
        form : str, optional
            Specific Wulff shape name to use (e.g., 'fccCubo').
            If None, all valid forms are used.
            
        max_size : float, optional
            Maximum NP size in nm (default: 50).
        
        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential
            for [Al, Cu, Ag, Au, Ni, Pd, Pt].
            
        instance_Debye : object, optional
            Instance for Debye I(q)/g(r) generation.
            
        create_iq : bool, optional
            If True, generate I(q) data.
            
        create_gr : bool, optional
            If True, generate g(r) data.
            
        noOutput : bool, optional
            If True, suppress printed output.
        """

        super().__init__(
            path, cif_data, [[k] for k in sizes],
            max_size, optimize_structure,
            instance_Debye, create_iq, create_gr,
            noOutput
        )
        self.wulff_shapes = wulff_shapes
        self.form = form

        self.create_wulff_shapes()

    def _generate_np(self, cif_info, form, i, d_hkl, index, structure):
        size = [i[0] * d_hkl]

        TestNP = cyNP.Crystal(
            crystal=f'{self.cif_name}',
            userDefCif=cif_info['cif_path'],
            shape=f"Wulff: {form}",
            sizesWulff=size,
            threshold=0.001,
            thresholdCoreSurface=2,
            postAnalyzis=True,
            jmolCrystalShape=True,
            noOutput=True,
            aseView=False,
            skipSymmetryAnalyzis=True
        )

        circumsphere_diameter = NPFiles._safe_radius(TestNP) * 2 * 0.1

        if circumsphere_diameter < self.max_size:
            if not self.noOutput:
                print(f'\033[1m Generating size is {size[0]:.4f} nm (dhkl × {i}).\033[0m')
                print(f'\033[1m Circumscribed sphere diameter = {circumsphere_diameter:.2f} nm\033[0m')
            self.writexyz_generalized_crystals(structure, TestNP, index)
            return True
        else:
            if not self.noOutput:
                print(f'\033[1m Circumscribed sphere diameter = {circumsphere_diameter:.2f} nm > max = {self.max_size} nm → skipped.\033[0m')
            return False

    def create_wulff_shapes(self):
        
        """
        Generate Wulff shapes and their files for all CIF compounds in the dataset.
        """

        for cif_name, cif_file in self.cif_data['cif file'].items():

            if not self.noOutput:
                print(f'\n\033[1m {bg.LIGHTBLUEB} {cif_name.center(50)}\033[0m\n')

            cif_info, d_hkl, structure = self._load_cif_and_get_dhkl(cif_name, cif_file)

            if not self.noOutput:
                print(f'\033[1m d_hkl = {d_hkl:.4f} nm \033[0m')

            if self.form is None:
                for form, row in self.wulff_shapes.iterrows():
                    lattices = [l.strip() for l in row['Bravais lattice'].split(',')]
                    if self.crystal_type in lattices:
                        if not self.noOutput:
                            print(f"\n {bg.LIGHTGREENB} {self.crystal_type} ∈ {lattices} → Wulff form: {form}\033[0m \n")
                        for index, i in enumerate(self.sizes, start=1):
                            success = self._generate_np(cif_info, form, i, d_hkl, index, structure)
                            if not success:
                                break
                    else:
                        if not self.noOutput:
                            print(f" {bg.LIGHTREDB} {self.crystal_type} ∉ {lattices} for Wulff form: {form} \033[0m")

            else:
                lattices = self.wulff_shapes['Bravais lattice'][self.form]
                if self.crystal_type in lattices:
                    if not self.noOutput:
                        print(f"\n {bg.LIGHTGREENB} {self.crystal_type} matches {lattices} → Wulff form: {self.form} \033[0m \n")
                    for index, i in enumerate(self.sizes, start=1):
                        success = self._generate_np(cif_info, self.form, i, d_hkl, index, structure)
                        if not success:
                            break
                else:
                    if not self.noOutput:
                        print(f" {bg.LIGHTREDB} {self.crystal_type} does not match {lattices} for Wulff form: {self.form} \033[0m")


class Crystals_EllipsoidsParallelepipeds(Crystal_Files):
    
    """
    A class for generating XYZ and CIF files of ellipsoidal
    and parallelepipedic nanoparticles (NPs) from a dataset
    of compounds (CIF dataset). This process enables the creation 
    of a well-structured database optimized for machine learning applications,
    ensuring consistency in format and data representation.
    """

    def __init__(self, path, cif_data, sizes, forms, max_size: float = 50, optimize_structure: bool = False,
                 instance_Debye=None, create_iq: bool = False, create_gr: bool = False, noOutput: bool = True):
        
        """
        Initialize the nanoparticle generator.

        Parameters
        ----------
        path : str
            Path to save generated files.
            
        cif_data : pd.DataFrame
            CIF files with structure data.
            
        sizes : list of lists
            List of size multipliers in 3D. Each element is a list
            like [2, 3, 4], producing final size [2*dhkl, 3*dhkl, 4*dhkl].
            
        forms : list of str
            List of forms to generate: ['ellipsoid', 'parallelepiped']
            or a subset.
            
        max_size : float, optional
            Maximum NP size in nm (circumscribed sphere diameter). 
            Default is 50 nm.

        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential for
            [Al, Cu, Ag, Au, Ni, Pd, Pt].
            
        instance_Debye : object, optional
            Instance for Debye I(q)/g(r) generation.
            
        create_iq : bool, optional
            If True, generate I(q) data.
            
        create_gr : bool, optional
            If True, generate g(r) data.
            
        noOutput : bool, optional
            If False, print progress and size details.
        """
        super().__init__(
            path, cif_data, sizes,
            max_size, optimize_structure,
            instance_Debye, create_iq, create_gr,
            noOutput
        )
        self.forms = forms

        self.create_shapes()

    def _generate_np(self, cif_info, form, i, d_hkl, index, structure):

        size = [i[0] * d_hkl, i[1] * d_hkl, i[2] * d_hkl]

        TestNP = cyNP.Crystal(
            crystal=self.cif_name,
            userDefCif=cif_info['cif_path'],
            shape=form,
            size=size,
            buildPPD='abc',
            threshold=0.001,
            thresholdCoreSurface=2,
            postAnalyzis=True,
            jmolCrystalShape=True,
            noOutput=True,
            aseView=False,
            skipSymmetryAnalyzis=True
        )

        circumsphere_diameter = NPFiles._safe_radius(TestNP) * 2 * 0.1  # nm

        if circumsphere_diameter < self.max_size:
            if not self.noOutput:
                print(f'\033[1m Generating size: {size} nm (dhkl × {i})\033[0m')
                print(f'\033[1m Circumscribed sphere diameter: {circumsphere_diameter:.2f} nm\033[0m')
            self.writexyz_generalized_crystals(structure, TestNP, index)
            return True
        else:
            if not self.noOutput:
                print(f'\033[1m Circumscribed sphere diameter {circumsphere_diameter:.2f} nm exceeds max {self.max_size} nm → skipped.\033[0m')
            return False

    def create_shapes(self):
        
        """
        Generate ellipsoids and parallelepipeds and their files
        for all CIF compounds.
        """

        if not self.forms:
            if not self.noOutput:
                print(f"{bg.LIGHTREDB} Please provide a valid list of forms: ['ellipsoid', 'paralllepiped']\033[0m")
            return

        for cif_name, cif_file in self.cif_data['cif file'].items():

            if not self.noOutput:
                print(f'\n\033[1m{bg.LIGHTBLUEB} {cif_name.center(50)} \033[0m\n')

            cif_info, d_hkl, structure = self._load_cif_and_get_dhkl(cif_name, cif_file)

            if not self.noOutput:
                print(f'\033[1m d_hkl = {d_hkl:.4f} nm \033[0m')

            for form in self.forms:
                if form not in ['ellipsoid', 'parallelepiped']:
                    if not self.noOutput:
                        print(f"{bg.LIGHTREDB} Shape '{form}' is not valid. Choose 'ellipsoid' or 'parallelepiped'.\033[0m")
                    continue

                if not self.noOutput:
                    print(f'\n\033[1m{bg.LIGHTBLUEB} Shape: {form} \033[0m\n')

                for index, i in enumerate(self.sizes, start=1):
                    success = self._generate_np(cif_info, form, i, d_hkl, index, structure)
                    if not success:
                        break


    
class Crystals_spheres(Crystal_Files):
    
    """
    A class for generating XYZ and CIF files of spherical nanoparticles (NPs)
    from a dataset of compounds (CIF dataset). This process enables the
    creation of a standardized database optimized for machine learning
    applications, ensuring consistency in format, size, and structure
    representation.
    """

    def __init__(self, path, cif_data, sizes, max_size: float = 50,
                 optimize_structure: bool = False, instance_Debye=None,
                 create_iq: bool = False, create_gr: bool = False,
                 noOutput: bool = True):
        
        """
        Initialize the Crystals_spheres class with CIF data and sphere size settings.

        Parameters 
        ----------
            path : str
                Directory where the generated XYZ and CIF files will be saved.
            
            cif_data : DataFrame
                DataFrame containing CIF files under the column 'cif file'.
            
            sizes : list of list
                Array of size scaling factors, e.g., [[2], [3]] results
                in sphere diameters [2×d_hkl], [3×d_hkl], etc.
            
            max_size : float, optional
                Maximum allowed nanoparticle size (diameter of the 
                circumscribed sphere), in nm. Default is 50.

            optimize_structure : bool, optional
                If True, optimize the structure using EMT potential 
                for [Al, Cu, Ag, Au, Ni, Pd, Pt].
            
            instance_Debye : object, optional
                Instance of a Debye class for computing scattering curves
                (I(q), g(r)).
            
            create_iq : bool, optional
                If True, generate I(q) CSV files along with metadata.
            
            create_gr : bool, optional
                If True, generate g(r) CSV files along with metadata.
            
            noOutput : bool, optional
                If False, display progress and detailed output 
                during generation.
        """
        super().__init__(
            path, cif_data, sizes,
            max_size, optimize_structure,
            instance_Debye, create_iq, create_gr,
            noOutput
        )

        self.create_shapes()

    def create_shapes(self):
        
        """
        Generate spherical nanoparticles and their corresponding files
        from the CIF dataset.
        """
        
        for cif_name, cif_file in self.cif_data['cif file'].items():

            if not self.noOutput:
                print(f'\n\033[1m{bg.LIGHTBLUEB}{cif_name.center(50)}\033[0m\n')

            cif_info, d_hkl, structure = self._load_cif_and_get_dhkl(cif_name, cif_file)

            if not self.noOutput:
                print(f'\033[1m d_hkl = {d_hkl:.3f} nm \033[0m')

            index = 0
            for i in self.sizes:
                size_nm = [i[0] * d_hkl]
                index += 1

                TestNP = cyNP.Crystal(
                    crystal=self.cif_name,
                    userDefCif=cif_info['cif_path'],
                    shape='sphere',
                    size=size_nm,
                    threshold=0.001,
                    thresholdCoreSurface=2,
                    postAnalyzis=True,
                    jmolCrystalShape=True,
                    noOutput=True,
                    aseView=False,
                    skipSymmetryAnalyzis=True
                )

                circumsphere_diameter = NPFiles._safe_radius(TestNP) * 2 * 0.1  # in nm

                if circumsphere_diameter < self.max_size:
                    if not self.noOutput:
                        print(f'\033[1m Generating sphere of size {size_nm[0]:.2f} nm (dhkl × {i[0]}) \033[0m')
                        print(f'\033[1m Circumscribed diameter = {circumsphere_diameter:.2f} nm \033[0m')

                    self.writexyz_generalized_crystals(structure, TestNP, index)
                else:
                    if not self.noOutput:
                        print(f'\033[1m Skipped: circumscribed diameter = {circumsphere_diameter:.2f} nm exceeds max size = {self.max_size} nm \033[0m')
                    break
   
                       
class Crystals_wires(Crystal_Files):
    
    """
    A class for generating XYZ and CIF files of nanowires from a dataset
    of compounds (CIF dataset).
    This process enables the creation of a well-structured database optimized
    for machine learning applications, ensuring consistency in format and
    data representation.

    Attributes:
    -----------
    cif_data (DataFrame):
        DataFrame containing the CIF files of different compounds.
        
    sizes (array-like):
        Array containing the [diameter multiplier, length multiplier] to be
        applied to dhkl for the wire size.
        Example: [[2,4], [3,6]] will generate wires of diameter 
        2*dhkl and length 4*dhkl, etc.

    optimize_structure (bool):
        Whether to optimize the structure using EMT potential
        for [Al, Cu, Ag, Au, Ni, Pd, Pt].

    create_iq (bool):
        Whether to generate CSV files containing I(q) data and metadata.
        
    create_gr (bool):
        Whether to generate CSV files containing g(r) data and metadata.
        
    crystal_type (str):
        Inferred from the CIF; determines how the wire direction
        and plane are chosen.

    Methods:
    --------
    create_shapes(noOutput, max_size, path, create_iq, create_gr, instance_Debye):
        Generate wires and their files (XYZ/CIF and optional I(q)/g(r))
        for all compounds in the dataset.

    Notes:
    ------
    - The direction and growth plane of the wire depend on the crystal type:
        - For cubic:    directionWire = [1,1,1], refPlaneWire = [0,1,-1]
        - For hcp:      directionWire = [0,0,1], refPlaneWire = [1,0,0]
    - For each wire, two cross-section shapes are tried: square (nRot=4) 
      and hexagonal (nRot=6).
    - Only wires with a circumscribed diameter smaller than `max_size`
      (in nm) are kept.
    - The unit used throughout is nanometers (nm).
    """

    def __init__(self, path, cif_data, sizes, max_size: float = 50, optimize_structure: bool = False, instance_Debye=None,
                 create_iq: bool = False, create_gr: bool = False, noOutput: bool = True):
        super().__init__(path, cif_data, sizes, max_size, optimize_structure, instance_Debye, create_iq, create_gr, noOutput)
        self.create_shapes()

    def create_shapes(self):
        import math
        for cif_name, cif_file in self.cif_data['cif file'].items():
            
            element = cif_name.split()[0]  # Extract chemical element from CIF name
            
            if element not in self.UNSUPPORTED_ELEMENTS:
                if not self.noOutput:
                    print(f'\n\033[1m {bg.LIGHTBLUEB} {cif_name.center(50)}\033[0m\n')
    
                cif_info, _, structure = self._load_cif_and_get_dhkl(cif_name, cif_file)
                crystal_system_name = self.ucBL.__class__.__name__
    
                if self.crystal_type in ['fcc', 'bcc', 'cubic']:
                    directionWire = [1, 1, 1]
                    refPlaneWire = [0, 1, -1]
                elif self.crystal_type == 'hcp':
                    directionWire = [0, 0, 1]
                    refPlaneWire = [1, 0, 0]
    
                if not self.noOutput:
                    print(f'refPlaneWire={refPlaneWire} and directionWire={directionWire}\n')
    
                if self.crystal_type == 'fcc':
                    d_hkl_diameter = pyNMBu.interPlanarSpacing(refPlaneWire, 
                                                               self.ucUnitcell, 
                                                               crystal_system_name) * 0.1
                    d_hkl_length = 2 * pyNMBu.interPlanarSpacing(directionWire, 
                                                                 self.ucUnitcell, 
                                                                 crystal_system_name) * 0.1
                else:
                    d_hkl_length = pyNMBu.interPlanarSpacing(directionWire, 
                                                             self.ucUnitcell, 
                                                             crystal_system_name) * 0.1
                    d_hkl_diameter = pyNMBu.interPlanarSpacing(refPlaneWire,
                                                                self.ucUnitcell, 
                                                                crystal_system_name) * 0.1
    
                if not self.noOutput:
                    print(f'd_hkl_length is {d_hkl_length} nm')
                    print(f'd_hkl_diameter is {d_hkl_diameter} nm')
    
                for nRot in [4, 6]:
                    if not self.noOutput:
                        print(f'\n\033[1m nRot (cross section of the wire) ={nRot}\033[0m\n')
                    index = 0
                    for i in self.sizes:
                        size = [i[0] * d_hkl_diameter, i[1] * d_hkl_length]
                        index += 1
                        TestNP = cyNP.Crystal(
                            crystal=f'{self.cif_name}',
                            userDefCif=cif_info['cif_path'],
                            shape='wire',
                            size=size,
                            directionWire=directionWire,
                            nRotWire=nRot,
                            refPlaneWire=refPlaneWire,
                            threshold=0.001,
                            thresholdCoreSurface=1,
                            postAnalyzis=True,
                            jmolCrystalShape=True,
                            noOutput=True,
                            aseView=False,
                            skipSymmetryAnalyzis=True
                        )
    
                        circumsphere_diameter = NPFiles._safe_radius(TestNP) * 2 * 0.1
                        if circumsphere_diameter < self.max_size:
                            if not self.noOutput:
                                print(f'\033[1m Generating size is {size} nm and is equal to dhkl multiplied by {i}.\033[0m ')
                                print(f'\033[1m  Circumscribed sphere diameter ={circumsphere_diameter} nm \033[0m')
                            self.writexyz_generalized_crystals(structure, TestNP, index)
                        else:
                            if not self.noOutput:
                                print(f'\033[1m Circumscribed sphere diameter ={circumsphere_diameter} nm is greater than the maximum diameter defined={self.max_size} nm \033[0m')
                            break

                   
                

                    
                 
################################################ Platonic Database ##################################################################################################                    
      
class Platonic_Files(NPFiles):
    
    """
    A class for generating XYZ and CIF files of Platonic nanoparticles (NPs) 
    from a dataset of compounds (CIF dataset). This process enables
    the creation  of a well-structured database optimized for machine
    learning applications, ensuring consistency in format and 
    data representation.
    """
    def __init__(self, path, cif_data, sizes, form: str = None, max_size: float = 50, optimize_structure: bool = False, instance_Debye=None, create_iq: bool = False, create_gr: bool = False, noOutput: bool = True):
        """
        Initialize the class with CIF data, shapes and sizes.

        Parameters
        ----------
        path : str
            Path to save generated files.
            
        cif_data : pd.DataFrame
            CIF files with structure data.
            
        sizes : list
            Array of the number of bonds per edges, ex: [[1],[2],[3]].
            
        form : str, optional
            If None, all the forms are created; otherwise, a specific form is selected.
            
        max_size : float, optional
            Maximum NP size in nm (default: 50 nm).
            
        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential for [Al, Cu, Ag, Au, Ni, Pd, Pt].
            
        instance_Debye : object, optional
            Instance for Debye I(q)/g(r) generation.
            
        create_iq : bool, optional
            If True, generate I(q) data.
            
        create_gr : bool, optional
            If True, generate g(r) data.
            
        noOutput : bool, optional
            If True, suppress printed output.

            Methods:
        create_platonic(noOutput, path): Generates Platonic nanoparticles and saves their files.
        """
        
        super().__init__(
            path, cif_data, sizes,
            max_size=max_size,
            optimize_structure=optimize_structure,
            instance_Debye=instance_Debye,
            create_iq=create_iq,
            create_gr=create_gr,
            noOutput=noOutput
        )
        self.form = form
        self.create_platonic(noOutput, max_size, path, create_iq, create_gr, instance_Debye)

    def _write_sasview(
        self, instance_class, id_file, natoms, Z, shape
    ):
        """Write SASView CSV for regfccOh shapes."""
        if shape == "regfccOh":
            sasview_filename = f"{id_file}_metadata_sasview.csv"
            sasview_path = os.path.join(
                self.path, sasview_filename
            )
            volume = (
                instance_class.volume()
                if callable(instance_class.volume)
                else instance_class.volume
            )
            metadata_flat_sasview = {
                "shape": "Oh",
                "length_a_A": round(
                    float(instance_class.sasview_dims[0]), 3
                ),
                "truncature_ratio": round(
                    float(instance_class.sasview_dims[1]), 3
                ),
                "normalization_factor_for_Debye_intensity"
                "_2/(N_atoms**2)*(Z**2)": round(
                    2 * (natoms ** 2) * (Z ** 2), 3
                ),
                "normalization_factor_for_SasView"
                "_intensity_(V*10**(-4))": round(
                    volume * 10 ** (-4), 3
                ),
            }
            df_sasview = pd.DataFrame(
                [metadata_flat_sasview]
            )
            df_sasview.to_csv(sasview_path, index=False)
            if not self.noOutput:
                print(
                    f"[SASVIEW] Saved SASView parameters: "
                    f"{sasview_path}"
                )

    def create_platonic(self, noOutput, max_size, path, create_iq, create_gr, instance_Debye):
        
        """
        Generate platonic NPs and their files from cif data.
        """
        plat_dict = {
            'regfccOh': 'nOrder',
            'cube': 'nOrder',
            'regIco': 'nShell',
            'regfccTd': 'nLayer',
            'regDD': 'nShell'
        }
    
        form_fcc = ['regfccOh', 'regfccTd', 'regDD', 'regIco', 'cube']
        form_bcc = ['regDD', 'regIco', 'cube']
        form_other = ['regDD', 'regIco']
    
        def create_and_write_NPs(element, dist, form_list, structure):
            for form in form_list:
                n_size = plat_dict[form]
                cls = getattr(pNP, form)
                if not noOutput:
                    print(f" {bg.LIGHTGREENB} xyz/cif files can be created for {self.cif_name} of Bravais lattice={self.crystal_type} for the form {form}. \033[0m")
                index = 0
                for i in self.sizes:
                    index += 1
                    if not noOutput:
                        print(f'{bg.LIGHTBLUEB} Number of bonds is {i}\033[0m')
                    kwargs = {
                        'element': element,
                        'Rnn': dist,
                        n_size: i + 1 if n_size == 'nLayer' else i,
                        'shape': form,
                        'postAnalyzis': True,
                        'aseView': False,
                        'thresholdCoreSurface': 1,
                        'skipSymmetryAnalyzis': True,
                        'noOutput': True
                    }
                    TestNP = cls(**kwargs)
                    circumsphere_diameter = NPFiles._safe_radius(TestNP) * 2 * 0.1
                    if circumsphere_diameter < max_size:
                        if not noOutput:
                            print(f'{bg.LIGHTBLUEB}Circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                        self.writexyz_generalized(structure, TestNP, index)
                    else:
                        if not noOutput :
                            print(f'{bg.LIGHTBLUEB} Size greater than the circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                        break
    
        for cif_name, cif_file in self.cif_data['cif file'].items():
            element = cif_name.split()[0]
            if element in self.UNSUPPORTED_ELEMENTS:
                continue
    
            self.cif_name = cif_name
            structure = self.cif_name.split()[1] if len(self.cif_name.split()) == 2 else None
    
            if not noOutput:
                print(f'\n\033[1m{bg.LIGHTBLUEB}{cif_name.center(50)}\033[0m\n')
    
            cif_info = pyNMBu.load_cif(self, cif_file, noOutput)
    
            if self.crystal_type == 'fcc':
                forms = form_fcc
            elif self.crystal_type == 'bcc':
                forms = form_bcc
            elif self.crystal_type == 'hcp':
                forms = form_other
            else:
                forms = None
                if not noOutput:
                    print(f" {bg.LIGHTREDB} xyz/cif files can't be created for {self.cif_name} because the interatomic distance is unknown.\033[0m")
                continue
    
            if not noOutput:
                print(f" \033[1m The forms possible of {self.cif_name}({self.crystal_type}) are {forms}.\033[0m")
    
            dist = pyNMBu.FindInterAtomicDist(self)
    
            if self.form is None:
                if forms is not None:
                    create_and_write_NPs(element, dist, forms, structure)
            else:
                if self.form in forms:
                    create_and_write_NPs(element, dist, [self.form], structure)


################################################ Archimedean Database ################################################################################################## 
class ArchimedeansFiles(NPFiles):
    
    """
    Generate an XYZ, CIF and script files of a archimedean nanoparticle's (NP)
    structure along with its associated metadata. Possibility to also create
    NPZ files containing I(q) and G(r) data. A CSV file is created containing
    all the metadata and the ID of the created files.
    """

    def __init__(self, path, cif_data, sizes, mode='fccCubo', max_size: float = 50,
                 instance_Debye=None, create_iq: bool = False, create_gr: bool = False, optimize_structure: bool = False, noOutput: bool = True):
        """
        Initialize the class with CIF data, shapes and sizes.

        Parameters
        ----------
        path : str
            Path to save generated files.
            
        cif_data : pd.DataFrame
            CIF files with structure data.
            
        sizes : list
            Array of the number of shells.
            
        form : str, optional
            If None, all the forms are created; otherwise,
            a specific form is selected.
            
        max_size : float, optional
            Maximum NP size in nm (default: 50 nm).
            
        instance_Debye : object, optional
            Instance for Debye I(q)/g(r) generation.
            
        create_iq : bool, optional
            If True, generate I(q) data.
            
        create_gr : bool, optional
            If True, generate g(r) data.
        
        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential
            for [Al, Cu, Ag, Au, Ni, Pd, Pt].
            
        noOutput : bool, optional
            If True, suppress printed output.

            Methods:
        create_archimedean(noOutput, max_size, path, create_iq, create_gr, instance_Debye):
            Generate archimedean NPs and their files from cif data.
        """

        super().__init__(
            path, cif_data, sizes,
            max_size=max_size,
            optimize_structure=optimize_structure,
            instance_Debye=instance_Debye,
            create_iq=create_iq,
            create_gr=create_gr,
            noOutput=noOutput
        )
        self.mode = mode

        if mode not in self.ARCHIMEDEAN_CONFIG:
            raise ValueError(
                f"Unsupported mode: {mode}. "
                f"Choose from {list(self.ARCHIMEDEAN_CONFIG)}."
            )
        self._create_archimedean(mode)


    def _get_truncation_info(self, instance_class, shape):
        """Return truncation status based on shape."""
        if shape in ('fccTrOh', 'fccTrCube', 'fccTrTd'):
            return True, None
        return False, None

    def _write_sasview(
        self, instance_class, id_file, natoms, Z, shape
    ):
        """Write SASView CSV for fccTrOh shapes."""
        if shape == "fccTrOh":
            sasview_filename = f"{id_file}_metadata_sasview.csv"
            sasview_path = os.path.join(
                self.path, sasview_filename
            )
            metadata_flat_sasview = {
                "shape": "Oh",
                "length_a_A": round(
                    float(instance_class.sasview_dims[0]), 3
                ),
                "truncature_ratio": round(
                    float(instance_class.sasview_dims[1]), 3
                ),
                "normalization_debye": round(
                    2 * (natoms ** 2) * (Z ** 2), 3
                ),
                "normalization_sasview": round(
                    float(instance_class.volume_diff_trunc)
                    * 10 ** (-4), 3
                ),
            }
            df_sasview = pd.DataFrame(
                [metadata_flat_sasview]
            )
            df_sasview.to_csv(sasview_path, index=False)
            if not self.noOutput:
                print(
                    f"[SASVIEW] Saved SASView parameters: "
                    f"{sasview_path}"
                )

    # ---------- Archimedean shape configuration ----------
    # Maps mode → (constructor, size_kwarg_name, size_description)
    ARCHIMEDEAN_CONFIG = {
        'fccCubo': {
            'cls': aNP.fccCubo,
            'size_param': 'nShell',
            'size_label': 'Number of shells is {i} (= number of bonds per edge)',
        },
        'fccTrOh': {
            'cls': aNP.fccTrOh,
            'size_param': 'nOrder',
            'size_label': 'nOrder is {i} (= number of octahedrons imbricated '
                          '= number of bonds per edge before truncation)',
        },
        'fccTrTd': {
            'cls': aNP.fccTrTd,
            'size_param': 'nLayer',
            'size_label': 'Number of layers is {i}.',
        },
        'fccTrCube': {
            'cls': aNP.fccTrCube,
            'size_param': 'nOrder',
            'size_label': 'nOrder is {i} (= number of cubes imbricated '
                          '= number of bonds per edge before truncation)',
        },
    }

    def _create_archimedean(self, mode):
        """
        Generate Archimedean NPs and their files from CIF data.

        Parameters
        ----------
        mode : str
            One of the keys in ``ARCHIMEDEAN_CONFIG``
            ('fccCubo', 'fccTrOh', 'fccTrTd', 'fccTrCube').
        """
        config = self.ARCHIMEDEAN_CONFIG[mode]
        np_cls = config['cls']
        size_param = config['size_param']
        size_label = config['size_label']

        for cif_name, cif_file in self.cif_data['cif file'].items():
            element = cif_name.split()[0]

            if element in self.UNSUPPORTED_ELEMENTS:
                continue

            self.cif_name = cif_name
            structure = (
                self.cif_name.split()[1]
                if len(self.cif_name.split()) == 2
                else None
            )

            if not self.noOutput:
                print(
                    f'\n\033[1m{cif_name.center(50)}\033[0m\n'
                )

            pyNMBu.load_cif(self, cif_file, self.noOutput)

            if self.crystal_type != 'fcc':
                if not self.noOutput:
                    print(
                        f" {bg.LIGHTREDB} xyz/cif files can't be "
                        f"created for {self.cif_name} because the "
                        f"crystal type isn't fcc.\033[0m"
                    )
                continue

            dist = pyNMBu.FindInterAtomicDist(self)

            if not self.noOutput:
                print(
                    f" {bg.LIGHTGREENB} xyz/cif files can be "
                    f"created for {self.cif_name} of Bravais "
                    f"lattice={self.crystal_type} for the form "
                    f"{mode}. \033[0m"
                )

            for index, i in enumerate(self.sizes, start=1):
                if not self.noOutput:
                    print(
                        f'{bg.LIGHTBLUEB} '
                        f'{size_label.format(i=i)}'
                    )

                TestNP = np_cls(
                    element=element,
                    Rnn=dist,
                    **{size_param: i},
                    postAnalyzis=True,
                    aseView=False,
                    thresholdCoreSurface=1,
                    skipSymmetryAnalyzis=True,
                    noOutput=True,
                )

                circumsphere_diameter = (
                    NPFiles._safe_radius(TestNP) * 2 * 0.1
                )

                if circumsphere_diameter < self.max_size:
                    if not self.noOutput:
                        print(
                            f'{bg.LIGHTBLUEB}Circumscribed sphere '
                            f'diameter ={circumsphere_diameter}'
                            f'\033[0m'
                        )
                    self.writexyz_generalized(
                        structure, TestNP, index
                    )
                else:
                    if not self.noOutput:
                        print(
                            f'{bg.LIGHTBLUEB} Size greater than '
                            f'the circumscribed sphere diameter '
                            f'={circumsphere_diameter}\033[0m'
                        )
                    break

################################################ Catalan Database ################################################################################################## 

class Catalan_Files(NPFiles):
    
    """
    A class for generating XYZ and CIF files of rhombic dodecahedron (bccrDD) 
    and dihedral rhombic dodecahedron (fccdrDD) Catalan nanoparticles (NPs) 
    from a dataset of compounds (CIF dataset). This process enables the
    creation of a well-structured database optimized for machine learning
    applications, ensuring consistency in format and data representation.
    """
    def __init__(self, path, cif_data, sizes, max_size: float = 50, instance_Debye=None, create_iq: bool = False, create_gr: bool = False, optimize_structure: bool = False, noOutput: bool = True):
        """
        Initialize the nanoparticle generator.

        Parameters
        ----------
        path : str
            Path to save generated files.
            
        cif_data : pd.DataFrame
            CIF files with structure data.
            
        sizes : list
            Array of the number of shells, ex: [2,3,4]
            
        max_size : float, optional
            Maximum NP size in nm (default: 50).
            
        instance_Debye : object, optional
            Instance for Debye I(q)/g(r) generation.
            
        create_iq : bool, optional
            If True, generate I(q) data.
            
        create_gr : bool, optional
            If True, generate g(r) data.
            
        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential
              for [Al, Cu, Ag, Au, Ni, Pd, Pt].
            
        noOutput : bool, optional
            If True, suppress printed output.

        Methods:
        --------
            create_catalan(noOutput, path): Generates Catalan nanoparticles
            and saves their files.
    
        """
        super().__init__(
            path, cif_data, sizes,
            max_size=max_size,
            optimize_structure=optimize_structure,
            instance_Debye=instance_Debye,
            create_iq=create_iq,
            create_gr=create_gr,
            noOutput=noOutput
        )
        self.create_catalan(noOutput, max_size, path, create_iq, create_gr, instance_Debye)



    def create_catalan(self, noOutput, max_size, path, create_iq, create_gr, instance_Debye):
        
        """
        Generate catalan (polyhedral) NPs and their files from CIF data.  
        
        Parameters
        ----------
        noOutput : bool, optional
            If True, suppress printed output.
            
        max_size : float, optional
            Maximum NP size in nm (default: 50).
            
        path : str
            Path to save generated files.
            
        create_iq : bool, optional
            If True, generate I(q) data.
            
        create_gr : bool, optional
            If True, generate g(r) data.
            
        instance_Debye : object, optional
            Instance for Debye I(q)/g(r) generation.
        
        """
        
        crystal_classes = {
            'fcc': cNP.fccdrDD,
            'bcc': cNP.bccrDD
        }
    
        for cif_name, cif_file in self.cif_data['cif file'].items():
            element = cif_name.split()[0]
    
            if element in self.UNSUPPORTED_ELEMENTS:
                continue
    
            self.cif_name = cif_name
    
            structure_parts = cif_name.split()
            structure = structure_parts[1] if len(structure_parts) == 2 else None
    
            if not noOutput:
                print(f'\n\033[1m{cif_name.center(50)}\033[0m\n')
    
            pyNMBu.load_cif(self, cif_file, noOutput)
            crystal_type = self.crystal_type
    
            if crystal_type not in crystal_classes:
                if not noOutput:
                    print(f"{bg.LIGHTREDB} xyz/cif files can't be created for {self.cif_name} (unsupported crystal type: {crystal_type}).\033[0m")
                continue
    
            dist = pyNMBu.FindInterAtomicDist(self)
            shape_class = crystal_classes[crystal_type]
    
            if not noOutput:
                print(f"{bg.LIGHTGREENB} xyz/cif files can be created for {self.cif_name} with Bravais lattice = {crystal_type}. \033[0m")
    
            for index, i in enumerate(self.sizes, start=1):
                if not noOutput:
                    print(f'{bg.LIGHTBLUEB} Number of atoms per edge is {i + 1}')
    
                TestNP = shape_class(
                    element=element,
                    Rnn=dist,
                    nShell=i,
                    postAnalyzis=True,
                    aseView=False,
                    thresholdCoreSurface=1,
                    skipSymmetryAnalyzis=True,
                    noOutput=True
                )
    
                circ_diam = NPFiles._safe_radius(TestNP) * 2 * 0.1  # nm

                if circ_diam < max_size:
                    if not noOutput:
                        print(f'{bg.LIGHTBLUEB}Circumscribed sphere diameter = {circ_diam:.3f} nm\033[0m')
                    self.writexyz_generalized(structure, TestNP, index)
                else:
                    if not noOutput :
                        print(f'{bg.LIGHTBLUEB} Size greater than the circumscribed sphere diameter ={circ_diam:.3f}\033[0m')
                    break
      
       

################################################ Johnson Database ################################################################################################## 
class Johnson_Files(NPFiles):
    
    """
    A unified class for generating XYZ and CIF files of Johnson
    nanoparticles (NPs), including:
    - 'fcctbp': fcc trigonal bipyramid shapes
    - 'epbpyM': pentagonal bipyramids and elongated bipyramids

    This class supports structured database generation for ML applications.
    """

    def __init__(self, path, cif_data, sizes, mode='fcctbp', max_size: float = 50,
                 instance_Debye=None, create_iq: bool = False, create_gr: bool = False, optimize_structure: bool = False, noOutput: bool = True):
        """
        Parameters 
        ----------
        path : str
            Output directory for XYZ/CIF files
            
        cif_data : DataFrame
            CIF dataset
        sizes : array-like
            For 'fcctbp' → list of int (bond layers), see conditions on the example notebook.
            For 'epbpyM' → ndarray with shape (N,3) for [sizeP, sizeE, Marks],
            see conditions on the example notebook.
            
        mode : str
            Either 'fcctbp' or 'epbpyM'
            
        max_size : float
            Max circumscribed sphere diameter in nm (default 50)
            
        instance_Debye : object, optional 
            Debye calculator instance
            
        create_iq : bool 
            I(q) data in a NPZ file : q, iq_saxs, iq_waxs
            
        create_gr : bool 
            G(r) data in a NPZ file : r, gr
            
        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential for [Al, Cu, Ag, Au, Ni, Pd, Pt].
            
        noOutput : bool
            If False, print file-writing feedback.
            
        Notes:
        ------
        Dimensions are in Å, MOI are in amu.Å², normalized MOI in Å².
        Filenames follow the format: Element_structure_shape_number_0000000.xyz

        """
        super().__init__(
            path, cif_data, sizes,
            max_size=max_size,
            optimize_structure=optimize_structure,
            instance_Debye=instance_Debye,
            create_iq=create_iq,
            create_gr=create_gr,
            noOutput=noOutput
        )
        self.mode = mode

        if mode == 'fcctbp':
            self.create_johnson_fcctbp(noOutput, sizes, max_size, path, create_iq, create_gr , instance_Debye)
        elif mode == 'epbpyM':
            self.sizeP = np.unique(sizes[:, 0])
            self.sizeE = np.unique(sizes[:, 1])
            self.Marks = np.unique(sizes[:, 2])
            self.create_johnson_epbpyM(noOutput, sizes, max_size, path, create_iq, create_gr, instance_Debye)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose 'fcctbp' or 'epbpyM'.")


    def _get_truncation_info(self, instance_class, shape):
        """Return truncation info for Johnson shapes."""
        if shape == 'epbpyM':
            if instance_class.Marks != 0:
                return True, int(instance_class.Marks)
        return False, None

    def _get_crystallization_type(self, instance_class):
        """Return crystallization type for Johnson shapes."""
        if instance_class.shape == 'epbpyM':
            return 'twinned'
        return 'monocrystalline'

    def create_johnson_fcctbp(self, noOutput, sizes, max_size, path, create_iq, create_gr, instance_Debye):
        
        """
        Generate johnson fcctbp NPs and their files from cif data.
        
        Parameters 
        ----------
        noOutput : bool 
            If False, prints details about the process.
            
        sizes : list
            list of int (bond layers), see conditions on the example notebook
            
        max_size (float, optional) : 
            Maximal size for the NPs, equals to the diameter of the circumscribed
            sphere, equals 50 nm by default.
            
        path : str
            The directory where files will be written.       
            
        create_iq : bool 
            I(q) data in a NPZ file : q, iq_saxs, iq_waxs
            
        create_gr : bool 
            G(r) data in a NPZ file : r, gr
            
        noOutput : bool
            If False, print file-writing feedback.
            
        instance_Debye (class instance, optional): 
            Instance of the Debye calculator module in order to compute I,q
            and G,r
    
        Note
        -----
        The function uses the johnson module "jNP" and the fcctbp class.
        """
        
        for cif_name, cif_file in self.cif_data['cif file'].items():
            element = cif_name.split()[0]
            if element in self.UNSUPPORTED_ELEMENTS:
                continue

            self.cif_name = cif_name
            structure = self.cif_name.split()[1] if len(self.cif_name.split()) == 2 else None

            if not noOutput:
                print(f'\n\033[1m{cif_name.center(50)}\033[0m\n')

            cif_info = pyNMBu.load_cif(self, cif_file, noOutput)

            if self.crystal_type == 'fcc':
                dist = pyNMBu.FindInterAtomicDist(self)
                index = 0

                if not noOutput:
                    print(f"{bg.LIGHTGREENB} xyz/cif files can be created for {self.cif_name} of Bravais lattice={self.crystal_type}. \033[0m")

                for i in self.sizes:
                    index += 1
                    if not noOutput:
                        print(f'Number of bonds is {i}')
                    TestNP = jNP.fcctbp(
                        element=element,
                        Rnn=dist,
                        nLayerTd=i,
                        postAnalyzis=True,
                        aseView=False,
                        thresholdCoreSurface=1,
                        skipSymmetryAnalyzis=True,
                        noOutput=True
                    )
                    circumsphere_diameter = NPFiles._safe_radius(TestNP) * 2 * 0.1
                    if circumsphere_diameter < max_size:
                        if not noOutput:
                            print(f'{bg.LIGHTBLUEB}Circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                        self.writexyz_generalized(structure, TestNP, index)
                    else:
                        if not noOutput:
                            print(f'{bg.LIGHTBLUEB} Size greater than the circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                        break
            else:
                if not noOutput:
                    print(f"{bg.LIGHTREDB} xyz/cif files can't be created for {self.cif_name} because the crystal type isn't fcc.\033[0m")

    def create_johnson_epbpyM(self, noOutput, sizes, max_size, path, create_iq, create_gr, instance_Debye):
        
        """
        Generate johnson epbpyM NPs and their files from cif data.
      
        Parameters 
        ----------
        noOutput : bool 
            If False, prints details about the process.
            
        sizes : ndarray
             ndarray with shape (N,3) for [sizeP, sizeE, Marks],
             see conditions on the example notebook.
             
        max_size (float, optional) : 
            Maximal size for the NPs, equals to the diameter of
            the circumscribed sphere, equals 50 nm by default.
            
        path : str
            The directory where files will be written.       
            
        create_iq : bool 
            I(q) data in a NPZ file : q, iq_saxs, iq_waxs
            
        create_gr : bool 
            G(r) data in a NPZ file : r, gr
            
        noOutput : bool
            If False, print file-writing feedback.
            
        instance_Debye (class instance, optional): 
            Instance of the Debye calculator module in order to
            compute I,q and G,r
    
        Note
        -----
        The function uses the johnson module "jNP" and the fcctbp class.
        """
        if not noOutput:
            print(f' Sizes = [bonds in pentagonal cross-section, elongated part, number of truncated atoms]')

        for cif_name, cif_file in self.cif_data['cif file'].items():
            element = cif_name.split()[0]
            if element in self.UNSUPPORTED_ELEMENTS:
                continue

            self.cif_name = cif_name
            structure = self.cif_name.split()[1] if len(self.cif_name.split()) == 2 else None

            if not noOutput:
                print(f'\n\033[1m{cif_name.center(50)}\033[0m\n')

            cif_info = pyNMBu.load_cif(self, cif_file, noOutput)

            if self.crystal_type in ['fcc', 'bcc', 'hcp']:
                dist = pyNMBu.FindInterAtomicDist(self)
                index=0 
                for i, y, z in sizes:
                    index += 1 
                    if not noOutput :
                        print(f'{bg.LIGHTBLUEB} [{i},{y},{z}]  ')
                    TestNP =jNP.epbpyM(
                        element=element,
                        Rnn=dist,
                        sizeP=i,
                        sizeE=y,
                        Marks=z,
                        postAnalyzis=True,
                        aseView=False,
                        thresholdCoreSurface=1,
                        skipSymmetryAnalyzis=True,
                        noOutput= True
                    )
       
                    circumsphere_diameter=NPFiles._safe_radius(TestNP)*2*0.1
                    if circumsphere_diameter<max_size :
                        if not noOutput :
                            print(f'{bg.LIGHTBLUEB}Circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                        self.writexyz_generalized(structure, TestNP, index)
                    else :
                        if not noOutput :
                            print(f'{bg.LIGHTBLUEB} Size greater than the circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                        break
                      
            else:
                dist = None
                if not noOutput:
                    print(f"{bg.LIGHTREDB} xyz/cif files can't be created for {self.cif_name} because the crystal type isn't supported.\033[0m")


################################################ OtherNPs Database ################################################################################################## 

class OtherNPs_Files(NPFiles):
    
    """
    A class for generating XYZ and CIF files of trigonal platelets
    from a dataset of compounds (CIF dataset). This process enables the creation
    of a well-structured database optimized for machine learning applications, 
    ensuring consistency in format and data representation.
    """
    def __init__(self, path, cif_data,sizes,sizes2,max_size: float=50,instance_Debye=None, create_iq: bool = False, create_gr: bool = False, optimize_structure: bool = False, noOutput:bool = True):
        
        """
        Initialize the class with CIF data and size parameters for NP generation.
    
        Parameters
        ----------
        path : str
            Path where the generated XYZ and CIF files will be stored.
        
        cif_data : pandas.DataFrame
            DataFrame containing CIF files of different compounds.
              The DataFrame should include a 
            'cif file' column with file paths.
    
        sizes : array-like
            Array of integers representing the sizes (e.g., number of bonds,
            shells, or base lengths) used for nanoparticle construction.
    
        sizes2 : array-like
            Secondary array of integers representing an additional size
            parameter (e.g., number of shells for elongation, or truncation
            depth).
    
        max_size : float, optional
            Maximum allowed diameter (in nm) of the circumscribed sphere for
            the generated nanoparticles. 
            Defaults to 50 nm.
    
        instance_Debye : object, optional
            Instance of a Debye calculator class, used to compute I(q)
            and optionally G(r). 
            If provided and `create_iq` or `create_gr` is True,
            the I(q)/G(r) data will be calculated and saved.
    
        create_iq : bool, optional
            If True, generates CSV files containing I(q) data along with NP
            metadata. Defaults to False.
    
        create_gr : bool, optional
            If True, generates CSV files containing G(r) data along with NP
            metadata. Defaults to False.
    
        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential
            for [Al, Cu, Ag, Au, Ni, Pd, Pt].
    
        noOutput : bool, optional
            If False, prints status messages and details during
            the generation process. Defaults to True.
    
        Notes
        -----
        Upon initialization, the method `create_OtherNPs` is
        immediately called with the provided parameters to generate
        and export the requested nanoparticle shapes (e.g., fcctpt).
        """

        super().__init__(
            path, cif_data, sizes,
            max_size=max_size,
            optimize_structure=optimize_structure,
            instance_Debye=instance_Debye,
            create_iq=create_iq,
            create_gr=create_gr,
            noOutput=noOutput
        )
        self.sizes2 = sizes2
        form = 'fcctpt'
        
        self.create_OtherNPs(noOutput, max_size, path, form, create_iq, create_gr, instance_Debye, optimize_structure)



    def create_OtherNPs(self,noOutput,max_size,path,form,create_iq, create_gr,instance_Debye, optimize_structure):
        
        """
        Genenrate fcc trigonal platelets NPs and their files from cif data.
        Parameters
        ---------- 
        noOutput : bool, optional
            If False, prints status messages and details during the generation
            process. Defaults to True.
    
        max_size : float, optional
            Maximum allowed diameter (in nm) of the circumscribed sphere
            for the generated nanoparticles. 
            Defaults to 50 nm.
            
        path : str
            The directory where files will be writtenn

        form : str
            'fcctpt' for now.
    
        instance_Debye : object, optional
            Instance of a Debye calculator class, used to compute I(q)
            and optionally G(r). 
            If provided and `create_iq` or `create_gr` is True, the I(q)/G(r)
            data will be calculated and saved.
    
        create_iq : bool, optional
            If True, generates CSV files containing I(q) data along with
            NP metadata. Defaults to False.
    
        create_gr : bool, optional
            If True, generates CSV files containing G(r) data along with
            NP metadata. Defaults to False.
    
        optimize_structure : bool, optional
            If True, optimize the structure using EMT potential
            for [Al, Cu, Ag, Au, Ni, Pd, Pt].
        
        instance_Debye : object, optional
            Instance of a Debye calculator class, used to compute I(q)
            and optionally G(r). 
            If provided and `create_iq` or `create_gr` is True, the I(q)/G(r)
            data will be calculated and saved.
    
        Args:
            noOutput (bool): If False, prints details about the process.
            path (str): Path where xyz/cif files will be created.
 
        """
      
        for cif_name, cif_file in self.cif_data['cif file'].items():
            
            # Load cif informations
            element=cif_name.split()[0] # Extract chemical element from CIF name
            
            if element not in self.UNSUPPORTED_ELEMENTS:
                self.cif_name=cif_name
                
                # Extract the structure name for the name of the files, for example 'Rutile' or 'Anatase' or 'Alpha'
                if len(self.cif_name.split())==2 : 
                    structure=self.cif_name.split()[1]
                else : 
                    structure=None
                if not noOutput :
                    print(f'\n\033[1m{cif_name.center(50)}\033[0m\n')
                cif_info = pyNMBu.load_cif(self,cif_file,noOutput)
                    
                # Determine forms based on crystal type
                if self.crystal_type=='fcc' :
                    dist= pyNMBu.FindInterAtomicDist(self)# Extract the interatomic distance
                    index=0 
                    if not noOutput :
                        print(f" {bg.LIGHTGREENB} xyz/cif files can be created for {self.cif_name} of Bravais lattice={self.crystal_type} for the form {form}. \033[0m ")
                        # Create instances for each form and size
                    for i in self.sizes :
                        stop_i = False
                        for y_idx, y in enumerate(self.sizes2):
                            if y < i :
                                index += 1 
                                if not noOutput :
                                    print(f'{bg.LIGHTBLUEB} Sizes [{i},{y}]')
                                TestNP =oNP.fcctpt(
                                    element=element,
                                    Rnn=dist,
                                    nLayerTd=i,
                                    nLayer = y, #number of layers per pyramid -> total number of layers = nLayer*2 + twinning plane
                                    postAnalyzis=True,
                                    aseView=False,
                                    thresholdCoreSurface=1,
                                    skipSymmetryAnalyzis=True,
                                    noOutput= True
                                )
        
                                circumsphere_diameter=NPFiles._safe_radius(TestNP)*2*0.1
                                if circumsphere_diameter<max_size :
                                    if not noOutput :
                                        print(f'{bg.LIGHTBLUEB}Circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                                    self.writexyz_generalized(structure, TestNP, index)
                                else :
                                    if y_idx == 0:
                                        stop_i = True
                                    if not noOutput :
                                        print(f'{bg.LIGHTBLUEB} Size greater than the circumscribed sphere diameter ={circumsphere_diameter}\033[0m')
                                    break
                        if stop_i:
                            break

                    
          

                else : 
                    if not noOutput :
                        print(f" {bg.LIGHTREDB} xyz/cif files can't be created for {self.cif_name} because the crystal type isn't fcc.\033[0m  ")   
                    pass
         
            else : 
                pass


############################################### Create a dataframe with all the data#################################################################################

def create_full_dataframe_from_csv_npz(csv_folder_path,
                                       csv_pattern="*.csv",
                                       base_npz_path=None,
                                       id_col="id"):
    """
    Load all CSV files from the folder `csv_folder_path`
    matching `csv_pattern`, concatenate them into a single DataFrame,
    then for each row reconstruct the NPZ paths from the `id` column:
        - `<id>_iq.npz` → columns q, iq_saxs, iq_waxs
        - `<id>_gr.npz` → columns r, gr

    If the NPZ file doesn't exist, the corresponding columns are filled
    with ``"no data"``.

    Args:
        csv_folder_path (str): folder containing CSV files.
        csv_pattern (str): glob pattern for CSV files (default ``"*.csv"``).
        base_npz_path (str or None): base folder where NPZ files are searched.
            If None, search in ``csv_folder_path``.
        id_col (str): name of the column containing the identifier
            (default ``"id"``).

    Returns:
        pd.DataFrame: DataFrame complet avec les données NPZ chargées.
    """

    import glob

    if base_npz_path is None:
        base_npz_path = csv_folder_path

    # 1. Find all CSV files and concatenates them into a single DataFrame
    csv_files = glob.glob(os.path.join(csv_folder_path, csv_pattern))
    # Don't keep the sasview CSV files for now
    csv_files = [f for f in csv_files if not f.endswith("_sasview.csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {csv_folder_path} "
            f"with pattern {csv_pattern}"
        )

    df = pd.concat(
        [pd.read_csv(f) for f in csv_files], ignore_index=True
    )

    # 2. For each row, reconstruct the NPZ paths from the `id` column and load the data
    def load_npz_arrays(row):
        data = {}
        file_id = row.get(id_col, None)

        if file_id is None or pd.isna(file_id):
            data["q"] = None
            data["iq_saxs"] = None
            data["iq_waxs"] = None
            data["r"] = None
            data["gr"] = None
            return pd.Series(data)

        iq_path = os.path.join(
            base_npz_path, f"{file_id}_iq.npz"
        )
        gr_path = os.path.join(
            base_npz_path, f"{file_id}_gr.npz"
        )

        # Loads I(q)
        if os.path.isfile(iq_path):
            with np.load(iq_path) as npz_iq:
                data["q"] = npz_iq["q"]
                data["iq_saxs"] = npz_iq["iq_saxs"]
                data["iq_waxs"] = npz_iq["iq_waxs"]
        else:
            data["q"] = None
            data["iq_saxs"] = None
            data["iq_waxs"] = None

        # Loads G(r)
        if os.path.isfile(gr_path):
            with np.load(gr_path) as npz_gr:
                data["r"] = npz_gr["r"]
                data["gr"] = npz_gr["gr"]
        else:
            data["r"] = None
            data["gr"] = None

        return pd.Series(data)

    # 3. Apply the loading function to each row 
    # and concatenate the results with the original DataFrame
    npz_data = df.apply(load_npz_arrays, axis=1)
    df_final = pd.concat([df, npz_data], axis=1)

    return df_final

















