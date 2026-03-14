# General external dependencies
import math
import re
import numpy as np
import pandas as pd
import os
from pathlib import Path

# External dependencies (ASE)
from ase import io
from ase.build import bulk
from ase.build.supercells import make_supercell
from ase.geometry import cellpar_to_cell
from ase.visualize import view

# Internal Relative Imports
from .visualID import fg, hl, bg
from . import visualID as vID
from . import data
from . import utils as pyNMBu

class Crystal:
    """
    A class for generating XYZ and CIF files of crystalline nanoparticles (NPs) 
    of various shapes and sizes, based on user-defined compounds (either by 
    name, e.g., "Fe bcc", or from a CIF file). The supported nanoparticle 
    shapes include:

    - Spheres
    - Ellipsoids
    - Parallelepipeds
    - Cylinders and wires with different cross-sections
    - Wulff constructions: cube, octahedron, cuboctahedron, dodecahedron, 
      spheroids, and their truncated versions

    Key Features:
    - Allows to choose the NP size, shape and composition.
    - Supports Wulff construction with customizable surface energies.
    - Enables the creation of wires with defined orientations and cross-sections.
    - Can analyze the structure in detail, including symmetry and properties.
    - Offers options for core/surface differentiation based on a threshold.
    - Generates outputs in XYZ and CIF formats for visualization and simulations.
    - Provides compatibility with jMol for 3D visualization.
    
    Additional Notes:
    - The symmetry analysis can be skipped to speed up computations.
    - Periodic boundary conditions (PBC) can be enabled if needed.
    - Customizable precision thresholds for structural analysis.
    """

    def __init__(
        self,
        crystal: str = "Au",
        scaleDmin2: float = None,
        setSymbols2: np.ndarray = None,
        userDefCif: str = None,
        shape: str = "sphere",
        size: float = None,
        directionsPPD: np.ndarray = None,
        buildPPD: str = "xyz",
        directionWire: float = None,
        directionCylinder: float = None,
        refPlaneWire: float = None,
        nRotWire: int = 6,
        hollow_sphere_diameter: float = None,
        surfacesWulff: np.ndarray = None,
        eSurfacesWulff: np.ndarray = None,
        sizesWulff: np.ndarray = None,
        symWulff: bool = True,
        jmolCrystalShape: bool = True,
        aseSymPrec: float = 1e-4,
        pbc: bool = False,
        threshold: float = 1e-3,
        dbFolder: str = None,
        postAnalyzis: bool = True,
        aseView: bool = False,
        thresholdCoreSurface: float = 1.0,
        skipSymmetryAnalyzis: bool = False,
        noOutput: bool = False,
        calcPropOnly: bool = False,
    ):
        """
        Initialize a Crystal nanoparticle generator with specified parameters.

        Args:
            crystal (str): Chemical element or compound name (default: "Au").
                See pyNMBu.ciflist() for available options.
            scaleDmin2 (float, optional): Scale factor for unit cell minimum dimension.
            setSymbols2 (np.ndarray, optional): Array of chemical symbols to replace default.
            userDefCif (str, optional): Path to user-defined CIF file.
            shape (str): Nanoparticle shape. Options: 'sphere', 'ellipsoid', 'parallelepiped',
                'wire', 'cylinder', or 'Wulff: <shape>' (default: "sphere").
            size (list, optional): Size specification (format depends on shape):
                - Sphere: [diameter]
                - Ellipsoid/Parallelepiped: [size_x, size_y, size_z]
                - Wire/Cylinder: [cross-section diameter, length]
            directionsPPD (np.ndarray, optional): Three direction vectors for parallelepiped.
            buildPPD (str): Parallelepiped coordinate system: 'xyz' or 'abc' (default: "xyz").
            directionWire (list, optional): Wire growth direction (default: [0, 0, 1]).
            directionCylinder (list, optional): Cylinder growth direction (default: [0, 0, 1]).
            refPlaneWire (list, optional): Miller indices of wire reference plane.
            nRotWire (int): Number of rotations for wire cross-section (default: 6).
            hollow_sphere_diameter (list, optional): Hollow sphere diameter (Kirkendall effect).
            surfacesWulff (np.ndarray, optional): Miller indices for Wulff surfaces.
            eSurfacesWulff (np.ndarray, optional): Surface energies for Wulff construction.
            sizesWulff (np.ndarray, optional): Size parameters for Wulff construction.
            symWulff (bool): Apply symmetry to Wulff construction (default: True).
            jmolCrystalShape (bool): Generate JMol visualization script (default: True).
            aseSymPrec (float): Symmetry precision threshold (default: 1e-4).
            pbc (bool): Enable periodic boundary conditions (default: False).
            threshold (float): Plane truncation distance threshold (default: 1e-3).
            dbFolder (str, optional): Database folder path for CIF files.
            postAnalyzis (bool): Perform post-construction analysis (default: True).
            aseView (bool): Enable ASE visualization (default: False).
            thresholdCoreSurface (float): Core/surface differentiation threshold (default: 1.0).
            skipSymmetryAnalyzis (bool): Skip symmetry analysis (default: False).
            noOutput (bool): Suppress printed output (default: False).
            calcPropOnly (bool): Calculate properties only without structure generation (default: False).
        """
        # Initialize default values to avoid mutable default arguments
        if size is None:
            size = [2, 2, 2]
        if directionsPPD is None:
            directionsPPD = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if directionWire is None:
            directionWire = [0, 0, 1]
        if directionCylinder is None:
            directionCylinder = [0, 0, 1]
        if refPlaneWire is None:
            refPlaneWire = [1, 0, 0]
        if hollow_sphere_diameter is None:
            hollow_sphere_diameter = [0.0]
        if dbFolder is None:
            dbFolder = data.pyNMBvar.dbFolder

        self.dbFolder = dbFolder  # Database folder containing CIF files
        self.crystal = crystal
        self.shape = shape.strip(" ")  # 'sphere', 'ellipsoid', 'parallelepiped', 'wire', 'cylinder', 'Wulff'
        self.size = size
        self.directionsPPD = directionsPPD
        self.buildPPD = buildPPD
        self.directionWire = directionWire
        self.directionCylinder = directionCylinder
        self.refPlaneWire = refPlaneWire
        self.nRotWire = nRotWire
        self.hollow_sphere_diameter = hollow_sphere_diameter
        self.surfacesWulff = surfacesWulff
        self.eSurfacesWulff = eSurfacesWulff
        self.sizesWulff = sizesWulff
        self.symWulff = symWulff
        self.jmolCrystalShape = jmolCrystalShape
        self.aseSymPrec = aseSymPrec
        self.pbc = pbc
        self.threshold = threshold
        self.nAtoms = 0
        self.cif = None
        self.cifname = None
        self.userDefCif = userDefCif
        self.trPlanes = None


        if "Wulff" in self.shape:
            self.WulffShape = self.shape.split(":")
            if len(self.WulffShape) == 2:
                self.WulffShape = self.WulffShape[1].lstrip(" ")  # Removes 'Wulff:'
                self.shape = "Wulff: " + self.WulffShape  # Normalize shape name
            else:
                self.WulffShape = None

        if self.userDefCif is not None:
            self.loadExternalCif()

        if not noOutput:
            vID.centerTitle(f"{self.crystal} {self.shape}")

        self.bulk(noOutput)
        if scaleDmin2 is not None:
            pyNMBu.scaleUnitCell(self, scaleDmin2, noOutput=noOutput)
        if setSymbols2 is not None:
            self.cif.set_chemical_symbols(setSymbols2)
        if aseView:
            view(self.cif)

        if not calcPropOnly:
            self.makeNP(noOutput)
            if aseView:
                view(self.sc)
                view(self.NP)
            if postAnalyzis:
                self.prop(noOutput)
                self.propPostMake(skipSymmetryAnalyzis, thresholdCoreSurface, noOutput)
                if aseView:
                    view(self.NPcs)
    def __str__(self):
        """Return string representation of the Crystal instance."""
        return f"Crystal = {self.crystal} {self.shape}"

    def loadExternalCif(self):
        """
        Load an external CIF file and extract the crystal name.

        Checks if a CIF file is already loaded to avoid redundant loading.
        Extracts the crystal name from specific CIF tags if available.

        Raises:
            SystemExit: If the CIF file is not found.
        """
        if hasattr(self, "cif"):
            return  # CIF already loaded

        self.cif = io.read(self.userDefCif)
        path2extCif = Path(self.userDefCif)
        if not path2extCif.exists():
            sys.exit(
                f"File {self.userDefCif} not found. "
                "Check the file name or its location."
            )

        # Search for specific CIF tags containing the crystal name
        with open(self.userDefCif, "r") as cifFile:
            cifFileLines = cifFile.readlines()

        re_name_systematic = re.compile("_chemical_name_systematic")
        re_name_sum = re.compile("_chemical_formula_sum")
        re_name_moiety = re.compile("_chemical_formula_moiety")

        crystal1 = None
        crystal2 = None
        crystal3 = None

        for line in cifFileLines:
            if re_name_systematic.search(line):
                parts = line.split()
                crystal1 = " ".join(parts[1:])
            if re_name_sum.search(line):
                parts = line.split()
                crystal2 = " ".join(parts[1:])
            if re_name_moiety.search(line):
                parts = line.split()
                crystal3 = " ".join(parts[1:])

        # Assign crystal name based on priority
        if crystal1 is not None:
            self.crystal = crystal1
        elif crystal3 is not None:
            self.crystal = crystal3
        elif crystal2 is not None:
            self.crystal = crystal2
        else:
            self.crystal = "unknown"

    def bulk(self, noOutput):
        """
        Retrieve bulk crystal structure parameters.

        Loads CIF data from either a user-defined file or the internal database.

        Args:
            noOutput (bool): If False, prints status information.

        Raises:
            SystemExit: If crystal not found and no external CIF provided.
        """
        if self.userDefCif is None:
            # Search for crystal in database
            if self.crystal.upper() in data.pyNMBcif.CIFdf.index.str.upper():
                dftmp = pd.DataFrame(index=data.pyNMBcif.CIFdf.index.copy())
                data.pyNMBcif.CIFdf.index = data.pyNMBcif.CIFdf.index.str.upper()
                self.cifname = data.pyNMBcif.CIFdf["cif file"].loc[
                    self.crystal.upper()
                ]
                data.pyNMBcif.CIFdf.index = dftmp.index.copy()  # Revert index
            else:
                if not noOutput:
                    display(data.pyNMBcif.CIFdf)
                sys.exit(
                    f"{fg.RED}{bg.LIGHTREDB}The database does not contain "
                    f"bulk parameters for '{self.crystal}' crystal.\n"
                    f"Please provide a CIF file.{fg.OFF}"
                )
            full_path_to_cif = pyNMBu.get_resource_path("resources/cif_database", self.cifname)
            print(f"######################### {full_path_to_cif=}")
            self.cif = io.read(full_path_to_cif)
        else:
            self.cif = io.read(self.userDefCif)
            path2extCif = Path(self.userDefCif)
            self.cifname = Path(*path2extCif.parts[-1:])

        pyNMBu.returnUnitcellData(self)
        if not noOutput:
            print(f"CIF parameters for {self.crystal} found in {self.cifname}")

    def makeSuperCell(self, noOutput):
        """
        Create a supercell based on the nanoparticle shape and size.

        Determines the appropriate supercell dimensions based on the particle shape,
        then constructs and centers the supercell at the origin.

        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput:
            chrono = pyNMBu.timer()
            chrono.chrono_start()
        if not noOutput:
            vID.centertxt(
                "Making a multiple cell",
                bgc="#cbcbcb",
                size="12",
                fgc="b",
                weight="bold",
            )
        extend_size_by_factor = 1.06  # Extension factor for sufficient cell size

        # Determine supercell dimensions based on shape
        if self.shape == "sphere":
            sphere_radius = self.size[0] / 2
            Ma = int(
                np.round(
                    extend_size_by_factor
                    * sphere_radius
                    * 2
                    * 10
                    / self.cif.cell.lengths()[0]
                )
            )
            Mb = int(
                np.round(
                    extend_size_by_factor
                    * sphere_radius
                    * 2
                    * 10
                    / self.cif.cell.lengths()[1]
                )
            )
            Mc = int(
                np.round(
                    extend_size_by_factor
                    * sphere_radius
                    * 2
                    * 10
                    / self.cif.cell.lengths()[2]
                )
            )
        elif self.shape in ("ellipsoid", "supercell", "parallelepiped"):
            Ma = int(
                np.round(
                    extend_size_by_factor
                    * self.size[0]
                    * 2
                    * 10
                    / self.cif.cell.lengths()[0]
                )
            )
            Mb = int(
                np.round(
                    extend_size_by_factor
                    * self.size[1]
                    * 2
                    * 10
                    / self.cif.cell.lengths()[1]
                )
            )
            Mc = int(
                np.round(
                    extend_size_by_factor
                    * self.size[2]
                    * 2
                    * 10
                    / self.cif.cell.lengths()[2]
                )
            )
        elif self.shape == "wire":
            max_dim = np.max(self.size) * 10 * 1.5
            Ma = int(
                np.round(
                    extend_size_by_factor * max_dim / self.cif.cell.lengths()[0]
                )
            )
            Mb = int(
                np.round(
                    extend_size_by_factor * max_dim / self.cif.cell.lengths()[1]
                )
            )
            Mc = int(
                np.round(
                    extend_size_by_factor * max_dim / self.cif.cell.lengths()[2]
                )
            )
        elif self.shape == "cylinder":
            diameter = self.size[0]
            length = self.size[1]
            max_dim = max(diameter, length)
            # Ensure supercell is large enough after rotation to any axis
            Ma = int(
                np.round(
                    extend_size_by_factor
                    * max_dim
                    * 10
                    * 1.5
                    / self.cif.cell.lengths()[0]
                )
            )
            Mb = int(
                np.round(
                    extend_size_by_factor
                    * max_dim
                    * 10
                    * 1.5
                    / self.cif.cell.lengths()[1]
                )
            )
            Mc = int(
                np.round(
                    extend_size_by_factor
                    * max_dim
                    * 10
                    * 1.5
                    / self.cif.cell.lengths()[2]
                )
            )
        elif "Wulff" in self.shape:
            if np.argmax(self.sizesWulff) == 1:
                max_dim = self.sizesWulff[0] * 10 * 1.5
            else:
                max_dim = np.max(self.sizesWulff) * 10 * 1.5
            Ma = int(
                np.round(
                    extend_size_by_factor * max_dim / self.cif.cell.lengths()[0]
                )
            )
            Mb = int(
                np.round(
                    extend_size_by_factor * max_dim / self.cif.cell.lengths()[1]
                )
            )
            Mc = int(
                np.round(
                    extend_size_by_factor * max_dim / self.cif.cell.lengths()[2]
                )
            )

        # Define minimum supercell size (at least 20 Å per direction)
        ma1nm = int(np.round(20 / self.cif.cell.lengths()[0]))
        mb1nm = int(np.round(20 / self.cif.cell.lengths()[1]))
        mc1nm = int(np.round(20 / self.cif.cell.lengths()[2]))
        ma1nm = min(ma1nm, Ma)
        mb1nm = min(mb1nm, Mb)
        mc1nm = min(mc1nm, Mc)

        if not noOutput:
            print(f"First making a {ma1nm}x{mb1nm}x{mc1nm} supercell")

        # Generate initial supercell
        M1nm = [[ma1nm, 0, 0], [0, mb1nm, 0], [0, 0, mc1nm]]
        sc1nm = make_supercell(self.cif, M1nm)

        # Scale up supercell size (find nearest even numbers)
        Ma = Ma / ma1nm
        Mb = Mb / mb1nm
        Mc = Mc / mc1nm
        Ma = math.ceil(Ma / 2.0) * 2
        Mb = math.ceil(Mb / 2.0) * 2
        Mc = math.ceil(Mc / 2.0) * 2

        if not noOutput:
            print(f"Making a {Ma}x{Mb}x{Mc} supercell of the supercell")
            print(f"       = {Ma * ma1nm}x{Mb * mb1nm}x{Mc * mc1nm} supercell")

        M = [[Ma, 0, 0], [0, Mb, 0], [0, 0, Mc]]
        sc = make_supercell(sc1nm, M)
        V = cellpar_to_cell(sc.cell.cellpar())

        # Center supercell at origin
        if not noOutput:
            com = sc.get_center_of_mass()
            print(f"Center of Mass: {[f'{c:.2f}' for c in com]} Å")
            print("Now translating the supercell to origin")

        sc.translate(-V[0] / 2)
        sc.translate(-V[1] / 2)
        sc.translate(-V[2] / 2)

        if not noOutput:
            print(f"Center of Mass after translation: {sc.get_center_of_mass()} Å")

        # Store final supercell and count atoms
        self.sc = sc.copy()
        n_atoms = len(self.sc.get_positions())

        if not noOutput:
            print(f"Total number of atoms = {n_atoms}")
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        
    def makeSphere(self, noOutput):
        """
        Create a spherical nanoparticle by removing atoms outside the defined radius.

        Computes dimensions from the provided diameter in nanometers.

        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput:
            vID.centertxt(
                "Removing atoms to make a sphere",
                bgc="#cbcbcb",
                size="12",
                fgc="b",
                weight="bold",
            )
            chrono = pyNMBu.timer()
            chrono.chrono_start()

        # Get center of mass
        com = self.sc.get_center_of_mass()

        # Compute sphere radius from diameter
        sphere_radius = self.size[0] / 2

        # Identify atoms to delete (distance > radius)
        del_atom = []
        del_atom_hollow = []
        for atom_coord in self.sc.positions:
            del_atom.extend(
                pyNMBu.Rbetween2Points(com, atom_coord) / 10 > [sphere_radius]
            )

        self.NP = self.sc.copy()
        del self.NP[del_atom]

        # Compute measured diameters
        positions = self.NP.get_positions()
        zmax = max(positions[:, 2])
        zmin = min(positions[:, 2])
        self.radius = (zmax - zmin) / 2
        self.sasview_dims = [self.radius]  # For consistency with ellipsoid

        # Compute volume
        self.volume = (4 / 3) * math.pi * (self.radius ** 3)

        if not noOutput:
            print(f"Measured radius = {self.radius:.2f} Å")

        # Handle hollow sphere (Kirkendall effect)
        if not self.hollow_sphere_diameter[0] == 0.0:
            for atom_coord in self.NP.positions:
                del_atom_hollow.extend(
                    pyNMBu.Rbetween2Points(com, atom_coord) / 10
                    < [self.hollow_sphere_diameter[0] / 2]
                )
            del self.NP[del_atom_hollow]

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()

    def makeEllipsoid(self, noOutput):
        """
        Create an ellipsoidal nanoparticle by removing atoms outside the defined ellipsoid.

        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput:
            vID.centertxt(
                "Removing atoms to make an ellipsoid",
                bgc="#cbcbcb",
                size="12",
                fgc="b",
                weight="bold",
            )
            chrono = pyNMBu.timer()
            chrono.chrono_start()

        # Get center of mass
        com = self.sc.get_center_of_mass()

        # Convert size from diameter (nm) to radius (Ångström) for each axis
        size = (np.array(self.size) / 2) * 10

        def outside(coord, com, size):
            """
            Check if atom is outside the ellipsoid.

            Args:
                coord (list): Atom position coordinates.
                com (list): Center of mass coordinates.
                size (list): Half sizes of ellipsoid along each axis.

            Returns:
                bool: True if atom is outside ellipsoid, False otherwise.
            """
            return (
                (coord[0] - com[0]) ** 2
                / (size[0]) ** 2
                + (coord[1] - com[1]) ** 2
                / (size[1]) ** 2
                + (coord[2] - com[2]) ** 2
                / (size[2]) ** 2
            )

        # Identify atoms to delete (outside ellipsoid)
        del_atom = np.array([outside(atom, com, size) > 1 for atom in self.sc.positions])

        self.NP = self.sc.copy()
        del self.NP[del_atom]

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()

        # Compute measured dimensions
        positions = self.NP.get_positions()
        mins = np.min(positions, axis=0)
        maxs = np.max(positions, axis=0)
        a_real = (maxs[0] - mins[0]) / 2  # Semi-axis a
        b_real = (maxs[1] - mins[1]) / 2  # Semi-axis b
        c_real = (maxs[2] - mins[2]) / 2  # Semi-axis c

        if not noOutput:
            print(
                f"Measured diameters: a = {a_real*2:.2f} Å, "
                f"b = {b_real*2:.2f} Å, c = {c_real*2:.2f} Å"
            )

        # Compute polar and equatorial radii for SASSview compatibility
        self.radius_polar = min(a_real, b_real, c_real)
        self.radius_equatorial = max(a_real, b_real, c_real)
        self.sasview_dims = [
            self.radius_equatorial,
            self.radius_polar,
        ]  # [equatorial, polar]

        self.volume = (4 / 3) * math.pi * a_real * b_real * c_real

    def makeWire(self, noOutput=False):
        """
        Create a nanowire by truncating atoms based on reference planes.

        Uses rotated planes around the growth direction to define wire cross-section.
        Measures and stores the wire length, radius, and volume.

        Args:
            noOutput (bool): If False, details are printed (default: False).

        Returns:
            ase.Atoms: The generated nanowire structure.
        """
        if not noOutput:
            vID.centertxt(
                "Removing atoms to make a wire",
                bgc="#cbcbcb",
                size="12",
                fgc="b",
                weight="bold",
            )
            chrono = pyNMBu.timer()
            chrono.chrono_start()

        # Construct reference plane if not provided
        if self.refPlaneWire is None:
            self.refPlaneWire = pyNMBu.returnPlaneParallel2Line(
                self.directionWire, [1, 0, 0], debug=False
            )

        normal = pyNMBu.normal2MillerPlane(
            self, self.refPlaneWire, printN=not noOutput
        )
        tr_planes = pyNMBu.planeRotation(
            self,
            normal,
            self.directionWire,
            self.nRotWire,
            debug=False,
            noOutput=noOutput,
        )
        tr_planes = np.array([pyNMBu.normV(p) for p in tr_planes])

        # Define radial constraint planes
        radius = 10 * self.size[0] / 2
        tradius = np.full((self.nRotWire, 1), -radius)
        tr_planes = np.append(tr_planes, tradius, axis=1)

        # Define axial constraint planes (if not periodic)
        if not self.pbc:
            half_length = 10 * self.size[1] / 2
            ptop = np.append(pyNMBu.normV(self.directionWire), -half_length)
            pbottom = np.append(-pyNMBu.normV(self.directionWire), -half_length)
            tr_planes = np.append(tr_planes, [ptop, pbottom], axis=0)

        # Truncate atoms
        atoms_above_planes = pyNMBu.truncateAbovePlanes(
            tr_planes, self.sc.positions, eps=self.threshold, noOutput=noOutput
        )
        self.NP = self.sc.copy()
        del self.NP[atoms_above_planes]
        self.trPlanes = tr_planes

        if not noOutput:
            vID.centertxt(
                "Nanowire moved to the center of the unit cell",
                bgc="#cbcbcb",
                size="12",
                fgc="b",
                weight="bold",
            )
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()


        return self.NP
       
    
    def makeCylinder(self, noOutput=False):
        """
        Create a cylindrical nanoparticle.

        Aligns the specified direction to the Z-axis, removes atoms outside the cylinder,
        and measures the resulting dimensions.

        Args:
            noOutput (bool): If False, details are printed (default: False).

        Returns:
            ase.Atoms: The generated cylindrical nanoparticle.

        Note:
            size = [diameter_nm, length_nm]
            directionCylinder = [h, k, l]
        """
        # Convert nanometers to Ångströms
        radius = self.size[0] * 10.0 / 2.0
        height = self.size[1] * 10.0
        half_height = height / 2.0

        if not noOutput:
            print(
                f"Target cylinder: length = {height:.2f} Å, radius = {radius:.2f} Å"
            )

        # 1) Align structure so requested direction becomes Z-axis
        axis = np.array(self.directionCylinder)
        self.NP = self.sc.copy()
        self.NP.positions = pyNMBu.rotateMoltoAlignItWithAxis(
            self.NP.positions, axis, targetAxis=np.array([0, 0, 1])
        )

        # 2) Center at center of mass
        com = self.sc.get_center_of_mass()
        self.NP.positions -= com

        # 3) Delete atoms outside the Z-aligned cylinder
        del_atom = [
            i
            for i, pos in enumerate(self.NP.positions)
            if (pos[0] ** 2 + pos[1] ** 2) > (radius ** 2)
            or (abs(pos[2]) > half_height)
        ]

        # 4) Copy and delete atoms
        del self.NP[del_atom]

        # 5) Recenter NP at its center of mass
        self.NP.positions -= self.NP.get_center_of_mass()

        # 6) Measure length and radius
        def measure_cylinder_dimensions(atoms):
            """
            Measure cylinder length and radius from atom positions.

            Args:
                atoms (ase.Atoms): The atomic structure to measure.

            Returns:
                tuple: (length, radius) measured values.
            """
            pos = atoms.get_positions()
            z_coord = pos[:, 2]
            length = z_coord.max() - z_coord.min()

            # Measure radius in central section to avoid conical edges
            z_mid = 0.5 * (z_coord.max() + z_coord.min())
            tol = max(1.0, 0.05 * length)  # At least 1 Å or 5% of length
            mid_mask = np.abs(z_coord - z_mid) < tol

            r_coord = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            if np.count_nonzero(mid_mask) >= 5:
                radius = r_coord[mid_mask].max()
            else:
                radius = r_coord.max()

            return length, radius

        length_meas, radius_meas = measure_cylinder_dimensions(self.NP)

        if not noOutput:
            print(
                f"Measured dimensions: "
                f"length = {length_meas:.2f} Å, radius = {radius_meas:.2f} Å"
            )

        self.length = length_meas
        self.radius = radius_meas
        self.sasview_dims = [radius_meas, length_meas]
        self.volume = math.pi * (radius_meas ** 2) * length_meas

        return self.NP


      
    def makeParallelepiped(self, noOutput):
        """
        Create a parallelepiped-shaped nanoparticle.

        Truncates atoms based on specified directions. If using Cartesian coordinates ("xyz"),
        directions are used directly. Otherwise, normal vectors are calculated and lattice
        transformation is applied.

        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput:
            vID.centertxt(
                "Removing atoms to make a parallelepiped",
                bgc="#cbcbcb",
                size="12",
                fgc="b",
                weight="bold",
            )
            chrono = pyNMBu.timer()
            chrono.chrono_start()

        # Use directions directly if Cartesian, otherwise compute normals
        if self.buildPPD == "xyz":
            tr_planes = self.directionsPPD
        else:
            normal = []
            for direction in self.directionsPPD:
                normal.append(
                    pyNMBu.normal2MillerPlane(self, direction, printN=not noOutput)
                )
            # Project from Bravais basis to Cartesian
            tr_planes = pyNMBu.lattice_cart(
                self, normal, Bravais2cart=True, printV=not noOutput
            )

        # Normalize each plane
        tr_planes = np.array([pyNMBu.normV(p) for p in tr_planes])

        # Define parallelepiped dimensions (convert from nm to Ångströms)
        size = -np.array(self.size) * 10 / 2
        size = np.append(size, size, axis=0)

        # Define 6 planes: [-a/2, a/2], [-b/2, b/2], [-c/2, c/2]
        tr_planes = np.append(tr_planes, -tr_planes, axis=0)
        tr_planes = np.append(tr_planes, size.reshape(6, 1), axis=1)

        # Identify atoms outside the parallelepiped
        atoms_above_planes = pyNMBu.truncateAbovePlanes(
            tr_planes,
            self.sc.positions,
            eps=self.threshold,
            debug=False,
            noOutput=noOutput,
        )
        self.NP = self.sc.copy()
        del self.NP[atoms_above_planes]
        self.trPlanes = tr_planes

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()

        # Compute measured dimensions
        if self.buildPPD == "xyz":
            positions = self.NP.get_positions()
            mins = np.min(positions, axis=0)
            maxs = np.max(positions, axis=0)
            a_real = maxs[0] - mins[0]  # Axis a
            b_real = maxs[1] - mins[1]  # Axis b
            c_real = maxs[2] - mins[2]  # Axis c

            if not noOutput:
                print(
                    f"Measured lengths: "
                    f"a = {a_real:.2f} Å, b = {b_real:.2f} Å, c = {c_real:.2f} Å"
                )

            self.sasview_dims = sorted([a_real, b_real, c_real])

        elif self.buildPPD == "abc":
            # NOTE: Not working for non-orthogonal lattices
            # TODO: Implement proper projection for non-90° angles
            self.sasview_dims = sorted(
                [self.size[0] * 10, self.size[1] * 10, self.size[2] * 10]
            )

        self.volume = (
            self.sasview_dims[0] * self.sasview_dims[1] * self.sasview_dims[2]
        )


    def makeWulff(self, noOutput):
        """
        Calculate truncation distances for Wulff nanoparticles.

        Determines truncation planes based on provided surfaces and surface energies.
        Removes atoms above these planes to create the final shape.

        Args:
            noOutput (bool): If False, details are printed.

        Note:
            Updates self.trPlanes with normalized truncation planes and distances.
            Updates self.NP with the truncated structure.
        """
        if not noOutput:
            vID.centertxt(
                "Calculating truncation distances",
                bgc="#cbcbcb",
                size="12",
                fgc="b",
                weight="bold",
            )
            chrono = pyNMBu.timer()
            chrono.chrono_start()

        tr_planes = []  # List to store truncation planes
        if self.eSurfacesWulff is None:
            sizes = []
        else:
            sizes = []
            e_surf = []

        # Loop over surface planes to compute truncation
        for i, p in enumerate(self.surfacesWulff):
            if self.symWulff:
                # sym_p = self.ucSG.equivalent_lattice_points(p) #deprecated (ase function)
                sym_p = pyNMBu.get_equivalent_miller_indices(self,p)
                normal = []
                for sp in sym_p:
                    normal.append(
                        pyNMBu.normal2MillerPlane(self, sp, printN=not noOutput)
                    )
                tr_planes += list(normal)
                if self.eSurfacesWulff is None:
                    sizes.append(len(sym_p) * [self.sizesWulff[i]])
                if self.eSurfacesWulff is not None:
                    e_surf += list(len(sym_p) * [self.eSurfacesWulff[i]])
            else:
                tr_planes.append(
                    pyNMBu.normal2MillerPlane(self, p, printN=not noOutput)
                )
                if self.eSurfacesWulff is None:
                    sizes.append(self.sizesWulff[i])
                if self.eSurfacesWulff is not None:
                    e_surf.append(self.eSurfacesWulff[i])

        tr_planes = np.array(tr_planes)
        # Convert Miller indices to Cartesian coordinates
        tr_planes = pyNMBu.lattice_cart(
            self, tr_planes, Bravais2cart=True, printV=not noOutput
        )
        # Normalize vectors
        tr_planes = np.array([pyNMBu.normV(p) for p in tr_planes])

        if self.eSurfacesWulff is None:
            sizes = -np.array(sizes) * 10 / 2
            tr_planes = np.append(
                tr_planes, sizes.reshape(len(tr_planes), 1), axis=1
            )
        else:
            # Use energy-weighted truncation distances
            most_stable_e = min(e_surf)
            sizes = []
            for i, e in enumerate(e_surf):
                sizes.append(-self.sizesWulff[0] * 10 * e / 2 / most_stable_e)
            sizes = np.array(sizes)
            tr_planes = np.append(
                tr_planes, sizes.reshape(len(tr_planes), 1), axis=1
            )

        # Remove atoms above truncation planes
        atoms_above_planes = pyNMBu.truncateAbovePlanes(
            tr_planes,
            self.sc.positions,
            allP=False,
            eps=self.threshold,
            debug=False,
            noOutput=noOutput,
        )
        self.NP = self.sc.copy()
        del self.NP[atoms_above_planes]
        self.trPlanes = tr_planes

        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()

    def makeNP(self, noOutput):
        """
        Generate a nanoparticle of the specified shape.

        Constructs different nanoparticle types based on user-defined shape and size.
        Supports spheres, ellipsoids, parallelepipeds, wires, cylinders, and Wulff
        constructions.

        Args:
            noOutput (bool): If False, details are printed.

        Note:
            Updates self.NP with the generated nanoparticle structure.
            Updates self.nAtoms, self.cog, and self.trPlanes attributes.
        """
        if not noOutput:
            vID.centertxt("Builder", bgc="#007a7a", size="14", weight="bold")

        # Set default size if not provided
        if self.size is None:
            self.size = [2, 2, 2]
            if not noOutput:
                print(f"Target size parameter set to: {self.size[0]} nm")

        # Construct nanoparticle based on shape
        if self.shape == "sphere":
            if not noOutput:
                print(f"Making a sphere with target radius = {self.size[0]/2:.3f} nm")
            self.makeSuperCell(noOutput)
            self.makeSphere(noOutput)

        elif self.shape == "ellipsoid":
            if not noOutput:
                print(
                    f"Making an ellipsoid with target radii = "
                    f"{self.size[0]/2} {self.size[1]/2} {self.size[2]/2} nm"
                )
            self.makeSuperCell(noOutput)
            self.makeEllipsoid(noOutput)

        elif self.shape == "parallelepiped":
            if not noOutput:
                print(
                    f"Making a parallelepiped with target side length = {self.size} nm, "
                    f"directions = {list(self.directionsPPD)}"
                )
            self.makeSuperCell(noOutput)
            self.makeParallelepiped(noOutput)

        elif self.shape == "supercell":
            if not noOutput:
                print(f"Supercell side length = {self.size} nm")
            if len(self.size) != 3:
                sys.exit(
                    "Please enter lengths along a, b, c axes: size=[l_a, l_b, l_c]"
                )
            self.makeSuperCell(noOutput)

        elif self.shape == "wire":
            if not noOutput:
                print(
                    f"Wire in the {self.directionWire} direction. "
                    f"Target dimensions: Length x Width = {self.size[1]} x {self.size[0]} nm"
                )
                print(
                    f"Reference plane = {self.refPlaneWire}, "
                    f"{self.nRotWire}-fold rotation around {self.directionWire}"
                )
            if not pyNMBu.isPlaneParrallel2Line(
                self.refPlaneWire, self.directionWire
            ):
                print(
                    f"{bg.DARKREDB}Warning! Reference plane is not parallel "
                    f"to {self.directionWire}. Are you sure?{fg.OFF}"
                )
                suggested_plane = pyNMBu.returnPlaneParallel2Line(
                    self.directionWire
                )
                print(f"You can try: {suggested_plane}")
            else:
                if not noOutput:
                    print(
                        f"{bg.LIGHTGREENB}Reference plane is parallel "
                        f"to {self.directionWire}{fg.OFF}"
                    )
            self.makeSuperCell(noOutput)
            self.makeWire(noOutput)

        elif self.shape == "cylinder":
            if not noOutput:
                print(
                    f"Cylinder in the {self.directionCylinder} direction. "
                    f"Length x Width = {self.size[1]} x {self.size[0]} nm"
                )
            self.makeSuperCell(noOutput)
            self.makeCylinder(noOutput)

        elif "Wulff" in self.shape:
            if self.WulffShape is not None:
                self.predefinedParameters4WulffShapes(noOutput)
            # Validate parameters
            if self.surfacesWulff is None:
                sys.exit(
                    "Wulff construction requested but no planes defined. "
                    "Set 'surfacesWulff' parameter."
                )
            if (
                self.eSurfacesWulff is None
                and self.sizesWulff is None
            ):
                sys.exit(
                    "Either 'eSurfacesWulff' or 'sizesWulff' must be set"
                )
            if (
                len(self.surfacesWulff) != len(self.eSurfacesWulff)
                and len(self.surfacesWulff) != len(self.sizesWulff)
            ):
                sys.exit(
                    "'surfacesWulff' and energy/size lists "
                    "have different dimensions"
                )
            self.makeSuperCell(noOutput)
            self.makeWulff(noOutput)

        # For Wulff-wire, also measure wire dimensions
        if "wire" in self.shape:
            axis = pyNMBu.normV(np.array(self.directionWire))
            proj = self.NP.positions @ axis
            length_measured = proj.max() - proj.min()
            perp = self.NP.positions - np.outer(proj, axis)
            radii = np.linalg.norm(perp, axis=1)
            radius_measured = radii.max()

            self.length = length_measured
            self.radius = radius_measured
            self.sasview_dims = [radius_measured, length_measured]
            self.volume = math.pi * (radius_measured ** 2) * length_measured
            if not noOutput:
                print(
                    f"Measured wire dimensions: "
                    f"length = {length_measured:.2f} Å, "
                    f"diameter = {2*radius_measured:.2f} Å"
                )

        # Update final attributes
        self.nAtoms = len(self.NP.get_positions())
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None:
            self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)

        if not noOutput:
            print(f"Total number of atoms = {self.nAtoms}")

    def predefinedParameters4WulffShapes(self, noOutput):
        """
        Assign pre-defined parameters for Wulff shapes.

        Retrieves pre-defined properties (truncation planes, symmetry, MOI) for
        Wulff shapes from the internal database. Validates shape compatibility
        with crystal lattice.

        Args:
            noOutput (bool): If False, details are printed.

        Note:
            Updates self.eSurfacesWulff, self.surfacesWulff, and self.symWulff.
        """
        if self.WulffShape in data.WulffShapes.WSdf.index:
            self.eSurfacesWulff = data.WulffShapes.WSdf["relative energies"].loc[
                self.WulffShape
            ]
            self.surfacesWulff = data.WulffShapes.WSdf["planes"].loc[
                self.WulffShape
            ]
            self.symWulff = data.WulffShapes.WSdf["apply symmetry"].loc[
                self.WulffShape
            ]

            if not noOutput:
                print(f"{hl.BOLD}Selected shape{hl.OFF}")
                display(data.WulffShapes.WSdf.loc[self.WulffShape])

                expected_lattice = data.WulffShapes.WSdf["lattice system"].loc[
                    self.WulffShape
                ]
                if expected_lattice != self.ucBL.lattice_system:
                    print(
                        f"{bg.DARKREDB}Expected lattice system: "
                        f"{expected_lattice}.\n"
                        f"Crystal lattice: {self.ucBL.lattice_system}. "
                        f"Miller indices may not be meaningful.{fg.OFF}"
                    )
                else:
                    print(
                        f"{bg.LIGHTGREENB}Lattice system "
                        f"({expected_lattice}) matches crystal "
                        f"({self.ucBL.lattice_system}){bg.OFF}"
                    )
        else:
            display(data.WulffShapes.WSdf)
            sys.exit(
                f"{fg.RED}{bg.LIGHTREDB}Wulff shape '{self.WulffShape}' "
                f"not found in predefined shapes.{fg.OFF}"
            )

    def prop(self, noOutput):
        """
        Display unit cell and nanoparticle properties.

        Args:
            noOutput (bool): If False, details are printed.
        """
        if not noOutput:
            vID.centertxt(
                "Unit cell properties", bgc="#007a7a", size="14", weight="bold"
            )
            pyNMBu.print_ase_unitcell(self)
            vID.centertxt("Properties", bgc="#007a7a", size="14", weight="bold")
            print(self)

    def propPostMake(
        self, skip_symmetry_analysis, thresholdCoreSurface, noOutput
    ):
        """
        Compute post-construction nanoparticle properties.

        Calculates moments of inertia, symmetry analysis (optional), core/surface
        differentiation, and geometric properties via convex hull.

        Args:
            skip_symmetry_analysis (bool): If True, skip symmetry analysis.
            thresholdCoreSurface (float): Threshold for core/surface differentiation.
            noOutput (bool): If False, details are printed.

        Note:
            Updates self.moi, self.moisize, MOI size from inertia tensor.
            Updates self.NPcs with surface atoms marked with Nobelium (102).
        """
        # Compute moment of inertia
        self.moi = pyNMBu.moi(self.NP, noOutput)
        self.moisize = np.array(
            pyNMBu.moi_size(self.NP, noOutput)
        )  # Mass-normalized MOI

        if not skip_symmetry_analysis:
            pyNMBu.MolSym(self.NP, noOutput=noOutput)

        # Analyze convex hull and core/surface atoms
        (
            self.vertices,
            self.simplices,
            self.neighbors,
            self.equations,
        ), surface_atoms = pyNMBu.coreSurface(
            self, thresholdCoreSurface, noOutput=noOutput
        )

        # Generate JMol visualization if enabled
        if self.jmolCrystalShape:
            self.jMolCS = pyNMBu.defCrystalShapeForJMol(self, noOutput=True)

        # Mark surface atoms (Nobelium=102 for visualization)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surface_atoms)] = 102
        self.surfaceatoms = self.NPcs[surface_atoms]

        # Compute inscribed and circumscribed sphere radii
        pyNMBu.Inscribed_circumscribed_spheres(self, noOutput=noOutput)
 