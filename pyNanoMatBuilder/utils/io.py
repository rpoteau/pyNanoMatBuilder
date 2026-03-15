import time, datetime
import importlib
import os
import pathlib
from pathlib import Path
import re

import numpy as np
from scipy import linalg
import math
import sys

from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase import io as ase_io
from ase.spacegroup import get_spacegroup
from ase.visualize import view

from importlib import resources

from pyNanoMatBuilder import data
from .core import (pyNMB_location, get_resource_path, timer, RAB, Rbetween2Points,
                   vector, vectorBetween2Points, coord2xyz, vertex, vertexScaled, RadiusSphereAfterV,
                   centerOfGravity, center2cog, normOfV, normV, centerToVertices, Rx, Ry, Rz,
                   EulerRotationMatrix, plotPalette, rgb2hex, clone, deleteElementsOfAList,
                   planeFittingLSF, AngleBetweenVV, signedAngleBetweenVV
                   )
from .core import centertxt, centerTitle, fg, bg, hl, color
from .crystals import G, Gstar, print_ase_unitcell

#############################################################################################
def returnUnitcellData(system):
    """
    Function that calculates various unit cell properties from the `system.cif` object 
    and assigns them to attributes within the `system` instance.

    Args:
        An instance of the Crystal class containing CIF file data.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # ase analyzis
    system.ucUnitcell = system.cif.cell.cellpar()
    system.ucV = cellpar_to_cell(system.ucUnitcell)
    system.ucBL = system.cif.cell.get_bravais_lattice()
    # system.ucSG = get_spacegroup(system.cif, symprec=system.aseSymPrec) #deprecated
    system.ucVolume = system.cif.cell.volume
    system.ucReciprocal = np.array(system.cif.cell.reciprocal())
    system.ucFormula = system.cif.get_chemical_formula()
    system.G = G(system)
    system.Gstar = Gstar(system)

    # Pymatgen Symmetry Analysis (Replacing ase get_spacegroup)
    # Convert ASE Atoms to Pymatgen Structure
    pmg_struct = AseAtomsAdaptor.get_structure(system.cif)
    # Analyze symmetry
    sga = SpacegroupAnalyzer(pmg_struct, symprec=system.aseSymPrec)
    # Store the analyzer object so other methods (like makeWulff) can use it
    system.sga = sga
    # Store the Spacegroup Info (Spglib format)
    # This replaces the old system.ucSG
    system.ucSG_symbol = sga.get_space_group_symbol()
    system.ucSG_number = sga.get_space_group_number()
    system.ucCrystalSystem = sga.get_crystal_system()


def listCifsOfTheDatabase():
    """Display all CIF filenames in the database."""

    try:
        # We target a dummy file or just the directory if supported by the OS
        db_path = Path(get_resource_path("resources", "cif_database"))
    except:
        # Fallback: get the path to the folder directly
        db_path = Path(pyNMB_location()) / "resources" / "cif_database"

    print(f"Path to cif database: {db_path}")
    sgITField = "_space_group_IT_number"
    sgHMField = "_symmetry_space_group_name_H-M"

    class Crystal:
        pass

    for cif in db_path.rglob('*.cif'):
        relative_name = cif.relative_to(db_path)
        centertxt(f"{relative_name}", size=14, weight='bold')
        cifContent = ase_io.read(cif)
        cifFile = open(cif, 'r')
        cifFileLines = cifFile.readlines()
        re_sgIT = re.compile(sgITField)
        re_sgHM = re.compile(sgHMField)
        for line in cifFileLines:
            if re_sgIT.search(line):
                sgIT = ' '.join(line.split()[1:])
            if re_sgHM.search(line):
                sgHM = ' '.join(line.split()[1:])
        cifFile.close()
        c = Crystal()
        c.cif = cifContent
        c.aseSymPrec = 1e-4
        returnUnitcellData(c)
        print_ase_unitcell(c)
        color = "fg.RED"
        print()
        if int(sgIT) == c.ucSG_number:
            print(
                f"{fg.GREEN}Symmetry in the cif file = {sgIT}   {sgHM}"
                f"{hl.BOLD} in agreement with the pymatgen symmetry analyzis{fg.OFF}"
            )
        else:
            print(
                f"{fg.RED}Symmetry in the cif file = {sgIT}   {sgHM}"
                f"{hl.BOLD} disagrees with the pymatgen symmetry analyzis (Group Number = {c.ucSG_number}){fg.OFF}"
            )

######################################## cif files informations
def get_crystal_type(self):
    """
    Find the Bravais lattice based on the space group number.

    Returns:
        str: Bravais lattice
    """
    spacegroup_number = self.ucSG_number  # space group number

    # Bravais lattice based on space group number https://fr.wikipedia.org/wiki/Groupe_d%27espace
    if 195 <= spacegroup_number <= 230:  # Cubic
        if spacegroup_number == 225:
            return 'fcc'
        elif spacegroup_number == 229:
            return 'bcc'
        else:
            return 'cubic'
    elif 168 <= spacegroup_number <= 194:  # Hexagonal
        return 'hcp'
    elif 75 <= spacegroup_number <= 142:  # Tetragonal
        return 'tetragonal'
    elif 16 <= spacegroup_number <= 74:  # Orthorhombic
        return 'orthorhombic'
    elif 3 <= spacegroup_number <= 15:  # Monoclinic
        return 'monoclinic'
    elif 1 <= spacegroup_number <= 2:  # Triclinic
        return 'triclinic'
    else:
        return 'unknown'

def extract_cif_info(self, cif_file):
    """
    Extract useful information from a CIF file.

    Args:
        cif_file: CIF file.

    Returns:
        dict: A dictionary containing extracted CIF information:
            - 'cif_path' (Path): Absolute path to the CIF file.
            - 'crystal_type' (str): The crystal type.
            - 'Unitcell_param' (list): Unit cell parameters [a, b, c, α, β, γ].
            - 'ucBL' (str): Bravais lattice type.

    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    # structure = read(cif_file)  # load structure with ase, not used ?
    # self.ucUnitcell[0]=a, self.ucUnitcell[1]=b, self.ucUnitcell[2]=c,
    # self.ucUnitcell[3]= α, etc

    # 1. Basic ASE extractions
    self.ucUnitcell = self.cif.cell.cellpar()
    self.parameters = self.cif.cell.lengths()
    self.ucBL = self.cif.cell.get_bravais_lattice()  # HEX, CUB, etc
    # self.ucSG = get_spacegroup(self.cif, symprec=float(1e-2)) #deprecated. Moved to pymatgen symmetry analyzer
    self.ucFormula = self.cif.get_chemical_formula()

    # Pymatgen Symmetry Analysis
    # Convert ASE Atoms to Pymatgen Structure
    pmg_struct = AseAtomsAdaptor.get_structure(self.cif)
    # Initialize the analyzer (symprec 1e-2 matches the original code)
    sga = SpacegroupAnalyzer(pmg_struct, symprec=1e-2)
    # Get trustworthy symmetry data
    self.ucSG_number = sga.get_space_group_number()
    self.ucSG_symbol = sga.get_space_group_symbol()

    self.crystal_type = get_crystal_type(self)
    return {
        # 'crystal_name': self.ucFormula,
        'cif_path': cif_file,
        'crystal_type': self.crystal_type,
        'Unitcell_param': self.ucUnitcell,
        'ucBL': self.ucBL,
    }


def load_cif(self, cif_file, noOutput):
    """
    Loads a CIF file and extracts its information if it has not been loaded before.
    Args:
        cif_file: a CIF file.
        noOutput (bool): If False, prints the CIF file path.
    Return:
        dict: Extracted CIF information (from `extract_cif_info`).
    Notes:
    - CIF files are assumed to be stored in the "cif_database" directory.
    - If the file has already been loaded, its information is retrieved from `self.loaded_cifs`.

    """
    cif_folder = "cif_database"
    path2cif = Path(get_resource_path('resources/cif_database', cif_file))
    self.cif = ase_io.read(path2cif)
    if not noOutput:
        print("Absolute path to CIF:", path2cif)
    if not path2cif.exists():
        raise FileNotFoundError(f"File {cif_file} not found.")
    if path2cif not in self.loaded_cifs:
        self.loaded_cifs[path2cif] = extract_cif_info(self, path2cif)
    return self.loaded_cifs[path2cif]


#######################################################################
def ciflist(dbFolder="resources/cif_database"):
    """
    Function that prints the CIF files in the built-in dataset.
    Args:
        dbFolder: The database folder name (default is `resources/cif_database`).
    """
    path2cif = Path(get_resource_path(dbFolder, ""))
    if path2cif.exists():
        print(os.listdir(path2cif))
    else:
        print(f"Folder {dbFolder} not found.")

######################################## coupling with Jmol & DebyeCalculator
def check_jmol():
    from pyNanoMatBuilder import data
    from pathlib import Path
    
    # We check the CURRENT value of the variable in RAM
    path = Path(data.pyNMBvar.path2Jmol) / "JmolData.jar"
    
    if not path.exists():
        print("---")
        print("💡 Jmol not found. To enable image rendering, please set the path:")
        print("from pyNanoMatBuilder import data")
        print("data.pyNMBvar.path2Jmol = '/your/path/to/jmol'")
        print("---")
        return False
    return True
    
def saveCoords_DrawJmol(asemol, prefix, scriptJ="", boundaries=False, noOutput=True):
    """
    Save coordinates and generate a Jmol visualization.

    Args:
        asemol: ASE Atoms object to visualize.
        prefix (str): Filename prefix for output files.
        scriptJ (str): Additional Jmol script commands.
        boundaries (bool): If True, draws boundaries without facets script.
        noOutput (bool): If True, suppresses command output.
    """
    from pyNanoMatBuilder import data
    path2Jmol = Path(data.pyNMBvar.path2Jmol)
    jar_file = path2Jmol / "JmolData.jar"
    # fxyz = "./figs/" + prefix + ".xyz"
    # writexyz(fxyz, asemol)

    # Output directory for the USER (Working Directory)
    # We save results in a local 'figs' folder so the user can see them
    user_output_dir = Path("figs")
    user_output_dir.mkdir(exist_ok=True)
    fxyz = user_output_dir / f"{prefix}.xyz"
    writexyz(str(fxyz), asemol)

    if not jar_file.exists():
        if not noOutput:
            print(f"\n{fg.RED}⚠️ Jmol not found at: {jar_file}{fg.OFF}")
            print("The .xyz file was saved, but the .png rendering was skipped.")
            print(f"To fix this, set: {hl.BOLD}data.pyNMBvar.path2Jmol = 'your/path'{hl.OFF}\n")
        return # Graceful exit
    
    # if not boundaries:
    #     jmolscript = (
    #         scriptJ + '; frank off; cpk 0; wireframe 0.05; '
    #         'script "./figs/script-facettes-345PtLight.spt"; '
    #         'facettes345ptlight; draw * opaque;'
    #     )
    # else:
    #     jmolscript = scriptJ + '; frank off; cpk 0; wireframe 0.0; draw * opaque;'
    # jmolscript = (
    #     jmolscript +
    #     'set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; '
    #     'set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    # )
    # jmolcmd = (
    #     "java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz +
    #     " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    # )
    # if not noOutput:
    #     print(jmolcmd)
    # os.system(jmolcmd)
    try:
        internal_spt = get_resource_path("resources/figs", "script-facettes-345PtLight.spt")
    except FileNotFoundError:
        internal_spt = None

    # Build the Jmol Script
    if not boundaries and internal_spt:
        jmolscript = (
            f"{scriptJ}; frank off; cpk 0; wireframe 0.05; "
            f"script '{internal_spt}'; "  # Points to the internal resource
            "facettes345ptlight; draw * opaque;"
        )
    else:
        jmolscript = f"{scriptJ}; frank off; cpk 0; wireframe 0.0; draw * opaque;"

    # Save the PNG to the USER'S local figs folder
    output_png = user_output_dir / f"{prefix}.png"
    jmolscript += (
        "set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; "
        f"set zShade ON; set zShadePower 1; write image pngt 1024 1024 '{output_png}';"
    )

    jmolcmd = (
        f"java -Xmx512m -jar {path2Jmol}/JmolData.jar {fxyz} "
        f"-ij \"{jmolscript}\" >/dev/null "
    )

    if not noOutput:
        print(f"Saving to: {output_png}")
    os.system(jmolcmd)

# def DrawJmol(mol, prefix, scriptJ=""):
#     """
#     Generate a Jmol visualization from an existing XYZ file.

#     Args:
#         mol (str): Molecule filename (without extension).
#         prefix (str): Output image filename prefix.
#         scriptJ (str): Additional Jmol script commands.
#     """
#     path2Jmol = '/usr/local/src/jmol-14.32.50'
#     fxyz = "./figs/" + mol + ".xyz"
#     jmolscript = (
#         scriptJ + '; frank off; set specularPower 80; set antialiasdisplay; '
#         'set background [xf1f2f3]; set zShade ON;set zShadePower 1; '
#         'write image pngt 1024 1024 ./figs/'
#     )
#     jmolcmd = (
#         "java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz +
#         " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
#     )
#     if not noOutput:
#         print(jmolcmd)
#     os.system(jmolcmd)

def DrawJmol(mol, prefix, scriptJ="", noOutput=True):
    """
    Generate a Jmol visualization from an existing XYZ file.
    """
    from pyNanoMatBuilder import data
    path2Jmol = data.pyNMBvar.path2Jmol  # Use the user-configurable path
    
    # 1. Define the local working directory for the user's files
    user_figs_dir = Path("figs")
    user_figs_dir.mkdir(exist_ok=True)
    
    # 2. Locate the input XYZ (assumed to be in the local figs folder)
    fxyz = user_figs_dir / f"{mol}.xyz"
    if not fxyz.exists():
        if not noOutput:
            print(f"Error: {fxyz} not found. Cannot generate image.")
        return

    # 3. Build the Jmol script
    # We save the .png to the local working directory 'figs/'
    output_png = user_figs_dir / f"{prefix}.png"
    
    jmolscript = (
        f"{scriptJ}; frank off; set specularPower 80; set antialiasdisplay; "
        "set background [xf1f2f3]; set zShade ON; set zShadePower 1; "
        f"write image pngt 1024 1024 '{output_png}';"
    )

    # 4. Assemble the command
    jmolcmd = (
        f"java -Xmx512m -jar {path2Jmol}/JmolData.jar {fxyz} "
        f"-ij \"{jmolscript}\" >/dev/null "
    )

    if not noOutput:
        print(f"Generating Jmol image: {output_png}")
    
    os.system(jmolcmd)

#######################################################################
######################################## Functions that writes xyz, cif, jmol script files

def write(filename: str, atoms, wa='w', **kwargs):
    """
    Unified write function for pyNanoMatBuilder.

    This function serves as a central hub for exporting data. It handles directory 
    creation automatically and routes the data to the appropriate writer based 
    on the file extension.

    Args:
        filename (str): Path to the output file (e.g., 'coords/np.xyz').
        atoms (ase.Atoms or str): The atomic structure to save, or a string 
            containing script content for .script/.spt files.
        wa (str, optional): Write mode. 'w' for overwrite (default) or 'a' 
            for append (useful for multi-frame trajectories).
        **kwargs: Additional arguments passed to the underlying ASE write 
            function (e.g., 'format').

    Note:
        - Automatically creates parent directories if they do not exist.
        - Uses internal 'writexyz' for .xyz files to preserve custom headers.
        - Uses ASE for crystallography formats (.cif, .res, .pdb, etc.).
        - Handles raw text writing for Jmol scripts (.script, .spt).
    """
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    extension = file_path.suffix.lower()

    if extension == ".xyz":
        # Call your custom function for the specific pyNMB header
        writexyz(filename, atoms, wa=wa)
    
    elif extension in [".script", ".spt"]:
        # Handle Jmol scripts (which are strings, not Atoms objects)
        with open(file_path, wa) as f:
            f.write(atoms)
            
    else:
        # For everything else (.cif, .pdb, etc.), use the power of ASE
        # Translate 'wa' for ASE
        # If wa is 'a', then is_append is True.
        is_append = (wa == 'a')
        ase_io.write(filename, atoms, append=is_append, **kwargs)
        
def writexyz(filename: str,
             atoms: Atoms,
             wa: str='w'):
    """
    Simple xyz writing, with atomic symbols/x/y/z and no other information.
    Automatically creates the parent directories if they do not exist.
    Args:
        filename (str): Output filename.
        atoms (Atoms): ASE Atoms object.
        wa (str): Write mode ('w' or 'a').
    """
    from collections import Counter
    element_array = atoms.get_chemical_symbols()
    # extract composition in dict form - optimized with Counter
    composition = dict(Counter(element_array))

    coord = atoms.get_positions()
    natoms = len(element_array)
    line2write = '%d \n' % natoms
    line2write += '%s\n' % str(composition)
    for i in range(natoms):
        line2write += (
            '%s' % str(element_array[i]) +
            '\t %.8f' % float(coord[i, 0]) +
            '\t %.8f' % float(coord[i, 1]) +
            '\t %.8f' % float(coord[i, 2]) + '\n'
        )
        
    # Ensure the directory exists
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, wa) as file:
        file.write(line2write)

def create_data_csv(path_of_files, path_of_csvfiles, noOutput):
    """
    Extract dictionaries from XYZ/CSV files and create new CSV files.

    Args:
        path_of_files: Path of the directory containing the files.
        path_of_csvfiles: Path of the directory where the CSV file will be created.
        noOutput: If True, suppresses print statements.

    Returns:
        None
    """
    number_created_files = 0

    # Check if the directories exist
    if not os.path.isdir(path_of_files):
        raise FileNotFoundError(f"Directory '{path_of_files}' does not exist.")
    if not os.path.isdir(path_of_csvfiles):
        raise FileNotFoundError(f"Directory '{path_of_csvfiles}' does not exist.")

    # Loop through the files in the directory
    for filename in os.listdir(path_of_files):
        if filename.endswith(".xyz") or filename.endswith(".csv"):
            if not noOutput:
                print('File used:', filename)

            structure_source = os.path.join(path_of_files, filename)
            base_name = os.path.splitext(os.path.basename(structure_source))[0]
            csv_file_name = f"{base_name}_data.csv"
            csv_file = os.path.join(path_of_csvfiles, csv_file_name)

            # Write the dictionary in the new CSV file
            with open(csv_file, 'w') as csvfile:
                number_created_files += 1
                with open(structure_source, 'r') as file:
                    lignes = file.readlines()
                    if len(lignes) >= 2:
                        line_metadata = lignes[1].strip()
                        csvfile.write(f"{line_metadata}\n")

                        if not noOutput:
                            print(f"\n\033[1mNew file created: {csv_file}\033[0m\n")
                    else:
                        print(f"Error: File {filename} does not have enough lines.")

        else:
            if not noOutput:
                print(f"File format error: {filename}")

    print('Total CSV files created:', number_created_files)

######################################## Misc for plots
def imageNameWithPathway(imgName):
    """
    Construct the full file path for an image.

    Constructs the full file path for an image by joining the base directory
    with the image name.

    Args:
        imgName (str): The name of the image file.

    Returns:
        str: The full file path to the image file.
    """
    return get_resource_path('resources/figs', imgName)


def plotImageInPropFunction(imageFile):
    """
    Plot an image using matplotlib with no axes and a specified size.

    Args:
        imageFile: The path to the image file to be displayed.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image = mpimg.imread(imageFile)
    plt.figure(figsize=(2, 10))
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()

######################################## simple file management utilities

def createDir(path2, forceDel=False):
    """
    Creates a directory at the specified path.

    If the directory already exists, it will either be left unchanged or
    deleted and recreated based on the 'forceDel' argument.

    Args:
        path2 (str): The path where the directory should be created.
        forceDel (bool, optional): If set to True, will delete the existing
            directory and recreate it. Default is False.

    Returns:
        None
    """
    import shutil

    if os.path.isdir(path2) and not forceDel:
        print(f"{path2} already exists. No need to recreate it")
    if os.path.isdir(path2) and forceDel:
        print(f"{fg.RED}Previously created {path2} is deleted{fg.OFF}")
        shutil.rmtree(path2)
    if (os.path.isdir(path2) and forceDel) or not os.path.isdir(path2):
        print(f"{fg.BLUE}{path2} is created{fg.OFF}")
        os.mkdir(path2)


