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
from .geometry import reduceHullFacets
from .prop import kDTreeCN
from .io import writexyz

######################################## coupling with Jmol 

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
    
    try:
        internal_spt = get_resource_path("resources/figs", "script-facettes-345PtLight.spt")
    except FileNotFoundError:
        internal_spt = None

    # Build the Jmol Script
    if not boundaries and internal_spt:
        jmolscript = scriptJ + (
            f"; frank off; cpk 0; wireframe 0.05; "
            f"script '{internal_spt}'; "  # Points to the internal resource
            "facettes345ptlight; draw * opaque;"
        )
    else:
        jmolscript = scriptJ + "; frank off; cpk 0; wireframe 0.0; draw * opaque;"

    # Save the PNG to the USER'S local figs folder
    output_png = user_output_dir / f"{prefix}.png"
    jmolscript += (
        "set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; "
        f"set zShade ON; set zShadePower 1; write image pngt 1024 1024 '{output_png}';"
    )

    jmolcmd = (
        f"java -Xmx512m -jar {path2Jmol}/JmolData.jar {fxyz} "
        f"-ij '{jmolscript}' >/dev/null "  
    )
    if not noOutput:
        print(f"Saving to: {output_png}")
        print(jmolcmd)
    os.system(jmolcmd)

#######################################################################

def defCrystalShapeForJMol(self,
                           noOutput: bool=True,
                          ):
    """
    Generate a Jmol command to visualize the crystal shape based on the facets of the NP.

    Args:
        noOutput (bool): If True, suppresses the output of the command.

    Returns:
        str: The Jmol command string for visualizing the crystal shape.
    """
    useWulff = hasattr(self, 'trPlanes_Wulff') and self.trPlanes_Wulff is not None

    target_planes = getattr(self, 'trPlanes_Wulff', None) if useWulff else None

    if target_planes is None:
        target_planes = getattr(self, 'trPlanes_opt', None) if self.is_optimized else getattr(self, 'trPlanes', None)

    # if target_planes is not None:
    #     vertices, redFacets = reduceHullFacets(self, noOutput=noOutput, useWulff=useWulff)

    if target_planes is not None:
        vertices, redFacets = reduceHullFacets(self, noOutput=noOutput, useWulff=useWulff)
        if not noOutput:
            centertxt(
                "generating the jmol command line to view the crystal shape",
                bgc='#cbcbcb',
                size='12',
                fgc='b',
                weight='bold',
            )
        cmd = ""
        for i, nf in enumerate(redFacets):
            cmd += "draw facet" + str(i) + " polygon "
            cmd += '['
            for at in nf:
                cmd += f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
            cmd += "]; "
        cmd += "color $facet* translucent 70 [x828282]"
        cmde = ""
        index = 0
        for nf in redFacets:
            nfcycle = np.append(nf, nf[0])
            for i, at in enumerate(nfcycle[:-1]):
                cmde += "draw line" + str(index) + " ["
                cmde += f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
                cmde += (
                    f"{{{vertices[nfcycle[i+1]][0]:.4f},"
                    f"{vertices[nfcycle[i+1]][1]:.4f},"
                    f"{vertices[nfcycle[i+1]][2]:.4f}}},"
                )
                cmde += "] width 0.2; "
                index += 1
        cmde += "color $line* [xd6d6d6]; "
        cmd = cmde + cmd
    else:  # sphere, ellipsoid
        cmd = ""
    if not noOutput:
        print("Jmol command: ", cmd)
    return cmd

def saveCN4JMol(Crystal: Atoms,
                save2: str='CN.dat',
                Rmax: float=3.0,
                noOutput: bool=False,
                ):
    """
    Calculates the coordination number (CN) for a given crystal and generates a Jmol command for visualization.
    
    Args:
        Crystal (Atoms): The crystal structure object.
        save2 (str, optional): The filename to save the coordination numbers. Defaults to 'CN.dat'.
        Rmax (float, optional): The maximum distance for neighbors when calculating CN. Defaults to 3.0.
        noOutput (bool, optional): If set to True, suppresses the output. Defaults to False.
    
    Returns:
        None
    """
    import seaborn as sns

    # Calculate the coordination number (CN) using a k-D tree method
    nn, CN = kDTreeCN(Crystal, Rmax, noOutput=noOutput)
    CNmin = np.min(CN)
    CNmax = np.max(CN)
    with open(save2, 'w') as f:
        for cn in CN:
            f.write(str(cn) + "\n")
    if not noOutput:
        uniqueCN = np.unique(CN)
        nColors = len(uniqueCN)
        print(f"CN range = [{CNmin} - {CNmax}]")
        print(f"CN = {uniqueCN}")
        CNMax = 16
        colorsFull = [
            (255, 0, 0), (255, 255, 153), (255, 255, 0), (255, 204, 0),
            (102, 255, 255), (51, 204, 255), (102, 153, 255), (249, 128, 130),
            (153, 255, 204), (0, 204, 153), (0, 134, 101), (0, 102, 102),
            (51, 51, 255), (102, 51, 0), (0, 51, 102), (77, 77, 77),
            (0, 0, 0)
        ]
        colorsFull = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in colorsFull]
        path, file = os.path.split(save2)
        prefix = file.split(".")
        fileColors = "./" + path + "/" + prefix[0] + "colors.png"
        fileColorsFull = "./" + path + "/" + "CN_color_palette.png"
        colorNamesFull = np.array(range(0, CNMax + 1))
        print("Full palette:")
        plotPalette(colorsFull, colorNamesFull, savePngAs=fileColorsFull)
        print(f"Palette specific to {prefix[0]}:")
        colors = []
        for c in uniqueCN:
            colors.append(colorsFull[c])
        plotPalette(colors, uniqueCN, savePngAs=fileColors)

        # Generate Jmol command for CN visualization
        print(f"{hl.BOLD}Jmol command:{hl.OFF}")
        command = f"{{*}}.valence = load('{file}'); "
        colorScheme = ""
        for c in colorsFull:
            colorScheme = colorScheme + rgb2hex(c) + " "
        command = command + f"color atoms property valence 'colorCN' RANGE 0 {CNMax} ;"
        command = (
            command +
            "label %2.0[valence]; color label yellow ; font label 24 ; set labeloffset 7 0;"
        )
        print(f"color 'colorCN = {colorScheme}';")
        print(command)

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
        print(jmolcmd)
    
    os.system(jmolcmd)

# def defHelixShapeForJMol(self, n_rings=50, n_sides=12, noOutput=True):
#     """
#     Generate a Jmol command to visualize the helical envelope as a triangulated tube.

#     The tube is built by connecting successive rings of n_sides points each,
#     placed perpendicular to the helix tangent at each sample point.
#     Each pair of adjacent rings generates 2*n_sides triangles.

#     Args:
#         n_rings (int): Number of rings along the helix (default: 50).
#         n_sides (int): Number of vertices per ring (default: 12).
#         noOutput (bool): If True, suppresses output. Default is True.

#     Returns:
#         str: Jmol command string for the helical tube.
#     """
#     from .geometry import normV
#     import numpy as np

#     if not hasattr(self, '_helix_params'):
#         print(f"{bg.DARKREDB}Warning: no helix parameters found. "
#               f"Call applyTorsion with profile='helix' first.{bg.OFF}")
#         return ""

#     p = self._helix_params
#     helix_radius = p['helix_radius']
#     pitch        = p['pitch']
#     axis_cart    = p['axis_cart']
#     L            = p['L']
#     e1           = p['e1']
#     e2           = p['e2']
#     wire_radius  = p['wire_radius']

#     pitch_factor = pitch / (2 * np.pi)
#     stretch = np.sqrt(1 + (helix_radius / pitch_factor)**2)

#     # Sample t values along the helix
#     t_values = np.linspace(0, L / pitch_factor / stretch, n_rings)

#     # --- Build all rings ---
#     # Each ring is a list of n_sides 3D points
#     rings = []
#     for t in t_values:
#         # Center of ring on the helix
#         center = (helix_radius * np.cos(t) * e1 +
#                   helix_radius * np.sin(t) * e2 +
#                   pitch_factor * t * axis_cart)
#         center += self.cog

#         # Frenet-Serret frame
#         tangent  = normV(-helix_radius * np.sin(t) * e1 +
#                           helix_radius * np.cos(t) * e2 +
#                           pitch_factor * axis_cart)
#         normal   = -np.cos(t) * e1 - np.sin(t) * e2
#         binormal = np.cross(tangent, normal)

#         # Ring points
#         angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
#         ring = [center + wire_radius * (np.cos(a) * normal +
#                                         np.sin(a) * binormal)
#                 for a in angles]
#         rings.append(ring)

#     # --- Center the tube on self.cog ---
#     # Compute center of gravity of all ring centers
#     all_centers = np.array([
#         helix_radius * np.cos(t) * e1 +
#         helix_radius * np.sin(t) * e2 +
#         pitch_factor * t * axis_cart
#         for t in t_values
#     ])
#     tube_cog = all_centers.mean(axis=0)
#     shift = self.cog - tube_cog
#     # Apply shift to all rings
#     rings = [[pt + shift for pt in ring] for ring in rings]

#     # --- Build triangulated surface between adjacent rings ---
#     cmd = ""
#     face_idx = 0
#     for i in range(len(rings) - 1):
#         r0 = rings[i]
#         r1 = rings[i + 1]
#         for j in range(n_sides):
#             j1 = (j + 1) % n_sides

#             # Two triangles per quad between rings
#             # Triangle 1: r0[j], r0[j1], r1[j]
#             p0 = r0[j]
#             p1 = r0[j1]
#             p2 = r1[j]
#             cmd += f"draw htube{face_idx} polygon ["
#             cmd += f"{{{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}}},"
#             cmd += f"{{{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}}},"
#             cmd += f"{{{p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f}}},"
#             cmd += "]; "
#             face_idx += 1

#             # Triangle 2: r1[j], r0[j1], r1[j1]
#             p0 = r1[j]
#             p1 = r0[j1]
#             p2 = r1[j1]
#             cmd += f"draw htube{face_idx} polygon ["
#             cmd += f"{{{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}}},"
#             cmd += f"{{{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}}},"
#             cmd += f"{{{p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f}}},"
#             cmd += "]; "
#             face_idx += 1

#     cmd += "color $htube* translucent 70 [x828282]; "

#     if not noOutput:
#         print(f"Helix tube: {face_idx} triangles, "
#               f"{n_rings} rings x {n_sides} sides")
#         print("Jmol command: ", cmd)
#     return cmd

def defHelixShapeForJMol(self, n_rings=50, n_sides=12, noOutput=True):
    """
    Generate a Jmol command to visualize the helical envelope as a triangulated tube.

    The tube is built by connecting successive rings of n_sides points each,
    placed perpendicular to the helix tangent at each sample point.
    Each pair of adjacent rings generates 2*n_sides triangles.
    The tube is automatically centered on the NP center of mass.

    Args:
        n_rings (int): Number of rings along the helix (default: 50).
        n_sides (int): Number of vertices per ring (default: 12).
        noOutput (bool): If True, suppresses output. Default is True.

    Returns:
        str: Jmol command string for the helical tube.
    """
    from .geometry import normV

    if not hasattr(self, '_helix_params'):
        print(f"{bg.DARKREDB}Warning: no helix parameters found. "
              f"Call applyTorsion with profile='helix' first.{bg.OFF}")
        return ""

    p            = self._helix_params
    helix_radius = p['helix_radius']
    pitch        = p['pitch']
    axis_cart    = p['axis_cart']
    L            = p['L']
    e1           = p['e1']
    e2           = p['e2']
    wire_radius  = p['wire_radius']
    cog_helix    = p['cog_helix']
    proj_min     = p['proj_min']

    pitch_factor = pitch / (2 * np.pi)
    stretch      = np.sqrt(1 + (helix_radius / pitch_factor)**2)

    # Sample t values along the helix
    t_start  = proj_min * 2 * np.pi / pitch / stretch
    t_end    = t_start + L * 2 * np.pi / pitch / stretch
    t_values = np.linspace(t_start, t_end, n_rings)

    # --- Build all rings ---
    rings = []
    for t in t_values:
        # Center of ring on the helix (unshifted)
        center = (helix_radius * np.cos(t) * e1 +
                  helix_radius * np.sin(t) * e2 +
                  pitch_factor * t * axis_cart)

        # Frenet-Serret frame
        tangent  = normV(-helix_radius * np.sin(t) * e1 +
                          helix_radius * np.cos(t) * e2 +
                          pitch_factor * axis_cart)
        normal   = -np.cos(t) * e1 - np.sin(t) * e2
        binormal = np.cross(tangent, normal)

        # Ring points
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        ring   = [center + wire_radius * (np.cos(a) * normal +
                                          np.sin(a) * binormal)
                  for a in angles]
        rings.append(ring)

    # --- Center the tube on self.cog ---
    # shift = self.cog - cog_helix
    # rings = [[pt + shift for pt in ring] for ring in rings]

    # --- Build triangulated surface between adjacent rings ---
    cmd      = ""
    face_idx = 0
    for i in range(len(rings) - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for j in range(n_sides):
            j1 = (j + 1) % n_sides

            # Triangle 1: r0[j], r0[j1], r1[j]
            p0, p1, p2 = r0[j], r0[j1], r1[j]
            cmd += f"draw htube{face_idx} polygon ["
            cmd += f"{{{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}}},"
            cmd += f"{{{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}}},"
            cmd += f"{{{p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f}}},"
            cmd += "]; "
            face_idx += 1

            # Triangle 2: r1[j], r0[j1], r1[j1]
            p0, p1, p2 = r1[j], r0[j1], r1[j1]
            cmd += f"draw htube{face_idx} polygon ["
            cmd += f"{{{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}}},"
            cmd += f"{{{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}}},"
            cmd += f"{{{p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f}}},"
            cmd += "]; "
            face_idx += 1

    cmd += "color $htube* translucent 70 [x828282]; "

    if not noOutput:
        print(f"Helix tube: {face_idx} triangles, "
              f"{n_rings} rings x {n_sides} sides")
        print("Jmol command: ", cmd)
    return cmd