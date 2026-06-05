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
from .core import kDTreeCN
from .io import writexyz

######################################## coupling with Jmol 

def check_jmol():
    
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
    
def saveCoords_DrawJmol(asemol, prefix, scriptJ="", boundaries=False, noOutput=True,
                        user_output_dir="figs", cpk=0, wireframe=0.05, saveXYZ=True,
                        widthLines=None):
    """
    Save coordinates and generate a Jmol visualization.

    Args:
        asemol: ASE Atoms object to visualize.
        prefix (str): Filename prefix for output files.
        scriptJ (str): Additional Jmol script commands.
        boundaries (bool): If True, draws boundaries without facets script.
        noOutput (bool): If True, suppresses command output.
        user_output_dir (str): Output directory for files.
        cpk (float): CPK radius for atom display.
        wireframe (float): Wireframe thickness.
        saveXYZ (bool): If True, saves the .xyz file.
        widthLines (float): If not None, overrides the default line width
            (0.2) used by the facettes345ptlight script for drawing edges.
    """
    path2Jmol = Path(data.pyNMBvar.path2Jmol)
    jar_file = path2Jmol / "JmolData.jar"
    # fxyz = "./figs/" + prefix + ".xyz"
    # writexyz(fxyz, asemol)

    # Output directory for the USER (Working Directory)
    # We save results in a local 'figs' folder so the user can see them
    user_output_dir = Path(user_output_dir)
    user_output_dir.mkdir(exist_ok=True)
    fxyz = user_output_dir / f"{prefix}.xyz"
    if saveXYZ:
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
    # if not boundaries and internal_spt:
    #     jmolscript = scriptJ + (
    #         f"; frank off; cpk 0; wireframe 0.05; "
    #         f"script '{internal_spt}'; "  # Points to the internal resource
    #         "facettes345ptlight; draw * opaque;"
    #     )
    # else:
    #     jmolscript = scriptJ + "; frank off; cpk 0; wireframe 0.0; draw * opaque;"
    if not boundaries and internal_spt:
        jmolscript = scriptJ + (
            f"; frank off; cpk {cpk}; wireframe {wireframe}; "
            f"script '{internal_spt}'; "
            "facettes345ptlight; draw * opaque;"
        )
    else:
        jmolscript = scriptJ + f"; frank off; cpk {cpk}; wireframe {wireframe}; draw * opaque;"
        
    # Save the PNG to the USER'S local figs folder
    output_png = user_output_dir / f"{prefix}.png"
    jmolscript += (
        "set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; "
        f"set zShade ON; set zShadePower 1; write image pngt 1024 1024 '{output_png}';"
    )
    
    if widthLines is not None:
        jmolscript = jmolscript.replace("width 0.2", f"width {widthLines}")
        
    # jmolcmd = (
    #     f"java -Xmx512m -jar {path2Jmol}/JmolData.jar {fxyz} "
    #     f"-ij '{jmolscript}' >/dev/null "  
    # )

    # write the Jmol script to a file (avoids "Argument list too long"
    # when the script contains thousands of draw commands)
    script_file = user_output_dir / f"{prefix}_tmp.spt"
    with open(script_file, "w") as f:
        f.write(jmolscript)

    jmolcmd = (
        f"java -Xmx512m -jar {path2Jmol}/JmolData.jar {fxyz} "
        f"-s '{script_file}' >/dev/null "
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

    For Wulff constructions and crystal shapes, uses HalfspaceIntersection to
    compute the reduced convex hull facets. For slicing planes (applySlicing),
    redirects to defSlicingPlanesForJMol since the resulting shape may be
    non-convex and cannot be represented by HalfspaceIntersection.

    Args:
        noOutput (bool): If True, suppresses the output of the command.

    Returns:
        str: The Jmol command string for visualizing the crystal shape.
    """
    useWulff   = hasattr(self, 'trPlanes_Wulff')   and self.trPlanes_Wulff   is not None
    useSlices  = hasattr(self, 'trPlanes_Slices')  and self.trPlanes_Slices  is not None

    # --- Slicing planes: non-convex possible → individual plane visualization ---
    if useSlices and not useWulff:
        self.jMolSlices = defSlicingPlanesForJMol(self, noOutput=noOutput)

    if useWulff:
        target_planes = self.trPlanes_Wulff
        # print("wulff")
    elif self.is_optimized:
        target_planes = getattr(self, 'trPlanes_opt', None)
        # print("is_optimized")
    else:
        target_planes = getattr(self, 'trPlanes', None)
        # print("other")

    # if target_planes is not None:
    #     vertices, redFacets = reduceHullFacets(self, noOutput=noOutput, useWulff=useWulff)

    # print(target_planes)

    if target_planes is not None:
        vertices, redFacets = reduceHullFacets(self, noOutput=noOutput, 
                                               tolAngle=4.0 if self.is_optimized else 2.0,
                                               useWulff=useWulff)
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
              f"Call applyTwist with profile='helix' first.{bg.OFF}")
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

def defSlabShapeForJMol(self, hkl, offset: float = 1.5, noOutput: bool = True):
    """
    Generate a Jmol command to visualize the slab surface plane as a
    translucent polygon, positioned just above the topmost atomic layer.

    The polygon is defined by the four corners of the slab surface cell
    (spanned by cell vectors a and b), centered on the slab center of mass,
    and placed at z_max + offset to aflush with the top atomic layer.
    Edges are drawn as thin lines and the plane is labeled with its
    Miller indices.

    Args:
        hkl (array-like): Miller indices [h, k, l] of the surface plane,
                          used for labeling only.
        offset (float): Vertical offset in Å above the topmost atomic layer
                        at which the surface polygon is placed. Should be
                        approximately half the atomic radius of the surface
                        element (default: 1.5 Å, suitable for Au, Ag, Pt).
        noOutput (bool): If True, suppresses printing of the Jmol command.
                         Default is True.

    Returns:
        str: Jmol command string defining the surface polygon, its edges,
             and its Miller index label.

    Note:
        This function is intended for use on objects returned by generateSlab().
        The polygon approximates the surface as a flat quadrilateral — it is
        exact only if the slab cell vectors a and b lie in the surface plane,
        which is the case for slabs generated by ASE or pymatgen surface builders.

    Example:

        slab = AuNP.generateSlab([1, 1, 5], size_a=2.0, size_b=2.0,
                                  min_thickness=10.0, backend='ase')
        script = slab.defSlabShapeForJMol([1, 1, 5], offset=1.7)
        pyNMBu.write('coords/Au_115.script', script)
    """
    pos = self.NP.get_positions()
    cell = self.NP.cell
    z_surface = pos[:, 2].max() + offset

    a = np.array(cell[0])
    b = np.array(cell[1])

    # Project onto xy plane
    a_xy = np.array([a[0], a[1], 0.0])
    b_xy = np.array([b[0], b[1], 0.0])

    # Center of the polygon = center of atomic positions in xy
    # NOT the cell origin
    xy_atoms_center = np.array([pos[:, 0].mean(), pos[:, 1].mean(), 0.0])
    xy_polygon_center = (a_xy + b_xy) / 2.0   # center of parallelogram

    # Shift origin so polygon center matches atomic center
    origin = xy_atoms_center - xy_polygon_center
    origin[2] = z_surface

    corners = np.array([
        origin,
        origin + a_xy,
        origin + a_xy + b_xy,
        origin + b_xy,
    ])

    h, k, l = hkl
    cmd = f"draw slab_face polygon ["
    for pt in corners:
        cmd += f"{{{pt[0]:.4f},{pt[1]:.4f},{pt[2]:.4f}}},"
    cmd += "]; "
    cmd += "color $slab_face translucent 60 [x828282]; "

    # Edges
    corners_cycle = np.vstack([corners, corners[0]])
    for i in range(len(corners)):
        p0 = corners_cycle[i]
        p1 = corners_cycle[i+1]
        cmd += f"draw slab_edge{i} [{{{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}}},"
        cmd += f"{{{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}}}] width 0.2; "
    cmd += "color $slab_edge* [xd6d6d6]; "

    # Label via JMol echo
    center = corners.mean(axis=0)
    cmd += (f"set echo slab_label {{{center[0]:.4f},{center[1]:.4f},{center[2]+2:.4f}}}; "
            f"echo ({h}{k}{l}); color echo yellow; font echo 24")

    if not noOutput:
        print(f"Jmol command for plane ({h}{k}{l}): ", cmd)
    return cmd

def defSlicingPlanesForJMol(self,
                             size: float = 30.0,
                             colors: list = None,
                             translucency: int = 70,
                             noOutput: bool = True):
    """
    Generate a Jmol command to visualize all slicing planes stored in
    self.trPlanes_Slicing as translucent square polygons.

    For each plane [nx, ny, nz, d], a square polygon is built centered
    on the closest point of the plane to the origin, in the plane itself.
    Planes from the same group share the same color.

    Args:
        self: pyNMBcore object with self.trPlanes_Slicing defined.
        size (float): Half-size of each plane polygon in Å. Default is 30.0.
        colors (list of str): List of hex colors (e.g. ['x0055ff', 'xff5500'])
            one per plane. If None, a default palette is used.
        translucency (int): Translucency of the planes in percent (0=opaque,
            100=translucent). Default is 70.
        noOutput (bool): If True, suppresses output. Default is True.

    Returns:
        str: Jmol command string.
    """
    from .geometry import normV, rotationMolAroundAxis

    if not hasattr(self, 'trPlanes_Slices') or self.trPlanes_Slices is None:
        print(f"{bg.LIGHTYELLOWB}Warning: no slicing planes found. "
              f"Run applySlicing() first.{bg.OFF}")
        return ""

    planes = np.array(self.trPlanes_Slices)

    # --- Compute NP extent for arrow length ---
    pos = self.NP.get_positions()
    extent = np.max(pos.max(axis=0) - pos.min(axis=0))  # max dimension in Å
    arrow_length = extent * 0.4  # 40% of NP size

    # Retrieve family index for each plane (saved by applySlicing)
    # Fallback: one family per plane if not available
    family = getattr(self, 'trPlanes_Slicing_groups',
                     list(range(len(planes))))

    # Map each family index to a color
    default_colors = [
        'x0055ff',  # blue
        'xff5500',  # orange
        'x00aa44',  # green
        'xcc0000',  # red
        'xaa00cc',  # purple
        'x00aacc',  # cyan
        'xffcc00',  # yellow
        'xff0088',  # pink
    ]
    color_by_family = [default_colors[f % len(default_colors)] for f in family]

    cmd = ""
    
    delete_by_plane = getattr(self, 'trPlanes_Slicing_delete',
                              ['above'] * len(planes))  # fallback
    
    for i, plane in enumerate(planes):
        n = plane[:3]
        d = plane[3]
        color = colors[i] if colors else color_by_family[i]

        # --- Center of the polygon: closest point of the plane to origin ---
        # P = -d * n  (since n is normalized)
        center = -d * n

        # --- Build two orthonormal vectors in the plane ---
        arbitrary = np.array([1, 0, 0]) if abs(n[0]) < 0.9 \
                    else np.array([0, 1, 0])
        u = np.cross(n, arbitrary)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        # --- 4 corners of the square polygon ---
        corners = np.array([
            center + size * ( u + v),
            center + size * (-u + v),
            center + size * (-u - v),
            center + size * ( u - v),
        ])

        # --- Polygon ---
        cmd += f"draw slplane{i} polygon ["
        for pt in corners:
            cmd += f"{{{pt[0]:.4f},{pt[1]:.4f},{pt[2]:.4f}}},"
        cmd += "]; "
        cmd += f"color $slplane{i} translucent {translucency} [{color}]; "

        # --- Edges ---
        corners_cycle = np.vstack([corners, corners[0]])
        for j in range(4):
            p0 = corners_cycle[j]
            p1 = corners_cycle[j+1]
            cmd += f"draw slpedge{i}_{j} ["
            cmd += f"{{{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}}},"
            cmd += f"{{{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}}},"
            cmd += "] width 0.15; "
        cmd += f"color $slpedge{i}_* [{color}]; "

        # --- Arrow points toward the deleted side ---
        delete = delete_by_plane[i] if i < len(delete_by_plane) else 'above'
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-10:
            outward = center / center_norm
        else:
            outward = n
        # outward points from origin toward the plane center
        # delete='below' → delete on the side OPPOSITE to outward → arrow = -outward
        # delete='above' → delete on the same side as outward → arrow = +outward
        # BUT if d was originally negative, outward is already inverted
        # → use sign of d to correct
        d_sign = 1.0 if d < 0 else -1.0
        arrow_dir = outward * d_sign if delete == 'above' else -outward * d_sign
        p_start = center
        p_end = center + arrow_length * arrow_dir
        cmd += f"draw slparrow{i} arrow "
        cmd += f"{{{p_start[0]:.4f},{p_start[1]:.4f},{p_start[2]:.4f}}} "
        cmd += f"{{{p_end[0]:.4f},{p_end[1]:.4f},{p_end[2]:.4f}}} "
        cmd += f"width 0.3; "
        cmd += f"color $slparrow{i} [{color}]; "

    if not noOutput:
        print(f"Jmol command for {len(planes)} slicing planes:")
        print(cmd)

    return cmd