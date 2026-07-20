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
from .core import kDTreeCN, _resolve_value_range
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
                           colorFacets=None,
                           useWulff=None,
                          ):
    """
    Generate a Jmol command to visualize the crystal shape based on the facets of the NP.

    For Wulff constructions and crystal shapes, uses HalfspaceIntersection to
    compute the reduced convex hull facets. For slicing planes (applySlicing),
    redirects to defSlicingPlanesForJMol since the resulting shape may be
    non-convex and cannot be represented by HalfspaceIntersection.

    Args:
        noOutput (bool): If True, suppresses the output of the command.
        colorFacets (str): If not None, overrides the default facet colour
            (gray [x828282]). Accepts a Jmol colour name ('red', 'gold', …),
            a hex code prefixed by 'x', '0x' or '#' ('xff3030'), normalised to
            Jmol's bracketed form, or a value already in brackets, passed
            through as-is.

    Returns:
        str: The Jmol command string for visualizing the crystal shape.
    """
        # auto-detect by default, but allow an explicit override
    if useWulff is None:
        useWulff = hasattr(self, 'trPlanes_Wulff') and self.trPlanes_Wulff is not None
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

        if colorFacets is None:
            cmd += "color $facet* translucent 70 [x828282]; "
        else:
            cf = str(colorFacets).strip()
            if cf.startswith('[') and cf.endswith(']'):
                body = cf
            elif cf.startswith(('#', '0x', 'x')):
                hexpart = cf.lstrip('#')
                if hexpart.startswith('0x'):
                    hexpart = hexpart[2:]
                elif hexpart.startswith('x'):
                    hexpart = hexpart[1:]
                body = f"[x{hexpart}]"
            else:
                body = cf                      # Jmol colour name
            cmd += f"color $facet* translucent 70 {body}; "
        
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

def saveCN4JMol(self,
                save2: str = 'CN.dat',
                Rmax: float = 3.0,
                is_optimized: bool = None,
                noOutput: bool = False,
                ):
    """
    Colour atoms by their coordination number (CN) in Jmol.

    Writes one CN class index per atom to a .dat file (loaded into Jmol via the
    'valence' channel) and prints the Jmol command to colour atoms by CN with a
    discrete palette (one colour per CN value). Reuses self.CN if it has already
    been computed by calculate_CN; otherwise computes it on the fly. The legend
    is a vertical set of circles, one per CN value present, each circle's area
    proportional to the number of atoms with that CN, annotated with
    'index | CN | atom count'.

    Note:
        The .dat file written for Jmol contains the per-atom CLASS INDEX (a
        contiguous 0-based index over the CN values present), not necessarily
        the raw CN, so that the colour palette is indexed contiguously. The raw
        CN is available in self.CN; the CN of each class is reported in the
        legend.

    Args:
        self: pyNMBcore instance (uses self.NP or self.NP_opt).
        save2 (str): file where the per-atom CN class index is written.
        Rmax (float): cutoff distance (Å), used only if the CN must be computed.
            Default 3.0.
        is_optimized (bool or None): target structure. None -> self.is_optimized.
        noOutput (bool): if True, suppresses output. Default False.

    Returns:
        None.
    """
    import numpy as np
    import os

    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    use_opt = is_optimized and getattr(self, 'NP_opt', None) is not None

    # --- CN of every atom (reuse self.CN if present) ---------------------
    from .prop import calculate_CN
    CN = getattr(self, 'CN_opt' if use_opt else 'CN', None)
    if CN is None:
        CN = calculate_CN(self, Rmax=Rmax, is_optimized=is_optimized,
                          noOutput=True)
    CN = np.asarray(CN, dtype=int)
    if CN.size == 0:
        print("No atoms; nothing to do.")
        return

    uniqueCN = np.unique(CN)                      # CN values present, sorted
    CNmin, CNmax = int(uniqueCN.min()), int(uniqueCN.max())

    # contiguous class index over the CN values present
    cn_to_class = {int(cn): i for i, cn in enumerate(uniqueCN)}
    cn_class = np.array([cn_to_class[int(c)] for c in CN], dtype=int)
    nClasses = len(uniqueCN)

    with open(save2, 'w') as f:
        for c in cn_class:
            f.write(str(int(c)) + "\n")

    # also write the real CN values (for Jmol labelling via a second channel)
    import os as _os
    root, ext = _os.path.splitext(save2)
    save2_values = root + "_values" + (ext or ".dat")
    with open(save2_values, 'w') as f:
        for c in CN:
            f.write(f"{int(c)}\n")

    if noOutput:
        return

    # --- WARNING: file holds the class index, not the raw CN -------------
    print(f"{bg.LIGHTYELLOWB}Warning: '{os.path.basename(save2)}' stores the "
          f"per-atom CLASS INDEX (0..{nClasses-1}) for Jmol colouring, not the "
          f"raw CN value. The raw CN is in self.CN; per-class CN values are in "
          f"the legend.{bg.OFF}")

    print(f"CN range = [{CNmin} - {CNmax}]")
    print(f"CN values present = {uniqueCN}")

    # --- palette indexed by CN value (keep your original colour mapping) -
    colorsFull = [
        (255, 0, 0), (255, 255, 153), (255, 255, 0), (255, 204, 0),
        (102, 255, 255), (51, 204, 255), (102, 153, 255), (249, 128, 130),
        (153, 255, 204), (0, 204, 153), (0, 134, 101), (0, 102, 102),
        (51, 51, 255), (102, 51, 0), (0, 51, 102), (77, 77, 77),
        (0, 0, 0)
    ]
    colorsFull = [(e[0]/255.0, e[1]/255.0, e[2]/255.0) for e in colorsFull]
    # colour of each class = colour indexed by its CN value (your convention)
    classColors = [colorsFull[int(cn)] for cn in uniqueCN]
    counts = np.array([int(np.count_nonzero(CN == cn)) for cn in uniqueCN])

    path, file = os.path.split(save2)
    prefix = file.split(".")[0]
    if path == "":
        path = "."
    fileColors = "./" + path + "/" + prefix + "colors.png"
    # value column = the CN value itself (as float for uniform formatting)
    plotPaletteBubbles(classColors, list(range(nClasses)),
                       uniqueCN.astype(float), counts,
                       savePngAs=fileColors, title=None,
                       val_header="CN", show=True)

    # --- Jmol command -----------------------------------------------------
    # 'valence' carries the class index -> drives the colour (indexed palette).
    # 'partialcharge' carries the real CN value -> drives the integer label.
    print(f"{hl.BOLD}Jmol command:{hl.OFF}")
    colorScheme = ""
    for c in classColors:
        colorScheme = colorScheme + rgb2hex(c) + " "
    file_values = os.path.basename(save2_values)
    command = (f"{{*}}.valence = load('{file}'); "
               f"color atoms property valence 'colorCN' RANGE 0 {nClasses-1} ; "
               f"{{*}}.partialcharge = load('{file_values}'); "
               f"label %.0[partialcharge]; color label yellow ; "
               f"font label 24 ; set labeloffset 7 0;")
    print(f"color 'colorCN = {colorScheme}';")
    print(command)
    print(f"{bg.LIGHTYELLOWB}Note: colour is driven by the class index "
          f"('{file}', channel valence); the label shows the real CN value "
          f"('{file_values}', channel partialcharge).{bg.OFF}")

def saveGCN4JMol(self,
                 save2: str = 'GCN.dat',
                 Rmax: float = 2.9,
                 cn_max: int = None,
                 nClasses: int = 12,
                 gcn_range: tuple = None,
                 split_at_bulk: bool = True,
                 is_optimized: bool = None,
                 noOutput: bool = False,
                 ):
    """
    Compute the generalized coordination number (GCN) of every atom, bin it
    into discrete classes, and generate a Jmol command that colours each atom
    by its GCN class.

    The GCN is continuous, so it is discretized into nClasses bins. A class
    boundary is forced at the bulk value cn_max by default (split_at_bulk=True)
    so that bulk-perfect sites (GCN = cn_max, e.g. 12 for FCC) and
    over-coordinated dense sites (GCN > cn_max, e.g. from CN = 16) always fall
    in different colour classes. The legend is a vertical set of circles, one
    per class, each circle's area proportional to the number of atoms in the
    class, annotated with 'index | segment mid-GCN | atom count'.

    Note:
        The .dat file written for Jmol contains the per-atom CLASS INDEX (an
        integer), not the real GCN value, because Jmol colours by an indexed
        palette. The real GCN is available in self.GCN; the segment value of
        each class is reported in the legend.

    Args:
        self: pyNMBcore instance (uses self.NP or self.NP_opt).
        save2 (str): file where the per-atom GCN class index is written.
        Rmax (float): cutoff distance (Å) for first neighbours. Default 2.9.
        cn_max (int or None): bulk reference coordination. None -> resolved
            from the crystal structure (FCC/HCP=12, BCC=8), else 12.
        nClasses (int): number of GCN colour classes. Default 12.
        gcn_range (tuple or None): (gcn_min, gcn_max) spanned by the classes.
            None uses the min and max GCN over all atoms.
        split_at_bulk (bool): force a class boundary at cn_max. Default True.
        is_optimized (bool or None): target structure. None -> self.is_optimized.
        noOutput (bool): if True, suppresses output. Default False.

    Returns:
        None.
    """
    import numpy as np
    import os

    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    use_opt = is_optimized and getattr(self, 'NP_opt', None) is not None

    # --- GCN of every atom (reuse self.GCN if present) --------------------
    from .prop import calculate_GCN
    GCN = getattr(self, 'GCN_opt' if use_opt else 'GCN', None)
    if GCN is None:
        GCN = calculate_GCN(self, Rmax=Rmax, cn_max=cn_max,
                            is_optimized=is_optimized, noOutput=True)
    GCN = np.asarray(GCN, dtype=float)
    if GCN.size == 0:
        print("No atoms; nothing to do.")
        return

    # resolve cn_max actually used
    cnmax_used = cn_max
    if cnmax_used is None:
        cs = getattr(self, 'crystalStructure', None) or \
             getattr(self, 'crystal_structure', None)
        cs_str = str(cs).lower() if cs is not None else ''
        cnmax_used = {'fcc': 12, 'hcp': 12, 'bcc': 8}.get(
            next((k for k in ('fcc', 'hcp', 'bcc') if k in cs_str), None), 12)

    # --- class edges ------------------------------------------------------
    if gcn_range is None:
        gcn_min, gcn_max = float(GCN.min()), float(GCN.max())
    else:
        gcn_min, gcn_max = float(gcn_range[0]), float(gcn_range[1])
    if gcn_max <= gcn_min:
        gcn_max = gcn_min + 1e-6

    if split_at_bulk and gcn_min < cnmax_used < gcn_max:
        n_below = max(1, int(round(nClasses * (cnmax_used - gcn_min) /
                                   (gcn_max - gcn_min))))
        n_above = max(1, nClasses - n_below)
        below = np.linspace(gcn_min, cnmax_used, n_below + 1)
        above = np.linspace(cnmax_used, gcn_max, n_above + 1)
        edges = np.concatenate([below[:-1], above])
        nClasses = len(edges) - 1
    else:
        edges = np.linspace(gcn_min, gcn_max, nClasses + 1)

    # --- assign a class to every atom; write the CLASS INDEX --------------
    cls = np.digitize(GCN, edges[1:-1])
    gcn_class = np.clip(cls, 0, nClasses - 1).astype(int)

    with open(save2, 'w') as f:
        for c in gcn_class:
            f.write(str(int(c)) + "\n")

    # also write the real GCN values (for Jmol labelling via a second channel)
    import os as _os
    root, ext = _os.path.splitext(save2)
    save2_values = root + "_values" + (ext or ".dat")
    with open(save2_values, 'w') as f:
        for g in GCN:
            f.write(f"{g:.4f}\n")

    if noOutput:
        return

    # --- WARNING: file holds the class index, not the real GCN -----------
    print(f"{bg.LIGHTYELLOWB}Warning: '{os.path.basename(save2)}' stores the "
          f"per-atom CLASS INDEX (0..{nClasses-1}) for Jmol colouring, not the "
          f"real GCN value. The real GCN is in self.GCN; per-class values are "
          f"in the legend.{bg.OFF}")

    # --- palette, per-class representative value and atom counts ---------
    colorsFull = [
        (255, 0, 0), (255, 255, 153), (255, 255, 0), (255, 204, 0),
        (102, 255, 255), (51, 204, 255), (102, 153, 255), (249, 128, 130),
        (153, 255, 204), (0, 204, 153), (0, 134, 101), (0, 102, 102),
        (51, 51, 255), (102, 51, 0), (0, 51, 102), (77, 77, 77),
    ]
    colorsFull = [(e[0]/255.0, e[1]/255.0, e[2]/255.0) for e in colorsFull]
    classColors = [colorsFull[i % len(colorsFull)] for i in range(nClasses)]
    counts = np.array([int(np.count_nonzero(gcn_class == i))
                       for i in range(nClasses)])

    # Representative value of each class = the actual MEAN of the GCN values of
    # its atoms, not the geometric mid-point of the interval. When a class holds
    # a single distinct GCN value (e.g. all (111) terrace centres at exactly
    # 7.50), this mean IS that exact value, so the legend matches the computed
    # GCN. When the class spans a range of values, the mean is a faithful
    # representative that lies among the real values.
    class_value = np.full(nClasses, np.nan)
    class_spread = np.zeros(nClasses)
    for i in range(nClasses):
        vals_i = GCN[gcn_class == i]
        if vals_i.size:
            class_value[i] = vals_i.mean()
            class_spread[i] = vals_i.max() - vals_i.min()
    # fall back to interval mid-point only for (empty) classes, never displayed
    seg_mid = (edges[:-1] + edges[1:]) / 2.0
    class_value = np.where(np.isnan(class_value), seg_mid, class_value)

    print(f"GCN range = [{gcn_min:.2f} - {gcn_max:.2f}], {nClasses} classes "
          f"(bulk cn_max = {cnmax_used}"
          f"{', boundary forced there' if split_at_bulk else ''})")
    for i in range(nClasses):
        if counts[i] == 0:
            continue
        spread_txt = (f", spread {class_spread[i]:.2f}"
                      if class_spread[i] > 1e-6 else " (single value)")
        print(f"  class {i:>2d}: GCN in [{edges[i]:.2f}, {edges[i+1]:.2f}), "
              f"mean {class_value[i]:.2f}{spread_txt}, {counts[i]} atoms")

    # keep only populated classes in the legend
    nz = np.where(counts > 0)[0]
    leg_colors = [classColors[i] for i in nz]
    leg_idx    = [int(i) for i in nz]
    leg_val    = class_value[nz]
    leg_counts = counts[nz]

    path, file = os.path.split(save2)
    prefix = file.split(".")[0]
    if path == "":
        path = "."
    fileColors = "./" + path + "/" + prefix + "colors.png"
    plotPaletteBubbles(leg_colors, leg_idx, leg_val, leg_counts,
                       savePngAs=fileColors, title=None,
                       val_header="GCN", show=True)

    # --- Jmol command -----------------------------------------------------
    # 'valence' carries the class index -> drives the colour (indexed palette).
    # 'partialcharge' carries the real GCN value -> drives the decimal label.
    print(f"{hl.BOLD}Jmol command:{hl.OFF}")
    colorScheme = ""
    for c in classColors:
        colorScheme = colorScheme + rgb2hex(c) + " "
    file_values = os.path.basename(save2_values)
    command = (f"{{*}}.valence = load('{file}'); "
               f"color atoms property valence 'colorGCN' RANGE 0 {nClasses-1} ; "
               f"{{*}}.partialcharge = load('{file_values}'); "
               f"label %.2[partialcharge]; color label yellow ; "
               f"font label 24 ; set labeloffset 7 0;")
    print(f"color 'colorGCN = {colorScheme}';")
    print(command)
    print(f"{bg.LIGHTYELLOWB}Note: colour is driven by the class index "
          f"('{file}', channel valence); the label shows the real GCN value "
          f"('{file_values}', channel partialcharge).{bg.OFF}")
    
def plotPaletteBubbles(colors, labels_idx, labels_val, counts,
                       savePngAs=None, title=None, val_header="GCN",
                       show=False):
    """
    Vertical colour legend: one circle per class, its area proportional to the
    number of atoms in the class, with 'index  value(2 dec)  N atoms' written
    to its right. Used by saveCN4JMol and saveGCN4JMol.

    The circle radius grows with the LOG of the atom count (with a floor radius)
    so that classes spanning several orders of magnitude all stay visible, a
    one-atom class included.

    Args:
        colors (list): (r, g, b) floats in [0, 1], one per class.
        labels_idx (list): class index (int) for each class.
        labels_val (list): representative value per class (float). For GCN this
            is the segment mid-value; for CN it is the integer CN.
        counts (list): number of atoms in each class.
        savePngAs (str): path to save the PNG (or None to skip saving).
        title (str): optional title above the legend.
        val_header (str): header of the value column ('GCN' or 'CN').
        show (bool): if True, also display the figure inline (e.g. in a
            notebook). Default False.

    Returns:
        savePngAs.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n = len(colors)
    counts = np.asarray(counts, dtype=float)
    # Circle radius grows with the LOG of the atom count, so that classes
    # spanning several orders of magnitude (a single surface atom vs thousands
    # in the bulk) all stay visible. A generous floor radius (r_min) keeps a
    # one-atom class as a clearly coloured disc, never a dot.
    r_min, r_max = 0.12, 0.42
    lc = np.log1p(counts)                       # log(1+count); 0 only if count=0
    span = lc.max() - lc.min()
    if span > 0:
        radii = r_min + (r_max - r_min) * (lc - lc.min()) / span
    else:
        radii = np.full_like(lc, 0.5 * (r_min + r_max))   # all equal
    radii = np.where(counts > 0, radii, 0.0)   # empty class -> no circle

    fig, ax = plt.subplots(figsize=(4.2, 0.6 * n + 0.8))
    for row, (col, idx, val, cnt, rad) in enumerate(
            zip(colors, labels_idx, labels_val, counts, radii)):
        y = n - row
        ax.add_patch(plt.Circle((0.5, y), rad, color=col, ec='black', lw=0.6))
        ax.text(1.15, y, f"{int(idx):>2d}   {val:6.2f}   {int(cnt):>7d} atoms",
                va='center', ha='left', family='monospace', fontsize=11)
    ax.text(1.15, n + 0.9, f"idx   {val_header:>5s}   {'count':>7s}",
            va='center', ha='left', family='monospace',
            fontsize=11, fontweight='bold')
    if title:
        ax.text(0.5, n + 1.6, title, va='center', ha='left',
                fontsize=12, fontweight='bold')
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, n + 2.2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    if savePngAs:
        plt.savefig(savePngAs, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return savePngAs
   
def saveStrain4JMol(self, prefix="strain", user_output_dir="./",
                    which='all', color=None, value_range=None,
                    symmetric=None, noOutput=False):
    """
    Write per-atom strain descriptors to real-value .dat files for Jmol and
    print the Jmol command that colours atoms by each field.

    One .dat file is written per requested quantity, each holding one real value
    per atom in atom-index order. Strain fields are continuous (and the volumetric
    strain is signed, negative under compression), so atoms are coloured by a
    continuous colourmap with an explicit range, in the same style as
    defLocalOrderColorForJMol, rather than by the discrete class palette used for
    CN and GCN.

    The Jmol command loads each .dat into the 'partialcharge' channel and colours
    by it. The real value is also shown as an atom label.

    Output files (written in user_output_dir):
        {prefix}_strainVol_values.dat    volumetric strain trace(eta)
        {prefix}_strainVM_values.dat      von Mises (deviatoric) strain
        {prefix}_strainD2min_values.dat   non-affine residual D2min

    Args:
        prefix (str, optional): Filename prefix for the output .dat files.
            Default is "strain".
        user_output_dir (str or pathlib.Path, optional): Output directory for the
            .dat files, created if absent. Default is "./".
        which (str, optional): Which quantities to write, one of 'all', 'vol',
            'vm', or 'd2min'. Default is 'all'.
        color (str or None, optional): Jmol colour scheme name. If None (default),
            a per-field default is used: a divergent 'rwb' (red = compression,
            white = zero, blue = dilation) for the signed volumetric strain, and a
            sequential 'turbo' for the non-negative von Mises and D2min fields.
            Pass an explicit name to force the same colour scheme on every field.
        value_range (list of float or None, optional): [vmin, vmax] colour-scale
            limits in the field's units. None uses each field's own min/max.
            Pass the same value_range to several structures to colour them on a
            common scale. Default is None.
        symmetric (bool or None, optional): If True, centre the colour range on
            zero (vmax = -vmin = max(|values|)), appropriate for the signed
            volumetric strain so that zero strain maps to the colourmap centre.
            If None, defaults to True for the volumetric field and False for the
            (non-negative) von Mises and D2min fields. Ignored when value_range
            is given. Default is None.
        noOutput (bool, optional): If True, suppress the printed Jmol command.
            Default is False.

    Raises:
        AttributeError: If calculate_atomic_strain() has not been run yet.
        ValueError: If which is not one of 'all', 'vol', 'vm', 'd2min'.
    """
    from pathlib import Path

    if not hasattr(self, 'strain_vol'):
        raise AttributeError(
            "No strain data found. Run calculate_atomic_strain() first."
        )
    if which not in ('all', 'vol', 'vm', 'd2min'):
        raise ValueError("which must be one of 'all', 'vol', 'vm', 'd2min'.")

    out_dir = Path(user_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

# field key -> (suffix, values, label, default symmetric, default colormap, interpretation)
    fields = {
        'vol':   ('_strainVol_values.dat',   self.strain_vol,
                  'volumetric strain trace(eta)', True,  'rwb',
                  "red = compression (local volume < bulk), "
                  "blue = dilation (local volume > bulk), white = no volume strain."),
        'vm':    ('_strainVM_values.dat',    self.strain_vm,
                  'von Mises (shear) strain',     False, 'turbo',
                  "0 = no local shear, high = strong shear "
                  "(concentrates on twin planes)."),
        'd2min': ('_strainD2min_values.dat', self.strain_d2min,
                  'non-affine residual D2min',    False, 'turbo',
                  "0 = locally affine (reliable strain), high = non-affine "
                  "(defect, surface step, or neighbour-pairing issue)."),
    }

    selected = fields.keys() if which == 'all' else (which,)

    for key in selected:
        suffix, values, label, sym_default, cmap_default, interp = fields[key]
        values = np.asarray(values, dtype=float)
        out = out_dir / f"{prefix}{suffix}"
        np.savetxt(out, values, fmt='%.6e')

        if noOutput:
            continue

        # colormap: explicit `color` arg overrides the per-field default
        cmap = cmap_default if color is None else color

        # --- colour range: explicit, symmetric, or data-driven ---
        sym = sym_default if symmetric is None else symmetric
        finite = values[np.isfinite(values)]
        if value_range is not None:
            vmin, vmax = _resolve_value_range(finite, value_range, label=key)
        elif sym and finite.size:
            vmax = float(np.max(np.abs(finite)))
            vmin = -vmax
        elif finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

        n_nan = int(np.isnan(values).sum())
        fname = out.name
        decs = 2 if key == 'd2min' else 4

        command = (f"{{*}}.partialcharge = load('{fname}'); "
                   f"color atoms property partialcharge '{cmap}' "
                   f"RANGE {vmin:.4f} {vmax:.4f}; "
                   f"label %.{decs}[partialcharge]; color label black ; "
                   f"font label 18 ; set labeloffset 7 0;")

        print(f"{bg.LIGHTBLUEB}{fg.DARKCYAN}Strain field '{key}'{bg.OFF} "
              f"{hl.ITALIC}({label}){hl.italic} written to {out}")
        print(f"  colour map '{cmap}', range {vmin:.4f} – {vmax:.4f}"
              f"{f', {n_nan} NaN atoms' if n_nan else ''}")
        print(f"  {fg.BLUE}interpretation: {interp}{fg.OFF}")
        print(f"{hl.BOLD}Jmol command:{hl.OFF}")
        print(command)
        print()
        
# def DrawJmol(mol, prefix, scriptJ="", noOutput=True):
#     """
#     Generate a Jmol visualization from an existing XYZ file.
#     """
#     from pyNanoMatBuilder import data
#     path2Jmol = data.pyNMBvar.path2Jmol  # Use the user-configurable path
    
#     # 1. Define the local working directory for the user's files
#     user_figs_dir = Path("figs")
#     user_figs_dir.mkdir(exist_ok=True)
    
#     # 2. Locate the input XYZ (assumed to be in the local figs folder)
#     fxyz = user_figs_dir / f"{mol}.xyz"
#     if not fxyz.exists():
#         if not noOutput:
#             print(f"Error: {fxyz} not found. Cannot generate image.")
#         return

#     # 3. Build the Jmol script
#     # We save the .png to the local working directory 'figs/'
#     output_png = user_figs_dir / f"{prefix}.png"
    
#     jmolscript = (
#         f"{scriptJ}; frank off; set specularPower 80; set antialiasdisplay; "
#         "set background [xf1f2f3]; set zShade ON; set zShadePower 1; "
#         f"write image pngt 1024 1024 '{output_png}';"
#     )

#     # 4. Assemble the command
#     jmolcmd = (
#         f"java -Xmx512m -jar {path2Jmol}/JmolData.jar {fxyz} "
#         f"-ij \"{jmolscript}\" >/dev/null "
#     )

#     if not noOutput:
#         print(f"Generating Jmol image: {output_png}")
#         print(jmolcmd)
    
#     os.system(jmolcmd)

def DrawJmol(mol, prefix, scriptJ="", noOutput=True,
             user_output_dir="figs", fmt=None):
    """
    Generate a Jmol visualization from an existing XYZ or CIF file.

    Backward compatible: DrawJmol(mol, prefix) still reads 'figs/{mol}.xyz'
    and writes 'figs/{prefix}.png'. The input directory and the file format
    can now be set explicitly.

    Args:
        mol (str): base name of the input file (without extension), OR a full
            path to the input file. If a path with a .xyz/.cif extension is
            given, it is used directly and fmt/user_output_dir are ignored for
            locating the input.
        prefix (str): base name of the output PNG (written as '{prefix}.png').
        scriptJ (str): Jmol script commands applied before writing the image.
        noOutput (bool): if True, suppresses printed messages. Default True.
        user_output_dir (str or Path): directory holding the input file and
            receiving the output PNG. Default "figs" (legacy behaviour).
        fmt (str or None): input format, 'xyz' or 'cif'. If None, inferred from
            the file extension when 'mol' is a path, otherwise defaults to
            'xyz'. CIF files are loaded natively by Jmol so the unit cell is
            preserved (use 'unitcell on' in scriptJ to draw it).

    Returns:
        str: full path of the PNG written, or None if the input was not found
            or Jmol is unavailable.
    """
    from pyNanoMatBuilder import data
    path2Jmol = Path(data.pyNMBvar.path2Jmol)

    # --- Resolve the input file: explicit path, or '{user_output_dir}/{mol}.{fmt}' ---
    mol_path = Path(mol)
    if mol_path.suffix.lower() in (".xyz", ".cif"):
        # 'mol' is already a path to the structure file
        finput = mol_path
        if fmt is None:
            fmt = mol_path.suffix.lower().lstrip(".")
        out_dir = Path(user_output_dir)
    else:
        # legacy form: base name inside user_output_dir
        if fmt is None:
            fmt = "xyz"
        out_dir = Path(user_output_dir)
        finput = out_dir / f"{mol}.{fmt}"

    if fmt not in ("xyz", "cif"):
        raise ValueError(f"fmt must be 'xyz' or 'cif', got '{fmt}'.")

    out_dir.mkdir(parents=True, exist_ok=True)

    if not finput.exists():
        if not noOutput:
            print(f"Error: {finput} not found. Cannot generate image.")
        return None

    jar_file = path2Jmol / "JmolData.jar"
    if not jar_file.exists():
        if not noOutput:
            print(f"Jmol not found at {jar_file}; PNG skipped.")
        return None

    # --- Build the Jmol script ---
    # For CIF, load explicitly so cell parameters are read; for XYZ, the file
    # passed on the command line is loaded automatically (legacy behaviour).
    output_png = out_dir / f"{prefix}.png"
    load_cmd = f"load '{finput}'; " if fmt == "cif" else ""
    jmolscript = (
        f"{load_cmd}{scriptJ}; frank off; set specularPower 80; "
        "set antialiasdisplay; set background [xf1f2f3]; "
        f"set zShade ON; set zShadePower 1; "
        f"write image pngt 1024 1024 '{output_png}';"
    )

    # --- Assemble the command ---
    # XYZ: pass the file as an argument (kept for backward compatibility).
    # CIF: the load is inside the script, so no file argument is passed.
    file_arg = "" if fmt == "cif" else f"{finput} "
    jmolcmd = (
        f"java -Xmx512m -jar {jar_file} {file_arg}"
        f"-ij \"{jmolscript}\" >/dev/null "
    )

    if not noOutput:
        print(f"Generating Jmol image: {output_png}")
        print(jmolcmd)

    os.system(jmolcmd)
    return str(output_png)
    
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

def defLocalOrderColorForJMol(self, descriptor='cnp', l=6, color='turbo',
                              value_range=None, is_optimized=None, noOutput=True):
    """
    Generate a Jmol command that writes a per-atom local-order descriptor
    (CNP or Steinhardt q_l) into an atom property and colours the atoms by it.

    The descriptor must already be stored on the object (self.cnp / self.q{l},
    with the _opt suffix for the optimized structure). Run
    common_neighbour_parameter() or steinhardt_q() first.

    Args:
        descriptor (str): 'cnp' (default) or 'q' for Steinhardt q_l.
        l (int): Harmonic degree for descriptor='q'. Default 6.
        color (str): Jmol colour scheme name. Default 'turbo'.
            (color schemes known by Jmol: batlow,cividis,kry,thermal,viridis,inferno,magma,plasma,turbo)
        value_range (list of float or None): [vmin, vmax] colour-scale limits
            in the descriptor's units (Å² for CNP). None (default) uses this
            structure's own min/max. Pass the SAME value_range to several
            structures to colour them on a common scale; a bound outside the
            data is clamped to the data with a warning. Applies to both the
            matplotlib figure and the generated Jmol colouring command.
        is_optimized (bool or None): Target structure. None -> self.is_optimized.
        noOutput (bool): If True, suppresses output. Default is True.

    Returns:
        str: Jmol command string (also stored in self.jMol_{descriptor}
            (e.g. self.jMol_cnp, self.jMol_q6, with the _opt suffix for the optimized structure) ).
    """
    if is_optimized is None:
        is_optimized = getattr(self, 'is_optimized', False)
    use_opt = is_optimized and getattr(self, 'NP_opt', None) is not None
    suffix = "_opt" if use_opt else ""

    if descriptor == 'cnp':
        attr = f"cnp{suffix}"
        prop_name = "cnp"
        label = "CNP (Å²)"
    elif descriptor == 'q':
        attr = f"q{l}{suffix}"
        prop_name = f"q{l}"
        label = f"q{l}"
    else:
        raise ValueError(f"descriptor must be 'cnp' or 'q', got '{descriptor}'.")

    values = getattr(self, attr, None)
    if values is None:
        print(f"{bg.LIGHTYELLOWB}Warning: self.{attr} not found. "
              f"Run the descriptor first (common_neighbour_parameter / "
              f"steinhardt_q).{bg.OFF}")
        return ""

    values = np.asarray(values)
    vmin, vmax = _resolve_value_range(values, value_range, label=prop_name)

    data_str = " ".join(f"{v:.4f}" for v in values)
    cmd = (f"data \"property_{prop_name}\"\n"
           f"{data_str}\n"
           f"end \"property_{prop_name}\"; "
           f"color atoms property_{prop_name} \"{color}\" "
           f"range {vmin:.4f} {vmax:.4f};")

    setattr(self, f"jMol_{prop_name}{suffix}", cmd)

    if not noOutput:
        print(f"  [Jmol command to colour atoms by {label}]:")
        print(f"  (range {vmin:.4f} – {vmax:.4f})")
        print(f"  {cmd}")

    return cmd

def defCarvePreviewForJMol(self,
                           B_copies,
                           color: str = 'xff3030',
                           translucency: int = 80,
                           noOutput: bool = True):
    """
    Generate a Jmol command to visualize the carving patterns placed by
    systematic_carve_by(preview=True), as translucent polygons outlining the
    convex hull of each placed pattern copy.

    For each placed copy of B, the faces of its convex hull are drawn as
    translucent polygons with their edges, so the exact volume that would be
    removed is visible as a clean solid — legible regardless of B's atomic
    spacing (i.e. even when scale > 1 spreads the marker atoms apart). This is
    the mode='hull' companion to the 'No' marker atoms stored in
    self.NP_preview; in mode='atoms' the marker atoms already show B's true
    (possibly concave) shape and no polygons are produced.

    Args:
        B_copies (list of ndarray): list of (N,3) arrays, the placed atomic
            positions of each pattern copy (one per carved face).
        color (str): hex colour of the polygons/edges (default 'xff3030', red).
        translucency (int): polygon translucency in percent (0=opaque,
            100=fully translucent). Default 80.
        noOutput (bool): If True, suppresses output. Default is True.

    Returns:
        str: Jmol command string. The caller (systematic_carve_by or
            systematic_stellate_by) stores it in its own attribute
            (self.jMolCarvePreview or self.jMolStellationPreview).
    """
    from scipy.spatial import ConvexHull

    cmd = ""
    fidx = 0
    eidx = 0
    for ic, Bp in enumerate(B_copies):
        try:
            hb = ConvexHull(Bp)
        except Exception:
            continue
        # --- hull faces as translucent polygons ---
        for simp in hb.simplices:
            cmd += f"draw carvef{fidx} polygon ["
            for at in simp:
                p = Bp[at]
                cmd += f"{{{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}}},"
            cmd += "]; "
            fidx += 1
        # --- hull edges as thin lines ---
        edges = set()
        for s in hb.simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                edges.add((min(a, b), max(a, b)))
        for a, b in edges:
            p0, p1 = Bp[a], Bp[b]
            cmd += f"draw carvee{eidx} ["
            cmd += f"{{{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}}},"
            cmd += f"{{{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}}},"
            cmd += "] width 0.15; "
            eidx += 1

    cmd += f"color $carvef* translucent {translucency} [{color}]; "
    cmd += f"color $carvee* [{color}]; "

    if not noOutput:
        print(f"Jmol command for {len(B_copies)} carving patterns "
              f"({fidx} polygons):")
        print(cmd)

    return cmd

############################################ OVITO ###################
def _normalize_rgb(color):
    """Normalize a color specification to an (r, g, b) float triple in [0, 1].

    Accepts any of:
      - a hex string '#RRGGBB' (or '0xRRGGBB', 'xRRGGBB', or bare 'RRGGBB'),
      - a triple of ints in [0, 255] (e.g. (255, 165, 0)),
      - a triple of floats already in [0, 1] (e.g. (1.0, 0.65, 0.0)).

    Args:
        color: the color specification (str or 3-element sequence).

    Returns:
        tuple of float: (r, g, b), each in [0, 1].

    Raises:
        ValueError: if the specification cannot be parsed as a color.
    """
    # Hex string form.
    if isinstance(color, str):
        s = color.strip().lstrip('#').lstrip('0x').lstrip('x')
        if len(s) != 6:
            raise ValueError(
                f"Invalid hex color '{color}'; expected 6 hex digits "
                f"(e.g. '#FFA500').")
        try:
            r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            raise ValueError(f"Invalid hex color '{color}'.")
        return (r / 255.0, g / 255.0, b / 255.0)

    # Sequence of three numbers.
    seq = tuple(color)
    if len(seq) != 3:
        raise ValueError(
            f"Color must be a hex string or a 3-element RGB sequence, "
            f"got {color!r}.")
    # Ints in [0, 255] are normalized; floats already in [0, 1] are kept.
    if all(isinstance(v, (int,)) and not isinstance(v, bool) for v in seq) \
            and any(v > 1 for v in seq):
        return tuple(v / 255.0 for v in seq)
    return tuple(float(v) for v in seq)
    
def export_png_with_ovito(self,
                          prefix="ovito",
                          user_output_dir="./",
                          radius=None,
                          colors=None,
                          azimuth_deg=30.0,
                          elevation_deg=65.0,
                          camera_pos=None,
                          camera_dir=None,
                          camera_up=None,
                          fov=None,
                          zoom_factor=None,
                          projection="perspective",
                          size=(2400, 2400),
                          background=(1, 1, 1),
                          transparent=False,
                          ambient_occlusion=True,
                          use_opt=False,
                          render_mode="atoms",
                          surface_radius=4.0,
                          surface_smoothing=12,
                          highlight_mesh_edges=False,
                          noOutput=False):
    """Render self.NP to a PNG image with OVITO Basic (free, no watermark).
    The function can render either the atoms as spheres (the default) or a
    solid enclosing surface, selected with the render_mode argument. In surface
    mode an alpha-shape surface is built around the atoms, the atoms are hidden,
    and the mesh is drawn as a uniform-colored solid, giving a faceted
    polyhedron-like view of the particle shape; a non-periodic bounding-box
    cell is assigned to the data first, as the surface construction requires a
    non-degenerate simulation cell.
    
    Uses OVITO's OpenGL renderer, the only non-watermarked backend in the free
    OVITO Basic edition. The free Ambient Occlusion modifier is added to the
    pipeline to give per-atom occlusion shading, which OpenGL can display and
    which makes facets readable. Facets look flat when a particle is viewed
    straight down, so the camera is placed off-axis.
    The output file is written to '<user_output_dir>/<prefix>.png'.
    The camera can use either a perspective or a parallel (orthographic)
    projection, selected with the projection argument. To reproduce a view set
    up interactively in the OVITO GUI, match the GUI projection here: read the
    projection type, the camera position, the camera direction and the field of
    view from the 'Adjust view...' dialog and pass them through projection,
    camera_pos, camera_dir and fov.
    Two ways to set the viewpoint:
      - azimuth_deg / elevation_deg: the camera looks at the origin from a
        point on a sphere (convenient for quick framing).
      - camera_pos / camera_dir: explicit camera placement, e.g. the values
        read from the interactive OVITO GUI. If camera_dir is given it takes
        priority over the azimuth/elevation direction; if camera_pos is also
        given, zoom_all() is skipped so the GUI framing is reproduced exactly.
    The OVITO Python module renders headlessly when imported from an external
    interpreter, so no window is opened; the image is written straight to disk.
    Args:
        prefix (str): base name of the output file (without extension); the
            image is saved as '<prefix>.png'.
        user_output_dir (str): directory where the image is written. Created
            if it does not exist.
        radius (float, optional): sphere radius in Angstrom. If None, uses
            half of self.Rnn (jointed spheres look).
        colors (dict, str, or sequence, optional): atom colors overriding
            OVITO's default CPK per-element colors. Each color may be given as:
              - a hex string '#RRGGBB' (also accepts '0xRRGGBB', 'xRRGGBB',
                or bare 'RRGGBB'),
              - a triple of ints in [0, 255], e.g. (255, 165, 0),
              - a triple of floats in [0, 1], e.g. (1.0, 0.65, 0.0).
            Two forms are accepted:
              - a dict mapping chemical symbol to a color, e.g.
                {'Au': '#D9A521', 'Co': (51, 102, 230)}; only the listed
                elements are recoloured, others keep their default,
              - a single color (hex string or RGB triple) applied uniformly
                to every element.
            A triple whose values are all <= 1 is read as floats in [0, 1];
            if any value exceeds 1, the triple is read as ints in [0, 255].
            If None, the default element colors are kept.
        azimuth_deg (float): camera azimuth around the vertical z axis, in
            degrees (0 = +x). Ignored if camera_dir is provided.
        elevation_deg (float): camera elevation above the xy plane, in degrees.
            90 looks straight down (facets flatten); ~60-70 keeps the top
            facets visible while giving them depth. Ignored if camera_dir is
            provided.
        camera_pos (array-like, optional): explicit camera position (x, y, z).
            When given together with camera_dir, the automatic zoom_all()
            framing is skipped to reproduce a GUI viewpoint exactly.
        camera_dir (array-like, optional): explicit camera viewing direction
            (look vector). Takes priority over azimuth/elevation.
        fov (float, optional): camera field of view, only used when camera_pos
            and camera_dir are both given (explicit GUI framing). Its meaning
            depends on projection: for 'perspective' it is the vertical opening
            angle in radians; for 'ortho' it is the vertical size of the
            visible region in Angstrom. Copy the value shown in the OVITO
            'Adjust view...' dialog. If None, OVITO's default field of view for
            the chosen projection is used, so the apparent zoom may not match
            the GUI.
        zoom_factor (float, optional): multiplies the field of view after
            framing, in either viewpoint mode. Values > 1 zoom out (add margin
            around the particle), values < 1 zoom in. Useful when the automatic
            zoom_all() framing is too tight and the particle spills over the
            image edges. If None, the framing is left unchanged.
        projection (str): camera projection type. 'perspective' (default) for a
            perspective projection, or 'ortho' for a parallel/orthographic
            projection (also accepts 'orthographic', 'orthogonal', 'parallel').
            To reproduce a GUI view, set this to the same projection as the
            viewport used in the GUI.
        size (tuple of int): output image size in pixels (width, height).
        background (tuple of float): RGB background color, each in [0, 1].
            Ignored where the image is transparent if transparent=True.
        transparent (bool): if True, render with a transparent background
            (alpha channel): areas not covered by atoms become transparent so
            the image can be overlaid on another backdrop. PNG output (used
            here) preserves the alpha channel.
        ambient_occlusion (bool): if True, append the Ambient Occlusion
            modifier for facet-revealing per-atom shading.
        use_opt (bool): if True, render the optimized structure (self.NP_opt)
            instead of self.NP.
        render_mode (str): what to draw. 'atoms' (default) renders the atoms as
            spheres. 'surface' builds a solid alpha-shape surface enclosing the
            atoms, hides the atoms, and renders that mesh instead, giving a
            faceted polyhedron-like view of the particle shape. In surface mode
            the per-sphere radius is ignored and colors sets the single surface
            color.
        surface_radius (float): in surface render mode, the alpha-shape probe
            sphere radius in Angstrom. Smaller values hug the atoms and may
            open holes in low-density regions; larger values smooth the surface
            and, in the limit, reproduce the convex hull. Ignored in atoms
            render mode.
        surface_smoothing (int): in surface render mode, the number of
            smoothing iterations applied to the surface mesh. Higher values
            round off the mesh. Ignored in atoms render mode.
        highlight_mesh_edges (bool): in surface render mode, if True draw the edges
            of the surface mesh triangles (the OVITO 'Highlight mesh edges'
            option), which exposes the triangulation. Default False gives
            smooth faces where only the facet edges show up through ambient
            occlusion shading. Ignored in atoms render mode.
        noOutput (bool): if True, suppress printed messages.
    Returns:
        str: the full path of the PNG file that was written.
    Raises:
        ImportError: if the ovito Python module is not installed.
        AttributeError: if use_opt is True but self.NP_opt is not available.
        ValueError: if projection is not a recognized projection name.
    """
    try:
        from ovito.io.ase import ase_to_ovito
        from ovito.pipeline import Pipeline, StaticSource
        from ovito.vis import Viewport, OpenGLRenderer, ParticlesVis
        from ovito.modifiers import AmbientOcclusionModifier, ConstructSurfaceModifier
    except ImportError as exc:
        raise ImportError(
            "OVITO is required for export_png_with_ovito. Install it with "
            "'conda install -c https://conda.ovito.org ovito' (preferred) or "
            "'pip install ovito'.") from exc
    if not noOutput:
        centertxt(
            "OVITO rendering",
            bgc='#007a7a', size='14', weight='bold')
        chrono = timer()
        chrono.chrono_start()
        
    # --- Build the output path '<user_output_dir>/<prefix>.png' ---
    outdir = pathlib.Path(user_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{prefix}.png"

    # --- Select the structure to render ---
    if use_opt:
        if not hasattr(self, 'NP_opt') or self.NP_opt is None:
            raise AttributeError(
                "use_opt=True but self.NP_opt is not available; optimize the "
                "structure first.")
        atoms = self.NP_opt
    else:
        atoms = self.NP

    if radius is None:
        radius = self.Rnn / 2.0

    # --- Build an OVITO pipeline directly from the ASE Atoms object ---
    # ase_to_ovito avoids a round-trip through a file on disk.
    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))

    mode = str(render_mode).lower()
    if mode not in ("atoms", "surface"):
        raise ValueError(
            f"render_mode must be 'atoms' or 'surface', got {render_mode!r}")

    if mode == "atoms":
        vis = pipeline.source.data.particles.vis
        vis.shape = ParticlesVis.Shape.Sphere
        vis.radius = radius

        # --- Optional atom recolouring (overrides default CPK colors) ---
        if colors is not None:
            if isinstance(colors, dict):
                color_map = {sym: _normalize_rgb(c)
                             for sym, c in colors.items()}
            else:
                rgb = _normalize_rgb(colors)
                color_map = {ptype.name: rgb
                             for ptype in data.particles.particle_types.types}
            for ptype in data.particles.particle_types.types:
                if ptype.name in color_map:
                    ptype.color = color_map[ptype.name]

        if ambient_occlusion:
            pipeline.modifiers.append(AmbientOcclusionModifier())

    else:  # mode == "surface"
                # ConstructSurfaceModifier requires a non-degenerate simulation cell.
        # pyNMB particles are isolated clusters with no periodic box, so give
        # the data a non-periodic bounding-box cell sized to the atoms (plus a
        # margin) to avoid a zero-volume cell error.
        pos = data.particles.positions
        pmin = np.min(pos, axis=0)
        pmax = np.max(pos, axis=0)
        margin = 2.0 * surface_radius
        cell_origin = pmin - margin
        cell_span = (pmax - pmin) + 2.0 * margin
        cell_matrix = np.zeros((3, 4))
        cell_matrix[0, 0] = cell_span[0]
        cell_matrix[1, 1] = cell_span[1]
        cell_matrix[2, 2] = cell_span[2]
        cell_matrix[:, 3] = cell_origin  # cell origin in the 4th column
        data.create_cell(cell_matrix, (False, False, False))

        # Build a solid alpha-shape surface enclosing the atoms and render that
        # instead of the individual spheres, giving a faceted polyhedron look.
        surf_mod = ConstructSurfaceModifier(
            method=ConstructSurfaceModifier.Method.AlphaShape,
            radius=surface_radius,
            smoothing_level=surface_smoothing,
            identify_regions=False)
        # Configure the surface appearance on the modifier's own vis element so
        # the settings persist through to rendering (a throwaway compute() does
        # not).
        surf_mod.vis.show_cap = False
        surf_mod.vis.highlight_edges = highlight_mesh_edges
        if colors is not None:
            surf_mod.vis.surface_color = _normalize_rgb(colors)
        pipeline.modifiers.append(surf_mod)

        # Hide the atoms so only the surface mesh is visible.
        pipeline.source.data.particles.vis.enabled = False

        if ambient_occlusion:
            pipeline.modifiers.append(AmbientOcclusionModifier())

    pipeline.add_to_scene()

    # --- Camera placement ---
    proj = str(projection).lower()
    if proj in ("ortho", "orthographic", "orthogonal", "parallel"):
        vp = Viewport(type=Viewport.Type.Ortho)
    elif proj in ("perspective", "persp"):
        vp = Viewport(type=Viewport.Type.Perspective)
    else:
        raise ValueError(
            f"projection must be 'ortho' or 'perspective', got {projection!r}")

    if camera_dir is not None:
        # Explicit direction from the GUI takes priority.
        vp.camera_dir = tuple(camera_dir)
    else:
        # Direction from azimuth/elevation: camera sits on a sphere and looks
        # back at the origin.
        az = np.radians(azimuth_deg)
        el = np.radians(elevation_deg)
        cam = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el)])
        vp.camera_dir = tuple(-cam)

    if camera_up is not None:
        # Explicit up vector fixes the roll (rotation about the viewing axis).
        # When None, OVITO keeps its default constraint (z stays vertical).
        vp.camera_up = tuple(camera_up)

    if camera_pos is not None and camera_dir is not None:
        # Full explicit framing: reproduce the GUI viewpoint, no auto-zoom.
        vp.camera_pos = tuple(camera_pos)
        if fov is not None:
            vp.fov = float(fov)
    else:
        # Auto-position the camera along camera_dir to fit all atoms.
        vp.zoom_all()

    if zoom_factor is not None:
        # Widen (>1) or tighten (<1) the framing relative to the current fov.
        # Useful when zoom_all() frames too tightly and the particle spills
        # over the image edges: zoom_factor=1.1 adds ~10% margin.
        vp.fov = vp.fov * float(zoom_factor)

    vp.render_image(filename=str(out_path), size=tuple(size),
                    renderer=OpenGLRenderer(),
                    background=tuple(background),
                    alpha=transparent)

    # Clean up the scene so repeated calls do not stack pipelines.
    pipeline.remove_from_scene()

    if not noOutput:
        print(f"OVITO render written to {out_path} "
              f"(radius={radius:.3f} A, transparent={transparent})")
        chrono.chrono_stop(hdelay=False)
        chrono.chrono_show()
    return str(out_path)

def export_surface_mesh(self, prefix, user_output_dir="mesh",
                        method='alpha', probe_radius=None, surface_smoothing=8,
                        cn_threshold=12, Rmax=None, surface_only=None,
                        atomic_radius=None,
                        color='x999999', translucency=0.15,
                        noOutput=False):
    """
    Build a surface mesh of the NP and export it as a Jmol pmesh file.

    For very large NPs, the mesh is built only from under-coordinated atoms
    (CN < cn_threshold), i.e. the surface shell — light on memory and, for
    method='alpha', faithful to concave features (nanostars, octopods).

    Args:
        prefix (str): basename of the output files; '{prefix}.pmesh' and
            '{prefix}.script' are written in user_output_dir.
        user_output_dir (str or Path): output directory, created if needed.
            Default 'mesh'.
        method (str): 'alpha' (default, follows concavities) or 'hull'
            (convex only).
        probe_radius (float): alpha-shape probe radius in Å. If None,
            defaults to 1.6 * Rnn. Ignored when method='hull'.
        surface_smoothing (int): number of Taubin smoothing iterations
            applied to the mesh, analogous to OVITO's smoothing_level.
            0 disables smoothing. Default 8.
        cn_threshold (int): keep only atoms with coordination number below
            this (default 12 = fcc/hcp bulk; use 8 for bcc).
        Rmax (float): neighbour cutoff in Å for the CN calculation. If None,
            defaults to 1.2 * Rnn.
        surface_only (bool): if True, use self.surfaceAtoms instead of the CN
            filter. Default None (use CN filter).
        atomic_radius (float): if given (Å), also compute the area and
            volume of the surface offset outwards by this radius (outer
            atomic envelope estimate, Steiner parallel-body formulas).
            Stored as meshArea_radius_nm2 / meshVolume_radius_nm3.
        color (str): Jmol colour. Default 'x999999'.
        translucency (float): 0 = opaque, 1 = invisible. Default 0.15.
        noOutput (bool): suppress messages. Default False.

    """
    import numpy as np
    from pathlib import Path
    from .geometry import build_surface_mesh
    from .core import kDTreeCN, centertxt, timer

    Rnn = getattr(self, 'Rnn', 2.9)
    if Rmax is None:
        Rmax = 1.2 * Rnn
    if probe_radius is None:
        probe_radius = 1.6 * Rnn

    if not noOutput:
        centertxt("Exporting surface mesh (pmesh for Jmol)",
                  bgc='#007a7a', size='14', weight='bold')
        chrono = timer(); chrono.chrono_start()

    # Select the shell
    if surface_only:
        if not hasattr(self, 'surfaceAtoms') or self.surfaceAtoms is None:
            raise AttributeError("surface_only=True requires self.surfaceAtoms "
                                 "(run coreSurface / propPostMake first).")
        shell = self.NP[self.surfaceAtoms]
        n_source = "surface atoms (surfaceAtoms)"
    else:
        from scipy.spatial import cKDTree
        _, CN = kDTreeCN(self.NP, Rmax=Rmax, returnD=False, noOutput=True)
        CN = np.asarray(CN)
        mask = CN < cn_threshold
        # Dilate the surface selection by one neighbour layer: a strictly
        # single-layer shell is locally coplanar on flat facets, which makes
        # every local tetrahedron degenerate (zero determinant) and opens
        # holes in the alpha-shape. Adding the first sub-surface layer gives
        # the tessellation the thickness it needs.
        pos = self.NP.get_positions()
        tree = cKDTree(pos)
        neigh = tree.query_ball_point(pos[mask], r=Rmax)
        extra = np.unique(np.concatenate([np.asarray(n, dtype=int) for n in neigh]))
        mask = mask.copy()
        mask[extra] = True
        shell = self.NP[mask]
        n_source = (f"under-coordinated atoms (CN < {cn_threshold}) "
                    f"+ one neighbour layer")

    if len(shell) < 4:
        raise ValueError(f"Only {len(shell)} atoms in the shell — not enough "
                         f"to build a 3D mesh. Lower cn_threshold or check Rmax.")

    # Build the mesh (pure geometry)
    vertices, faces, comp_labels = build_surface_mesh(
        shell, method=method, probe_radius=probe_radius,
        return_components=True)

    # Area and volume of the outer skin, on the raw (unsmoothed) mesh
    from .prop import mesh_area_volume
    result = mesh_area_volume(vertices, faces, labels=comp_labels,
                              atomic_radius=atomic_radius)
    area, volume = result[0], result[1]
    self.meshArea_nm2 = area * 1e-2
    self.meshVolume_nm3 = volume * 1e-3
    if atomic_radius is not None:
        area_corr, volume_corr = result[2], result[3]
        self.meshArea_radius_nm2 = area_corr * 1e-2
        self.meshVolume_radius_nm3 = volume_corr * 1e-3
    
    if surface_smoothing and surface_smoothing > 0:
        from .geometry import smooth_mesh_taubin
        vertices = smooth_mesh_taubin(vertices, faces,
                                      iterations=surface_smoothing)

    # Optional Taubin smoothing, equivalent to OVITO's smoothing_level
    if surface_smoothing and surface_smoothing > 0:
        from .geometry import smooth_mesh_taubin
        vertices = smooth_mesh_taubin(vertices, faces,
                                      iterations=surface_smoothing)

    # Normalize colour to Jmol's [xRRGGBB] form (accept '#RRGGBB' or 'xRRGGBB')
    c = color.strip()
    if c.startswith('#'):
        c = c[1:]
    elif c.startswith('x'):
        c = c[1:]
    color_jmol = f"x{c}"

    # Write the Jmol-native pmesh file
    output_dir = Path(user_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pmesh = output_dir / f"{prefix}.pmesh"
    with open(output_pmesh, 'w') as f:
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        f.write(f"{len(faces)}\n")
        for t in faces:
            f.write(f"4 {t[0]} {t[1]} {t[2]} {t[0]}\n")

    jmol_cmd = f'pmesh npmesh "{output_pmesh.name}"; color $npmesh [{color_jmol}] translucent {translucency};'

    # Write a ready-to-run Jmol script next to the pmesh
    script_path = output_pmesh.with_suffix('.script')
    with open(script_path, 'w') as f:
        f.write(jmol_cmd + "\n")
        
    if not noOutput:
        print(f"  - Source: {n_source} ({len(shell)} atoms)")
        print(f"  - Method: {method}"
              + (f", probe_radius = {probe_radius:.2f} Å" if method == 'alpha' else ""))
        print(f"  - Mesh: {len(vertices)} vertices, {len(faces)} triangles")
        print(f"  - Outer surface (through atomic centres): "
              f"area = {self.meshArea_nm2:.2f} nm², volume = {self.meshVolume_nm3:.2f} nm³")
        if atomic_radius is not None:
            print(f"  - Corrected for atomic radius R = {atomic_radius:.3f} Å: "
                  f"area = {self.meshArea_radius_nm2:.2f} nm², "
                  f"volume = {self.meshVolume_radius_nm3:.2f} nm³")
        print(f"  - Mesh and Script written to  {output_pmesh}")
        print(f"\n  To visualize in Jmol:")
        print(f"    - optionally load the atoms first: load \"{output_pmesh.stem}.xyz\"")
        print(f"    - then run in the Jmol console: script \"{output_pmesh.stem}.script\"")
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

