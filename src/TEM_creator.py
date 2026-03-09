import sys
import tkinter
import os
import gzip
from pathlib import Path
import re
import importlib

import numpy as np
import pandas as pd
import visualID as vID
from visualID import  fg, hl, bg

from pyNanoMatBuilder import crystalNPs as cyNP
from pyNanoMatBuilder import platonicNPs as pNP
from pyNanoMatBuilder import archimedeanNPs as aNP
from pyNanoMatBuilder import catalanNPs as cNP
from pyNanoMatBuilder import johnsonNPs as jNP
from pyNanoMatBuilder import otherNPs as oNP
from pyNanoMatBuilder import utils as pyNMBu
from pyNanoMatBuilder import data
import abtem

from ase import io
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase.spacegroup import get_spacegroup
from ase.io import read
from ase.io import write
from ase.visualize import view

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math


class CreateHRTEMStructure:
    """
    A class for generating HRTEM images from CIF compounds. First the class creates XYZ files, viewable on JMOL,
    containing the coordinates of a nanoparticule laying on a transmission electronic miscroscope (TEM) grid of carbon.
    This class allows to create images for different NPs (various compounds, shapes, sizes) laying  on their surfaces or edges
    with different angles rotations, meaning it is possible to explore different orientation of the NP on the grid.
    Then the class uses AbTEM to generate their HRTEM images. Many parameters are explored, see the HRTEM_parameter.ipynb notebook
    for explanations.

    Additional Notes:
    - The supported nanoparticle shapes include: Wulff constructions: cube, octahedron, cuboctahedron, dodecahedron, 
      spheroids, and their truncated versions
    - The NPs are laying on  flat areas of the carbon substrate.
    - The name of the output XYZ files is : {element}_{structure}_{form}_{counter}.xyz
    - The name of the output PNG files is : {element}_{structure}_{form}_{xyz_counter}_{microscope_counter}.png
    - Orientation (surface/edge), angles and size metadata are stored in CSV files.
    - The NPs are created using the crystalNPs class.
    - Files are created for each NP surface plane and each carbon grid flat surface.
    - A distance tolerance between the NP and the grid can be choosen by the user.
    
    """

    def __init__(self, path, xyz_gz_file, cif_data, wulff_shapes, 
                 nRot, sizes, min_size : float=0, max_size: float=50, tolerance: int=3, noOutput:bool = True):
      

        """
        Initialize the class with CIF data, Wulff shapes information and size, and the tolerance distance between the NP and the carbon grid.
            Args:
        path (str) : Path that will contain the output XYZ files.
        xyz_gz_file (str): Path to the gzipped (.gz) XYZ file containing atomic coordinates of the carbon substrate.
        cif_data (dataframe) : the CIF of the compounds.
        wulff_shapes (dataframe): the Wulff shapes and their informations.
        nRot (int): Number of rotation of the NP laying on its surface along z (angle = 360/nRot)
        sizes (array-like): Array of the sizes of the nanoparticles.
        min_size (float, optional): Minimal size for the NPs, equals to the diameter of the circumscribed sphere, equals 0 nm by default.
        max_size (float, optional): Maximal size for the NPs, equals to the diameter of the circumscribed sphere, equals 50 nm by default.
        tolerance (float, optional):  Tolerance distance between the NP and the carbon grid, be careful, if too small, chemical bonds appear in the interface.
        noOutput (bool): if bool=False: details of the files
            Methods:
        self.create_NP_TEMimages(tolerance, noOutput,min_size, max_size, nRot,path,xyz_gz_file)
        """
        self.path = path
        self.xyz_gz_file = Path(xyz_gz_file)
        self.cif_data = cif_data  # DataFrame containing CIF data
        self.wulff_shapes = wulff_shapes  # DataFrame of Wulff  forms
        self.loaded_cifs = {}  # stock loaded cif files
        self.nRot = nRot
        self.sizes= [[k] for k in sizes] #nested list of the sizes 
        self.tolerance= tolerance 
        self._xyz_counter = 0
        self._xyz_metadata = []
        
        self.create_NP_TEMimages(tolerance, noOutput,min_size, max_size, nRot,path,xyz_gz_file)


    def find_surface_atoms(self,xyz_gz_file, grid_size=2, z_tolerance=6):
        """
        Identify surface atoms from a compressed XYZ file using a grid-based method.
            Args:
        xyz_gz_file (str): Path  to the gzipped (.gz) XYZ file containing atomic coordinates of the carbon substrate.
        grid_size : float Size of the xy-grid cells in angstroms, used to discretize the xy-plane. Smaller grid size results in finer surface resolution, larger grid size is coarser. 
        z_tolerance : float Maximum allowed vertical distance from global maximum z to consider an atom as a surface atom.
            Note:
        If some xy-grid cells do not contain any atoms near the surface (for instance due to vertical gaps or columns), 
        the algorithm might select atoms located deep within the material. The `z_tolerance` parameter ensures that only 
        atoms close enough to the global maximum z-value are considered, thus preventing deep internal atoms from being mistakenly identified as surface atoms.
            Returns:
        Two files in a subdirectory named 'output_xyz':
        - `<original_filename>_surface.xyz`: 
            XYZ file copy where surface carbon atoms are replaced by oxygen atoms ('O') for visualization purposes.
        - `<original_filename>_surface_atoms.txt`:
            Text file listing indices (1-based) of atoms identified as surface atoms.
            Procedure:
        1. Reads atomic coordinates from the gzipped XYZ file.
        2. Defines a regular xy-grid over the atomic coordinates.
        3. Identifies the topmost atom (maximum z-coordinate) in each grid cell, only if close enough to the global maximum z.
        4. Marks these atoms as surface atoms and replaces their type with 'O'.
        5. Saves the modified XYZ file and a text file listing surface atom indices.
            Example:
        find_surface_atoms('path/to/file.xyz.gz', 'path/to/output_directory', grid_size=2.0, z_tolerance=5.0)
        """

        with gzip.open(xyz_gz_file, 'rt', encoding='utf-8' ) as f:
            lines = f.readlines()
    
        num_atoms = int(lines[0].strip())
        header = lines[:2]
        atom_lines = lines[2:2 + num_atoms]
        coords = []
        for line in atom_lines:
            parts = line.split()
            atom_type = parts[0]
            x, y, z = map(float, parts[1:4])
            coords.append((atom_type, x, y, z))
    
        coords_array = np.array(coords, dtype=object)
        xy_values = coords_array[:, 1:3].astype(float)
        z_values = coords_array[:, 3].astype(float)
        z_global_max = z_values.max()
        x_min, y_min = xy_values.min(axis=0)
        x_max, y_max = xy_values.max(axis=0)
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)
    
        surface_indices = []
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                in_cell = np.where((xy_values[:, 0] >= x_bins[i]) & (xy_values[:, 0] < x_bins[i+1]) &
                                   (xy_values[:, 1] >= y_bins[j]) & (xy_values[:, 1] < y_bins[j+1]))[0]
                if len(in_cell) > 0:
                    z_local_max_idx = in_cell[np.argmax(z_values[in_cell])]
                    if z_values[z_local_max_idx] >= z_global_max - z_tolerance:
                        surface_indices.append(z_local_max_idx)
    
        surface_indices = np.unique(surface_indices)
        modified_coords = coords.copy()
        for idx in surface_indices:
            atom_type, x, y, z = modified_coords[idx]
            modified_coords[idx] = ('O', x, y, z)
    
        return surface_indices 
    
    
    def flat_areas_atoms(self,xyz_gz_file, surface_indices, diameter_nm=2, min_cluster_span=20, max_z_variation=5, anisotropy_threshold= 1.1, overlap_tolerance=0.2):
        """
        Identify candidate grafting sites for a nanoparticle of given diameter using a sliding grid over the xy-plane.
        Each candidate cluster must:
        - Contain at least two atoms,
        - Cover a minimum lateral spatial extent (min_cluster_span),
        - Remain sufficiently flat (within max_z_variation in the z direction),
        - Be reasonably isotropic in xy (to avoid long narrow stripes).
      
            Args:
        xyz_gz_file : Path  to the gzipped (.gz) XYZ file containing atomic coordinates of the carbon substrate.
        surface_indices : list of int
            List of surface atom indices (0-based) to consider for clustering.
        diameter_nm : float Diameter of the nanoparticle (in nanometers), used to define the size of sliding xy grid cells. (1 nm = 10 Å)
        min_cluster_span : floatMinimum lateral spatial extent (in Å) required between atoms in a cluster (in xy-plane).Prevents selecting groups that are too small to accommodate the NP laterally.
        max_z_variation : float Maximum allowed difference in z-coordinates (in Å) within a cluster.Ensures that the candidate site is sufficiently flat.
        anisotropy_threshold : float Maximum allowed ratio between std(x) and std(y) (or vice versa) in the cluster.Rejects elongated, anisotropic clusters.
    
            Returns
        accepted_cluster_coords (list): List of the coordinates of the substrate flat area.

        """
        import numpy as np
        from pathlib import Path
        import gzip
        import random
        from scipy.spatial.distance import pdist
        from matplotlib import colormaps
        
        log_lines = [] 
        with gzip.open(xyz_gz_file, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    
        num_atoms = int(lines[0].strip())
        atom_lines = lines[2:2 + num_atoms]
    
        coords = []
        for line in atom_lines:
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            coords.append((x, y, z))
    
        xyz = np.array(coords)
        surface_xyz = xyz[surface_indices]
    
        # Define sliding grid parameters
        diameter_angstrom = diameter_nm * 10
    
        x_min, y_min = surface_xyz[:, :2].min(axis=0)
        x_max, y_max = surface_xyz[:, :2].max(axis=0)
        
        accepted_clusters = []
        cluster_dict = {}
        cluster_id = 0
        cluster_rejected_dict = {}
        cluster_rejected_id = 0
    
        x_starts = np.arange(x_min, x_max, 5.0)
        y_starts = np.arange(y_min, y_max, 10.0)
        accepted_cluster_coords=[]

        
        for i, x0 in enumerate(x_starts):
            for j, y0 in enumerate(y_starts):
                local_indices = [idx for idx, (x, y, _) in enumerate(surface_xyz)
                                 if x0 <= x < x0 + diameter_angstrom and y0 <= y < y0 + diameter_angstrom]
                msg = f"🔵🔵🔵 Grid cell ({i},{j}) at ({x0:.1f}, {y0:.1f}) → {len(local_indices)} atoms"
                # print(msg)
                log_lines.append(msg)
    
                if len(local_indices) >= 2:
                    coords_patch = surface_xyz[local_indices]
                    xy_patch = coords_patch[:, :2]
                    max_xy_dist = np.max(pdist(xy_patch)) if len(xy_patch) >= 2 else 0
                    std_x, std_y = np.std(xy_patch[:, 0]), np.std(xy_patch[:, 1])
                    anisotropy = max(std_x, std_y) / max(1e-6, min(std_x, std_y))
                    x_values= coords_patch[:, 0]
                    y_values= coords_patch[:, 1]
                    z_values = coords_patch[:, 2]
                    z_range = z_values.max() - z_values.min()
    
                    reasons = []
                    ok = []
                    if max_xy_dist < min_cluster_span:
                        reasons.append(f"too compact (xy span = {max_xy_dist:.2f} Å < {min_cluster_span} Å)")
                    else:
                        ok.append(f"compact (xy span = {max_xy_dist:.2f} Å > {min_cluster_span} Å)")
                    if z_range > max_z_variation:
                        reasons.append(f"not flat (z_range = {z_range:.2f} Å > {max_z_variation} Å)")
                    else:
                        ok.append(f"flat (z_range = {z_range:.2f} Å < {max_z_variation} Å)")
                    if anisotropy > anisotropy_threshold:
                        reasons.append(f"too elongated (anisotropy = {anisotropy:.2f} > {anisotropy_threshold})")
                    else:
                        ok.append(f"regular shape (anisotropy = {anisotropy:.2f} < {anisotropy_threshold})")


                    # NEW 

                     # Check overlap
                    atom_ids = [surface_indices[i] + 1 for i in local_indices]
                    overlap_found = False
                    for existing in accepted_clusters:
                        common = set(atom_ids) & set(existing)
                        if len(common) / max(len(atom_ids), len(existing)) > overlap_tolerance:
                            reasons.append(f"overlap > {int(overlap_tolerance*100)}% with existing cluster")
                            overlap_found = True
                            break


                    
                    if not reasons:
                        cluster_dict[cluster_id] = [surface_indices[i] + 1 for i in local_indices]
                        atom_ids = cluster_dict[cluster_id]
                        accepted_clusters.append(atom_ids)
                        select_line = f"select all; color atoms grey; select {', '.join(f'@{idx}' for idx in atom_ids)}; color atoms green"
                        z_mean = z_values.mean()
                        msg = f"  ✅ Cluster {cluster_id:2d} ✓ n = {len(local_indices):2d}, z mean = {z_mean:.2f}, min = {z_values.min():.2f}, max = {z_values.max():.2f}, max_xy_dist={max_xy_dist:.2f} [{min_cluster_span} Å], z_range={z_range:.2f} [{max_z_variation} Å], anisotropy={anisotropy:.2f}"
                        # print(msg)
                        log_lines.append(msg)
                        log_lines.append(select_line)
                        log_lines.append("")  # blank line between clusters
                        cluster_id += 1
                        accepted_cluster_coords.append(coords_patch) # NEW
                    else:
                        cluster_rejected_dict[cluster_rejected_id] = [surface_indices[i] + 1 for i in local_indices]
                        atom_ids = cluster_rejected_dict[cluster_rejected_id]
                        select_line = f"select all; color atoms grey; select {', '.join(f'@{idx}' for idx in atom_ids)}; color atoms red"
                        out_str = "; ".join(reasons)
                        ok_str = "; ".join(ok)
                        out_str = f"{out_str}; ✅ FULFILLED CRITERIA = {ok_str}"
                        msg = f"  ❌ Rejected: {out_str}"
                        # print(msg)
                        log_lines.append(msg)
                        log_lines.append(select_line)
                        log_lines.append("")  # blank line between clusters
                        cluster_rejected_id += 1
        return accepted_cluster_coords

   
    def place_NPsurface_on_grid(self, path, xyz_gz_file, surface_indices,tolerance,instanceWulff,surfaces_indices, element, structure, form, nRot, number, noOutput, circumsphere_diameter):

        """
        Creates XYZ files of NPs laying on one of their surface on a carbon grid.
        the interface NP/carbon substrate files, the NP laying on its surface facet.
        Initialize the class with CIF data, Wulff shapes information and size, and the tolerance distance between the NP and the carbon grid.
            Args:
        xyz_gz_file (str) : Path  to the gzipped (.gz) XYZ file containing atomic coordinates of the carbon substrate.
        tolerance (float, optional):  Tolerance distance between the NP and the carbon grid, be careful, if too small, chemical bonds appear in the interface.
        instanceWulff(class instance): Instance of the crystalNPs to create the Wulff NP.
        element (str): Compound elements, defined in create_NP_TEMimages()
        structure (str): Lattice, defined in create_NP_TEMimages()
        form (str): Wulff form,  defined in create_NP_TEMimages()
        nRot (int): Number of rotation of the NP along z (angle = 360/nRot)
        number (int): Index for the size, defined in create_NP_TEMimages()
        
            Returns:
        The XYZ files for each NPs (each compounds and sizes) laying on one of their surfaces on the carbon grid.
        
        """

        # 1. The NP surface
        # Place the NP on its surface
        plane = instanceWulff.trPlanes[0]
        if noOutput == False :
            print(f'Surface plane of the NP used = {plane}.')
        normal_plane= np.array(plane[:3]) # Normal of the NP surface plane
        
        # 2. The carbon surface
        # Find the plane that fits the best the cluster points (flat area of the carbon substrate surface where the NP will be on)
        accepted_cluster_coords= self.flat_areas_atoms(xyz_gz_file, surface_indices)

        # Select the flat-area cluster depending on the substrate file
        substrate_name = str(xyz_gz_file)
        if substrate_name.endswith('aC_relax_clean_10x10.xyz.gz'):
            cluster = accepted_cluster_coords[10]
        elif substrate_name.endswith('aC_relax_clean5x5.xyz.gz'):
            cluster = accepted_cluster_coords[2]
        else:
            cluster = accepted_cluster_coords[0]  # default: first flat area

        carbon_plane_positions = cluster # Positions of the atoms of the flat carbon surface
        pca = PCA(n_components=3)
        pca.fit(carbon_plane_positions)
        normal_carbon = pca.components_[-1]  # Normal vector but we don't know the sense (positive or negative)
        if normal_carbon[2] < 0: # positive z
            normal_carbon = -normal_carbon
        normal_carbon_unit = normal_carbon / np.linalg.norm(normal_carbon)
        center_carbon_plane = carbon_plane_positions.mean(axis=0)
                   
        # 3. Place the NPs close to the substrate
        # Compute the carbon surface distance from the origin
        dist_carbon = -np.dot(normal_carbon, center_carbon_plane)  # -d
        # Make the NP surface plane parallel to the carbon surface 
        rotated_positions = pyNMBu.rotateMoltoAlignItWithAxis(instanceWulff.NP.positions,axis=normal_plane,targetAxis=normal_carbon) 
        rotated_positions_on_carbon_unit = np.dot(rotated_positions - center_carbon_plane, normal_carbon_unit) # Project the NP atoms positions on the carbon surface normal vector : gives height from the carbon surface
        
        min_point_proj=np.min(rotated_positions_on_carbon_unit) # Find the closest NP atom from the carbon surface 
                
        # 3.1  Move the np towards the carbon surface horizontally
        center_np = rotated_positions.mean(axis=0)
        # On veut projeter center_np sur le plan défini par center_carbon_plane et normal_carbon_unit
        vec_to_plane = center_np - center_carbon_plane
        dist_to_plane = np.dot(vec_to_plane, normal_carbon_unit)
        proj_center_np_on_plane = center_np - dist_to_plane * normal_carbon_unit
        # Décalage latéral à appliquer
        lateral_shift = center_carbon_plane - proj_center_np_on_plane
        # print('Horizontal translation vector = ',lateral_shift)
        rotated_and_shifted_positions = rotated_positions + lateral_shift
               
        # 3.2 Move the np towards the carbon surface (1 Angs above) vertically
        # Translation_vector is for translating the NP vertically only
        translation_vector = (-min_point_proj + tolerance) * normal_carbon_unit
        translated_positions = rotated_and_shifted_positions + translation_vector
        center_of_rotation = translated_positions.mean(axis=0)
        
        # 3.3 Change the orientation of the NP along the plane xy
        for i in np.random.randint(0, 360, nRot)  :
            angle = i
            # translated_positions = pyNMBu.rotationMolAroundAxis(translated_positions, angle, normal_carbon_unit)
            new_positions = pyNMBu.rotation_around_axis_through_point(translated_positions,angle_deg=angle,axis=normal_carbon,center=center_of_rotation)
        
            # 4. Create the files for each NPs
            self._xyz_counter += 1
            xyz_filename = f"{path}/{element}_{structure}_{form}_{self._xyz_counter:06d}.xyz"
            self._xyz_metadata.append({
                "xyz_file": f"{element}_{structure}_{form}_{self._xyz_counter:06d}.xyz",
                "orientation": "surface",
                "angle_xy": int(angle),
                "angle_tilt": 0,
                "circumsphere_diameter_nm": round(circumsphere_diameter, 4),
            })
            # Read the carbone files
            carbon_lines = []
            with gzip.open(xyz_gz_file, 'rt', encoding='utf-8') as f0:
                i = 0
                for line in f0:
                    i += 1
                    if i >= 3:  # on saute les 2 premières lignes (nombre d'atomes et commentaire)
                        carbon_lines.append(line.strip())     
            n_carbon = len(carbon_lines)
            n_au = len(new_positions)
            total_atoms = n_carbon + n_au
            with open(xyz_filename, "w") as f:
                f.write(f"{total_atoms}\n")
                f.write("Carbon substrate + translated nanoparticle\n")
            
                # Écriture des atomes carbone
                for line in carbon_lines:
                    f.write(f"{line}\n")
            
                # Écriture des atomes or (NP)
                for pos in new_positions:
                    f.write(f"{element} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            if noOutput == False :
                print(f" \033[34m File written : {xyz_filename} \033[0m")  
                print('')
    
    def place_NPedge_on_grid(self, path,xyz_gz_file,surface_indices,tolerance,instanceWulff,surfaces_indices, element, structure, form, nRot, number, noOutput, circumsphere_diameter):

        """
        Creates XYZ files of the NPs laying on one of their edges on a carbon grid.
        First, the edge and the two adjacent planes of the NP are computed. Then,  the edge of the NP is made
        parallel to the carbon flat surface, and the bisector of the two planes  aligned with the normal of the carbon flat surface. 
        The idea is to have a perfectly centered NP laying on its edge and be able to tilt it both sides.
        
            Args:
        xyz_gz_file (str) : Path  to the gzipped (.gz) XYZ file containing atomic coordinates of the carbon substrate.
        tolerance (float, optional):  Tolerance distance between the NP and the carbon grid, be careful, if too small, chemical bonds appear in the interface.
        instanceWulff(class instance): Instance of the crystalNPs to create the Wulff NP.
        surfaces_indices (list): Indices of the atoms of the flat carbon surface
        element (str): Compound elements, defined in create_NP_TEMimages()
        structure (str): Lattice, defined in create_NP_TEMimages()
        form (str): Wulff form,  defined in create_NP_TEMimages()
        nRot (int): Number of rotation of the NP (angle = 360/nRot)
        number (int): Index for the size, defined in create_NP_TEMimages()

            Returns:
        The XYZ files for each NPs (each compounds and sizes) laying on one of their edges on the carbon grid.
        
        """
     
        # 1. Find the NP edge and the two adjacent planes.
        from collections import defaultdict
        edge_count = defaultdict(int)
        
        # Using simplices of the Hull algorithm
        for triangle in instanceWulff.simplices:
            for i in range(3):
                a = triangle[i]
                b = triangle[(i + 1) % 3]
                edge = tuple(sorted((a, b)))
                edge_count[edge] += 1 
        # Extract the edge
        true_edges = [edge for edge, count in edge_count.items() if count == 2]

        for edge in true_edges:
            a, b = edge
            atom_a = instanceWulff.NP.positions[a]
            atom_b = instanceWulff.NP.positions[b]
            edge = pyNMBu.vector(instanceWulff.NP.positions,a,b)
            adjacent_planes = []
        
            for planes in instanceWulff.trPlanes:
                a1, b1, c1, d1 = planes
                if (abs(a1 * atom_a[0] + b1 * atom_a[1] + c1 * atom_a[2] + d1) < 1e-2 and
                    abs(a1 * atom_b[0] + b1 * atom_b[1] + c1 * atom_b[2] + d1) < 1e-2):
                    adjacent_planes.append(planes)
        
            if len(adjacent_planes) >= 2:
                p1, p2 = adjacent_planes
                n1 = np.array(p1[:3]) / np.linalg.norm(p1[:3])
                n2 = np.array(p2[:3]) / np.linalg.norm(p2[:3])
                bisector = n1 + n2
                if np.linalg.norm(bisector) > 1e-6:
                    break  # good edge found
            
        
        # print('adjacent_planes',adjacent_planes)
        if len(adjacent_planes) >= 2 :
            if noOutput == False :
                print('Two adjacent planes found.')

            # Normals of the two planes in order to compute the bisector
            normal_p1 = np.array(p1[:3])
            normal_p2 = np.array(p2[:3])
            normal_p1_unit = normal_p1 / np.linalg.norm(normal_p1)
            normal_p2_unit = normal_p2 / np.linalg.norm(normal_p2)
            
            # Compute the bisector of the two planes
            bisector = normal_p1_unit + normal_p2_unit
            if np.linalg.norm(bisector) < 1e-6:
                if not noOutput :
                    print("Bisector not well defined, normals of the planes opposite.")
            bisector_unit = bisector / np.linalg.norm(bisector)
            
            # Carbon surface : compute the normal of the surface and a vector contained on the surface (using PCA)
            accepted_cluster_coords = self.flat_areas_atoms(xyz_gz_file, surface_indices)

            # Select the flat-area cluster depending on the substrate file
            substrate_name = str(xyz_gz_file)
            if substrate_name.endswith('aC_relax_10x10.xyz.gz'):
                cluster = accepted_cluster_coords[10]
            elif substrate_name.endswith('aC_relax_clean5x5.xyz.gz'):
                cluster = accepted_cluster_coords[2]
            else:
                cluster = accepted_cluster_coords[0]  # default: first flat area

            carbon_plane_positions = cluster
            pca = PCA(n_components=3)
            pca.fit(carbon_plane_positions)
            normal_carbon = pca.components_[-1]
            plane_x = pca.components_[0]  # vector in the plane (xy) = carbon plane

            # Watchout the sign (z)
            if normal_carbon[2] < 0:
                normal_carbon = -normal_carbon
            normal_carbon_unit = normal_carbon / np.linalg.norm(normal_carbon)
            center_carbon_plane = carbon_plane_positions.mean(axis=0)
            
            # Unique rotation to : 1) make the edge of the NP parallel to the carbon surface, and 2) align the bisector with the carbon normal
            # Local coordinate system
            edge_unit = edge / np.linalg.norm(edge)
            z_NP = bisector_unit
            x_NP = edge_unit
            y_NP = np.cross(z_NP, x_NP)
            y_NP /= np.linalg.norm(y_NP)
            x_NP = np.cross(y_NP, z_NP)
            R_NP = np.stack([x_NP, y_NP, z_NP], axis=1)

            # Target coordinate system
            z_target = normal_carbon_unit
            x_target = plane_x / np.linalg.norm(plane_x)
            y_target = np.cross(z_target, x_target)
            y_target /= np.linalg.norm(y_target)
            x_target = np.cross(y_target, z_target)
            R_target = np.stack([x_target, y_target, z_target], axis=1)
            
            # Global rotation
            center_of_rotation = instanceWulff.NP.positions.mean(axis=0)
            R = R_target @ R_NP.T
            rotated_positions = (R @ (instanceWulff.NP.positions - center_of_rotation).T).T + center_of_rotation
            
            # Lateral translation
            center_np = rotated_positions.mean(axis=0)
            vec_to_plane = center_np - center_carbon_plane
            dist_to_plane = np.dot(vec_to_plane, normal_carbon_unit)
            proj_center_np_on_plane = center_np - dist_to_plane * normal_carbon_unit
            lateral_shift = center_carbon_plane - proj_center_np_on_plane
            rotated_and_shifted_positions = rotated_positions + lateral_shift
            
            # Vertical translation
            rotated_positions_on_carbon_unit = np.dot(rotated_and_shifted_positions - center_carbon_plane, normal_carbon_unit)
            min_point_proj = np.min(rotated_positions_on_carbon_unit)
            translation_vector = (-min_point_proj + tolerance) * normal_carbon_unit
            translated_positions = rotated_and_shifted_positions + translation_vector
            new_center = translated_positions.mean(axis=0)
            
            # Angle between the 2 planes
            dot_product = np.dot(normal_p1_unit, normal_p2_unit)
            angle_n1_n2 = np.arccos(dot_product)
            angle_n1_n2 = np.degrees(angle_n1_n2)
            if not noOutput :
                print(f"Angle entre les 2 plans = {angle_n1_n2:.2f}°")   
            # Max angle between the planes and the (xy) plane
            angle_max = (180 - angle_n1_n2 ) / 2
            if not noOutput :
                print(f"Angle max entre les plans et le plan (xy) = {angle_max:.2f}°")  
            from scipy.spatial.transform import Rotation as R
            rotation_axis = edge_unit

            # Tilt the NP along z
            for angle in np.random.randint(-angle_max, angle_max, nRot) : 
                tilt_rotation = R.from_rotvec(angle * rotation_axis)
                tilt_center = new_center
                positions_tilted = tilt_rotation.apply(translated_positions - tilt_center) + tilt_center
                center_of_rotation = positions_tilted.mean(axis=0)
                
                 #  Change the orientation of the NP along the plane xy
                for i in np.random.randint(0, 360, nRot)  :
                    angle2 = i
                    # translated_positions = pyNMBu.rotationMolAroundAxis(translated_positions, angle, normal_carbon_unit)
                    new_positions = pyNMBu.rotation_around_axis_through_point(positions_tilted,angle_deg = angle2,axis = normal_carbon,center = center_of_rotation)
                       
                    # Write and save the XYZ file
                    self._xyz_counter += 1
                    xyz_filename = f"{path}/{element}_{structure}_{form}_{self._xyz_counter:06d}.xyz"
                    self._xyz_metadata.append({
                        "xyz_file": f"{element}_{structure}_{form}_{self._xyz_counter:06d}.xyz",
                        "orientation": "edge",
                        "angle_xy": int(angle2),
                        "angle_tilt": int(angle),
                        "circumsphere_diameter_nm": round(circumsphere_diameter, 4),
                    })
                    carbon_lines = []
                    with gzip.open(xyz_gz_file, 'rt', encoding='utf-8' ) as f0:
                        i = 0
                        for line in f0:
                            i += 1
                            if i >= 3:
                                carbon_lines.append(line.strip())
                    
                    n_carbon = len(carbon_lines)
                    n_au = len(new_positions)
                    total_atoms = n_carbon + n_au
                    
                    with open(xyz_filename, "w") as f:
                        f.write(f"{total_atoms}\n")
                        f.write("Carbon substrate + translated nanoparticle\n")
                        for line in carbon_lines:
                            f.write(f"{line}\n")
                        for pos in new_positions:
                            f.write(f"{element} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                    if not noOutput :
                        print(f"\033[32m File written : {xyz_filename} \033[0m")
                        print('')


    def create_NP_TEMimages(self, tolerance, noOutput,min_size,max_size, nRot, path, xyz_gz_file):
        """
        Generate Wulff shapes and their files for a single specific CIF coumpound.
         Args:
        cif file (file): singular cif file
        noOutput (bool): if bool=False : details of the files
        min_size (float, optional) : Mainimal size for the NPs, equals to the diameter of the circumscribed sphere, equals 0 nm by default.
        max_size (float, optional) : Maximal size for the NPs, equals to the diameter of the circumscribed sphere, equals 50 nm by default.
        path (str) : path that will contain the created xyz/CIF files
        
        """   
        surface_indices = self.find_surface_atoms(xyz_gz_file, grid_size=2, z_tolerance=6)
        for cif_name, cif_file in self.cif_data['cif file'].items():
                    
            # Extract the structure name for the name of the files, for example 'Rutile' or 'Anatase' or 'Alpha'
            self.cif_name=cif_name
            if len(self.cif_name.split())==2 : # For the name of the files, for example 'Rutile' or 'Anatase'
                structure=self.cif_name.split()[1]
            else : 
                structure=None
            if not noOutput :
                print(f'\n\033[1m {bg.LIGHTBLUEB} {cif_name.center(50)}\033[0m\n') 
            cif_info = pyNMBu.load_cif(self,cif_file,noOutput)
            crystal_system_name = self.ucBL.__class__.__name__
            direction=[0,0,1]
            d_hkl=pyNMBu.interPlanarSpacing(direction,self.ucUnitcell,crystal_system_name)*0.1 #nm    
            if not noOutput:
                print(f'\033[1m d_hkl={d_hkl} nm \033[0m')
            
            if len(cif_name.split()) >= 2 :  # to exclude TiO2 and NaCl
                for form, row in self.wulff_shapes.iterrows():
                    lattices = [l.strip() for l in row['Bravais lattice'].split(',')]
                    if self.crystal_type  in lattices:  # Verify if the lattice of the compound matches the lattice of the wulff form
                        index=0 
                        if not noOutput :
                            print(f"\n {bg.LIGHTGREENB} {self.crystal_type} corresponds to the lattices {lattices} of the Wulff {form} form. \033[0m \n")
                        number=0
                        # Create instances for each form and size
                        for i in self.sizes :
                            # number+=1
                            size= [i[0]*d_hkl]
                            index += 1 
                            TestNP = cyNP.Crystal(
                                crystal=f'{cif_name}',
                                userDefCif=cif_info['cif_path'],
                                shape=f"Wulff: {form}",
                                sizesWulff= size,
                                threshold=0.001,
                                thresholdCoreSurface=2,
                                postAnalyzis=True,
                                jmolCrystalShape=True,
                                noOutput=True,
                                aseView=False,
                                skipSymmetryAnalyzis=True) 
                            
                            circumsphere_diameter=TestNP.radiusCircumscribedSphere*2*0.1 # Setting a maximal size of NPs : circumscribed sphere diameter
                            if min_size<=circumsphere_diameter<max_size :
                                number+=1
                                if not noOutput :
                                    print(f'\033[1m Generating size is {size[0]:.4f} nm and is equal to dhkl multiplied by {i}.\033[0m ')
                                    print(f'\033[1m  Circumscribed sphere diameter ={circumsphere_diameter} nm and is in the interval [{min_size},{max_size}].\033[0m')
                                # Names of the files 
                                element = self.cif_name.split()[0]
                                structure=self.crystal_type
                                if not noOutput :
                                    print('')
                                    print(f' Generate NPs laying on one of their surfaces.')
                                self.place_NPsurface_on_grid(path,xyz_gz_file, surface_indices,tolerance, TestNP, surface_indices, element, structure, form, nRot, number, noOutput, circumsphere_diameter)
                                if not noOutput :
                                    print(f' Generate NPs laying on one of their edges.')
                                self.place_NPedge_on_grid(path, xyz_gz_file, surface_indices,tolerance, TestNP, surface_indices, element, structure, form, nRot, number, noOutput, circumsphere_diameter)
                                
                            else :
                                if  min_size>=circumsphere_diameter :
                                    if not noOutput :
                                        print(f'\033[1m The circumscribed sphere diameter of the NP ={circumsphere_diameter} nm is smaller than the minimal size : {min_size}nm chosen. \033[0m') 
                                if circumsphere_diameter>max_size :   
                                    if not noOutput :
                                        print(f'\033[1m The circumscribed sphere diameter of the NP ={circumsphere_diameter} nm is greater than the maximal size : {max_size}nm chosen. \033[0m') 
                                    break


                # Add regIco
                form = 'regico'
                element = self.cif_name.split()[0]
    

                if self.crystal_type=='fcc' :
                    dist= pyNMBu.FindInterAtomicDist(self)# Extract the interatomic distance
                    if not noOutput :
                        print(f"{bg.LIGHTGREENB}Addinng icosahedron with crystal type {self.crystal_type} and interatomic distance = {dist:.4f} nm.")
                    
                    for i in np.arange(1,1000) :
                        index += 1 
                        if not noOutput :
                            print(f'{bg.LIGHTBLUEB} Number of bonds is {i}')
                        TestNP2 =pNP.regIco(
                            element=element,
                            Rnn=dist,
                            nShell=i,
                            shape='regico',
                            postAnalyzis=True,
                            aseView=False,
                            thresholdCoreSurface=1,
                            skipSymmetryAnalyzis=True,
                            noOutput= True
                        )
                        
                        circumsphere_diameter=TestNP2.radiusCircumscribedSphere()*2*0.1 # Setting a maximal size of NPs : circumscribed sphere diameter
                   
                        if min_size<=circumsphere_diameter<max_size :
                            number+=1
                            if not noOutput :
                                print(f'\033[1m Generating size is {size[0]:.4f} nm.\033[0m ')
                                print(f'\033[1m  Circumscribed sphere diameter ={circumsphere_diameter} nm and is in the interval [{min_size},{max_size}].\033[0m')
                            # Names of the files 
                            element = self.cif_name.split()[0]
                            structure=self.crystal_type
                            if not noOutput :
                                print('')
                                print(f' Generate NPs laying on one of their surfaces.')
                            self.place_NPsurface_on_grid(path,xyz_gz_file, surface_indices,tolerance, TestNP2, surface_indices, element, structure, form, nRot, number, noOutput, circumsphere_diameter)
                            if not noOutput :
                                print(f' Generate NPs laying on one of their edges.')
                            self.place_NPedge_on_grid(path, xyz_gz_file, surface_indices,tolerance, TestNP2, surface_indices, element, structure, form, nRot, number, noOutput, circumsphere_diameter)
                            
                        else :
                            if  min_size>=circumsphere_diameter :
                                if not noOutput :
                                    print(f'\033[1m The circumscribed sphere diameter of the NP ={circumsphere_diameter} nm is smaller than the minimal size : {min_size}nm chosen. \033[0m') 
                            if circumsphere_diameter>max_size :   
                                if not noOutput :
                                    print(f'\033[1m The circumscribed sphere diameter of the NP ={circumsphere_diameter} nm is greater than the maximal size : {max_size}nm chosen. \033[0m') 
                                break

                elif self.crystal_type=='bcc': # do not make an icosahedron 
                    # dist= pyNMBu.FindInterAtomicDist(self) # Extract the interatomic distance
                    if not noOutput :
                        print(f'{bg.LIGHTREDB} No icosahedron for bcc lattice. ')

                elif self.crystal_type=='hcp': # do not make an icosahedron
                    dist= pyNMBu.FindInterAtomicDist(self) # Extract the interatomic distance
                    if not noOutput :
                        print(f'{bg.LIGHTREDB} No icosahedron for hcp lattice.')
                else : 
                    dist=None
                    if not noOutput :
                        print(f'{bg.LIGHTREDB} No interatomic distance found.')

        # Save XYZ metadata CSV
        if self._xyz_metadata:
            df_xyz_meta = pd.DataFrame(self._xyz_metadata)
            df_xyz_meta.to_csv(f"{path}/xyz_metadata.csv", index=False)
            if not noOutput:
                print(f"[CSV] Saved XYZ metadata: {path}/xyz_metadata.csv")


############################################## Class for HRTEM simulations


class CreateHRTEMImage:
    """
    Simulate High-Resolution Transmission Electron Microscopy (HRTEM) images
    from XYZ atomic structure files using the abTEM multislice framework.

    This class reads XYZ files produced by ``CreateHRTEMStructure`` (nanoparticle
    on an amorphous-carbon substrate) and generates realistic HRTEM images by
    chaining the following physical steps:

    **Simulation pipeline (per XYZ file)**

    1. ``place_atoms_grid``        – Crop the carbon substrate to a thin slab.
    2. ``calculate_potentials``     – Build frozen-phonon potentials (Debye-Waller).
    3. ``create_wave_function``     – Create an incident plane wave.
    4. ``perform_multislice``       – Propagate the wave through the potential slices.
    5. ``calculate_ctf``            – Apply the CTF (Cs, defocus, astigmatism C12/phi12,
                                      temporal coherence via focal_spread).
    6. ``apply_astigmatism``        – Copy CTF for incoherent imaging.
    7. ``apply_partial_coherence``  – Compute intensity from the ensemble of frozen-phonon
                                      exit waves convolved with the CTF.
    8. ``apply_poisson_noise``      – Add shot noise at a given electron dose.
    9. ``calculate_mtf``            – Apply the detector Modulation Transfer Function.
    10. ``generate_and_save_image`` – Save the PNG image and a per-image CSV metadata file.
    11. ``generate_mask_image``     – (optional) Save a binary segmentation mask of the NP.

    **Aberration notation** follows the Krivanek convention used by abTEM:

    ============  ============================================  ==========
    Symbol        Physical meaning                              Unit
    ============  ============================================  ==========
    Cs (C30)      3rd-order spherical aberration                Angstrom
    C10           Defocus (set to Scherzer by default)          Angstrom
    C12 / phi12   2-fold astigmatism amplitude / azimuth        Angstrom / rad
    focal_spread  Temporal-coherence envelope = Cc * dE / E     Angstrom
    ============  ============================================  ==========

    **Outputs**

    For each input ``{name}.xyz`` the class produces:

    - ``{name}_{index}.png``              – HRTEM image (grayscale, 512x512 px).
    - ``{name}_{index}_metadata.csv``     – All simulation parameters + NP metadata.
    - ``{name}_{index}_mask.png``          – Binary mask (if ``masking_images=True``).

    Parameters
    ----------
    path_input : str
        Directory containing the input ``.xyz`` files and optionally
        ``xyz_metadata.csv`` (produced by ``CreateHRTEMStructure``).
    path_output : str
        Directory where output PNG and CSV files are saved.
    sampling : float, default 0.05
        Grid sampling for each dimension in Ångstrom per grid point.
        Impact on the time computation.
    masking_images : bool, default True
        If True, generate a binary segmentation mask for each image.
    phonon_config : int, default 8
        The number of configurations around the equilibrium position. 
    sigmas : float, default 0.1
        The standard deviation of the displacements in Angstrom for frozen phonons.
    Cs_value : float, default -80 (i.e. -8e-6 * 1e10 Angstrom)
        Spherical aberration coefficient C30 in Angstrom.
        Negative for aberration-corrected microscopes.
    C12 : float, default 0
        Two-fold astigmatism amplitude in Angstrom.
    phi12 : float, default 0
        Azimuthal angle of the two-fold astigmatism in radians.
    Cc_value : float, default 1e7 (i.e. 1.0e-3 * 1e10 Angstrom = 1 mm)
        Chromatic aberration coefficient in Angstrom.
    semiangle_cutoff_value : int, default 45
        Objective aperture semiangle cutoff in mrad.
    energy_spread : float, default 0.35
        Energy spread dE of the electron source in eV.
        Combined with Cc and E to compute the focal spread:
        ``focal_spread = Cc_value * energy_spread / energy``.
    c1 : float, default -0.6
        MTF parameter – asymptotic contrast floor.
    c2 : float, default 0.1
        MTF parameter – half-power spatial frequency scaling.
    c3 : float, default 1.0
        MTF parameter – roll-off exponent.
    dose_poisson_noise : float, default 1e4
        Electron dose for Poisson shot noise in e-/Angstrom^2.
    slice_thickness : float, default 1
        Thickness of each potential slice for the multislice algorithm
        in Angstrom.
        Smaller values are more accurate but slower.
    noOutput : bool, default True
        Print diagnostic information (defocus, focal_spread, etc.).
    energy : float, default 200e3
        Electron beam energy in eV  (200 keV by default).
    substrate_size : list of float, default [10, 10, 10]
        Simulation super-cell size [x, y, z] in nanometers.
    device : str, default 'gpu'
        Computation device: ``'gpu'`` (cupy/CUDA) or ``'cpu'``.

    Attributes
    ----------
    focal_spread : float
        Temporal-coherence envelope width computed as
        ``Cc_value * energy_spread / energy`` (Angstrom).
    defocus : float
        Scherzer defocus C10 in Angstrom (set after ``calculate_ctf``).

    Examples
    --------
    >>> CreateHRTEMImage(
    ...     path_input="output_xyz/",
    ...     path_output="output_hrtem/",
    ...     Cs_value=-8e-6 * 1e10,
    ...     energy=200e3,
    ...     device="gpu",
    ... )
    """
    
    def __init__(self,  path_input: str, path_output: str, sampling: float = 0.05, masking_images: bool = True,   
             phonon_config: int = 8, Cs_value: float = -8e-6 * 1e10, C12: float = 0,
             phi12: float = 0, Cc_value: float = 1.0e-3 * 1e10, 
             semiangle_cutoff_value: int = 45, energy_spread: float = 0.35, 
             c1 : float = -0.6, c2 : float = 0.1,  c3 : float= 1.0,
             dose_poisson_noise: float = 1e4,
             sigmas: float = 0.1, slice_thickness: float = 1,
             noOutput: bool = True, energy: float = 200e3, substrate_size: list = [10, 10, 10],
             device: str = 'gpu'):

        self.path_input = path_input
        self.path_output = path_output
        abtem.config.set({"device": device, "fft": "fftw"})
        self.masking_images = masking_images
        self.sampling = sampling
        self.phonon_config = phonon_config
        self.Cs_value = Cs_value
        self.C12 = C12            # 2-fold astigmatism amplitude (Å)
        self.phi12 = phi12        # 2-fold astigmatism angle (rad)
        self.Cc_value = Cc_value  # chromatic aberration coefficient (Å)
        self.semiangle_cutoff_value = semiangle_cutoff_value
        self.energy_spread = energy_spread  # energy spread dE (eV)
        # focal_spread = Cc * dE / E  (temporal coherence envelope)
        self.focal_spread = Cc_value * energy_spread / energy
        self.noOutput = noOutput
        self.energy = energy
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.dose_poisson_noise = dose_poisson_noise  # e⁻/Å²
        self.sigmas = sigmas              # Debye-Waller rms displacement (Å)
        self.slice_thickness = slice_thickness  # multislice slice thickness (Å)
        self.substrate_size = substrate_size
        self.device = device
        self.cluster = None
        self.substrate = None
        self.atoms = None
        self.frozen_phonons = None
        self.potential = None
        self.wave = None
        self.exit_wave = None
        self.ctf = None
        self.incoherent_ctf = None
        self.measurement_ensemble = None
    
        self.hrtem_image()
        

    def hrtem_image(self):
        """Main method to generate the HRTEM image."""
        
        input_files=Path(self.path_input)

        # Load XYZ metadata (orientation, angles, size) if available
        xyz_meta_path = input_files / "xyz_metadata.csv"
        xyz_meta_lookup = {}
        if xyz_meta_path.exists():
            df_xyz_meta = pd.read_csv(xyz_meta_path)
            xyz_meta_lookup = {row["xyz_file"]: row.to_dict() for _, row in df_xyz_meta.iterrows()}

        index = 0
        for f in sorted(input_files.glob("*.xyz")): # loop on all the output XYZ files
            index += 1
            xyz_meta = xyz_meta_lookup.get(f.name, {})
            self.atoms = read(f) # from XYZ files to ASE objects for AbTEM
            self.place_atoms_grid()
            self.calculate_potentials()
            self.create_wave_function()
            self.perform_multislice()
            self.calculate_ctf()
            self.apply_astigmatism()
            self.apply_partial_coherence()
            self.apply_poisson_noise()
            self.calculate_mtf(f, index, xyz_meta) 
            if self.masking_images:
                self.generate_mask_image(f)


    def place_atoms_grid(self):
        """ 
        Function that only takes 2nm of the carbon substrate.
        """
        # Define the z-limits for the substrate slice (z is perpendicular to the substrate plane)
        z_min = 30
        z_max = 100
        substrat_size = np.array(self.substrate_size) * 10  # Convert nm to Å
        atoms = self.atoms[[z_min <= atom.position[2] <= z_max for atom in self.atoms]]
        atoms.set_cell(substrat_size)  # Convert nm to Å
        atoms.center(axis=2, vacuum=2)
        self.atoms = atoms


    def display_cluster_views(self):
        """Display the cluster from different views."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        abtem.show_atoms(self.cluster, plane="xy", ax=ax1, title="Beam view")
        abtem.show_atoms(self.cluster, plane="yz", ax=ax2, title="Side view")


    def display_combined_views(self):
        """Display the combined views of the substrate and cluster."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        abtem.show_atoms(self.atoms, plane="xy", ax=ax1, title="Beam view")
        abtem.show_atoms(self.atoms, plane="xz", ax=ax2, title="Side view")


    def calculate_potentials(self):
        """Calculate the potentials for the system."""
        self.frozen_phonons = abtem.FrozenPhonons(self.atoms, self.phonon_config, sigmas=self.sigmas)
        self.potential = abtem.Potential(
            self.frozen_phonons,
            sampling= self.sampling,
            projection="infinite",
            slice_thickness=self.slice_thickness,
        )


    def create_wave_function(self):
        """Create the wave function for the simulation."""
        self.wave = abtem.PlaneWave(energy=self.energy)


    def perform_multislice(self):
        """Perform the multislice calculation."""
        self.exit_wave = self.wave.multislice(self.potential)
        self.exit_wave.compute()


    def calculate_ctf(self):
        """Calculate the contrast transfer function (CTF) with astigmatism and focal spread."""
        self.ctf = abtem.CTF(
            Cs=self.Cs_value,
            energy=self.wave.energy,
            defocus="scherzer",
            semiangle_cutoff=self.semiangle_cutoff_value,
            C12=self.C12,
            phi12=self.phi12,
            focal_spread=self.focal_spread,
        )
        if self.noOutput == False :
            print(f"defocus = {self.ctf.defocus:.2f} Å")
        self.defocus = self.ctf.defocus  # C10 in Å
        if self.C12 != 0:
            print(f"C12 (astigmatism) = {self.C12:.2f} Å, phi12 = {self.phi12:.4f} rad")
        if self.focal_spread != 0:
            print(f"focal_spread = {self.focal_spread:.2f} Å  (Cc={self.Cc_value:.2e} Å, dE={self.energy_spread} eV)")


    def apply_astigmatism(self):
        """Copy CTF for incoherent imaging (astigmatism already set in calculate_ctf)."""
        self.incoherent_ctf = self.ctf.copy()


    def apply_partial_coherence(self):
        """Apply partial coherence to the exit wave."""
        self.measurement_ensemble = self.exit_wave.apply_ctf(self.incoherent_ctf).intensity()
    

    def apply_poisson_noise(self):
        measurement = self.measurement_ensemble.mean(0)
        self.noisy_measurement = measurement.poisson_noise(dose_per_area=self.dose_poisson_noise)


    def calculate_mtf(self, f, index, xyz_meta=None):
        """
        The Modulation Transfer Function (MTF) is a measure of how well the contrast in an object is transferred to an image by a detector.*
        It characterizes the fidelity of the spatial frequency content of the object in the resulting image.
        """
        if xyz_meta is None:
            xyz_meta = {}
        from numpy.fft import fft2, ifft2, fftshift, ifftshift

        # parameters
        pixel_size = self.sampling
        # Compute the spatial frequencies q

        # cmb de angs sont couverts par un pixel
        q_N = 1 / (2 * pixel_size)
        # q_N = abtem.transfer.nyquist_sampling(self.semiangle_cutoff_value, self.energy)
        # print('q_N',q_N)
        # Compute the spatial frequencies q
        # noisy_measurement is already averaged over phonon configs (2D)
        mean_data = self.noisy_measurement.array
        # mean_data = np.mean(image_data, axis=0)
        # print('mean_data.shape',mean_data.shape)
        
        ny, nx = mean_data.shape
        qx = np.fft.fftfreq(nx, d=pixel_size)
        qy = np.fft.fftfreq(ny, d=pixel_size)
        qx, qy = np.meshgrid(qx, qy)
        q = np.sqrt(qx**2 + qy**2)

        # for c1 in self.c1 :
        #     for c2 in self.c2:
        #         for c3 in self.c3:
        # Compute the MTF
        if self.device == 'cpu' :
            mtf = (1 - self.c1) / (1 + (q / (2 * self.c2 * q_N))**self.c3) + self.c1
            # Apply MTF
            image_fft = fft2(mean_data) # fourier transform of the image: image in the frequency space (where low and high frequencies are separated)
            image_fft_filtered = image_fft * fftshift(mtf) # each spatial frequency is multiplied by its MFT value
            self.measurement_ensemble = np.real(ifft2(image_fft_filtered)) # back to the real space of the iamge 
            self.generate_and_save_image(f, index, xyz_meta)
        if self.device == 'gpu' : 
            import cupy as cp
            from cupy.fft import fftshift 
            mtf =cp.asarray((1 - self.c1) / (1 + (q / (2 * self.c2 * q_N))**self.c3) + self.c1)
            # Apply MTF
            image_fft = fft2(mean_data) # fourier transform of the image: image in the frequency space (where low and high frequencies are separated)
            image_fft_filtered = image_fft * fftshift(mtf) # each spatial frequency is multiplied by its MFT value
            self.measurement_ensemble = np.real(ifft2(image_fft_filtered)) # back to the real space of the iamge 
            self.generate_and_save_image(f, index, xyz_meta)

    def generate_and_save_image(self, f, index, xyz_meta=None):
        """Generate and save the final image."""
        if xyz_meta is None:
            xyz_meta = {}
  
        # 1. The PNG HRTEM image
        
        self.final_filename = f'{self.path_output}/{f.stem}_{index:07d}'
        print(f"File is {self.final_filename}.png")
        plt.figure(figsize=(5.12, 5.12))  # Taille en pouces (ex: 6x6)
        plt.imshow(self.measurement_ensemble, cmap='gray', origin='lower')
        plt.axis('off')  # Pas d’axes
        plt.tight_layout(pad=0)  # Pas de bordures
        plt.savefig(f"{self.final_filename}.png", dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 2. The CSV file of metadata
        
        filename_csv = f'{self.final_filename}_metadata'
        path_csv = f"{filename_csv}.csv"
        # Parse element/structure/shape from simplified filename
        parts = f.stem.split('_')
        # Flatten metadata for CSV
        metadata_flat = {
            "id": f'{f.stem}_{index:07d}',
            "element": parts[0] if len(parts) > 0 else "",
            "crystal_structure": parts[1] if len(parts) > 1 else "",
            "shape": parts[2] if len(parts) > 2 else "",
            "orientation": xyz_meta.get("orientation", ""),
            "angle_xy_deg": xyz_meta.get("angle_xy", ""),
            "angle_tilt_deg": xyz_meta.get("angle_tilt", ""),
            "circumsphere_diameter_nm": xyz_meta.get("circumsphere_diameter_nm", ""),
            "sampling_A-1": self.sampling,
            "phonon_config": self.phonon_config,
            "Cs_value_A": self.Cs_value,
            "defocus_C10_A": self.defocus,
            "C12_A": self.C12,
            "phi12_rad": self.phi12,
            "Cc_value_A": self.Cc_value,
            "energy_spread_eV": self.energy_spread,
            "focal_spread_A": self.focal_spread,
            "semiangle_cutoff_value_mrad": self.semiangle_cutoff_value,
            "energy_eV": self.energy,
            "mtf_c1":self.c1,
            "mtf_c2":self.c2,
            "mtf_c3":self.c3,
            "dose_poisson_noise_e-A-2": self.dose_poisson_noise,
            "sigmas_A": self.sigmas,
            "slice_thickness_A": self.slice_thickness,
            "substrate_size_x_nm":self.substrate_size[0],
            "substrate_size_y_nm":self.substrate_size[1]
        }
    
        df = pd.DataFrame([metadata_flat])
        df.to_csv(path_csv, index=False)
        print(f"[CSV] Saved metadata: {path_csv}")
  
        print('--------Finished---------------')


    def generate_mask_image(self, f):
        """
        Generate and save the masked HRTEM image for segmentation purposes.
        The background is set to zero and the nanoparticule to 1.
        A disk corresponding to the Van der Waals radius is drawn for each atom.
        """
        from ase.data import vdw_radii, covalent_radii
        from skimage.draw import disk

        # 1. Dimensions of the HRTEM image just generated
        shape = self.measurement_ensemble.shape
        pixel_resolution = self.sampling  # spatial resolution between pixels (angstroms per pixel)

        # 2. Initialize the mask image with zeros (background)
        mask_image = np.zeros(shape, dtype=np.uint8)

        # 3. Get the atoms of the nanoparticle (exclude Carbon)
        np_atoms = [atom for atom in self.atoms if atom.symbol != 'C']

        # 4. Draw a disk for each atom on the mask
        for atom in np_atoms:
            # Atom coordinates in Angstroms
            x, y = atom.position[0], atom.position[1]

            # Convert to pixel coordinates
            # The image origin (0,0) is top-left, but atom coordinates can be anywhere.
            # We need to align them with the image grid.
            px, py = int(y / pixel_resolution), int(x / pixel_resolution)

            # Get Van der Waals radius in Angstroms and convert to pixels
            radius_ang = vdw_radii[atom.number]
            if np.isnan(radius_ang): # sometimes not defined
                radius_ang = covalent_radii[atom.number] # Fallback to covalent radius

            if np.isnan(radius_ang): # sometimes not defined
                radius_ang = 1.5  # Default value if both are undefined

            radius_px = int(radius_ang / pixel_resolution)

            # Draw a disk for the atom on the mask, checking boundaries
            rr, cc = disk((py, px), radius_px, shape=shape)
            mask_image[rr, cc] = 1

        # 5. Save the mask image
        base_filename = self.final_filename
        mask_filename = f"{base_filename}_mask.png"
        
        # Use origin='lower' to match the orientation of the HRTEM image

        plt.figure(figsize=(5.12, 5.12))  # Taille en pouces (ex: 6x6)
        plt.imshow(mask_image, cmap='gray', origin='lower')
        plt.axis('off')  # Pas d’axes
        plt.tight_layout(pad=0)  # Pas de bordures
        plt.savefig(mask_filename, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        # plt.imsave(mask_filename, mask_image, cmap='gray', origin='lower')
        print(f"Mask image saved as {mask_filename}")




def create_csv(tem_image_paths, output_csv, noOutput=True):
    """
    Function that creates a single CSV file containing the metadata of all the PNG files.
    Usefull for Machine Learning applications.
    Inputs : tem_image_paths (str): path of the PNG files.
            output_csv (str): name of the output CSV file.
    """
    import csv

    # create a datadrame concatenating all the metadata CSV files
    all_metadata = []
    for img_path in Path((tem_image_paths)).iterdir():
        if img_path.is_file() and img_path.suffix == ".png":
            metadata_csv = img_path.with_name(img_path.stem + "_metadata.csv")
            if metadata_csv.exists():
                df_meta = pd.read_csv(metadata_csv)
                all_metadata.append(df_meta)
            else:
                if not noOutput:
                    print(f"Metadata CSV not found for {img_path}")
    
    if all_metadata:
        combined_df = pd.concat(all_metadata, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        if not noOutput:
            print(f" CSV containing the metadata created : {output_csv}")
    else:
        if not noOutput:
            print(f"No metadata files found in {tem_image_paths}")

    # save the csv
    if all_metadata:
        combined_df.to_csv(output_csv, index=False)

    return combined_df



