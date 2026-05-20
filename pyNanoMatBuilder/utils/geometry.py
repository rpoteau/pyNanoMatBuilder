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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyNanoMatBuilder import data
from .core import (pyNMB_location, get_resource_path, timer, RAB, Rbetween2Points,
                   vector, vectorBetween2Points, coord2xyz, vertex, vertexScaled, RadiusSphereAfterV,
                   centerOfGravity, center2cog, normOfV, normV, centerToVertices, Rx, Ry, Rz,
                   EulerRotationMatrix, plotPalette, rgb2hex, clone, deleteElementsOfAList,
                   planeFittingLSF, faces_to_planes, AngleBetweenVV, signedAngleBetweenVV
                   )
from .core import centertxt, centerTitle, fg, bg, hl, color
from .crystals import lattice_cart, convertuvwh2hkld
from .core import round_to_Miller
from .core import kDTreeCN
######################################## Fill edges and facets

def MakeFaceCoord(Rnn,f,coord,nAtomsOnFaces,coordFaceAt):
    """
    Interpolates atom positions on a given face of a polyhedron by distributing atoms 
    between two relevant edges.

    Args:
        Rnn (float): Nearest neighbor distance.
        f (list): List of vertex indices defining the face.
        coord (np.ndarray): Array containing the coordinates of all atoms.
        nAtomsOnFaces (int): Counter for the number of atoms placed on faces.
        coordFaceAt (list): List of face atom coordinates to be updated.

    Returns:
        tuple: 
            - nAtomsOnFaces (int): Updated count of face atoms.
            - coordFaceAt (list): Updated list of coordinates of atoms placed on faces.

    Method:
        1. Determines two relevant edges based on the number of vertices in the face.
        2. Interpolates atoms along these edges.
        3. Fills the face by interpolating between interpolated edge atoms.
    """
    # the idea here is to interpolate between edge atoms of two relevant edges
    # (for example two opposite edges of a squared face)
    # be careful of the vectors orientation of the edges!
    if (len(f) == 3):  #triangular facet
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    elif (len(f) == 4):  #square facet 0-1-2-3-4-0
        edge1 = [f[3],f[0]]
        edge2 = [f[2],f[1]]
    elif (len(f) == 5):  #pentagonal facet #not working
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    elif (len(f) == 6):  #hexagonal facet #not working
        edge1 = [f[0],f[1]]
        edge2 = [f[5],f[4]]
    else:
        raise ValueError("Face type not supported (only 3, 4, 5, or 6 vertices).")
        
    # Determine the number of atoms along the edges
    nAtomsOnEdges = int((RAB(coord,f[1],f[0])+1e-6)/Rnn) - 1
    nIntervalsE = nAtomsOnEdges + 1

    # Interpolate atoms along the edges
    for n in range(nAtomsOnEdges):
        CoordAtomOnEdge1 = coord[edge1[0]]+vector(coord,edge1[0],edge1[1])*(n+1) / nIntervalsE
        CoordAtomOnEdge2 = coord[edge2[0]]+vector(coord,edge2[0],edge2[1])*(n+1) / nIntervalsE

        # Compute distance and interpolate atoms between edge atoms
        distBetween2EdgeAtoms = Rbetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)
        nAtomsBetweenEdges = int((distBetween2EdgeAtoms+1e-6)/Rnn) - 1
        nIntervalsF = nAtomsBetweenEdges + 1
        for m in range(nAtomsBetweenEdges):
            coordFaceAt.append(CoordAtomOnEdge1 + vectorBetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)*(m+1) / nIntervalsF)
            nAtomsOnFaces += 1
    return nAtomsOnFaces,coordFaceAt

###############################################################################################
def reduceHullFacets(self,
                     noOutput: bool=False,
                     tolAngle: float=2.0,
                     useWulff: bool=False,
                    ):
    """
    Reduce crystal facets based on convex hull and coplanarit of facets.

    Args:
        noOutput (bool): If True, suppresses output to the console. Default is False.
        tolAngle (float): Tolerance angle to define coplanarity. Default is 2.0 degree.
        useWulff (bool): If True, uses the Wulff construction planes (trPlanes_Wulff)
                         regardless of whether the structure is optimized or not.
                         If False, uses trPlanes_opt for optimized structures,
                         or trPlanes for initial structures. Default is True.     
    Returns:
        tuple: The vertices and reduced faces.

    Note:
        - Previous hull.simplices must have been saved as Crystal.trPlanes.
        - trPlanes_wWlff must have been saved during the Wulff construction.
          The result is always saved in self.trPlanes (or self.trPlanes_opt).
    """

    from scipy.spatial import HalfspaceIntersection
    from scipy.spatial import ConvexHull
    import scipy as sp

    # --- Select source planes ---
    if useWulff and hasattr(self, 'trPlanes_Wulff') and self.trPlanes_Wulff is not None:
        target_planes = np.array(self.trPlanes_Wulff)
        status = "Wulff construction"
    elif self.is_optimized:
        target_planes = np.array(self.trPlanes_opt)
        status = "optimized structure"
    else:
        target_planes = np.array(self.trPlanes)
        status = "initial structure"

    target_cog = self.cog_opt if self.is_optimized else self.cog
        
    if target_planes is None or target_cog is None:
        raise ValueError(f"Missing data for the {status}. Please check the building process or the optimization results.")

    # --- Deduplicate and clean planes before HalfspaceIntersection ---
    if not useWulff:
        if not noOutput:
            print(f"  Planes before deduplication: {len(target_planes)}")
        # Round normals to remove near-duplicate planes from ConvexHull
        planes_rounded = np.round(target_planes[:, :3], decimals=4)
        _, unique_idx = np.unique(planes_rounded, axis=0, return_index=True)
        target_planes = target_planes[np.sort(unique_idx)]
        if not noOutput:
            print(f"  Planes after deduplication: {len(target_planes)}")
        
    feasible_point = np.array(target_cog)
    hs = HalfspaceIntersection(target_planes, feasible_point)
    vertices = hs.intersections
    hull = ConvexHull(vertices)

    faces = hull.simplices
    neighbours = hull.neighbors
    if not noOutput:
        centertxt("Boundaries figure", bgc='#007a7a', size='14', weight='bold')
        centertxt(
            f"Half space intersection of the planes followed by a convex Hull analysis"
            f" -- source: {status}",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
        print("Found:")
        print(f"  - {len(hull.vertices)} convex Hull vertices")
        print(f"  - {len(hull.simplices)} convex Hull simplices before reduction")

    def sortVCW(V, C):
        """
        Sort vertices of a planar polygon clockwise.

        Args:
            V (list): List of vertex indices.
            C (np.ndarray): Array of vertex coordinates.
        
        Returns:
            list: Sorted vertex indices in clockwise order.
        """
        coords = []
        for v in V : coords.append(C[v])
        cog = np.mean(coords, axis=0)
        radialV = coords-cog
        angle = []
        V = list(V)
        normal = planeFittingLSF(np.array(coords), False, False)
        for i in range(len(radialV)):
            angle.append(signedAngleBetweenVV(radialV[0], radialV[i], normal[0:3]))
        ind = np.argsort(angle)
        Vs = np.array(list(V))
        return Vs[ind]
    

    def isCoplanar(p1, p2, tolAngle=tolAngle):
        """Check if two planes p1 and p2 are coplanar."""
        angle = AngleBetweenVV(p1[0:3], p2[0:3])
        return abs(angle) < tolAngle or abs(angle - 180) <= tolAngle

    def reduceFaces(F, coordsVertices):
        """Reduce the number of faces by merging coplanar ones."""
        import networkx as nx
        flatten = lambda l: [item for sublist in l for item in sublist]

        # create a graph in which nodes represent triangles
        # nodes are connected if the corresponding triangles are adjacent and coplanar
        G = nx.Graph()
        G.add_nodes_from(range(len(F)))
        pList = []
        for i, f in enumerate(F):
            planeDef = []
            for v in f:
                planeDef.append(coordsVertices[v])
            planeDef = np.array(planeDef)
            pList.append(planeFittingLSF(planeDef, printErrors=False, printEq=False))

        for i, p1 in enumerate(pList):
            for n in neighbours[i]:
                p2 = pList[n]
                if isCoplanar(p1, p2):
                    G.add_edge(i, n)
        components = list(nx.connected_components(G))
        simplified = [
            set(flatten(F[index] for index in component)) for component in components
        ]
        # pList = []
        # for f in F:
        #     planeDef = coordsVertices[f] # Direct indexing with numpy is faster
        #     pList.append(planeFittingLSF(planeDef, printErrors=False, printEq=False))
        # # Compare every face with every other face (Global check)
        # for i in range(len(pList)):
        #     for j in range(i + 1, len(pList)):
        #         # If they are coplanar, they belong to the same large facet
        #         if isCoplanar(pList[i], pList[j], tolAngle):
        #             G.add_edge(i, j)
    
        # # Group connected triangles into single facets
        # components = list(nx.connected_components(G))
        # simplified = [
        #     set(flatten(F[index] for index in component)) for component in components
        # ]
        return simplified

    new_faces = reduceFaces(faces, vertices)
    new_facesS = []
    for i, nf in enumerate(new_faces):
        new_facesS.append(sortVCW(nf, vertices).tolist())
    if not noOutput:
        print(f"  - {len(new_faces)} facets after reduction")
        print("New trPlanes saved in self.trPlanes")
    trPlanes = []
    for i, f in enumerate(new_faces):
        planeDef = []
        for v in f:
            planeDef.append(vertices[v])
        planeDef = np.array(planeDef)
        trPlanes.append(planeFittingLSF(planeDef, printErrors=False, printEq=False))
    if self.is_optimized:
        self.trPlanes_opt = setdAsNegative(np.array(trPlanes))
    else:
        self.trPlanes = setdAsNegative(np.array(trPlanes))
    return vertices, new_facesS


######################################################################################
def RotationMol(coords, angle, axis="z"):
    """
    Perform a rotation of the molecule's coordinates around a specified axis.

    Args:
        coords (numpy.ndarray): Coordinates of the molecule as a matrix (n x 3).
        angle (float): The angle of rotation in degrees.
        axis (str): The axis around which to perform the rotation ('x', 'y', or 'z').

    Returns:
        numpy.ndarray: R[0] The coordinates of the molecule after rotation, as a matrix (n x 3).
    """
    import math as m
    angler = angle * m.pi / 180
    if axis == 'x':
        R = np.array(Rx(angler) @ coords.transpose())
    elif axis == 'y':
        R = np.array(Ry(angler) @ coords.transpose())
    elif axis == 'z':
        R = np.array(Rz(angler) @ coords.transpose())

    return R


def EulerRotationMol(coords, gamma, beta, alpha, order="zyx"):
    """
    Perform an Euler rotation of the molecule's coordinates.

    Args:
        coords (numpy.ndarray): Coordinates of the molecule as a matrix (n x 3).
        gamma (float): Angle gamma in degrees.
        beta (float):  Angle beta in degrees.
        alpha (float): Angle alpha in degrees.
        order (str): The order of the Euler rotations (default is "zyx").

    Returns:
        numpy.ndarray: The coordinates of the molecule after the Euler rotation, as a matrix (n x 3).
    """
    return np.array(
        EulerRotationMatrix(gamma, beta, alpha, order) @ coords.transpose()
    ).transpose()

def RotationMatrixFromAxisAngle(u, angle):
    """
    Generates a 3x3 rotation matrix from a unit vector representing the axis of rotation and a rotation angle.
    Args:
        u (numpy.ndarray): A unit vector representing the axis of rotation (3 elements).
        angle (float): The angle of rotation in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    a = angle * np.pi / 180
    ux = u[0]
    uy = u[1]
    uz = u[2]
    return np.array(
        [
            [
                np.cos(a) + ux**2 * (1 - np.cos(a)),
                ux * uy * (1 - np.cos(a)) - uz * np.sin(a),
                ux * uz * (1 - np.cos(a)) + uy * np.sin(a)
            ],
            [
                uy * ux * (1 - np.cos(a)) + uz * np.sin(a),
                np.cos(a) + uy**2 * (1 - np.cos(a)),
                uy * uz * (1 - np.cos(a)) - ux * np.sin(a)
            ],
            [
                uz * ux * (1 - np.cos(a)) - uy * np.sin(a),
                uz * uy * (1 - np.cos(a)) + ux * np.sin(a),
                np.cos(a) + uz**2 * (1 - np.cos(a))
            ]
        ]
    )

def rotationMolAroundAxis(coords, angle, axis):
    """
    Return coordinates after rotation by a given angle around an [u,v,w] axis.

    Args:
        coords: natoms x 3 numpy array.
        angle: Angle of rotation.
        axis: Directions given under the form [u,v,w].

    Returns:
        numpy.ndarray: Rotated coordinates.
    """
    normalizedAxis = normV(axis)
    return np.array(
        RotationMatrixFromAxisAngle(normalizedAxis, angle) @ coords.transpose()
    ).transpose()

def rotation_around_axis_through_point(coords, angle_deg, axis, center):
    """
    Rotate coords (n,3) around a given axis (3,) passing through a point `center` (3,)
    by a given angle (in degrees).
    """
    # Mise en radians
    angle_rad = math.radians(angle_deg)
    
    # Normalise l'axe
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    # Translation pour mettre le point 'center' à l'origine
    shifted = coords - center

    # Matrice de rotation (formule de Rodrigues)
    ux, uy, uz = axis
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta),
         ux * uy * (1 - cos_theta) - uz * sin_theta,
         ux * uz * (1 - cos_theta) + uy * sin_theta],
        
        [uy * ux * (1 - cos_theta) + uz * sin_theta,
         cos_theta + uy**2 * (1 - cos_theta),
         uy * uz * (1 - cos_theta) - ux * sin_theta],
        
        [uz * ux * (1 - cos_theta) - uy * sin_theta,
         uz * uy * (1 - cos_theta) + ux * sin_theta,
         cos_theta + uz**2 * (1 - cos_theta)]
    ])

    # Appliquer la rotation
    rotated_shifted = shifted @ R.T

    # Replacer autour de 'center'
    rotated = rotated_shifted + center

    return rotated

######################################## symmetry
# def reflection(plane, points, dontDoItForAtomsThatLieInTheReflectionPlane=True):
#     """
#     Apply a mirror-image symmetry operation w.r.t. a plane of symmetry.

#     Args:
#         plane: [u,v,w,d] parameters that define a plane.
#         points: (N, 3) array of points.
#         dontDoItForAtomsThatLieInTheReflectionPlane: Self-explanatory.

#     Returns:
#         np.ndarray: (N, 3) array of mirror-image points.
#     """
#     pr = []
#     eps = 1.e-4
#     for p in points:
#         vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane, p)
#         if (
#             (dontDoItForAtomsThatLieInTheReflectionPlane and dp2plane >= eps) or
#             not dontDoItForAtomsThatLieInTheReflectionPlane
#         ):
#             ptmp = p + 2 * vp2plane
#             pr.append(ptmp)
#     return np.array(pr)

def reflection(plane,points,doItForAtomsThatLieInTheReflectionPlane=False,eps=1e-2):
    '''
    Apply a mirror-image symmetry operation to an array of points.

    Calculates the reflection of each point across a symmetry plane defined by
    the general equation $ax + by + cz + d = 0$.

    Args:
        plane (list or numpy.ndarray): The four parameters $[a, b, c, d]$ 
            that define the reflection plane equation.
        points (numpy.ndarray): An (N, 3) array of Cartesian coordinates 
            representing the points to be reflected.
        doItForAtomsThatLieInTheReflectionPlane (bool, optional): If True, points located exactly 
            on the reflection plane +- eps are processed. If False, they are 
            skipped. Defaults to False.
        eps (float, default: 1e-2): threshold associated to doItForAtomsThatLieInTheReflectionPlane

    Returns:
        numpy.ndarray: An (N, 3) array containing the coordinates of the 
        reflected mirror-image points.
    '''
    import numpy as np
    pr = []
    # print(f"inside reflection. {doItForAtomsThatLieInTheReflectionPlane}")
    for p in points:
        vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane,p)
        is_on_plane = dp2plane < eps
        if is_on_plane and not doItForAtomsThatLieInTheReflectionPlane:
            # print(dp2plane, vp2plane, p)
            continue
        ptmp = p+2*vp2plane
        pr.append(ptmp)
    return np.array(pr)
    
def reflection_with_face_update(plane, points, faces, 
                                current_indices, 
                                eps=1e-2, debug=False):
    """
    Apply a reflection to a set of points across a plane while mapping indices.

    This function identifies points lying on the reflection plane (shared atoms) 
    and those to be reflected (new atoms). Instead of duplicating shared atoms, 
    it maps their local indices to their existing global indices in the assembly. 
    New atoms are marked with a "NEW" placeholder to be indexed in the global 
    coordinate array later.

    Args:
        plane (np.ndarray): Plane equation [a, b, c, d] for ax + by + cz + d = 0.
        points (np.ndarray): (N, 3) Cartesian coordinates of the current unit.
        faces (list of tuples): Local face definitions (vertex indices).
        current_indices (list of int): Global indices in `all_coords` for 
            each point in `points`.
        eps (float): Distance threshold to consider a point as lying on the 
            plane. Defaults to 1e-2.
        debug (bool): If True, prints mapping information. Defaults to False.

    Returns:
        tuple:
            - reflected_points (np.ndarray): Coordinates of the truly new atoms.
            - old_to_new (dict): Dictionary mapping local indices (0 to N-1) 
              to either a global integer index or the string "NEW".
    """
    reflected_points = []
    old_to_new = {}

    for i, p in enumerate(points):
        vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane, p)
        is_on_plane = dp2plane < eps

        if is_on_plane:
            # L'atome existe déjà, on récupère son index global via current_indices
            old_to_new[i] = current_indices[i]
        else:
            # Nouveau sommet : on marque pour le mapping global plus tard
            old_to_new[i] = "NEW" 
            reflected_points.append(p + 2 * vp2plane)

    return np.array(reflected_points), old_to_new

def helical_assembly(seed_coords, seed_faces, n_units,
                     face_sequence=None, eps=1e-2, debug=False):
    """
    Generate a helical assembly of polyhedra via successive face reflections.

    This engine builds complex structures (like the Boerdijk-Coxeter helix) 
    by reflecting a seed unit across its faces. It prevents atomic duplication 
    by tracking global indices and only adding "new" vertices to the 
    coordinate array at each step.

    Args:
        seed_coords (np.ndarray): (N, 3) coordinates of the initial polyhedral 
            unit (e.g., a tetrahedron).
        seed_faces (list of tuples): Vertex index tuples defining the faces 
            of the seed unit.
        n_units (int): Total number of polyhedral units to assemble.
        face_sequence (list of int, optional): Sequence of face indices to 
            cycle through for reflections. Defaults to range(len(seed_faces)).
        eps (float): Precision threshold for detecting atoms on the reflection 
            plane. Defaults to 1e-2.
        debug (bool): If True, prints step-by-step assembly logs and updated 
            global face indices. Defaults to False.

    Returns:
        tuple:
            - all_coords (np.ndarray): (M, 3) array of all distinct atomic 
              coordinates in the final helix.
            - n_atoms (int): Total number of unique atoms (M).
    
    Notes:
        The function maintains a 'current_global_indices' list to ensure 
        topological continuity between successive units without using 
        nearest-neighbor searches (like KDTree).
    """
    
    if face_sequence is None:
        face_sequence = list(range(len(seed_faces)))
    n_seq = len(face_sequence)

    all_coords = np.array(seed_coords)
    current_coords = np.array(seed_coords)
    current_faces = list(seed_faces) # Faces locales au tétraèdre courant

    all_faces = [f for f in seed_faces]
    current_global_indices = list(range(len(seed_coords)))

    for i in range(1, n_units):
        face_index = face_sequence[i % n_seq]

        # Calcul du plan sur les coordonnées actuelles
        reflection_plane = faces_to_planes(
            [current_faces[face_index]], current_coords)[0]

        # APPEL CORRIGÉ : on passe current_global_indices
        new_coords, old_to_new = reflection_with_face_update(
            reflection_plane,
            current_coords,
            current_faces,
            current_indices=current_global_indices,
            eps=eps, debug=debug
        )
        
        # 1. Calcul du mapping global pour ce tour
        first_new_idx = len(all_coords)
        temp_mapping = {}
        new_count = 0
        for old_idx, val in old_to_new.items():
            if val == "NEW":
                temp_mapping[old_idx] = first_new_idx + new_count
                new_count += 1
            else:
                temp_mapping[old_idx] = val
        
        # 2. Ajout des nouveaux atomes à l'hélice
        if len(new_coords) > 0:
            all_coords = np.vstack([all_coords, new_coords])
        
        # 3. Mise à jour pour le tour suivant
        # On reconstruit les coordonnées du prochain tétraèdre (les 4 points)
        current_coords = all_coords[[temp_mapping[k] for k in range(len(current_coords))]]
        # Les faces restent les mêmes (topologie locale), mais elles pointeront 
        # vers les bons points grâce au mapping au début du prochain tour
        current_global_indices = [temp_mapping[k] for k in range(len(current_coords))]

        if debug:
            this_step_faces = [tuple(temp_mapping[idx] for idx in f) for f in current_faces]
            print(f"Step {i}: face_index={face_index}, new faces {this_step_faces}")
            print(f"Added {len(new_coords)} atoms, total so far: {len(all_coords)}")

    n_atoms = len(all_coords)
    return all_coords, n_atoms

def reflection_tetra(plane,points):
    """
    Simplified reflection function for the helix of tetrahedrons.

    Args:
        plane: [u,v,w,d] parameters that define a plane.
        points: (N, 3) array of points.

    Returns:
        np.ndarray: (N, 3) array of reflected points.
    """

    points = np.asarray(points)
    vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane, points)
    return points + 2 * vp2plane

#######################################################################
def remove_duplicate_atoms(coordinates, reference_coords, tolerance):
    """
    Robustly removes duplicate atoms from coordinates using adaptive tolerance.
    
    Identifies atoms in 'coordinates' that are within 'tolerance' of any atom
    in 'reference_coords' and removes them. This is ideal for removing atoms
    added by reflection that already exist on shared faces.
    
    Args:
        coordinates (np.ndarray): (N, 3) array of new atoms to filter
        reference_coords (np.ndarray): (M, 3) array of existing atoms (reference)
        tolerance (float): Distance threshold for duplicate detection.
                          Recommended: 0.1 * Rnn for molecular structures.
    
    Returns:
        tuple: (unique_coords, n_removed)
            - unique_coords: (K, 3) filtered coordinates
            - n_removed: (int) number of duplicates removed
    
    Notes:
        - Atoms from 'coordinates' that are far from all atoms in 'reference_coords'
          are considered unique and kept.
        - Efficiency: O(N*M) where N=len(coordinates), M=len(reference_coords)
        - Use this when adding incrementally to existing structure (helix generation)
    
    Example:
        >>> new_atoms = np.array([[0.5, 0, 0], [5, 0, 0]])
        >>> existing = np.array([[0, 0, 0], [1, 0, 0]])
        >>> tol = 0.1
        >>> unique, n_dup = remove_duplicate_atoms(new_atoms, existing, tol)
        >>> print(n_dup)  # 1 (first atom is duplicate)
    """
    coordinates = np.asarray(coordinates, dtype=float)
    reference_coords = np.asarray(reference_coords, dtype=float)
    tolerance = float(tolerance)
    
    if len(coordinates) == 0:
        return coordinates, 0
    
    if len(reference_coords) == 0:
        return coordinates, 0
    
    # Compute distances from each new atom to closest reference atom
    # Using broadcasting: (N, 3) - (1, 3) → (N, 3), then norm → (N,)
    distances_matrix = np.linalg.norm(
        coordinates[:, np.newaxis, :] - reference_coords[np.newaxis, :, :],
        axis=2
    )  # Shape: (N, M)
    
    min_distances = np.min(distances_matrix, axis=1)  # Shape: (N,)
    
    # Keep atoms that are far enough from all reference atoms
    mask_unique = min_distances > tolerance
    unique_coords = coordinates[mask_unique]
    n_removed = np.sum(~mask_unique)
    
    return unique_coords, n_removed
    
def planeRotation(Crystal,
                  refPlane,
                  rotAxis,
                  nRot=6,
                  returnMiller: bool=False,
                  debug: bool=False,
                  noOutput: bool=False
                 ):
    """
    Return planes obtained by rotating a reference plane around an axis,
    expressed in Cartesian coordinates.

    Args:
        Crystal: pyNanoMatBuilder object.
        refPlane (array-like): Reference plane in Miller indices [h, k, l]
            to be rotated.
        rotAxis (array-like): Rotation axis in crystallographic coordinates
            [u, v, w].
        nRot (int): Number of rotations. The rotation angle is 360°/nRot.
            Default is 6.
        returnMiller (bool): If True, also returns the rotated planes
            expressed as Miller indices in the crystallographic basis.
            Default is False.
        debug (bool): If True, prints normalized plane normals in both
            Cartesian and Miller index representations. Default is False.
        noOutput (bool): If True, suppresses all output. Default is False.

    Returns:
        np.ndarray: Array of shape (nRot, 3) containing the rotated plane
            normals in Cartesian coordinates.
        np.ndarray (optional): Array of shape (nRot, 3) containing the
            rotated plane normals as Miller indices in the crystallographic
            basis. Only returned if returnMiller=True.
    """
    pRef = np.array([refPlane])
    aRot = np.array([rotAxis])
    msg = (
        f"Projection of the ({pRef[0][0]: .2f} {pRef[0][1]: .2f} {pRef[0][2]: .2f}) "
        f"reference truncation plane around the [{rotAxis[0]: .2f}  {rotAxis[1]: .2f}  "
        f"{rotAxis[2]: .2f}] axis, after projection in the cartesian coordinate system"
    )
    if not noOutput:
        centertxt(msg, bgc='#cbcbcb', size='12', fgc='b', weight='bold')
    pRefCart = lattice_cart(Crystal, pRef, True, printV=not noOutput)
    rotAxisCart = lattice_cart(Crystal, aRot, True, printV=not noOutput)
    msg = (
        f"{nRot}th order rotation around {rotAxisCart[0][0]: .2f} "
        f"{rotAxisCart[0][1]: .2f} {rotAxisCart[0][2]: .2f}"
        f"of the ({pRefCart[0][0]: .2f} {pRefCart[0][1]: .2f} "
        f"{pRefCart[0][2]: .2f}) truncation plane"
    )
    if not noOutput:
        centertxt(msg, bgc='#cbcbcb', size='12', fgc='b', weight='bold')
    planesCart = []
    for i in range(0, nRot):
        angle = i * 360 / nRot
        # print("rot around z    = ",RotationMol(pRefCart[0],angle,'z'))
        x = rotationMolAroundAxis(pRefCart[0], angle, rotAxisCart[0])
        # print("rot around axis = ",x)
        planesCart.append(x)
    if debug:
        print(np.array(planesCart))
    if not noOutput:
        centertxt(
            f"Just for your knowledge: indexes of the {nRot} normal directions to the "
            f"truncation planes after projection to the {Crystal.cif.cell.get_bravais_lattice()} unitcell",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    Miller_indexes = lattice_cart(Crystal, np.array(planesCart), False, printV=not noOutput)
    Miller_indexes, is_integer = round_to_Miller(Miller_indexes)
    
    if not noOutput:
        centertxt("Miller indices of rotated planes",
                  bgc='#cbcbcb', size='12', fgc='b', weight='bold')
        for i, p in enumerate(Miller_indexes):
            print(f"  plane {i}: ({p[0]} {p[1]} {p[2]})")
    if debug:
        centertxt("Normalized Miller indexes", bgc='#cbcbcb', size='12', fgc='b', weight='bold')
        for i, p in enumerate(Miller_indexes):
            print(i, normV(p))
        print()
        centertxt(
            "Normalized cartesian planes", bgc='#cbcbcb', size='12', fgc='b', weight='bold'
        )
    if returnMiller:
        return np.array(planesCart), Miller_indexes
    else:
        return np.array(planesCart)

def alignV1WithV2_returnR(v1,v2=np.array([0, 0, 1])):
    """
    Return the rotation matrix that aligns v1 with v2.

    Args:
        v1 (np.ndarray): Source vector.
        v2 (np.ndarray): Target vector.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    from scipy.spatial.transform import Rotation
    v1 = np.reshape(v1, (1, -1))
    v2 = np.reshape(v2, (1, -1))
    rMat = Rotation.align_vectors(v2, v1)
    rMat = rMat[0].as_matrix()
    v1_rot = rMat @ v1[0]
    aligned = np.allclose(v1_rot / np.linalg.norm(v1_rot), v2 / np.linalg.norm(v2))
    if not aligned:
        sys.exit(f"Was unable to align {v1} with {v2}. Check your data")
    return rMat

def rotateMoltoAlignItWithAxis(coords, axis, targetAxis=np.array([0, 0, 1])):
    """
    Return coordinates after rotation aligning axis with targetAxis.

    Args:
        coords (np.ndarray): (n_atoms, 3) array of coordinates.
        axis (np.ndarray): Axis direction [u, v, w].
        targetAxis (np.ndarray): Target axis direction.

    Returns:
        np.ndarray: Rotated coordinates (n_atoms, 3).
    """
    if isinstance(axis, list):
        axis = np.array(axis)
    if isinstance(targetAxis, list):
        targetAxis = np.array(targetAxis)
    rMat = alignV1WithV2_returnR(axis, targetAxis)
    return np.array(rMat @ coords.transpose()).transpose()

def setdAsNegative(planes):
    """
    Flip plane signs so that d is negative.

    Args:
        planes (np.ndarray): Array of planes.

    Returns:
        np.ndarray: Updated planes.
    """
    for i,p in enumerate(planes):
        if p[3] > 0:
            p = -p
            planes[i] = p
    return planes

def returnPlaneParallel2Line(V, shift=[1,0,0], debug = False):
    """
    Return plane parameters for a plane parallel to a direction vector.

    Args:
        V (np.ndarray): Direction vector.
        shift (list): Shift vector used to build an arbitrary non-parallel vector.
        debug (bool): If True, prints intermediate values.

    Returns:
        np.ndarray: Plane normal [a, b, c]; d must be found separately.

    Method:
    
        - choose any arbitrary vector not parallel to V[i,j,k] such as V[i+1,j,k]
        - calculate the vector perpendicular to both of these, i.e. the cross product
        - this is the normal to the plane, i.e. you directly obtain the equation of the plane ax+by+cz+d = 0, d being indeterminate
          (to find d, it would be necessary to provide an (x0,y0,z0) point that does not belong to the line, hence d = -ax0-by0-cz0)
        
    """
    arbV = np.array(V.copy())
    arbV = arbV + np.array(shift)
    plane = np.cross(V, arbV)
    if areDirectionsParallel(V, arbV):
        sys.exit(
            f"Error in returnPlaneParallel2Line(): plane {V} is parallel to {arbV}. "
            f"Are you sure of your data?\n(this function wants to return an equation for a "
            f"plane parallel to the direction V = {V}.\n"
            f" Play with the shift variable - current problematic value = {shift})"
        )
    if debug:
        print(areDirectionsParallel(V, arbV), V, arbV, "cross product = ", plane)
    return plane

def normal2MillerPlane(Crystal,MillerIndexes,printN=True):
    """
    Return the normal direction to the plane defined by h,k,l Miller indices
    defined as [n1 n2 n3] = (hkl) x G*, where G* is the reciprocal metric tensor (G* = G-1).
    convertuvwh2hkld() function applied here converts real plane indexes to integers.

    Args:
        Crystal: Crystal object with G*.
        MillerIndexes (np.ndarray): Miller indices [h, k, l].
        printN (bool): If True, prints the computed normals.

    Returns:
        np.ndarray: Integer-normalized plane normal.
    """
    normal = MillerIndexes @ Crystal.Gstar
    normal = np.append(normal, 0.0)  # convertuvwh2hkld expects (u v w h)
    normalI = convertuvwh2hkld(normal, False)[0:3]
    if printN:
        print(
            f"Normal to the ({MillerIndexes[0]:2} {MillerIndexes[1]:2} "
            f"{MillerIndexes[2]:2}) user-defined plane > "
            f"[{normal[0]: .3e} {normal[1]: .3e} {normal[2]: .3e}]"
            f" = [{normalI[0]: .2f} {normalI[1]: .2f} {normalI[2]: .2f}]"
        )
    return normalI

def isPlaneParrallel2Line(v1,v2,tol=1e-5):
    """
    Return True if line direction and plane normal are parallel.
    A line direction vector and a plane are parallel if $|angle|$
    between the line and the normal vector of the plane is 90°.

    Args:
        v1 (np.ndarray): Line direction vector. 
        v2 (np.ndarray): Plane normal vector.
        tol (float): Tolerance for angle comparison in degrees.

    Returns:
        bool: True if line and plane are parallel, False otherwise.
    """
    return (
        np.abs(np.abs(AngleBetweenVV(v1, v2)) - 90) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 270) < tol
    )

def isPlaneOrthogonal2Line(v1, v2, tol=1e-5):
    """
    Return True if line direction and plane normal are orthogonal.
    A line direction vector and a plane are orthogonal if $|angle|$ 
    between the line and the normal vector of the plane is 0° or 180°.

    Args:
        v1 (np.ndarray): Line direction vector.
        v2 (np.ndarray): Plane normal vector.
        tol (float): Tolerance for angle comparison in degrees.

    Returns:
        bool: True if line and plane are orthogonal, False otherwise.
    """
    return (
        np.abs(AngleBetweenVV(v1, v2)) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 180) < tol
    )

def areDirectionsOrthogonal(v1, v2, tol=1e-6):
    """
    Return True if directions are orthogonal.
    Lines are orthogonal if the $|angle|$ between their direction vector is 90°

    Args:   
        v1 (np.ndarray): First direction vector.
        v2 (np.ndarray): Second direction vector.
        tol (float): Tolerance for angle comparison in degrees.
    
    Returns:
        bool: True if directions are orthogonal, False otherwise.
    """
    return (
        np.abs(np.abs(AngleBetweenVV(v1, v2)) - 90) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 270) < tol
    )

def areDirectionsParallel(v1, v2, tol=1e-6):
    """
    Return True if directions are parallel.
    Lines are parallel if the $|angle|$ between their direction vector is 0° or 180°.

    Args:
        v1 (np.ndarray): First direction vector.
        v2 (np.ndarray): Second direction vector.
        tol (float): Tolerance for angle comparison in degrees. 
    
    Returns:
        bool: True if directions are parallel, False otherwise.
    """
    return (
        np.abs(AngleBetweenVV(v1, v2)) < tol
        or np.abs(np.abs(AngleBetweenVV(v1, v2)) - 180) < tol
    )

def shortestPoint2PlaneVectorDistance(plane:np.ndarray,
                                      point:np.ndarray):
    """
    Return the shortest distance, d, from a point X0 to a plane p (projection of X0 on p = P), as well as the PX0 vector.

    Args:
        plane (np.ndarray): [u v w h] definition of the p plane 
        point (np.ndarray): [x0 y0 z0] coordinates of the X0 point or (N, 3) array of points.

    Returns:
        v,d (tuple): the PX0 vector and ||PX0||
    """
    point = np.asarray(point)
    norm_squared = np.sum(plane[0:3]**2)
    if point.ndim == 1:

        t = (plane[3] + np.dot(plane[0:3], point)) / norm_squared
        v = -t * plane[0:3]
        d = np.linalg.norm(v)
    
    else:
        # Multiple points (N, 3)
        t = (plane[3] + point @ plane[0:3]) / norm_squared
        v = -t[:, np.newaxis] * plane[0:3]
        d = np.linalg.norm(v, axis=1)

    return v, d


def Pt2planeSignedDistance(plane,point):
    """
    Return the orthogonal distance of a given point X0 to the plane p in a metric space (projection of X0 on p = P), 
    with the sign determined by whether or not X0 is in the interior of p with respect to the center of gravity [0 0 0].

    Args:
        plane (np.ndarray): Array [u, v, w, h] defining the plane.
        point (np.ndarray): Array [x0, y0, z0] or (N, 3) array of points.

    Returns:
        np.ndarray: Signed distance(s) to the plane.
    """
    plane = np.asarray(plane)
    point = np.asarray(point)
    plane_norm = np.linalg.norm(plane[0:3])

    if point.ndim == 1:
        # Single point: use dot product
        sd = (plane[3] + np.dot(plane[0:3], point)) / plane_norm
    else:
        # Multiple points (N, 3): use matrix multiply
        sd = (plane[3] + point @ plane[0:3]) / plane_norm

    return sd

def planeAtVertices(coordVertices: np.ndarray,
                    cog: np.ndarray):
    """
    Returns the equation of the plane defined by vectors between the center of gravity (cog) and each vertex of a polyhedron
    and that is located at the vertex.

    Args:
        coordVertices (np.ndarray): Coordinates of the vertices (n_vertices, 3).
        cog (np.ndarray): Center of gravity of the NP.

    Returns:
        np.array(plane): the (cog-nvertices)x3 coordinates of the plane 
    """
    planes = []
    for vx in coordVertices:
        vector = vx - cog
        d = -np.dot(vx, vector)
        vector = np.append(vector, d)
        planes.append(vector)

    # TO BE TESTED: the following vectorized version of the above loop, should be faster
    # coordVertices = np.asarray(coordVertices)
    # cog = np.asarray(cog)
    # vectors = coordVertices - cog
    # d = -np.einsum('ij,ij->i', coordVertices, vectors)  # Compute d for each vertex
    # planes = np.column_stack((vectors, d))

    return np.array(planes)

def planeAtPoint(plane: np.ndarray,
                 P0: np.ndarray):
    """
    Recalculate plane d so the plane passes through P0.

    Args:
        plane (np.ndarray): Array [a, b, c, d].
        P0 (np.ndarray): Coordinates [x0, y0, z0].

    Returns:
        np.ndarray: Plane parameters [a, b, c, -(ax0+by0+cz0)].
    """
    d = np.dot(plane[0:3], P0)
    planeAtP = plane.copy()
    planeAtP[3] = -d
    return planeAtP


def normalizePlane(p):
    """Normalize plane parameters [a, b, c, d] by the normal norm."""
    return p / normOfV(p[0:3])

def point2PlaneDistance(point: np.float64,
                              plane: np.float64):
    """
    Compute the shortest distance between a point and a plane in 3D space.

    Args:
        point (np.ndarray): A 3D point as [x, y, z].
        plane (np.ndarray): A plane defined by [A, B, C, D] where Ax + By + Cz + D = 0.

    Returns:
        float: The shortest distance from the point to the plane.
    """
    from numpy.linalg import norm
    distance = abs(np.dot(point, plane[0:3]) + plane[3]) / norm(plane[0:3])
    return distance

######################################## cut above planes
def calculateTruncationPlanesFromVertices(planes, cutFromVertexAt, nAtomsPerEdge, debug=False, noOutput=False, 
                                          trTd=False):
    """
    Calculate truncation planes from vertex planes.

    Args:
        planes (np.ndarray): Array of planes [u, v, w, d].
        cutFromVertexAt (float): Fraction of edge length to cut from each vertex.
        nAtomsPerEdge (int): Number of atoms per edge.
        debug (bool): If True, prints debug information.
        noOutput (bool): If True, suppresses output.

    Returns:
        np.ndarray: Truncation planes.
    """
    n = int(round(1/cutFromVertexAt))
    if not noOutput:
        print(
            f"factor = {cutFromVertexAt:.3f} ▶ {round(nAtomsPerEdge / n)} layer(s) "
            "will be removed, starting from each vertex"
        )

    trPlanes = []
    # for the truncated tetrahedron, the substracted number of layers may be under estimated for big NPs
    # let's introduce a small eps proportionnal to the size of the tetrahedron
    if trTd:
        eps = 0.1 # without it, the truncated tetrahedron is not enough truncated (one layer is missing)
    else:
        eps = 0 # works with the other shapes


    for p in planes:
        pNormalized = normalizePlane(p.copy())
        pNormalized[3] = pNormalized[3] - pNormalized[3] * (cutFromVertexAt + eps)
        trPlanes.append(pNormalized)
    if debug and not noOutput:
        print("normalized original plane = ", normalizePlane(p))
        print("cut plane = ", pNormalized, "... norm = ", normOfV(pNormalized[0:3]))
        print(
            "signed distance between original plane and origin = ",
            Pt2planeSignedDistance(p, [0, 0, 0]),
        )
        print(
            "signed distance between cut plane and origin = ",
            Pt2planeSignedDistance(pNormalized, [0, 0, 0]),
        )
        print(
            "pcut/pRef = ",
            Pt2planeSignedDistance(pNormalized, [0, 0, 0])
            / Pt2planeSignedDistance(p, [0, 0, 0]),
        )
    if not noOutput:
        print(
            "Will remove atoms just above plane "
            f"{pNormalized[0]:.2f} {pNormalized[1]:.2f} {pNormalized[2]:.2f} "
            f"d:{pNormalized[3]:.3f}"
        )
       
    return np.array(trPlanes)

def truncateAboveEachPlane(planes: np.ndarray,
                           coords,
                           debug: bool=False,
                           delAbove: bool=True,
                           noOutput: bool=False,
                           eps: float = 1e-3):
    """
    Return atom indices above (or below) each plane.

    Args:
        planes (np.ndarray): Array with plane definitions [u, v, w, d].
        coords (np.ndarray): (N, 3) array of coordinates.
        debug (bool): If True, prints selected atom indices.
        delAbove (bool): If True, deletes atoms above planes + eps; below otherwise.
        noOutput (bool): If True, suppresses output.
        eps (float): Atom-to-plane signed distance threshold.
    Returns:
        np.ndarray: Unique indices of atoms above/below any input plane.
    """
    coords = np.asarray(coords)
    planes = np.asarray(planes)

    
    delAtoms = []
    
    for p in planes:
        # Vectorized: compute signed distances for all atoms at once
        signedDistances = Pt2planeSignedDistance(p, coords)
        
        if delAbove:
            atomsToDelete = np.where(signedDistances > eps)[0]
        else:
            atomsToDelete = np.where(signedDistances < eps)[0]
        
        delAtoms.extend(atomsToDelete)
        
        if debug and not noOutput:
            for a in atomsToDelete:
                print(f"@{a+1}", end=',')
            print("", end='\n')
    
    delAtoms = np.unique(np.array(delAtoms))
    return delAtoms

def truncateAbovePlanes(planes: np.ndarray,
                        coords: np.ndarray,
                        allP: bool=False,
                        delAbove: bool = True,
                        debug: bool=False,
                        noOutput: bool=False,
                        eps: float=1e-3,
                        depth_max: float=None):
    """
    Return a boolean mask of atoms above/below plane(s).

    Args:
        planes (np.ndarray): Array with plane definitions [u, v, w, d].
        coords (np.ndarray): (N, 3) array of coordinates.
        allP (bool): If True, deleted atoms must lie simultaneously above ALL individual plane.
        delAbove (bool): If True, delete atoms above planes (+eps); below otherwise
        (use with precaution, could return no atoms as a function of their definition)
        debug (bool): If True, prints atoms matching the conditions.
        noOutput (bool): If True, suppresses output.
        eps (float): Atom-to-plane signed distance threshold.
        depth_max (float): Reserved for depth-based filtering.

    Returns:
        np.ndarray: Boolean mask for atoms above/below planes.
    """
    coords = np.asarray(coords)
    planes = np.asarray(planes)
    
    if not noOutput: centertxt(f"Plane truncation (all planes condition: {allP}, delete above planes: {delAbove}, initial number of atoms = {len(coords)})",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    
    if allP:
        delAtoms = np.ones(len(coords), dtype=bool)
    else:
        delAtoms = np.zeros(len(coords), dtype=bool)
    
    for p in planes:
        # Vectorized: compute signed distances for all atoms at once
        signedDistances = Pt2planeSignedDistance(p, coords)
        
        if delAbove and allP:
            delAtomsP = signedDistances > eps
            delAtoms = delAtoms & delAtomsP
        elif delAbove and not allP:
            delAtomsP = signedDistances > eps
            delAtoms = delAtoms | delAtomsP
        elif not delAbove and allP:
            delAtomsP = signedDistances < -eps
            delAtoms = delAtoms & delAtomsP
        elif not delAbove and not allP:
            delAtomsP = signedDistances < -eps
            delAtoms = delAtoms | delAtomsP
        
        nOfDeletedAtomsP = np.count_nonzero(delAtomsP)
        nOfDeletedAtoms = np.count_nonzero(delAtoms)
        
        if debug and not allP:
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtomsP} atoms deleted")
            for i in np.where(delAtomsP)[0]:
                print(f"@{i+1}",end=',')
            print("",end='\n')
        if debug and allP:
            print("allP is True => deletion of all atoms that simultaneously lie above/below each plane")
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtoms} atoms deleted")
            for i in np.where(delAtoms)[0]:
                print(f"@{i+1}",end=',')
            print("",end='\n')
    
    if not noOutput: 
        if delAbove:
            print(f"{len(coords)-np.count_nonzero(delAtoms)} atoms lie below the plane(s)")
        else:
            print(f"{np.count_nonzero(delAtoms)} atoms lie below the plane(s)")
    return delAtoms

def returnPointsThatLieInPlanes(planes: np.ndarray,
                                coords: np.ndarray,
                                debug: bool=False,
                                threshold: float=1e-3,
                                noOutput: bool=False,
                               ):
    """
    Finds all points (atoms) that lie within the given planes based on a signed distance criterion.

    Args:
        planes (np.ndarray): A 2D array with plane equations [a, b, c, d].
        coords (np.ndarray): A 2D array of atom coordinates [x, y, z].
        debug (bool, optional): If True, prints additional debugging information.
        threshold (float, optional): Tolerance for distance to consider a point in plane.
        noOutput (bool, optional): If True, suppresses output messages.

    Returns:
        AtomsInPlane (np.ndarray): A boolean array where True indicates that the atom lies in one of the planes.
    """
    coords = np.asarray(coords)
    planes = np.asarray(planes)
    
    if not noOutput:
        centertxt(
            "Find all points that lie in the given planes",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    AtomsInPlane = np.zeros(len(coords), dtype=bool)
    
    for p in planes:
        # Vectorized: compute signed distances for all atoms at once
        signedDistances = Pt2planeSignedDistance(p, coords)
        atomsInThisPlane = np.abs(signedDistances) < threshold
        AtomsInPlane = AtomsInPlane | atomsInThisPlane
        
        nOfAtomsInPlane = np.count_nonzero(AtomsInPlane)
        if debug:
            print(
                f"- plane",
                [f"{x: .2f}" for x in p],
                f"> {nOfAtomsInPlane} atoms lie in the planes",
            )
            for i in np.where(atomsInThisPlane)[0]:
                print(f"@{i+1}", end=',')
            print("", end='\n')

    if not noOutput:
        print(f"{np.count_nonzero(AtomsInPlane)} atoms lie in the plane(s)")
    return AtomsInPlane

######################################## Core/surface identification / Convex Hull analysis
def coreSurface(self,
                threshold_CoreSurface: float=1e-3,
                noOutput=False,
               ):
    """
    Identify the core and surface atoms of a nanoparticle.
    This method distinguishes surface atoms from core atoms by calculating the 
    geometric convex hull of the atomic coordinates. It can target either the 
    initial structure (NP) or the relaxed structure (NP_opt).

    Args:
        threshold_CoreSurface (float): The threshold used to identify surface atoms.
        noOutput (bool): If True, suppresses output during the analysis.

    Returns:
        tuple: A tuple containing:
            - list: [Hull vertices, Hull simplices, Hull neighbors, Hull equations]
            - surfaceAtoms (numpy.ndarray): A boolean array where True indicates that the atom lies on the surface.
    """
    from ase.visualize import view
    from scipy.spatial import ConvexHull
    if not noOutput:
        centertxt("Core/Surface analyzis", bgc='#007a7a', size='14', weight='bold')
    if not noOutput:
        chrono = timer()
        chrono.chrono_start()
    if self.is_optimized and self.NP_opt is not None:
        target_atoms = self.NP_opt
        status = "optimized structure"
    else:
        target_atoms = self.NP
        status = "initial structure"

    coords = target_atoms.get_positions()
    if not noOutput:
        centertxt(
            "Convex Hull analyzis",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    hull = ConvexHull(coords)

    if status == "optimized structure":
        self.trPlanes_opt = hull.equations
        current_planes = self.trPlanes_opt
        self.vol_Hull_opt = hull.volume/1000
        self.area_Hull_opt = hull.area/100
    else:
        self.trPlanes = hull.equations
        current_planes = self.trPlanes
        self.vol_Hull = hull.volume/1000
        self.area_Hull = hull.area/100
        
    if not noOutput:
        print("Found:")
        print(f"  - {len(hull.vertices)} vertices")
        print(f"  - {len(hull.simplices)} simplices")
        print(f"  - Volume: {hull.volume/1000:.2f} nm³")
        print(f"  - Area: {hull.area/100:.2f} nm²")        

    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    if not noOutput: chrono = timer(); chrono.chrono_start()
    surfaceAtoms = returnPointsThatLieInPlanes(current_planes, coords, noOutput=noOutput, threshold=threshold_CoreSurface)
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return [hull.vertices, hull.simplices, hull.neighbors, hull.equations],surfaceAtoms

################################################################

def peel_by_coordination(self, threshold_peeling=6, Rmax=2.9, noOutput=False):
    """
    Remove surface atoms with low coordination numbers to simulate truncation or 
    incomplete shell growth.

    This method uses a KDTree-based neighbor search (via self.kDTreeCN) to identify 
    and delete atoms that are weakly bonded, such as vertices or edge atoms.

    The method updates self.NP in place. If an optimized structure (NPopt) 
    existed, it is used as the coordinate source for the peeling, then 
    deleted. self.is_optimized is reset to False.

    Args:
        threshold_peeling (int): Minimum coordination number required to keep an atom. 
        Rmax (float): The cutoff distance (in Angstroms) for defining a 
                      nearest neighbor.
        noOutput (bool): If True, suppresses output messages.

    Returns:
        ase.Atoms: The updated nanoparticle (self.NP) after peeling.
    """
    import numpy as np
    
    # 1. Determine which structure to peel
    if self.is_optimized and hasattr(self, 'NP_opt'):
        target_attr = 'NP_opt'
        status = "optimized structure"
    else:
        target_attr = 'NP'
        status = "initial structure"

    target_atoms = getattr(self, target_attr)
    
    # 2. Retrieve Coordination Numbers (CN)
    # Using your internal KDTree tool
    _, CN = kDTreeCN(target_atoms, Rmax=Rmax, returnD=False, noOutput=noOutput)

    # 3. Identify atoms to keep
    CN = np.array(CN)
    indices_to_keep = np.where(CN >= threshold_peeling)[0]
    
    # 4. Update the structure in place
    old_count = len(target_atoms)
    self.NP = target_atoms[indices_to_keep]
    self.NP.positions -= self.NP.get_center_of_mass()
    self.nAtoms = len(self.NP)
    
    if not noOutput:
        centertxt("Removing surface atoms with low coordination numbers", bgc='#007a7a', size='14', weight='bold')
        print(f"Peeling the {status} (CN < {threshold_peeling}): "
              f"removed {old_count - self.nAtoms} atoms. self.NP updated.")
    
    # Sync Metadata and Clean Stale Data !!!
    self._flush_stale_data(shape_update="_peeled_CN")
    self.is_optimized = False
    self.propPostMake(skipChiralityCalculation=self.skipChiralityCalculation,
                      skipSymmetryAnalyzis=self.skipSymmetryAnalyzis,
                      skipFacetInfo=self.skipFacetInfo,
                      thresholdCoreSurface=self.thresholdCoreSurface,
                      noOutput=False, is_optimized=False)
    

def peel_by_shifted_ellipsoid(self, shift_dist=2.5, noOutput=False):
    """
    Truncate the nanoparticle using a shape-adaptive envelope shifted in a 
    random direction.

    This method simulates asymmetric growth or partial dissolution by shifting 
    a volume of control (defined by the particle's own inertia tensor) and 
    removing atoms that fall outside. By projecting coordinates into the 
    Principal Component Analysis (PCA) local frame, the truncation volume 
    perfectly matches the eccentricity of the NP, making it robust for both 
    spherical and cylindrical (nanorod) geometries.

    The method updates self.NP in place. If an optimized structure (NP_opt) 
    existed, it is used as the coordinate source, then deleted. 
    self.is_optimized is reset to False.

    Args:
        shift_dist (float): The distance to shift the envelope (in Angstroms). 
                            2.5 A corresponds to approximately one atomic layer.
        noOutput (bool): If True, suppresses output messages.
    """
    import numpy as np
    
    # 1. Identify source data using your updated keys
    if self.is_optimized and hasattr(self, 'NP_opt'):
        target_atoms = self.NP_opt
        key = 'optimized structure'
    else:
        target_atoms = self.NP
        key = 'initial structure'
        
    # 2. Re-run ellipsoid analysis to get evecs aligned with surface atoms
    self.get_ellipsoid_analysis(noOutput=True, mode='vertices')
    res = self.ellipsoid[key]
    a, b, c = res['D1']/2, res['D2']/2, res['D3']/2

    # Get evecs from hull vertices — SAME as get_ellipsoid_analysis(mode='vertices')
    if self.is_optimized:
        hull_indices = getattr(self, 'vertices_opt', None)
    else:
        hull_indices = getattr(self, 'vertices', None)
    
    hull_pts = target_atoms.get_positions()[hull_indices]
    center_orig = hull_pts.mean(axis=0)
    pos_c = hull_pts - center_orig
    S = (pos_c.T @ pos_c) / len(pos_c)
    evals, evecs = np.linalg.eigh(S)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    # 3. Define shift
    if shift_dist == 0:
        shift_vec = np.zeros(3)
    else:
        random_vec = np.random.normal(size=3)
        random_vec /= np.linalg.norm(random_vec)
        shift_vec = random_vec * shift_dist
        
    new_center = center_orig + shift_vec

    # 4. Apply ellipsoid in local frame
    pos = target_atoms.get_positions()
    relative_pos = pos - new_center
    local_pos = relative_pos @ evecs

    inside = (local_pos[:,0]**2 / a**2 +
              local_pos[:,1]**2 / b**2 +
              local_pos[:,2]**2 / c**2) <= 1.0 + 1e-6 # small tolerance
    
    # 5. Update and Reset
    old_count = len(target_atoms)
    self.NP = target_atoms[inside]
    self.NP.positions -= self.NP.get_center_of_mass()
    self.nAtoms = len(self.NP)
    
    if not noOutput:
        # Calculating the two ratios (Major/Intermediate and Major/Minor)
        # This reflects the full 3D shape (Cylindrical vs Spheroidal)

        centertxt("Truncating the nanoparticle using a shape-adaptive envelope", bgc='#007a7a', size='14', weight='bold')
        print(f"Shifted Truncation ({key}):")
        print(f"  - Envelope matched to particle length ({a*0.1:.2f} nm) and shape.")
        print(f"  - Aspect Ratios: a/b = {a/b:.2f} ; a/c = {a/c:.2f}")
        print(f"  - Atoms removed: {old_count - self.nAtoms}. self.NP updated.")
    
    # Sync Metadata and Clean Stale Data !!!
    self._flush_stale_data(shape_update="_peeled_ellipsoid")
    self.is_optimized = False
    self.propPostMake(skipChiralityCalculation=self.skipChiralityCalculation,
                      skipSymmetryAnalyzis=self.skipSymmetryAnalyzis,
                      skipFacetInfo=self.skipFacetInfo, 
                      thresholdCoreSurface=self.thresholdCoreSurface,
                      noOutput=False, is_optimized=False)


def plot_npr_triangle(self=None, is_optimized: bool = False, save_path: str = None, 
                      external_data: dict = None, color_by: str = 'Rg', color: str='viridis'):
    """
    Hybrid Sauer Plot for pyNanoMatBuilder with precise colorbar scaling and bold formatting.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # 0. Larger figure size
    fig, ax = plt.subplots(figsize=(14, 12))

    # 1. Draw Triangle Boundaries
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [1, 1, 0.5, 1]
    ax.plot(triangle_x, triangle_y, color='black', linestyle='--', linewidth=1.5, zorder=1)

    # 2. Data Preparation
    if external_data:
        npr_list = np.array(external_data['NPR'])
        rg_list = np.array(external_data['Rg'])
        
        if color_by == 'Rg':
            sc = ax.scatter(npr_list[:,0], npr_list[:,1], c=rg_list, cmap=color,
                            s=120, edgecolors='black', alpha=0.8, zorder=3)
            
            # --- FIX: Match colorbar height to axes height ---
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label('Radius of Gyration $R_g$ / nm', size=13, weight='bold')
            
            # Apply bold/size 12 to colorbar ticks as well
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(12)
                t.set_fontweight('bold')
        else:
            for shape_type in set(external_data['shapes']):
                mask = [s == shape_type for s in external_data['shapes']]
                ax.scatter(npr_list[mask, 0], npr_list[mask, 1], label=shape_type,
                           s=120, edgecolors='black', alpha=0.8, zorder=3)
            ax.legend(prop={'weight': 'bold', 'size': 12}, loc='best')
    else:
        # Plotting the single instance (self)
        npr = self.NPR_opt if is_optimized else self.NPR
        rg = self.Rg_opt if is_optimized else self.Rg
        
        ax.scatter(npr[0], npr[1], color='tab:blue', s=200, edgecolors='black', 
                   linewidth=1.5, zorder=3)
        
        ax.text(npr[0] + 0.03, npr[1] + 0.01, f"$R_g$: {rg:.2f} nm", 
                fontsize=12, fontweight='bold', color='darkblue')

    # 3. Vertices Labels
    ax.text(0, 1.04, 'Rod', fontsize=14, fontweight='bold', ha='center', va='bottom')
    ax.text(1, 1.04, 'Sphere', fontsize=14, fontweight='bold', ha='center', va='bottom')
    ax.text(0.5, 0.45, 'Disk', fontsize=14, fontweight='bold', ha='center', va='top')

    # 4. Styling
    status = "Optimized" if is_optimized else "Unoptimized"
    title = "NP Population Analysis" if external_data else f"{self.shape} Analysis ({status})"
    ax.set_title(title, pad=25, fontsize=16, fontweight='bold')
    ax.set_xlabel('NPR1 ($I_1$ / $I_3$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('NPR2 ($I_2$ / $I_3$)', fontsize=14, fontweight='bold')

    # Formatting Tick Labels: Size 12 and Bold
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
        label.set_fontweight('bold')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.4, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

    # 5. Save Logic
    if save_path:
        if save_path.lower().endswith('.svg'):
            import matplotlib
            matplotlib.rcParams['svg.fonttype'] = 'none'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")

    plt.show()

###################################### chirality
def _applyRotationRodrigues(positions, axis_cart, angles_deg):
    """
    Vectorized rotation of N positions around a single axis, each by its own
    angle, using Rodrigues' rotation formula. No Python loop — all operations
    are performed in numpy/C for maximum performance.

    Args:
        positions (np.ndarray): (N, 3) centered atomic positions.
        axis_cart (np.ndarray): (3,) unit vector of the rotation axis.
        angles_deg (np.ndarray): (N,) rotation angles in degrees, one per atom.

    Returns:
        np.ndarray: (N, 3) rotated positions.

    Note:
        Rodrigues' formula: v' = v·cos(θ) + (k × v)·sin(θ) + k·(k·v)·(1 - cos(θ))
        where k is the unit rotation axis and θ is the rotation angle.
    """
    angles_rad = np.deg2rad(angles_deg)               # (N,)
    cos_a = np.cos(angles_rad)[:, np.newaxis]         # (N, 1)
    sin_a = np.sin(angles_rad)[:, np.newaxis]         # (N, 1)

    k = axis_cart                                     # (3,)
    kdotv = positions @ k                             # (N,)   k·v per atom
    kcrossv = np.cross(positions, k)                  # (N, 3) v × k per atom

    rotated = (positions * cos_a
               - kcrossv * sin_a                      # sign: cross(v,k) = -cross(k,v)
               + k * (kdotv * (1 - cos_a.squeeze()))[:, np.newaxis])
    return rotated

def applyTwist(self,
                 axis: [0,0,1],
                 axis_def: str = 'hkl',
                 rate: float = 1.0,
                 profile: str = 'linear',
                 custom_profile=None,
                 pitch: float = None,
                 helix_radius: float = None,
                 chirality = 'RH',
                 noOutput: bool = False,
                ):
    """
    Apply a Twist to the atomic positions of a NP along a given crystallographic axis.

    Each atomic plane perpendicular to the Twist axis is rotated by an angle
    that depends on its position along the axis, according to the chosen profile.
    The 'helical' profile additionally translates atoms along the axis proportionally
    to their rotation angle, producing a helix-like deformation.
    The 'helix' profile bends the entire wire to follow a circular helical path,
    like a spring.
    Rotations are computed using a fully vectorized implementation of Rodrigues'
    formula — no Python loop over atoms.

    Args:
        self: A pyNMBcore instance (Crystal, or any object with self.NP and self.cog).
        axis (array-like): Twist axis in crystallographic coordinates [h, k, l]
                           or Cartesian [x, y, z] depending on axis_def.
        axis_def (str): Defines the coordinate system of the axis argument. Options:
                        'hkl' — axis is given as Miller indices in the crystallographic
                                 basis; requires a crystal object with a reciprocal metric
                                 tensor (Gstar). Use for Crystal NPs built by pyNanoMatBuilder.
                        'cart' — axis is given as a Cartesian direction [x, y, z]; works
                                 for any NP, including those loaded from file via from_file().
                        Default is 'hkl'.
        rate (float): Twist rate in degrees. Interpretation depends on profile:
            ``linear`` — degrees per Å.
            ``sinusoidal`` — peak amplitude in degrees.
            ``gaussian`` — peak amplitude in degrees.
            ``helical`` — degrees per Å.
            ``helix`` — not used.
            ``custom`` — scaling factor applied to custom_profile(z, L).
            Default is 1.0.
        profile (str): Twist profile along the axis. Options:
            ``linear`` — angle(z) = rate * z.
            ``sinusoidal`` — angle(z) = rate * sin(2π * z / L).
            ``gaussian`` — angle(z) = rate * exp(-z² / 2σ²), σ = L/4.
            ``helical`` — angle(z) = rate * z, plus translation along axis.
            ``helix`` — bends wire to helical path, requires helix_radius and pitch.
            ``custom`` — angle(z) = rate * custom_profile(z, L).
            Default is ``'linear'``.
        custom_profile (callable, optional): A user-defined function f(z, L) -> float
                       where z is the signed distance along the axis in Å, and L is
                       the total length of the NP along the axis in Å.
                       Required when profile='custom'. Example:
                           custom_profile = lambda z, L: np.sin(4 * np.pi * z / L)
        pitch (float, optional): Helix pitch in Å per turn (Å/360°).
                       Required when profile='helical' or 'helix'.
        helix_radius (float, optional): Radius of the helical path in Å.
                       Required when profile='helix'.
        chirality (str): Handedness of the Twist or helix.
            ``'RH'`` — Right-Handed (default): counter-clockwise when viewed
            from the positive axis direction.
            ``'LH'`` — Left-Handed: mirror image of RH, obtained by flipping
            the rotation direction or the helix winding direction.
            Default is ``'RH'``.
        noOutput (bool): If True, suppresses output. Default is False.

    Returns:
        None. Updates self.NP.positions, self.cog, and calls propPostMake.

    Raises:
        ValueError: If profile is unknown.
        ValueError: If custom_profile is None when profile='custom'.
        ValueError: If custom_profile is not callable.
        ValueError: If pitch is None when profile='helical' or 'helix'.
        ValueError: If helix_radius is None when profile='helix'.
        ValueError: If axis_def is unknown.
        ValueError: If axis_def='hkl' and Gstar is not available.

    Note:
        The Twist is applied around the center of mass of the NP.
        The axis is normalized internally.
        For profiles other than 'helix', rotation is performed using a vectorized
        Rodrigues formula for maximum performance (no Python loop over atoms).
        Each atomic slice perpendicular to the axis rotates as a rigid body —
        no intra-slice distortion. Inter-slice bond stretching increases with
        radial distance from the axis.
        For profile='helix', each slice is displaced and reoriented using the
        Frenet-Serret frame of the helix — no internal distortion of each slice.

    Examples:
        # Linear Twist along [0, 0, 1], 2°/Å
        NP.applyTwist(axis=[0, 0, 1], rate=2.0, profile='linear')

        # Sinusoidal Twist with 45° peak amplitude
        NP.applyTwist(axis=[0, 0, 1], rate=45.0, profile='sinusoidal')

        # Wire twisted on itself, 5°/Å, 200 Å/turn
        NP.applyTwist(axis=[0, 0, 1], rate=5.0, profile='helical', pitch=200.0)

        # Wire bent to a helical path, radius 50 Å, pitch 100 Å/turn
        NP.applyTwist(axis=[0, 0, 1], rate=1.0, profile='helix',
                        helix_radius=50.0, pitch=100.0, axis_def='cart')

        # Custom double-period sinusoidal Twist
        NP.applyTwist(axis=[0, 0, 1], rate=45.0, profile='custom',
                        custom_profile=lambda z, L: np.sin(4 * np.pi * z / L))
    """
    from .crystals import lattice_cart, normV

    if chirality not in ["RH", "LH"]:
        raise ValueError(
            f"Invalid chirality '{chirality}'. "
            "Must be either 'RH' (Right-Handed) or 'LH' (Left-Handed)."
        )
    self.chirality = chirality

    chiral_str = "Right-Handed"
    if chirality == "LH":
        chiral_str = "Left-Handed"

    # --- Input validation ---
    valid_profiles = ('linear', 'sinusoidal', 'gaussian', 'helical', 'helix', 'custom')
    if profile not in valid_profiles:
        raise ValueError(f"Unknown Twist profile '{profile}'. "
                         f"Choose from {valid_profiles}.")
    if profile == 'helix' and helix_radius is None:
        raise ValueError("profile='helix' requires a helix_radius value in Å.")
    if profile in ('helix', 'helical') and pitch is None:
        raise ValueError(f"profile='{profile}' requires a pitch value in Å/turn.")
    if profile == 'custom' and custom_profile is None:
        raise ValueError("profile='custom' requires a callable custom_profile(z, L).")
    if custom_profile is not None and not callable(custom_profile):
        raise ValueError("custom_profile must be a callable function f(z, L).")
    if profile == 'helix' and rate != 1.0 and not noOutput:
        print("  Warning: rate is not used for profile='helix' and will be ignored.")

    valid_axis_def = ('hkl', 'cart')
    if axis_def not in valid_axis_def:
        raise ValueError(f"Unknown axis definition '{axis_def}'. "
                         f"Choose from {valid_axis_def}.")
    if axis_def == 'hkl' and not hasattr(self, 'Gstar'):
        raise ValueError("axis_def='hkl' requires a crystal object with a reciprocal "
                         "metric tensor (Gstar). Use axis_def='cart' for non-crystal objects.")

    # --- Convert axis to Cartesian and normalize ---
    if axis_def == 'cart':
        axis_cart = normV(np.array(axis, dtype=float))
    else:  # 'hkl'
        axis_cart = lattice_cart(self, [axis], Bravais2cart=True, printV=not noOutput)[0]
        axis_cart = normV(axis_cart)

    # --- Determine which structure to use ---
    if self.is_optimized and hasattr(self, 'NP_opt'):
        target_attr = 'NP_opt'
        status = "optimized structure"
    else:
        target_attr = 'NP'
        status = "initial structure"

    target_atoms = getattr(self, target_attr)

    # --- Center positions on cog ---
    positions = target_atoms.get_positions() - self.cog

    # --- Project each atom onto the Twist axis (signed distance in Å) ---
    proj = positions @ axis_cart                          # (N,)

    # --- Total length of NP along axis ---
    L = proj.max() - proj.min()
    
    # Inter-slice bond stretching check
    radial = np.linalg.norm(
        positions - proj[:, np.newaxis] * axis_cart, axis=1
    )

    # =========================================================
    # --- Profile 'helix' : bend wire to a helical path ---
    # =========================================================
    if profile == 'helix':
        # Build local orthonormal frame (e1, e2, e3)
        e3 = axis_cart
        arbitrary = np.array([1, 0, 0]) if abs(axis_cart[0]) < 0.9 else np.array([0, 1, 0])
        e1 = normV(arbitrary - np.dot(arbitrary, e3) * e3)
        e2 = np.cross(e3, e1)

        # Apply chirality — flipping e2 mirrors the helix
        if chirality == 'LH':
            e2 = -e2

        # Arc length correction factor
        # ds/dz = sqrt(1 + (2πR/pitch)²)
        # Without correction, inter-plane distances are stretched by this factor
        pitch_factor = pitch / (2 * np.pi)
        stretch = np.sqrt(1 + (helix_radius / pitch_factor)**2)

        # Corrected parameter t along the helix (radians)
        # Dividing by stretch ensures equal arc-length spacing between slices
        t = proj * 2 * np.pi / pitch / stretch                # (N,)

        # Helix center positions for each atom's slice
        # proj is also corrected to match the compressed z-spacing
        helix_centers = (helix_radius * np.cos(t)[:, np.newaxis] * e1 +
                         helix_radius * np.sin(t)[:, np.newaxis] * e2 +
                         (proj / stretch)[:, np.newaxis] * e3)  # (N, 3)

        # Frenet-Serret local frame
        tangent = (-helix_radius * np.sin(t)[:, np.newaxis] * e1 +
                    helix_radius * np.cos(t)[:, np.newaxis] * e2 +
                    pitch_factor * e3)
        tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)

        normal = (-np.cos(t)[:, np.newaxis] * e1 -
                   np.sin(t)[:, np.newaxis] * e2)

        binormal = np.cross(tangent, normal)

        # Transverse displacement in original frame
        transverse = positions - proj[:, np.newaxis] * e3
        tr_e1 = transverse @ e1
        tr_e2 = transverse @ e2

        # Re-express in Frenet-Serret frame
        new_positions = (helix_centers +
                         tr_e1[:, np.newaxis] * normal +
                         tr_e2[:, np.newaxis] * binormal)

        if not noOutput:
            centertxt("Helix bending analysis", bgc='#007a7a', size='14', weight='bold')
            centertxt(
                f"{chiral_str} Bending wire to helical path on the {status}",
                bgc='#cbcbcb', size='12', fgc='b', weight='bold',
            )
            print(f"  Cartesian axis          : {axis_cart}")
            print(f"  Wire length             : {L:.2f} Å")
            print(f"  Helix radius            : {helix_radius:.2f} Å")
            print(f"  Pitch                   : {pitch:.2f} Å/turn")
            print(f"  Arc stretch factor      : {stretch:.3f}  "
                  f"({'warning: significant distortion !' if stretch > 1.5 else 'ok'})")
            print(f"  Inter-plane distortion  : {(stretch-1)*100:.1f} %")
            print(f"  Number of turns         : {L / pitch / stretch:.2f}")
            print(f"  Effective helix length  : {L / stretch:.2f} Å  "
                  f"(compressed from {L:.2f} Å)")

    # =========================================================
    # --- Other profiles : rotation-based Twist ---
    # =========================================================
    else:
        # Compute Twist angle for each atom (degrees)
        if profile == 'linear':
            angles = rate * proj
        elif profile == 'sinusoidal':
            angles = rate * np.sin(2 * np.pi * proj / L)
        elif profile == 'gaussian':
            sigma = L / 4
            angles = rate * np.exp(-proj**2 / (2 * sigma**2))
        elif profile == 'helical':
            angles = rate * proj
        elif profile == 'custom':
            angles = rate * np.array([custom_profile(z, L) for z in proj])

        # Apply chirality — LH flips the rotation direction
        if chirality == 'LH':
            angles = -angles

        R_max = radial.max()
        R_mean = radial.mean()
        proj_sorted = np.sort(proj)
        dz_mean = np.mean(np.diff(proj_sorted[::max(1, len(proj_sorted)//200)]))
        delta_theta_rad = np.deg2rad(rate * dz_mean)
        delta_tang_surface = R_max * delta_theta_rad
        delta_tang_core = R_mean * delta_theta_rad

        if not noOutput:
            centertxt("Twist analysis", bgc='#007a7a', size='14', weight='bold')
            centertxt(
                f"Applying {chiral_str} '{profile}' Twist along axis {axis} on the {status}",
                bgc='#cbcbcb', size='12', fgc='b', weight='bold',
            )
            print(f"  Cartesian axis          : {axis_cart}")
            print(f"  NP length along axis    : {L:.2f} Å")
            print(f"  Max radial distance     : {R_max:.2f} Å")
            print(f"  rate                    : {rate} °/Å")
            print(f"  max angle               : {np.max(np.abs(angles)):.2f}°")
            if profile == 'helical':
                print(f"  pitch                   : {pitch:.2f} Å/turn")
            print(f"  --- Inter-slice bond stretching estimate ---")
            print(f"  Mean slice thickness    : {dz_mean:.2f} Å")
            print(f"  Angular increment Δθ    : {np.rad2deg(delta_theta_rad):.4f}°/slice")
            print(f"  Tangential displacement : {delta_tang_surface:.4f} Å (surface, r={R_max:.1f} Å)")
            print(f"                           {delta_tang_core:.4f} Å (mean core, r={R_mean:.1f} Å)")

        # Apply vectorized Rodrigues rotation
        if profile == 'helical':
            translations = pitch * np.deg2rad(angles) / (2 * np.pi)
            new_positions = (_applyRotationRodrigues(positions, axis_cart, angles)
                             + translations[:, np.newaxis] * axis_cart)
        else:
            new_positions = _applyRotationRodrigues(positions, axis_cart, angles)

    # --- Restore cog offset and update ASE object ---
    new_positions = new_positions + self.cog
    self.NP.set_positions(new_positions)

    # Update center of mass after Twist
    self.cog = self.NP.get_center_of_mass()

    # Invalidate trPlanes — geometry has changed, old planes are no longer valid
    self.trPlanes = None
    self.trPlanes_Wulff = None
    self.trPlanes_opt = None

    # Sync Metadata and Clean Stale Data
    self._flush_stale_data(shape_update="_Twist")
    self.is_optimized = False
    if profile == 'helix':
        # Store helix parameters for later visualization
        self._helix_params = {
            'helix_radius': helix_radius,
            'pitch':        pitch,
            'axis_cart':    axis_cart,
            'L':            L,
            'e1':           e1,
            'e2':           e2,
            'wire_radius':  np.max(radial),
            'cog_helix':    helix_centers.mean(axis=0) + self.cog,
            'proj_min':     proj.min()
        }

        self.jmolCrystalShape = False
        if not noOutput:
            print(f"{bg.DARKREDB}Warning: jmolCrystalShape disabled for profile='helix' — "
                  f"crystal shape analysis is not meaningful for a bent wire. "
                  f"jMolCS and jMolCS_opt will not be available.{bg.OFF}")
            print(f"{bg.LIGHTGREENB}To visualize the helical envelope, use: "
                  f"script = NP.defHelixShapeForJMol(){bg.OFF}")
    self.propPostMake(skipChiralityCalculation=self.skipChiralityCalculation,
                      skipSymmetryAnalyzis=self.skipSymmetryAnalyzis,
                      skipFacetInfo = self.skipFacetInfo,
                      thresholdCoreSurface=self.thresholdCoreSurface,
                      noOutput=noOutput, is_optimized=False)
