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
from .crystals import lattice_cart, convertuvwh2hkld

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

#######################################################################
######################################## Momenta of inertia
def moi(model: Atoms,
        noOutput: bool=False,
       ):
    """
    Get the moments of inertia along the principal axes.

    Notes:
        Units of the moments of inertia are amu.angstrom**2.
        Periodic boundary conditions are ignored.
    """
    if not noOutput:
        centertxt(
            "Moments of inertia", bgc='#007a7a', size='14', weight='bold'
        )
    model.moi = model.get_moments_of_inertia()  # in amu*angstrom**2
    if not noOutput:
        print(
            f"Moments of inertia = {model.moi[0]:.2f} {model.moi[1]:.2f} "
            f"{model.moi[2]:.2f} amu.Å2"
        )
    model.masses = model.get_masses()
    model.M = model.masses.sum()
    model.moiM = model.moi/(model.M)
    moiM=model.moiM
    if not noOutput:
        print(
            f"Moments of inertia / M = {model.moiM[0]:.2f} {model.moiM[1]:.2f} "
            f"{model.moiM[2]:.2f} amu.Å2"
        )
    return moiM
   


#NEW
def get_moments_of_inertia_for_size(self, vectors=False):  # from ASE with mass modification
    """
    Get the moments of inertia along the principal axes with mass normalization.

    Args:
        vectors (bool, optional): If True, returns both eigenvalues and eigenvectors.
            If False, returns only eigenvalues. Defaults to False.

    Returns:
        np.ndarray: Principal moments of inertia (3 values).
        np.ndarray: Principal axes (3x3 matrix) if vectors is True.

    Notes:
        Periodic boundary conditions are ignored.
        Units of the moments of inertia are angstrom**2.
    """
    com = self.get_center_of_mass()
    positions = self.get_positions()
    #number_atoms=len(positions)
    positions -= com  # translate center of mass to origin
    # masses = self.get_masses() # mass normalization is done by setting all masses to 1 in the inertia tensor calculation

    # Initialize elements of the inertial tensor
    I11 = np.sum(positions[:, 1]**2 + positions[:, 2]**2)
    I22 = np.sum(positions[:, 0]**2 + positions[:, 2]**2)
    I33 = np.sum(positions[:, 0]**2 + positions[:, 1]**2)
    I12 = -np.sum(positions[:, 0] * positions[:, 1])
    I13 = -np.sum(positions[:, 0] * positions[:, 2])
    I23 = -np.sum(positions[:, 1] * positions[:, 2])
    Itensor = np.array([[I11, I12, I13],
                        [I12, I22, I23],
                        [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(Itensor)  # valeurs propes de la matrice
    if vectors:
        return evals, evecs.transpose()
    else:
        return evals


def moi_size(model: Atoms,  # normalized moment of inertia with masses=1
             noOutput: bool=False,
            ):
    """
    Get the moments of inertia along the principal axes with mass normalization.

    Notes:
        Units of the moments of inertia are angstrom**2.
    """

    model.moi_size_all = get_moments_of_inertia_for_size(model)
    positions = model.get_positions()
    number_atoms = len(positions)
    model.moi_size = model.moi_size_all/(number_atoms)
    if not noOutput:
        print(
            f"Moments of inertia with mass=1/M = {model.moi_size[0]:.2f} "
            f"{model.moi_size[1]:.2f} {model.moi_size[2]:.2f} Å2"
        )
    return [model.moi_size[0], model.moi_size[1], model.moi_size[2]]
  
###############################################################################################
def reduceHullFacets(Crystal: Atoms,
                     noOutput: bool=False,
                     tolAngle: float=2.0,
                    ):
    """
    Reduce crystal facets based on convex hull and coplanarit of facets.

    Args:
        Crystal (Atoms): The crystal object containing the planes for the facet reduction.
        feasible_point (np.ndarray): A feasible point for half-space intersection. Default is [0, 0, 0].
        tolAngle (float): Tolerance angle to define coplanarity. Default is 2.0.
        noOutput (bool): If True, suppresses output to the console. Default is False.
        
    Returns:
        tuple: The vertices and reduced faces.

    Note:
        Previous hull.simplices must have been saved as Crystal.trPlanes.
    """
    from scipy.spatial import HalfspaceIntersection
    from scipy.spatial import ConvexHull
    import networkx as nx
    import scipy as sp

    cog = Crystal.cog
    feasible_point = cog
    hs = HalfspaceIntersection(Crystal.trPlanes, feasible_point)
    vertices = hs.intersections
    hull = ConvexHull(vertices)

    faces = hull.simplices
    neighbours = hull.neighbors
    if not noOutput:
        centertxt("Boundaries figure", bgc='#007a7a', size='14', weight='bold')
        centertxt(
            "Half space intersection of the planes followed by a convex Hull analyzis",
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
    Crystal.trPlanes = setdAsNegative(np.array(trPlanes))
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

def reflection(plane,points,doItForAtomsThatLieInTheReflectionPlane=False):
    '''
    Apply a mirror-image symmetry operation to an array of points.

    Calculates the reflection of each point across a symmetry plane defined by
    the general equation $ax + by + cz + d = 0$.

    Args:
        points (numpy.ndarray): An (N, 3) array of Cartesian coordinates 
            representing the points to be reflected.
        plane (list or numpy.ndarray): The four parameters $[a, b, c, d]$ 
            that define the reflection plane equation.
        include_plane_atoms (bool, optional): If True, points located exactly 
            on the reflection plane are processed. If False, they are 
            skipped. Defaults to True.

    Returns:
        numpy.ndarray: An (N, 3) array containing the coordinates of the 
        reflected mirror-image points.
    '''
    import numpy as np
    pr = []
    eps = 1.e-4
    for p in points:
        vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane,p)
        if dp2plane >= eps and not doItForAtomsThatLieInTheReflectionPlane: # otherwise the point belongs to the reflection plane
            # print(dp2plane, vp2plane, p)
            ptmp = p+2*vp2plane
            pr.append(ptmp)
        else:
            ptmp = p+2*vp2plane
            pr.append(ptmp)
    return np.array(pr)

def reflection_tetra(plane,points):
    """
    Simplified reflection function for the helix of tetrahedrons.

    Args:
        plane: [u,v,w,d] parameters that define a plane.
        points: (N, 3) array of points.

    Returns:
        np.ndarray: (N, 3) array of reflected points.
    """
    # pr = []
    # for p in points:
    #     vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane,p)
        
    #     ptmp = p+2*vp2plane
    #     pr.append(ptmp)
    # return np.array(pr)
    points = np.asarray(points)
    vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane, points)
    return points + 2 * vp2plane

def Inscribed_circumscribed_spheres(self, noOutput):
    """
    Calculates the diameters of the inscribed and circumscribed spheres for the nanoparticle (NP) based on 
    its positions and the plane equations.

    The circumscribed sphere is the smallest sphere that completely encloses the NP, while the inscribed sphere 
    is the largest sphere that fits within the NP.

    Args:
        noOutput (bool, optional): If set to True, suppresses output during the
            analysis. Default is False.

    Returns:
        None: The function updates the object's attributes with the calculated
            diameters of the spheres.

    Notes:
        The circumscribed sphere radius is calculated as the maximum distance from the center of gravity 
        (COG) to the NP positions, and the inscribed sphere radius is calculated as the minimum distance 
        from the NP positions to the planes (based on Hull equations)
    """
    if self.shape == 'ellipsoid':
        self.radiusInscribedSphere = min(self.sasview_dims)
        self.radiusCircumscribedSphere = max(self.sasview_dims)
    elif self.shape == 'sphere':
        self.radiusInscribedSphere = self.radius
        self.radiusCircumscribedSphere = self.radius
    else:
        distances = np.linalg.norm(self.NP.positions - self.cog, axis=1)
        self.radiusCircumscribedSphere = np.max(distances)
        distances = [
            abs(d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
            for a, b, c, d in self.equations
        ]
        self.radiusInscribedSphere = np.min(distances)

    if not noOutput:
        centertxt(
            "Diameters of the inscribed and circumscribed sphere using the "
            "Hull equations",
            bgc='#007a7a',
            size='14',
            weight='bold'
        )
    if not noOutput:
        print(
            f"diameters of the circumscribed sphere: "
            f"{self.radiusCircumscribedSphere * 2 * 0.1:.2f} nm"
        )
        print(
            f"diameters of the inscribed sphere: "
            f"{self.radiusInscribedSphere * 2 * 0.1:.2f} nm"
        )

    return self.radiusInscribedSphere, self.radiusCircumscribedSphere

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

def planeRotation(Crystal: Atoms,
                  refPlane,
                  rotAxis,
                  nRot=6,
                  debug: bool=False,
                  noOutput: bool=False
                 ):
    """
    Return planes obtained by rotating a reference plane around an axis.

    Args:
        Crystal: Crystal object.
        refPlane: Plane to rotate.
        rotAxis: Rotation axis.
        nRot (int): Rotation count, angle is 360°/nRot.
        debug (bool): If True, prints normalized planes.
        noOutput (bool): If True, suppresses output.

    Returns:
        np.ndarray: Rotated planes in cartesian coordinates.
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
    planesHCP = lattice_cart(Crystal, np.array(planesCart), False, printV=not noOutput)
    if debug:
        centertxt("Normalized HCP planes", bgc='#cbcbcb', size='12', fgc='b', weight='bold')
        for i, p in enumerate(planesHCP):
            print(i, normV(p))
        print()
        centertxt(
            "Normalized cartesian planes", bgc='#cbcbcb', size='12', fgc='b', weight='bold'
        )
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
        np.ndarray: A boolean array where True indicates that the atom lies in one of the planes.
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
def coreSurface(Crystal: Atoms,
                threshold,
                noOutput=False,
               ):
    """
    Identify the core and surface atoms of a crystal using Convex Hull analysis.

    Args:
        Crystal (Atoms): Crystal structure object.
        threshold (float): The threshold used to identify surface atoms.
        noOutput (bool): If True, suppresses output during the analysis.

    Returns:
        tuple: A tuple containing:
            - list: [Hull vertices, Hull simplices, Hull neighbors, Hull equations]
            - surfaceAtoms (numpy.ndarray): The atomic positions of atoms on the surface.
    """
    from ase.visualize import view
    from scipy.spatial import ConvexHull
    if not noOutput:
        centertxt("Core/Surface analyzis", bgc='#007a7a', size='14', weight='bold')
    if not noOutput:
        chrono = timer()
        chrono.chrono_start()
    coords = Crystal.NP.get_positions()
    if not noOutput:
        centertxt(
            "Convex Hull analyzis",
            bgc='#cbcbcb',
            size='12',
            fgc='b',
            weight='bold',
        )
    hull = ConvexHull(coords)
    if not noOutput:
        print("Found:")
        print(f"  - {len(hull.vertices)} vertices")
        print(f"  - {len(hull.simplices)} simplices")
    Crystal.trPlanes = hull.equations
    # print("Crystal.trplanes inside coreSurface")
    # print(Crystal.trPlanes)
    # print(np.unique(Crystal.trPlanes, axis=0, return_counts=True))
    # print(Crystal.trPlanes.shape)
    #_ = defCrystalShapeForJMol(Crystal,noOutput=noOutput)
    # print("Crystal.trplanes inside coreSurface after defCrystalShapeForJMol")
    # print(Crystal.trPlanes)
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    if not noOutput: chrono = timer(); chrono.chrono_start()
    surfaceAtoms = returnPointsThatLieInPlanes(Crystal.trPlanes, coords, noOutput=noOutput,threshold=threshold)
    if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return [hull.vertices, hull.simplices, hull.neighbors, hull.equations],surfaceAtoms


