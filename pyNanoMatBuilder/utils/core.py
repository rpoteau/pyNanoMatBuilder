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

class fg:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    LIGHTGRAY = "\033[37m"
    DARKGRAY = "\033[90m"    
    BLACK = '\033[30m'
    WHITE = "\033[38;5;231m"
    OFF = '\033[0m'
class hl:
    BLINK = "\033[5m"
    blink = "\033[25m" #reset blink
    BOLD = '\033[1m'
    bold = "\033[21m" #reset bold
    UNDERL = '\033[4m'
    underl = "\033[24m" #reset underline
    ITALIC = "\033[3m"
    italic = "\033[23m"
    OFF = '\033[0m'
class bg:
    DARKRED = "\033[38;5;231;48;5;52m"
    DARKREDB = "\033[38;5;231;48;5;52;1m"
    LIGHTRED = "\033[48;5;217m"
    LIGHTREDB = "\033[48;5;217;1m"
    LIGHTYELLOW = "\033[48;5;228m"
    LIGHTYELLOWB = "\033[48;5;228;1m"
    LIGHTGREEN = "\033[48;5;156m"
    LIGHTGREENB = "\033[48;5;156;1m"
    LIGHTBLUE = "\033[48;5;117m"
    LIGHTBLUEB = "\033[48;5;117;1m"
    OFF = "\033[0m"
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    LIGHTGRAY = "\033[37m"
    DARKGRAY = "\033[90m"    
    BLACK = '\033[30m'
    WHITE = "\033[38;5;231m"
    BOLD = '\033[1m'
    OFF = '\033[0m'

def pyNMB_location():
    """
    Returns the absolute path to the root of the installed package.
    """
    # This points to the directory containing __init__.py of pyNanoMatBuilder
    return Path(str(resources.files("pyNanoMatBuilder")))
        
def get_resource_path(sub_folder: str, filename: str) -> str:
    """
    Retrieve the absolute path of a resource file within the package.
    
    This helper uses the modern 'importlib.resources.files' API (Python 3.9+) 
    to ensure compatibility with various installation environments like 
    Google Colab or virtual environments.

    Args:
        sub_folder (str): Folder path relative to the package root 
                          (e.g., 'resources/figs' or 'resources/cif_database').
        filename (str): Name of the file (e.g., 'banner.png').

    Returns:
        str: The absolute string path to the file.
    """
    # Start at the package root
    traversable_path = resources.files("pyNanoMatBuilder")
    
    # Walk down the sub-folders (supports 'a/b/c' notation)
    for folder in sub_folder.split('/'):
        traversable_path = traversable_path / folder
        
    # Target the specific file
    file_path = traversable_path / filename

    if not file_path.exists():
        raise FileNotFoundError(f"❌ Resource not found: {sub_folder}/{filename}")

    # Use as_file to ensure the path is accessible on the physical disk
    with resources.as_file(file_path) as p:
        return str(p)

######################################## time
class timer:
    """
    Timer class to measure elapsed time in seconds and display it in hh:mm:ss ms.
    """

    def __init__(self):
        self._chrono_start = None
        self._chrono_stop = None

    # delay can be timedelta or seconds
    def hdelay_ms(self, delay):
        """
        Converts a delay into a human-readable format: hh:mm:ss ms.

        Args:
            delay: A timedelta object or a float representing a duration in seconds.
        Return:
            A formatted string in hh:mm:ss ms.
        """
        if type(delay) is not datetime.timedelta:
            delay = datetime.timedelta(seconds=delay)
        sec = delay.total_seconds()
        hh = sec // 3600
        mm = (sec // 60) - (hh * 60)
        ss = sec - hh * 3600 - mm * 60
        ms = (sec - int(sec)) * 1000
        return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'

    def chrono_start(self):
        """
        Starts the chrono.
        """
        self._chrono_start = time.time()

    # return delay in seconds or in humain format
    def chrono_stop(self, hdelay=False):
        """
        Stops the chrono and returns the elapsed time.
        """
        if self._chrono_start is None:
            return "00:00:00 000ms" if hdelay else 0.0
            
        self._chrono_stop = time.time()
        sec = self._chrono_stop - self._chrono_start
        if hdelay:
            return self.hdelay_ms(sec)
        return sec

    def chrono_show(self):
        """
        Prints the elapsed time.
        """
        if self._chrono_start is None:
            print(f'{fg.BLUE}Duration : Timer not started{fg.OFF}')
            return
            
        elapsed = time.time() - self._chrono_start
        print(f'{fg.BLUE}Duration : {self.hdelay_ms(elapsed)}{fg.OFF}')

#######################################################################

######################################## Coordinates, vectors, etc
def RAB(coord, a, b):
    """
    Calculate the distance between two atoms by indices.
    
    Args:
        coord (array-like): Array of 3D coordinates.
        a (int): Index of the first atom.
        b (int): Index of the second atom.
    
    Returns:
        float: Distance between atoms a and b.
    """
    return np.linalg.norm(np.asarray(coord[a]) - np.asarray(coord[b]))


def Rbetween2Points(p1, p2):
    """
    Calculate the distance between two points.
    
    Args:
        p1 (array-like): 3D coordinates of the first point.
        p2 (array-like): 3D coordinates of the second point.
    
    Returns:
        float: Distance between p1 and p2.
    """
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))


def vector(coord, a, b):
    """
    Compute the vector from point a to point b given a list of coordinates.
    
    Args:
        coord (array-like): Array of 3D coordinates.
        a (int): Index of the starting point.
        b (int): Index of the ending point.
    
    Returns:
        np.ndarray: Vector from a to b.
    """
    return np.asarray(coord[b]) - np.asarray(coord[a])


def vectorBetween2Points(p1, p2):
    """
    Compute the vector between two 3D points.
    
    Args:
        p1 (array-like): 3D coordinates of the first point.
        p2 (array-like): 3D coordinates of the second point.
    
    Returns:
        np.ndarray: Vector from p1 to p2.
    """
    return np.asarray(p2) - np.asarray(p1)


def coord2xyz(coord):
    """
    Extracts x, y, and z from a list of 3D coordinates.

    Args:
        A list or array of 3D coordinates in the format [[x1, y1, z1], [x2, y2, z2], ...].

    Returns:
        Three NumPy arrays containing the x, y, and z coordinates separately.
    """
    x = np.array(coord)[:, 0]
    y = np.array(coord)[:, 1]
    z = np.array(coord)[:, 2]
    return x, y, z


def vertex(x, y, z, scale):
    """Return vertex coordinates fixed to the unit sphere."""
    length = np.sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x, y, z)]


def vertexScaled(x, y, z, scale):
    """Return vertex coordinates multiplied by the scale factor."""
    return [i * scale for i in (x, y, z)]


def RadiusSphereAfterV(V):
    """
    Computes the radius of a sphere given its volume.

    Args:
        V (float): Volume of the sphere in cubic units.

    Returns:
        float: Radius of the sphere.

    Formula: R = (3V / (4π))^(1/3)
    """
    return (3 * V / (4 * np.pi)) ** (1 / 3)

# def centerOfGravity(c: np.ndarray,
#                     select=None):
#     """
#     Computes the center of gravity (geometric center) of a set of points.

#     Args:
#         c (np.ndarray): An array of shape (N, 3) representing N atomic positions (x, y, z).
#         select (np.ndarray, optional): Indices of selected atoms to include in the calculation.
#                                        If None, all atoms are used.
#     Returns:
#         np.ndarray: A 3-element array representing the center of gravity coordinates (x, y, z).

#     Notes:
#     - The center of gravity is computed as the average of the selected atomic positions.
#     """
#     import numpy as np
#     if select is None:
#         select = np.array((range(len(c))))
#     nselect = len(select)
#     xg = 0
#     yg = 0
#     zg = 0
#     for at in select:
#         xg += c[at][0]
#         yg += c[at][1]
#         zg += c[at][2]
#     cog = [xg/nselect, yg/nselect, zg/nselect]
#     return np.array(cog)

def centerOfGravity(c: np.ndarray, select=None):
    """Compute center of gravity of selected atoms."""
    c = np.asarray(c)
    if select is None:
        select = np.arange(len(c))
    return np.mean(c[select], axis=0)


def center2cog(c: np.ndarray):
    """
    Centers a set of atomic coordinates to their center of gravity.

    Args:
        c (np.ndarray): An array of shape (N, 3) representing N atomic positions (x, y, z).

    Returns:
        np.ndarray: A new array of centered coordinates where the center of gravity is at (0,0,0).

    Notes:
    - Uses `centerOfGravity(c)` to compute the center of gravity.
    - Each atomic position is shifted by subtracting the center of gravity.
    """
    c = np.asarray(c)
    cog = centerOfGravity(c)
    return c - cog

########################################################################################
def normOfV(V):
    '''
    Returns the norm of a vector V.
    Args:
        V (np.ndarray): A 3-element array representing a vector [Vx, Vy, Vz].
    Returns:
        float: The norm of the vector.
    '''
    V = np.asarray(V)
    return np.linalg.norm(V)


def normV(V):
    '''
    Computes the normalized unit vector of a vector V.
    Args:
        V (np.ndarray): A 3-element array representing a vector [Vx, Vy, Vz].
    Returns:
        np.ndarray: A 3-element array representing the normalized vector.
        
    '''
    V = np.asarray(V)
    N = normOfV(V)
    if N == 0:
        return np.zeros_like(V)
    return V / N

#########################################################################################

def AngleBetweenVV(lineDV, planeNV):
    """Return the angle, in degrees, between two vectors."""
    ldv = np.array(lineDV)
    pnv = np.array(planeNV)
    numerator = np.dot(ldv, pnv)
    denominator = normOfV(lineDV) * normOfV(planeNV)
    if denominator == 0:
        alpha = np.NaN
    else:
        alpha = 180 * np.arccos(np.clip(numerator / denominator, -1, 1)) / np.pi
    return alpha

def signedAngleBetweenVV(v1, v2, n):
    """
    Return the signed angle between two vectors in degrees in [0, 360].

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        n (np.ndarray): Normal of the plane formed by the two vectors.

    Returns:
        float: Signed angle in degrees.
    """
    cosTh = np.dot(v1, v2)
    sinTh = np.dot(np.cross(v1, v2), n)
    angle = np.rad2deg(np.arctan2(sinTh, cosTh))
    if angle >= 0:
        return angle
    return 360 + angle

#########################################################################################
def centerToVertices(coordVertices: np.ndarray,
                     cog: np.ndarray):
    """
    Computes the vectors and distances between the center of gravity (cog) 
    and each vertex of a polyhedron.
    Args:
        coordVertices (np.ndarray): Array of shape (n_vertices, 3) containing the coordinates of the vertices.
        cog (np.ndarray): A 3-element array representing the center of gravity of the nanoparticle.

    Returns:
        tuple:
            - directions (np.ndarray): Array of shape (n_vertices, 3) containing the vectors from cog to each vertex.
            - distances (np.ndarray): Array of shape (n_vertices,) containing the distances from cog to each vertex.
    """
    coordVertices = np.asarray(coordVertices)
    cog = np.asarray(cog)
    distances = Rbetween2Points(coordVertices, cog)  # Vectorisé pour tous les vertices
    directions = coordVertices - cog

    return directions, distances

######################################## rotation
def Rx(a):
    """Return the R/x rotation matrix."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ]
    )
  
def Ry(a):
    """Return the R/y rotation matrix."""
    return np.array(
        [
            [np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0, np.cos(a)]
        ]
    )
  
def Rz(a):
    """Return the R/z rotation matrix."""
    return np.array(
        [
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ]
    )

def EulerRotationMatrix(gamma, beta, alpha, order="zyx"):
    """
    Return a 3x3 Euler rotation matrix.

    Args:
        gamma: Rot/x (°).
        beta: Rot/y (°).
        alpha: Rot/z (°).
        order: If (order="zyx"): returns Rz(alpha) * Ry(beta) * Rx(gamma).

    Returns:
        np.ndarray: A 3x3 Euler matrix.
    """
    R = 1.
    gammarad = gamma * np.pi / 180
    betarad = beta * np.pi / 180
    alpharad = alpha * np.pi / 180
    for i in range(3):
        if order[i] == "x":
            R = R * Rx(gammarad)
        if order[i] == "y":
            R = R * Ry(betarad)
        if order[i] == "z":
            R = R * Rz(alpharad)
    return R

def plotPalette(Pcolors, namePC, angle=0, savePngAs=None):
    """
    Plot a 1D palette of colors with names.

    Args:
        Pcolors: 1D list with hex colors.
        namePC: Label for each color.
        angle: Rotation angle of the text.
        savePngAs: Also saves the palette in a png file (default: None).
    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.palplot(sns.color_palette(Pcolors))
    ax = plt.gca()

    for i, name in enumerate(namePC):
        ax.set_xticks(np.arange(len(namePC)))
        ax.tick_params(length=0)
        ax.set_xticklabels(namePC, weight='bold', size=10, rotation=angle)
    if savePngAs is not None:
        plt.tight_layout()
        plt.savefig(savePngAs, dpi=600, transparent=True)
    plt.show()
    return

def rgb2hex(c, frac=True):
    """
    Convert an RGB color to its hexadecimal representation.

    It has an optional frac argument to handle the case where the RGB values
    are provided as fractions (ranging from 0 to 1) or as integers
    (ranging from 0 to 255).

    Args:
        c: RGB color tuple.
        frac (bool): If True, RGB values are fractions (0-1).

    Returns:
        str: Hexadecimal representation.
    """
    if frac:
        r = int(round(c[0] * 255))
        g = int(round(c[1] * 255))
        b = int(round(c[2] * 255))
    else:
        r = c[0]
        g = c[1]
        b = c[2]
    return f"[x{r:02X}{g:02X}{b:02X}]"

def clone(system):
    """
    Create and return a deep copy of any pyNanoMatBuilder system.
    
    This ensures that all internal arrays, ASE atoms objects, 
    and Pymatgen structures are fully independent from the original.
    
    Args:
        system: A pyNanoMatBuilder object (e.g., Crystal, Icosahedron).
        
    Returns:
        A completely independent clone of the input system.
    """
    import copy
    return copy.deepcopy(system)

def deleteElementsOfAList(t,
                          list2Delete: bool):
    """
    Return a new list with elements deleted based on a boolean mask.

    Args:
        t: List or array-like.
        list2Delete (bool): Boolean mask, list2Delete[i] = True deletes t[i].

    Returns:
        list: Filtered list.
    """

    if len(t) != len(list2Delete):
        sys.exit(
            "the input list and the array of booleans must have the same dimension. "
            "Check your code"
        )
    if type(t) == list:
        tloc = np.array(t.copy())
    else:
        tloc = t.copy()
    tloc = np.delete(tloc, list2Delete, axis=0)
    return list(tloc)

def centerTitle(content=None):
    """
    Centers and renders as HTML a text in the notebook
    font size = 16px, background color = dark grey, foreground color = white
    """
    
    from IPython.display import display, HTML
    display(HTML(f"<div style='text-align:center; font-weight: bold; font-size:16px;background-color: #343132;color: #ffffff'>{content}</div>"))

def centertxt(content=None,font='sans', size=12,weight="normal",bgc="#000000",fgc="#ffffff"):
    """
    Centers and renders as HTML a text in the notebook
    
    input: 
        - content = the text to render (default: None)
        - font = font family (default: 'sans', values allowed =  'sans-serif' | 'serif' | 'monospace' | 'cursive' | 'fantasy' | ...)
        - size = font size (default: 12)
        - weight = font weight (default: 'normal', values allowed = 'normal' | 'bold' | 'bolder' | 'lighter' | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 )
        - bgc = background color (name or hex code, default = '#ffffff')
        - fgc = foreground color (name or hex code, default = '#000000')
        
    """
    
    from IPython.display import display, HTML
    display(HTML(f"<div style='text-align:center; font-family: {font}; font-weight: {weight}; font-size:{size}px;background-color: {bgc};color: {fgc}'>{content}</div>"))

######################################## Planes & Directions
# def planeFittingLSF(coords: np.float64,
#                     printErrors: bool=False,
#                     printEq: bool=True):
#     """
#     Fit a plane ux + vy + wz + h = 0 to 3D points using least squares.

#     Args:
#         coords (np.ndarray): Array with shape (N, 3) containing point coordinates.
#         printErrors (bool): If True, prints error/residual details.
#         printEq (bool): If True, prints the fitted plane equation.

#     Returns:
#         np.ndarray: Array [u, v, w, h].
#     """
#     from numpy import linalg as la
#     xs = coords[:, 0]
#     ys = coords[:, 1]
#     zs = coords[:, 2]
#     nat = len(xs)
#     cog = centerOfGravity(coords)
#     # mat = np.zeros((3,3))
#     # for i in range(nat):
#     #     mat[0,0]=mat[0,0]+(xs[i]-cog[0])**2
#     #     mat[1,1]=mat[1,1]+(ys[i]-cog[1])**2
#     #     mat[2,2]=mat[2,2]+(zs[i]-cog[2])**2
#     #     mat[0,1]=mat[0,1]+(xs[i]-cog[0])*(ys[i]-cog[1])
#     #     mat[0,2]=mat[0,2]+(xs[i]-cog[0])*(zs[i]-cog[2])
#     #     mat[1,2]=mat[1,2]+(ys[i]-cog[1])*(zs[i]-cog[2])    
#     # mat[1,0]=mat[0,1]
#     # mat[2,0]=mat[0,2]
#     # mat[2,1]=mat[1,2]

#     coords_centered = coords - cog
#     mat = coords_centered.T @ coords_centered

#     eigenvalues, eigenvectors = la.eig(mat)
#     # the eigenvector associated with the smallest eigenvalue is the vector normal to the plane
#     # print(eigenvalues)
#     # print(eigenvectors)
#     indexMinEigenvalue = np.argmin(eigenvalues)
#     # print(indexMinEigenvalue)
#     # print(la.norm(eigenvectors[:,indexMinEigenvalue]))
#     u, v, w = eigenvectors[:, indexMinEigenvalue]
#     h = -u * cog[0] - v * cog[1] - w * cog[2]
#     if printEq:
#         print(
#             f"bare solution: {u:.5f} x + {v:.5f} y + {w:.5f} z + {h:.5f} = 0"
#         )
#     tmp = coords.copy()
#     ones = np.ones(nat)
#     tmp = np.column_stack((coords, ones))
#     fit = np.array([u, v, w, h])
#     fit = fit.reshape(4, 1)
#     errors = tmp @ fit
#     residual = la.norm(errors)
#     if printErrors:
#         print("errors:")
#         print(errors)
#         print(f"residual: {residual}")
#     return np.array([u, v, w, h]).real

def planeFittingLSF(coords: np.float64,
                    printErrors: bool=False,
                    printEq: bool=True):
    '''
    Least-square fitting of the equation of a plane ux + vy + wz + h = 0
    that passes as close as possible to a set of 3D points
    Args:
        - coords (np.ndarray): array with shape (N,3) that contains the 3 coordinates for each of the N points
        - printErrors (bool): if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed 
        - printEq (bool): if True, prints equation.
    Returns:
        numpy array([u v w h])
    '''
    import numpy as np
    from numpy import linalg as la
    xs = coords[:,0]
    ys = coords[:,1]
    zs = coords[:,2]
    nat = len(xs)
    select=[i for i in range(nat)]
    cog = centerOfGravity(coords, select)
    mat = np.zeros((3,3))
    for i in range(nat):
        mat[0,0]=mat[0,0]+(xs[i]-cog[0])**2
        mat[1,1]=mat[1,1]+(ys[i]-cog[1])**2
        mat[2,2]=mat[2,2]+(zs[i]-cog[2])**2
        mat[0,1]=mat[0,1]+(xs[i]-cog[0])*(ys[i]-cog[1])
        mat[0,2]=mat[0,2]+(xs[i]-cog[0])*(zs[i]-cog[2])
        mat[1,2]=mat[1,2]+(ys[i]-cog[1])*(zs[i]-cog[2])    
    mat[1,0]=mat[0,1]
    mat[2,0]=mat[0,2]
    mat[2,1]=mat[1,2]
    eigenvalues, eigenvectors = la.eig(mat)
    # the eigenvector associated with the smallest eigenvalue is the vector normal to the plane
    # print(eigenvalues)
    # print(eigenvectors)
    indexMinEigenvalue = np.argmin(eigenvalues)
    # print(indexMinEigenvalue)
    # print(la.norm(eigenvectors[:,indexMinEigenvalue]))
    u,v,w = eigenvectors[:,indexMinEigenvalue]
    h = -u*cog[0] - v*cog[1] - w*cog[2]
    if printEq: print(f"bare solution: {u:.5f} x + {v:.5f} y + {w:.5f} z + {h:.5f} = 0")
    tmp = coords.copy()
    ones = np.ones(nat)
    tmp = np.column_stack((coords,ones))
    fit = np.array([u,v,w,h])
    fit = fit.reshape(4,1)
    errors = tmp@fit
    residual = la.norm(errors)
    if printErrors:
        print(f"errors:")
        print(errors)
        print(f"residual: {residual}")
    return np.array([u,v,w,h]).real
