import visualID as vID
from visualID import  fg, hl, bg

import sys
import numpy as np
import pyNanoMatBuilder.utils as pNMBu
import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view
from ase.cluster.cubic import FaceCenteredCubic

from pyNanoMatBuilder import platonicNPs as pNP
from pyNanoMatBuilder import johnsonNPs as jNP

###########################################################################################################
class fcctpt:
    nFaces = 8
    nEdges = 9
    nVertices = 9
    edgeLengthF = 2

    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayerTd: int=1,
                 nLayer: int = 3):
        self.element = element
        self.Rnn = Rnn
        self.nLayerTd = int(nLayerTd)
        self.tbp = jNP.fcctbp(self.element,self.Rnn,self.nLayerTd)
        self.nLayertbp = 2*self.nLayerTd - 1
        self.nLayer = nLayer
        self.nAtoms = 0
        self.interLayerDistance = self.tbp.interLayerDistance
        self.nAtomsPerEdge = self.nLayerTd+1
        self.cog = np.array([0., 0., 0.])
        if self.nLayer%2 != 1:
            self.nLayer = self.nLayer + 1
            print(f"Number of layers of the platelet must be an even number. nLayer parameter forced to be = {self.nLayer}")
        if self.nLayer > self.nLayertbp:
            sys.exit(f"Number of layers of the triangular platelet ({self.nLayer}) cannot be > to the total number of layers of the trigonal bipyramid {self.nLayertbp}")

    def __str__(self):
        return(f"Truncated fcc double tetrahedron with {self.nLayer} layer(s) and Rnn = {self.Rnn}") 

    def edgeLength(self):
        return self.tbp.edgeLength()

    def coords(self):
        chrono = pNMBu.timer(); chrono.chrono_start()
        vID.centertxt("Generation of the coordinates of the trigonal bipyramid, based on the fcc tetrahedron",bgc='#007a7a',size='14',weight='bold')
        tbp = jNP.fcctbp(self.element,self.Rnn,self.nLayerTd+1)
        asetbp = tbp.coords()
        view(asetbp)
        nAtoms = asetbp.get_global_number_of_atoms()
        # cog = pNMBu.centerOfGravity(asetbp.get_positions())
        # print("cog = ",cog)
        vID.centertxt("Truncation of the trigonal bipyramid",bgc='#007a7a',size='14',weight='bold')
        print("Now calculating the coordinates of the twin plane (defined by atoms 0,1,2)")
        coordTwPVertices = asetbp.get_positions()[[0,1,2]]
        twinningPlane = pNMBu.hklPlaneFitting(coordTwPVertices)
        twinningPlane = pNMBu.normalizePlane(twinningPlane)
        print("Now calculating the coordinates of the truncation planes")
        truncationPlane1 = twinningPlane.copy()
        truncationPlane1[3] = -(self.nLayer-1)*self.interLayerDistance/2
        print("signed distance between truncation plane and origin = ",pNMBu.Pt2planeSignedDistance(truncationPlane1,[0,0,0]))
        truncationPlane2 = -twinningPlane.copy()
        truncationPlane2[3] = -(self.nLayer-1)*self.interLayerDistance/2
        print("signed distance between truncation plane and origin = ",pNMBu.Pt2planeSignedDistance(truncationPlane2,[0,0,0]))
        trPlanes = [truncationPlane1, truncationPlane2]
        AtomsAbovePlanes = pNMBu.truncateAboveEachPlane(trPlanes,asetbp.get_positions())
        del asetbp[AtomsAbovePlanes]
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        nAtoms = asetbp.get_global_number_of_atoms()
        print(f"Total number of atoms = {nAtoms}")
        return asetbp
        
    def prop(self):
        print(self)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge at the twin boundary = {self.nAtomsPerEdge}")
        print(f"inter-layer distance = {self.interLayerDistance:.2f} Å")
        print(f"height of the platelet = {self.interLayerDistance*(self.nLayer-1)*0.1:.2f} nm")
        # print(f"area = {6*self.Td.area()/4*1e-2:.1f} nm2")
        # print(f"volume = {2*self.Td.volume()*1e-3:.1f} nm3")
        print(f"face-vertex-edge angle in Td = {self.tbp.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle in Td = {self.tbp.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond) angle in Td = {self.tbp.vcvAngle:.1f}°")
        print(f"coordinates of the center of gravity = {self.cog}")
