import sys
import numpy as np
import pyNanoMatBuilder.utils as pnmbu

###########################################################################################################
class fccCubo:
    nFaces = 14
    nEdges = 24
    nVertices = 12
    edgeLengthF = 1
    radiusCSF = 1
    interShellF = 1/radiusCSF
    radiusISF = 3/4
  
    def __init__(self, Rnn, nShell):
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
          
    def __str__(self):
        return(f"Cuboctahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of a cuboctahedron of size i"""
        return round((10*i**3 + 11*i)/3 + 5*i**2 + 1)
    
    def nAtomsPerShellAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni-Sum)
        return n
    
    def nAtomsPerShellCumulativeAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n
    
    def nAtomsAnalytic(self):
        n = self.nAtomsF(self.nShell)
        return n
    
    def edgeLength(self):
        return self.Rnn*self.nShell

    def radiusCircumscribedSphere(self):
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        return self.radiusISF*self.edgeLength()

    def area(self):
        el = self.edgeLength()
        return (6 + 2*np.sqrt(3)) * el**2 
    
    def volume(self):
        el = self.edgeLength()
        return (5 * np.sqrt(2)/3) * el**3
    
    def MakeVertices(self,i):
        """
        input:
            - i = index of the shell
        returns:
            - CoordVertices = the 12 vertex coordinates of the ith shell of an icosahedron
            - edges = indexes of the 30 edges
            - faces = indexes of the 20 faces 
        """
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell = {self.nShell}")
        else:            
            scale = self.interShellDistance * i
            CoordVertices = [ pnmbu.vertex(-1, 1, 0, scale),\
                              pnmbu.vertex( 1, 1, 0, scale),\
                              pnmbu.vertex(-1,-1, 0, scale),\
                              pnmbu.vertex( 1,-1, 0, scale),\
                              pnmbu.vertex( 0,-1, 1, scale),\
                              pnmbu.vertex( 0, 1, 1, scale),\
                              pnmbu.vertex( 0,-1,-1, scale),\
                              pnmbu.vertex( 0, 1,-1, scale),\
                              pnmbu.vertex( 1, 0,-1, scale),\
                              pnmbu.vertex( 1, 0, 1, scale),\
                              pnmbu.vertex(-1, 0,-1, scale),\
                              pnmbu.vertex(-1, 0, 1, scale) ]
            CoordVertices = np.array(CoordVertices)
            edges = [( 4, 2), ( 4, 3), ( 5, 0), ( 5, 1), ( 6, 2), ( 6, 3), ( 7, 0), ( 7, 1),\
                     ( 8, 1), ( 8, 3), ( 8, 6), ( 8, 7), ( 9, 1), ( 9, 3), ( 9, 4), ( 9, 5),\
                     ( 10, 0), ( 10, 2), ( 10, 6), ( 10, 7), ( 11, 0), ( 11, 2), ( 11, 4), ( 11, 5)]
            faces3 = [( 4, 2, 11), ( 4, 3, 9), ( 5, 0, 11), ( 5, 1, 9), ( 6, 2, 10), ( 6, 3, 8), ( 7, 0, 10), ( 7, 1, 8)]
            faces4 = [( 2, 4, 3, 6), ( 0, 5, 1, 7), ( 1, 8, 3, 9), ( 6, 8, 7, 10), ( 4, 9, 5, 11), ( 0, 10, 2, 11)]
            edges = np.array(edges)
            faces3 = np.array(faces3)
            faces4 = np.array(faces4)
            # print("len = ",len(CoordVertices))
            # for i in range(len(CoordVertices)):
            #     print("i, CoordV ",i,CoordVertices[i])
        return CoordVertices, edges, faces3, faces4

    def coords(self):
        # central atom = "1st shell"
        c = [[0., 0., 0.]]
        self.nAtoms = 1
        self.nAtomsPerShell = [0]
        self.nAtomsPerShell[0] = 1
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFace3Atoms = []
        indexFace4Atoms = []
        for i in range(1,self.nShell+1):
            # vertices
            nAtoms0 = self.nAtoms
            cshell, E, F3, F4 = self.MakeVertices(i)
            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

            # intermediate atoms on edges e
            nAtoms0 = self.nAtoms
            # print("nAtoms0 = ",nAtoms0)
            Rvv = pnmbu.RAB(cshell,E[0,0],E[0,1]) #distance between two vertex atoms
            nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
            nIntervals = nAtomsOnEdges + 1
            # print("nAtomsOnEdges = ",nAtomsOnEdges,"  len(E) = ",len(E))
            coordEdgeAt = []
            for n in range(nAtomsOnEdges):
                for e in E:
                    a = e[0]
                    b = e[1]
                    coordEdgeAt.append(cshell[a]+pnmbu.vector(cshell,a,b)*(n+1) / nIntervals)
            self.nAtomsPerShell[i] += nAtomsOnEdges * len(E) # number of edges x nAtomsOnEdges
            self.nAtoms += nAtomsOnEdges * len(E)
            c.extend(coordEdgeAt)
            indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
            
            # now, triangular facet atoms
            coordFace3At = []
            nAtomsOnFaces3 = 0
            nAtoms0 = self.nAtoms
            for f in F3:
                nAtomsOnFaces3,coordFace3At = pnmbu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces3,coordFace3At)
            self.nAtomsPerShell[i] += nAtomsOnFaces3
            self.nAtoms += nAtomsOnFaces3
            c.extend(coordFace3At)
            indexFace3Atoms.extend(range(nAtoms0,self.nAtoms))

            # now, square facet atoms
            coordFace4At = []
            nAtomsOnFaces4 = 0
            nAtoms0 = self.nAtoms
            for f in F4:
                nAtomsOnFaces4,coordFace4At = pnmbu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces4,coordFace4At)
            self.nAtomsPerShell[i] += nAtomsOnFaces4
            self.nAtoms += nAtomsOnFaces4
            c.extend(coordFace4At)
            indexFace4Atoms.extend(range(nAtoms0,self.nAtoms))
            
        # print(indexVertexAtoms)
        # print(indexEdgeAtoms)
        # print(indexFaceAtoms)
        return c,[indexVertexAtoms,indexEdgeAtoms,indexFace3Atoms,indexFace4Atoms]
    
    def prop(self):
        print(self)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pnmbu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: rhombic dodecahedron")

