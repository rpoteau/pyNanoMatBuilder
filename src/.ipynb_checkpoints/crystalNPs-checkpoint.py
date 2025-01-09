import sys, os, pathlib
import re
import math
import numpy as np
import pandas as pd
import pyNanoMatBuilder.utils as pyNMBu
from pyNanoMatBuilder import data
from ase.build import bulk
from ase import io
from ase.visualize import view
from ase.build.supercells import make_supercell
from ase.geometry import cellpar_to_cell

from visualID import  fg, hl, bg
import visualID as vID

class Crystal:
    
    def __init__(self,
                 crystal: str='Au',
                 scaleDmin2: float=None,
                 setSymbols2: np.ndarray=None,
                 userDefCif: str=None,
                 shape: str='sphere',
                 size: float=[2,2,2],
                 directionsPPD: np.ndarray=np.array([[1,0,0],[0,1,0],[0,0,1]]),
                 buildPPD: str='xyz',
                 directionWire: float=[0,0,1],
                 #NEW CYLINDER
                 directionCylinder: float=[0,0,1],
                 refPlaneWire: float=[1,0,0],
                 nRotWire: int=6,
                 surfacesWulff: np.ndarray=None,
                 eSurfacesWulff: np.ndarray=None,
                 sizesWulff: np.ndarray=None,
                 symWulff: bool = True,
                 jmolCrystalShape: bool=True,
                 aseSymPrec: float=1e-4,
                 pbc: bool=False,
                 threshold: float=1e-3,
                 dbFolder: str=data.pyNMBvar.dbFolder,
                 postAnalyzis: bool=True,
                 aseView: bool= False,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis: bool = False,
                 noOutput: bool = False,
                 calcPropOnly: bool = False,
                ):
        self.dbFolder = dbFolder #database folder that contains cif files
        self.crystal = crystal # see list with the pyNMBu.ciflist() command
        self.shape = shape.strip(' ') # 'sphere', 'ellipsoid', 'cube', 'wire', 'Wulff', 'cylinder'
        self.size = size
        self.directionsPPD = directionsPPD
        self.buildPPD = buildPPD
        self.directionWire = directionWire
        #NEW for cylinder
        self.directionCylinder = directionCylinder
        self.refPlaneWire = refPlaneWire
        self.nRotWire = nRotWire
        self.surfacesWulff = surfacesWulff
        self.eSurfacesWulff = eSurfacesWulff
        self.sizesWulff = sizesWulff
        self.symWulff = symWulff
        self.jmolCrystalShape = jmolCrystalShape
        self.aseSymPrec = aseSymPrec
        self.pbc = pbc
        self.threshold = threshold
        self.nAtoms = 0
        self.cif = None
        self.cifname = None
        self.userDefCif = userDefCif
        self.trPlanes = None
        self.dim=[0,0,0] # main dimensions for files 
        if "Wulff" in self.shape :
            self.WulffShape = self.shape.split(":")
            if len(self.WulffShape) == 2:
                self.WulffShape = self.WulffShape[1].lstrip(' ') #removes 'Wulff:'
                self.shape="Wulff: " + self.WulffShape #normalizes the name of the Wulff shape
            else:
                self.WulffShape = None

        if self.userDefCif is not None: self.loadExternalCif()
        #NEW CYLINDER
        # if self.shape in data.pyNMBimg.IMGdf.index:
        #     self.imageFile = pyNMBu.imageNameWithPathway(data.pyNMBimg.IMGdf["png file"].loc[self.shape])
        # else:
        #     sys.exit("Shape {self.shape} is unknown")
        if not noOutput: vID.centerTitle(f"{self.crystal} {self.shape}")

        self.bulk(noOutput)
        if scaleDmin2 is not None: pyNMBu.scaleUnitCell(self,scaleDmin2,noOutput=noOutput)
        if setSymbols2 is not None: self.cif.set_chemical_symbols(setSymbols2)
        if aseView: view(self.cif)
         
        
        if not calcPropOnly:
            self.makeNP(noOutput)
            if aseView: 
                view(self.sc)
                view(self.NP)
            if postAnalyzis:
                self.prop(noOutput)
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface, noOutput)
                if aseView: view(self.NPcs)
        # self.NP.center() #must be done before the surface calculation, since the Hull analysis assulmes that the structure is centered in [0,0,0]
          
    def __str__(self):
        return(f"Crystal = {self.crystal} {self.shape}")

    def loadExternalCif(self):
        self.cif = io.read(self.userDefCif)
        path2extCif = pathlib.Path(self.userDefCif)
        if not path2extCif.exists():
            sys.exit(f"file {self.userDefCif} not found. Check the file name or its location")
        cifFile =  open(self.userDefCif, 'r')
        name1 = "_chemical_name_systematic"
        name2 = "_chemical_formula_sum"
        name3 = "_chemical_formula_moiety"
        cifFileLines = cifFile.readlines()
        re_name_systematic = re.compile(name1)
        re_name_sum = re.compile(name2)
        re_name_moiety = re.compile(name3)
        crystal1 = None
        crystal2 = None
        crystal3 = None
        for line in cifFileLines:
            if re_name_systematic.search(line):
                parts = line.split()
                crystal1 = ' '.join(parts[1:])
            if re_name_sum.search(line):
                parts = line.split()
                crystal2 = ' '.join(parts[1:])
            if re_name_moiety.search(line):
                parts = line.split()
                crystal3 = ' '.join(parts[1:])
        cifFile.close()
        if crystal1 is not None:
            self.crystal = crystal1
        elif crystal3 is not None:
            self.crystal = crystal3
        elif crystal2 is not None:
            self.crystal = crystal2
        else: self.crystal = "unknown"

    # def real_size(self):
    #     positions=self.NP.get_positions()
    #     x_min, y_min, z_min = positions.min(axis=0) #columns
    #     x_max, y_max, z_max = positions.max(axis=0)
    #     xlength=x_max-x_min
    #     ylength=y_max-y_min
    #     zlength=z_max-z_min
    #     return xlength,ylength,zlength
    
    # #NEW
    # def radiusInscribedSphere(self):
    #     if self.shape == 'sphere' :
    #         return self.size[0]
    #     if self.shape == 'cylinder' :
    #         return self.size[0]
    #     if self.shape == 'ellipsoid' :
    #         return min(self.size)
                

    #     #not done yet   
    #     if self.shape == 'parallepiped' :
            
    #     if self.shape == 'wire' :
    #         return 

    # def radiusCircumscribedSphere(self):
    #     if self.shape == 'sphere' :
    #         return self.size[0]
    #     if self.shape == 'cylinder' :
    #         return np.sqrt((self.size[0]/2)**2*+(self.size[1]/2)**2)
    #     if self.shape == 'ellipsoid' :
    #         return max(self.size)
    #     #not done yet
    #     if self.shape == 'parallepiped' :
            
    #     if self.shape == 'wire' :
            
            

            
    def bulk(self, noOutput):

        if self.userDefCif is None:
            path2cif = os.path.join(pyNMBu.pyNMB_location(),self.dbFolder)
            # if not noOutput: vID.centertxt(f"List of stored cif files in pyNanoMatBuilder ('{self.dbFolder}' folder)",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
            # if not noOutput: display(data.pyNMBcif.CIFdf)
            if  self.crystal.upper() in data.pyNMBcif.CIFdf.index.str.upper():
                dftmp = pd.DataFrame(index=data.pyNMBcif.CIFdf.index.copy())
                data.pyNMBcif.CIFdf.index = data.pyNMBcif.CIFdf.index.str.upper()
                self.cifname = data.pyNMBcif.CIFdf["cif file"].loc[self.crystal.upper()]
                data.pyNMBcif.CIFdf.index = dftmp.index.copy() # revert. Ugly, but I do not know how to manage it differently
            else:
                if noOutput: display(data.pyNMBcif.CIFdf) #because the databse has not been displayed just above
                sys.exit(f"{fg.RED}{bg.LIGHTREDB}The database does not contain bulk parameters for the '{self.crystal}' crystal.\n"\
                         f"Please provide a cif file{fg.OFF}")
            self.cif = io.read(os.path.join(path2cif,self.cifname))
        else:
            self.cif = io.read(self.userDefCif)
            path2extCif = pathlib.Path(self.userDefCif)
            self.cifname = pathlib.Path(*path2extCif.parts[-1:])

        
        pyNMBu.returnUnitcellData(self)
        if not noOutput: print(f"cif parameters for {self.crystal} found in {self.cifname}")
        return 

    def makeSuperCell(self,noOutput):
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        if not noOutput: vID.centertxt(f"Making a multiple cell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        extendSizeByFactor = 1.06
        if (self.shape == 'sphere'):
            # first calculate the size of the supercell
            sphereRadius = self.size[0]
            Ma = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'ellipsoid' or self.shape == 'supercell' or self.shape == 'parallepiped'):
            # first calculate the size of the supercell
            Ma = int(np.round(extendSizeByFactor * self.size[0]*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * self.size[1]*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * self.size[2]*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'wire'):
            maxDim = np.max(self.size)*10*1.5
            Ma = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[2]))
        #NEW CYLINDER
        elif (self.shape == 'cylinder'):
            cylinderRadius = self.size[0]
            half_height=self.size[1]
            Ma = int(np.round(extendSizeByFactor * cylinderRadius*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * cylinderRadius*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * half_height*10/self.cif.cell.lengths()[2]))
        elif ('Wulff' in self.shape):
            if np.argmax(self.sizesWulff) == 1:
                maxDim = self.sizesWulff[0]*10*1.5
                print('maxDim',maxDim)
            else:
                maxDim = np.max(self.sizesWulff)*10*1.5
            # print(f"{maxDim = }")
            Ma = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[2]))
        Ma1nm =  int(np.round(20/self.cif.cell.lengths()[0]))
        Mb1nm =  int(np.round(20/self.cif.cell.lengths()[1]))
        Mc1nm =  int(np.round(20/self.cif.cell.lengths()[2]))
        Ma1nm = min(Ma1nm,Ma)
        Mb1nm = min(Mb1nm,Mb)
        Mc1nm = min(Mc1nm,Mc)
        if not noOutput: print(f"First making a {Ma1nm}x{Mb1nm}x{Mc1nm} supercell")
        M1nm = [[Ma1nm, 0, 0], [0, Mb1nm, 0], [0, 0, Mc1nm]]
        sc1nm=make_supercell(self.cif,M1nm)
        Ma = Ma/Ma1nm
        Mb = Mb/Mb1nm
        Mc = Mc/Mc1nm
        #finds the nearest even numbers
        Ma = math.ceil(Ma / 2.) * 2
        Mb = math.ceil(Mb / 2.) * 2
        Mc = math.ceil(Mc / 2.) * 2
        if not noOutput: print(f"Making a {Ma}x{Mb}x{Mc} supercell of the supercell")
        if not noOutput: print(f"       = {Ma*Ma1nm}x{Mb*Mb1nm}x{Mc*Mc1nm} supercell")
        M = [[Ma, 0, 0], [0, Mb, 0], [0, 0, Mc]]
        sc=make_supercell(sc1nm,M)
        # print(cif.cell.cellpar())
        # print(cellpar_to_cell(cif.cell.cellpar()))
        # print(sc.cell.cellpar())
        # print(cellpar_to_cell(sc.cell.cellpar()))
        V = cellpar_to_cell(sc.cell.cellpar())
        if not noOutput: print(f"Center of Mass:", [f"{c:.2f}" for c in sc.get_center_of_mass()]," Å")
        if not noOutput: print("Now translating the supercell to O")
        #sc.center(about=(0.0,0.0,0.0))
        sc.translate(-V[0]/2)
        sc.translate(-V[1]/2)
        sc.translate(-V[2]/2)
        if not noOutput: print(f"Center of Mass after translation of the supercell: {sc.get_center_of_mass()} Å")
        self.sc = sc.copy()
        nAtoms=len(self.sc.get_positions())
        if not noOutput: print(f"Total number of atoms = {nAtoms}")
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
    def makeSphere(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make a sphere",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        com = self.sc.get_center_of_mass()
        sphereRadius = self.size[0]
        delAtom = []
        for atomCoord in self.sc.positions:
            delAtom.extend(pyNMBu.Rbetween2Points(com,atomCoord)/10 > [sphereRadius])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
               
    def makeEllipsoid(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make an ellipsoid",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        com = self.sc.get_center_of_mass()
        size = np.array(self.size)*10 #nm to angstrom
        def outside(coord,com,size):
            return (coord[0]-com[0])**2/(size[0])**2+(coord[1]-com[1])**2/(size[1])**2+(coord[2]-com[2])**2/(size[2])**2
        delAtom = []
        for atom in self.sc.positions:
            delAtom.extend([outside(atom,com,size) > 1])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        #NEW
         
    def makeWire(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make a wire",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        if self.refPlaneWire is None: self.refPlaneWire = pyNMBu.returnPlaneParallel2Line(self.directionWire,[1,0,0],debug=False)
        normal = pyNMBu.normal2MillerPlane(self,self.refPlaneWire,printN=not noOutput)
        trPlanes = pyNMBu.planeRotation(self,normal,self.directionWire,self.nRotWire,debug=False,noOutput=noOutput)
        for i,p in enumerate(trPlanes):
            trPlanes[i] = pyNMBu.normV(p)
        radius = 10*self.size[0]/2
        tradius = np.full((self.nRotWire,1),-radius)
        trPlanes = np.append(trPlanes,tradius,axis=1)
        if not self.pbc:
            halfLength = 10*self.size[1]/2
            ptop = np.append(pyNMBu.normV(self.directionWire),-halfLength)
            pbottom = np.append(-pyNMBu.normV(self.directionWire),-halfLength)
            trPlanes = np.append(trPlanes,ptop)
            trPlanes = np.append(trPlanes,pbottom)
            trPlanes = np.reshape(trPlanes,(self.nRotWire+2,4))
        AtomsAbovePlanes = pyNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,eps=self.threshold,noOutput=noOutput)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        self.trPlanes = trPlanes
        if not noOutput: vID.centertxt(f"Nanowire moved to the center of the unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        # self.NP.center()
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        #NEW
        

    def makeCylinder(self,noOutput): #in def init, the two entries are specified : size=[diameter,length] and directionCylinder=[h,k,l]
        """
        Create a cylinder along the direction [hkl], [radius,length] are specified by user in the class Crystal
        """
    
        if not noOutput: vID.centertxt(f"Removing atoms to make a cylinder",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        #dimension
        radius = 10*self.size[0]/2 
        radius_squared=radius**2
        half_height = 10 * self.size[1] / 2
        axis=np.array(self.directionCylinder)
        com = self.sc.get_center_of_mass()
        #delAtom = []
    
    
        #delete atoms in the supercell
        delAtom = [i for i, pos in enumerate(self.sc.positions) if pyNMBu.isnt_inside_cylinder(pos,radius, radius_squared, half_height)]
        
        #rotating the coordinates again to have the hkl orientation
        self.sc.positions= pyNMBu.rotateMoltoAlignItWithAxis(self.sc.positions,axis,targetAxis=np.array([0, 0, 1]))
        self.NP = self.sc.copy()
        del self.NP[delAtom]
    
        if not noOutput: vID.centertxt(f"Nanocylinder moved to the center of the unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        # self.NP.center()
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        #NEW
      
    def makeParallelepiped(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make a parallelepiped",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        if self.buildPPD == "xyz":
            trPlanes = self.directionsPPD
        else:
            normal = []
            for d in self.directionsPPD:
                normal.append(pyNMBu.normal2MillerPlane(self,d,printN=not noOutput))
            trPlanes = pyNMBu.lattice_cart(self,normal,Bravais2cart=True,printV=not noOutput)
        for i,p in enumerate(trPlanes): trPlanes[i] = pyNMBu.normV(p)
        # 6 planes defined to cut between 
        # [-a/2 direction, a/2 direction], [-b/2 direction, b/2 direction], [-c/2 direction, c/2 direction]
        size = -np.array(self.size)*10/2 #nm!
        size = np.append(size,size,axis=0)
        trPlanes = np.append(trPlanes,-trPlanes,axis=0)
        trPlanes = np.append(trPlanes,size.reshape(6,1),axis=1)
        AtomsAbovePlanes = pyNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,eps=self.threshold,debug=False,noOutput=noOutput)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        #self.NP.center()
        self.trPlanes = trPlanes
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
  
    def makeWulff(self,noOutput):
        if not noOutput: vID.centertxt(f"Calculating truncation distances",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        if not noOutput: chrono = pyNMBu.timer(); chrono.chrono_start()
        trPlanes = []
        if self.eSurfacesWulff is None: sizes = []
        if self.eSurfacesWulff is not None: 
            sizes = []
            eSurf = []
        for i,p in enumerate(self.surfacesWulff):
            if self.symWulff:
                symP = self.ucSG.equivalent_lattice_points(p)
                normal = []
                for sp in symP:
                    normal.append(pyNMBu.normal2MillerPlane(self,sp,printN=not noOutput))
                trPlanes += list(normal)
                if self.eSurfacesWulff is None: sizes.append(len(symP)*[self.sizesWulff[i]])
                if self.eSurfacesWulff is not None: eSurf += (len(symP)*[self.eSurfacesWulff[i]])
            else:
                trPlanes.append(pyNMBu.normal2MillerPlane(self,p,printN=not noOutput))
                if self.eSurfacesWulff is None: sizes.append(self.sizesWulff[i])
                if self.eSurfacesWulff is not None: eSurf.append(self.eSurfacesWulff[i])
        trPlanes = np.array(trPlanes)
        trPlanes = pyNMBu.lattice_cart(self,trPlanes,Bravais2cart=True,printV=not noOutput)
        for i,p in enumerate(trPlanes): trPlanes[i] = pyNMBu.normV(p)
        # print("inside makeWulff, just after trPlanes normalization. ",trPlanes.tolist())
        if self.eSurfacesWulff is None: 
            sizes = -np.array(sizes)*10/2
            trPlanes = np.append(trPlanes,sizes.reshape(len(trPlanes),1),axis=1)
        else:
            mostStableE = min(eSurf)
            for i, e in enumerate(eSurf):
                sizes.append(-self.sizesWulff[0]*10*e/2/mostStableE)
            sizes = np.array(sizes)
            trPlanes = np.append(trPlanes,sizes.reshape(len(trPlanes),1),axis=1)
        # print("inside makeWulff, just after trPlanes size calculation. ",trPlanes)
        # print("sizes = ",sizes)
        AtomsAbovePlanes = pyNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,allP=False,\
                                                     eps=self.threshold,debug=False,noOutput=noOutput)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        # self.NP.center()
        self.trPlanes = trPlanes
        print(' self.trPlanes', self.trPlanes)
        if not noOutput: chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    def makeNP(self,noOutput):
        import os
        if not noOutput: vID.centertxt("Builder",bgc='#007a7a',size='14',weight='bold')
        if self.size is None:
            self.length = [2,2,2]
            if not noOutput: print(f"length parameter set up as = {self.size} nm")
        if self.shape == "sphere":
            if not noOutput: print(f"Sphere radius = {self.size[0]} nm")
            self.makeSuperCell(noOutput)
            self.makeSphere(noOutput)
        elif self.shape == "ellipsoid":
            if not noOutput: print(f"Ellipsoid radii = {self.size} nm")
            self.makeSuperCell(noOutput)
            self.makeEllipsoid(noOutput)
        elif self.shape == "parallepiped":
            if not noOutput: print(f"Parallepiped side length = {self.size} nm, directions = {list(self.directionsPPD)}")
            self.makeSuperCell(noOutput)
            self.makeParallelepiped(noOutput)
        elif self.shape == "supercell":
            if not noOutput: print(f"Supercell side length = {self.size} nm")
            if len(self.size) != 3: sys.exit("Please enter lengths along a,b and c axis, i.e. size=[l_a,l_b,l_c]")
            self.makeSuperCell(noOutput)
        elif self.shape == "wire":
            if not noOutput: print(f"Wire in the {self.directionWire} directionWire. Length x width = {self.size[1]} x {self.size[0]} nm")
            if not noOutput: print(f"Reference plane = {self.refPlaneWire}, {self.nRotWire}-th order rotation around {self.directionWire}")
            if not pyNMBu.isPlaneParrallel2Line(self.refPlaneWire, self.directionWire):
                print(f"{bg.DARKREDB}Warning! The reference truncation plane is not parallel to {self.directionWire}. Are you sure?{fg.OFF}")
                suggestedPlane = pyNMBu.returnPlaneParallel2Line(self.directionWire)
                print(f"Among other possibilities, you can try {suggestedPlane}")            
            else:
                if not noOutput: print(f"{bg.LIGHTGREENB}The reference truncation plane is parallel to {self.directionWire}{fg.OFF}")
            self.makeSuperCell(noOutput)
            self.makeWire(noOutput)

        #NEW CYLINDER
        elif self.shape == "cylinder":  
            if not noOutput: print(f"Cylinder in the {self.directionCylinder} directionCylinder. Length x width = {self.size[1]} x {self.size[0]} nm")
            self.makeSuperCell(noOutput)
            self.makeCylinder(noOutput)
        elif "Wulff" in self.shape :
            if self.WulffShape is not None:
                self.predefinedParameters4WulffShapes(noOutput)
            if self.surfacesWulff == None:
                sys.exit("Wulff construction requested, but no planes were given. Define them with the 'surfacesWulff' variable")
            if self.eSurfacesWulff == None and self.sizesWulff == None: 
                sys.exit("Either 'eSurfacesWulff' or 'sizesWulff' variables must be set up")
            if len(self.surfacesWulff) != len(self.eSurfacesWulff) and len(self.surfacesWulff) != len(self.sizesWulff):
                sys.exit("'surfacesWulff' and ('eSurfacesWulff' or 'sizesWulff') lists have different dimensions")
            self.makeSuperCell(noOutput)
            self.makeWulff(noOutput)
        self.nAtoms=len(self.NP.get_positions())
        # self.NP.center(about=(0.0,0.0,0.0))
        self.cog = self.NP.get_center_of_mass()
        if self.trPlanes is not None: self.trPlanes = pyNMBu.setdAsNegative(self.trPlanes)
        if self.jmolCrystalShape: self.jMolCS = pyNMBu.defCrystalShapeForJMol(self,noOutput)
        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")

    def predefinedParameters4WulffShapes(self,noOutput):
        # if not noOutput: vID.centertxt("List of pre-defined Wulff shapes in pyNanoMatBuilder",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        # if not noOutput: display(data.WulffShapes.WSdf)
        if self.WulffShape in data.WulffShapes.WSdf.index:
            self.eSurfacesWulff = data.WulffShapes.WSdf["relative energies"].loc[self.WulffShape]
            self.surfacesWulff = data.WulffShapes.WSdf["planes"].loc[self.WulffShape]
            self.symWulff = data.WulffShapes.WSdf["apply symmetry"].loc[self.WulffShape]
            if not noOutput: 
                print(f"{hl.BOLD}Selected shape{hl.OFF}")
                display(data.WulffShapes.WSdf.loc[self.WulffShape])
                if data.WulffShapes.WSdf["lattice system"].loc[self.WulffShape] != self.ucBL.lattice_system:
                    print(f"{bg.DARKREDB}The expected lattice system of this Wulff shape is ",
                          f"{data.WulffShapes.WSdf['lattice system'].loc[self.WulffShape]}.\n",
                          f"It does not correspond to the {self.ucBL.lattice_system} cif lattice. The Wulff pre-defined [h,k,l] indexes are meaningless.\n",
                          f"{fg.RED}{bg.LIGHTREDB}I hope you know what you are doing{bg.OFF}")
                else:
                    print(f"{bg.LIGHTGREENB}The lattice system of this Wulff shape ",
                          f"({data.WulffShapes.WSdf['lattice system'].loc[self.WulffShape]}) matches with ",
                          f"the {self.ucBL.lattice_system} cif lattice{bg.OFF}")
        else:
            display(data.WulffShapes.WSdf)
            sys.exit(f"{fg.RED}{bg.LIGHTREDB}The {self.WulffShape} Wulff shape is not a pre-defined shortcut "\
                    "in the WSdf dataframe (see data.py and the index column of the table just above){fg.OFF}")
  
   
    def prop(self,noOutput):
        
        #pyNMBu.plotImageInPropFunction(self.imageFile)
        vID.centertxt("Unit cell properties",bgc='#007a7a',size='14',weight='bold')
        pyNMBu.print_ase_unitcell(self)
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        if "Wulff" in self.shape :
            distances = np.linalg.norm(self.NP.positions- self.cog, axis=1)
            self.radiusCircumscribedSphere= np.max(distances)
            self.radiusInscribedSphere= np.min(distances)
            self.dim[0]= self.radiusCircumscribedSphere      # main dimensions for files 
            self.dim[1]= self.dim[0]
            self.dim[2]= self.dim[0]
            if not noOutput : 
                print(f"diameters of the circumscribed sphere: {self.radiusCircumscribedSphere * 2* 0.1:.2f}  {self.radiusCircumscribedSphere* 2 * 0.1:.2f}  {self.radiusCircumscribedSphere* 2* 0.1:.2f} nm")
                print(f"diameters of the inscribed sphere: { self.radiusInscribedSphere* 2* 0.1:.2f}  {self.radiusInscribedSphere* 2 * 0.1:.2f}  {self.radiusInscribedSphere * 2* 0.1:.2f} nm")
               

    def propPostMake(self,skipSymmetryAnalyzis, thresholdCoreSurface, noOutput): #accès aux faces voisines etc
        
        self.moi=pyNMBu.moi(self.NP, noOutput)
        self.moisize=np.array(pyNMBu.moi_size(self.NP, noOutput))# MOI mass normalized (m of each atoms=1)

        # find the size using the MOI mass normalized
        if self.shape == 'ellipsoid': # https://scienceworld.wolfram.com/physics/MomentofInertiaEllipsoid.html
            self.dim[0] = 2*np.sqrt((5 *self.moisize[1] + 5 * self.moisize[2] - 5 * self.moisize[0]) / 2)
            self.dim[1] = 2*np.sqrt((5 * self.moisize[0] + 5 * self.moisize[2] - 5 * self.moisize[1]) / 2)
            self.dim[2] = 2*np.sqrt((5 * self.moisize[0] + 5 * self.moisize[1] - 5 * self.moisize[2]) / 2)
            if not noOutput:
                print(f"Diameters of the ellipsoid  { self.dim[0]* 0.1:.2f}  { self.dim[1] * 0.1:.2f}  { self.dim[2] * 0.1:.2f} nm")
        if self.shape == 'sphere': #wikipedia
            self.dim[0] = 2 * np.sqrt(5 / 2 * self.moisize[0])  # même formule pour les 3 directions avec Ix, Iy, Iz égaux
            self.dim[1] = 2 * np.sqrt(5 / 2 * self.moisize[0])
            self.dim[2] = 2 * np.sqrt(5 / 2 * self.moisize[0])
            if not noOutput:
                print(f"Diameter of the sphere = {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm")
        if self.shape == 'cylinder': #wikipedia
            self.dim[0] = np.sqrt(12 * self.moisize[1] - 6 * self.moisize[0]) #longest distance
            self.dim[1] = 2 * np.sqrt(2 * self.moisize[0]) 
            self.dim[2] = 2 * np.sqrt(2 * self.moisize[0])
            if not noOutput:
                print(f"Size of the cylinder= {self.dim[0] * 0.1:.2f} {self.dim[1] * 0.1:.2f} {self.dim[2] * 0.1:.2f} nm")
                
        if self.shape == 'parallepiped': #wikipedia
            self.dim[0] = np.sqrt(6 *(self.moisize[1] + self.moisize[2] - self.moisize[0])) #longest distance
            self.dim[1] = np.sqrt(6 * (self.moisize[0] + self.moisize[2] - self.moisize[1])) # 2nd longest distance
            self.dim[2] = np.sqrt(6 * (self.moisize[0] + self.moisize[1] - self.moisize[2])) # 3rd longest distance
            if not noOutput:
                print(f"Size of the parallepiped=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm")

        if self.shape == 'wire': #wikipedia
            if self.nRotWire==4 :
                self.dim[0]=np.sqrt(12*self.moisize[1]-6*self.moisize[0]) #longest distance
                self.dim[1]=np.sqrt(6*self.moisize[0])
                self.dim[2]=np.sqrt(6*self.moisize[0])
                if not noOutput:
                    print(f"Size of the wire=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm")
            if self.nRotWire==6 :
                self.dim[0]=np.sqrt(12*self.moisize[1]-6*self.moisize[0]) #longest distance
                self.dim[1]=2*np.sqrt(2*self.moisize[0])
                self.dim[2]=2*np.sqrt(2*self.moisize[0])
                if not noOutput:
                    print(f"Size of the wire=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm")

        #NEW WULFF PREDEFINED
        # if "Wulff" and "cube" in self.shape :
        #     self.dim[0] = np.sqrt(6*self.moisize[0])
        #     self.dim[1] = np.sqrt(6*self.moisize[1])
        #     self.dim[2] = np.sqrt(6*self.moisize[2])
        #     if not noOutput:
        #         print(f"Length of the cube  { self.dim[0]* 0.1:.2f}  { self.dim[1] * 0.1:.2f}  { self.dim[2] * 0.1:.2f} nm")


        # if "Wulff" and "Oh" in self.shape :
        #     a=np.sqrt(10*self.moisize[0]) #arete https://www.vcalc.com/collection/?uuid=1a8912a2-f145-11e9-8682-bc764e2038f2
        #     self.dim[0] =a*math.sqrt(2) #diameter of the circumscribed sphere
        #     self.dim[1] = self.dim[0]
        #     self.dim[2] = self.dim[0]
        #     #https://fr.wikipedia.org/wiki/Dod%C3%A9ca%C3%A8dre_r%C3%A9gulier#:~:text=Les%2020%20%C3%97%206%20%3D%2012,sur%20les%20faces%20du%20poly%C3%A8dre.
        #     if not noOutput:
        #         print(f"Size of the octahedron (diameter of the circumscribed sphere) :  { self.dim[0]* 0.1:.2f}  { self.dim[1] * 0.1:.2f}  { self.dim[2] * 0.1:.2f} nm")
        #         print(f"Edge  of the icosahedron:  { a* 0.1:.2f}   nm")


        # if "Wulff" and "hcpwire" in self.shape :
        #     self.dim[0]=np.sqrt(12*self.moisize[1]-6*self.moisize[0]) #longest distance
        #     self.dim[1]=2*np.sqrt(2*self.moisize[0])
        #     self.dim[2]=2*np.sqrt(2*self.moisize[0])
        #     if not noOutput:
               # print(f"Size of the wire=  {self.dim[0] * 0.1:.2f}  {self.dim[1] * 0.1:.2f}  {self.dim[2] * 0.1:.2f} nm") 
        

        if not skipSymmetryAnalyzis: pyNMBu.MolSym(self.NP, noOutput=noOutput)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pyNMBu.coreSurface(self,thresholdCoreSurface, noOutput=noOutput)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
       
        
        
