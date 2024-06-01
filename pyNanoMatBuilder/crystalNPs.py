import sys, os, pathlib
import re
import math
import numpy as np
import pyNanoMatBuilder.utils as pNMBu
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
                 userDefCif: str=None,
                 shape: str='sphere',
                 size: float=[2,2,2], #nm
                 direction: float=[0,0,1],
                 refPlane: float=[1,0,0],
                 nRot: int=6,
                 pbc: bool=False,
                 threshold: float=1e-3,
                 dbFolder: str='cif_database',
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.dbFolder = dbFolder #database folder that contains cif files
        self.crystal = crystal # see list with the pNMBu.ciflist() command
        self.shape = shape # 'sphere', 'ellipsoid', 'cube', 'wire'
        self.size = size
        self.direction = direction
        self.refPlane = refPlane
        self.nRot = nRot
        self.pbc = pbc
        self.threshold = threshold
        self.nAtoms = 0
        self.cif = None
        self.cifname = None
        self.userDefCif = userDefCif
        if self.userDefCif is not None: self.loadExternalCif()

        match self.shape:
            case "sphere":
                self.imageFile = pNMBu.imageNameWithPathway("sphere-C.png")
            case "ellipsoid":
                self.imageFile = pNMBu.imageNameWithPathway("ellipsoid-C.png")
            case "wire":
                print("image does not exist yet")
                self.imageFile = pNMBu.imageNameWithPathway("underConstruction.png")
            case _:
                sys.exit("Shape {self.shape} is unknown")
        if not noOutput: vID.centerTitle(f"{self.crystal} {self.shape}")

        self.bulk(noOutput)
        if aseView: view(self.cif)
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.makeNP(noOutput)
            if aseView: 
                view(self.sc)
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)
          
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

    def print_unitcell(self):
        unitcell = self.cif.cell.cellpar()
        bl = self.cif.cell.get_bravais_lattice()
        print(f"Bravais lattice: {bl}")
        print(f"Chemical formula: {self.cif.get_chemical_formula()}")
        print(f"Crystal family = {bl.crystal_family} (lattice system = {bl.lattice_system})")
        print(f"Name = {bl.longname} (Pearson symbol = {bl.pearson_symbol})")
        print(f"Variant names = {bl.variant_names}")
        print()
        print(f"a: {unitcell[0]:.3f} Å, b: {unitcell[1]:.3f} Å, c: {unitcell[2]:.3f} Å. (c/a = {unitcell[2]/unitcell[0]:.3f})")
        print(f"α: {unitcell[3]:.3f} °, β: {unitcell[4]:.3f} °, γ: {unitcell[5]:.3f} °")
        print()
        print(f"Volume: {self.cif.cell.volume:.3f} Å3")

    def return_unitcell(self):
        unitcell = self.cif.cell.cellpar()
        V = cellpar_to_cell(unitcell)
        return unitcell, V

    def bulk(self, noOutput):

        if self.userDefCif is None:
            path2cif = os.path.join(pNMBu.pNMB_location(),self.dbFolder)
            match self.crystal.upper():
                case "NACL":
                    self.cifname = "cod1000041_NaCl.cif"
                case "TIO2 RUTILE":
                    self.cifname = "cod9015662-TiO2-rutile.cif"
                case "TIO2 ANATASE":
                    self.cifname = "cod90159291-TiO2-anatase.cif"
                case "RU":
                    self.cifname = "cod9008513_Ru_hcp.cif"
                case "PT":
                    self.cifname = "cod9012957_Pt_fcc.cif"
                case "AU":
                    self.cifname = "cod9008463_Au_fcc.cif"
                case _:
                    sys.exit(f"The database does not contain bulk parameters for the {self.crystal} crystal.\nPlease provide a cif file")
            self.cif = io.read(os.path.join(path2cif,self.cifname))
        else:
            self.cif = io.read(self.userDefCif)
            path2extCif = pathlib.Path(self.userDefCif)
            self.cifname = pathlib.Path(*path2extCif.parts[-1:])
        if not noOutput: print(f"cif parameters for {self.crystal} found in {self.cifname}")
        return 

    def makeSuperCell(self,noOutput):
        chrono = pNMBu.timer(); chrono.chrono_start()
        if not noOutput: vID.centertxt(f"Making a multiple cell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        extendSizeByFactor = 1.1
        if (self.shape == 'sphere'):
            # first calculate the size of the supercell
            sphereRadius = self.size[0]
            Ma = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'ellipsoid' or self.shape == 'supercell'):
            # first calculate the size of the supercell
            Ma = int(np.round(extendSizeByFactor * self.size[0]*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * self.size[1]*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * self.size[2]*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'wire'):
            if np.argmax(self.size) == 1:
                maxDim = self.size[1]
            else:
                maxDim = self.size[0]*1.5 # add space
            Ma = int(np.round(extendSizeByFactor * maxDim*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * maxDim*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * maxDim*10/self.cif.cell.lengths()[2]))
        #finds the nearest even numbers
        Ma = math.ceil(Ma / 2.) * 2
        Mb = math.ceil(Mb / 2.) * 2
        Mc = math.ceil(Mc / 2.) * 2
        if not noOutput: print(f"Making a {Ma}x{Mb}x{Mc} supercell")
        M = [[Ma, 0, 0], [0, Mb, 0], [0, 0, Mc]]
        sc=make_supercell(self.cif,M)
        # print(cif.cell.cellpar())
        # print(cellpar_to_cell(cif.cell.cellpar()))
        # print(sc.cell.cellpar())
        # print(cellpar_to_cell(sc.cell.cellpar()))
        V = cellpar_to_cell(sc.cell.cellpar())
        if not noOutput: print(f"Center of Mass: {sc.get_center_of_mass()} Å")
        if not noOutput: print("Now translating the supercell")
        sc.translate(-V[0]/2)
        sc.translate(-V[1]/2)
        sc.translate(-V[2]/2)
        if not noOutput: print(f"Center of Mass after translation of the supercell: {sc.get_center_of_mass()} Å")
        self.sc = sc.copy()
        nAtoms=len(self.sc.get_positions())
        if not noOutput: print(f"Total number of atoms = {nAtoms}")
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
    def makeSphere(self,noOutput):
        vID.centertxt(f"Removing atoms to make a sphere",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        com = self.sc.get_center_of_mass()
        sphereRadius = self.size[0]
        delAtom = []
        for atomCoord in self.sc.positions:
            delAtom.extend(pNMBu.Rbetween2Points(com,atomCoord)/10 > [sphereRadius])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
                
    def makeEllipsoid(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make an ellipsoid",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        com = self.sc.get_center_of_mass()
        size = np.array(self.size)*10 #nm to angstrom
        def outside(coord,com,size):
            return (coord[0]-com[0])**2/(size[0])**2+(coord[1]-com[1])**2/(size[1])**2+(coord[2]-com[2])**2/(size[2])**2
        delAtom = []
        for atom in self.sc.positions:
            delAtom.extend([outside(atom,com,size) > 1])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    def makeWire(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make a wire",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        if self.refPlane is None: self.refPlane = pNMBu.returnPlaneParallel2Line(self.direction,[1,0,0],debug=True)
        trPlanes = pNMBu.planeRotation(self,self.refPlane,self.direction,self.nRot)
        for i,p in enumerate(trPlanes):
            trPlanes[i] = pNMBu.normV(p)
        radius = 10*self.size[0]/2
        tradius = np.full((self.nRot,1),-radius)
        trPlanes = np.append(trPlanes,tradius,axis=1)
        if not self.pbc:
            halfLength = 10*self.size[1]/2
            ptop = np.append(pNMBu.normV(self.direction),-halfLength)
            pbottom = np.append(-pNMBu.normV(self.direction),-halfLength)
            trPlanes = np.append(trPlanes,ptop)
            trPlanes = np.append(trPlanes,pbottom)
            trPlanes = np.reshape(trPlanes,(self.nRot+2,4))
        AtomsAbovePlanes = pNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,eps=self.threshold)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        vID.centertxt(f"Nanowire moved to the center of the unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        self.NP.center()
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
                
    def makeNP(self,noOutput):
        import os
        if not noOutput: vID.centertxt("Builder",bgc='#007a7a',size='14',weight='bold')
        if (self.size is None):
            self.length = [2,2,2]
            if not noOutput: print(f"length parameter set up as = {self.size} nm")
        if (self.shape == "sphere"):
            if not noOutput: print((f"Sphere radius = {self.size[0]} nm"))
            self.makeSuperCell(noOutput)
            self.makeSphere(noOutput)
        elif (self.shape == "ellipsoid"):
            if not noOutput: print((f"Ellipsoid radii = {self.size} nm"))
            self.makeSuperCell(noOutput)
            self.makeEllipsoid(noOutput)
        elif (self.shape == "cube"):
            if not noOutput: print((f"Cube side length = {self.size[0]} nm"))
        elif (self.shape == "rectangular cuboid"):
            if not noOutput: print((f"Rectangular cuboid side lengths = {self.size} nm"))
        elif (self.shape == "supercell"):
            if not noOutput: print((f"Supercell side length = {self.size} nm"))
            if len(self.size) != 3: sys.exit("Please enter lengths along a,b and c axis, i.e. size=[l_a,l_b,l_c]")
            self.makeSuperCell(noOutput)
        elif (self.shape == "wire"):
            if not noOutput: print((f"Wire in the {self.direction} direction. Length x width = {self.size[1]} x {self.size[0]} nm"))
            if not noOutput: print((f"Reference plane = {self.refPlane}, {self.nRot}-th order rotation around {self.direction}"))
            if not pNMBu.isPlaneParrallel2Line(self.refPlane, self.direction):
                print(f"{fg.RED}Warning! The reference truncation plane is not parallel to {self.direction}. Are you sure?{fg.OFF}")
                suggestedPlane = pNMBu.returnPlaneParallel2Line(self.direction)
                print(f"Among other possibilities, you can try {suggestedPlane}")
            else:
                if not noOutput: print(f"{fg.GREEN}The reference truncation plane is parallel to {self.direction}{fg.OFF}")
            self.makeSuperCell(noOutput)
            self.makeWire(noOutput)
        self.nAtoms=len(self.NP.get_positions())
        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")

    def prop(self):
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        vID.centertxt("Unit cell properties",bgc='#007a7a',size='14',weight='bold')
        self.print_unitcell()

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
