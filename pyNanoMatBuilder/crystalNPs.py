import sys
import math
import numpy as np
import pyNanoMatBuilder.utils as pNMBu
from ase.build import bulk
from ase import io
from ase.visualize import view
from ase.build.supercells import make_supercell
from ase.geometry import cellpar_to_cell

import visualID as vID

class Crystal:
    
    def __init__(self,
                 crystal: str='Au',
                 shape: str='sphere',
                 size: float=[2],
                 direction: float=[1,1,1],
                 dbFolder: str='cif_database'):
        self.dbFolder = dbFolder #database folder that contains cif files
        self.crystal = crystal # see list with the pNMBu.ciflist() command
        self.shape = shape # 'sphere', 'ellipsoid', 'cube'
        self.size = size
        self.direction = direction
        self.nAtoms = 0
        self.cif = None
        self.cifname = None
          
    def __str__(self):
        return(f"Crystal = {self.crystal} {self.shape}")

    def bulk(self,
             fformat: str='cif',
             crystal: str=None,
             pNMBlib: bool=True):
        '''
        input:
            - fformat = file format, either 'cif' or 'poscar'
            - crystal = name of the crystal
            - pNMBlib, Boolean = pypyNanoMatBuilder library of cf files (default: True). If False, pNMB will look for user-defined files
              in the default user folder
        returns:
            - cif or poscar file
            - name of the cif or POSCAR file
        '''
        import os

        path2cif = os.path.join(pNMBu.pNMB_location(),self.dbFolder)
        match self.crystal.upper():
            case "RU":
                self.cifname = "9008513_Ru_hcp.cif"
            case "PT":
                self.cifname = "9012957_Pt_fcc.cif"
            case "AU":
                self.cifname = "9008463_Au_fcc.cif"
            case _:
                sys.exit(f"The database does not contain bulk parameters for the {self.crystal} crystal.\nPlease provide parameters")
        self.cif = io.read(os.path.join(path2cif,self.cifname))
        return 

    def print_unitcell(self):
        unitcell = self.cif.cell.cellpar()
        print(f"a: {unitcell[0]:.3f} Å, b: {unitcell[1]:.3f} Å, c: {unitcell[2]:.3f} Å. (c/a = {unitcell[2]/unitcell[0]:.3f})")
        print(f"α: {unitcell[3]:.3f} °, β: {unitcell[4]:.3f} °, γ: {unitcell[5]:.3f} °")
        print()
        print(f"Bravais lattice: {self.cif.cell.get_bravais_lattice()}")
        print(f"Volume: {self.cif.cell.volume:.3f} Å3")
        print(f"Chemical formula: {self.cif.get_chemical_formula()}")

    def return_unitcell(self):
        unitcell = self.cif.cell.cellpar()
        V = cellpar_to_cell(unitcell)
        return unitcell, V

    def makeSuperCell(self):
        view(self.cif)
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
        #finds the nearest even numbers
        Ma = math.ceil(Ma / 2.) * 2
        Mb = math.ceil(Mb / 2.) * 2
        Mc = math.ceil(Mc / 2.) * 2
        print(f"Making a {Ma}x{Mb}x{Mc} supercell")
        M = [[Ma, 0, 0], [0, Mb, 0], [0, 0, Mc]]
        sc=make_supercell(self.cif,M)
        # print(cif.cell.cellpar())
        # print(cellpar_to_cell(cif.cell.cellpar()))
        # print(sc.cell.cellpar())
        # print(cellpar_to_cell(sc.cell.cellpar()))
        V = cellpar_to_cell(sc.cell.cellpar())
        print(f"Center of Mass: {sc.get_center_of_mass()} Å")
        print("Now translating the supercell")
        sc.translate(-V[0]/2)
        sc.translate(-V[1]/2)
        sc.translate(-V[2]/2)
        print(f"Center of Mass after translation of the supercell: {sc.get_center_of_mass()} Å")
        view(sc)
        return sc
        
    def makeSphere(self,sc):
        com = sc.get_center_of_mass()
        delAtom = []
        for atom in sc.positions:
            delAtom.extend(pNMBu.Rbetween2Points(com,atom)/10 > self.size)
        del sc[delAtom]
        view(sc)
        return sc
                
    def makeEllipsoid(self,sc):
        com = sc.get_center_of_mass()
        size = np.array(self.size)*10 #nm to angstrom
        def outside(coord,com,size):
            return (coord[0]-com[0])**2/(size[0])**2+(coord[1]-com[1])**2/(size[1])**2+(coord[2]-com[2])**2/(size[2])**2
        delAtom = []
        for atom in sc.positions:
            delAtom.extend([outside(atom,com,size) > 1])
        del sc[delAtom]
        view(sc)
        return sc
                
    def makeNP(self):
        import os
        print(self)
        # print(f"Crystal = {self.crystal}")
        chrono = pNMBu.timer(); chrono.chrono_start()
        print(self.bulk())

        self.bulk()
        
        vID.centertxt("Unit cell properties",bgc='#007a7a',size='14',weight='bold')
        print(f"cif parameters for {self.crystal} found in {self.cifname}")
        self.print_unitcell()

        vID.centertxt("Builder",bgc='#007a7a',size='14',weight='bold')
        if (self.size is None):
            self.length = [2,2,2]
            print(f"length parameter set up as = {self.size} nm")
        if (self.shape == "sphere"):
            print((f"Sphere radius = {self.size[0]} nm"))
            sc = self.makeSuperCell()
            NP = self.makeSphere(sc)
        elif (self.shape == "ellipsoid"):
            print((f"Ellipsoid radii = {self.size} nm"))
            sc = self.makeSuperCell()
            NP = self.makeEllipsoid(sc)
        elif (self.shape == "cube"):
            print((f"Cube side length = {self.size[0]} nm"))
        elif (self.shape == "rectangular cuboid"):
            print((f"Rectangular cuboid side lengths = {self.size} nm"))
        elif (self.shape == "supercell"):
            print((f"Supercell side length = {self.size} nm"))
            NP = self.makeSuperCell()
        elif (self.shape == "cylinder"):
            print((f"Cylinder in the {self.direction} direction. Length x width = {self.size[1]} x {self.size[0]} nm"))
        else:
            sys.exit("Shape {self.shape} is unknown")
        self.nAtoms=len(NP.get_positions())
        print(f"Total number of atoms = {self.nAtoms}")
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        return NP
