# External dependencies
import sys
import numpy as np

import ase
from ase.visualize import view

# Internal Relative Imports
from . import visualID as vID
from . import data
from . import utils as pyNMBu
from .utils import hl, fg, bg
from .pyNMBcore import pyNMBcore
from . import platonicNPs as pNP

class bch(pyNMBcore):
    """
    Boerdijk-Coxeter Helix (BCH) builder for FCC nanoparticles.

    This class generates a chiral assembly of regular FCC tetrahedrons. 
    It leverages the base `regfccTd` class to construct the initial 
    tetrahedral seed and then applies successive face reflections to 
    build a Boerdijk-Coxeter helix.

    Attributes:
        element (str): Chemical symbol of the atoms (e.g., 'Pt', 'Au').
        Rnn (float): Nearest-neighbor distance in Ångström.
        nLayer (int): Number of atomic layers in the unit tetrahedron.
        n_Td (int): Total number of tetrahedrons in the helical assembly.
        nAtoms_helix (int): Final count of unique atoms in the helix.
        NP (ase.Atoms): The final helical nanoparticle object.
    """

    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayerTd: int=1, 
                 n_Td: int = 15,
                 chirality = 'RH',
                 **kwargs
                ):
        """
        Initialize the Boerdijk-Coxeter helix by setting up the seed parameters.

        Args:
            element: Chemical element symbol.
            Rnn: Interatomic distance in Å.
            nLayerTd: Size of each tetrahedron (number of atoms per edge).
            n_Td: Total units to stack in the helix.
            chirality: right-(RH) or left-handed (LH)?
            **kwargs: Additional arguments passed to the parent classes 
                (e.g., postAnalyzis, aseView).
        """
        # Ensure the parent class generates exactly one tetrahedron as the seed
        super().__init__(**kwargs)
        self.element = element
        self.shape = 'bch'
        self.Rnn = Rnn
        self.nLayerTd = int(nLayerTd)
        self.n_Td = n_Td
        self.tdprop = pNP.regfccTd(self.element, self.Rnn, self.nLayerTd, noOutput=True, calcPropOnly=True)
        self.interLayerDistance = self.tdprop.interLayerDistance()
        self.nAtomsPerEdge = self.nLayerTd + 1
        self.dim = [0, 0, 0]
        self.nVertices = self.n_Td + 3
        self.nEdges = 3 * self.n_Td + 3
        self.nFaces = 2 * self.n_Td + 2
        self.area = self.tdprop.area()/4 * self.nFaces
        self.volume = self.tdprop.volume() * n_Td
        self.imageFile = pyNMBu.imageNameWithPathway("bch-C.png")
        if chirality not in ["RH", "LH"]:
            raise ValueError(
                f"Invalid chirality '{chirality}'. "
                "Must be either 'RH' (Right-Handed) or 'LH' (Left-Handed)."
            )
        self.chirality = chirality
        noOutput = self.noOutput
        if not noOutput:
            chiral_str = "Right-Handed"
            if chirality == "LH":
                chiral_str = "Left-Handed"
            pyNMBu.centerTitle(
                f"{n_Td} - {chiral_str} Boerdijk-Coxeter helix, made from an fcc tetrahedron with {self.nLayerTd} shells per tetrahedral unit"
            )

        if not self.calcPropOnly:
            self.coords(noOutput)
            if self.aseView:
                view(self.NP)
            if self.postAnalyzis:
                self.propPostMake(self.skipChiralityCalculation, self.skipSymmetryAnalyzis, self.thresholdCoreSurface, noOutput)
                if self.aseView:
                    view(self.NPcs)
        if not noOutput: self.prop()
                    
    def coords(self, noOutput: bool):
        """
        Generate the helical coordinates by assembling tetrahedral units.

        This method overrides the parent `coords` method. It first 
        generates the base tetrahedral unit, then calls the helical 
        assembly engine to reflect the unit across its faces in a 
        specific sequence to create chirality.

        Args:
            noOutput: If True, suppresses console prints during generation.
        """

        if not noOutput:
            pyNMBu.centertxt("Generation of coordinates", bgc='#007a7a', size='14', weight='bold')
        chrono = pyNMBu.timer()
        chrono.chrono_start()
        
        if not noOutput:
            pyNMBu.centertxt(
                "Generation of the coordinates of the fcc tetrahedron",
                bgc='#cbcbcb',
                size='12',
                fgc='b',
                weight='bold',
            )
        td = pNP.regfccTd(self.element, self.Rnn, self.nLayerTd, skipSymmetryAnalyzis=True, postAnalyzis=False, noOutput=True)
        self.NP0 = td.NP.copy()
        asetd = td.NP
        nAtoms = asetd.get_global_number_of_atoms()
        print(f"{nAtoms=}")

        # helical algorithm
        seed_c = asetd.get_positions()

        if self.n_Td > 1:
            if not noOutput: pyNMBu.centertxt("Making the helical geometry",bgc='#cbcbcb',size='12',fgc='b',weight='bold')        
            
            # Standard face definitions for a tetrahedron 
            # Consistent with the MakeVertices convention
            seed_faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
            if self.chirality == "RH":
                face_sequence = [0, 1, 2, 3]             
            elif self.chirality == "LH":
                face_sequence = [0, 3, 2, 1] 
            
            # The Boerdijk-Coxeter pattern cycles through the 4 faces
            
            # Call the optimized assembly engine
            helix_c, self.nAtoms_helix = pyNMBu.helical_assembly(
                seed_coords=seed_c,
                seed_faces=seed_faces,
                n_units=self.n_Td,
                face_sequence=face_sequence,
                debug=False
            )
            
            # Update the class with the full helical structure
            self.nAtoms = len(helix_c)
            helix_c = pyNMBu.center2cog(helix_c)
            # Reconstruct the ASE object with the full coordinate list
            self.NP = ase.Atoms(self.element * self.nAtoms, positions=helix_c)
            # Update the center of gravity for the entire helix
            self.cog = self.NP.get_center_of_mass()
        else:
            # For a single unit, helix count equals base count
            self.NP = self.NP.copy()
            self.nAtoms = nAtoms
            self.nAtoms_helix = self.nAtoms
        if not noOutput:
            chrono.chrono_stop(hdelay=False)
            chrono.chrono_show()
        print(self.nAtoms)
            
    def __str__(self):
        """String representation of the helical nanoparticle."""
        return (f"Boerdijk-Coxeter Helix: {self.n_Td} units of size "
                f"{self.nLayerTd} ({self.element})")
        
    def edgeLength(self):
        """
        Computes the edge length of the triangular platelet in Å..
        """
        return self.tdprop.edgeLength()
        
    def prop(self):
        """
        Display unit cell and nanoparticle properties.
        """
        pyNMBu.centertxt("Properties", bgc='#007a7a', size='14', weight='bold')
        print(self)
        pyNMBu.plotImageInPropFunction(self.imageFile, figsize=(10, 2), rot=90)
        print("element = ",self.element)
        print(f"Total units (tetrahedrons): {self.n_Td}")
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"Total atoms in helical assembly: {self.nAtoms}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length of the tetrahedrons used to create the helix = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge of the tetrahedron = {self.nAtomsPerEdge + 1}")
        print(f"inter-layer distance = {self.interLayerDistance:.2f} Å")
        print(f"area = {self.area * 1e-2:.1f} nm2")
        print(f"volume = {self.volume * 1e-3:.1f} nm3")
        print(f"coordinates of the center of gravity = {self.cog}")
        