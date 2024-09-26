# nanostructure_scatter_tools.py
from debyecalculator import DebyeCalculator
from PyQt6.QtCore import QThread, QObject, pyqtSignal, QRunnable, QThreadPool
import numpy as np
import ase
from ase.io import write
import pyNanoMatBuilder.utils as pNMBu
from pyNanoMatBuilder import crystalNPs as cyNP
from pyNanoMatBuilder import platonicNPs as pNP
import os
import sys
import shutil

# NanostructureWorker is responsible for creating nanostructures in a separate thread
class NanostructureWorker(QObject):
    """
    Worker class for creating nanostructures in a separate thread.
    
    Attributes:
        finished (pyqtSignal): Signal emitted when nanostructure creation is complete.
        error (pyqtSignal): Signal emitted if an error occurs.
    """
    finished = pyqtSignal(object)  # Signal emitted when nanostructure creation is complete
    error = pyqtSignal(str)        # Signal emitted if an error occurs

    def __init__(self, structure_params):
        """
        Initialize the NanostructureWorker instance.
        
        Parameters:
            structure_params (dict): Dictionary containing structure parameters.
        """
        super().__init__()
        self.atom = structure_params.get('atom')  # Atomic species
        self.shape = structure_params.get('shape')  # Shape of the nanostructure
        self.distance = structure_params.get('distance')  # Atomic distance parameter
        self.size_0 = structure_params.get('size_0')  # Primary size parameter
        self.size_1 = structure_params.get('size_1') # Secondary size parameter (optional)

    def run(self):
        """
        Run the nanostructure creation in a separate thread.
        
        This method creates a nanostructure based on the provided parameters and saves it to a file.
        """
        try:
            result = None
            if self.shape == 'Sphere':
                # Create a spherical nanostructure
                self.MyNP = cyNP.Crystal(self.atom, size=[self.size_0], shape='sphere')
                result = self.MyNP.makeNP()
            elif self.shape == 'Icosahedron':
                # Create an icosahedral nanostructure
                self.MyNP = pNP.regIco(self.atom, self.distance, nShell=int(self.size_0),
                                       aseView=False, thresholdCoreSurface=0.,
                                       skipSymmetryAnalyzis=True, noOutput=True)
            # Save the structure to a file
            xyzfile = "MaNanoParticule.xyz"
            write(xyzfile, self.MyNP.NP)
            self.finished.emit(result)  # Emit finished signal with the result
        except Exception as e:
            self.error.emit(str(e))  # Emit error signal if an exception occurs


class DebyeWorker(QObject):
    """
    Worker class for performing Debye calculations in a separate thread.
    
    Attributes:
        debye_finished (pyqtSignal): Signal emitted when Debye calculation is complete.
        error (pyqtSignal): Signal emitted if an error occurs.
    """

    debye_finished = pyqtSignal(object)  # Signal emitted when Debye calculation is complete
    error = pyqtSignal(str)        # Signal emitted if an error occurs

    def __init__(self, debye_params):
        """
        Initialize the DebyeWorker instance.
        
        Parameters:
            debye_params (dict): Dictionary containing Debye calculation parameters.
        """
        super().__init__()
        self.file_path = debye_params.get('file_path')
        self.qmin = debye_params.get('q_min')  # Minimum Q value
        self.qmax = debye_params.get('q_max')  # Maximum Q value
        self.qstep = debye_params.get('q_step')  # Step size for Q
        self.rmin = debye_params.get('r_min')  # Minimum r value
        self.rmax = debye_params.get('r_max')  # Maximum r value
        self.rstep = debye_params.get('r_step')  # Step size for r
        self.biso = debye_params.get('biso')  # Isotropic B factor
        self.device = debye_params.get('device')  # Device ('cpu' or 'cuda')
        self.curve_type = debye_params.get('curve_type')  # Type of curve ('iq' or 'gr')
        self.scale = debye_params.get('scale')  # Scale factor

    def run(self):
        """
        Perform the Debye calculation.
        
        This method performs the Debye calculation using the provided parameters and emits the result.
        """
        try:
            # Perform the Debye calculation and emit results
            result = self.debye_calculation(self.file_path, self.qmin, self.qmax, self.qstep, self.rmin, self.rmax, self.rstep, self.biso, self.device, self.curve_type, self.scale)
            self.debye_finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def debye_calculation(self, file_path=None, qmin=None, qmax=None, qstep=None, rmin=None, rmax=None, rstep=None,  biso=None, device=None, curve_type=None, scale=None):
        """
        Perform Debye calculation using the DebyeCalculator class.
        
        Parameters:
            file_path (str): Path to the structure file.
            qmin (float): Minimum value of Q.
            qmax (float): Maximum value of Q.
            qstep (float): Step size for Q.
            rmin (float): Minimum value of r.
            rmax (float): Maximum value of r.
            rstep (float): Step size for r.
            biso (float): Isotropic B factor.
            device (str): Device to use ('cpu' or 'cuda').
            curve (str): Type of calculation ('iq' or 'gr').
        """
        if file_path is None:
            file_path = "MaNanoParticule.xyz"  # Default file if not provided

        calc = DebyeCalculator(qmin=qmin, qmax=qmax, qstep=qstep, rmin=rmin, rmax=rmax, rstep=rstep, biso=biso, device=device)
        if curve_type == 'iq':
            x_calc, y_calc = calc.iq(structure_source=file_path)
        elif curve_type == 'gr':
            x_calc, y_calc = calc.gr(structure_source=file_path)
        # Apply the scale factor

        y_calc *= scale
        return x_calc, y_calc

