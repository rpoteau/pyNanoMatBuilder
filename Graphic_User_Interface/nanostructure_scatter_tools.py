# nanostructure_scatter_tools.py

# Import necessary modules and libraries
print("Importing debyecalculator")
from debyecalculator import DebyeCalculator
print("debyecalculator imported")
from PyQt6.QtCore import QThread, QObject, pyqtSignal
import numpy as np
import ase
from ase.io import write
import pyNanoMatBuilder.utils as pNMBu
from pyNanoMatBuilder import crystalNPs as cyNP
from pyNanoMatBuilder import platonicNPs as pNP
import os
import sys
print(os.getcwd())
cwd0 = './styles/'
sys.path.append(cwd0)

sys.stdout.reconfigure(encoding='utf-8')
import shutil

# NanostructureWorker is responsible for creating nanostructures in a separate thread
class NanostructureWorker(QObject):
    finished = pyqtSignal(object)  # Signal emitted when nanostructure creation is complete
    error = pyqtSignal(str)        # Signal emitted if an error occurs

    def __init__(self, atom, shape, size0, size1=None):
        super().__init__()
        self.atom = atom  # Atomic species for nanostructure
        self.shape = shape  # Shape of the nanostructure ('Sphere' or 'Icosahedron')
        self.size0 = size0  # Primary size parameter
        self.size1 = size1  # Secondary size parameter (optional)


    def run(self):
        """Run the nanostructure creation in a separate thread."""
        try:
            result = None
            if self.shape == 'Sphere':
                # Create a spherical nanostructure
                self.MyNP = cyNP.Crystal(self.atom, size=[self.size0], shape='sphere')
                result = self.MyNP.makeNP()
            elif self.shape == 'Icosahedron':
                # Create an icosahedral nanostructure
                self.MyNP = pNP.regIco(self.atom, self.size0, nShell=int(self.size1),
                                       aseView=False, thresholdCoreSurface=0.,
                                       skipSymmetryAnalyzis=True, noOutput=True)
            # Save the structure to a file
            xyzfile = "MaNanoParticule.xyz"
            write(xyzfile, self.MyNP.NP)
            self.finished.emit(result)  # Emit finished signal with the result
        except Exception as e:
            self.error.emit(str(e))  # Emit error signal if an exception occurs

# DebyeWorker performs Debye scattering calculations in a separate thread
class DebyeWorker(QObject):
    debye_finished = pyqtSignal(list)  # Signal emitted when Debye calculation is complete
    error = pyqtSignal(str)  # Signal emitted if an error occurs

    def __init__(self, file_path, qmin, qmax, qstep, biso, device):
        super().__init__()
        self.file_path = file_path  # Path to the file with nanostructure data
        self.qmin = qmin  # Minimum Q value
        self.qmax = qmax  # Maximum Q value
        self.qstep = qstep  # Step size for Q
        self.biso = biso  # Isotropic B factor
        self.device = device  # Device to use for calculation ('cpu' or 'cuda')
        self._is_running = True  # Flag to control the running state

    def run(self):
        """Perform the Debye calculation."""
        try:
            # Perform the Debye calculation and emit results
            print(self.file_path, self.qmin, self.qmax, self.qstep, self.biso, self.device, 'gr')
            debye_array = self.debye_calculation(self.file_path, self.qmin, self.qmax, self.qstep, self.biso, self.device, 'gr')
            self.debye_finished.emit(debye_array)
        except Exception as e:
            self.error.emit(str(e))  # Emit error signal if an exception occurs

    def debye_calculation(self, file_path=None, qmin=None, qmax=None, qstep=None, biso=None, device=None, curve=None):
        """
        Perform Debye calculation using the DebyeCalculator class.
        
        Parameters:
            file_path (str): Path to the structure file.
            qmin (float): Minimum value of Q.
            qmax (float): Maximum value of Q.
            qstep (float): Step size for Q.
            biso (float): Isotropic B factor.
            device (str): Device to use ('cpu' or 'cuda').
            curve (str): Type of calculation ('iq' or 'gr').
        """
        if file_path is None:
            file_path = "MaNanoParticule.xyz"  # Default file if not provided

        calc = DebyeCalculator(qmin=qmin, qmax=qmax, qstep=qstep, device=device, biso=biso)


        ###§ ATTENTION!!! x_calc est en fait q , même pour les g(r)  Comment convertir q en r???
        if curve == 'iq':
            x_calc, y_calc = calc.iq(structure_source=file_path)
        elif curve == 'gr':
            x_calc, y_calc = calc.gr(structure_source=file_path)

        return x_calc, y_calc

    def stop(self):
        """Stop the calculation."""
        self._is_running = False

# DebyeCalculationManager manages the thread for Debye calculation
class DebyeCalculationManager:
    def __init__(self, file_path, qmin, qmax, qstep, biso, device):
        self.thread = QThread()  # Create a new thread
        self.worker = DebyeWorker(file_path, qmin, qmax, qstep, biso, device)  # Create a worker instance

        self.worker.moveToThread(self.thread)  # Move worker to the new thread
        self.thread.started.connect(self.worker.start)  # Connect thread start signal to worker start method
        self.worker.debye_finished.connect(self.handle_result)  # Connect worker's finished signal to result handler
        self.thread.finished.connect(self.thread.quit)  # Connect thread finished signal to thread quit method

    def start(self):
        self.thread.start()  # Start the thread

    def stop(self):
        self.worker.stop()  # Stop the worker
        self.thread.quit()  # Quit the thread

    def handle_result(self, result):
        print("Debye calculation finished. Result:", result)
        # Handle the result here (e.g., update UI, save data)

