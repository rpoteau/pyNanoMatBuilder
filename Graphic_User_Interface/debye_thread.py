print("start import debye")
from PyQt6.QtCore import QThread, pyqtSignal
from debyecalculator import DebyeCalculator
print("finish import debye")




class DebyeThread(QThread):
    """QThread subclass to handle Debye calculations in a separate thread."""
    debye_finished = pyqtSignal(list)

    def __init__(self, file_path, qmin, qmax, qstep, biso, device):
        super().__init__()
        self.file_path = file_path
        self.qmin = qmin
        self.qmax = qmax
        self.qstep = qstep
        self.biso = biso
        self.device = device
        self._is_running = True 

    def run(self):
        """Perform the Debye calculation and emit the result."""
        debye_array = self.debye_calculation(self.file_path, self.qmin, self.qmax, self.qstep, self.biso, self.device)
        self.debye_finished.emit(debye_array)

    def debye_calculation(self, file_path, qmin, qmax, qstep, biso, device):
        """
        Perform Debye calculation using the DebyeCalculator class.

        Parameters:
            file_path (str): Path to the structure file.
            qmin (float): Minimum value of Q.
            qmax (float): Maximum value of Q.
            qstep (float): Step size for Q.
            biso (float): Isotropic B factor.
            device (str): Device to use ('cpu' or 'cuda').

        Returns:
            tuple: Two lists, Q and I, representing the scattering vector and intensity.
        """
        calc = DebyeCalculator(qmin=qmin, qmax=qmax, qstep=qstep, device=device, biso=biso)
        Q, I = calc.iq(structure_source=file_path)
        return Q, I
    def stop(self):
        """Stop the thread."""
        self._is_running = False