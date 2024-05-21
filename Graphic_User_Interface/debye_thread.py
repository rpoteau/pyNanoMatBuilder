print("start import debye")
from PyQt6.QtCore import QThread, pyqtSignal
from debyecalculator import DebyeCalculator
print("finish import debye")
def debye_calculation(file_path):
    """
    Perform Debye calculation using the DebyeCalculator class.

    Parameters:
        file_path (str): Path to the structure file.

    Returns:
        tuple: Two lists, Q and I, representing the scattering vector and intensity.
    """
    # Create instance of DebyeCalculator class and specify the device to use ('cpu' for CPU or 'cuda' for GPU)
    calc = DebyeCalculator(qmin=0.001, qmax=20, qstep=0.001, device='cpu', biso=0)
    # Compute I(q)
    Q, I = calc.iq(structure_source=file_path)
    return Q, I



class DebyeThread(QThread):
    """QThread subclass to handle Debye calculations in a separate thread."""
    debye_finished = pyqtSignal(list)

    def __init__(self, file_path):
        """
        Initialize the thread with the file path for Debye calculation.

        Parameters:
            file_path (str): Path to the structure file.
        """
        super().__init__()
        self.file_path = file_path

    def run(self):
        """Perform the Debye calculation and emit the result."""
        debye_array = self.debye_calculation(self.file_path)
        self.debye_finished.emit(debye_array)

    def debye_calculation(self, file_path):
        """
        Perform Debye calculation using the DebyeCalculator class.

        Parameters:
            file_path (str): Path to the structure file.

        Returns:
            tuple: Two lists, Q and I, representing the scattering vector and intensity.
        """
        # Create instance of DebyeCalculator class and specify the device to use ('cpu' for CPU or 'cuda' for GPU)
        calc = DebyeCalculator(qmin=0.001, qmax=20, qstep=0.001, device='cpu', biso=0)
        # Compute I(q)
        Q, I = calc.iq(structure_source=file_path)
        return Q, I
