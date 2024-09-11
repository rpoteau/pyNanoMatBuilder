# window_fit.py

from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QFileDialog
from PyQt6.uic import loadUi
import os
import sys
import pyqtgraph as pg
from PyQt6.QtCore import QThread
from nanostructure_scatter_tools import NanostructureWorker, DebyeWorker
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

class Fit(QDialog):
    """
    Fit experimental data
    """
    
    def __init__(self):
        """Initialize the dialog and set up connections."""
        super().__init__()
        loadUi("fit.ui", self)  # Load the UI from a .ui file
        # Connect UI elements to corresponding methods
        self.pushButton_open.clicked.connect(self.select_file)
        self.pushButton_refine.clicked.connect(self.start_creation)
        self.pushButton_debye.clicked.connect(self.start_debye_calculation) 
        self.pushButton_clear_graph.clicked.connect(self.clear_graph) 
        # Connect signals to update parameters when values change
        self.comboBox_atom.currentIndexChanged.connect(self.update_parameters)
        self.comboBox_shape.currentIndexChanged.connect(self.update_parameters)
        self.spinBox_distance.valueChanged.connect(self.update_parameters)
        self.spinBox_size_0.valueChanged.connect(self.update_parameters)
        self.spinBox_size_1.valueChanged.connect(self.update_parameters)
        self.spinBox_qmin.valueChanged.connect(self.update_parameters)
        self.spinBox_qmax.valueChanged.connect(self.update_parameters)
        self.spinBox_qstep.valueChanged.connect(self.update_parameters)
        self.spinBox_biso.valueChanged.connect(self.update_parameters)
        
        # Read initial parameters (method to be implemented)
        self.get_params()

    # PART 1: Display Experimental Graph
    def select_file(self):
        """Open a dialog to select the file and plot the data."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Experimental Data File", "", "Text Files (*.gr);;All Files (*)")
        
        if file_path:
            x, y = self.read_experimental_file(file_path)
            self.plot_graph(x, y, 'b', 2)  # Plot data

    def read_experimental_file(self, fichier):
        """Read and parse experimental data from a file."""
        x = []
        y = []
        lire_donnees = False

        with open(fichier, 'r') as f:
            for ligne in f:
                if '#### start data' in ligne:
                    lire_donnees = True
                    continue
                if lire_donnees and not ligne.startswith('#'):
                    try:
                        colonnes = ligne.split()
                        if len(colonnes) == 2:
                            x.append(float(colonnes[0]))
                            y.append(float(colonnes[1]))
                    except ValueError:
                        continue
        return x, y

    def plot_graph(self, x, y, color, width):
        """Plot data with specified color and width."""
        pen = pg.mkPen(color=color, width=width)
        self.plot_widget.plot(x, y, pen=pen)

    def clear_graph(self):
        """Clear the plotted graph."""
        self.plot_widget.clear()

    # PART 2: Create Structure
    def get_params(self):
        """Retrieve parameters from the UI."""
        atom = self.comboBox_atom.currentText()
        shape = self.comboBox_shape.currentText()
        distance = self.spinBox_distance.value()
        size0 = self.spinBox_size_0.value()
        size1 = self.spinBox_size_1.value()
        qmin = self.spinBox_qmin.value()
        qmax = self.spinBox_qmax.value()
        qstep = self.spinBox_qstep.value()
        biso = self.spinBox_biso.value()
        print(f"Retrieved Params - Atom: {atom}, Shape: {shape}, Distance: {distance}, Size0: {size0}, Size1: {size1}, qmin: {qmin}, qmax: {qmax}, qstep: {qstep}, biso: {biso}")
        return atom, shape, distance, size0, size1, qmin, qmax, qstep, biso

    def update_parameters(self):
        """Update parameters and print to console."""
        atom, shape, distance, size0, size1, qmin, qmax, qstep, biso = self.get_params()
        print(f"Parameters updated: Atom={atom}, Shape={shape}, Distance={distance}, Size0={size0}, Size1={size1}, qmin={qmin}, qmax={qmax}, qstep={qstep}, biso={biso}")

    def start_creation(self):
        """Start nanostructure creation in a separate thread."""
        atom, shape, distance, size0, size1, qmin, qmax, qstep, biso = self.get_params()

        self.thread = QThread()  # Create a new thread
        self.worker = NanostructureWorker(atom, shape, size0, size1)  # Create worker instance
        self.worker.moveToThread(self.thread)  # Move worker to the thread

        # Connect worker signals to methods
        self.worker.finished.connect(self.on_creation_finished)
        self.worker.error.connect(self.on_error)

        # Connect thread started signal to worker's run method
        self.thread.started.connect(self.worker.run)
        self.thread.start()  # Start the thread

    def on_creation_finished(self, result):
        """Handle the result of the nanostructure creation."""
        self.thread.quit()  # Stop the thread
        self.thread.wait()  # Wait for the thread to finish
        self.display_result(result)  # Display the result

    def on_error(self, error_message):
        """Handle errors during nanostructure creation."""
        self.thread.quit()  # Stop the thread
        self.thread.wait()  # Wait for the thread to finish
        QMessageBox.critical(self, "Error", error_message)  # Show error message

    def display_result(self, result):
        """Display the created nanostructure result."""
        # Implement result display logic here

    # PART 3: Handle Debye Calculation
    def start_debye_calculation(self):
        """Start Debye calculation in a separate thread."""
        atom, shape, distance, size0, size1, qmin, qmax, qstep, biso = self.get_params()

        file_path = "MaNanoParticule.xyz"  # File generated by NanostructureWorker

        self.debye_thread = QThread()  # Create a new thread
        self.debye_worker = DebyeWorker(file_path, qmin, qmax, qstep, biso, "cpu")  # Create worker instance
        self.debye_worker.moveToThread(self.debye_thread)  # Move worker to the thread

        # Connect worker signals to methods
        self.debye_worker.debye_finished.connect(self.on_debye_finished)
        self.debye_worker.error.connect(self.on_error)

        # Connect thread started signal to worker's run method
        self.debye_thread.started.connect(self.debye_worker.run)
        self.debye_thread.start()  # Start the thread

    def on_debye_finished(self, result):
        """Handle the result of the Debye calculation."""
        self.debye_thread.quit()  # Stop the thread
        self.debye_thread.wait()  # Wait for the thread to finish
        self.display_debye_result(result)  # Display the result

    def display_debye_result(self, result):
        """Display the Debye calculation result."""
        #Q, I = result 
        #self.plot_graph(Q, I, 'r', 2)  # Plot Debye results
        x_calc, y_calc = result
        self.plot_graph(x_calc, y_calc, 'r', 2)
        print('qmin= ',qmin, 'qmax= ', qmax)

    def on_error(self, error_message):
        """Handle errors during Debye calculation."""
        self.debye_thread.quit()  # Stop the thread
        self.debye_thread.wait()  # Wait for the thread to finish
        QMessageBox.critical(self, "Error", error_message)  # Show error message

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = Fit()
    dialog.show()
    sys.exit(app.exec())
