# window_fit.py

# Import necessary libraries
print('importing libraries')
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QFileDialog, QMainWindow
from PyQt6.uic import loadUi
import os
import sys
import pyqtgraph as pg
from PyQt6.QtCore import QThread, QUrl,QWaitCondition, QMutex, QThreadPool
 # Import worker classes
import numpy as np
import lmfit
import ase
from ase.io import write
import pyNanoMatBuilder.utils as pNMBu
from pyNanoMatBuilder import crystalNPs as cyNP
from pyNanoMatBuilder import platonicNPs as pNP
from html_script_jsmol import create_html_from_structure
from lmfit import Minimizer
import random
print('imports done')

from nanostructure_scatter_tools import NanostructureWorker, DebyeWorker 

class My_window(QMainWindow):
    """
    Fit experimental data
    """

    def __init__(self):
        """
        Initialize the dialog and set up connections.
        
        This method initializes the main window and sets up connections between UI elements and their corresponding methods.
        It also creates threads for nanostructure creation and Debye calculation, as well as synchronization objects for Debye calculation.
        """
        super().__init__()
        loadUi("fit2.ui", self)  # Load the UI from a .ui file

        # Connect UI elements to corresponding methods
        self.pushButton_open.clicked.connect(self.select_file)
        self.pushButton_compute.clicked.connect(self.start_creation)
        self.pushButton_debye.clicked.connect(self.start_debye_calculation)
        self.pushButton_clear_graph.clicked.connect(self.clear_graph)
        self.pushButton_refine.clicked.connect(self.get_fit_params)

        # Connect signals to update parameters when values change
        self.comboBox_atom.currentIndexChanged.connect(self.update_parameters)
        self.comboBox_shape.currentIndexChanged.connect(self.update_parameters)
        self.spinBox_distance.valueChanged.connect(self.update_parameters)
        self.spinBox_size_0.valueChanged.connect(self.update_parameters)
        self.spinBox_size_1.valueChanged.connect(self.update_parameters)
        self.spinBox_qmin.valueChanged.connect(self.update_parameters)
        self.spinBox_qmax.valueChanged.connect(self.update_parameters)
        self.spinBox_qstep.valueChanged.connect(self.update_parameters)
        self.spinBox_rmin.valueChanged.connect(self.update_parameters)
        self.spinBox_rmax.valueChanged.connect(self.update_parameters)
        self.spinBox_rstep.valueChanged.connect(self.update_parameters)
        self.spinBox_biso.valueChanged.connect(self.update_parameters)
        # spinbox and checkbox for refinement
        self.spinBox_scale_min.valueChanged.connect(self.update_parameters)
        self.spinBox_scale_max.valueChanged.connect(self.update_parameters)
        self.spinBox_distance_min.valueChanged.connect(self.update_parameters)
        self.spinBox_distance_max.valueChanged.connect(self.update_parameters)
        self.spinBox_size_0_min.valueChanged.connect(self.update_parameters)
        self.spinBox_size_0_max.valueChanged.connect(self.update_parameters)
        self.spinBox_biso_min.valueChanged.connect(self.update_parameters)
        self.spinBox_biso_max.valueChanged.connect(self.update_parameters)
        self.spinBox_qdamp_min.valueChanged.connect(self.update_parameters)
        self.spinBox_qdamp_max.valueChanged.connect(self.update_parameters)
        self.checkBox_scale_factor.stateChanged.connect(self.update_parameters)
        self.checkBox_distance.stateChanged.connect(self.update_parameters)
        self.checkBox_size_0.stateChanged.connect(self.update_parameters)
        self.checkBox_biso.stateChanged.connect(self.update_parameters)
        self.checkBox_qdamp.stateChanged.connect(self.update_parameters)

        # Initialize parameters
        self.structure_params, self.debye_params = self.get_params()
        self.colors = ['w', 'r', 'c', 'm', 'y', 'l']  # list of colors
        self.plots = []

        # Create threads for nanostructure creation and Debye calculation
        self.debye_thread = QThread()
        self.creation_thread = QThread()

        # Create synchronization objects for Debye calculation
        self.debye_calculation_finished = QWaitCondition()
        self.debye_calculation_mutex = QMutex()

    # PART 1: Display Experimental Graph
    def select_file(self):
        """
        Open a dialog to select the file and plot the data.
        
        This method opens a file dialog for the user to select an experimental data file.
        It then reads the file and plots the data.
        """
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "Open Experimental Data File", "", "Text Files (*.gr);;All Files (*)")

        if self.file_path:
            self.x_exp, self.y_exp = self.read_experimental_file(self.file_path)
            self.plot_graph(self.x_exp, self.y_exp, 'b', 2)  # Plot data

    def read_experimental_file(self, fichier):
        """
        Read and parse experimental data from a file.
        
        This method reads a file and extracts the experimental data.
        It returns the x and y values of the data.
        """
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

    def plot_graph(self, x, y, color, width, label=None):
        """
        Plot data with specified color and width.
        
        This method plots data with a specified color and width.
        It adds the plot to the list of plots.
        """
        pen = pg.mkPen(color=color, width=width)
        plot = self.plot_widget.plot(x, y, pen=pen)
        self.plots.append(plot)

    def clear_graph(self):
        """
        Clear the plotted graph.
        
        This method clears the plotted graph by removing all plots and resetting the plot widget.
        """
        for plot in self.plots:
            self.plot_widget.removeItem(plot)
        self.plots = []
        self.plot_widget.plotItem.clear()
        self.plot_widget.clear()

    # PART 2: Create Structure
    def get_params(self):
        """
        Retrieve parameters from the UI.
        
        This method retrieves the current parameters from the UI.
        It returns the structure and Debye parameters.
        """
        structure_params = {
            "atom": self.comboBox_atom.currentText(),
            "shape": self.comboBox_shape.currentText(),
            "distance": self.spinBox_distance.value(),
            "size_0": self.spinBox_size_0.value(),
            "size_1": self.spinBox_size_1.value()
        }

        debye_params = {
            "file_path": "MaNanoParticule.xyz",
            "q_min": self.spinBox_qmin.value(),
            "q_max": self.spinBox_qmax.value(),
            "q_step": self.spinBox_qstep.value(),
            "r_min": self.spinBox_rmin.value(),
            "r_max": self.spinBox_rmax.value(),
            "r_step": self.spinBox_rstep.value(),
            "biso": self.spinBox_biso.value(),
            "scale": self.spinBox_scale.value(),
            "device": "cpu",
            "curve_type": 'gr' if self.radioButton_gr.isChecked() else 'iq' if self.radioButton_iq.isChecked() else None
        }
        return structure_params, debye_params

    def update_parameters(self):
        """
        Update parameters and print to console.
        
        This method updates the current parameters and prints them to the console.
        """
        self.structure_params, self.debye_params = self.get_params()

    def start_creation(self):
        """
        Start nanostructure creation in a separate thread.
        
        This method starts the nanostructure creation process in a separate thread.
        It creates a worker instance and moves it to the thread.
        It then connects the worker signals to the corresponding methods and starts the thread.
        """
        structure_params, debye_params = self.get_params()
        self.worker = NanostructureWorker(structure_params)  # Create worker instance
        self.worker.moveToThread(self.creation_thread)  # Move worker to the thread
        # Connect worker signals to methods
        self.worker.finished.connect(self.on_creation_finished)
        self.worker.error.connect(self.on_error)
        # Connect thread started signal to worker's run method
        self.creation_thread.started.connect(self.worker.run)
        if not self.creation_thread.isRunning():
            self.creation_thread.start()  # Start the thread
        else:
            self.creation_thread.quit()
            self.creation_thread.wait()
            self.creation_thread.start()

    def on_creation_finished(self, result):
        """
        Handle the result of the nanostructure creation.
        
        This method handles the result of the nanostructure creation process.
        It stops the creation thread and waits for it to finish.
        It then displays the structure and starts the Debye calculation if necessary.
        """
        self.creation_thread.quit()  # Stop the thread
        self.creation_thread.wait()  # Wait for the thread to finish
        if hasattr(self, 'thread') and self.debye_thread.isRunning():
            self.debye_thread.quit()  # Stop the thread
            self.debye_thread.wait()  # Wait for the thread to finish
            del self.worker
        self.display_structure()
        if self.checkBox_structure_only.isChecked():
            pass
        else:
            self.start_debye_calculation(self.get_params())

    def on_error(self, error_message):
        """
        Handle errors during nanostructure creation.
        
        This method handles errors that occur during the nanostructure creation process.
        It prints the error message to the console.
        """
        print(f"Error: {error_message}")

    # PART 3: Handle Debye Calculation
    def on_debye_finished(self, result):
        """
        Handle the result of the Debye calculation.
        
        This method handles the result of the Debye calculation process.
        It unlocks the Debye calculation mutex and wakes up any waiting threads.
        It then displays the Debye calculation result.
        """
        self.debye_calculation_mutex.lock()
        self.x_calc, self.y_calc = result
        self.debye_calculation_mutex.unlock()
        self.debye_calculation_finished.wakeAll()
        self.display_debye_result(result)  # Display the result

    def start_debye_calculation(self, params=None, plot='True'):
        """
        Start Debye calculation in a separate thread.
        
        This method starts the Debye calculation process in a separate thread.
        It creates a worker instance and moves it to the thread.
        It then connects the worker signals to the corresponding methods and starts the thread.
        """
        if params is not None:
            structure_params, debye_params = self.get_params()
        else:
            structure_params, debye_params = params
        self.debye_worker = DebyeWorker(debye_params)  # Create a new worker instance
        self.debye_worker.moveToThread(self.debye_thread)  # Move worker to the thread
        # Connect worker signals to methods
        self.debye_worker.debye_finished.connect(self.on_debye_finished)
        self.debye_worker.error.connect(self.on_error)
        # Connect thread started signal to worker's run method
        self.debye_thread.started.connect(self.debye_worker.run)
        if not self.debye_thread.isRunning():
            self.debye_thread.start()  # Start the thread
        else:
            self.debye_thread.quit()
            self.debye_thread.wait()
            self.debye_thread.start()

    def display_debye_result(self, result=None):
        """
        Display the Debye calculation result.
        
        This method displays the result of the Debye calculation process.
        It clears the graph and plots the experimental data and the calculated data.
        It also plots the residuals.
        """
        self.clear_graph()
        if self.y_exp is not None:
            self.plot_graph(self.x_exp, self.y_exp, 'b', 2)

        if result is not None:
            self.x_calc, self.y_calc = result
            self.plot_graph(self.x_calc, self.y_calc, 'r', 2)
            residuals_array = self.y_exp - self.y_calc
            self.plot_graph(self.x_calc, 0.5 * residuals_array - 10, 'g', 2)

    def display_structure(self):
        """
        Display the 3D structure of the generated nanostructure in the web engine.
        
        This method displays the 3D structure of the generated nanostructure in the web engine.
        It generates an HTML file for the structure using `create_html_from_structure` and loads it into the WebEngineView widget.
        """
        # Generate the HTML file for the selected structure ('MaNanoParticule.xyz')
        create_html_from_structure("MaNanoParticule.xyz")
        # Join the current working directory with the relative path to 'index.html'
        self.html_file_path = os.path.join(os.getcwd(), 'index.html')
        # Convert the file path to a local URL that the web engine can load
        local_html_file_path = QUrl.fromLocalFile(self.html_file_path)
        # Load the generated HTML file into the WebEngineView widget for display
        self.webEngineView.load(local_html_file_path)

    # PART 4: Refinement (or fitting)
    def residuals(self, params):
        """
        Calculate the residuals for the fitting process.
        
        This method calculates the residuals for the fitting process.
        It starts the Debye calculation process and waits for the result.
        It then calculates the residuals and returns them.
        """
        # Pass the params object to the start_debye_calculation method
        self.start_debye_calculation(params)

        # Wait for the result
        self.debye_calculation_mutex.lock()
        while not hasattr(self, 'x_calc') or not hasattr(self, 'y_calc'):
            self.debye_calculation_mutex.unlock()
            self.debye_calculation_finished.wait()
            self.debye_calculation_mutex.lock()
        # Calculate the residuals
        residuals_array = self.y_exp - self.y_calc
        # Replace NaNs with 0
        residuals_array = np.nan_to_num(residuals_array, nan=0)
        self.debye_calculation_mutex.unlock()
        return residuals_array

    def get_fit_params(self):
        """
        Get the fit parameters and perform the fitting using the lmfit library.
        
        This method gets the fit parameters and performs the fitting using the lmfit library.
        It creates a Params object and adds the parameters to it.
        It then performs the fitting and updates the spinboxes with the refined parameters.
        """
        # Create a Params object
        params = lmfit.Parameters()
        # Add the parameters to the Params object
        params.add('scalefactor', value=self.spinBox_scale.value(), vary=self.checkBox_scale_factor.isChecked(), min=self.spinBox_scale_min.value(), max=self.spinBox_scale_max.value())
        params.add('distance', value=self.spinBox_distance.value(), vary=self.checkBox_distance.isChecked(), min=self.spinBox_distance_min.value(), max=self.spinBox_distance_max.value())
        params.add('size_0', value=self.spinBox_size_0.value(), vary=self.checkBox_size_0.isChecked(), min=self.spinBox_size_0_min.value(), max=self.spinBox_size_0_max.value())
        params.add('biso', value=self.spinBox_biso.value(), vary=self.checkBox_biso.isChecked(), min=self.spinBox_biso_min.value(), max=self.spinBox_biso_max.value())
        params.add('qdamp', value=self.spinBox_qdamp.value(), vary=self.checkBox_qdamp.isChecked(), min=self.spinBox_qdamp_min.value(), max=self.spinBox_qdamp_max.value())

        # Perform the fitting using the lmfit library
        result = lmfit.minimize(self.residuals, params, method='leastsq', iter_cb=self.iteration_callback, max_nfev=100)

        # Get the refined parameters
        refined_params = result.params.valuesdict()
        print("Refined parameters:")
        for key, value in refined_params.items():
            print(f"{key}: {value:.4f}")
            # Update the spinboxes with the refined parameters
        self.spinBox_scale.setValue(refined_params['scalefactor'])
        self.spinBox_distance.setValue(refined_params['distance'])
        self.spinBox_size_0.setValue(int(refined_params['size_0']))
        self.spinBox_biso.setValue(refined_params['biso'])

        # Update the structure parameters with the refined values
        self.structure_params, self.debye_params = self.get_params()

        # Stop and wait for the creation thread to finish
        if self.creation_thread.isRunning():
            self.creation_thread.quit()
            self.creation_thread.wait()

        # Continue with the new Debye calculation using the updated spinbox values
        self.start_debye_calculation(self.get_params())
        self.start_creation()

        # Calculate R²
        r2 = self.r2calc(self.y_exp, self.y_calc)
        self.label_R2.setText(str(r2))
        print('R² = ', r2)

    def iteration_callback(self, params, iter, resid, *args, **kwargs):
        """
        Callback function for the fitting iteration.
        
        This method is a callback function for the fitting iteration.
        It prints the current iteration number, parameters, and residual.
        """
        print(f"Iteration {iter}:")
        print("Parameters:")
        for key, value in params.valuesdict().items():
            print(f"{key}: {value:.4f}")
        print(f"Residual:")
        print(resid)
        print()

    def r2calc(self, y_exp, y_calc):
        """
        Calculate R².
        
        This method calculates R².
        It returns the R² value.
        """
        mean = np.mean(y_exp)
        TSS = np.sum((y_exp - mean) ** 2)
        RSS = np.sum((y_exp - y_calc) ** 2)
        return 1 - RSS / TSS

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = My_window()
    dialog.show()
    sys.exit(app.exec())