import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QSplashScreen, QProgressBar
from PyQt6.uic import loadUi
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QPixmap
import os
import pyqtgraph as pg

from debye_thread import DebyeThread
from html_script_jsmol import create_html_from_structure
from window_wyz_creator import XYZCreatorDialog

class MyMainWindow(QMainWindow):
    """Main window class for the application."""

    def __init__(self):
        """Initialize the main window and set up the UI components and connections."""
        super().__init__()
        loadUi("interface27.ui", self)  # Load the UI design from the .ui file

        # Connect signals to slots
        # self.comboBox_selection.currentIndexChanged.connect(self.update_selected_file)
        self.actionOpen_structure.triggered.connect(self.open_structure_file)
        self.pushButton_debye.clicked.connect(self.debye)
        self.pushButton_canceldebye.clicked.connect(self.cancel_debye)
        self.actionSave_I_q.triggered.connect(self.save_I_q)  # Connect actionSave_I_q to save_I_q function
        self.actionNew_Structure_File.triggered.connect(self.open_dialog_xyz_creator)
        self.actionNew_Structure_File.triggered.connect(self.open_dialog_xyz_creator)

        self.selected_files = []  # List to store the paths of selected XYZ or PDB files
        self.html_file_path = None  # Variable to store the path of the generated HTML file

        # Set up the plot widget
        self.plot_widget = self.findChild(pg.PlotWidget, 'plot_widget')
        self.plot_widget.setBackground('w')

        # Configure log-log scale
        self.plot_widget.getAxis('bottom').setLogMode(True)
        self.plot_widget.getAxis('left').setLogMode(True)
        self.plot_widget.setLogMode(x=True, y=True)
        
        # Enable auto-scaling
        self.plot_widget.enableAutoRange('xy', True)

        # Define a list of colors to cycle through
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        self.color_index = 0

        # Set labels and title
        self.plot_widget.setLabel('left', 'Intensity')
        self.plot_widget.setLabel('bottom', 'Diffusion Vector')
        self.plot_widget.setTitle('Calculated I(q)')

        # Add legend to the plot
        self.legend = self.plot_widget.addLegend()

        # Set up the progress bar
        self.progressBar.hide()
        self.progress_bar = self.findChild(QProgressBar, 'progressBar')
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.progress_value = 0

        self.debye_array = None  # Initialize debye_array to store the Debye calculation results

    def open_dialog_xyz_creator(self):
        dialog = XYZCreatorDialog()
        dialog.file_saved.connect(self.load_new_xyz_file)  # Connect the signal to the slot
        dialog.exec()

    def load_new_xyz_file(self, file_path):
        """Slot to handle loading the new XYZ file."""
        self.file_path = file_path
        self.update_selected_file()

    def open_structure_file(self):
        """Open a file dialog to select a structure file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Structure File", "", "Structure Files (*.xyz *.pdb)")
        if file_path:
            self.file_path = file_path
            self.update_selected_file()
            self.label_file_display.setText(os.path.basename(file_path))
            # os.path.basename(self.file_path).
            

    def update_selected_file(self):
        """Update the displayed file in the web view based on the selected file."""
        if self.file_path:
            create_html_from_structure(self.file_path)  # Generate the HTML file for the selected structure file
            # Get the absolute path of the generated HTML file
            directory = os.path.dirname(self.file_path)
            self.html_file_path = os.path.join(directory, "index.html")
            local_html_file_path = QUrl.fromLocalFile(self.html_file_path)
            self.webEngineView_structure.load(local_html_file_path)

            
    def debye(self):
        """Start the Debye calculation in a separate thread."""
        if hasattr(self, 'file_path'):
            # Retrieve values from UI
            qmin = self.doubleSpinBox_qmin.value()
            qmax = self.doubleSpinBox_qmax.value()
            qstep = self.doubleSpinBox_qstep.value()
            biso = self.doubleSpinBox_biso.value()
            device = 'cuda' if self.radioButton_gpu.isChecked() else 'cpu'

            self.pushButton_debye.setEnabled(False)  # Disable the button
            self.progressBar.show()
            self.progress_bar.setValue(0)  # Reset the progress bar
            self.progress_timer.start(100)  # Start the progress timer with an interval of 100 ms

            self.debye_thread = DebyeThread(self.file_path, qmin, qmax, qstep, biso, device)
            self.debye_thread.debye_finished.connect(self.plot_debye)
            self.debye_thread.debye_finished.connect(self.enable_debye_button)
            self.debye_thread.start()

    def cancel_debye(self):
        """Cancel the ongoing Debye calculation."""
        if hasattr(self, 'debye_thread'):
            self.debye_thread.stop()  # Set the flag to stop the thread
            # self.debye_thread.wait()  # Wait for the thread to finish

            self.pushButton_debye.setEnabled(True)  # Re-enable the button
            self.progressBar.hide()  # Hide the progress bar
    def update_progress_bar(self):
        """Update the progress bar value."""
        self.progress_value = (self.progress_value + 1) % 101  # Increment and wrap around at 100
        self.progress_bar.setValue(self.progress_value)

    def plot_debye(self, debye_array):
        """Plot the Debye calculation results on the plot widget."""
        # Stop the progress timer
        self.progress_timer.stop()
        self.progress_bar.setValue(100)  # Set the progress bar to 100% when done

        # Select the next color
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1

        # Plot the new curve with the selected color
        plot_item = self.plot_widget.plot(debye_array[0], debye_array[1], pen=pg.mkPen(color=color, width=2))

        # Get the filename for the legend
        filename = os.path.basename(self.file_path)

        # Add the curve to the legend
        self.legend.addItem(plot_item, filename)

        # Save the array for future use in this class (save)
        self.debye_array = debye_array

    def enable_debye_button(self):
        """Re-enable the Debye calculation button."""
        self.pushButton_debye.setEnabled(True)
        self.progressBar.hide()

    def save_I_q(self):
        """
        Save the Debye calculation results to a text file with two columns.
        The first column contains the diffusion vector (Q) values and the second column contains the intensity (I(Q)) values.
        """
        if self.debye_array is not None:
            # Open a file dialog to choose the save location and file name
            file_path, _ = QFileDialog.getSaveFileName(self, "Save I(q)", "", "Text Files (*.txt);;All Files (*)")
            
            if file_path:
                # Save debye_array to the selected file
                with open(file_path, 'w') as file:
                    file.write(os.path.basename(self.file_path))
                    file.write('\n')
                    file.write("Q\tI(Q)\n")  # Use tab-separated columns for better readability in a text file
                    for q, i_q in zip(self.debye_array[0], self.debye_array[1]):
                        file.write(f"{q}\t{i_q}\n")
                
                print(f"Debye data saved to {file_path}")

    def closeEvent(self, event):
        """Handle the close event to clean up resources, such as deleting the generated HTML file."""
        if self.html_file_path and os.path.exists(self.html_file_path):
            os.remove(self.html_file_path)
            print(f"HTML file removed: {self.html_file_path}")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Show the splash screen immediately
    splash_pix = QPixmap('splash.png')
    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()  # Ensure the splash screen is shown immediately

    # Initialize the main window
    window = MyMainWindow()
    window.show()

    # Finish the splash screen
    splash.finish(window)

    # Start the application event loop
    sys.exit(app.exec())
