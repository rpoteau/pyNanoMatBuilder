import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QSplashScreen, QProgressBar
from PyQt6.uic import loadUi
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QPixmap
import os
import pyqtgraph as pg

from debye_thread import DebyeThread
from html_script_jsmol import create_html_from_structure

class MyMainWindow(QMainWindow):
    """Main window class for the application."""

    def __init__(self):
        """Initialize the main window and set up the UI components and connections."""
        super().__init__()
        loadUi("interface23.ui", self)  # Load the UI design from the .ui file

        # Connect signals to slots
        self.comboBox_selection.currentIndexChanged.connect(self.update_selected_file)
        self.actionOpen_structure.triggered.connect(self.open_structure_file)
        self.pushButton_debye.clicked.connect(self.debye)
        self.actionSave_I_q.triggered.connect(self.save_I_q)  # Connect actionSave_I_q to save_I_q function

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

    def open_structure_file(self):
        """Open a file dialog to select structure files and update the combo box with the selected files."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Structure Files", "", "Structure Files (*.xyz *.pdb)")
        if file_paths:
            self.selected_files = file_paths
            self.comboBox_selection.clear()
            for file_path in self.selected_files:
                self.comboBox_selection.addItem(os.path.basename(file_path))
            self.update_selected_file()

    def update_selected_file(self):
        """Update the displayed file in the web view based on the selected file in the combo box."""
        index = self.comboBox_selection.currentIndex()
        if index >= 0 and index < len(self.selected_files):
            self.file_path = self.selected_files[index]
            create_html_from_structure(self.file_path)  # Generate the HTML file for the selected structure file
            # Get the absolute path of the generated HTML file
            directory = os.path.dirname(self.file_path)
            self.html_file_path = os.path.join(directory, "index.html")
            local_html_file_path = QUrl.fromLocalFile(self.html_file_path)
            self.webEngineView_structure.load(local_html_file_path)

    def debye(self):
        """Start the Debye calculation in a separate thread."""
        if hasattr(self, 'file_path'):
            self.pushButton_debye.setEnabled(False)  # Disable the button
            self.progressBar.show()
            self.progress_bar.setValue(0)  # Reset the progress bar
            self.progress_timer.start(100)  # Start the progress timer with an interval of 100 ms

            self.debye_thread = DebyeThread(self.file_path)
            self.debye_thread.debye_finished.connect(self.plot_debye)
            self.debye_thread.debye_finished.connect(self.save_I_q)
            self.debye_thread.debye_finished.connect(self.enable_debye_button)
            self.debye_thread.start()

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

    def enable_debye_button(self):
        """Re-enable the Debye calculation button."""
        self.pushButton_debye.setEnabled(True)
        self.progressBar.hide()

    def save_I_q(self, debye_array):
        """
        Save the Debye calculation results to a text file with two columns.
        The first column contains the diffusion vector (Q) values and the second column contains the intensity (I(Q)) values.
        Parameters:
        debye_array (list): A list of two arrays, where debye_array[0] contains Q values and debye_array[1] contains I(Q) values.
        """
        # Open a file dialog to choose the save location and file name
        file_path, _ = QFileDialog.getSaveFileName(self, "Save I(q)", "", "Text Files (*.txt);;All Files (*)")
        
        if file_path:
            # Save debye_array to the selected file
            with open(file_path, 'w') as file:
                file.write("Q\tI(Q)\n")  # Use tab-separated columns for better readability in a text file
                for q, i_q in zip(debye_array[0], debye_array[1]):
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
