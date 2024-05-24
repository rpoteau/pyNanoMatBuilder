# vID.init(cwd0)
# import sys
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QProgressBar, QFileDialog
from PyQt6.uic import loadUi
# import os
# import numpy as np
# from ase.io import write
# from pyNanoMatBuilder import crystalNPs as cyNP

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
print(os.getcwd())
# Set the correct path to the styles directory
# cwd0 = '/home-local/ratel-ra/anaconda3_c/envs/py311/lib/python3.11/site-packages/styles/'
cwd0 = 'C:/Users/cayez/Anaconda3/envs/pynanomatbuilder2/Lib/site-packages/styles/'
# Add the path to the system path
sys.path.append(cwd0)

import visualID as vID
from visualID import fg, hl, bg
# Initialize visualID with the correct path
vID.init(cwd0)

import numpy as np
import ase
from ase.io import write
# Comment out the next line to prevent Tkinter window from opening
# from ase.visualize import view
import pyNanoMatBuilder.utils as pNMBu

# Build nanosphere Ru-hcp radius=4nm
from pyNanoMatBuilder import crystalNPs as cyNP
from pyNanoMatBuilder import platonicNPs as pNP

from PyQt6.QtCore import QThread, pyqtSignal

import shutil

class Worker(QThread):
    """
    Worker thread for creating nanostructures and writing XYZ files.

    Args:
        atom (str): The type of atom.
        size (float): The size of the nanos.

    Signals:
        finished (str): Signal emitted upon successful completion with the filename.
        error (str): Signal emitted if an error occurs, with the error message.
    """

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, atom,shape, size0, size1,size2):
        super().__init__()
        self.atom = atom
        self.shape = shape
        self.size0 = size0
        self.size1 = size1
        self.size2 = size2


    def run(self):
        try:
            # Create the nanosphere
            self.create_structure()
            # Write the XYZ file
            xyzfile = "MaNanoParticule.xyz"
            self.writexyz(self.MyNP, xyzfile)
            # Emit the finished signal with the filename
            self.finished.emit(xyzfile)
        except Exception as e:
            # Emit the error signal with the error message
            self.error.emit(str(e))

    def create_structure(self):
        """Create the nanostructure."""


        if self.shape == 'Sphere':
        # vID.centerTitle(f"{self.atom} Sphere")
            self.MyNP = cyNP.Crystal(self.atom, size=[self.size0], shape = 'sphere')
            self.MyNP = self.MyNP.makeNP()

        if self.shape == 'Icosahedron':
            myIco = pNP.regIco(self.atom,self.size0, nShell=int(self.size1))
            # Auico=pNP.regIco('Au',2.7,nShell=4)
            #Auico=pNP.regDD('Au',2.7,nShell=4)
            self.MyNP,_=myIco.coords()




    def writexyz(self, atoms, filename):
        """Write the XYZ file."""
        element_array = atoms.get_chemical_symbols()
        composition = {}
        for element in element_array:
            if element in composition:
                composition[element] += 1
            else:
                composition[element] = 1

        coord = atoms.get_positions()
        natoms = len(element_array)
        line2write = '%d \n' % natoms
        line2write += '%s\n' % str(composition)

        for i in range(natoms):
            line2write += '%s' % str(element_array[i]) + '\t %.6f' % float(coord[i, 0]) + '\t %.6f' % float(coord[i, 1]) + '\t %.6f' % float(coord[i, 2]) + '\n'

        with open(filename, 'w') as file:
            file.write(line2write)

class XYZCreatorDialog(QDialog):
    """
    Dialog for creating nanospheres and writing XYZ files.

    Inherits from QDialog.

    Methods:
        __init__(self): Initialize the dialog.
        start_thread(self): Start the worker thread for creating the nano and writing the XYZ file.
        on_finished(self, xyzfile): Handle the finished signal from the worker thread.
        on_error(self, error_message): Handle the error signal from the worker thread.
    """
    file_saved = pyqtSignal(str) 
    def __init__(self):
        """Initialize the dialog."""
        super().__init__()
        loadUi("xyz_creator.ui", self)
        self.pushButton_create.clicked.connect(self.start_thread)
        self.comboBox_shape.currentIndexChanged.connect(self.on_shape_changed)

    def on_shape_changed(self, index):
        selected_shape = self.comboBox_shape.currentText()
        if selected_shape == "Sphere":
            print('sphere')

            self.stackedWidget_size0.setCurrentIndex(1)
            self.stackedWidget_labelSize0.setCurrentIndex(1)
            
            self.stackedWidget_labelSize1.setCurrentIndex(0)
            self.stackedWidget_size1.setCurrentIndex(0)

            self.stackedWidget_labelSize2.setCurrentIndex(0)
            self.stackedWidget_size2.setCurrentIndex(0)

            self.spinBox_Size0.setDecimals(1)

            self.stackedWidget_img_shape.setCurrentIndex(1)
            
        if selected_shape == "Icosahedron":
            print('ico')

            self.stackedWidget_size0.setCurrentIndex(1)
            self.stackedWidget_labelSize0.setCurrentIndex(2)
            
            self.stackedWidget_labelSize1.setCurrentIndex(1)
            self.stackedWidget_size1.setCurrentIndex(1)

            self.stackedWidget_labelSize2.setCurrentIndex(0)
            self.stackedWidget_size2.setCurrentIndex(0)
            self.spinBox_Size0.setDecimals(1)
            self.spinBox_Size1.setDecimals(0)

            self.stackedWidget_img_shape.setCurrentIndex(2)

    def start_thread(self):
        """Start the worker thread for creating the nanosphere and writing the XYZ file."""
        self.pushButton_create.setEnabled(False)  # Disable the button
        atom = self.comboBox_atom.currentText()
        shape = self.comboBox_shape.currentText()
        size0 = self.spinBox_Size0.value()
        size1 = self.spinBox_Size1.value()
        size2 = self.spinBox_Size2.value()

        self.thread = Worker(atom,shape, size0, size1, size2)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)

        self.thread.start()

    def on_finished(self, xyzfile):
        # """Handle the finished signal from the worker thread."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save XYZ File", "", "XYZ Files (*.xyz);;All Files (*)")
        
        if file_path:
            try:
                shutil.copy(xyzfile, file_path)
                QMessageBox.information(self, "Success", f"{xyzfile} has been created and saved successfully.")
                
                self.file_saved.emit(file_path)  # Emit the signal with the saved file path
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save the file: {str(e)}")
               
        else:
            QMessageBox.warning(self, "Warning", "Save operation was cancelled.")
           
        self.pushButton_create.setEnabled(True)  # Re-enable the button

    def on_error(self, error_message):
        """Handle the error signal from the worker thread."""
        QMessageBox.critical(self, "Error", error_message)
        self.pushButton_create.setEnabled(True)  # Re-enable the button

    def closeEvent(self, event):
        print("Dialog close event triggered")
        self.close()
        event.accept() # if not need to click 2 times x to close

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = XYZCreatorDialog()
    dialog.show()

    sys.exit(app.exec())