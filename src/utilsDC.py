import importlib

import os
# Helper functions for generating different particle models
from debyecalculator.utility.generate import (
    generate_substitutional_alloy_models,
    generate_periodic_plane_substitution,
    generate_spherical_particle,
)

# Libraries for timing and visualization
import time
import torch
import numpy as np
from tqdm.auto import tqdm, trange

# ASE for particle visualization
import ase
from ase.io import write
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols
# Matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 8, 'lines.linewidth': 1})


# Plotting functions
def plot_loglogiq( q, iq, title="Scattering Intensity", xlabel='Q [$Å^{-1}$]', ylabel='I(Q) [a.u.]',figsize=(3,4),color='blue',linestyle='-'):
        """
    
    Args:
        q (array-like): Q values.
        iq (array-like): I(Q) values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple): Figure size.
        color (str): Line color.
        linestyle (str): Line style.
       
    """
       
        fig, ax = plt.subplots(figsize=figsize)
        ax.loglog(q, iq,color=color,linestyle=linestyle)
        ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
        ax.grid(alpha=0.2)
        ax.set_title(title)
        plt.show()
   
def plot_gr(r, gr, title="Reduced pair distribution function", xlabel='r [$Å$]', ylabel='(G(r) [a.u.]',figsize=(6, 3),color='blue',linestyle='-'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(r, gr,color=color,linestyle=linestyle)
    ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
    ax.grid(alpha=0.2)
    ax.set_title(title)
    plt.show()

def plot_sq( q, sq, title="Structure function", xlabel='Q [$Å^{-1}$]', ylabel='loglog(S(q)) [a.u.]',figsize=(6, 3),color='blue',linestyle='-'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(q, sq,color=color,linestyle=linestyle)
    ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
    ax.grid(alpha=0.2)
    ax.set_title(title)
    plt.show()
def plot_fq( q, fq, title="Reduced structure function", xlabel='Q [$Å^{-1}$]', ylabel='loglog(F(q)) [a.u.]',figsize=(6, 3),color='blue',linestyle='-'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(q, fq,color=color,linestyle=linestyle)
    ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
    ax.grid(alpha=0.2)
    ax.set_title(title)
    plt.show()



# Generalize it 
def create_iqfiles_from_xyzfiles(self, path_of_xyzfiles, path_of_csvfiles,noOutput):
    """
    Convert XYZ files into CSV files containing q and I(q) values using DebyeCalculator.
    Note:
        - The XYZ files must be named in the format: "nanoparticle_name.xyz" and stored in the same directory.
        - The generated CSV files will have the format: "nanoparticle_name_DC_iq.csv" and will be created in the specified directory.


    Args:
        self : instance of the DebyeCalculator class
        path_of_xyzfiles: Path of the directory containing the .xyz files.
        path_of_csvfiles: Path of the directory where the CSV file will be created.
        noOutput (bool): If False, prints details of the process.
    Returns:
        int: Total number of CSV files created.
    """
    number_created_files=0
    
    # Check if the directories exist 
    if not os.path.isdir(path_of_xyzfiles):
        raise FileNotFoundError(f"Directory '{path_of_xyzfiles}'does not exist.")
    if not os.path.isdir(path_of_csvfiles):
        raise FileNotFoundError(f"Directory '{path_of_csvfiles}' does not exist.")

    # Loop through all files in the XYZ directory
    for filename in os.listdir(path_of_xyzfiles): 
        if filename.endswith(".xyz"):   # Ensure the file has the correct format
            if not noOutput :
                print('file used :',filename)
            structure_source = os.path.join(path_of_xyzfiles, filename)
            try :
                # Read the first two lines of the file (number of atoms and metadata)
                with open(structure_source,'r') as file :
                    i=0
                    line_numberatoms= None
                    line_metadata= None
                    for line in file :
                        i+=1
                        if i==1 :
                            line_numberatoms=line.strip()
                            
                        elif i==2 :
                            line_metadata=line.strip()
                            
                        else :
                            break
    
                # Compute q and I(q) values
                q, iq= self.iq(structure_source)
                
                if not noOutput :
                    print('q',len(q),'iq',len(iq))
                # Validate the computed data
                if not len(q) == len(iq) and len(q) > 0:
                    print(f"Invalid q, iq data in file: {filename}")
                    continue
    
                # extract the name of the nanoparticle (name of the xyz file)
                base_name = os.path.splitext(os.path.basename(structure_source))[0]
        
                # Create the new file name
                csv_file_name = f"{base_name}_DC_iq.csv"
                csv_file = os.path.join(path_of_csvfiles, csv_file_name)
        
                data = np.column_stack((q, iq))
               
                # Save the data to the new CSV file
                with open(csv_file, 'w') as csvfile:
                    number_created_files+=1
                    # Write the number of atoms (first line)
                    csvfile.write(f"{line_numberatoms}\n")
                    # Write the metadata (second line)
                    csvfile.write(f"{line_metadata}\n")
                    # Write the q, iq data
                    np.savetxt(csvfile, data, delimiter=",",header="q,sq")

                if not noOutput :
                    print(f'\n\033[1m New file created:{csv_file}\033[0m\n')
            
            except FileNotFoundError:
                print(f"Error: The file {filename} was not found.")
            except ValueError:
                print(f"Error: The file {filename} contains invalid data.")
            except Exception as e:
                print(f"Unexpected error with {filename}: {e}")
                            
        else:
            if not noOutput :
                print(f"File format Error(not .xyz) : {filename}")
    print('Total of csv files created',number_created_files) # Return the total number of created files


#adding cif files 
def create_sqfiles_from_xyzfiles(self, path_of_xyzfiles, path_of_csvfiles,radii):
    """
    Calculate and store q, sq values in csv files for multiple nanoparticules : 
    The nanoparticules structures must be in specific format :  "nanoparticulename.xyz" and in a same directory (path_of_xyzfiles)
    The csv files will be created in a same directory aswell (path_of_csvfiles)

    Args:

        path_of_xyzfiles: Path of the directory containing the .xyz and .cif files
        path_of_csvfiles: Path of the directory where the CSV file will be created.

    Returns:
        None
    """
    # Check if the directories exist 
    if not os.path.isdir(path_of_xyzfiles):
        raise FileNotFoundError(f"Directory '{path_of_xyzfiles}'does not exist.")
    if not os.path.isdir(path_of_csvfiles):
        raise FileNotFoundError(f"Directory '{path_of_csvfiles}' does not exist.")

    # Creation of csv files for each xyz files   
    for filename in os.listdir(path_of_xyzfiles):#loop on the xyz files
        # try :
        if filename.endswith(".xyz"): 
            print('file used :',filename)
            structure_source = os.path.join(path_of_xyzfiles, filename)
            with open(structure_source,'r') as file :
                i=0
                line_numberatoms= None
                line_metadata= None
                for line in file :
                    i+=1
                    if i==1 :
                        line_numberatoms=line.strip()
                        
                    elif i==2 :
                        line_metadata=line.strip() 
                    else :
                        break
            q, sq= self.sq(structure_source,radii)
            print('q',len(q),'sq',len(sq))
            if not len(q) == len(sq) and len(q) > 0:
                print(f"Invalid q, sq data in file: {filename}")
                continue

            # extract the name of the nanoparticle (name of the xyz file)
            base_name = os.path.splitext(os.path.basename(structure_source))[0]
            # Create the new file name
            csv_file_name = f"{base_name}_DC_sq.csv"
            csv_file = os.path.join(path_of_csvfiles, csv_file_name)
            data = np.column_stack((q, sq))

            # Save the data to the new CSV file
            with open(csv_file, 'w') as csvfile:
                # Write the number of atoms (first line)
                csvfile.write(f"{line_numberatoms}\n")
                # Write the metadata (second line)
                csvfile.write(f"{line_metadata}\n")
                # Write the q, sq data
                np.savetxt(csvfile, data, delimiter=",",header="q,sq")
            print('New file created:', csv_file)
            
        elif filename.endswith(".cif"):
            print('file used :',filename)
            #print('file used :',filename)
            structure_source = os.path.join(path_of_xyzfiles, filename) 
            q, sq= self.sq(structure_source,radii)
            print('q',len(q),'sq',len(sq))
            if not len(q) == len(sq) and len(q) > 0:
                print(f"Invalid q, sq data in file: {filename}")
                continue

            # extract the name of the nanoparticle (name of the xyz file)
            base_name = os.path.splitext(os.path.basename(structure_source))[0]
    
            # Create the new file name
            csv_file_name = f"{base_name}_DC_sq.csv"
            csv_file = os.path.join(path_of_csvfiles, csv_file_name)
            data = np.column_stack((q, sq))
           
            # Save the data to the new CSV file
            with open(csv_file, 'w') as csvfile:
                np.savetxt(csvfile, data, delimiter=",",header="q,sq")
            print('New file created:', csv_file)


        # except FileNotFoundError:
        #     print(f"Error: The file {filename} was not found.")
        # except ValueError:
        #     print(f"Error: The file {filename} contains invalid data.")
        # except Exception as e:
        #     print(f"Unexpected error with {filename}: {e}")
                        
        # else:
        #     print(f"File format Error(not .xyz or .cif) : {filename}")


def create_fqfiles_from_xyzfiles(self, path_of_xyzfiles, path_of_csvfiles):
    """
    Calculate and store q, fq values in csv files for multiple nanoparticules : 
    The nanoparticules structures must be in specific format :  "nanoparticulename.xyz" and in a same directory (path_of_xyzfiles)
    The csv files will be created in a same directory aswell (path_of_csvfiles)

    Args:

        path_of_xyzfiles: Path of the directory containing the .xyz files.
        path_of_csvfiles: Path of the directory where the CSV file will be created.

    Returns:
        None
    """
    # Check if the directories exist 
    if not os.path.isdir(path_of_xyzfiles):
        raise FileNotFoundError(f"Directory '{path_of_xyzfiles}'does not exist.")
    if not os.path.isdir(path_of_csvfiles):
        raise FileNotFoundError(f"Directory '{path_of_csvfiles}' does not exist.")

    # Creation of csv files for each xyz files   
    for filename in os.listdir(path_of_xyzfiles): #loop on the xyz files
        if filename.endswith(".xyz"): 
            print('file used :',filename)
            structure_source = os.path.join(path_of_xyzfiles, filename)
            try :
                with open(structure_source,'r') as file :
                    i=0
                    line_numberatoms= None
                    line_metadata= None
                    for line in file :
                        i+=1
                        if i==1 :
                            line_numberatoms=line.strip()
                            
                        elif i==2 :
                            line_metadata=line.strip()
                            
                        else :
                            break
    
               
                q, fq= self.fq(structure_source)
                print('q',len(q),'fq',len(fq))
                if not len(q) == len(fq) and len(q) > 0:
                    print(f"Invalid q, fq data in file: {filename}")
                    continue
    
                # extract the name of the nanoparticle (name of the xyz file)
                base_name = os.path.splitext(os.path.basename(structure_source))[0]
        
                # Create the new file name
                csv_file_name = f"{base_name}_DC_fq.csv"
                csv_file = os.path.join(path_of_csvfiles, csv_file_name)
        
                data = np.column_stack((q, fq))
               
                # Save the data to the new CSV file
                with open(csv_file, 'w') as csvfile:
                    # Write the number of atoms (first line)
                    csvfile.write(f"{line_numberatoms}\n")
                    # Write the metadata (second line)
                    csvfile.write(f"{line_metadata}\n")
                    # Write the q, fq data
                    np.savetxt(csvfile, data, delimiter=",",header="q,fq")
                print('New file created:', csv_file)
            
            except FileNotFoundError:
                print(f"Error: The file {filename} was not found.")
            except ValueError:
                print(f"Error: The file {filename} contains invalid data.")
            except Exception as e:
                print(f"Unexpected error with {filename}: {e}")
                            
        else:
            print(f"File format Error(not .xyz) : {filename}")


def create_grfiles_from_xyzfiles(self, path_of_xyzfiles, path_of_csvfiles):
    """
    Calculate and store r, gr values in csv files for multiple nanoparticules : 
    The nanoparticules structures must be in specific format :  "nanoparticulename.xyz" and in a same directory (path_of_xyzfiles)
    The csv files will be created in a same directory aswell (path_of_csvfiles)

    Args:

        path_of_xyzfiles: Path of the directory containing the .xyz files.
        path_of_csvfiles: Path of the directory where the CSV file will be created.

    Returns:
        None
    """
    # Check if the directories exist 
    if not os.path.isdir(path_of_xyzfiles):
        raise FileNotFoundError(f"Directory '{path_of_xyzfiles}'does not exist.")
    if not os.path.isdir(path_of_csvfiles):
        raise FileNotFoundError(f"Directory '{path_of_csvfiles}' does not exist.")

    # Creation of csv files for each xyz files   
    for filename in os.listdir(path_of_xyzfiles): #loop on the xyz files
        if filename.endswith(".xyz"): 
            print('file used :',filename)
            structure_source = os.path.join(path_of_xyzfiles, filename)
            try :
                with open(structure_source,'r') as file :
                    i=0
                    line_numberatoms= None
                    line_metadata= None
                    for line in file :
                        i+=1
                        if i==1 :
                            line_numberatoms=line.strip()
                            
                        elif i==2 :
                            line_metadata=line.strip()
                            
                        else :
                            break
    
               
                r, gr= self.gr(structure_source)
                print('r',len(r),'gr',len(r))
                if not len(r) == len(gr) and len(r) > 0:
                    print(f"Invalid r, gr data in file: {filename}")
                    continue
    
                # extract the name of the nanoparticle (name of the xyz file)
                base_name = os.path.splitext(os.path.basename(structure_source))[0]
        
                # Create the new file name
                csv_file_name = f"{base_name}_DC_gr.csv"
                csv_file = os.path.join(path_of_csvfiles, csv_file_name)
        
                data = np.column_stack((r, gr))
               
                # Save the data to the new CSV file
                with open(csv_file, 'w') as csvfile:
                    # Write the number of atoms (first line)
                    csvfile.write(f"{line_numberatoms}\n")
                    # Write the metadata (second line)
                    csvfile.write(f"{line_metadata}\n")
                    # Write the r, gr data
                    np.savetxt(csvfile, data, delimiter=",",header="r,gr")
                print('New file created:', csv_file)
            
            except FileNotFoundError:
                print(f"Error: The file {filename} was not found.")
            except ValueError:
                print(f"Error: The file {filename} contains invalid data.")
            except Exception as e:
                print(f"Unexpected error with {filename}: {e}")
                            
        else:
            print(f"File format Error(not .xyz) : {filename}")

