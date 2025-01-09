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
def plot_loglogiq(self, q, iq, title="Scattering Intensity", xlabel='Q [$Å^{-1}$]', ylabel='loglog(I(Q)) [a.u.]',figsize=(3,4),color='blue',linestyle='-'):
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
        ax.plot(q, np.log(np.log(iq)),color=color,linestyle=linestyle)
        ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
        ax.grid(alpha=0.2)
        ax.set_title(title)
        plt.show()
   
def plot_gr(self, r, gr, title="Reduced pair distribution function", xlabel='r [$Å$]', ylabel='loglog(G(r)) [a.u.]',figsize=(6, 3),color='blue',linestyle='-'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(r, gr,color=color,linestyle=linestyle)
    ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
    ax.grid(alpha=0.2)
    ax.set_title(title)
    plt.show()

def plot_sq(self, q, sq, title="Structure function", xlabel='Q [$Å^{-1}$]', ylabel='loglog(S(q)) [a.u.]',figsize=(6, 3),color='blue',linestyle='-'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(q, sq,color=color,linestyle=linestyle)
    ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
    ax.grid(alpha=0.2)
    ax.set_title(title)
    plt.show()
def plot_fq(self, q, fq, title="Reduced structure function", xlabel='Q [$Å^{-1}$]', ylabel='loglog(F(q)) [a.u.]',figsize=(6, 3),color='blue',linestyle='-'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(q, fq,color=color,linestyle=linestyle)
    ax.set(xlabel=xlabel, ylabel=ylabel, yticks=[])
    ax.grid(alpha=0.2)
    ax.set_title(title)
    plt.show()



# Generalize it 
def create_iqfiles_from_xyzfiles(self, path_of_xyzfiles, path_of_csvfiles):
    """
    Calculate and store q, iq values in csv files for multiple nanoparticules : 
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
    
               
                q, iq= self.iq(structure_source)
                print('q',len(q),'iq',len(iq))
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
                    # Write the number of atoms (first line)
                    csvfile.write(f"{line_numberatoms}\n")
                    # Write the metadata (second line)
                    csvfile.write(f"{line_metadata}\n")
                    # Write the q, iq data
                    np.savetxt(csvfile, data, delimiter=",",header="q,sq")
                print('New file created:', csv_file)
            
            except FileNotFoundError:
                print(f"Error: The file {filename} was not found.")
            except ValueError:
                print(f"Error: The file {filename} contains invalid data.")
            except Exception as e:
                print(f"Unexpected error with {filename}: {e}")
                            
        else:
            print(f"File format Error(not .xyz) : {filename}")



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

#old code# def create_grfiles_from_xyzfiles(self, path_of_xyzfiles, path_of_csvfiles):
#     """
#     Calculate and store r, gr values in csv files for multiple nanoparticules : 
#     The nanoparticules structures must be in specific format :  "nanoparticulename.xyz" and in a same directory (path_of_xyzfiles)
#     The csv files will be created in a same directory aswell (path_of_csvfiles)

#     Args:

#         path_of_xyzfiles: Path of the directory containing the .xyz files.
#         path_of_csvfiles: Path of the directory where the CSV file will be created.

#     Returns:
#         None
#     """
   
#     # Check if the directories exist 
#     if not os.path.isdir(path_of_xyzfiles):
#         raise FileNotFoundError(f"Directory '{path_of_xyzfiles}'does not exist.")
#     if not os.path.isdir(path_of_csvfiles):
#         raise FileNotFoundError(f"Directory '{path_of_csvfiles}' does not exist.")



#     # Creation of csv files for each xyz files   
#     for filename in os.listdir(path_of_xyzfiles): #loop on the xyz files
#         if filename.endswith(".xyz"): 
#             try :
#                 structure_source = os.path.join(path_of_xyzfiles, filename)
#                 r, gr= self.gr(structure_source)
#                 print('r',len(r),'gr',len(gr))
#                 if not len(r) == len(gr) and len(r) > 0:
#                     print(f"Invalid r, gr data in file: {filename}")
#                     continue

#                 # extract the name of the nanoparticle (name of the xyz file)
#                 base_name = os.path.splitext(os.path.basename(structure_source))[0]
        
#                 # Create the new file name
#                 csv_file_name = f"{base_name}_gr.csv"
#                 csv_file = os.path.join(path_of_csvfiles, csv_file_name)
        
#                 # Combine r and gr values into a single array
#                 data = np.column_stack((r, gr))
        
#                 # Save the data to the new CSV file
#                 np.savetxt(csv_file, data, delimiter=",", header="r,gr")
#                 print('nouveau fichier créé=',csv_file)
            
#             except FileNotFoundError:
#                 print(f"Error: The file {filename} was not found.")
#             except ValueError:
#                 print(f"Error: The file {filename} contains invalid data.")
#             except Exception as e:
#                 print(f"Unexpected error with {filename}: {e}")
                            
#         else:
#             print(f"File format Error(not .xyz) : {filename}")
