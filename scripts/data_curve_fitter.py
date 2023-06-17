"""
This script processes and fits the cleaned trajectory data of particles using Legendre polynomials.

Description:
    The script reads configuration parameters from a config.ini file, loads cleaned particle trajectory data
    from an HDF5 file and then fits the 3D trajectories using Legendre polynomials. The calculated parameters
    such as the tangent, normal, binormal vectors, curvature, torsion, velocity, and acceleration are then
    stored back to an HDF5 file for each particle track. There is also an option to plot the original and fitted
    data, but it is turned off by default.

Dependencies:
    os, configparser, h5py, matplotlib, numpy, pandas, sys

Modules:
    curve_fitting: contains the curve_fit_legendre function which is used to fit the 3D trajectories of particles.

Usage:
    Run this script through the command line or an IDE. The script is intended to be executed as a standalone 
    file, and it uses values from a configuration file 'config.ini'.

Configuration File (config.ini):
    DEFAULT:
        data_directory : Path to the directory containing the data files.
    CLEANING:
        filename : Name of the file containing cleaned data.
    Fitting:
        window_size : Window size for the curve fitting.
        order : Order of the Legendre polynomial.
        energy_stretch : Binary flag for stretch energy.
        energy_bend : Binary flag for bend energy.
        energy_twist : Binary flag for twist energy.
        filename : Name of the output file to save the fitted data.

Output:
    The output is an HDF5 file containing the fitted trajectory data along with tangent, normal, binormal vectors,
    curvature, torsion, velocity, and acceleration for each particle track.
"""

import os
import configparser
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Append the root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary functions from modules
from curve_fitting import curve_fit_legendre

def main():
    
    # Read configuration file
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)

    ## Retrieve values from the configuration file
    
    # For loading cleanded data & setting up parameters
    
    data_directory = config.get('DEFAULT', 'data_directory')
    filename = config.get('CLEANING', 'filename')
    
    
    window_size = config.getint('Fitting', 'window_size')
    order = config.getint('Fitting', 'order')
    stretch_flag = config.getint('Fitting', 'energy_stretch')
    bend_flag = config.getint('Fitting', 'energy_bend')
    twist_flag = config.getint('Fitting', 'energy_twist')
    
    gamma_S = stretch_flag*0 
    gamma_B = bend_flag*0
    gamma_T = twist_flag*0
    # Define the file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    file_path = os.path.join(parent_dir, data_directory, filename)
    
    # Load the data as a pandas DataFrame
    # Load the data from the HDF5 file
    with h5py.File(file_path, 'r') as file:
        data = np.array(file['data'])

    # Convert the array to a DataFrame for easier manipulation
    columns = ['Particle_no', 'frame_no', 'x_position', 'y_position', 'z_position']
    cleaned_df = pd.DataFrame(data, columns=columns)
    
    unique_tracks = cleaned_df['Particle_no'].unique()
    
    #unique_tracks = unique_tracks[1:2]
    
    output_filename = config.get('Fitting', 'filename')
    output_file_path = os.path.join(parent_dir, data_directory,output_filename)
    with h5py.File(output_file_path, 'w') as file:       
    
        # Iterate over the tracks
        for track_no in unique_tracks:
            print(f"Track: {track_no}")
            track_data = cleaned_df[cleaned_df['Particle_no'] == track_no]
            sorted_track_data = track_data.sort_values(by='frame_no')
            xdata = sorted_track_data[['x_position', 'y_position', 'z_position']].values.T
            frame_numbers = sorted_track_data['frame_no'].values
            # Call curve_fit_legendre function
            x_fitted, T, N, B, Curvature, Torsion, Velocity, Acceleration = curve_fit_legendre(xdata, gamma_S, gamma_B, gamma_T, window_size, order)
    
    
            # pitch = 2 * np.pi * (Torsion/1000) / ((Curvature/1000)**2 + (Torsion/1000)**2)
            # radius = (Curvature/1000) / ((Curvature/1000)**2 + (Torsion/1000)**2)
            
            # print(np.median(pitch))
            # print(np.median(radius))
    
            # Create a group for this track
            group = file.create_group(f'{track_no}')
    
            # Save the matrices and vectors to the file
            group.create_dataset('x_fitted', data=x_fitted)
            group.create_dataset('Frame_no', data=frame_numbers)
            group.create_dataset('T', data=T)
            group.create_dataset('N', data=N)
            group.create_dataset('B', data=B)
            group.create_dataset('Curvature', data=Curvature)
            group.create_dataset('Torsion', data=Torsion)
            group.create_dataset('Velocity', data=Velocity*20)
            group.create_dataset('Acceleration', data=Acceleration*20*20)     
            
            if False:
                # Plot the original data
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(xdata[0], xdata[1], xdata[2])
                ax.plot(x_fitted[0], x_fitted[1], x_fitted[2],'r-', linewidth = 1)
                plt.rcParams['font.family'] = 'Helvetica'
                plt.rcParams['font.size'] = 15
    
                # Add labels and legend
                ax.set_xlabel('X $[\mu m]$', fontsize=20, labelpad=20)
                ax.set_ylabel('Y $[\mu m]$', fontsize=20, labelpad=20)
                ax.set_zlabel('Z $[\mu m]$', fontsize=20, labelpad=20)
    
                # Set tick label font size and line width
                ax.tick_params(axis='both', which='major', labelsize=15, width=0.5, pad=10)
    
                # Turn off the grid
                ax.grid(False)
    
                # Show plot
                plt.show()
     

if __name__ == "__main__":
    main()

    