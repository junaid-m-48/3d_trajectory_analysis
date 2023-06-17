import h5py
import os
import numpy as np
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def main():
    
    """
    This script processes and analyzes the fitted trajectory data using kernel density estimation (KDE) to compute
    the distribution of parameters such as velocity, pitch, and radius. It also generates 2D kernel density plots
    for radius vs velocity, pitch vs velocity, and pitch vs radius.
    
    Dependencies:
        h5py, os, numpy, configparser, pandas, matplotlib, sklearn.neighbors.KernelDensity
    
    Usage:
        Run this script through the command line or an IDE. The script is intended to be executed as a standalone
        file, and it uses values from a configuration file 'config.ini'.
    
    Configuration File (config.ini):
        DEFAULT:
            data_directory : Path to the directory containing the data files.
        Fitting:
            filename : Name of the HDF5 file containing the fitted trajectory data.
        PLOTS:
            plot_directory : Path to the directory to save the generated plots.
    
    Output:
        The output of this script is a set of histograms and 2D kernel density plots showing the distribution of
        velocity, pitch, radius, and omega values computed from the fitted trajectory data. The plots are saved
        as PNG images in the specified plot directory.
    """
     # Read configuration file
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)

    ## Retrieve values from the configuration file
    # Define the file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    filename = config.get('Fitting', 'filename')
    data_directory = config.get('DEFAULT', 'data_directory')
    file_path = os.path.join(parent_dir, data_directory, filename)
    
    # plot_directory
    plot_directory = os.path.join(parent_dir, 'plots')
    os.makedirs(plot_directory, exist_ok=True)
    
    # Dictionaries to store the data
    curvatures = {}
    torsions = {}
    velocities = {}
    frame_nos = {}
    
    # plot_directory
    plot_directory = os.path.join('..', 'Plots')
    os.makedirs(plot_directory, exist_ok=True)
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:       
        
        # Iterate over each group (track) in the file
        for track_name in file.keys():
                   
            # Print the track name
            print(f"Track: {track_name}")
            # Load the datasets for the current track into memory
            curvatures[track_name] = np.array(file[track_name]['Curvature']) / 1000 # converting to um
            torsions[track_name] = np.array(file[track_name]['Torsion']) / 1000 # # converting to um
            
            # Load the Velocity dataset
            velocity_vectors = np.array(file[track_name]['Velocity'])
            # Compute the magnitudes of the velocity vectors
            velocity_magnitudes = np.linalg.norm(velocity_vectors, axis=0)
            # Store the magnitudes in the velocities dictionary
            velocities[track_name] = velocity_magnitudes*1000
            
            frame_nos[track_name] = np.array(file[track_name]['Frame_no'])
     # List to store all the data
    all_data = []
    
    # Iterate through the keys in the dictionaries (track names)
    for track_name in curvatures.keys():
        # Zip the data
        data = zip(frame_nos[track_name], curvatures[track_name], torsions[track_name], velocities[track_name])
        # Append to all_data
        all_data.extend([(track_name, frame_no, curvature, torsion, velocity) for frame_no, curvature, torsion, velocity in data])
    
    # Create a DataFrame
    columns = ['track_no', 'frame_no', 'Curvature', 'Torsion', 'Velocity']
    fitted_df = pd.DataFrame(all_data, columns=columns)

    fitted_df['Pitch'] = 2 * np.pi * fitted_df['Torsion'] / (fitted_df['Curvature']**2 + fitted_df['Torsion']**2)
    fitted_df['Radius'] = fitted_df['Curvature'] / (fitted_df['Curvature']**2 + fitted_df['Torsion']**2)
    fitted_df['Omega'] = fitted_df['Velocity'] / np.sqrt(fitted_df['Pitch']**2 + fitted_df['Radius']**2)
    
    # Group by Index (track) and compute median for each track
    grouped = fitted_df.groupby('track_no')
    velocity_median = grouped['Velocity'].median()
    pitch_median = grouped['Pitch'].median()
    pitch_median_absolute = grouped['Pitch'].apply(lambda x: np.median(np.abs(x)))
    radius_median = grouped['Radius'].median() 
    omega_median = grouped['Omega'].median() 
    
    
    # Compute the sign of the median for each track
    signs = fitted_df.groupby('track_no')['Pitch'].transform(lambda x: np.sign(x.median()))
    
    # Compute the updated pitch values and store them in a new column 'Updated_Pitch'
    fitted_df['Updated_Pitch'] = np.abs(fitted_df['Pitch']) * signs
    
    # Compute the median of the updated 'Pitch' values
    mod_pitch_median = fitted_df.groupby('track_no')['Updated_Pitch'].median()
    
    
    # Set global parameters for the plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    
    # Create figure and axis objects
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    
    # First subplot
    axs[0].hist(velocity_median, bins=np.linspace(0, 250, 61), color='skyblue', edgecolor='k')
    axs[0].set_xlabel('Velocity $(\\bar{V}) \\, [\\mu m/s]$', fontsize=16)
    axs[0].set_title('Velocity Distribution', fontsize=16)
    
    # Second subplot
    axs[1].hist(pitch_median_absolute, bins=np.linspace(0, 400, 61), color='salmon', edgecolor='k')
    axs[1].set_xlabel('Pitch $(\\bar{|P|}) \\, [\\mu m]$', fontsize=16)
    axs[1].set_title('Pitch Distribution', fontsize=16)
    
    # Third subplot
    axs[2].hist(radius_median, bins=np.linspace(0, 50, 61), color='lightgreen', edgecolor='k')
    axs[2].set_xlabel('Radius $(\\bar{R}) \\, [\\mu m]$', fontsize=16)
    axs[2].set_title('Radius Distribution', fontsize=16)
    
    # Fourth subplot
    axs[3].hist(omega_median, bins=np.linspace(0, 3, 61), color='orchid', edgecolor='k')
    axs[3].set_xlabel('Omega $(\\bar{\\Omega}) \\, [\\mu m]$', fontsize=16)
    axs[3].set_title('Omega Distribution', fontsize=16)
    
    # Set common labels
    fig.text(0.04, 0.5, 'No. of Tracks', va='center', rotation='vertical', fontsize=18)
    
    
     # Save the figure in high resolution

    plot_path = os.path.join(plot_directory, "hist_parameters.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    

    rad_edge = np.linspace(0,30,51);
    vel_edge = np.linspace(30,200,51);
    pitch_edge = np.linspace(-400,400,51);
    pitch_edge_2 = np.linspace(0,400,51);
    
    
    
    # Creating mesh grids
    xv, yv = np.meshgrid(rad_edge, vel_edge)
    xv2, yv2 = np.meshgrid(pitch_edge, vel_edge)
    xv3, yv3 = np.meshgrid(pitch_edge_2, rad_edge)
    
    # Calculating Kernel Density Estimates
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    values = np.exp(kde.fit(np.vstack([radius_median, velocity_median]).T).score_samples(np.vstack([xv.ravel(), yv.ravel()]).T))
    values = values / np.max(values)
    
    kde2 = KernelDensity(bandwidth=1.0, kernel='gaussian')
    values2 = np.exp(kde2.fit(np.vstack([mod_pitch_median, velocity_median]).T).score_samples(np.vstack([xv2.ravel(), yv2.ravel()]).T))
    values2 = values2 / np.max(values2)
    
    kde3 = KernelDensity(bandwidth=1.0, kernel='gaussian')
    values3 = np.exp(kde3.fit(np.vstack([pitch_median_absolute, radius_median]).T).score_samples(np.vstack([xv3.ravel(), yv3.ravel()]).T))
    values3 = values3 / np.max(values3)
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    # Set global parameters for the plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    # Colormap
    cmap = 'inferno_r'
    
    # First plot
    cf1 = axs[0].contourf(xv, yv, values.reshape(xv.shape), cmap=cmap, levels=50)
    axs[0].set_xlabel('Radius $(\\bar{R}) \\, [\\mu m]$', fontsize=16, labelpad=10)
    axs[0].set_ylabel('Velocity $(\\bar{V}) \\, [\\mu m/s]$', fontsize=16, labelpad=10)
    axs[0].tick_params(labelsize=14)
    axs[0].set_xlim([-0.2, 30.2])
    axs[0].set_ylim([28, 202])
    axs[0].grid(False)
    
    # Second plot
    cf2 = axs[1].contourf(xv2, yv2, values2.reshape(xv2.shape), cmap=cmap, levels=50)
    axs[1].set_xlabel('Pitch $(\\bar{P}) \\, [\\mu m]$', fontsize=16, labelpad=10)
    axs[1].set_ylabel('Velocity $(\\bar{V}) \\, [\\mu m/s]$', fontsize=16, labelpad=10)
    axs[1].axvline(x=0, color='k', linestyle='--', linewidth=1)
    axs[1].tick_params(labelsize=14)
    axs[1].set_xlim([-402, 402])
    axs[1].set_ylim([28, 202])
    axs[1].grid(False)
    
    # Third plot
    cf3 = axs[2].contourf(xv3, yv3, values3.reshape(xv3.shape), cmap=cmap, levels=50)
    axs[2].set_xlabel('Pitch $(\\bar{|P|}) \\, [\\mu m]$', fontsize=16, labelpad=10)
    axs[2].set_ylabel('Radius $(\\bar{R}) \\, [\\mu m]$', fontsize=16, labelpad=10)
    axs[2].tick_params(labelsize=14)
    axs[2].set_xlim([-2, 402])
    axs[2].set_ylim([-0.1, 30.1])
    axs[2].grid(False)
    
    # Adding colorbar
    cbar = fig.colorbar(cf3, ax=axs[2])
    cbar.set_label('Density', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    
    #
    plot_path = os.path.join(plot_directory, "2D_Kernel_Density.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Showing plot
    plt.show()

    
    

if __name__ == "__main__":
    main()
