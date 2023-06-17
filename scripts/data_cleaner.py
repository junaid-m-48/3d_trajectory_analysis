import os
import random
import configparser
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    
    """
    Load, analyze, and clean particle trajectory data.
    
    This script reads a configuration file to get parameters and file paths. It then loads particle trajectory
    data from an HDF5 file, visualizes a subset of the trajectories in a 3D plot, cleans the data based
    on velocity, length, and standard deviation, and saves the cleaned data to a new HDF5 file.
    
    The configuration file should have the following format:
    [DOWNLOAD]
    filename = <file_name>
    
    [DEFAULT]
    data_directory = <data_directory_path>
    
    [PLOT]
    num_particles_raw_plot = <number_of_particles_to_plot>
    
    [CLEANING]
    min_velocity = <minimum_velocity>
    min_length = <minimum_length>
    min_position_std = <minimum_position_standard_deviation>
    filename = <output_file_name>
    """
    
    # Read configuration file
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)

    ## Retrieve values from the configuration file
    
    # For loading data
    filename = config.get('DOWNLOAD', 'filename')
    data_directory = config.get('DEFAULT', 'data_directory')
    
    # To check raw data file
    num_particles_raw_plot = config.getint('PLOT', 'num_particles_raw_plot')
    
    #To clean the data
    min_velocity = config.getint('CLEANING', 'min_velocity')
    min_length = config.getint('CLEANING', 'min_length')
    min_position_std = config.getint('CLEANING', 'min_position_std')
    
    
    # Define the file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    file_path = os.path.join(parent_dir, data_directory, filename)
    
    # Load the data as a pandas DataFrame
    # Load the data from the HDF5 file
    with h5py.File(file_path, 'r') as file:
        data = np.array(file['particle_trajectories'])

    # Convert the array to a DataFrame for easier manipulation
    columns = ['Particle_no', 'frame_no', 'x_position', 'y_position', 'z_position']
    raw_df = pd.DataFrame(data, columns=columns)
    
   

    # plot few of the tracks from data frame to check if everything is alright
    particle_counts = raw_df.groupby('Particle_no').size()
    long_tracks = particle_counts[particle_counts > 200].index.tolist()

    # Randomly select particle numbers
    selected_particles = random.sample(long_tracks, num_particles_raw_plot)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the selected tracks
    for particle in selected_particles:
        track = raw_df[raw_df['Particle_no'] == particle]

        # Extract X, Y, Z coordinates
        x_coords = track['x_position'].values * 1000
        y_coords = track['y_position'].values * 1000
        z_coords = track['z_position'].values * 1000

        # Plot the track
        ax.plot(x_coords, y_coords, z_coords, label=f'Particle {particle}')

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

    # Set plot limits
    ax.set_xlim([-1500, 1500])
    ax.set_ylim([-1500, 1500])
    ax.set_zlim([-1500, 1500])

    # Show plot
    plt.show()
    
    # Cleaning the data based on velocity,length and standard deviation
   
    cleaned_df = clean_data(raw_df, min_velocity, min_length, min_position_std, True)
    
    
    # Create an h5 file and save the data
    output_filename = config.get('CLEANING', 'filename')
    output_file_path = os.path.join(parent_dir, data_directory,output_filename)
    data = cleaned_df.to_numpy()
    with h5py.File(output_file_path, 'w') as file:
        file.create_dataset('data', data=data)

def clean_data(data_frame, min_velocity=20, min_length=20, min_position_std=20, plot=False):
    """
    Clean and filter the data frame based on specified criteria.

    Args:
        data_frame (pd.DataFrame): The pandas DataFrame containing the raw data.
        min_velocity (float): The minimum velocity threshold in um/s (default: 20).
        min_length (int): The minimum length of the particle track in frames (default: 20).
        min_position_std (float): The minimum position standard deviation threshold in um (default: 20).
        plot (bool): A flag indicating whether to plot histograms of raw and cleaned velocities (default: False).

    Returns:
        pd.DataFrame: The cleaned data frame.
    """
    # Calculate velocity for each particle and multiply by 20 and 1000 (beacuse Frame rate was 20 and values were in mm/s)
    data_frame['velocity'] = (data_frame.groupby('Particle_no')['x_position'].diff() ** 2 +
                              data_frame.groupby('Particle_no')['y_position'].diff() ** 2 +
                              data_frame.groupby('Particle_no')['z_position'].diff() ** 2).pow(0.5)
    data_frame['velocity'] = data_frame['velocity'] * 20 * 1000

    # Calculate median velocity for each particle
    median_velocities = data_frame.groupby('Particle_no')['velocity'].median()
    
    if plot:
        fig =plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist(median_velocities, bins=200)
        ax1.set_xlabel('Velocity $(um/s)$')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Raw Data')
        plt.show()
    
    # Calculate position standard deviation for each particle
    position_std = ((data_frame.groupby('Particle_no')['x_position'].std() ** 2 +
                    data_frame.groupby('Particle_no')['y_position'].std() ** 2 +
                    data_frame.groupby('Particle_no')['z_position'].std() ** 2).pow(0.5))*1000

    # Calculate the length of each particle track
    particle_counts = data_frame.groupby('Particle_no').size()
    
    # Filter based on minimum median velocity, length, and position standard deviation
    cleaned_data_frame = data_frame[data_frame['Particle_no'].isin(
        median_velocities[median_velocities > min_velocity].index) & 
        data_frame['Particle_no'].isin(
            particle_counts[particle_counts > min_length].index) &
        data_frame['Particle_no'].isin(
            position_std[position_std > min_position_std].index)]

    median_velocities = cleaned_data_frame.groupby('Particle_no')['velocity'].median()
    cleaned_data_frame = cleaned_data_frame.drop('velocity', axis=1)
    
    if plot:
        ax2.hist(median_velocities, bins=200)
        ax2.set_xlabel('Velocity $(um/s)$')
        ax2.set_title('Cleaned Data')
        plt.show()
    
    return cleaned_data_frame

    
if __name__ == "__main__":
    main()
