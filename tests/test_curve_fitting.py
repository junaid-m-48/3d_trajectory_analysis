import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from curve_fitting import curve_fit_legendre

def main():
    r = 5
    gap = 2
    fps = 20

    t = np.arange(0, 40 * np.pi/gap, 2 * np.pi / (gap * fps))
    st = r * np.sin(t)
    ct = r * np.cos(t)
    
    xdata = np.vstack((st, ct, t)) #+ 1 * np.vstack((np.random.rand(len(t)), np.random.rand(len(t)), np.random.rand(len(t))))
    
    # Set parameters for curve fitting
    gamma_S = 0
    gamma_B = 0
    gamma_T = 0
    window_size = 40
    order = 7
    # Call curve_fit_legendre function
    x_new, T, N, B, Curvature, Torsion, Velocity, Acceleration = curve_fit_legendre(xdata, gamma_S, gamma_B, gamma_T, window_size, order)

    pitch = 2 * np.pi * Torsion / (Curvature**2 + Torsion**2)
    radius = Curvature / (Curvature**2 + Torsion**2)
    # Plot the original data
    print(np.median(pitch))
    print(np.median(radius))
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xdata[0], xdata[1], xdata[2])
        ax.plot(x_new[0], x_new[1], x_new[2],'r-', linewidth = 1)
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
