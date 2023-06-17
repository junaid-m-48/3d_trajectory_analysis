"""
This script contains functions for fitting 3D trajectories to Legendre polynomials
and calculating various properties like Frenet-Serret frames, curvature, torsion, etc.

Modules:
    - numpy
    - matplotlib
    - scipy
    - construct_eq_mat
    - hessian_mat_obj
    - gen_fit
    - frenet_serret
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from .construct_eq_mat import construct_eq_mat
from .hessian_mat_obj import hessian_mat_obj
from .gen_fit import gen_fit
from .frenet_serret import frenet_serret

def curve_fit_legendre(X, gamma_S, gamma_B, gamma_T, window_size, order):
    """
    Fit 3D trajectories to Legendre polynomials and calculate Frenet-Serret frames,
    curvature, torsion, velocity, and acceleration.

    Args:
        X (numpy.ndarray): The input 3D positions (shape: 3 x n).
        gamma_S (float): Weighting factor for smoothing.
        gamma_B (float): Weighting factor for bending.
        gamma_T (float): Weighting factor for torsion.
        window_size (int): The window size for fitting.
        order (int): Order of the Legendre polynomial.

    Returns:
        tuple: A tuple containing fitted 3D positions, Frenet-Serret frames (T, N, B),
               curvature, torsion, velocity, and acceleration.
    """
    # ... rest of the code remains the same.

    # Calculate the mirrored version of the minimum x-coordinate
    x_min = 2 * (X[:, [0]] * np.ones((3, X.shape[1]))) - X
    # Calculate the mirrored version of the maximum x-coordinate
    x_max = 2 * (X[:, [-1]] * np.ones((3, X.shape[1]))) - X
    
    # Pad the input positions with mirrored values
    X_padded = np.hstack((np.fliplr(x_min[:, 1:window_size + 1]), X, np.fliplr(x_max[:, -(window_size + 1):-1])))
    xdata = X_padded
    
    # To check the padding
    if False:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the track
        ax.plot(xdata[0], xdata[1], xdata[2])

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
    
    # Determine the number of elements in the x-data
    x = xdata.shape[1] - 1
    elements_in = np.hstack((window_size * np.ones(x // window_size, dtype=int), np.array([1 + x % window_size])))

    # Calculate the positions of the knots
    knots_positions = np.cumsum(np.hstack(([1], elements_in)))
    if elements_in[-1] < elements_in[0] / 2:
        knots_positions = np.delete(knots_positions, -2)
    knots_positions[-1] = knots_positions[-1] - 1
    
    time_len = xdata.shape[1] - 1
    
    fit_parameters = {"Window_size": window_size, "Terms": [len(knots_positions) - 1, order]}
    
    # Call custom functions
    Aeq, beq = construct_eq_mat(xdata, fit_parameters, knots_positions, 3)
    H, A = hessian_mat_obj(xdata, knots_positions, fit_parameters, gamma_S, gamma_B, gamma_T)
    
    B = xdata.flatten(order='C')  # Extract the positions from the padded x-data

    Aeq_T = Aeq.transpose()
    zero_mat = sp.csr_matrix((Aeq.shape[0], Aeq_T.shape[1]))
    conc_prob = sp.vstack([
        sp.hstack([H, Aeq_T]),
        sp.hstack([Aeq, zero_mat])
        ], format='csr')

    # Calculate the concatenated problem vector
    A_T = A.transpose()
    BB = np.concatenate([
        (2 / xdata.shape[1]) * A_T*(B),
        beq
        ])
    
    conc_prob_dense = conc_prob.toarray()
    # Solve the concatenated problem to obtain the coefficient vector
    k = np.linalg.solve(conc_prob_dense,BB)
    
    # Extract coefficients
    coeff_x = k[0:1 + np.prod(fit_parameters["Terms"]) - 1]
    coeff_y = k[np.prod(fit_parameters["Terms"]): 1 + 2 * np.prod(fit_parameters["Terms"]) - 1]
    coeff_z = k[2 * np.prod(fit_parameters["Terms"]): 1 + 3 * np.prod(fit_parameters["Terms"]) - 1]
    
    # Generate the fitted values
    x = gen_fit(np.arange(time_len + 1), coeff_x, fit_parameters["Terms"], fit_parameters["Window_size"])
    y = gen_fit(np.arange(time_len + 1), coeff_y, fit_parameters["Terms"], fit_parameters["Window_size"])
    z = gen_fit(np.arange(time_len + 1), coeff_z, fit_parameters["Terms"], fit_parameters["Window_size"])
    x_new = np.vstack((x, y, z))
    x_new = x_new[:, window_size:-window_size]
    
    
    if False:
        # Plot the original data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[0], X[1], X[2],'go',markersize = 1)
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
    
    coeff_all = np.vstack((coeff_x, coeff_y, coeff_z))
    dimensions = ['X', 'Y', 'Z']
    coeff_structure = {}
    
    for j, dim in enumerate(dimensions):
        coeffic = np.reshape(coeff_all[j], (fit_parameters["Terms"][1], fit_parameters["Terms"][0]), order='F').copy().T
        for i in range(coeffic.shape[0]):
            field_name = f'coeff_{dim}{i + 1}'
            field_value = coeffic[i, :]
            coeff_structure[field_name] = field_value
    
    # Call another custom function for frenet_serret
    T, N, B, Curvature, Torsion, Velocity, Acceleration = frenet_serret(coeff_structure, order, time_len, knots_positions)
    
    # Return the result
    return x_new, T, N, B, Curvature, Torsion, Velocity, Acceleration


# Usage example (assuming you have input data in the variable 'X' and appropriate parameters):
# x_new, T, N, B, Curvature, Torsion, Velocity, Acceleration = curve_fit_legendre(X, gamma_S, gamma_B, gamma_T, window_size, order)
