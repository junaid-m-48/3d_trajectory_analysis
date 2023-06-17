"""
Module for calculating the Hessian matrix and the matrix A in the context of curve fitting.

This module contains functions for calculating the Hessian matrix and matrix A which are used in
curve fitting problems, specifically in the context of fitting curves using orthogonal polynomials.

Functions:
    hessian_mat_integral(poly_order) -> tuple
        Returns Hessian matrices computed for first, second, and third derivatives
        of Langrange polynomials for a given polynomial order.

    normalize(array, range=(-1, 1)) -> numpy.ndarray
        Normalizes an array to a given range [-1, 1] by default.

    hessian_mat_obj(xdata, knots_positions, fit_parameters, gamma_s, gamma_b, gamma_t) -> tuple
        Calculates the Hessian matrix H and the matrix A for the curve fitting problem.

Imports:
    numpy (as np)
    scipy.sparse.block_diag
    scipy.sparse.csr_matrix
    orth_order_diff_mat from the local module
"""

import numpy as np
from scipy.sparse import block_diag, csr_matrix
from .orth_order_diff_mat import orth_order_diff_mat

def hessian_mat_integral(poly_order):
    # Hardcoded values for speedup, these hessians were computed considering
    # first, 2nd and 3rd derivative of Langrange polynomials 
    H_1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
        [0, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12],
        [0, 4, 0, 24, 0, 24, 0, 24, 0, 24, 0],
        [0, 0, 12, 0, 40, 0, 40, 0, 40, 0, 40],
        [0, 4, 0, 24, 0, 60, 0, 60, 0, 60, 0],
        [0, 0, 12, 0, 40, 0, 84, 0, 84, 0, 84],
        [0, 4, 0, 24, 0, 60, 0, 112, 0, 112, 0],
        [0, 0, 12, 0, 40, 0, 84, 0, 144, 0, 144],
        [0, 4, 0, 24, 0, 60, 0, 112, 0, 180, 0],
        [0, 0, 12, 0, 40, 0, 84, 0, 144, 0, 220]
    ])
    
    H_2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 36, 0, 120, 0, 252, 0, 432, 0, 660],
        [0, 0, 0, 300, 0, 840, 0, 1620, 0, 2640, 0],
        [0, 0, 120, 0, 1380, 0, 3360, 0, 6060, 0, 9480],
        [0, 0, 0, 840, 0, 4620, 0, 10080, 0, 17220, 0],
        [0, 0, 252, 0, 3360, 0, 12600, 0, 25200, 0, 41160],
        [0, 0, 0, 1620, 0, 10080, 0, 29736, 0, 55440, 0],
        [0, 0, 432, 0, 6060, 0, 25200, 0, 63000, 0, 110880],
        [0, 0, 0, 2640, 0, 17220, 0, 55440, 0, 122760, 0],
        [0, 0, 660, 0, 9480, 0, 41160, 0, 110880, 0, 223740]
    ])

    H_3 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 900, 0, 6300, 0, 22680, 0, 59400, 0],
        [0, 0, 0, 0, 14700, 0, 79380, 0, 249480, 0, 600600],
        [0, 0, 0, 6300, 0, 123480, 0, 532980, 0, 1496880, 0],
        [0, 0, 0, 0, 79380, 0, 703080, 0, 2536380, 0, 6486480],
        [0, 0, 0, 22680, 0, 532980, 0, 3071880, 0, 9604980, 0],
        [0, 0, 0, 0, 249480, 0, 2536380, 0, 11060280, 0, 30810780],
        [0, 0, 0, 59400, 0, 1496880, 0, 9604980, 0, 34345080, 0],
        [0, 0, 0, 0, 600600, 0, 6486480, 0, 30810780, 0, 94877640]
    ])
    H_dR = np.array(H_1[0:poly_order, 0:poly_order], dtype=float)
    H_ddR = np.array(H_2[0:poly_order, 0:poly_order], dtype=float)
    H_dddR = np.array(H_3[0:poly_order, 0:poly_order], dtype=float)
    return H_dR, H_ddR, H_dddR


def normalize(array, range=(-1, 1)):
    """
    Normalizes an array to a given range [-1, 1] by default.
    """
    min_val, max_val = range
    min_array, max_array = np.min(array), np.max(array)
    normalized_array = (max_val - min_val) * (array - min_array) / (max_array - min_array) + min_val
    return normalized_array

def hessian_mat_obj(xdata, knots_positions, fit_parameters, gamma_s, gamma_b, gamma_t):
   
    """
    Calculates the Hessian matrix H and the matrix A for the curve fitting problem.
    
    Parameters:
        xdata (numpy.ndarray): The x data points for the curve fitting.
        knots_positions (list): List of positions of knots.
        fit_parameters (dict): Dictionary containing fit parameters such as "Terms".
        gamma_s (float): Scaling factor for the stretch energy.
        gamma_b (float): Scaling factor for the bend energy.
        gamma_t (float): Scaling factor for the twist energy.
    
    Returns:
        tuple: A tuple (H, A) where H is the Hessian matrix and A is the matrix for
               least-square-minimization.
    """
    
    time_len = xdata.shape[1] - 1
    t = np.arange(time_len + 1)
    A = csr_matrix((0, 0))  # For least-square-minimization
    S = csr_matrix((0, 0))  # For stretch energy
    B = csr_matrix((0, 0))  # For bend energy
    T = csr_matrix((0, 0))  # For twist energy
    
    # Loop over the knot positions
    for i in range(len(knots_positions) - 1):
        si = knots_positions[i]
        li = knots_positions[i + 1]
        t_process = t[si-1:li]
        
        t_new = normalize(t_process, range=(-1, 1))
        
        # Calculate the orthogonal polynomials and hessian matrix integral
        polynomials_var = orth_order_diff_mat(fit_parameters["Terms"][1], t_new, 0)
        H_dR, H_ddR, H_dddR = hessian_mat_integral(fit_parameters["Terms"][1])
        
        if i < len(knots_positions) - 2:
            polynomials_var = np.delete(polynomials_var, -1, axis=1)
        
        # Multiply the polynomials by appropriate scaling factors
        polynomials_var_1 = H_dR * ((t_new[1] - t_new[0]) ** 1)
        polynomials_var_2 = H_ddR * ((t_new[1] - t_new[0]) ** 2)
        polynomials_var_3 = H_dddR * ((t_new[1] - t_new[0]) ** 3)
        
        # Append the polynomials to the matrices A, S, B, T
        A = block_diag([A, polynomials_var.T], format='csr')
        S = block_diag([S, polynomials_var_1], format='csr')
        B = block_diag([B, polynomials_var_2], format='csr')
        T = block_diag([T, polynomials_var_3], format='csr')
    
    # Duplicate and stack the matrices A, B, C, D for the three coordinate axes
    A = block_diag([A]*3, format='csr')
    S = block_diag([S]*3, format='csr')
    B = block_diag([B]*3, format='csr')
    T = block_diag([T]*3, format='csr')
    
    # Calculate the combined Hessian matrix H
    H = (2. / len(t)) * A.T.dot(A) + gamma_s * S + gamma_b * B + gamma_t * T
    
    return H, A
