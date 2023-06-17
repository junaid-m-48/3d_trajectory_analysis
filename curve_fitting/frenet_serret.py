import numpy as np
from .orth_order_diff_mat import orth_order_diff_mat

def normalize(array, range=(-1, 1)):
    """
    Normalizes an array to a given range [-1, 1] by default.
    """
    min_val, max_val = range
    min_array, max_array = np.min(array), np.max(array)
    normalized_array = (max_val - min_val) * (array - min_array) / (max_array - min_array) + min_val
    return normalized_array


def frenet_serret(coeff_struct, order, time_len, knots_positions):
    """
    Calculates the Frenet-Serret frames, curvature, torsion, velocity, and acceleration of a space curve.
    
    The function computes the unit tangent vector (T), normal vector (N), binormal vector (B), curvature, torsion, velocity, and acceleration
    for a given set of control points representing a space curve. These quantities are computed at various points along the curve
    using the Frenet-Serret formulas.
    
    Parameters:
        coeff_struct (dict): Dictionary containing coefficients for the polynomial representation of the space curve.
        order (int): The order of the polynomial used to represent the space curve.
        time_len (int): The total number of time steps.
        knots_positions (list of int): List of indices representing the positions of knots in the space curve.
        
    Returns:
        T (numpy array): The unit tangent vectors at various points along the curve.
        N (numpy array): The normal vectors at various points along the curve.
        B (numpy array): The binormal vectors at various points along the curve.
        Curvature (numpy array): The curvature at various points along the curve.
        Torsion (numpy array): The torsion at various points along the curve.
        Velocity (numpy array): The velocity vectors at various points along the curve.
        Acceleration (numpy array): The acceleration vectors at various points along the curve.
    """
    dimensions = ['X', 'Y', 'Z']
    dR = []  # Initialize the first derivative array
    ddR = []  # Initialize the second derivative array
    dddR = []  # Initialize the third derivative array
    total_stencils = len(knots_positions) - 1
    window_size = knots_positions[1] - knots_positions[0]
    
    for j in range(3):
        fR = []  # Initialize the array for the first derivatives
        ffR = []  # Initialize the array for the second derivatives
        fffR = []  # Initialize the array for the third derivatives
        t = np.arange(time_len + 1)
        
        for i in range(total_stencils):
            field_name = f'coeff_{dimensions[j]}{i + 1}'
            coeff = coeff_struct[field_name]
            si = knots_positions[i]  # Determine the starting index for the current knot
            li = knots_positions[i + 1]  # Determine the ending index for the current knot
            t_process = t[si-1:li]  # Extract the time values for the current knot
            
            t_new = normalize(t_process)  # Normalize the time values to the range [-1, 1]
            
            polynomials_var = orth_order_diff_mat(order, t_new, 1)  # Generate the orthogonal polynomials for the first derivative
            f_interm = (coeff @ polynomials_var) * (t_new[1] - t_new[0])  # Calculate the first derivatives for the current knot
            polynomials_var = orth_order_diff_mat(order, t_new, 2)  # Generate the orthogonal polynomials for the second derivative
            f2_interm = (coeff @ polynomials_var) * (t_new[1] - t_new[0]) ** 2  # Calculate the second derivatives for the current knot
            polynomials_var = orth_order_diff_mat(order, t_new, 3)  # Generate the orthogonal polynomials for the third derivative
            f3_interm = (coeff @ polynomials_var) * (t_new[1] - t_new[0]) ** 3  # Calculate the third derivatives for the current knot
            
            if i < total_stencils - 1:
                f_interm = f_interm[:-1]  # Remove the last element of the first derivatives if not the last knot
                f2_interm = f2_interm[:-1]  # Remove the last element of the second derivatives if not the last knot
                f3_interm = f3_interm[:-1]  # Remove the last element of the third derivatives if not the last knot
                
            fR.extend(f_interm)  # Concatenate the first derivatives for all knots
            ffR.extend(f2_interm)  # Concatenate the second derivatives for all knots
            fffR.extend(f3_interm)  # Concatenate the third derivatives for all knots
            
        dR.append(fR)  # Concatenate the first derivatives for all dimensions
        ddR.append(ffR)  # Concatenate the second derivatives for all dimensions
        dddR.append(fffR)  # Concatenate the third derivatives for all dimensions
        
    dR = np.array(dR)[:, window_size:-window_size]  # Remove the padded values from the first derivatives
    ddR = np.array(ddR)[:, window_size:-window_size]  # Remove the padded values from the second derivatives
    dddR = np.array(dddR)[:, window_size:-window_size]  # Remove the padded values from the third derivatives
    
    T = dR / np.sqrt(np.sum(dR ** 2, axis=0))  # Calculate the unit tangent vector along the curve
    

    cross_ddR_dR = np.cross(ddR, dR, axisa=0, axisb=0).T  # Cross product between ddR and dR for each column (time point)
    
    cross_dR_ddR = np.cross(dR, ddR, axisa=0, axisb=0).T
    
    N = np.cross(dR, cross_ddR_dR, axisa=0, axisb=0).T / (np.sqrt(np.sum(dR ** 2, axis=0)) * np.sqrt(np.sum(cross_ddR_dR ** 2, axis=0)))  # Calculate the normal vector to the curve (Frenet trihedron)
    
    B = np.cross(dR, ddR, axisa=0, axisb=0).T / np.sqrt(np.sum(np.cross(dR, ddR, axisa=0, axisb=0).T ** 2, axis=0))  # Calculate the binormal vector to the curve (Frenet trihedron)
    
    Curvature = np.sqrt(np.sum(np.cross(dR, ddR, axisa=0, axisb=0).T ** 2, axis=0)) / np.sum(dR ** 2, axis=0) ** 1.5  # Calculate the curvature of the curve
    
    Torsion = np.sum(dddR * cross_dR_ddR, axis=0).T / np.sum(cross_ddR_dR ** 2, axis=0)  # Calculate the torsion of the curve
    
    Velocity = dR  # Assign the first derivatives as the velocity
    
    Acceleration = ddR  # Assign the second derivatives as the acceleration
    
    return T, N, B, Curvature, Torsion, Velocity, Acceleration
