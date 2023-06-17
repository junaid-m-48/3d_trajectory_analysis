import numpy as np
from sklearn.preprocessing import minmax_scale
from .orth_order_diff_mat import orth_order_diff_mat

def construct_eq_mat(xdata, fit_parameters, knots_positions, deriv_cond):
    """
    Constructs the equality matrix (Aeq) and right-hand side vector (beq) for curve fitting using spline interpolation.

    This function is used to construct the matrix and vector required for enforcing continuity conditions
    between spline segments. These conditions include matching of position, derivative, and curvature
    at the interior knots.

    Parameters:
        xdata (numpy array): 2D array where each row represents a point in 3D space.
        fit_parameters (dict): Dictionary containing fitting parameters such as the order of polynomial terms and window size.
        knots_positions (list of int): List of indices representing the positions of knots in the space curve.
        deriv_cond (int): The highest order derivative condition to enforce (e.g., 0 for position, 1 for first derivative).

    Returns:
        Aeq (numpy array): The equality matrix for enforcing continuity conditions.
        beq (numpy array): The right-hand side vector for the equality conditions, usually set to zeros for continuity.
    """
    
    x = xdata.shape[1] - 1

    # Calculate the number of elements in each stencil based on the window size
    elements_in = [fit_parameters["Window_size"]] * (x // fit_parameters["Window_size"]) + [1 + x % fit_parameters["Window_size"]]
    
    idx_data = np.cumsum([1] + elements_in)

    # Adjust the index positions based on the element counts
    if elements_in[-1] < elements_in[0] / 2:
        idx_data = np.delete(idx_data, -2)
    idx_data[-1] = idx_data[-1] - 1

    order = fit_parameters["Terms"][1]

    Aeq = np.zeros(((len(idx_data) - 2) * ((deriv_cond + 1) * 3), 3 * np.prod(fit_parameters["Terms"])))

    total_cond = list(range(deriv_cond + 1))
    ro = 0
    co = 0
    t = np.arange(x + 1)

    # Calculate the normalized parameter values within each stencil
    si = knots_positions[0]
    li = knots_positions[1]
    t_process = t[si-1:li]
    t_new = minmax_scale(t_process, feature_range=(-1, 1))
    m1 = (t_new[1] - t_new[0])

    si = knots_positions[-2]
    li = knots_positions[-1]
    t_process = t[si-1:li]
    t_new = minmax_scale(t_process, feature_range=(-1, 1))
    m2 = (t_new[1] - t_new[0])

    st_p = [1] * (len(idx_data) - 2)
    enp_p = [len(t_new)] * (len(idx_data) - 2)

    # Loop over derivative conditions
    for dr in total_cond:
        # Start with X Coordinates
        # Loop over stencils
        for i in range(len(idx_data) - 2):
            # Calculate the first and last multiplier for derivative condition
            first_multiplier_curv = orth_order_diff_mat(order, t_new[enp_p[i] - 1], dr).T * m1 ** dr
            
            if i == len(idx_data) - 3:
                last_multiplier_curv = -orth_order_diff_mat(order, t_new[st_p[i] - 1], dr).T * m2 ** dr
            else:
                last_multiplier_curv = -orth_order_diff_mat(order, t_new[st_p[i] - 1], dr).T * m1 ** dr
                
            curv_cond = np.hstack((first_multiplier_curv, last_multiplier_curv))

            # Assign the derivative condition to Aeq
            Aeq[ro, co:co + 2 * order] = curv_cond
            co = co + order
            ro = ro + 1
        
        # Update the column offset for Y Coordinates
        co = (len(idx_data) - 1) * order
        for i in range(len(idx_data) - 2):
            # Calculate the first and last multiplier for curvature condition
            first_multiplier_curv = orth_order_diff_mat(order, t_new[enp_p[i] - 1], dr).T * m1 ** dr
            
            if i == len(idx_data) - 3:
                last_multiplier_curv = -orth_order_diff_mat(order, t_new[st_p[i] - 1], dr).T * m2 ** dr
            else:
                last_multiplier_curv = -orth_order_diff_mat(order, t_new[st_p[i] - 1], dr).T * m1 ** dr
                
            curv_cond = np.hstack((first_multiplier_curv, last_multiplier_curv))

            # Assign the curvature condition to Aeq
            Aeq[ro, co:co + 2 * order] = curv_cond
            co = co + order
            ro = ro + 1
        
        # Update the column offset for Z Coordinates
        co = (len(idx_data) - 1) * order * 2
        for i in range(len(idx_data) - 2):
            # Calculate the first and last multiplier for curvature condition
            first_multiplier_curv = orth_order_diff_mat(order, t_new[enp_p[i] - 1], dr).T * m1 ** dr
            
            if i == len(idx_data) - 3:
                last_multiplier_curv = -orth_order_diff_mat(order, t_new[st_p[i] - 1], dr).T * m2 ** dr
            else:
                last_multiplier_curv = -orth_order_diff_mat(order, t_new[st_p[i] - 1], dr).T * m1 ** dr
                
            curv_cond = np.hstack((first_multiplier_curv, last_multiplier_curv))

            # Assign the curvature condition to Aeq
            Aeq[ro, co:co + 2 * order] = curv_cond
            co = co + order
            ro = ro + 1
        
        # Reset the column offset for the next derivative condition
        co = 0

    beq = np.zeros((len(idx_data) - 2) * ((deriv_cond + 1) * 3))

    return Aeq, beq