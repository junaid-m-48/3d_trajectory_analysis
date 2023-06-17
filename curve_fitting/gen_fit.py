import numpy as np
from .orth_order_diff_mat import orth_order_diff_mat

def gen_fit(t, coeff, parameters, ws=25):
    """
    Generate a fitted function based on input parameters.
    
    Parameters:
        t (numpy array): The array of time values.
        coeff (numpy array): The array of coefficients.
        parameters (int or list of int): Polynomial degree, or a list where the second element is the polynomial degree.
        ws (int): Window size (default is 25).
    
    Returns:
        numpy array: The fitted values.
    """

    # Determine the degree of the fitting polynomial based on the parameters
    if isinstance(parameters, (list, tuple)) and len(parameters) > 1:
        n = parameters[1]
    else:
        n = parameters

    # Ensure that coeff is a column vector
    coeff = coeff.flatten()

    # Compute the size of t
    X = len(t) - 1

    # Determine the number of elements in each window
    N = ws
    elements_in = np.array([N] * int(X / N) + [1 + X % N])

    # Compute the cumulative sum of elements_in to obtain the indices of data segments
    idx_data = np.cumsum(np.array([1] + elements_in.tolist()))

    # Adjust the last element of idx_data if the last window has fewer elements than the first half
    if elements_in[-1] < elements_in[0] / 2:
        idx_data = np.delete(idx_data, -2)

    # Adjust the last element of idx_data to exclude the last data point of the last window
    idx_data[-1] = idx_data[-1] - 1

    # Reshape the coefficients into a matrix with dimensions (number of windows) x (degree of polynomial)
    coeffic = coeff.reshape((n, len(idx_data) - 1), order='F').copy().T

    # Initialize variables for storing the fitted values and the indices of all data points
    C = []
    idx_all = []

    # Perform fitting for each window
    for i in range(coeffic.shape[0]):
        si = idx_data[i]
        li = idx_data[i + 1]
        t_process = t[si-1:li]

        # Normalize t_process to the range [-1, 1]
        t_new = (t_process - np.min(t_process)) / (np.max(t_process) - np.min(t_process)) * 2 - 1

        # Compute the orthogonal polynomials for the given degree and normalized t_process
        # You will have to define or import the orth_poly_diff_mat function to use here
        polynomials_var = orth_order_diff_mat(n, t_new, 0)
        
        # Compute the fitted values by multiplying the coefficients with the orthogonal polynomials
        c_interm = np.dot(coeffic[i, :], polynomials_var)
        
        # Exclude the last data point and fitted value if it belongs to the last window
        if i < coeffic.shape[0] - 1:
            t_process = t_process[:-1]
            c_interm = c_interm[:-1]
        
        # Concatenate the indices of all data points and the fitted values
        idx_all.extend(t_process.flatten())
        C.extend(c_interm)

    # Assign the fitted values to the output variable X
    return np.array(C)
