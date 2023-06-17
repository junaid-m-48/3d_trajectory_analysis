import numpy as np

def orth_order_diff_mat(n, t, deriv, m=None):
    """
    Compute orthogonal Legendre polynomials and their derivatives.
    
    This function computes orthogonal Legendre polynomials and their derivatives with respect
    to the variable `t`. It uses explicit formulas for the polynomials and their derivatives
    up to a certain degree. 
    
    Parameters:
    -----------
    n : int
        The number of Legendre polynomials to be computed.
    t : array-like
        The values at which the Legendre polynomials and their derivatives are evaluated.
    deriv : int
        The order of the derivative. 0 for the polynomial itself, 1 for the first derivative, etc.
    m : int, optional
        If specified, the function returns only the m-th row of the computed matrix.
        
    Returns:
    --------
    R : numpy array
        If `m` is None, a 2D array of size (n, len(t)) containing the values of the Legendre
        polynomials and their derivatives evaluated at the points in `t`. Each row corresponds
        to a different polynomial or derivative.
        If `m` is specified, a 2D array of size (1, len(t)) containing the values of the m-th
        polynomial or derivative.
        
    Example:
    --------
    >>> t = [0.1, 0.2]
    >>> deriv = 1
    >>> n = 5
    >>> orth_order_diff_mat(n, t, deriv)
    """
    t = np.array(t, dtype=float) # Make sure t is an array to allow element-wise operations

    if deriv == 0:
        R = np.vstack([
            t**0,
            t,
            0.5 * (3 * t**2 - 1),
            0.5 * (5 * t**3 - 3 * t),
            (1/8) * (35 * t**4 - 30 * t**2 + 3),
            (1/8) * (63 * t**5 - 70 * t**3 + 15 * t),
            (1/16) * (231 * t**6 - 315 * t**4 + 105 * t**2 - 5),
            (1/16) * (429 * t**7 - 693 * t**5 + 315 * t**3 - 35 * t),
            (1/128) * (6435 * t**8 - 12012 * t**6 + 6930 * t**4 - 1260 * t**2 + 35),
            (1/128) * (12155 * t**9 - 25740 * t**7 + 18018 * t**5 - 4620 * t**3 + 315 * t),
            (1/256) * (46189 * t**10 - 109395 * t**8 + 90090 * t**6 - 30030 * t**4 + 3465 * t**2 - 63)
        ])
    elif deriv == 1:
        R = np.vstack([
            0 * t**0,
            t**0,
            3 * t,
            7.5 * t**2 - 1.5,
            17.5 * t**3 - 7.5 * t,
            (315 * t**4)/8 - (105 * t**2)/4 + 15/8,
            (693 * t**5)/8 - (315 * t**3)/4 + (105 * t)/8,
            (3003 * t**6)/16 - (3465 * t**4)/16 + (945 * t**2)/16 - 35/16,
            (6435 * t**7)/16 - (9009 * t**5)/16 + (3465 * t**3)/16 - (315 * t)/16,
            (109395 * t**8)/128 - (45045 * t**6)/32 + (45045 * t**4)/64 - (3465 * t**2)/32 + 315/128,
            (230945 * t**9)/128 - (109395 * t**7)/32 + (135135 * t**5)/64 - (15015 * t**3)/32 + (3465 * t)/128
        ])
    elif deriv == 2:
        R = np.vstack([
            0 * t**0,
            0 * t**0,
            3 * t**0,
            15 * t,
            52.5 * t**2 - 7.5,
            157.5 * t**3 - 52.5 * t,
            (3465 * t**4)/8 - (945 * t**2)/4 + 105/8,
            (9009 * t**5)/8 - (3465 * t**3)/4 + (945 * t)/8,
            (45045 * t**6)/16 - (45045 * t**4)/16 + (10395 * t**2)/16 - 315/16,
            (109395 * t**7)/16 - (135135 * t**5)/16 + (45045 * t**3)/16 - (3465 * t)/16,
            (2078505 * t**8)/128 - (765765 * t**6)/32 + (675675 * t**4)/64 - (45045 * t**2)/32 + 3465/128
        ])
    elif deriv == 3:
        R = np.vstack([
            0 * t**0,
            0 * t**0,
            0 * t**0,
            15 * t**0,
            105 * t,
            472.5 * t**2 - 52.5,
            1732.5 * t**3 - 472.5 * t,
            (45045 * t**4)/8 - (10395 * t**2)/4 + 945/8,
            (135135 * t**5)/8 - (45045 * t**3)/4 + (10395 * t)/8,
            (765765 * t**6)/16 - (675675 * t**4)/16 + (135135 * t**2)/16 - 3465/16,
            (2078505 * t**7)/16 - (2297295 * t**5)/16 + (675675 * t**3)/16 - (45045 * t)/16
        ])
    elif deriv == 4:
        R = np.vstack([
            0 * t**0,
            0 * t**0,
            0 * t**0,
            0 * t**0,
            105 * t**0,
            945 * t,
            5197.5 * t**2 - 472.5,
            22522.5 * t**3 - 5197.5 * t,
            (675675 * t**4)/8 - (135135 * t**2)/4 + 10395/8,
            (2297295 * t**5)/8 - (675675 * t**3)/4 + (135135 * t)/8,
            (14549535 * t**6)/16 - (11486475 * t**4)/16 + (2027025 * t**2)/16 - 45045/16
        ])

    if m is None:
        return R[:n, :]
    else:
        return R[m-1:m, :]

# Example usage:
# t = [0.1, 0.2]
# deriv = 1
# n = 5
# print(orth_poly_diff_mat(n, t, deriv))
