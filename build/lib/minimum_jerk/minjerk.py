"""Minimum jerk algorithm.
"""
import numpy as np
from typing import List

import minjerk_analytic as mja
import minjerk_penalty as mjp
from mytypes import Array, Array2D

def shape_check(*arrays: np.ndarray) -> List:
    """
    Validate and convert input arrays to 2D arrays, 
    and check if all arrays have the same dimensions.
    
    parameters:
    -----------
    *arrays : a variable number of numpy arrays to be validated and converted.
    
    returns:
    --------
    converted_arrays: a list of 2D numpy arrays with the same shape if all inputs have consistent dimensions.
        
    raises:
    -------
    ValueError
        If the input arrays have inconsistent dimensions.
    """
    converted_arrays = []
    target_shape = None

    for array in arrays:
        # Check if the input is a numpy array
        if not isinstance(array, np.ndarray):
            raise TypeError("All inputs must be numpy arrays.")

        # Convert to 2D if necessary
        if array.ndim == 1:
            array = array[np.newaxis, :]  # Convert 1D array to 2D (1, n)

        # Check if all arrays have the same shape
        if target_shape is None:
            target_shape = array.shape
        elif array.shape != target_shape:
            raise ValueError("All input arrays must have the same shape.")

        converted_arrays.append(array)
    return converted_arrays

def is_integer_multiple(a: float, b: float, 
                        tolerance=1e-12):
    """
    Check if float a is an integer multiple of float b.

    parameters:
    -----------
    a: the number to be checked
    b: the reference number
    tolerance: the tolerance for floating-point comparison

    returns:
    --------
    p: the integer multiple
    bool: true if a is an integer multiple of b, otherwise False
    """
    if b == 0:
        raise ValueError(f"Divisor cannot be zero.")
    # Calculate the ratio
    ratio = a / b
    # Check if the ratio is approximately an integer
    return abs(round(ratio) - ratio) < tolerance

def integer_check(T: Array, dt: float) -> None:
    """Check if every number in T is an integer multiple of dt
    """
    for i in range(len(T)):
        t = T[i]
        if not is_integer_multiple(t, dt):
            raise ValueError(f'T is not an integer multiple of dt!') 

def minimum_jerk_trajectory(X: Array2D, V: Array2D, 
                            A: Array2D, T: Array,
                            dt: float, N: Array2D=None):
    """Using analytic solver or penalty solver based on the
    value of N.
    """
    integer_check(T, dt)
    X, V, A = shape_check(X, V, A)

    if N is None:
        return mja.path_planning(X, V, A, T, dt)
    else:
        if N.ndim == 1:
            N = N.reshape(1, -1)
        if N.shape[1] != (X.shape[1]-1):
            raise ValueError(f'Point and constraint dimensions do not match')
        return mjp.path_planning(X, V, A, T, dt, N)