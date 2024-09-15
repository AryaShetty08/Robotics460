import numpy as np

epsilon = 0.01

#What is epsilon??
def check_SOn(m):
    """
    Checks if the matrix satisifies conditions for Special Orthogonal Groups

    Input:
    - m: matrix

    Returns:
    - boolean: Returns True if satisfies conditions, False if not 
    """

    # Check transpose of matrix 
    mTranspose = np.transpose(m)
    result = np.dot(m, mTranspose)

    if not (np.allclose(result, np.identity(len(m)), atol=epsilon)):
        return False

    # Check if determinant is equal to 1 
    mDeterminant = np.linalg.det(m)
    
    if abs(mDeterminant - 1) > epsilon:
        return False

    return True

def check_quaternion(v):
    """
    Checks if the matrix satisifies conditions for Quarternions

    Input:
    - v: vector

    Returns:
    - boolean: Returns True if satisfies conditions, False if not 
    """

    return True

def check_SEn(m):
    """
    Checks if the matrix satisifies conditions for Special Euclidean Groups

    Input:
    - m: matrix

    Returns:
    - boolean: Returns True if satisfies conditions, False if not 
    """

    return True