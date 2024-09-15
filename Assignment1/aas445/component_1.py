import numpy as np

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

    if result != np.identiy(len(m)):
        return False

    # Check if determinant is equal to 1 
    mDeterminant = np.linalg.det(m)

    if mDeterminant != 1:
        return False

    return True