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

    #Just have to check unit vector equal to 1?
    # make sure its right vector length 

    return True

def check_SEn(m):
    """
    Checks if the matrix satisifies conditions for Special Euclidean Groups

    Input:
    - m: matrix

    Returns:
    - boolean: Returns True if satisfies conditions, False if not 
    """

    #check the so again, then check the vector make sure its good, then check the bottom row is 0 0 1, easier to fix
    
    # Check whether it is a rotation for 2D or 3D 
    # Check the epsilon for this one !!!

    if len(m) == 3:
        rotationMatrix = m[:2,:2]
        # Check rotation matrix
        if check_SOn(rotationMatrix):
            translationVector = m[:2, 2]
            # Check bottom row for 0 and 1
            bottomRow = m[2, :]
            if bottomRow == [0, 0, 1]:
                return True    
    elif len(m) == 4:
        rotationMatrix = m[:3, :3]
        if check_SOn(rotationMatrix):
            translationVector = m[:3, 3]
            # Check bottom row for 0 and 1
            bottomRow = m[3, :]
            if bottomRow == [0, 0, 0, 1]:
                return True

    return False