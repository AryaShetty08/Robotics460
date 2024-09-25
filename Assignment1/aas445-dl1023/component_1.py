import numpy as np

epsilon = 0.01

def check_SOn(m):
    """
    Checks if the matrix satisifies conditions for Special Orthogonal Groups

    Input:
    - m: matrix

    Returns:
    - boolean: Returns True if satisfies conditions, False if not 
    """

    # Make sure input is matrix
    if len(m.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")
    
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
    Checks if the matrix satisifies conditions for Quaternions

    Input:
    - v: vector

    Returns:
    - boolean: Returns True if satisfies conditions, False if not 
    """

    """
    The conditions for a vector to be in Quaternions are:
    1. The vector is a unit vector
    2. The vector is of length 4
    """

    # Make sure input is vector
    if len(v.shape) != 1:
        raise ValueError("Input must be a 1D vector.")
    
    # Check if vector is of length 4
    if len(v) != 4:
        return False
    
    # Check if vector is a unit vector
    if not np.allclose(np.linalg.norm(v), 1, atol=epsilon):
        return False

    return True

def check_SEn(m):
    """
    Checks if the matrix satisifies conditions for Special Euclidean Groups

    Input:
    - m: matrix

    Returns:
    - boolean: Returns True if satisfies conditions, False if not 
    """
    
    # Check whether it is a rotation for 2D or 3D 

    # Make sure input is matrix
    if len(m.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")
    
    if len(m) == 3:
        rotationMatrix = m[:2,:2]
        # Check rotation matrix
        if check_SOn(rotationMatrix):
            translationVector = m[:2, 2]
            # Check bottom row for 0 and 1
            bottomRow = m[2, :]
            if np.allclose(bottomRow, [0, 0, 1], atol=0):
                return True    
    elif len(m) == 4:
        rotationMatrix = m[:3, :3]
        if check_SOn(rotationMatrix):
            translationVector = m[:3, 3]
            # Check bottom row for 0 and 1
            bottomRow = m[3, :]
            if np.allclose(bottomRow, [0, 0, 0, 1], atol=0):
                return True

    return False

def correct_SOn(m):
    # Make sure input is matrix
    if len(m.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")
    
    # Check transpose of matrix 
    mTranspose = np.transpose(m)
    result = np.dot(m, mTranspose)

    if not (np.allclose(result, np.identity(len(m)), atol=epsilon)):
        # Correct the matrix
        mCorrected = np.dot(m, np.linalg.inv(result))
        return mCorrected

    # Check if determinant is equal to 1 
    mDeterminant = np.linalg.det(m)
    
    if abs(mDeterminant - 1) > epsilon:
        # Correct the matrix
        mCorrected = np.dot(m, np.linalg.inv(mDeterminant))
        return mCorrected

    return m

def correct_quaternion(v):
    # Make sure input is vector
    if len(v.shape) != 1:
        raise ValueError("Input must be a 1D vector.")
    
    # Check if vector is of length 4
    if len(v) != 4:
        return False
    
    # Check if vector is a unit vector
    if not np.allclose(np.linalg.norm(v), 1, atol=epsilon):
        # Correct the vector
        vCorrected = v / np.linalg.norm(v)
        return vCorrected

    return v

def correct_SEn(m):
    # Check whether it is a rotation for 2D or 3D 

    # Make sure input is matrix
    if len(m.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")
    
    if len(m) == 3:
        rotationMatrix = m[:2,:2]
        # Check rotation matrix
        if check_SOn(rotationMatrix):
            translationVector = m[:2, 2]
            # Check bottom row for 0 and 1
            bottomRow = m[2, :]
            if np.allclose(bottomRow, [0, 0, 1], atol=0):
                return m
            else:
                # Correct the matrix
                m[2, :] = [0, 0, 1]
                return m
        else: 
            rotationMatrix = correct_SOn(rotationMatrix)
            m[:2,:2] = rotationMatrix
            # Check bottom row for 0 and 1
            bottomRow = m[2, :]
            if np.allclose(bottomRow, [0, 0, 1], atol=0):
                return m
            else:
                # Correct the matrix
                m[2, :] = [0, 0, 1]
                return m 
    elif len(m) == 4:
        rotationMatrix = m[:3, :3]
        if check_SOn(rotationMatrix):
            translationVector = m[:3, 3]
            # Check bottom row for 0 and 1
            bottomRow = m[3, :]
            if np.allclose(bottomRow, [0, 0, 0, 1], atol=0):
                return m
            else:
                # Correct the matrix
                m[3, :] = [0, 0, 0, 1]
                return m
        else:
            rotationMatrix = correct_SOn(rotationMatrix)
            m[:3, :3] = rotationMatrix
            # Check bottom row for 0 and 1
            bottomRow = m[3, :]
            if np.allclose(bottomRow, [0, 0, 0, 1], atol=0):
                return m
            else:
                # Correct the matrix
                m[3, :] = [0, 0, 0, 1]
                return m

    return m

if __name__ == '__main__':
    m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(check_SOn(m))
    print(correct_SOn(m))

    v = np.array([1, 0, 0, 0])
    print(check_quaternion(v))
    print(correct_quaternion(v))

    m = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    print(check_SEn(m))
    print(correct_SEn(m))