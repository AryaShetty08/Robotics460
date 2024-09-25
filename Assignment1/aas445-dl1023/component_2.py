import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from component_1 import check_SOn, check_quaternion

def random_rotation_matrix(naive):
     """
    Generates random rotation matrix that satisifies conditions for Special Euclidean Groups

    Input:
    - naive: boolean that determines naive solution or not 

    Returns:
    - matrix: Returns valid, randomly generated rotation matrix 
    """
     if naive:
         # Generate random euler angles
         alpha = np.random.uniform(0, 2*np.pi)
         beta = np.random.uniform(0, 2*np.pi)
         gamma = np.random.uniform(0, 2*np.pi)

         # Create rotation matrices
         rotationX = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]
         rotationY = [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
         rotationZ = [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]

         m = np.matmul(rotationX, rotationY)
         m = np.matmul(m, rotationZ)

         # Already satisfies SO(n) conditions since its using rotation matrices
         return m
            
     else:
         # Followed algorithm in paper
         m = np.zeros((3, 3))
         # Runs until condition met
         while not check_SOn(m):
            x1, x2, x3 = np.random.uniform(0, 1, 3)

            v = [np.cos(2*np.pi*x2)*np.sqrt(x3), np.sin(2*np.pi*x2)*np.sqrt(x3), np.sqrt(1-x3)]
            v = np.reshape(v, (3,1))

            mReflection = 2*np.dot(v,np.transpose(v)) - np.eye(3)
            rotationZ = [[np.cos(2*np.pi*x1), -np.sin(2*np.pi*x1), 0], [np.sin(2*np.pi*x1), np.cos(2*np.pi*x1), 0], [0, 0, 1]]

            m = np.dot(mReflection, rotationZ)
         return m

def random_quaternion(naive):
    """
    Generates random quaternion that satisifies conditions 

    Input:
    - naive: boolean that determines naive solution or not 

    Returns:
    - vector: Returns valid, randomly generated rotation quaternion

    Input: A boolean that defines how the rotation is generated. If naive is true, implement a naive solution (for example, random euler angles and convert to rotation matrix). If naive is false, implement the function as defined in Algorithm 2 in [2]. Return: A randomly generated element q ∈ S3.
    """
    if naive:
        # Generate random euler angles
        alpha = np.random.uniform(0, 2*np.pi)
        beta = np.random.uniform(0, 2*np.pi)
        gamma = np.random.uniform(0, 2*np.pi)

        # Create rotation matrices
        rotationX = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]
        rotationY = [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
        rotationZ = [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]

        m = np.matmul(rotationX, rotationY)
        m = np.matmul(m, rotationZ)

        # Convert to quaternion
        q = np.zeros(4)
        q[0] = np.sqrt(1 + m[0,0] + m[1,1] + m[2,2]) / 2
        q[1] = (m[2,1] - m[1,2]) / (4*q[0])
        q[2] = (m[0,2] - m[2,0]) / (4*q[0])
        q[3] = (m[1,0] - m[0,1]) / (4*q[0])

        return q
    else:
        q = np.zeros(4)
        # Runs until condition met
        while not check_quaternion(q):
            x1, x2, x3 = np.random.uniform(0, 1, 3)

            q[0] = np.sqrt(1-x1)*np.sin(2*np.pi*x2)
            q[1] = np.sqrt(1-x1)*np.cos(2*np.pi*x2)
            q[2] = np.sqrt(x1)*np.sin(2*np.pi*x3)
            q[3] = np.sqrt(x1)*np.cos(2*np.pi*x3)
        return q
        
def quaternion_to_rotation_matrix(q):
    """
    Converts quaternion to rotation matrix

    Input:
    - q: quaternion

    Returns:
    - matrix: rotation matrix
    """
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    matrix = np.array([[1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2], [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1], [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]])
    return matrix

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(1000):
        rotationMatrix = random_rotation_matrix(False)
        startVector = [0, 1, 0]
        v = [0.1, 0, 0]
        rotatedStart = np.matmul(rotationMatrix, startVector)
        rotated_v = np.matmul(rotationMatrix, v)
        #ax.quiver(startVector[0], startVector[1], startVector[2], v[0], v[1], v[2], color='b')
        ax.quiver(rotatedStart[0], rotatedStart[1], rotatedStart[2], rotated_v[0], rotated_v[1], rotated_v[2], color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title('Random Rotation Visualization')
    plt.show()

    # plots the quaternion

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')

    for i in range(1000):
        quaternion = random_quaternion(False)
        rotationMatrix = quaternion_to_rotation_matrix(quaternion)
        startVector = [0, 1, 0]
        v = [0.1, 0, 0]
        rotatedStart = np.matmul(rotationMatrix, startVector)
        rotated_v = np.matmul(rotationMatrix, v)
        #ax.quiver(startVector[0], startVector[1], startVector[2], v[0], v[1], v[2], color='b')
        ax.quiver(rotatedStart[0], rotatedStart[1], rotatedStart[2], rotated_v[0], rotated_v[1], rotated_v[2], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title('Random Quaternion Visualization')
    plt.show()