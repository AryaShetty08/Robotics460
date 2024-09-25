import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from component_1 import check_SOn

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
        w = np.sqrt(1 + m[0,0] + m[1,1] + m[2,2])/2
        x = (m[2,1] - m[1,2])/(4*w)
        y = (m[0,2] - m[2,0])/(4*w)
        z = (m[1,0] - m[0,1])/(4*w)

        return [w, x, y, z]
    else:
        # Followed algorithm in paper
        while True:
            x1, x2, x3 = np.random.uniform(0, 1, 3)
            w = np.sqrt(1-x1)*np.sin(2*np.pi*x2)
            x = np.sqrt(1-x1)*np.cos(2*np.pi*x2)
            y = np.sqrt(x1)*np.sin(2*np.pi*x3)
            z = np.sqrt(x1)*np.cos(2*np.pi*x3)

            if np.linalg.norm([w, x, y, z]) == 1:
                return [w, x, y, z]

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
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(500):
        [w, x, y, z] = random_quaternion(False)
        [w1, x1, y1, z1] = random_quaternion(False)
        # Normalize the vector part (x, y, z) to ensure it's a unit vector on the sphere
        vec = np.array([x, y, z])
        vec /= np.linalg.norm(vec)  # Normalize to unit length
    
        # Generate a random origin in 3D space
        origin = np.random.uniform(-1, 1, 3)

        # Plot vector originating from the random origin and pointing outwards
        ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], color='b', length=0.2)

    #print(w, x, y, z)

    # plots the quaternion
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title('Random Quaternion Visualization')
    plt.show()