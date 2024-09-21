import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from component_1 import check_SOn

def random_rotation_matrix(naive):
     """
    Checks if the matrix satisifies conditions for Special Euclidean Groups

    Input:
    - naive: boolean that determines naive solution or not 

    Returns:
    - matrix: Returns valid, randomly generated rotation matrix 
    """
     if naive:
         alpha = np.random.uniform(0, 2*np.pi)
         beta = np.random.uniform(0, 2*np.pi)
         gamma = np.random.uniform(0, 2*np.pi)

         rotationX = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]
         rotationY = [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
         rotationZ = [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]

         m = np.matmul(rotationX, rotationY)
         m = np.matmul(m, rotationZ)

         #print(check_SOn(m))
         return m
            
     else:
         m = np.zeros((3, 3))
         while not check_SOn(m):
            x1, x2, x3 = np.random.uniform(0, 1, 3)

            v = [np.cos(2*np.pi*x2)*np.sqrt(x3), np.sin(2*np.pi*x2)*np.sqrt(x3), np.sqrt(1-x3)]
            v = np.reshape(v, (3,1))

            mReflection = 2*np.dot(v,np.transpose(v)) - np.eye(3)
            rotationZ = [[np.cos(2*np.pi*x1), -np.sin(2*np.pi*x1), 0], [np.sin(2*np.pi*x1), np.cos(2*np.pi*x1), 0], [0, 0, 1]]

            m = np.dot(mReflection, rotationZ)
         #print(check_SOn(m))
         return m

if __name__ == "__main__":
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')

     for i in range(500):
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