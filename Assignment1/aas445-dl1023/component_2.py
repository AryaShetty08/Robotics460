import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def random_rotation_matrix(naive):
     """
    Checks if the matrix satisifies conditions for Special Euclidean Groups

    Input:
    - naive: boolean that determines naive solution or not 

    Returns:
    - matrix: Returns valid, randomly generated rotation matrix 
    """
    
     alpha = np.degrees(np.random.uniform(0, 2*np.pi))
     beta = np.degrees(np.random.uniform(0, 2*np.pi))
     gamma = np.degrees(np.random.uniform(0, 2*np.pi))

     testVector =  [0, 0, 1]
     x, y, z = testVector
     #zip(*vectors)

     #if naive:
     rotationX = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]
     rotationY = [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
     rotationZ = [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]

     m = np.matmul(rotationX, rotationY)
     m = np.matmul(m, rotationZ)


          
     #else:


     #return m

if __name__ == "__main__":
     testVectors =  [[0, 0, 1], [0, 0, 2]]
     x, y, z = zip(*testVectors)
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     #ax.set_xlim([-1,1])
     ax.quiver(0, 0, 0, x, y, z, color='b')
     ax.set_xlabel('X')
     ax.set_ylabel('Y')
     ax.set_zlabel('Z')
     ax.set_title('Random Rotation Visualization')
     plt.show()