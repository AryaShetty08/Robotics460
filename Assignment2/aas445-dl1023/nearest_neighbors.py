import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

def nearest_neighbors(robot, target, k, configs):
    """
    Given input find the three nearest robot configurations to the target 

    Input:
    - robot - Either arm or freebody to define the robot type
    - target - N numbers that define the robot's initial configuration
    - k - Integer that defines number of nearest neighbors to output
    - configs - Filename that contains configurations

    Returns:
    - K nearest robot configs, visualized by showing k + 1 robots 
    """

    #traget is a tuple??
    

    configurations = []
    with open(configs, "r") as file:
        for line in file:
            tupleValues = tuple(map(float, line.split(' ')))
            configurations.append(tupleValues)
    nearestConfigs = []

    if robot == "arm":
        for i in range(len(configurations)):
            distance = math.sqrt(math.pow((target[0] - configurations[i][0]), 2) + math.pow((target[1] - configurations[i][1]), 2))
            nearestConfigs.append((distance, i))


    elif robot == "freeBody":
        #ax.set_zlim(0, 20)
        #ax.set_zlabel('Z')
        for i in range(len(configurations)):
            distance = math.sqrt(math.pow((target[0] - configurations[i][0]), 2) + math.pow((target[1] - configurations[i][1]), 2) + math.pow((target[2] - configurations[i][2]), 2))
            nearestConfigs.append((distance, i))
    
    sortedConfigs = sorted(nearestConfigs, key=lambda x: x[0])
    nearestConfigs = []
    for i in range(k):
        nearestConfigs.append(configurations[sortedConfigs[i][1]])

    # visualize the stuff for nearestConfigs + target on plot
    
    if robot == "arm":
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        x_values, y_values = zip(*nearestConfigs)
        ax.scatter(x_values, y_values)
        ax.grid(True) 
    elif robot == "freeBody":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_values, y_values, z_values = zip(*nearestConfigs)
        ax.scatter(x_values, y_values, z_values)  

    ax.set_title('KNN Visualization')
    plt.show()

    return nearestConfigs

if __name__ == "__main__":
    testRobot = "freeBody"
    testTarget = (0, 0, 0)
    k = 2
    testFile = "configs.txt"

    print(nearest_neighbors(testRobot, testTarget, k, testFile))