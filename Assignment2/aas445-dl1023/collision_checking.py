import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

def getCorners(obstacle):

        x = obstacle[0]
        y = obstacle[1]
        theta = obstacle[2]
        w = obstacle[3]
        h = obstacle[4]

        obsV1 = ((x-(w/2)), (y+(h/2)))
        obsV2 = ((x+(w/2)), (y+(h/2)))
        obsV3 = ((x-(w/2)), (y-(h/2)))
        obsV4 = ((x+(w/2)), (y-(h/2)))

        corners = [obsV1, obsV2, obsV3, obsV4]
        rotatedCorners = []

        for i in range(len(corners)):
            # corner position from center
            x_rel = corners[i][0] - x
            y_rel = corners[i][1] - y

            #apply rotation matrix for rotated obstacles
            x_rot = math.cos(theta) * x_rel - math.sin(theta) * y_rel + x
            y_rot = math.sin(theta) * x_rel + math.cos(theta) * y_rel + y

            rotatedCorners.append((x_rot, y_rot))

        return rotatedCorners
 
def getProjection(axes, corners):
    a = np.array([axes[0], axes[1]])

    min = math.inf
    max = -math.inf

    for i in range(len(corners)):
        b = np.array([corners[i][0], corners[i][1]])
        product = np.dot(a, b)
        if product > max:
            max = product
        if product < min:
            min = product

    return min, max

def checkCollision(obstacle, env):

    if len(env) == 0:
        return False
    
    obsCorners = getCorners(obstacle)

    obsEdges = [(obsCorners[1][0] - obsCorners[0][0], obsCorners[1][1] - obsCorners[0][1]), 
                (obsCorners[2][0] - obsCorners[0][0], obsCorners[2][1] - obsCorners[0][1])]
    indexHit = []

    
    for i in range(len(env)):
        checkCorners = getCorners(env[i])
        checkEdges = [(checkCorners[1][0] - checkCorners[0][0], checkCorners[1][1] - checkCorners[0][1]),
            (checkCorners[2][0] - checkCorners[0][0], checkCorners[2][1] - checkCorners[0][1])]

        normalVectors = []

        for j in range(len(obsEdges)):
            mag = math.sqrt(math.pow(-obsEdges[j][1], 2) + math.pow(obsEdges[j][0], 2))
            normalVectors.append((-obsEdges[j][1] / mag, obsEdges[j][0] / mag))

        for j in range(len(checkEdges)):
            mag = math.sqrt(math.pow(-checkEdges[j][1], 2) + math.pow(checkEdges[j][0], 2))
            normalVectors.append((-checkEdges[j][1] / mag, checkEdges[j][0] / mag))

        collision = True
        # project the corners onto the axis
        for j in range(len(normalVectors)):
            min1, max1 = getProjection(normalVectors[j], checkCorners)
            minObs, maxObs = getProjection(normalVectors[j], obsCorners) 
            #print(normalVectors)
            if max1 < minObs or maxObs < min1:
                collision = False
                break
            
        if collision:
            print("Hit")
            indexHit.append(i)
            #check for collision here 

        #normalVectors = normalVectors[:-2]
        # get rid of last two vectors for the next two vectors that will appear

    return indexHit

def generate_obstacle(width, height):
    
    w = width
    h = height
    
    x = float(f"{random.uniform(w/2, 20 - w/2):.2f}") 
    y = float(f"{random.uniform(h/2, 20 - h/2):.2f}") 

    theta = float(f"{random.uniform(0, 2*math.pi):.2f}")  
    
    return (x,y,theta,w,h)

# Different generation of configs for arm robots
def generate_arm_configs():
    l1, l2 = 2.0, 1.5 
    theta1 = random.uniform(0, 2*math.pi)
    theta2 = random.uniform(0, 2*math.pi)
    
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    
    # Use the joint angle sum for the rectangle orientation
    theta = theta1 + theta2
    
    return (x + 10, y + 10, theta, 0.5, 0.3)  # Centered in workspace so not at origin because it would be negative possibly

def visualize_all_configs(configs, env, all_hits):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('All Configurations Collision Visualization')
    ax.grid(True)
    
    # Plot environment obstacles
    for i, obstacle in enumerate(env):
        x, y, theta, w, h = obstacle
        color = 'green'  # Default color
        
        # Check if this obstacle was hit in any configuration
        for hits in all_hits:
            if i in hits:
                color = 'red'
                break
                
        t = Affine2D().rotate_around(x, y, theta) + ax.transData
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, color=color, alpha=0.5, transform=t)
        ax.add_patch(rect)
    
    # Plot all robot configurations with different alpha values
    for i, robot in enumerate(configs):
        x, y, theta, w, h = robot
        t = Affine2D().rotate_around(x, y, theta) + ax.transData
        alpha = 0.3 + (0.7 * i / len(configs))  # Varying transparency
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, color='blue', alpha=alpha, transform=t)
        ax.add_patch(rect)
    
    plt.show()

def collision_checking(robot, map):
    """
    Given a robot dimensions and a predefined enviornment with obstacles
    determines collision in map

    Input:
    - robot - Either "arm" or "freeBody" robot 
    - map - Environment that is predefined

    Returns:
    - Visualization of robot either colliding with obstacles or not
    """

    configs = []
    all_hits = []

    map = load_map(map)
    
    for n in range(10):
        if robot == "freeBody":
            obstacle = generate_obstacle(0.5, 0.3)
        else:  # arm
            obstacle = generate_arm_configs()
            
        configs.append(obstacle)
        hits = checkCollision(obstacle, map)
        all_hits.append(hits)
        print(f"Configuration {n}: {obstacle}")
        print(f"Collisions with obstacles: {hits}\n")
    
    visualize_all_configs(configs, map, all_hits)

def load_map(filename):
    """
    Load the map from a file

    Input:
    - filename - Name of the file to load the map from

    Returns:
    - List of obstacles in the map
    """
    env = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().strip('()').split(',')
            x, y, theta, w, h = [float(i) for i in line]
            env.append((x, y, theta, w, h))
    return env

if __name__ == "__main__":
    # testBot = "freeBody"
    # testEnv = [(4.94, 5.36, 3.29, 1.37, 1.87), (14.77, 11.11, 2.41, 1.14, 1.55), (6.61, 9.62, 5.21, 0.78, 1.53), 
    #            (17.37, 2.0, 1.55, 0.58, 1.88), (5.92, 14.2, 4.18, 0.72, 0.64), (9.77, 18.92, 2.61, 0.67, 0.85), 
    #            (1.05, 11.62, 3.33, 1.63, 0.97), (19.58, 10.42, 5.48, 0.72, 1.05), (7.72, 0.93, 1.33, 1.76, 0.74), (14.36, 18.27, 5.43, 1.31, 1.03)]
    
    # # Test both robot types
    # print("Testing freeBody robot:")
    # collision_checking("freeBody", testEnv)
    
    # print("\nTesting arm robot:")
    # collision_checking("arm", testEnv)

    parser = argparse.ArgumentParser(description='Collision Checking')
    parser.add_argument('--robot', type=str, choices=['arm', 'freeBody'], required=True)
    parser.add_argument('--map', type=str, required=True)

    args = parser.parse_args()

    collision_checking(args.robot, args.map)
