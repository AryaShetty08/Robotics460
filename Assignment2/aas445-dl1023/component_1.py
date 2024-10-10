import random
import math
import numpy as np

def generate_environment(number_of_obstacles):
    """
    Generate random 20x20 environment with defined number of obstacles

    Input:
    - number_of_obstacles

    Returns:
    - env - 20x20 grid with obstacles
    """

    # 20x20 environment is defined by [0, 20] [0,20]

    # store all the obstacles 
    env = []

    def generate_obstacle():
        
        w = float(f"{random.uniform(0.5, 2):.2f}")
        h = float(f"{random.uniform(0.5, 2):.2f}")
        
        x = float(f"{random.uniform(w/2, 20 - w/2):.2f}") 
        y = float(f"{random.uniform(h/2, 20 - h/2):.2f}") 

        theta = float(f"{random.uniform(0, 2*math.pi):.2f}")  
        
        return (x,y,theta,w,h)

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

        obsEdges = [(obsCorners[1][0] - obsCorners[0][0], obsCorners[1][1] - obsCorners[0][1]), (obsCorners[2][0] - obsCorners[0][0], obsCorners[2][1] - obsCorners[0][1])]
        normalVectors = []

        for i in range(len(obsEdges)):
            mag = math.sqrt(math.pow(-obsEdges[i][1], 2) + math.pow(obsEdges[i][0], 2))
            normalVectors.append((-obsEdges[i][1] / mag, obsEdges[i][0] / mag))
            
       
        for i in range(len(env)):
            checkCorners = getCorners(env[i])
            checkEdges = [(checkCorners[1][0] - checkCorners[0][0], checkCorners[1][1] - checkCorners[0][1]), (checkCorners[2][0] - checkCorners[0][0], checkCorners[2][1] - checkCorners[0][1])]

            for j in range(len(checkEdges)):
                mag = math.sqrt(math.pow(-checkEdges[j][1], 2) + math.pow(checkEdges[j][0], 2))
                normalVectors.append((-checkEdges[j][1] / mag, checkEdges[j][0] / mag))

            # project the corners onto the axis
            for j in range(len(normalVectors)):
               min1, max1 = getProjection(normalVectors[j], checkCorners)
               minObs, maxObs = getProjection(normalVectors[j], obsCorners) 
               print(normalVectors)
               if max1 < minObs or maxObs < min1:
                   return False
               #check for collision here 

            normalVectors = normalVectors[:-2]
            # get rid of last two vectors for the next two vectors that will appear

        return True


    for i in range(number_of_obstacles):

        obstacle = generate_obstacle()

        while(checkCollision(obstacle, env)):
            obstacle = generate_obstacle()
       
        env.append(obstacle)

    return env


def scene_to_file(env, filename):
    return

def scene_from_file(filename):
    return env

def visualize_scene(env):
    return

if __name__ == "__main__":
    num = 2
    print(generate_environment(num))
