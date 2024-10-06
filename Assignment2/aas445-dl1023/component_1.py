import random
from math import pi

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

        theta = float(f"{random.uniform(0, 2*pi):.2f}")  
        
        return (x,y,theta,w,h)


    def checkCollision(obstacle, env):

        if len(env) == 0:
            return False
        
        for i in range(len(env)):
            curr = env[i]
            # have to account for theta???
            xBounds = [curr[0]-curr[2]/2, curr[0]+curr[2]/2]
            yBounds = [curr[1]-curr[3]/2, curr[1]+curr[3]/2]

            xObstacleBounds = [obstacle[0]-obstacle[2]/2, obstacle[0]+obstacle[2]/2]
            yObstacleBounds = [obstacle[1]-obstacle[3]/2, obstacle[1]+obstacle[3]/2]

            if any(element in xObstacleBounds for element in xBounds):
             return True

            if any(element in yObstacleBounds for element in yBounds):
             return True

        return False


    for i in range(number_of_obstacles):

        obstacle = generate_obstacle()

        while(checkCollision(obstacle, env)):
            obstacle = generate_obstacle
       
        env.append(obstacle)

    return env


def scene_to_file(env, filename):
    return

def scene_from_file(filename):
    return env

def visualize_scene(env):
    return

if __name__ == "__main__":
    num = 3
    print(generate_environment(num))
