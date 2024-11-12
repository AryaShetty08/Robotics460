import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from matplotlib.patches import Rectangle
import time

# define the potetntials
# attractive
# repulsive
# calculate and use gradient to find force for robot to move this is in 2d no rotation 
# and implement

# this one is for goal
def attractive_potential(robot_pose, goal):
    # they used a quadratic here??
    att_pot = 0.5 * ((robot_pose[0] - goal[0])**2 + (robot_pose[1] - goal[1])**2)
    return att_pot

# for the obstacles
def repulsive_potential(robot_pose, obs):
    # they used inverse function?
    dist = max(0.01, ((robot_pose[0] - obs[0])**2 + (robot_pose[1] - obs[1])**2))
    return

def total_potential(robot_pose, goal, obstacles):
    u_att = attractive_potential(robot_pose, goal)
    u_rep = 0
    for obs in obstacles:
        u_rep += repulsive_potential(robot_pose, obs)
    return u_att + u_rep

# gradient


def main():
    print("hello")
    parser = argparse.ArgumentParser(description="Potential Function")

    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)

    args = parser.parse_args()
    
    # Convert arguments to numpy arrays
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)
    
if __name__ == "__main__":
    main()