import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D



def interpolate_rigid_body(start_pose, goal_pose):
     """
    Gives path for the robot to traverse from thes start_pose to goal_pose

    Input:
    - start_pose: tuple (x0, y0, theta0) that determines robots starting position and angle
    - goal_pose: tuple (xG, yG, thetaG)that determines robots ending position and angle

    Returns:
    - path: sequence of poses
    """
     
     x0, y0, theta0 = start_pose
     xG, yG, thetaG = goal_pose
     

def forward_propagate_rigid_body(start_pose, plan):
     """
    Gives path robot takes based on the start pose and plan

    Input:
    - start_pose: tuple (x0, y0, theta0) that determines robots starting position and angle
    - plan: sequence of N tuples (velocity, duration)

    Returns:
    - path: sequence of poses
    """
     
def visualize_path(path):
     """
    Visualizes path and animates robot's movement

    Input:
    - path: sequence of poses 

    Returns:
    - visualization
    """