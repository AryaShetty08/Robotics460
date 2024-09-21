import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def interpolate_arm(start, goal):
     """
    Gives path for the robot to traverse from thes start_pose to goal_pose

    Input:
    - start: tuple (first_theta0, second_theta0) that determines robots starting configuration
    - goal: tuple (first_thetaG, second_thetaG) that determines robots ending configuration

    Returns:
    - path: sequence of poses
    """
     
     
def forward_propagate_arm(start_pose, plan):
     """
    Gives path robot takes based on the start pose and plan

    Input:
    - start_pose: tuple (first_theta0, second_theta0) that determines robots starting configuration
    - plan: sequence of N tuples (velocity, duration)

    Returns:
    - path: sequence of poses
    """
     
          
def visualize_path(path):
     """
    Visualizes path and animates robot's arm movement

    Input:
    - path: sequence of poses 

    Returns:
    - visualization
    """
     
if __name__ == "__main__":
     start_pose =  (0, 0, 0)
