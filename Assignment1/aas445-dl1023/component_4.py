import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

L1 = 2
L2 = 1.5

#same as rigid body calc
def slerp(r1, r2, t):

    R_rel = np.dot(r1, r2.T)
    
    theta_rel = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    
    theta_interp = theta_rel * t
    
    new_matrix = np.array([[np.cos(theta_interp), -np.sin(theta_interp)], [np.sin(theta_interp),  np.cos(theta_interp)]])

    R_interp = np.dot(new_matrix, r1)
    
    return R_interp

def interpolate_arm(start, goal):
     """
    Gives path for the robot to traverse from thes start_pose to goal_pose

    Input:
    - start: tuple (first_theta0, second_theta0) that determines robots starting configuration
    - goal: tuple (first_thetaG, second_thetaG) that determines robots ending configuration

    Returns:
    - path: sequence of poses
    """
     theta1_s, theta2_s = start
     theta1_g, theta2_g = goal

     r1_start =  np.array([[np.cos(theta1_s), -np.sin(theta1_s)], [np.sin(theta1_s),  np.cos(theta1_s)]])
     r1_goal =  np.array([[np.cos(theta1_g), -np.sin(theta1_g)], [np.sin(theta1_g),  np.cos(theta1_g)]])

     r2_start =  np.array([[np.cos(theta2_s), -np.sin(theta2_s)], [np.sin(theta2_s),  np.cos(theta2_s)]])
     r2_goal =  np.array([[np.cos(theta2_g), -np.sin(theta2_g)], [np.sin(theta2_g),  np.cos(theta2_g)]])

     path = []
     steps = 10

     for i in range(steps):
          t = i / (steps - 1)

          r1_interp = slerp(r1_start, r1_goal, t)
          r2_interp = slerp(r2_start, r2_goal, t)

          theta1_interp = np.arctan2(r1_interp[1,0], r1_interp[0,0])
          theta2_interp = np.arctan2(r2_interp[1,0], r2_interp[0,0])

          path.append((theta1_interp, theta2_interp))

     return path

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
