import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Button

L1 = 2
L2 = 1.5

#same as rigid body calc
def slerp(r1, theta_rel, t):
    """
    Uses spherical linear interpolation to smoothly traverse arms of robot

    Input:
    - r1
    - theta_rel
    - t 

    Returns:
    - next_theta: next angle for pose 
    """

    # amount of angle by steps
    theta_interp = theta_rel * t
    # new rotation matrix 
    new_matrix = np.array([[np.cos(theta_interp), -np.sin(theta_interp)], [np.sin(theta_interp),  np.cos(theta_interp)]])
    # angle for pose
    R_interp = np.dot(new_matrix, r1)
    next_theta = np.arctan2(R_interp[1,0], R_interp[0,0])

    return next_theta

def f_kinematics(theta1, theta2):
     """
    Goes from angles to pose positions of robot arms

    Input:
    - theta1
    - theta2

    Returns:
    - tuple: World space positions of links, and the link relative positions
     """

     x1 = L1*np.cos(theta1)
     y1 = L1*np.sin(theta1)

     # based on first link
     x2 = x1 + L2*np.cos(theta1+theta2)
     y2 = y1 + L2*np.sin(theta1+theta2)

     #need explanation for this
     return (x1/2, y1/2), (x1+ L2/2 * np.cos(theta1 + theta2), y1+ L2/2 * np.sin(theta1 + theta2)), (x1, y1), (x2, y2)


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

     path = []
     path_angles = []
     steps = 10
     
     # Unwrap angles to ensure continuous interpolation, not SLERP in this case
     theta1_s = np.unwrap([theta1_s])[0]
     theta1_g = np.unwrap([theta1_g])[0]
    
     theta2_s = np.unwrap([theta2_s])[0]
     theta2_g = np.unwrap([theta2_g])[0]

     for i in range(steps):
        t = i / (steps - 1)

        # Linear interpolation for angles
        next_theta1 = (1 - t) * theta1_s + t * theta1_g
        next_theta2 = (1 - t) * theta2_s + t * theta2_g

        path_angles.append((next_theta1, next_theta2))
        
        # forward kinematics for the current angles
        path.append(f_kinematics(next_theta1, next_theta2))
     
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
     path = []
     theta1, theta2 = start_pose
     path.append(f_kinematics(theta1, theta2))

     for velocity, duration in plan:
          
          theta1 += velocity[0] * duration
          theta2 += velocity[1] * duration

          path.append(f_kinematics(theta1, theta2))

     return path
              
def visualize_path(path):
     """
    Visualizes path and animates robot's arm movement

    Input:
    - path: sequence of poses 

    Returns:
    - visualization
    """
     fig, ax = plt.subplots()
     ax.set_xlim([-4, 4])
     ax.set_ylim([-4, 4])

     link1, = ax.plot([], [], 'b-', lw=5, label='Link 1')
     link2, = ax.plot([], [], 'g-', lw=5, label='Link 2')
     xVec, = ax.plot([], [], 'ro', lw=2, label='x-vec')
     yVec, = ax.plot([], [], 'yo', lw=2, label='y-vec')
     
     start_ax = plt.axes([0.7, 0.9, 0.1, 0.075])
     pause_ax = plt.axes([0.81, 0.9, 0.1, 0.075])

     # Define the button objects
     start_button = Button(start_ax, 'Start')
     pause_button = Button(pause_ax, 'Pause')

     # This variable will control the animation state
     anim_running = True
     
     def init():
         link1.set_data([], [])
         link2.set_data([], [])
         return link1, link2
     
     def update(frame):
          (cx1, cy1), (cx2, cy2), (x1, y1), (x2, y2) = path[frame]
          link1.set_data([0, x1], [0, y1])
          link2.set_data([x1, x2], [y1, y2])
          xVec.set_data([cx1, cx2], [cy1, cy2]) #center of links as points 

          return link1, link2, xVec, yVec

     def start(event):
        nonlocal anim_running
        if not anim_running:
            anim.event_source.start()
            anim_running = True

     def pause(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False

     # Connect the buttons to the event handlers
     start_button.on_clicked(start)
     pause_button.on_clicked(pause)
     
     anim = FuncAnimation(fig, update, frames=len(path), interval=1000, blit=True)
     plt.legend()

     #anim.save("component_4_interpolate_arm.gif", writer="imagemagick")
     #anim.save("component_4_forward_propagate_arm.gif", writer="imagemagick")
     plt.show()

if __name__ == "__main__":
     start =  (np.radians(0), np.radians(45))
     goal = (np.radians(135), 0)

     path = interpolate_arm(start, goal)
     #visualize_path(path)

     plan = [((0, np.radians(30)), 2), ((np.radians(45), 0), 1), ((np.radians(60), 0), 1)]
     new_path = forward_propagate_arm(start, plan)
     visualize_path(new_path)