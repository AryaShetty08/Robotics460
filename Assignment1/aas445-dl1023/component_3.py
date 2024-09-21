import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


def interpolate_rigid_body(start_pose, goal_pose):
     """
    Gives path for the robot to traverse from thes start_pose to goal_pose

    Input:
    - start_pose: tuple (x0, y0, theta0) that determines robots starting position and angle
    - goal_pose: tuple (xG, yG, thetaG)that determines robots ending position and angle

    Returns:
    - path: sequence of poses
    """
     
     # maybe have checks for ir in -10, 10 grid where robot is supposed to move 
     
     x0, y0, theta0 = start_pose
     xG, yG, thetaG = goal_pose

     if x0 < -10 or x0 > 10 or y0 < -10 or y0 > 10 or xG < -10 or xG > 10 or yG < -10 or yG > 10:
          print("Start and End poses out of bounds")
          return (0, 0, 0)
     
     # have to determine how many steps to take to reach goal 
     # Figure out distance in 2d by dividing the length and iterating those steps 
     steps = 10 
     path_x = np.linspace(x0, xG, steps)
     path_y = np.linspace(y0, yG, steps)

     # angles must handle singularity cause of wrapping 
     if abs(thetaG - theta0) > np.pi:
          if thetaG > theta0:
               theta0 += 2 * np.pi
          else:
               thetaG += 2 * np.pi

     path_theta = np.linspace(theta0, thetaG, steps) % (2 * np.pi)
     
     path = [(path_x[i], path_y[i], path_theta[i]) for i in range(steps)]

     return path

def forward_propagate_rigid_body(start_pose, plan):
     """
    Gives path robot takes based on the start pose and plan

    Input:
    - start_pose: tuple (x0, y0, theta0) that determines robots starting position and angle
    - plan: sequence of N tuples (velocity, duration)

    Returns:
    - path: sequence of poses
    """

     # what should theta input be?

     x0, y0, theta0 = start_pose

     if x0 < -10 or x0 > 10 or y0 < -10 or y0 > 10:
          print("Start and End poses out of bounds")
          return (0, 0, 0)
     
     path = [(None, None, None) for _ in range(len(plan) + 1)]
     path[0] = start_pose
     
     for i in range(len(plan)):
          x, y, theta = path[i]
          Vx, Vy, Vtheta, duration = plan[i]
          velocity = (Vx, Vy, Vtheta)
          
          deltaX, deltaY, deltaTheta = (v * duration for v in velocity)

          # apply rotation matrix ??
          x = deltaX*np.cos(theta) - deltaY*np.sin(theta)
          y = deltaX*np.sin(theta) + deltaY*np.cos(theta)
          theta = (theta + deltaTheta) % (2*np.pi)

          path[i+1] = (x, y, theta)
     
     return path 
          
     
def visualize_path(path):
     """
    Visualizes path and animates robot's movement

    Input:
    - path: sequence of poses 

    Returns:
    - visualization
    """
     robot_dim = (0.5, 0.3)

     fig, ax = plt.subplots()

     ax.set_xlim(-10, 10)
     ax.set_ylim(-10, 10)

     # path robot takes 
     xVal = [pose[0] for pose in path]
     yVal = [pose[1] for pose in path]
     ax.plot(xVal, yVal, 'b-', label="Path")

     # robot box 
     initial_pose = path[0]
     box = patches.Rectangle((initial_pose[0] - robot_dim[0] / 2, initial_pose[1] - robot_dim[1] / 2),
                             robot_dim[0], robot_dim[1],
                             angle=initial_pose[2] * 180 / np.pi, 
                             fill=True, color='red', alpha=0.3)
     
     ax.add_patch(box)

     #animate 
     def update(frame):
          pose = path[frame]
          x, y, theta = pose

          box.set_xy((x - robot_dim[0] / 2, y - robot_dim[1] /2))
          box.angle = theta * 180 / np.pi

          return box,

     # mess with this or make it another input, maybe slider??
     interval = 500
     anim = FuncAnimation(fig, update, frames=len(path), interval=interval, blit=True)

     ax.set_xlabel('X')
     ax.set_ylabel('Y')
     ax.set_title('Robot Path Animation')
     ax.grid(True)

     plt.show()


if __name__ == "__main__":
     start_pose =  (0, 0, 0)
     end_pose = (5, 5, np.pi/2)
     #print(interpolate_rigid_body(start_pose, end_pose))

     plan = [
     (1, 0, 0.1, 2), 
     (0.5, 0, -0.2, 3),
     (0, 0, 0, 1)     
     ]
     #print(forward_propagate_rigid_body(start_pose, plan))

     example_path = [(0, 0, 0), (1, 1, 0.5), (2, 2, 1.0), (3, 3, 1.5), (4, 4, 2.0)]

     visualize_path(example_path)
