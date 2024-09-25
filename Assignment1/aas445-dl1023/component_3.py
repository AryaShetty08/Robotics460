import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Button


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

     path = []
     
     # First pose is the start
     path.append((x0, y0, theta0))

     if x0 < -10 or x0 > 10 or y0 < -10 or y0 > 10 or xG < -10 or xG > 10 or yG < -10 or yG > 10:
          print("Start and End poses out of bounds")
          return path
     
     # Determine how many steps to take to reach goal, can change if needed
     # Figure out distance in 2d by dividing the length and iterating those steps 
     steps = 10
     
     path_x = np.linspace(x0, xG, steps)
     path_y = np.linspace(y0, yG, steps)

     # angles must handle singularity cause of wrapping 
     r2 = np.array([[np.cos(thetaG), -np.sin(thetaG), 0], [np.sin(thetaG), np.cos(thetaG), 0], [0, 0, 1]])

     r1 = np.array([[np.cos(theta0), -np.sin(theta0), 0], [np.sin(theta0), np.cos(theta0), 0], [0, 0, 1]])

     # get relative rotation matrix
     r_rel = np.dot(r2, np.linalg.inv(r1))

     # find angle
     angle_rel = np.arctan2(r_rel[1, 0], r_rel[0, 0])

     for i in range(1, steps):

          interpolation_fraction = i / (steps - 1)

          # get new angle
          new_angle = interpolation_fraction * angle_rel

          # apply matrix to r1
          new_matrix = np.array([[np.cos(new_angle), -np.sin(new_angle), 0], [np.sin(new_angle), np.cos(new_angle), 0], [0, 0, 1]])
          r_interp = np.dot(new_matrix, r1)

          # get next angle for path 
          theta_interp = np.arctan2(r_interp[1, 0], r_interp[0 , 0])

          path.append((path_x[i], path_y[i], theta_interp))

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
          print("Start pose out of bounds")
          return (0, 0, 0)
     
     path = [(None, None, None) for _ in range(len(plan) + 1)]
     path[0] = start_pose
     
     for i in range(len(plan)):
          x, y, theta = path[i]
          Vx, Vy, Vtheta, duration = plan[i]
          velocity = (Vx, Vy, Vtheta)
          
          # find distance and angle traveled in time
          deltaX, deltaY, deltaTheta = (v * duration for v in velocity)

          # apply rotation matrix
          x += deltaX*np.cos(theta) - deltaY*np.sin(theta)
          y += deltaX*np.sin(theta) + deltaY*np.cos(theta)
          theta = (theta + deltaTheta) % (2*np.pi)

          path[i+1] = (x, y, theta)
     
     return path 
          
def draw_vectors(ax, center, theta, length=0.5):
     """
    Draws vectors of direction robot is headed for specfic pose

    Input:
    - ax
    - center
    - theta

    """
     x, y = center
     dx = length*np.cos(theta)
     dy = length*np.sin(theta)

     ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='green', ec='green')


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
     plt.subplots_adjust(bottom=0.2)

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

     # Define button axes for Start and Pause
     start_ax = plt.axes([0.7, 0.9, 0.1, 0.075])
     pause_ax = plt.axes([0.81, 0.9, 0.1, 0.075])
     
     # Define the button objects
     start_button = Button(start_ax, 'Start')
     pause_button = Button(pause_ax, 'Pause')

     # Animation state control
     anim_running = True

     def init():
         pose = path[0]
         box.set_xy((pose[0] - robot_dim[0] / 2, pose[1] - robot_dim[1] / 2))  # Use starting pose
         box.angle = pose[2] * 180 / np.pi  # Use starting angle
         return box,

     #animate 
     def update(frame):
          pose = path[frame]
          x, y, theta = pose

          box.set_xy((x - (robot_dim[0] / 2), y - (robot_dim[1] /2)))
          box.angle = theta * 180 / np.pi

          #ax.patches = [box]

          draw_vectors(ax, (x, y), theta)

          return box,

     # Button click events
     def start(event):
        nonlocal anim_running
        if not anim_running:
            ani.event_source.start()
            anim_running = True

     def pause(event):
        nonlocal anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False

     # Connect the buttons to the event handlers
     start_button.on_clicked(start)
     pause_button.on_clicked(pause)

     # Animation object
     ani = FuncAnimation(fig, update, frames=len(path), init_func=init, interval=500, repeat=True)

     ax.set_xlabel('X')
     ax.set_ylabel('Y')
     ax.set_title('Robot Path Animation')
     ax.grid(True)

     plt.show()


if __name__ == "__main__":
     start_pose =  (0, 0, 0)
     end_pose = (4, 5, np.pi/2)
     print(interpolate_rigid_body(start_pose, end_pose))
     visualize_path(interpolate_rigid_body(start_pose, end_pose))

     plan = [
     (1, 1, np.radians(45), 1), 
     (1, 0, np.radians(45), 2),   
     (1, 0, 0, 2),   
     ]
     #print(forward_propagate_rigid_body(start_pose, plan))
     #visualize_path(forward_propagate_rigid_body(start_pose, plan))

     example_path = [(0, 0, 0), (1, 1, 0.5), (2, 2, 1.0), (3, 3, 1.5), (4, 4, 2.0)]

     #visualize_path(example_path)
