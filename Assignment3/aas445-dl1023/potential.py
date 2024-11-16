import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from matplotlib.patches import Rectangle, Polygon
import time

# define the potetntials
# attractive
# repulsive
# calculate and use gradient to find force for robot to move this is in 2d no rotation 
# and implement

class PotentialPlanner:
    def __init__(self, k_att=1.0, k_rep=100.0, rho_0=2.0):
        # attractive, repulsive, and influence of radius of obs
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
# this one is for goal
    def attractive_potential(self, robot_pose, goal):
        # they used a quadratic here??
        att_pot = 0.5 * self.k_att * np.sum(np.square(robot_pose - goal))
        return att_pot
    
    def attractive_gradient(self, robot_pose, goal):
        att_grad = self.k_att * (robot_pose - goal)
        return att_grad
    
    def get_vertices(self, obs):
        x, y, theta, w, h = obs

        vertices = np.array([
            [-w/2, -h/2],
            [w/2, -h/2],
            [w/2, h/2],
            [-w/2, h/2]
        ])

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # apply rotation and translation 
        rotated_vertices = vertices @ R.T + np.array([x, y])
        return rotated_vertices
    
    def point_to_obs(self, point, obs):
        vertices = self.get_vertices(obs)

        def point_to_segment(p, v1, v2):
            segment = v2 - v1
            segment_length_squared = np.sum(segment**2)
            if segment_length_squared == 0:
                return np.linalg.norm(p - v1)

            t = max(0, min(1, np.dot(p - v1, segment) / segment_length_squared))
            projection = v1 + t * segment 
            return np.linalg.norm(p - projection)
        
        # get each egde distance
        d_min = float('inf')
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            dist = point_to_segment(point, v1, v2)
            d_min = min(d_min, dist)
        
        return d_min

    # for the obstacles
    def repulsive_potential(self, robot_pose, obs):
        # they used inverse function?
        dist = self.point_to_obs(robot_pose, obs)
        if dist <= self.rho_0:
            return 0.5 * self.k_rep * (1/dist - 1/self.rho_0)**2
        return 0.0

    def repulsive_gradient(self, robot_pose, obs):
        epsilon = 0.01
        grad = np.zeros(2)

        for i in range(2):
            dp = np.zeros(2)
            dp[i] = epsilon
            grad[i] = (self.repulsive_potential(robot_pose + dp, obs) - self.repulsive_potential(robot_pose - dp, obs)) / (2*epsilon)

        return grad
    
    def total_gradient(self, robot_pose, goal, obstacles):
        grad = -self.attractive_gradient(robot_pose, goal)
        for obs in obstacles:
            grad += -self.repulsive_gradient(robot_pose, obs)
        return grad

# gradient
def gradient_descent(planner, start, goal, obstacles, step_size=0.01, max_iterations=2000, threshold=0.01):
    path = [start.copy()]
    pos = start.copy()

    for i in range(max_iterations):
        grad = planner.total_gradient(pos, goal, obstacles)
        if np.linalg.norm(grad) < threshold:
            break
        pos = pos - step_size * grad

        #robot bounds
        pos[0] = np.clip(pos[0], 0, 20)
        pos[1] = np.clip(pos[1], 0, 20)

        path.append(pos.copy())

        if np.linalg.norm(pos - goal) < threshold:
            break
    
    return np.array(path)

def animate_path(path, goal, obstacles, xlim=(0, 20), ylim=(0, 20)):
    fig, ax = plt.subplots(figsize=(10, 10))
    line, = ax.plot([], [], 'b-')  # Path line
    point, = ax.plot([], [], 'bo', markersize=8)  # Moving point
    
    # Plot static elements outside of init function
    # Plot obstacles
    for obs in obstacles:
        vertices = planner.get_vertices(obs)
        polygon = Polygon(vertices, color='red', alpha=0.5)
        ax.add_patch(polygon)
        
    # Plot goal and start
    ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    ax.plot(path[0, 0], path[0, 1], 'bo', markersize=10, label='Start')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.legend()
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def update(frame):
        # Update the line to show the path up to current position
        line.set_data(path[:frame+1, 0], path[:frame+1, 1])
        # Update the point position
        point.set_data(path[frame, 0], path[frame, 1])
        return line, point
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(path),
                        interval=50, repeat=False, blit=True)
    plt.show()

def load_obstacles(filename):
    obstacles = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, theta, w, h = map(float, line.strip('()\n').split(','))
            obstacles.append([x, y, theta, w, h])
    return obstacles

def main():
    print("hello")
    parser = argparse.ArgumentParser(description="Potential Function")

    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)

    parser.add_argument('--obstacles', type=str, required=True, help='Path to obstacles file')
    args = parser.parse_args()
    
    # Convert arguments to numpy arrays
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)

    obstacles = load_obstacles(args.obstacles)
    
    global planner  # Make planner accessible to animation function
    planner = PotentialPlanner(k_att=1.0, k_rep=100.0, rho_0=2.0)
    
    # Generate path
    path = gradient_descent(planner, start_config, goal_config, obstacles)
    
    # Animate the result
    animate_path(path, goal_config, obstacles)
    
if __name__ == "__main__":
    main()