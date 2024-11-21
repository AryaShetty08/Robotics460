import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from matplotlib.patches import Rectangle, Polygon

"""
Potential Function Planner which gives path that robot should take based
on attractive and replusive potentials defined. Specifcally on 20x20 grid
with environment file loaded obstacles.

Input:
- start - x and y position of where the robot begins, must be collision free
- goal - x and y position of where robot must end
- obstacles - environment file with defined obstacles (x,y,theta,w,h)

Returns:
- path - path of poses that robot takes when traversing to goal
- success - bool that is True if robot makes it to goal or False if planner fails
"""
class PotentialFunctionPlanner:
    def __init__(self, k_att=0.5, k_rep=200.0, rho_0=0.25):
        self.k_att = k_att  # Attractive force
        self.k_rep = k_rep  # Repulsive force
        self.rho_0 = rho_0  # Influences radius of how close robot can go 
    
    # To get vertices of obstacles that are rotated 
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
        # rotation matrix and translation applied to get vertices of obstacle
        return vertices @ R.T + np.array([x, y])
    
    # Get distance from point to obstacle 
    def distance_to_obstacle(self, point, obs):
        vertices = self.get_vertices(obs)
        min_dist = float('inf')
        
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            segment = v2 - v1
            segment_length_squared = np.sum(segment**2)
            
            if segment_length_squared == 0:
                dist = np.linalg.norm(point - v1)
            else:
                t = np.clip(np.dot(point - v1, segment) / segment_length_squared, 0, 1)
                projection = v1 + t * segment
                dist = np.linalg.norm(point - projection)
            
            min_dist = min(min_dist, dist)
        return min_dist
    
    # attractive gradient
    def attractive_gradient(self, q, goal):
        return self.k_att * (q - goal)
    
    # repulsive gradient 
    def repulsive_gradient(self, q, obstacle):
        d = self.distance_to_obstacle(q, obstacle)
        
        if d <= self.rho_0:
            # Numerical gradient using central differences
            epsilon = 0.01
            grad = np.zeros(2)
            
            for i in range(2):
                dp = np.zeros(2)
                dp[i] = epsilon
                dist_plus = self.distance_to_obstacle(q + dp, obstacle)
                dist_minus = self.distance_to_obstacle(q - dp, obstacle)
                
                grad[i] = self.k_rep * (1/d - 1/self.rho_0) * (dist_plus - dist_minus) / (2 * epsilon * d**2)
            
            return -grad  # Negative because we want to move away from obstacles
        return np.zeros(2)
    
    # Total gradient for current robot position 
    def total_gradient(self, q, goal, obstacles):
        grad = -self.attractive_gradient(q, goal)  # Negative because we want to move downhill
        
        for obs in obstacles:
            grad += -self.repulsive_gradient(q, obs)
        
        # Normalize gradient if it's too large
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
            
        return grad
    
    # Generate path for robot
    # added iterations so doesn't run forever if convergence check is not met
    def plan_path(self, start, goal, obstacles, step_size=0.1, max_iters=2000):
        path = [start.copy()]
        q = start.copy()
        
        for _ in range(max_iters):
            grad = self.total_gradient(q, goal, obstacles)
            
            if np.linalg.norm(grad) < 1e-3:  # Convergence check
                break
                
            q = q + step_size * grad  # Algo 1
            q = np.clip(q, [0, 0], [20, 20])  # No out of bounds
            path.append(q.copy())
            
            if np.linalg.norm(q - goal) < 0.1: 
                return np.array(path), True
                
        return np.array(path), False

# Same function for loading obstacles from environment 
def load_obstacles(filename):
    obstacles = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                x, y, theta, w, h = map(float, line.strip('()\n').split(','))
                obstacles.append([x, y, theta, w, h])
    except Exception as e:
        print(f"Error loading obstacles: {e}")
        return []
    return np.array(obstacles)

# Animate robot path 
def animate_path(path, goal, obstacles, planner, success):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot obstacles as rectangles
    for obs in obstacles:
        vertices = planner.get_vertices(obs)
        polygon = Polygon(vertices, color='red', alpha=0.5)
        ax.add_patch(polygon)
    
    # Plot start and goal
    ax.plot(path[0, 0], path[0, 1], 'bo', label='Start')
    ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    
    # Create line
    line, = ax.plot([], [], 'b-', lw=2, label='Path')
    point, = ax.plot([], [], 'bo', markersize=8)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Potential Function Path Animation')
    outputStr = f"Path planning {'succeeded' if success else 'failed'} \n Path length: {len(path)} points"
    plt.gcf().text(0.02, 0.94, outputStr, fontsize=14)
    ax.grid(True)
    ax.legend()
    
    def update(frame):
        line.set_data(path[:frame+1, 0], path[:frame+1, 1])
        point.set_data([path[frame, 0]], [path[frame, 1]])
        return line, point
    
    anim = FuncAnimation(
        fig, update, frames=len(path),
        interval=50, blit=True, repeat=False
    )
    
    # save here 
    anim.save("Env5PotentialTest1.gif", writer="imagemagick")

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=float, nargs=2, required=True)
    parser.add_argument('--goal', type=float, nargs=2, required=True)
    parser.add_argument('--obstacles', type=str, required=True)
    
    args = parser.parse_args()
    
    # Load obstacles from file
    obstacles = load_obstacles(args.obstacles)
    
    planner = PotentialFunctionPlanner()
    path, success = planner.plan_path(
        np.array(args.start),
        np.array(args.goal),
        obstacles
    )
    
    print(f"Path planning {'succeeded' if success else 'failed'}")
    print(f"Path length: {len(path)} points")
    
    animate_path(path, np.array(args.goal), obstacles, planner, success)

if __name__ == "__main__":
    main()