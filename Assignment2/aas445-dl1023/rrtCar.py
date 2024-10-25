import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from matplotlib.patches import Rectangle
import time

# Easy way to keep track of nodes on tree
class Node:
    def __init__(self, config):
        self.config = config
        self.parent = None
        self.children = []

# Easy way to keep track of obstacles info, and way to get corners for collision checking
class Obstacle:
    def __init__(self, x, y, theta, width, height):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width
        self.height = height
    
    # Get corners in real world 
    def get_corners(self):
        w, h = self.width/2, self.height/2
        corners_local = np.array([
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]
        ])
        
        # Rotate and translate corners
        corners_world = np.dot(corners_local, R.T) + np.array([self.x, self.y])
        return corners_world

class RRTPlanner:
    def __init__(self, start_config, goal_config, goal_radius, map_filename, robot_type):
        self.start_config = start_config
        self.goal_config = goal_config
        self.goal_radius = goal_radius
        self.robot_type = robot_type
        self.nodes = [Node(start_config)]
        self.goal_node = None
        
        # Robot-specific parameters took out arm and freebody
        
        if robot_type == "car":
            self.car_length = 1.0  # Length between front and rear axles
            self.car_width = 0.5   # Width of the car
            self.max_steering_angle = np.pi/4  # Maximum steering angle
            self.max_velocity = 2.0  # Maximum velocity
            # State space bounds [x, y, theta, beta]
            self.bounds = [
                (0, 20),      # x position
                (0, 20),      # y position
                (-np.pi, np.pi), # heading angle theta
                (-self.max_steering_angle, self.max_steering_angle)  # steering angle beta
            ]
        
        self.start_config = self.clip_config(start_config)
        self.goal_config = self.clip_config(goal_config)

        self.load_map(map_filename)

    # needed for car, since we need to follow the movement constraints 
    def simulate_car_motion(self, config, controls, dt=0.1, steps=10):
        V, delta = controls  # velocity and steering angle
        x, y, theta, beta = config
        
        # Initialize trajectory
        trajectory = [config]
        current_config = np.array(config)
        
        for _ in range(steps):
            # Update beta (steering angle) towards desired angle
            beta_diff = delta - current_config[3]
            beta_step = np.clip(beta_diff, -0.5, 0.5)  # Limit steering rate
            new_beta = current_config[3] + beta_step
            
            # Apply kinematic equations
            dx = V * np.cos(current_config[2] + new_beta) * dt
            dy = V * np.sin(current_config[2] + new_beta) * dt
            dtheta = (2 * V / self.car_length) * np.sin(new_beta) * dt
            
            # Update configuration
            new_config = np.array([
                current_config[0] + dx,
                current_config[1] + dy,
                current_config[2] + dtheta,
                new_beta
            ])
            
            # Normalize theta to [-pi, pi]
            new_config[2] = np.arctan2(np.sin(new_config[2]), np.cos(new_config[2]))
            
            current_config = new_config
            trajectory.append(current_config)
            
        return trajectory

    # Pretty much like the get corners of robot of freebody
    def get_car_corners(self, config):
        x, y, theta, _ = config
        w, h = self.car_width/2, self.car_length/2
        
        # Local coordinates of corners
        corners_local = np.array([
            [-w, -h],  # rear left
            [w, -h],   # rear right
            [w, h],    # front right
            [-w, h]    # front left
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Rotate and translate corners
        corners_world = np.dot(corners_local, R.T) + np.array([x, y])
        return corners_world

    # Make sure config stays in bounds 
    def clip_config(self, config):
        return np.clip(config, 
                       [lower for lower, _ in self.bounds], 
                       [upper for _, upper in self.bounds])
    #Make sure to get all obstacles in environment     
    def load_map(self, filename):
        self.obstacles = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip().strip('()').split(',')
                    if len(line) == 5:
                        x, y, theta, w, h = map(float, line)
                        self.obstacles.append(Obstacle(x, y, theta, w, h))
        except Exception as e:
            print(f"Error loading map file: {e}")
            self.obstacles = []

    # Check if segments intersect with Counter clockwise test
    def segments_intersect(self, seg1_start, seg1_end, seg2_start, seg2_end):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        A = np.array(seg1_start)
        B = np.array(seg1_end)
        C = np.array(seg2_start)
        D = np.array(seg2_end)
        
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    # Check collision for car, since it's not straight edge you need to account for trajectory between configurations 
    def check_collision(self, config1, config2):
        if self.robot_type == "car":
            # Simulate the motion between configurations
            dx = config2[0] - config1[0]
            dy = config2[1] - config1[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Generate trajectory between configurations
            controls = (1.0, config2[3])  # Use constant velocity and target steering
            trajectory = self.simulate_car_motion(config1, controls)
            
            # Check collision for each configuration in trajectory
            for config in trajectory:
                car_corners = self.get_car_corners(config)
                
                # Check each obstacle
                for obs in self.obstacles:
                    obs_corners = obs.get_corners()
                    
                    # Check if any car corner is inside obstacle
                    for corner in car_corners:
                        if self.point_inside_polygon(corner, obs_corners):
                            return True
                    
                    # Check if car edges intersect with obstacle edges
                    car_edges = list(zip(car_corners, np.roll(car_corners, -1, axis=0)))
                    obs_edges = list(zip(obs_corners, np.roll(obs_corners, -1, axis=0)))
                    
                    for car_edge in car_edges:
                        for obs_edge in obs_edges:
                            if self.segments_intersect(car_edge[0], car_edge[1],
                                                     obs_edge[0], obs_edge[1]):
                                return True
            
            return False
        
    # Check if points are in obstacles using raycasting
    def point_inside_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    #Generate random configurations for car, 5% towards goal, other is random uniform
    def random_config(self):
        if len(self.nodes) > 0:
            # 0% chance to sample near existing nodes
            if random.random() < -1:
                random_node = random.choice(self.nodes)
                pos_noise = np.random.normal(0, 1.0, 2)  # Position noise
                angle_noise = np.random.normal(0, 0.3)   # Orientation noise
                steering_noise = np.random.normal(0, 0.1) # Steering noise
                config = np.array([
                    random_node.config[0] + pos_noise[0],
                    random_node.config[1] + pos_noise[1],
                    random_node.config[2] + angle_noise,
                    random_node.config[3] + steering_noise
                ])
            
            # 5% chance to sample towards goal
            elif random.random() < 0.05:
                t = random.random()
                # Interpolate position and orientation separately
                pos = self.start_config[:2] + t * (self.goal_config[:2] - self.start_config[:2])
                # Handle angle interpolation properly
                angle_diff = np.arctan2(np.sin(self.goal_config[2] - self.start_config[2]),
                                    np.cos(self.goal_config[2] - self.start_config[2]))
                angle = self.start_config[2] + t * angle_diff
                # For steering angle, interpolate directly
                steering = self.start_config[3] + t * (self.goal_config[3] - self.start_config[3])
                config = np.array([pos[0], pos[1], angle, steering])
                
            else:
                # Regular uniform sampling
                config = np.array([
                    random.uniform(self.bounds[0][0], self.bounds[0][1]),  # x
                    random.uniform(self.bounds[1][0], self.bounds[1][1]),  # y
                    random.uniform(self.bounds[2][0], self.bounds[2][1]),  # theta
                    random.uniform(self.bounds[3][0], self.bounds[3][1])   # steering angle
                ])
        else:
            # Regular uniform sampling for first node
            config = np.array([
                random.uniform(self.bounds[0][0], self.bounds[0][1]),  # x
                random.uniform(self.bounds[1][0], self.bounds[1][1]),  # y
                random.uniform(self.bounds[2][0], self.bounds[2][1]),  # theta
                random.uniform(self.bounds[3][0], self.bounds[3][1])   # steering angle
            ])
        
        return self.clip_config(config)

    # Find nearest neighbor that will extend to new configuration found     
    def nearest_neighbor(self, config):
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            if self.robot_type == "freeBody":
                # Weight position more heavily than orientation
                pos_dist = np.linalg.norm(node.config[:2] - config[:2])
                angle_dist = abs(np.arctan2(np.sin(node.config[2] - config[2]), 
                                          np.cos(node.config[2] - config[2])))
                dist = pos_dist + 0.3 * angle_dist  # Reduce weight of orientation
            else:
                dist = np.linalg.norm(node.config - config)
                
            # Add bonus for nodes with fewer children to encourage branching
            child_penalty = len(node.children) * 0.1
            dist += child_penalty
            
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node
    
    # How much to translate towards the new configuration with the nearest node
    def steer(self, from_config, to_config, step_size=0.2):  # Increased step size further
        """Generate a new configuration with adaptive step size."""
        diff = to_config - from_config
        
        if self.robot_type == "car":
            # Generate control inputs
            dx = to_config[0] - from_config[0]
            dy = to_config[1] - from_config[1]
            
            # Desired heading
            desired_heading = np.arctan2(dy, dx)
            
            # Current heading
            current_heading = from_config[2]
            
            # Compute heading difference [-pi, pi]
            heading_diff = np.arctan2(np.sin(desired_heading - current_heading),
                                    np.cos(desired_heading - current_heading))
            
            # Generate steering angle based on heading difference
            desired_steering = np.clip(heading_diff, -self.max_steering_angle, self.max_steering_angle)
            
            # Calculate distance to target
            dist = np.sqrt(dx**2 + dy**2)
            
            # Get velocity
            velocity = self.max_velocity * (1 - abs(desired_steering)/self.max_steering_angle)
            velocity = max(0.5, velocity)  # Ensure minimum velocity
            
            # Simulate motion with these controls
            controls = (velocity, desired_steering)
            trajectory = self.simulate_car_motion(from_config, controls)
            
            # Return the last configuration from the trajectory
            return trajectory[-1]

    # Actually add the node to the tree after we figured out where to steer 
    def extend(self, random_config):
        nearest = self.nearest_neighbor(random_config)
        new_config = self.steer(nearest.config, random_config)
        
        new_config = self.clip_config(new_config)

        # First check if the new configuration is valid and different enough
        if np.allclose(new_config, nearest.config, atol=1e-2):
            return None
            
        # Check collision before proceeding
        if self.check_collision(nearest.config, new_config):
            return None
            
        # If we get here, add the new node
        new_node = Node(new_config)
        new_node.parent = nearest
        nearest.children.append(new_node)
        self.nodes.append(new_node)
        
        # Check if we've reached the goal
        if self.robot_type == "car":
            # For car, check position and orientation separately
            pos_dist = np.linalg.norm(new_config[:2] - self.goal_config[:2])
            angle_dist = abs(np.arctan2(np.sin(new_config[2] - self.goal_config[2]), 
                                    np.cos(new_config[2] - self.goal_config[2])))
            
            # Only check position and heading angle for goal, ignore steering angle
            if pos_dist < self.goal_radius and angle_dist < self.goal_radius:
                # Create a temporary config that matches the goal steering angle
                final_config = new_config.copy()
                final_config[3] = self.goal_config[3]  # Match goal steering angle
                
                # Verify path to goal is collision-free
                if not self.check_collision(new_config, final_config):
                    self.goal_node = new_node
                    return new_node
        
        return new_node

    # Where RRT tree actually grows 
    def build_tree(self, max_iterations=10000):
        start_time = time.time()
        iteration = 0
        stall_count = 0
        last_best_dist = float('inf')
        best_node = None
        
        while iteration < max_iterations:
           
            random_config = self.random_config()
            
            node = self.extend(random_config)
            
            # Update progress tracking
            best_dist = float('inf')
            for n in self.nodes:
                dist = np.linalg.norm(n.config[:2] - self.goal_config[:2])
                if dist < best_dist:
                    best_dist = dist
                    best_node = n
            
            # Track progress
            if abs(best_dist - last_best_dist) < 0.01:
                stall_count += 1
            else:
                stall_count = 0
            last_best_dist = best_dist
            
            if iteration % 100 == 0 or best_dist < self.goal_radius * 2:
                print(f"Iteration {iteration}, nodes: {len(self.nodes)}")
                print(f"Best distance: {best_dist:.3f}, at config: {best_node.config}")
                print(f"Goal config: {self.goal_config}")
                print(f"Stall count: {stall_count}\n")

            if len(self.nodes) == 1000:
                print(f"Max Nodes ({1000}) reached without finding goal")
                return False
            
            # Check if we have found a valid path to goal
            if self.goal_node is not None:
                # Double check the path is valid
                path = self.get_path()
                valid = True
                for i in range(len(path)-1):
                    if self.check_collision(path[i], path[i+1]):
                        valid = False
                        break
                
                if valid:
                    final_dist = np.linalg.norm(self.goal_node.config[:2] - self.goal_config[:2])
                    print(f"Goal reached after {iteration} iterations!")
                    print(f"Final distance to goal: {final_dist:.3f}")
                    print(f"Total roadmap building time: {time.time() - start_time:.3f} seconds")
                    return True  # Exit immediately when valid path is found
                else:
                    print("Invalid path detected, continuing search...")
                    self.goal_node = None
            
            iteration += 1
            
            if iteration == max_iterations:
                print(f"Max iterations ({max_iterations}) reached without finding goal")
                if best_node:
                    print(f"Best distance achieved: {best_dist:.3f}")
                print(f"Total roadmap building time: {time.time() - start_time:.3f} seconds")
                return False
    # Backtrack from goal to get the path taken form parents 
    def get_path(self):
        if not self.goal_node:
            return []
            
        path = []
        current = self.goal_node
        while current:
            path.append(current.config)
            current = current.parent
        return path[::-1]

    # Animate the RRT Tree actually growing 
    def animate_tree_growth(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            # freeBody robot or in this car
            
            # Plot obstacles
            for obs in self.obstacles:
                corners = obs.get_corners()
                corners = np.vstack([corners, corners[0]])
                ax.plot(corners[:, 0], corners[:, 1], 'k-')
                ax.fill(corners[:, 0], corners[:, 1], 'gray', alpha=0.5)
            
            # Plot nodes and edges with color based on theta
            for node in self.nodes[:frame]:
                if node.parent:
                    # Color based on average theta of parent and current
                    avg_theta = (node.parent.config[2] + node.config[2]) / 2
                    color = plt.cm.hsv(((avg_theta + np.pi) / (2 * np.pi)))
                    ax.plot([node.parent.config[0], node.config[0]], 
                        [node.parent.config[1], node.config[1]], 
                        color=color, alpha=0.3)
            
            # Plot start and goal
            ax.plot(self.start_config[0], self.start_config[1], 'go')
            ax.plot(self.goal_config[0], self.goal_config[1], 'ro')
            
            # Draw goal region
            goal_circle = plt.Circle((self.goal_config[0], self.goal_config[1]), 
                                self.goal_radius, color='r', fill=False)
            ax.add_artist(goal_circle)
            
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
        
            ax.set_title(f'RRT Tree Growth in C-space (Frame {frame})')
            ax.set_aspect('equal')
            
        anim = FuncAnimation(fig, update, frames=len(self.nodes)+1, 
                        interval=50, repeat=False)
        
        #anim.save("Env5Car_rrt_grow.gif", writer="imagemagick")

        plt.show()

    # Animate robots moving in workspace
    def animate_robot_path(self):
        path = self.get_path()
        if not path:
            print("No path found to animate")
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            config = path[frame]
            
            # Draw obstacles
            for obs in self.obstacles:
                corners = obs.get_corners()
                corners = np.vstack([corners, corners[0]])
                ax.plot(corners[:, 0], corners[:, 1], 'k-')
                ax.fill(corners[:, 0], corners[:, 1], 'gray', alpha=0.5)
            
            if self.robot_type == "car":
                # Draw car body
                corners = self.get_car_corners(config)
                corners = np.vstack([corners, corners[0]])
                ax.fill(corners[:, 0], corners[:, 1], 'blue', alpha=0.7)
                
                # Draw direction arrow and steering angle
                x, y, theta, beta = config
                # Draw heading arrow
                dx = 0.5 * np.cos(theta)
                dy = 0.5 * np.sin(theta)
                ax.arrow(x, y, dx, dy, head_width=0.1, color='red')
                
                # Draw steering angle indicator
                front_center = np.array([x + self.car_length/2 * np.cos(theta), 
                                       y + self.car_length/2 * np.sin(theta)])
                steering_dx = 0.3 * np.cos(theta + beta)
                steering_dy = 0.3 * np.sin(theta + beta)
                ax.arrow(front_center[0], front_center[1], 
                        steering_dx, steering_dy, 
                        head_width=0.05, color='green')
                
                # Draw ghost images of previous configurations
                alpha = 0.2
                for prev_config in path[:frame]:
                    prev_corners = self.get_car_corners(prev_config)
                    prev_corners = np.vstack([prev_corners, prev_corners[0]])
                    ax.fill(prev_corners[:, 0], prev_corners[:, 1], 
                           'blue', alpha=alpha)
                    
            ax.set_title(f'Robot Motion in Workspace (Frame {frame}/{len(path)-1})')
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.set_aspect('equal')
            if self.robot_type == "arm":
                ax.legend()
            
        anim = FuncAnimation(fig, update, frames=len(path), 
                        interval=100, repeat=False)
        
        #anim.save("Env5Car_rrt_traj.gif", writer="imagemagick")

        plt.show()

def main():
    parser = argparse.ArgumentParser(description='RRT Path Planning')
    parser.add_argument('--robot', type=str, required=True, 
                       choices=['car'])
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--goal_rad', type=float, required=True)
    parser.add_argument('--map', type=str, required=True)
    
    args = parser.parse_args()
    
    # Convert arguments to numpy arrays
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)
    
    # Create and run RRT planner
    planner = RRTPlanner(start_config, goal_config, args.goal_rad, 
                        args.map, args.robot)
    
    #if args.robot == "arm":
        #planner.visualize_problem()

    if planner.build_tree():
        print("Path found!")
        planner.animate_tree_growth()
        planner.animate_robot_path()
    else:
        print("No path found within iteration/node limit")
        planner.animate_tree_growth()
        planner.animate_robot_path()

if __name__ == "__main__":
    main()