import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from matplotlib.patches import Rectangle
import time

class Node:
    def __init__(self, config):
        self.config = config
        self.parent = None
        self.children = []

class Obstacle:
    def __init__(self, x, y, theta, width, height):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width
        self.height = height
        
    def get_corners(self):
        """Get corners of the obstacle in world coordinates."""
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
        
        # Robot-specific parameters
        
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

    def simulate_car_motion(self, config, controls, dt=0.1, steps=10):
        """Simulate car motion using kinematic equations."""
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

    def get_car_corners(self, config):
        """Get corners of the car at given configuration."""
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

    def clip_config(self, config):
        """Clip the configuration to be within the bounds."""
        return np.clip(config, 
                       [lower for lower, _ in self.bounds], 
                       [upper for _, upper in self.bounds])
            
    def load_map(self, filename):
        """Load obstacles from file."""
        self.obstacles = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    # Remove parentheses and split by comma
                    line = line.strip().strip('()').split(',')
                    if len(line) == 5:
                        x, y, theta, w, h = map(float, line)
                        self.obstacles.append(Obstacle(x, y, theta, w, h))
        except Exception as e:
            print(f"Error loading map file: {e}")
            self.obstacles = []

    def get_arm_points(self, config):
        """Get the points defining the arm segments for a given configuration."""
        points = [self.base_position]  # Base of the arm
        x, y = self.base_position
        angle_sum = 0
        
        for theta, length in zip(config, self.link_lengths):
            angle_sum += theta
            x += length * np.cos(angle_sum)
            y += length * np.sin(angle_sum)
            points.append((x, y))
            
        return points

    def get_freebody_corners(self, config):
        """Get corners of the freeBody robot at a given configuration."""
        x, y, theta = config
        w, h = self.robot_width/2, self.robot_height/2
        
        # Local coordinates of corners
        corners_local = np.array([
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Rotate and translate corners
        corners_world = np.dot(corners_local, R.T) + np.array([x, y])
        return corners_world

    def point_segment_distance(self, p, seg_start, seg_end):
        """Calculate distance between point and line segment."""
        segment = np.array(seg_end) - np.array(seg_start)
        point = np.array(p) - np.array(seg_start)
        
        # Project point onto segment
        t = max(0, min(1, np.dot(point, segment) / np.dot(segment, segment)))
        projection = np.array(seg_start) + t * segment
        
        return np.linalg.norm(np.array(p) - projection)

    def segments_intersect(self, seg1_start, seg1_end, seg2_start, seg2_end):
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        A = np.array(seg1_start)
        B = np.array(seg1_end)
        C = np.array(seg2_start)
        D = np.array(seg2_end)
        
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def check_collision(self, config1, config2):
        """Check if the path between two configurations collides with obstacles."""
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

    def point_inside_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm."""
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

    def check_arm_collision(self, config1, config2):
        """Enhanced collision checking for arm robot."""
        steps = 20  # Increased from 10 for finer resolution
        
        # Interpolate between configurations
        for i in range(steps):
            t = i / float(steps-1)
            config = config1 + t * (config2 - config1)
            
            # Get arm segment points
            points = self.get_arm_points(config)
            
            # Check each arm segment against each obstacle
            for i in range(len(points) - 1):
                seg_start = points[i]
                seg_end = points[i + 1]
                
                # Add intermediate points along the arm segment for more thorough checking
                num_intermediate = 5
                for j in range(num_intermediate):
                    t = j / float(num_intermediate-1)
                    point = (
                        seg_start[0] + t * (seg_end[0] - seg_start[0]),
                        seg_start[1] + t * (seg_end[1] - seg_start[1])
                    )
                    
                    for obs in self.obstacles:
                        obs_corners = obs.get_corners()
                        
                        # Check if point is inside or very close to obstacle
                        if self.point_inside_polygon(point, obs_corners):
                            return True
                        
                        # Check distance to obstacle edges
                        for k in range(len(obs_corners)):
                            obs_start = obs_corners[k]
                            obs_end = obs_corners[(k+1)%len(obs_corners)]
                            if self.point_segment_distance(point, obs_start, obs_end) < 0.1:  # Small safety margin
                                return True
                        
            # Also check endpoints specifically
            for point in points:
                for obs in self.obstacles:
                    obs_corners = obs.get_corners()
                    if self.point_inside_polygon(point, obs_corners):
                        return True
                        
        return False

    def check_freebody_collision(self, config1, config2):
        """Check collision for freeBody robot."""
        steps = 10
        
        # Interpolate between configurations
        for i in range(steps):
            t = i / float(steps-1)
            config = config1 + t * (config2 - config1)
            robot_corners = self.get_freebody_corners(config)
            
            # Check robot corners against each obstacle
            for obs in self.obstacles:
                obs_corners = obs.get_corners()
                
                # Check robot edges against obstacle edges
                for i in range(len(robot_corners)):
                    r_start = robot_corners[i]
                    r_end = robot_corners[(i+1)%len(robot_corners)]
                    
                    for j in range(len(obs_corners)):
                        obs_start = obs_corners[j]
                        obs_end = obs_corners[(j+1)%len(obs_corners)]
                        if self.segments_intersect(r_start, r_end, obs_start, obs_end):
                            return True
                            
        return False

    def random_config(self):
        """Generate a random configuration with improved C-space sampling."""
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
        
    def nearest_neighbor(self, config):
        """Find the nearest node in the tree using a weighted distance metric."""
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
            
            # Generate velocity (slow down for sharp turns)
            velocity = self.max_velocity * (1 - abs(desired_steering)/self.max_steering_angle)
            velocity = max(0.5, velocity)  # Ensure minimum velocity
            
            # Simulate motion with these controls
            controls = (velocity, desired_steering)
            trajectory = self.simulate_car_motion(from_config, controls)
            
            # Return the last configuration from the trajectory
            return trajectory[-1]
    
    def extend(self, random_config):
        """Extend the tree with proper goal checking."""
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

    def build_tree(self, max_iterations=10000):
        """Build the RRT with improved progress tracking and visualization."""
        start_time = time.time()
        iteration = 0
        stall_count = 0
        last_best_dist = float('inf')
        best_node = None
        
        while iteration < max_iterations:
            # Adaptive goal bias
            goal_bias = 0.2 if stall_count < 100 else 0.1
            
            if random.random() < goal_bias:
                random_config = self.goal_config
            else:
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

    def get_path(self):
        """Extract the path from start to goal if one is found."""
        if not self.goal_node:
            return []
            
        path = []
        current = self.goal_node
        while current:
            path.append(current.config)
            current = current.parent
        return path[::-1]

    def animate_tree_growth(self):
        """Enhanced visualization with C-space for arm and workspace for freeBody."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if self.robot_type == "arm":
            print("Computing C-space obstacles...")
            theta1_range = np.linspace(self.bounds[0][0], self.bounds[0][1], 50)
            theta2_range = np.linspace(self.bounds[1][0], self.bounds[1][1], 50)
            collision_map = np.zeros((len(theta1_range), len(theta2_range)))
            
            # Check collisions for each configuration
            for i, theta1 in enumerate(theta1_range):
                for j, theta2 in enumerate(theta2_range):
                    config = np.array([theta1, theta2])
                    
                    # Get arm points for this configuration
                    points = self.get_arm_points(config)
                    
                    # Check collision with obstacles
                    is_collision = False
                    for k in range(len(points) - 1):
                        seg_start = points[k]
                        seg_end = points[k + 1]
                        
                        for obs in self.obstacles:
                            obs_corners = obs.get_corners()
                            # Check against obstacle edges
                            for m in range(len(obs_corners)):
                                obs_start = obs_corners[m]
                                obs_end = obs_corners[(m+1)%len(obs_corners)]
                                if self.segments_intersect(seg_start, seg_end, obs_start, obs_end):
                                    is_collision = True
                                    break
                            if is_collision:
                                break
                                
                            # Check if arm endpoint is inside the obstacle
                            if self.point_inside_polygon(seg_end, obs_corners):
                                is_collision = True
                                break
                        
                        if is_collision:
                            break
                            
                    if is_collision:
                        collision_map[j, i] = 1
        
        def update(frame):
            ax.clear()
            
            if self.robot_type == "arm":
                # Plot C-space obstacles
                ax.imshow(collision_map, extent=[self.bounds[0][0], self.bounds[0][1],
                                            self.bounds[1][0], self.bounds[1][1]],
                        origin='lower', cmap='Reds', alpha=0.5)
                
                # Plot tree nodes and edges in C-space
                for node in self.nodes[:frame]:
                    if node.parent:
                        ax.plot([node.parent.config[0], node.config[0]],
                            [node.parent.config[1], node.config[1]],
                            'b-', alpha=0.3)
                
                # Plot start and goal configurations
                ax.plot(self.start_config[0], self.start_config[1], 'go', markersize=8, label='Start')
                ax.plot(self.goal_config[0], self.goal_config[1], 'ro', markersize=8, label='Goal')
                
                # Draw goal region
                goal_circle = plt.Circle((self.goal_config[0], self.goal_config[1]), 
                                    self.goal_radius, color='r', fill=False)
                ax.add_artist(goal_circle)
                
                ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
                ax.set_ylim(self.bounds[1][0], self.bounds[1][1])
                ax.set_xlabel('θ₁ (rad)')
                ax.set_ylabel('θ₂ (rad)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            else:  # freeBody robot
                # Plot in C-space (x, y, theta)
                # We'll show a 2D projection (x, y) with color representing theta
                
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
        
        #anim.save("Env4TreeGrowthA2.gif", writer="imagemagick")

        plt.show()

    def animate_robot_path(self):
        """Create an animation of the robot moving along the solution path in workspace."""
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
        
        #anim.save("Env4RobotPathA2.gif", writer="imagemagick")

        plt.show()

    def visualize_problem(self):
        fig, ax = plt.subplots()
        theta1_range = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
        theta2_range = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)

        for theta1 in theta1_range:
            for theta2 in theta2_range:
                points = self.get_arm_points([theta1, theta2])
                ax.plot(points[-1][0], points[-1][1], 'b.', alpha=0.1)
        
        # Plot obstacles
        for obs in self.obstacles:
            corners = obs.get_corners()
            corners = np.vstack([corners, corners[0]])
            ax.plot(corners[:, 0], corners[:, 1], 'k-')
            ax.fill(corners[:, 0], corners[:, 1], 'gray', alpha=0.5)
        
        # Plot start configuration
        start_points = self.get_arm_points(self.start_config)
        ax.plot([p[0] for p in start_points], [p[1] for p in start_points], 'go-')
        
        # Plot goal configuration
        goal_points = self.get_arm_points(self.goal_config)
        ax.plot([p[0] for p in goal_points], [p[1] for p in goal_points], 'ro-')
        
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_aspect('equal')
        ax.set_title('Arm Robot Workspace and Problem Setup')
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