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
        self.cost = float('inf')  # Cost from start to this node

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

class RRTStarPlanner:
    def __init__(self, start_config, goal_config, goal_radius, map_filename, robot_type):
        self.start_config = start_config
        self.goal_config = goal_config
        self.goal_radius = goal_radius
        self.robot_type = robot_type
        self.goal_node = None
        
        # Robot-specific parameters
        if robot_type == "arm":
            self.link_lengths = [2.0, 1.5]  # Length of arm segments
            self.bounds = [(-np.pi, np.pi)] * len(start_config)
            self.base_position = (10,10)
        else:  # freeBody
            self.robot_width = 0.5
            self.robot_height = 0.3
            self.bounds = [(0, 20), (0, 20), (-np.pi, np.pi)]
        
        self.start_config = self.clip_config(start_config)
        self.goal_config = self.clip_config(goal_config)

        # Initialize the tree with start node
        start_node = Node(start_config)
        start_node.cost = 0  # Cost to start is 0
        self.nodes = [start_node]
        
        self.load_map(map_filename)
        
        # RRT* specific parameters
        self.dimension = len(start_config)
        # Tuning parameter for RRT*
        self.gamma = 2.0  # Can be tuned based on environment
        

    def get_neighborhood_radius(self):
        """Calculate the neighborhood radius based on current tree size."""
        n = len(self.nodes)
        if n < 2:
            return float('inf')
        
        # Calculate volume of the C-space (approximation)
        if self.robot_type == "freeBody":
            space_volume = (self.bounds[0][1] - self.bounds[0][0]) * \
                        (self.bounds[1][1] - self.bounds[1][0]) * \
                        (2 * np.pi)  # Full rotation range
        else:  # arm
            space_volume = np.prod([upper - lower for lower, upper in self.bounds])
        
        # RRT* radius calculation
        dimension = len(self.bounds)
        gamma = 2.0  # Tuning parameter
        
        # Calculate radius using the RRT* formula
        radius = min(gamma * (np.log(n) / n) ** (1.0 / dimension), 5.0)  # Added upper limit
        
        return radius

    def get_neighbors(self, config, radius):
        """Find all nodes within radius of config."""
        neighbors = []
        for node in self.nodes:
            if self.robot_type == "freeBody":
                # For freeBody, weight position and orientation differently
                pos_dist = np.linalg.norm(node.config[:2] - config[:2])
                angle_dist = abs(np.arctan2(np.sin(node.config[2] - config[2]), 
                                          np.cos(node.config[2] - config[2])))
                # Combined weighted distance
                dist = pos_dist + 0.3 * angle_dist
            else:
                dist = np.linalg.norm(node.config - config)
                
            if dist < radius:
                neighbors.append(node)
        return neighbors

    def cost_to_come(self, from_node, to_config):
        """Calculate the cost to come from a node to a configuration."""
        if self.robot_type == "freeBody":
            # Weight position more heavily than orientation for cost
            pos_dist = np.linalg.norm(from_node.config[:2] - to_config[:2])
            angle_dist = abs(np.arctan2(np.sin(from_node.config[2] - to_config[2]), 
                                      np.cos(from_node.config[2] - to_config[2])))
            return from_node.cost + pos_dist + 0.3 * angle_dist
        else:
            return from_node.cost + np.linalg.norm(from_node.config - to_config)


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
        if self.robot_type == "arm":
            return self.check_arm_collision(config1, config2)
        else:
            return self.check_freebody_collision(config1, config2)

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
                if self.robot_type == "arm":
                    # Sample in joint angle space
                    config = random_node.config + np.random.normal(0, 0.5, len(self.bounds))
                else:  # freeBody
                    # Sample position and orientation separately
                    pos_noise = np.random.normal(0, 1.0, 2)  # Position noise
                    angle_noise = np.random.normal(0, 0.3)   # Orientation noise
                    config = np.array([
                        random_node.config[0] + pos_noise[0],
                        random_node.config[1] + pos_noise[1],
                        random_node.config[2] + angle_noise
                    ])
            
            # 5% chance to sample towards goal
            elif random.random() < 0.05:
                t = random.random()
                if self.robot_type == "arm":
                    # Interpolate joint angles
                    config = self.start_config + t * (self.goal_config - self.start_config)
                    config += np.random.normal(0, 0.1, len(self.bounds))
                else:  # freeBody
                    # Interpolate position and orientation separately
                    pos = self.start_config[:2] + t * (self.goal_config[:2] - self.start_config[:2])
                    # Handle angle interpolation properly
                    angle_diff = np.arctan2(np.sin(self.goal_config[2] - self.start_config[2]),
                                        np.cos(self.goal_config[2] - self.start_config[2]))
                    angle = self.start_config[2] + t * angle_diff
                    config = np.array([pos[0], pos[1], angle])
                    
            else:
                # Regular uniform sampling
                if self.robot_type == "arm":
                    config = np.array([random.uniform(lower, upper) 
                                    for lower, upper in self.bounds])
                else:  # freeBody
                    config = np.array([
                        random.uniform(self.bounds[0][0], self.bounds[0][1]),  # x
                        random.uniform(self.bounds[1][0], self.bounds[1][1]),  # y
                        random.uniform(-np.pi, np.pi)  # theta
                    ])
        else:
            # Regular uniform sampling for first node
            config = np.array([random.uniform(lower, upper) 
                            for lower, upper in self.bounds])
        
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
        
        if self.robot_type == "freeBody":
            # Wrap orientation difference
            diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))
            
            # Adaptive step size based on distance to goal
            goal_dist = np.linalg.norm(self.goal_config[:2] - from_config[:2])
            adaptive_step = min(step_size, max(0.5, goal_dist / 5.0))
            
            pos_diff = diff[:2]
            pos_dist = np.linalg.norm(pos_diff)
            
            if pos_dist > adaptive_step:
                # Scale position components
                new_pos = from_config[:2] + (pos_diff / pos_dist) * adaptive_step
                # Interpolate orientation with reduced weight
                t = adaptive_step / pos_dist
                new_angle = from_config[2] + t * diff[2]
                new_config = np.array([new_pos[0], new_pos[1], new_angle])
            else:
                new_config = to_config
        else:
            dist = np.linalg.norm(diff)
            if dist < step_size:
                new_config = to_config
            else:
                new_config = from_config + (diff / dist) * step_size
            
        # Ensure the new configuration stays within bounds
        for i, (lower, upper) in enumerate(self.bounds):
            new_config[i] = max(lower, min(upper, new_config[i]))
                
        return self.clip_config(new_config)

    def extend(self, random_config):
        """RRT* extend function with improved nearest neighbor selection and rewiring."""
        # First find nearest neighbor as a backup
        nearest = self.nearest_neighbor(random_config)
        if nearest is None:
            return None
            
        # Generate new configuration
        new_config = self.steer(nearest.config, random_config)
        if new_config is None:
            return None
            
        # Check for collision
        if self.check_collision(nearest.config, new_config):
            return None
            
        # Find all neighbors within radius
        radius = self.get_neighborhood_radius()
        neighbors = self.get_neighbors(new_config, radius)
        
        # Find best parent from neighbors
        min_cost = float('inf')
        best_parent = None
        
        for neighbor in neighbors:
            # Calculate potential cost through this neighbor
            potential_cost = neighbor.cost + np.linalg.norm(neighbor.config - new_config)
            
            # Check if this gives better path
            if potential_cost < min_cost:
                # Verify connection is collision-free
                if not self.check_collision(neighbor.config, new_config):
                    min_cost = potential_cost
                    best_parent = neighbor
        
        # If no valid parent found in neighborhood, use nearest neighbor
        if best_parent is None:
            if not self.check_collision(nearest.config, new_config):
                best_parent = nearest
                min_cost = nearest.cost + np.linalg.norm(nearest.config - new_config)
            else:
                return None
        
        # Create new node
        new_node = Node(new_config)
        new_node.parent = best_parent
        new_node.cost = min_cost
        best_parent.children.append(new_node)
        self.nodes.append(new_node)
        
        # Rewire the tree through the new node
        self.rewire(new_node, neighbors)
        
        # Check if we've reached goal
        if self.robot_type == "freeBody":
            pos_dist = np.linalg.norm(new_node.config[:2] - self.goal_config[:2])
            angle_dist = abs(np.arctan2(np.sin(new_node.config[2] - self.goal_config[2]), 
                                    np.cos(new_node.config[2] - self.goal_config[2])))
            
            if pos_dist < self.goal_radius and angle_dist < self.goal_radius:
                if not self.check_collision(new_node.config, self.goal_config):
                    if self.goal_node is None or new_node.cost < self.goal_node.cost:
                        self.goal_node = new_node
        else:  # arm robot
            dist = np.linalg.norm(new_node.config - self.goal_config)
            if dist < self.goal_radius:
                if not self.check_collision(new_node.config, self.goal_config):
                    if self.goal_node is None or new_node.cost < self.goal_node.cost:
                        self.goal_node = new_node
        
        return new_node

    def rewire(self, new_node, neighbors):
        """Rewire the tree to optimize paths through the new node."""
        for neighbor in neighbors:
            if neighbor == new_node.parent:
                continue
                
            # Calculate potential new cost through new_node
            potential_cost = new_node.cost + np.linalg.norm(new_node.config - neighbor.config)
            
            # Check if this gives a better path
            if potential_cost < neighbor.cost:
                # Verify connection is collision-free
                if not self.check_collision(new_node.config, neighbor.config):
                    # Remove neighbor from old parent's children
                    if neighbor.parent:
                        neighbor.parent.children.remove(neighbor)
                        
                    # Rewire through new_node
                    neighbor.parent = new_node
                    neighbor.cost = potential_cost
                    new_node.children.append(neighbor)
                    
                    # Update costs of all descendants
                    self.update_descendant_costs(neighbor) 

    def update_descendant_costs(self, node):
        """Update the costs of all descendants after rewiring."""
        for child in node.children:
            child.cost = self.cost_to_come(node, child.config)
            self.update_descendant_costs(child)

    def build_tree(self, max_iterations=500):
        """Build the RRT* tree with improved progress tracking."""
        start_time = time.time()
        iteration = 0
        best_cost = float('inf')
        
        while iteration < max_iterations:
            # Goal biasing
            if random.random() < 0.05:  # Reduced goal bias for better exploration
                random_config = self.goal_config
            else:
                random_config = self.random_config()
            
            # Extend tree
            new_node = self.extend(random_config)
            
            # Track progress
            if self.goal_node and self.goal_node.cost < best_cost:
                best_cost = self.goal_node.cost
                print(f"Iteration {iteration}: New best path found with cost {best_cost:.3f}")
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, nodes: {len(self.nodes)}")
                if self.goal_node:
                    print(f"Current best cost: {best_cost:.3f}")
            
            iteration += 1

            if len(self.nodes) == 1000:
                print(f"Max Nodes ({1000}) reached without finding goal")
                return False
            
            # Success criteria: found a path and had time to optimize it
            if self.goal_node and iteration > min(1000, max_iterations/2):
                path = self.get_path()
                if self.validate_path(path):
                    print(f"Valid path found after {iteration} iterations with cost {self.goal_node.cost:.3f}")
                    print(len(self.nodes))
                    print(f"Total roadmap building time: {time.time() - start_time:.3f} seconds")
                    return True
        
        print(f"Max iterations ({max_iterations}) reached")
        print(f"Total roadmap building time: {time.time() - start_time:.3f} seconds")
        return self.goal_node is not None

    def validate_path(self, path):
        """Validate that the path is collision-free."""
        if not path:
            return False
            
        for i in range(len(path)-1):
            if self.check_collision(path[i], path[i+1]):
                return False
        return True

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
        
        #anim.save("Env4AOTreeGrowthA2.gif", writer="imagemagick")

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
            
            if self.robot_type == "arm":
                # Draw current arm configuration
                points = self.get_arm_points(config)
                for i in range(len(points) - 1):
                    ax.plot([points[i][0], points[i+1][0]], 
                        [points[i][1], points[i+1][1]], 'b-', linewidth=3)
                ax.plot([p[0] for p in points], [p[1] for p in points], 'bo')
                
                # Draw ghost images of previous configurations
                alpha = 0.2
                for prev_config in path[:frame]:
                    prev_points = self.get_arm_points(prev_config)
                    for i in range(len(prev_points) - 1):
                        ax.plot([prev_points[i][0], prev_points[i+1][0]], 
                            [prev_points[i][1], prev_points[i+1][1]], 
                            'b-', alpha=alpha, linewidth=1)
                
                # Draw start and goal configurations
                start_points = self.get_arm_points(self.start_config)
                goal_points = self.get_arm_points(self.goal_config)
                
                ax.plot([p[0] for p in start_points], [p[1] for p in start_points], 
                    'g--', linewidth=2, label='Start')
                ax.plot([p[0] for p in goal_points], [p[1] for p in goal_points], 
                    'r--', linewidth=2, label='Goal')
                
            else:  # freeBody robot
                # Draw robot
                corners = self.get_freebody_corners(config)
                corners = np.vstack([corners, corners[0]])
                ax.fill(corners[:, 0], corners[:, 1], 'blue', alpha=0.7)
                
                # Draw direction arrow
                x, y, theta = config
                dx = 0.2 * np.cos(theta)
                dy = 0.2 * np.sin(theta)
                ax.arrow(x, y, dx, dy, head_width=0.1, color='red')
                
                # Draw ghost images of previous configurations
                alpha = 0.2
                for prev_config in path[:frame]:
                    prev_corners = self.get_freebody_corners(prev_config)
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
        
        #anim.save("Env4AORobotPathA2.gif", writer="imagemagick")
        
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
                       choices=['arm', 'freeBody'])
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--goal_rad', type=float, required=True)
    parser.add_argument('--map', type=str, required=True)
    
    args = parser.parse_args()
    
    # Convert arguments to numpy arrays
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)
    
    # Create and run RRT planner
    planner = RRTStarPlanner(start_config, goal_config, args.goal_rad, 
                        args.map, args.robot)
    
    #if args.robot == "arm":
        #planner.visualize_problem()

    if planner.build_tree():
        print("Path found!")
        #planner.animate_tree_growth()
        planner.animate_robot_path()
    else:
        print("No path found within iteration limit")
        #planner.animate_tree_growth()
        planner.animate_robot_path()

if __name__ == "__main__":
    main()