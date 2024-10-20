import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from matplotlib.patches import Rectangle

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

        self.load_map(map_filename)

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
        """Check collision for arm robot."""
        steps = 10
        
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
                
                for obs in self.obstacles:
                    obs_corners = obs.get_corners()
                    # Check against obstacle edges
                    for j in range(len(obs_corners)):
                        obs_start = obs_corners[j]
                        obs_end = obs_corners[(j+1)%len(obs_corners)]
                        if self.segments_intersect(seg_start, seg_end, obs_start, obs_end):
                            return True
                    
                    # Check if arm endpoint is inside the obstacle
                    if self.point_inside_polygon(seg_end, obs_corners):
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
        """Generate a random configuration with adaptive sampling."""
        if len(self.nodes) > 0:
            # 30% chance to sample near existing nodes to refine the search
            if random.random() < 0.3:
                random_node = random.choice(self.nodes)
                config = random_node.config + np.random.normal(0, 2.0, len(self.bounds))
            # 20% chance to sample along the direct path to goal
            elif random.random() < 0.2:
                t = random.random()
                config = self.start_config + t * (self.goal_config - self.start_config)
                config += np.random.normal(0, 0.5, len(self.bounds))  # Add some noise
            else:
                # Regular uniform sampling
                config = []
                for lower, upper in self.bounds:
                    config.append(random.uniform(lower, upper))
        else:
            # Regular uniform sampling for first node
            config = []
            for lower, upper in self.bounds:
                config.append(random.uniform(lower, upper))
                
        config = np.array(config)
        # Ensure configuration stays within bounds
        for i, (lower, upper) in enumerate(self.bounds):
            config[i] = max(lower, min(upper, config[i]))
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
        """Extend the tree with proper goal checking."""
        nearest = self.nearest_neighbor(random_config)
        new_config = self.steer(nearest.config, random_config)
        
        new_config = self.clip_config(new_config)

        # First check if the new configuration is valid and different enough
        if self.robot_type == "freeBody":
            pos_close = np.allclose(new_config[:2], nearest.config[:2], atol=1e-2)
            angle_close = np.allclose(new_config[2], nearest.config[2], atol=1e-3)
            if pos_close and angle_close:
                return None
        else:
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
        if self.robot_type == "freeBody":
            pos_dist = np.linalg.norm(new_config[:2] - self.goal_config[:2])
            angle_dist = abs(np.arctan2(np.sin(new_config[2] - self.goal_config[2]), 
                                      np.cos(new_config[2] - self.goal_config[2])))
            
            # Print debug info for goal checks
            if pos_dist < self.goal_radius * 2:  # Debug nearby nodes
                print(f"Near goal - pos dist: {pos_dist:.3f}, angle dist: {angle_dist:.3f}, " +
                      f"goal radius: {self.goal_radius}")
            
            # Strict goal checking
            if pos_dist < self.goal_radius and angle_dist < self.goal_radius:
                # Verify path to goal is collision-free
                if not self.check_collision(new_config, self.goal_config):
                    self.goal_node = new_node
                    return new_node
        else:
            end_effector = self.get_arm_points(new_config)[-1]
            goal_end_effector = self.get_arm_points(self.goal_config)[-1]
            dist_to_goal = np.linalg.norm(np.array(end_effector) - np.array(goal_end_effector))
            if dist_to_goal < self.goal_radius:
                if not self.check_collision(new_config, self.goal_config):
                    self.goal_node = new_node
                    return new_node
        
        return new_node

    def build_tree(self, max_iterations=10000):
        """Build the RRT with improved progress tracking and visualization."""
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
            if self.robot_type == "arm":
                best_dist = float('inf')
                for n in self.nodes:
                    end_effector = self.get_arm_points(n.config)[-1]
                    goal_end_effector = self.get_arm_points(self.goal_config)[-1]
                    dist = np.linalg.norm(np.array(end_effector) - np.array(goal_end_effector))
                    #print(f"New node added. Distance to goal: {dist}")
                    if dist < best_dist:
                        best_dist = dist
                        best_node = n
            else:  # freeBody
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
            
            # Only declare success if we actually reached the goal
            if self.goal_node:
                # Double check the path is valid
                path = self.get_path()
                valid = True
                for i in range(len(path)-1):
                    if self.check_collision(path[i], path[i+1]):
                        valid = False
                        break
                
                if valid:
                    if self.robot_type == "arm":
                        end_effector = self.get_arm_points(self.goal_node.config)[-1]
                        goal_end_effector = self.get_arm_points(self.goal_config)[-1]
                        final_dist = np.linalg.norm(np.array(end_effector) - np.array(goal_end_effector))
                    else:  # freeBody
                        final_dist = np.linalg.norm(self.goal_node.config[:2] - self.goal_config[:2])
                    print(f"Goal reached after {iteration} iterations!")
                    print(f"Final distance to goal: {final_dist:.3f}")
                    return True
                else:
                    print("Invalid path detected, continuing search...")
                    self.goal_node = None
            
            iteration += 1
            
            if iteration == max_iterations:
                print(f"Max iterations ({max_iterations}) reached without finding goal")
                if best_node:
                    print(f"Best distance achieved: {best_dist:.3f}")
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
        """Enhanced visualization with goal region and current best path."""
        fig, ax = plt.subplots()
        
        def update(frame):
            ax.clear()
            
            # Plot obstacles
            for obs in self.obstacles:
                corners = obs.get_corners()
                corners = np.vstack([corners, corners[0]])
                ax.plot(corners[:, 0], corners[:, 1], 'k-')
                ax.fill(corners[:, 0], corners[:, 1], 'gray', alpha=0.5)
            
            # Plot nodes and edges
            for node in self.nodes[:frame]:
                if node.parent:
                    parent_config = node.parent.config
                    current_config = node.config
                    if self.robot_type == "freeBody":
                        ax.plot([parent_config[0], current_config[0]], 
                               [parent_config[1], current_config[1]], 'b-', alpha=0.3)
            
            # Plot start and goal
            if self.robot_type == "freeBody":
                ax.plot(self.start_config[0], self.start_config[1], 'go')
                ax.plot(self.goal_config[0], self.goal_config[1], 'ro')
                
                # Draw goal region
                goal_circle = plt.Circle((self.goal_config[0], self.goal_config[1]), 
                                      self.goal_radius, color='r', fill=False)
                ax.add_artist(goal_circle)
            
            # Plot current best path if available
            if frame > 0 and frame == len(self.nodes):
                best_dist = float('inf')
                best_node = None
                for node in self.nodes:
                    dist = np.linalg.norm(node.config[:2] - self.goal_config[:2])
                    if dist < best_dist:
                        best_dist = dist
                        best_node = node
                
                if best_node:
                    current = best_node
                    path = []
                    while current:
                        path.append(current.config)
                        current = current.parent
                    path = path[::-1]
                    
                    # Plot the current best path
                    path = np.array(path)
                    ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, alpha=0.5)
            
            ax.set_title(f'RRT Tree Growth (Frame {frame})')
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.set_aspect('equal')
            
        anim = FuncAnimation(fig, update, frames=len(self.nodes)+1, 
                           interval=50, repeat=False)
        plt.show()

    def animate_robot_path(self):
        """Create an animation of the robot moving along the solution path."""
        path = self.get_path()
        if not path:
            print("No path found to animate")
            return
            
        fig, ax = plt.subplots()
        
        def update(frame):
            ax.clear()
            config = path[frame]
            
            # Draw obstacles
            for obs in self.obstacles:
                corners = obs.get_corners()
                corners = np.vstack([corners, corners[0]])  # Close the polygon
                ax.plot(corners[:, 0], corners[:, 1], 'k-')
                ax.fill(corners[:, 0], corners[:, 1], 'gray', alpha=0.5)
            
            # Draw full solution path
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'g--', linewidth=2, alpha=0.5)
            
            if self.robot_type == "arm":
                # Draw robot arm
                points = self.get_arm_points(config)
                for i in range(len(points) - 1):
                    ax.plot([points[i][0], points[i+1][0]], 
                           [points[i][1], points[i+1][1]], 'b-', linewidth=2)
                ax.plot([p[0] for p in points], [p[1] for p in points], 'bo')
            else:
                # Draw freeBody robot
                corners = self.get_freebody_corners(config)
                corners = np.vstack([corners, corners[0]])  # Close the polygon
                ax.fill(corners[:, 0], corners[:, 1], 'blue', alpha=0.7)
                # Draw direction arrow
                x, y, theta = config
                dx = 0.2 * np.cos(theta)
                dy = 0.2 * np.sin(theta)
                ax.arrow(x, y, dx, dy, head_width=0.1, color='red')
            
            ax.set_title(f'Robot Motion (Frame {frame})')
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.set_aspect('equal')
            
        anim = FuncAnimation(fig, update, frames=len(path), 
                           interval=100, repeat=False)
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
    planner = RRTPlanner(start_config, goal_config, args.goal_rad, 
                        args.map, args.robot)
    
    if args.robot == "arm":
        planner.visualize_problem()

    if planner.build_tree():
        print("Path found!")
        planner.animate_tree_growth()
        planner.animate_robot_path()
    else:
        print("No path found within iteration limit")

if __name__ == "__main__":
    main()