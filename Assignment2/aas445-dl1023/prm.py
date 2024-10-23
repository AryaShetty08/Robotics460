import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from scipy.spatial import KDTree
import heapq

class Node:
    def __init__(self, config):
        self.config = np.array(config)  # Ensure config is numpy array
        self.neighbors = []  # List of tuples (neighbor, distance)
        self.g_cost = float('inf')
        self.parent = None

class Obstacle:
    def __init__(self, x, y, theta, width, height):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width
        self.height = height
        self._corners = None  # Cache corners
        
    def get_corners(self):
        if self._corners is None:
            w, h = self.width/2, self.height/2
            corners_local = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
            R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                         [np.sin(self.theta), np.cos(self.theta)]])
            self._corners = np.dot(corners_local, R.T) + np.array([self.x, self.y])
        return self._corners

class PRMPlanner:
    def __init__(self, start_config, goal_config, map_filename, robot_type):
        self.start_config = np.array(start_config)
        self.goal_config = np.array(goal_config)
        self.robot_type = robot_type
        self.nodes = []
        
        # Robot parameters
        if robot_type == "arm":
            self.link_lengths = [2.0, 1.5]
            self.bounds = [(-np.pi, np.pi)] * len(start_config)
            self.base_position = (10, 10)
            self.collision_check_steps = 5  # Reduced for arm
        else:
            self.robot_width = 0.5
            self.robot_height = 0.3
            self.bounds = [(0, 20), (0, 20), (-np.pi, np.pi)]
            self.collision_check_steps = 3  # Reduced for freeBody
        
        if robot_type == "arm":
            self.collision_check_steps = 10  # Increased from 5
        else:
            self.collision_check_steps = 5   # Increased from 3
            self.collision_margin = 0.1      # Add safety margin for freebody

        self.load_map(map_filename)
        
    def load_map(self, filename):
        self.obstacles = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    x, y, theta, w, h = map(float, line.strip().strip('()').split(','))
                    self.obstacles.append(Obstacle(x, y, theta, w, h))
        except Exception as e:
            print(f"Error loading map: {e}")
            
    def config_distance(self, config1, config2):
        if self.robot_type == "arm":
            diff = np.abs(config1 - config2)
            diff = np.minimum(diff, 2*np.pi - diff)
            return np.sum(diff * np.array([1.0, 0.5]))  # Weight second joint less
        else:
            pos_dist = np.linalg.norm(config1[:2] - config2[:2])
            angle_diff = abs(config1[2] - config2[2])
            angle_dist = min(angle_diff, 2*np.pi - angle_diff)
            return pos_dist + 0.3 * angle_dist
            
    def interpolate_configs(self, config1, config2, t):
        if self.robot_type == "arm":
            diff = config2 - config1
            # Handle angle wrapping
            diff = np.where(diff > np.pi, diff - 2*np.pi, diff)
            diff = np.where(diff < -np.pi, diff + 2*np.pi, diff)
            return config1 + t * diff
        else:
            pos = config1[:2] + t * (config2[:2] - config1[:2])
            angle = config1[2] + t * (config2[2] - config1[2])
            return np.array([*pos, angle])

    def check_collision(self, config1, config2):
        # Quick self-collision check
        if self.check_config_collision(config1) or self.check_config_collision(config2):
            return True
            
        # Check intermediate configurations
        for i in range(self.collision_check_steps):
            t = i / (self.collision_check_steps - 1)
            config = self.interpolate_configs(config1, config2, t)
            if self.check_config_collision(config):
                return True
        return False
        
    def check_config_collision(self, config):
        if self.robot_type == "arm":
            return self.check_arm_collision_single(config)
        else:
            return self.check_freebody_collision_single(config)
            
    def check_arm_collision_single(self, config):
        points = self.get_arm_points(config)
        
        # Check each arm segment against each obstacle
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            for obs in self.obstacles:
                obs_corners = obs.get_corners()
                # Check if segment endpoints are inside obstacle
                if self.point_inside_polygon(p1, obs_corners) or \
                   self.point_inside_polygon(p2, obs_corners):
                    return True
                # Check segment intersection with obstacle edges
                for j in range(len(obs_corners)):
                    if self.segments_intersect(p1, p2,
                                            obs_corners[j],
                                            obs_corners[(j+1)%len(obs_corners)]):
                        return True
        return False
    
    def _expand_polygon(self, corners, margin):
        """Expand polygon by moving corners outward by margin"""
        center = np.mean(corners, axis=0)
        expanded = []
        for corner in corners:
            dir_vec = corner - center
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            expanded.append(corner + margin * dir_vec)
        return np.array(expanded)

    def check_freebody_collision_single(self, config):
        # Add safety margin to robot dimensions
        robot_corners = self.get_freebody_corners(config)
        expanded_corners = self._expand_polygon(robot_corners, self.collision_margin)
        
        for obs in self.obstacles:
            obs_corners = obs.get_corners()
            
            # Quick AABB check first
            if not self.aabb_overlap(expanded_corners, obs_corners):
                continue
                
            # Check if any corner is inside the other polygon
            for corner in expanded_corners:
                if self.point_inside_polygon(corner, obs_corners):
                    return True
            for corner in obs_corners:
                if self.point_inside_polygon(corner, expanded_corners):
                    return True
                    
            # Check edge intersections
            for i in range(len(expanded_corners)):
                r_start = expanded_corners[i]
                r_end = expanded_corners[(i+1)%len(expanded_corners)]
                
                for j in range(len(obs_corners)):
                    if self.segments_intersect(r_start, r_end,
                                            obs_corners[j],
                                            obs_corners[(j+1)%len(obs_corners)]):
                        return True
        return False
        
    def aabb_overlap(self, corners1, corners2):
        min1 = np.min(corners1, axis=0)
        max1 = np.max(corners1, axis=0)
        min2 = np.min(corners2, axis=0)
        max2 = np.max(corners2, axis=0)
        return not (max1[0] < min2[0] or min1[0] > max2[0] or
                   max1[1] < min2[1] or min1[1] > max2[1])

    def segments_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def point_inside_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                     (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
                inside = not inside
            j = i
        return inside

    def get_arm_points(self, config):
        points = [np.array(self.base_position)]
        x, y = self.base_position
        angle_sum = 0
        
        for theta, length in zip(config, self.link_lengths):
            angle_sum += theta
            x += length * np.cos(angle_sum)
            y += length * np.sin(angle_sum)
            points.append(np.array([x, y]))
            
        return points

    def get_freebody_corners(self, config):
        x, y, theta = config
        w, h = self.robot_width/2, self.robot_height/2
        corners_local = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
        
        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        return np.dot(corners_local, R.T) + np.array([x, y])

    def build_roadmap(self, n_samples=5000, k=8):  # Increased k from 6
        print("Building roadmap...")
        
        # Add start and goal with extra validation
        if self.check_config_collision(self.start_config):
            raise ValueError("Start configuration is in collision!")
        if self.check_config_collision(self.goal_config):
            raise ValueError("Goal configuration is in collision!")
            
        start_node = Node(self.start_config)
        goal_node = Node(self.goal_config)
        self.nodes = [start_node, goal_node]
        
        # Sample valid configurations
        configs = []
        attempts = 0
        max_attempts = n_samples * 20  # Increased from 10
        
        while len(configs) < n_samples and attempts < max_attempts:
            config = self.sample_config()
            # Add margin check for configuration sampling
            if not self.check_config_collision(config):
                # Check minimum distance from obstacles
                if self.robot_type == "freebody":
                    corners = self.get_freebody_corners(config)
                    if self._min_obstacle_distance(corners) > self.collision_margin:
                        configs.append(config)
                else:
                    configs.append(config)
            attempts += 1
            
        if len(configs) < n_samples * 0.5:  # At least 50% of desired samples
            print(f"Warning: Only generated {len(configs)} valid configurations")
            
        # Create nodes and build connections
        for config in configs:
            self.nodes.append(Node(config))
            
        # Use KDTree for efficient nearest neighbor search
        node_configs = np.array([node.config for node in self.nodes])
        tree = KDTree(node_configs)
        
        print("Connecting nodes...")
        for i, node in enumerate(self.nodes):
            # Use more neighbors for start and goal
            local_k = k * 2 if i < 2 else k
            distances, indices = tree.query(node.config, k=min(local_k+1, len(self.nodes)))
            
            for j, idx in enumerate(indices[1:]):  # Skip self
                neighbor = self.nodes[idx]
                if neighbor not in [n for n, _ in node.neighbors]:
                    if not self.check_collision(node.config, neighbor.config):
                        dist = self.config_distance(node.config, neighbor.config)
                        node.neighbors.append((neighbor, dist))
                        neighbor.neighbors.append((node, dist))
        
        # Verify and improve connectivity
        if not self.verify_connectivity():
            print("Initial roadmap not connected. Adding additional connections...")
            self.improve_connectivity(k * 2)  # Try with more neighbors
            if not self.verify_connectivity():
                print("Warning: Failed to connect start and goal configurations")

    def animate_roadmap(self, show_animation=True, n_frames=100):
        if self.robot_type != "arm":
            print("C-space visualization is only available for robotic arms")
            return
            
        fig, ax2 = plt.subplots(figsize=(10, 10))

        # Setup C-space plot (right)
        ax2.set_title('Configuration Space')
        ax2.set_xlim(-np.pi, np.pi)
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_xlabel('θ1')
        ax2.set_ylabel('θ2')
        ax2.grid(True)
        
        # Pre-compute C-space obstacles through sampling
        resolution = 50
        theta1_range = np.linspace(-np.pi, np.pi, resolution)
        theta2_range = np.linspace(-np.pi, np.pi, resolution)
        cspace_obstacles = np.zeros((resolution, resolution))
        
        for i, theta1 in enumerate(theta1_range):
            for j, theta2 in enumerate(theta2_range):
                config = np.array([theta1, theta2])
                if self.check_config_collision(config):
                    cspace_obstacles[j, i] = 1
        
        # Plot C-space obstacles
        ax2.imshow(cspace_obstacles, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                origin='lower', cmap='YlOrRd', alpha=0.5)
        
        # Pre-compute node configurations and edges
        node_configs = np.array([node.config for node in self.nodes])
        edge_pairs = []
        if show_animation:
            for i, node in enumerate(self.nodes):
                for neighbor, _ in node.neighbors:
                    j = self.nodes.index(neighbor)
                    edge_pairs.append((i, j))
        
        # Create interpolated frame indices
        total_nodes = len(self.nodes)
        frame_indices = np.linspace(0, total_nodes - 1, n_frames, dtype=int)
        
        def update(frame):
            # Clear previous frame
            ax2.clear()
            
            # Reset plot settings            
            ax2.set_title('Configuration Space')
            ax2.set_xlim(-np.pi, np.pi)
            ax2.set_ylim(-np.pi, np.pi)
            ax2.set_xlabel('θ1')
            ax2.set_ylabel('θ2')
            ax2.grid(True)
            
            # Plot C-space obstacles
            ax2.imshow(cspace_obstacles, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                    origin='lower', cmap='YlOrRd', alpha=0.5)
            
            # Calculate how many nodes to show in this frame
            current_index = frame_indices[frame]
            
            # Plot visible nodes in C-space
            visible_configs = node_configs[:current_index+1]
            ax2.scatter(visible_configs[:, 0], visible_configs[:, 1], 
                    c='b', s=20, alpha=0.6)
            
            # Plot visible edges in C-space
            for i, j in edge_pairs:
                if i <= current_index and j <= current_index:
                    config1 = node_configs[i]
                    config2 = node_configs[j]
                    ax2.plot([config1[0], config2[0]], 
                            [config1[1], config2[1]], 
                            'k-', alpha=0.2)
            
            # Highlight start and goal configurations
            ax2.scatter([self.start_config[0]], [self.start_config[1]], 
                    c='g', s=100, label='Start')
            ax2.scatter([self.goal_config[0]], [self.goal_config[1]], 
                    c='r', s=100, label='Goal')
            ax2.legend()
            
            plt.suptitle(f'Frame {frame+1}/{n_frames}')
        
        if show_animation:
            anim = FuncAnimation(
                fig, 
                update, 
                frames=n_frames,
                interval=40,
                blit=False,
                cache_frame_data=False
            )
            plt.show()
        else:
            update(n_frames-1)
            plt.show()

    def compute_cspace_obstacles(self, resolution=50):
        """Compute C-space obstacles through sampling"""
        theta1_range = np.linspace(-np.pi, np.pi, resolution)
        theta2_range = np.linspace(-np.pi, np.pi, resolution)
        cspace_obstacles = np.zeros((resolution, resolution))
        
        for i, theta1 in enumerate(theta1_range):
            for j, theta2 in enumerate(theta2_range):
                config = np.array([theta1, theta2])
                if self.check_config_collision(config):
                    cspace_obstacles[j, i] = 1
                    
        return cspace_obstacles

    def _min_obstacle_distance(self, corners):
        """Calculate minimum distance from polygon corners to any obstacle"""
        min_dist = float('inf')
        for corner in corners:
            for obs in self.obstacles:
                obs_corners = obs.get_corners()
                for obs_corner in obs_corners:
                    dist = np.linalg.norm(corner - obs_corner)
                    min_dist = min(min_dist, dist)
        return min_dist

    def sample_config(self):
        if self.robot_type == "arm":
            config = np.array([random.uniform(lower, upper) 
                             for lower, upper in self.bounds])
        else:
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            theta = random.uniform(self.bounds[2][0], self.bounds[2][1])
            config = np.array([x, y, theta])
        return config

    def verify_connectivity(self):
        """Check if start and goal are in the same connected component"""
        visited = set()
        queue = [self.nodes[0]]  # Start node
        visited.add(self.nodes[0])
        
        while queue:
            current = queue.pop(0)
            if current == self.nodes[1]:  # Found path to goal
                return True
            
            for neighbor, _ in current.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def improve_connectivity(self, k):
        """Add more connections to improve connectivity"""
        node_configs = np.array([node.config for node in self.nodes])
        tree = KDTree(node_configs)
        
        for node in self.nodes:
            if len(node.neighbors) < k:
                distances, indices = tree.query(node.config, k=k*2)
                for idx in indices[1:]:  # Skip self
                    neighbor = self.nodes[idx]
                    if neighbor not in [n for n, _ in node.neighbors]:
                        if not self.check_collision(node.config, neighbor.config):
                            dist = self.config_distance(node.config, neighbor.config)
                            node.neighbors.append((neighbor, dist))
                            neighbor.neighbors.append((node, dist))

    def plan(self):
        path = self.a_star()
        if path is None:
            print("No path found in A*!")
            return None
        if len(path) < 2:
            print("Path too short to smooth!")
            return path
            
        smoothed_path = self.smooth_path(path)
        print(f"Path length: Original = {len(path)}, Smoothed = {len(smoothed_path)}")
        return smoothed_path

    def a_star(self):
        def heuristic(config1, config2):
            return self.config_distance(config1, config2)
        
        # Initialize
        start_node = self.nodes[0]
        goal_node = self.nodes[1]
        
        for node in self.nodes:
            node.g_cost = float('inf')
            node.parent = None
        
        start_node.g_cost = 0
        
        # Add counter to break ties in priority queue
        counter = 0
        queue = [(0, counter, start_node)]
        visited = set()
        
        while queue:
            _, _, current = heapq.heappop(queue)
            
            if current == goal_node:
                return self.extract_path(goal_node)
                
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor, cost in current.neighbors:
                if neighbor in visited:
                    continue
                    
                tentative_g = current.g_cost + cost
                
                if tentative_g < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g
                    f_score = tentative_g + heuristic(neighbor.config, goal_node.config)
                    counter += 1
                    heapq.heappush(queue, (f_score, counter, neighbor))
        
        return None

    def extract_path(self, goal_node):
        path = []
        current = goal_node
        while current is not None:
            path.append(current.config)
            current = current.parent
        return path[::-1]

    def smooth_path(self, path, iterations=50):
        if len(path) <= 2:
            return path
            
        smoothed_path = path.copy()
        min_points = max(len(path) // 4, 10)  # Preserve at least 1/4 of original points or 10 points
        
        for _ in range(iterations):
            if len(smoothed_path) <= min_points:  # Stop if path is too short
                break
                
            i = random.randint(0, len(smoothed_path)-3)
            
            if i + 2 >= len(smoothed_path)-1:
                continue
                
            j = random.randint(i+2, min(i+5, len(smoothed_path)-1))  # Limit smoothing window
            
            if not self.check_collision(smoothed_path[i], smoothed_path[j]):
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
        
        # Interpolate additional points if path is too sparse
        dense_path = []
        for i in range(len(smoothed_path)-1):
            dense_path.append(smoothed_path[i])
            # Add 5 interpolated configurations between each pair of waypoints
            for t in np.linspace(0, 1, 6)[1:-1]:
                interp_config = self.interpolate_configs(smoothed_path[i], smoothed_path[i+1], t)
                dense_path.append(interp_config)
        dense_path.append(smoothed_path[-1])
        
        return dense_path

    def animate_path(self, path):
        if not path:
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        self.anim = None
        
        def update(frame):
            ax.clear()
            self.plot_environment(ax)
            
            config = path[frame]
            
            if self.robot_type == "arm":
                points = self.get_arm_points(config)
                # Plot arm links
                for i in range(len(points)-1):
                    ax.plot([points[i][0], points[i+1][0]], 
                        [points[i][1], points[i+1][1]], 'b-', linewidth=2)
                # Plot joints
                ax.scatter([p[0] for p in points], [p[1] for p in points], 
                        c='r', s=50)
                
                # Plot full path lightly
                for i in range(len(path)):
                    path_points = self.get_arm_points(path[i])
                    for j in range(len(path_points)-1):
                        ax.plot([path_points[j][0], path_points[j+1][0]], 
                            [path_points[j][1], path_points[j+1][1]], 
                            'c-', alpha=0.1, linewidth=1)
            else:
                corners = self.get_freebody_corners(config)
                # Plot robot body
                ax.fill([c[0] for c in corners], [c[1] for c in corners], 
                    'b', alpha=0.5)
                # Plot direction indicator
                center = np.mean(corners, axis=0)
                direction = center + 0.3 * np.array([np.cos(config[2]), 
                                                np.sin(config[2])])
                ax.plot([center[0], direction[0]], [center[1], direction[1]], 
                    'r-', linewidth=2)
                
                # Plot full path lightly
                path_points = np.array([[c[0], c[1]] for c in path])
                ax.plot(path_points[:,0], path_points[:,1], 'c-', alpha=0.3)
            
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title(f'Frame {frame+1}/{len(path)}')
        
        # Create animation with slower interval
        self.anim = FuncAnimation(fig, update, frames=len(path), interval=50, repeat=False)
        
        plt.show(block=True)
        
    def plot_environment(self, ax):
        # Plot obstacles
        for obs in self.obstacles:
            corners = obs.get_corners()
            corners = np.vstack([corners, corners[0]])  # Close the polygon
            ax.fill(corners[:, 0], corners[:, 1], 'gray', alpha=0.5)
            
        # Plot start and goal configurations
        if self.robot_type == "arm":
            # Plot start configuration
            start_points = self.get_arm_points(self.start_config)
            for i in range(len(start_points)-1):
                ax.plot([start_points[i][0], start_points[i+1][0]], 
                       [start_points[i][1], start_points[i+1][1]], 
                       'g--', linewidth=2, alpha=0.5)
            
            # Plot goal configuration
            goal_points = self.get_arm_points(self.goal_config)
            for i in range(len(goal_points)-1):
                ax.plot([goal_points[i][0], goal_points[i+1][0]], 
                       [goal_points[i][1], goal_points[i+1][1]], 
                       'r--', linewidth=2, alpha=0.5)
        else:
            # Plot start configuration
            start_corners = self.get_freebody_corners(self.start_config)
            ax.fill([c[0] for c in start_corners], [c[1] for c in start_corners], 
                   'g', alpha=0.3)
            
            # Plot goal configuration
            goal_corners = self.get_freebody_corners(self.goal_config)
            ax.fill([c[0] for c in goal_corners], [c[1] for c in goal_corners], 
                   'r', alpha=0.3)

    def visualize_roadmap(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if self.robot_type == "arm":
            # Plot C-space roadmap for arm
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.set_xlabel('θ1')
            ax.set_ylabel('θ2')
            
            # Plot nodes
            configs = np.array([node.config for node in self.nodes])
            ax.scatter(configs[:, 0], configs[:, 1], c='b', s=20)
            
            # Plot edges
            for node in self.nodes:
                for neighbor, _ in node.neighbors:
                    ax.plot([node.config[0], neighbor.config[0]], 
                           [node.config[1], neighbor.config[1]], 
                           'k-', alpha=0.2)
            
            # Highlight start and goal
            ax.scatter([self.start_config[0]], [self.start_config[1]], 
                      c='g', s=100, label='Start')
            ax.scatter([self.goal_config[0]], [self.goal_config[1]], 
                      c='r', s=100, label='Goal')
        else:
            # Plot workspace roadmap for freebody
            self.plot_environment(ax)
            
            # Plot nodes
            configs = np.array([node.config for node in self.nodes])
            ax.scatter(configs[:, 0], configs[:, 1], c='b', s=20)
            
            # Plot edges
            for node in self.nodes:
                for neighbor, _ in node.neighbors:
                    ax.plot([node.config[0], neighbor.config[0]], 
                           [node.config[1], neighbor.config[1]], 
                           'k-', alpha=0.2)
            
        ax.grid(True)
        ax.legend()
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, required=True, 
                       choices=['arm', 'freeBody'])
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--map', type=str, required=True)
    
    args = parser.parse_args()
    
    # Create planner
    planner = PRMPlanner(args.start, args.goal, args.map, args.robot)
    
    # Build and visualize roadmap
    planner.build_roadmap()
    print("Visualizing roadmap...")
    planner.visualize_roadmap()

    # # Animate roadmap
    # print("Animating roadmap...")
    # try:
    #     planner.animate_roadmap()
    # except Exception as e:
    #     print(f"Error animating roadmap: {e}")
    
    # Plan path
    print("Planning path...")
    path = planner.plan()
    
    if path is not None:
        print("Path found! Animating solution...")
        planner.animate_path(path)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()