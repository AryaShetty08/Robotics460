import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

# To visualize the arm robot configs
def plot_robot_arm(ax, config, color='blue', alpha=1.0):
    # start at origin and all same lengths 
    l1, l2 = 2.0, 1.5  
    x0, y0 = 0, 0      
    
    # First link 
    x1 = x0 + l1 * np.cos(config[0])
    y1 = y0 + l1 * np.sin(config[0])
    
    # Second link 
    x2 = x1 + l2 * np.cos(config[0] + config[1])
    y2 = y1 + l2 * np.sin(config[0] + config[1])
    
    # Draw links
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=2, alpha=alpha)
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=alpha)
    # Draw joints
    ax.scatter([x0, x1], [y0, y1], color=color, s=50, alpha=alpha)
    ax.scatter([x2], [y2], color=color, s=50, alpha=alpha)

# To visualize the freebody robot configs
def plot_freebody(ax, config, color='blue', alpha=1.0):
    x, y, theta = config
    width, height = 0.5, 0.3
    
    rect = patches.Rectangle(
        (-width/2, -height/2),
        width, height,
        facecolor=color,
        alpha=alpha
    )
    
    # Apply transformation
    transform = Affine2D().rotate(theta).translate(x, y) + ax.transData
    rect.set_transform(transform)
    
    ax.add_patch(rect)
    ax.scatter([x], [y], color=color, s=20, alpha=alpha)

def nearest_neighbors(robot, target, k, configs):
    """
    Find k nearest robot configurations to the target configuration

    Input:
    - robot: Either "arm" or "freebody"
    - target: N numbers that define the robot's target configuration
    - k: Integer that defines number of nearest neighbors to output
    - configs: Filename that contains configurations

    Returns:
    - K nearest robot configurations
    """
    # Read configurations from file
    configurations = []
    with open(configs, "r") as file:
        for line in file:
            config = tuple(map(float, line.strip().split()))
            configurations.append(config)

    # Calculate distances
    distances = []
    for i, config in enumerate(configurations):
        if robot == "arm":
            # For arm, use angular distance
            d1 = abs((target[0] - config[0] + np.pi) % (2*np.pi) - np.pi)
            d2 = abs((target[1] - config[1] + np.pi) % (2*np.pi) - np.pi)
            distance = np.sqrt(d1**2 + d2**2)
        else:  # freebody
            # For freebody, use Euclidean distance for position and angular difference
            pos_dist = np.sqrt((target[0] - config[0])**2 + (target[1] - config[1])**2)
            angle_dist = abs((target[2] - config[2] + np.pi) % (2*np.pi) - np.pi)
            distance = pos_dist + 0.5 * angle_dist  # Weight angular difference

        distances.append((distance, i))
    
    # Sort and get k nearest
    sorted_configs = sorted(distances, key=lambda x: x[0])
    nearest_configs = [configurations[idx] for _, idx in sorted_configs[:k]]

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    if robot == "arm":
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        # Plot target configuration
        plot_robot_arm(ax, target, color='red', alpha=1.0)
        # Plot nearest neighbors
        for config in nearest_configs:
            plot_robot_arm(ax, config, color='blue', alpha=0.5)
    else:  # freebody
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        # Plot target configuration
        plot_freebody(ax, target, color='red', alpha=1.0)
        # Plot nearest neighbors
        for config in nearest_configs:
            plot_freebody(ax, config, color='blue', alpha=0.5)

    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Robot Configurations (Red: Target, Blue: {k} Nearest Neighbors)')
    plt.show()

    return nearest_configs

if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "robot": "arm",
            "target": (np.pi/4, np.pi/3),
            "k": 3,
            "configs": "configs2arm.txt"
        },
        {
            "robot": "freebody",
            "target": (0.5, 0.5, np.pi/4),
            "k": 3,
            "configs": "configs1.txt"
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting {test['robot']} robot...")
        nearest = nearest_neighbors(**test)
        print(f"Nearest configurations: {nearest}")