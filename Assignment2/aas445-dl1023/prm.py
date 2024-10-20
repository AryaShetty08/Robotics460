import random
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation
import heapq

# Using SAT collision checking for the obstacles 
def getCorners(obstacle):
    x = obstacle[0]
    y = obstacle[1]
    theta = obstacle[2]
    w = obstacle[3]
    h = obstacle[4]

    obsV1 = ((x-(w/2)), (y+(h/2)))
    obsV2 = ((x+(w/2)), (y+(h/2)))
    obsV3 = ((x-(w/2)), (y-(h/2)))
    obsV4 = ((x+(w/2)), (y-(h/2)))

    corners = [obsV1, obsV2, obsV3, obsV4]
    rotatedCorners = []

    for i in range(len(corners)):
        # corner position from center
        x_rel = corners[i][0] - x
        y_rel = corners[i][1] - y

        # apply rotation matrix for rotated obstacles
        x_rot = math.cos(theta) * x_rel - math.sin(theta) * y_rel + x
        y_rot = math.sin(theta) * x_rel + math.cos(theta) * y_rel + y

        rotatedCorners.append((x_rot, y_rot))

    return rotatedCorners

def getProjection(axes, corners):
    a = np.array([axes[0], axes[1]])

    min_val = math.inf
    max_val = -math.inf

    for i in range(len(corners)):
        b = np.array([corners[i][0], corners[i][1]])
        product = np.dot(a, b)
        if product > max_val:
            max_val = product
        if product < min_val:
            min_val = product

    return min_val, max_val

# Link lengths from 1st assignment 
L1 = 2  
L2 = 1.5  

def f_kinematics(theta1, theta2):
    """
    Computes the world space positions of robot arms' joints and end-effector.

    Input:
    - theta1: Angle of the first link
    - theta2: Angle of the second link

    Returns:
    - tuple: Contains relative positions of (base to joint1), (joint1 to joint2), and final positions
    """
    # Joint 1 (based on base)
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    # Joint 2 (based on joint 1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    # Return (joint1, joint2, end-effector)
    return (0, 0), (x1, y1), (x2, y2)

def checkCollision(obstacle, env):
    if len(env) == 0:
        return False
    
    obsCorners = getCorners(obstacle)
    obsEdges = [(obsCorners[1][0] - obsCorners[0][0], obsCorners[1][1] - obsCorners[0][1]), 
                (obsCorners[2][0] - obsCorners[0][0], obsCorners[2][1] - obsCorners[0][1])]
    
    for i in range(len(env)):
        checkCorners = getCorners(env[i])
        checkEdges = [(checkCorners[1][0] - checkCorners[0][0], checkCorners[1][1] - checkCorners[0][1]),
                      (checkCorners[2][0] - checkCorners[0][0], checkCorners[2][1] - checkCorners[0][1])]

        normalVectors = []

        for j in range(len(obsEdges)):
            mag = math.sqrt(math.pow(-obsEdges[j][1], 2) + math.pow(obsEdges[j][0], 2))
            normalVectors.append((-obsEdges[j][1] / mag, obsEdges[j][0] / mag))

        for j in range(len(checkEdges)):
            mag = math.sqrt(math.pow(-checkEdges[j][1], 2) + math.pow(checkEdges[j][0], 2))
            normalVectors.append((-checkEdges[j][1] / mag, checkEdges[j][0] / mag))

        collision = True
        # project the corners onto the axis
        for j in range(len(normalVectors)):
            min1, max1 = getProjection(normalVectors[j], checkCorners)
            minObs, maxObs = getProjection(normalVectors[j], obsCorners) 

            if max1 < minObs or maxObs < min1:
                collision = False
                break
            
        if collision:
            return True

    return False

# Necessary for arm robot, treating the links as segments to check for collisions
def checkCollision_arm(theta1, theta2, polygons):
    base, joint1, joint2 = f_kinematics(theta1, theta2)
 
    for polygon in polygons:
        obsCorners = getCorners(polygon)
        for i in range(len(obsCorners)):
            next_i = (i + 1) % len(obsCorners)
            # Check if arm links intersect with obstacle edges
            if is_intersecting_line(base, joint1, obsCorners[i], obsCorners[next_i]) or \
               is_intersecting_line(joint1, joint2, obsCorners[i], obsCorners[next_i]):
                return True
    return False


# PRM part of code, and creating the path 
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Nearest Node to connect samples 
def find_k_nearest_neighbors(node, nodes, k):
    distances = [(euclidean_distance(node, other_node), other_node) for other_node in nodes if other_node != node]
    distances.sort(key=lambda x: x[0])
    return [neighbor for _, neighbor in distances[:k]]

# Sampling random points in space 
def generate_random_node(robot_type):
    if robot_type == 'arm':
        return (random.uniform(0, math.pi), random.uniform(-math.pi /2, math.pi / 2))  # Joint angles for the arm
    elif robot_type == 'freeBody':
        return (random.uniform(0, 20), random.uniform(0, 20), random.uniform(-math.pi, math.pi))  # Pose for freeBody

def is_intersecting_line(p1, p2, q1, q2):
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        # Check if point b lies on segment a-c
        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

    # all possibiltiies 
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

# Check collision between node and neighbor
def connection_collision(node1, node2, polygons):
    for obstacle in polygons:
        obsCorners = getCorners(obstacle)
        for i in range(len(obsCorners)):
            next_i = (i+1) % len(obsCorners)
            if is_intersecting_line(node1, node2, obsCorners[i], obsCorners[next_i]):
                return True
    return False

# Path planner
def prm_planner(start, goal, robot_type, polygons, num_nodes=500, k=6):
    nodes = [tuple(start), tuple(goal)]
    
    while len(nodes) < num_nodes:
        node = generate_random_node(robot_type)

        if robot_type == "freeBody":
            # add the width and height for collision checking
            robot_config = (*node, 0.5, 0.3)
            if not checkCollision(robot_config, polygons):
                nodes.append(node)
        elif robot_type == "arm":
            theta1, theta2 = node
            if not checkCollision_arm(theta1, theta2, polygons):
                nodes.append(node)
    
    edges = []
    
    # Connect nodes with their k nearest neighbors
    for node in nodes:
        neighbors = find_k_nearest_neighbors(node, nodes, k)
        for neighbor in neighbors:
            if robot_type == "freeBody":
                robot_node = (*node, 0.5, 0.3)
                robot_neighbor = (*neighbor, 0.5, 0.3)
                if not checkCollision(robot_node, polygons) and not checkCollision(robot_neighbor, polygons):
                    # remeber to checkif the line crosses an obstacle 
                    if not connection_collision(node, neighbor, polygons):
                        edges.append((node, neighbor))
            elif robot_type == "arm":
                theta1_node, theta2_node = node
                theta1_neighbor, theta2_neighbor = neighbor
                if not checkCollision_arm(theta1_node, theta2_node, polygons) and \
                not checkCollision_arm(theta1_neighbor, theta2_neighbor, polygons):
                    if not connection_collision(node, neighbor, polygons):
                        edges.append((node,neighbor))

    
    return nodes, edges

# Dijkstra's Algorithm to find complete path 
def dijkstra(nodes, edges, start, goal):
    graph = {node: [] for node in nodes}
    
    for edge in edges:
        node1, node2 = edge
        distance = euclidean_distance(node1, node2)
        graph[node1].append((node2, distance))
        graph[node2].append((node1, distance))
    
    queue = [(0, start)]
    distances = {node: float('inf') for node in nodes}
    distances[start] = 0
    predecessors = {node: None for node in nodes}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Reconstruct the path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = predecessors[node]

    return path[::-1]  # Return reversed path

# Get environment
def scene_from_file(filename):
    env = []

    with open(filename, "r") as file:
        for line in file:
            line = line.strip().strip('()')
            tupleValues = tuple(map(float, line.split(',')))
            env.append(tupleValues)

    return env


# Visualization of PRM 
def visualize_prm(nodes, edges, polygons, robot_type, path=None):
    fig, ax = plt.subplots()
    
    # Plot obstacles
    for obstacle in polygons:
        x, y, theta, w, h = obstacle
        t = Affine2D().rotate_around(x, y, theta) + ax.transData
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, color='gray', alpha=0.5, transform=t)
        ax.add_patch(rect)
    
    # Plot edges
    for edge in edges:
        x_values = [node[0] for node in edge]
        y_values = [node[1] for node in edge]
        ax.plot(x_values, y_values, 'bo-', markersize=2)

    # Plot path
    if path:
        x_values = [node[0] for node in path]
        y_values = [node[1] for node in path]
        ax.plot(x_values, y_values, 'ro-', linewidth=2)

    #plt.show()

    def animate_robot(fig, ax, path, robot_type):
        if robot_type == 'freeBody':
            width, height = 0.5, 0.3
            robot_shape = patches.Rectangle((0, 0), width, height, angle=0, color='blue', alpha=0.5)
            ax.add_patch(robot_shape)
        else:  # arm
            link1, = ax.plot([], [], 'b-', linewidth=4)
            link2, = ax.plot([], [], 'r-', linewidth=4)
        
        def update(frame):
            if frame < len(path):
                if robot_type == 'freeBody':
                    x, y, theta = path[frame] 
                    robot_shape.set_xy((x, y)) 
                    robot_shape.angle = np.degrees(theta)
                    return robot_shape,
                else:  # arm
                    theta1, theta2 = path[frame]
                    base, joint1, joint2 = f_kinematics(theta1, theta2)
                    link1.set_data([base[0], joint1[0]], [base[1], joint1[1]])
                    link2.set_data([joint1[0], joint2[0]], [joint1[1], joint2[1]])
                    return link1, link2

        ani = animation.FuncAnimation(fig, update, frames=len(path), interval=1000, blit=True)
        return ani
    
    anim = animate_robot(fig, ax, path, robot_type)
    plt.show()
            
    return fig, ax



# Argument Parser for running PRM from input
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'])
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--map', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    polygons = scene_from_file(args.map)

    # Create the graph of connected nodes from PRM 
    nodes, edges = prm_planner(args.start, args.goal, args.robot, polygons)

    # Get path using dijkstra
    path = dijkstra(nodes, edges, tuple(args.start), tuple(args.goal))

    fig, ax = visualize_prm(nodes, edges, polygons, args.robot, path)
    print("animate")
    #animate_robot(fig, ax, path, args.robot)
