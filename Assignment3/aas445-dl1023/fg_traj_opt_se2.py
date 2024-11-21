import numpy as np
import gtsam
from functools import partial
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from mpl_toolkits.mplot3d import Axes3D


# Error function for qt_next = qt + qd * dt
def trajectory_error(dt: float, this: gtsam.CustomFactor, 
                    v: gtsam.Values, H: List[np.ndarray]):
    # Get keys 
    qt_key = this.keys()[0]
    qd_key = this.keys()[1]
    qt_next_key = this.keys()[2]
    
    # Get values (Pose2 objects for q and qd)
    qt = v.atPose2(qt_key)
    qd = v.atPose2(qd_key)
    qt_next = v.atPose2(qt_next_key)
    
    # Predict next state based on dynamics
    pred_qt_next = gtsam.Pose2(qt.x() + qd.x() * dt, 
                              qt.y() + qd.y() * dt, 
                              qt.theta() + qd.theta() * dt)
    
    # Compute error (3D) instead of subtracting
    error = pred_qt_next.between(qt_next)
    print("qt:", qt, "qd:", qd, "qt_next:", qt_next)
    print("pred_qt_next:", pred_qt_next)

    # Compute Jacobians, matrix 3x3
    if H is not None:
        H[0] = -np.eye(3)          # derror/dqt
        H[1] = -dt * np.eye(3)     # derror/dqd
        H[2] = np.eye(3)           # derror/dqt_next
        
    return error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='Start state (x, y, theta)')
    parser.add_argument('--goal', nargs=3, type=float, default=[5.0, 5.0, 1.57], help='Goal state (x, y, theta)')
    parser.add_argument('--x0', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='Input state 1 (x, y, theta)')
    parser.add_argument('--x1', nargs=3, type=float, default=[2.0, 2.0, 0.79], help='Input state 2 (x, y, theta)')
    parser.add_argument('--T', type=int, default=50, help='Number of timesteps')
    
    args = parser.parse_args()
    
    # Parameters
    start_state = np.array(args.start)
    goal_state = np.array(args.goal)
    x0 = np.array(args.x0)
    x1 = np.array(args.x1)   
    T = args.T
    dt = 0.1
    
    graph = gtsam.NonlinearFactorGraph()
    
    # Create noise model for 3D
    sigma = 1
    noise_model = gtsam.noiseModel.Isotropic.Sigma(3, sigma)
    
    # Create initial values
    initial_values = gtsam.Values()
    
    # Initialize all variables first
    for t in range(T):
        qt_key = gtsam.symbol('q', t)
        qd_key = gtsam.symbol('d', t)
        
        # Linear interpolation for initial guess, needed?
        alpha = float(t) / (T-1)
        qt_guess = start_state * (1-alpha) + goal_state * alpha
        qd_guess = (goal_state - start_state) / (T * dt)
        #print("qt_guess:", qt_guess)
        #print("qd_guess:", qd_guess)

        initial_values.insert(qt_key, gtsam.Pose2(qt_guess[0], qt_guess[1], qt_guess[2]))
        initial_values.insert(qd_key, gtsam.Pose2(qd_guess[0], qd_guess[1], qd_guess[2]))
    
    # Add start and goal priors
    start_prior = gtsam.PriorFactorPose2(
        gtsam.symbol('q', 0),
        gtsam.Pose2(start_state[0], start_state[1], start_state[2]),
        noise_model
    )
    graph.add(start_prior)
    
    goal_prior = gtsam.PriorFactorPose2(
        gtsam.symbol('q', T-1),
        gtsam.Pose2(goal_state[0], goal_state[1], goal_state[2]),
        noise_model
    )
    graph.add(goal_prior)
    
    # add in the extra constraints here
    x0_prior = gtsam.PriorFactorPose2(
        gtsam.symbol('q', int(T/3)),
        gtsam.Pose2(x0[0], x0[1], x0[2]),
        noise_model
    )
    graph.add(x0_prior)

    x1_prior = gtsam.PriorFactorPose2(
        gtsam.symbol('q', int(2*T/3)),
        gtsam.Pose2(x1[0], x1[1], x1[2]),
        noise_model
    )
    graph.add(x1_prior)

    # Add dynamics factors
    for t in range(T-1):
        qt_key = gtsam.symbol('q', t)
        qd_key = gtsam.symbol('d', t)
        qt_next_key = gtsam.symbol('q', t+1)
        
        # Create ordered key vector
        keys = gtsam.KeyVector()
        keys.append(qt_key)
        keys.append(qd_key)
        keys.append(qt_next_key)
        
        # Create dynamics factor
        factor = gtsam.BetweenFactorPose2(
            qt_key, qt_next_key, gtsam.Pose2(0, 0, 0), noise_model
        )
        graph.add(factor)

    
    # Add smoothness factors for velocity
    velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    for t in range(T-1):
        qd_key = gtsam.symbol('d', t)
        qd_next_key = gtsam.symbol('d', t+1)
        velocity_factor = gtsam.BetweenFactorPose2(
            qd_key,
            qd_next_key,
            gtsam.Pose2(0, 0, 0),  # prefer constant velocity
            velocity_noise
        )
        graph.add(velocity_factor)
    
    # debug
    print("Factor Graph Size:", graph.size())
    print("Number of Variables:", initial_values.size())
    
    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    result = optimizer.optimize()
    
    # Get results
    trajectory_x = []
    trajectory_y = []
    trajectory_theta = []
    for t in range(T):
        qt_key = gtsam.symbol('q', t)
        pos = result.atPose2(qt_key)
        trajectory_x.append(pos.x())
        trajectory_y.append(pos.y())
        trajectory_theta.append(pos.theta())

    #print(trajectory_x)
    #print(trajectory_y)
    #print(trajectory_theta)

    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].plot(trajectory_x, trajectory_y, '-b', label='Trajectory')
    ax[0].plot(trajectory_x, trajectory_y, 'ob', markersize=4, label='Waypoints')  # Points
    ax[0].plot([x0[0]], [x0[1]], 'mo', label='Input State 1')
    ax[0].plot([x1[0]], [x1[1]], 'co', label='Input State 2')
    ax[0].set_xlabel('X Position')
    ax[0].set_ylabel('Y Position')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].axis('equal')

    timesteps = range(len(trajectory_theta))
    ax[1].plot(timesteps, trajectory_theta, '-g', label='Orientation (theta)')
    ax[1].plot(timesteps, trajectory_theta, 'ob', markersize=4, label='Waypoints')  # Points
    ax[1].axhline(y=[x0[2]], color='g', linestyle='-', label='Input State 1')
    ax[1].axhline(y=[x1[2]], color='r', linestyle='-', label='Input State 2')
    ax[1].set_xlabel('Time Step')
    ax[1].set_ylabel('Theta (Orientation)')
    ax[1].legend()
    ax[1].grid(True)
    plt.show()

    # Plot the 3D trajectory
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_x, trajectory_y, trajectory_theta, '-b', label='Trajectory')
    ax.scatter([start_state[0]], [start_state[1]], [start_state[2]], 'go', label='Start')
    ax.scatter([goal_state[0]], [goal_state[1]], [goal_state[2]], 'ro', label='Goal')
    ax.scatter([x0[0]], [x0[1]], [x0[2]], 'mo', label='Input State 1')
    ax.scatter([x1[0]], [x1[1]], [x1[2]], 'co', label='Input State 2')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Orientation (theta)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()