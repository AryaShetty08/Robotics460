import numpy as np
import gtsam
from functools import partial
from typing import List
import matplotlib.pyplot as plt
import argparse

# Error function for qt_next = qt + qd * dt
def trajectory_error(dt: float, this: gtsam.CustomFactor, 
                    v: gtsam.Values, H: List[np.ndarray]):
    # Get keys 
    qt_key = this.keys()[0]
    qd_key = this.keys()[1]
    qt_next_key = this.keys()[2]
    
    # Get values (vectors for x y)
    qt = v.atVector(qt_key)
    qd = v.atVector(qd_key)
    qt_next = v.atVector(qt_next_key)
    
    # Predict next state based on dynamics
    pred_qt_next = qt + qd * dt
    
    # Compute error (2D)
    error = qt_next - pred_qt_next
    
    # Compute Jacobians, matrix 2d
    if H is not None:
        H[0] = -np.eye(2)          # derror/dqt
        H[1] = -dt * np.eye(2)     # derror/dqd
        H[2] = np.eye(2)           # derror/dqt_next
        
    return error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0], help='Start state')
    parser.add_argument('--goal', nargs=2, type=float, default=[5.0, 5.0], help='Goal state')
    parser.add_argument('--x0', nargs=2, type=float, default=[0.0, 0.0], help='Input state 1')
    parser.add_argument('--x1', nargs=2, type=float, default=[2.0, 2.0], help='Input state 2')
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
    
    # Create noise model for 2D
    sigma = 1
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
    
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
        
        initial_values.insert(qt_key, gtsam.Point2(qt_guess[0], qt_guess[1]))
        initial_values.insert(qd_key, gtsam.Point2(qd_guess[0], qd_guess[1]))
    
    # Add start and goal priors
    start_prior = gtsam.PriorFactorVector(
        gtsam.symbol('q', 0),
        gtsam.Point2(start_state[0], start_state[1]),
        noise_model
    )
    graph.add(start_prior)
    
    goal_prior = gtsam.PriorFactorVector(
        gtsam.symbol('q', T-1),
        gtsam.Point2(goal_state[0], goal_state[1]),
        noise_model
    )
    graph.add(goal_prior)
    
    # add in the extra constraints here
    x0_prior = gtsam.PriorFactorVector(
        gtsam.symbol('q', int(T/3)),
        gtsam.Point2(x0[0], x0[1]),
        noise_model
    )
    graph.add(x0_prior)

    x1_prior = gtsam.PriorFactorVector(
        gtsam.symbol('q', int(2*T/3)),
        gtsam.Point2(x1[0], x1[1]),
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
        factor = gtsam.CustomFactor(
            noise_model,
            keys,
            partial(trajectory_error, dt)
        )
        graph.add(factor)
    
    # Add smoothness factors for velocity
    velocity_noise = gtsam.noiseModel.Isotropic.Sigma(2, 0.1)
    for t in range(T-1):
        qd_key = gtsam.symbol('d', t)
        qd_next_key = gtsam.symbol('d', t+1)
        velocity_factor = gtsam.BetweenFactorVector(
            qd_key,
            qd_next_key,
            gtsam.Point2(0, 0),  # prefer constant velocity
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
    for t in range(T):
        qt_key = gtsam.symbol('q', t)
        pos = result.atVector(qt_key)
        trajectory_x.append(pos[0])
        trajectory_y.append(pos[1])

    #print(trajectory_x)
    #print(trajectory_y)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(trajectory_x, trajectory_y, '-b', alpha=0.5, label='Trajectory')  # Line
    plt.plot(trajectory_x, trajectory_y, 'ob', markersize=4, label='Waypoints')  # Points
    plt.plot([start_state[0]], [start_state[1]], 'go', label='Start')
    plt.plot([goal_state[0]], [goal_state[1]], 'ro', label='Goal')
    plt.plot([x0[0]], [x0[1]], 'mo', label='Input State 1')
    plt.plot([x1[0]], [x1[1]], 'co', label='Input State 2')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()

# draw the factor graph by hand?