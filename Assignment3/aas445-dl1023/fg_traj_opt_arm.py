import gtsam.noiseModel
import numpy as np
import gtsam
from functools import partial
from typing import List
import matplotlib.pyplot as plt
import argparse

# Error function for 2 arm robot
def arm_trajectory_error(dt: float, this: gtsam.CustomFactor, 
                        v: gtsam.Values, H: List[np.ndarray]):
    # Get keys
    theta0_key = this.keys()[0]
    theta1_key = this.keys()[1]
    theta0_next_key = this.keys()[2]
    theta1_next_key = this.keys()[3]
    
    # Get the angles of arms
    theta0 = v.atRot2(theta0_key).theta()
    theta1 = v.atRot2(theta1_key).theta()
    theta0_next = v.atRot2(theta0_next_key).theta()
    theta1_next = v.atRot2(theta1_next_key).theta()
    
    # Predict next state based on dynamics
    pred_theta0_next = theta0 + (theta0_next - theta0) / dt
    pred_theta1_next = theta1 + (theta1_next - theta1) / dt
    
    # Compute error
    error = np.array([theta0_next - pred_theta0_next, theta1_next - pred_theta1_next])
    
    # Compute Jacobians, matrix 2x4
    if H is not None:
        # Create the 2x4 Jacobian matrix
        J = np.zeros((2, 4))
        
        # Derivatives of error[0] with respect to variables
        J[0, 0] = -1 / dt  # d(error[0])/d(theta0)
        J[0, 2] = 1 / dt   # d(error[0])/d(theta0_next)
        
        # Derivatives of error[1] with respect to variables
        J[1, 1] = -1 / dt  # d(error[1])/d(theta1)
        J[1, 3] = 1 / dt   # d(error[1])/d(theta1_next)
        
        # Assign the computed Jacobian to H[0]
        H[0] = J
        print(f"Jacobian assigned to H[0]:\n{H[0]}")  # Should be (2, 4)
        
    return error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0], help='Start state (theta0, theta1)')
    parser.add_argument('--goal', nargs=2, type=float, default=[3.14, 1.57], help='Goal state (theta0, theta1)')
    parser.add_argument('--T', type=int, default=50, help='Number of timesteps')
    
    args = parser.parse_args()
    
    # Parameters
    start_state = np.array(args.start)
    goal_state = np.array(args.goal)
    T = args.T
    dt = 0.1
    
    graph = gtsam.NonlinearFactorGraph()
    
    # Create noise model for angles
    # need two different noise models for dynamics factors and prior factors 
    sigma = 0.1
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma]))
    prior_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

    # Create initial values
    initial_values = gtsam.Values()
    
    # Initialize all variables first
    # IMPORTANT 
    # had to rename the theta0 and theta1 since they only accept characters to a and b
    for t in range(T):
        theta0_key = gtsam.symbol('a', t)
        theta1_key = gtsam.symbol('b', t)
        
        # Linear interpolation for initial guess, needed again
        alpha = float(t) / (T-1)
        theta0_guess = start_state[0] * (1-alpha) + goal_state[0] * alpha
        theta1_guess = start_state[1] * (1-alpha) + goal_state[1] * alpha
        
        initial_values.insert(theta0_key, gtsam.Rot2.fromAngle(theta0_guess))
        initial_values.insert(theta1_key, gtsam.Rot2.fromAngle(theta1_guess))
    
    # Add start and goal priors
    # have to do them separately since it doesn't like multiple arguments

    start_prior = gtsam.PriorFactorRot2(
        gtsam.symbol('a', 0), gtsam.Rot2.fromAngle(start_state[0]),
        prior_noise_model
    )
    graph.add(start_prior)

    start_prior_b = gtsam.PriorFactorRot2(
        gtsam.symbol('b', 0), gtsam.Rot2.fromAngle(start_state[1]),
        prior_noise_model
    )
    graph.add(start_prior_b)

    goal_prior = gtsam.PriorFactorRot2(
        gtsam.symbol('a', T-1), gtsam.Rot2.fromAngle(goal_state[0]),
        prior_noise_model
    )
    graph.add(goal_prior)

    goal_prior_b = gtsam.PriorFactorRot2(
        gtsam.symbol('b', T-1), gtsam.Rot2.fromAngle(goal_state[1]),
        prior_noise_model
    )
    graph.add(goal_prior_b)

    
    # Add dynamics factors
    for t in range(T-1):
        theta0_key = gtsam.symbol('a', t)
        theta1_key = gtsam.symbol('b', t)
        theta0_next_key = gtsam.symbol('a', t+1)
        theta1_next_key = gtsam.symbol('b', t+1)
        
        # Create ordered key vector
        keys = gtsam.KeyVector()
        keys.append(theta0_key)
        keys.append(theta1_key)
        keys.append(theta0_next_key)
        keys.append(theta1_next_key)
        
        # Create dynamics factor, have to use nonlinear
        factor = gtsam.CustomFactor(
            noise_model,
            keys,
            partial(arm_trajectory_error, dt)
        )
        graph.add(factor)

    print(graph)

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    result = optimizer.optimize()
    
    # Get results
    trajectory_theta0 = []
    trajectory_theta1 = []
    for t in range(T):
        theta0_key = gtsam.symbol('a', t)
        theta1_key = gtsam.symbol('b', t)
        trajectory_theta0.append(result.atRot2(theta0_key).theta())
        trajectory_theta1.append(result.atRot2(theta1_key).theta())

    print(trajectory_theta0)
    print(trajectory_theta1)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(trajectory_theta0, trajectory_theta1, '-b', label='Trajectory')
    plt.plot([start_state[0]], [start_state[1]], 'go', label='Start')
    plt.plot([goal_state[0]], [goal_state[1]], 'ro', label='Goal')
    plt.xlabel('Theta0')
    plt.ylabel('Theta1')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()

# Remember to make factor graph