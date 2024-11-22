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
    u0_key = this.keys()[2]
    u1_key = this.keys()[3]
    theta0_next_key = this.keys()[4]
    theta1_next_key = this.keys()[5]
    
    # Get the angles and velocities
    theta0 = v.atRot2(theta0_key)
    theta1 = v.atRot2(theta1_key)
    u0 = v.atDouble(u0_key)
    u1 = v.atDouble(u1_key)
    theta0_next = v.atRot2(theta0_next_key)
    theta1_next = v.atRot2(theta1_next_key)
    
    # Predict next state based on dynamics
    pred_theta0_next = theta0 * gtsam.Rot2(u0 * dt)
    pred_theta1_next = theta1 * gtsam.Rot2(u1 * dt)
    
    # Compute error
    error0 = theta0_next.between(pred_theta0_next)
    error1 = theta1_next.between(pred_theta1_next)
    
    error = np.array([error0.theta(), error1.theta()])

    # Compute Jacobians
    if H is not None:
        H[0] = np.array([[1.0], [0.0]])  # d/d theta0
        H[1] = np.array([[0.0], [1.0]])  # d/d theta1
        H[2] = np.array([[dt], [0.0]])   # d/d u0
        H[3] = np.array([[0.0], [dt]])   # d/d u1
        H[4] = np.array([[-1.0], [0.0]]) # d/d theta0_next
        H[5] = np.array([[0.0], [-1.0]]) # d/d theta1_next
        
    return error

# just for me to graph and see it
def end_effector_pos(theta0, theta1):
    x = np.cos(theta0) + np.cos(theta0 + theta1)
    y = np.sin(theta0) + np.sin(theta0 + theta1)
    return x, y

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
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    
    # Noise models
    sigma = 1.0
    vel_sigma = 1.0
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
    vel_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, vel_sigma)
    prior_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)  # Tighter constraint for priors
    
    # Create initial values
    initial_values = gtsam.Values()
    
    # Initialize all variables to zero
    for t in range(T):
        # Initialize angles and velocities 
        initial_values.insert(gtsam.symbol('a', t), gtsam.Rot2(0.0))
        initial_values.insert(gtsam.symbol('b', t), gtsam.Rot2(0.0))
        initial_values.insert(gtsam.symbol('u', t), -0.1)
        initial_values.insert(gtsam.symbol('v', t), 0.1)
    
    # Add start and goal constraints
    graph.add(gtsam.PriorFactorRot2(gtsam.symbol('a', 0), 
                                   gtsam.Rot2(start_state[0]),
                                   prior_noise_model))
    graph.add(gtsam.PriorFactorRot2(gtsam.symbol('b', 0), 
                                   gtsam.Rot2(start_state[1]),
                                   prior_noise_model))
    graph.add(gtsam.PriorFactorRot2(gtsam.symbol('a', T-1), 
                                   gtsam.Rot2(goal_state[0]),
                                   prior_noise_model))
    graph.add(gtsam.PriorFactorRot2(gtsam.symbol('b', T-1), 
                                   gtsam.Rot2(goal_state[1]),
                                   prior_noise_model))
    
    # Add velocity constraints
    graph.add(gtsam.PriorFactorDouble(gtsam.symbol('u', 0), -0.1, vel_noise_model))
    graph.add(gtsam.PriorFactorDouble(gtsam.symbol('v', 0), 0.1, vel_noise_model))
    graph.add(gtsam.PriorFactorDouble(gtsam.symbol('u', T-1), -0.1, vel_noise_model))
    graph.add(gtsam.PriorFactorDouble(gtsam.symbol('v', T-1), 0.1, vel_noise_model))
    
    # Add dynamics factors
    for t in range(T-1):
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('a', t))    # theta0
        keys.append(gtsam.symbol('b', t))    # theta1
        keys.append(gtsam.symbol('u', t))    # u0
        keys.append(gtsam.symbol('v', t))    # u1
        keys.append(gtsam.symbol('a', t+1))  # theta0_next
        keys.append(gtsam.symbol('b', t+1))  # theta1_next
        
        factor = gtsam.CustomFactor(noise_model, keys, partial(arm_trajectory_error, dt))
        graph.add(factor)
        
        # Add velocity smoothness
        if t < T-2:  # Changed to T-2 to avoid adding constraints beyond available variables
            graph.add(gtsam.BetweenFactorDouble(
                gtsam.symbol('u', t), gtsam.symbol('u', t+1), 0.0, vel_noise_model))
            graph.add(gtsam.BetweenFactorDouble(
                gtsam.symbol('v', t), gtsam.symbol('v', t+1), 0.0, vel_noise_model))
    
    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()
    
    # Extract results
    trajectory_theta0 = []
    trajectory_theta1 = []
    velocities_u0 = []
    velocities_u1 = []
    
    for t in range(T):
        trajectory_theta0.append(result.atRot2(gtsam.symbol('a', t)).theta())
        trajectory_theta1.append(result.atRot2(gtsam.symbol('b', t)).theta())
        velocities_u0.append(result.atDouble(gtsam.symbol('u', t)))
        velocities_u1.append(result.atDouble(gtsam.symbol('v', t)))

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(trajectory_theta0, label='Theta0')
    plt.plot(trajectory_theta1, label='Theta1')
    plt.xlabel('Time step')
    plt.ylabel('Angle (rad)')
    plt.title('Joint Angles')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(velocities_u0, label='Velocity0')
    plt.plot(velocities_u1, label='Velocity1')
    plt.xlabel('Time step')
    plt.ylabel('Angular velocity')
    plt.title('Joint Velocities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot end effector trajectory
    plt.figure(figsize=(8, 8))
    startX, startY = end_effector_pos(start_state[0], start_state[1])
    goalX, goalY = end_effector_pos(goal_state[0], goal_state[1])

    plt.plot(startX, startY, 'go', label='Start')
    plt.plot(goalX, goalY, 'ro', label='Goal')

    end_effector_x = []
    end_effector_y = []
    for t in range(T):
        x, y = end_effector_pos(trajectory_theta0[t], trajectory_theta1[t])
        end_effector_x.append(x)
        end_effector_y.append(y)
    
    plt.plot(end_effector_x, end_effector_y, '-b', label='Trajectory')
    plt.plot(end_effector_x, end_effector_y, 'ob', markersize=4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('End Effector Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()