import argparse
from functools import partial
import numpy as np
import gtsam
from typing import List, Optional

# "True" function with its respective parameters
# f(x) = ax^3 + bx^2 + cx + d
def f(x, a=0.045, b=0.2, c=0.7, d=4.86):
    return a * x**3 + b * x**2 + c * x + d

def error_func(y: np.ndarray, x: np.ndarray, this: gtsam.CustomFactor, v:
    gtsam.Values, H: List[np.ndarray]):
    """
    :param y: { Given data point at x: y = f(x) }
    :type y: { array of one element }
    :param x: { Value that produces y for some function f: y = f(x) }
    :type x: { Array of one element }
    :param this: The factor
    :type this: { CustomFactor }
    :param v: { Set of Values, accessed via a key }
    :type v: { Values }
    :param H: { List of Jacobians: dErr/dInput. The inputs of THIS
    factor (the values) }
    :type H: { List of matrices }
    """
    # First, get the keys associated to THIS factor. The keys are in the same order as when the factor is constructed
    key_a = this.keys()[0]
    key_b = this.keys()[1]
    key_c = this.keys()[2]
    key_d = this.keys()[3]

    # Access the values associated with each key. Useful function include: atDouble, atVector, atPose2, atPose3...
    a = v.atDouble(key_a)
    b = v.atDouble(key_b)
    c = v.atDouble(key_c)
    d = v.atDouble(key_d)

    # Compute the prediction (the function h(.))
    yp = a * x**3 + b * x**2 + c * x + d

    # Compute the error: H(.) - zi. Notice that zi here is "fixed" per factor
    error = yp - y

    # For comp. efficiency, only compute jacobians when requested
    if H is not None:
    # GTSAM always expects H[i] to be matrices.
        # Jacobian for a
        H[0] = np.eye(1) * x**3
        # Jacobian for b
        H[1] = np.eye(1) * x**2
        H[2] = np.eye(1) * x
        H[3] = np.eye(1)
    
    return error

# Plots the ground truth, the noisy data and the resulting polynomial
def plot(T: int, GT: List[float], Z: List[float], a: float, b: float, c: float, d: float):
    import matplotlib.pyplot as plt
    x = np.arange(T)
    y = a * x**3 + b * x**2 + c * x + d
    plt.plot(x, GT, label="Ground Truth")
    plt.plot(x, Z, label="Noisy Data")
    plt.plot(x, y, label="Resulting Polynomial")
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Factor Graph with 3rd order polynomial")
    
    parser.add_argument('--initial', type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0], help="Initial guess for the parameters a, b, c, d")
    args = parser.parse_args()

    # Initial guess for the parameters a, b, c, d
    a = args.initial[0]
    b = args.initial[1]
    c = args.initial[2]
    d = args.initial[3]

    graph = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    T = 100
    GT = [] # The ground truth, for comparison
    Z = [] # GT + Normal(0, Sigma)

    # Create the key associated to the parameters a, b, c, d
    ka = gtsam.symbol('a', 0)
    kb = gtsam.symbol('b', 0)
    kc = gtsam.symbol('c', 0)
    kd = gtsam.symbol('d', 0)

    # Insert the initial guess of each key
    v.insert(ka, a)
    v.insert(kb, b)
    v.insert(kc, c)
    v.insert(kd, d)

    # Create the \Sigma (a n x n matrix, here n=1)
    sigma = 1               # change this value from 1, 5, or 10 for different noise levels
    noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

    for i in range(T):
        GT.append(f(i))
        Z.append(f(i) + np.random.normal(0.0, sigma)) # Produce the noisy data

        # This are the keys associate to each factor.
        keys = gtsam.KeyVector([ka, kb, kc, kd])

        # Create the factor:
        # Noise model - The Sigma associated to the factor
        # Keys - The keys associated to the neighboring Variables of the factor
        # Error function - The function that computes the error: h(.) - z
        # The function expected by CustomFactor has the signature
        # F(this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray])
        # Because our function has more parameters (z and i), we need to *fix* this
        # which can be done via partial.
        gf = gtsam.CustomFactor(noise_model, keys, partial(error_func, np.array([Z[i]]), np.array([i])))

        # add the factor to the graph.
        graph.add(gf)
    
    # Construct the optimizer and call with default parameters
    result = gtsam.LevenbergMarquardtOptimizer(graph, v).optimize()

    # We can print the graph, the values and evaluate the graph given some values:
    # result.print()
    # graph.print()
    # graph.printErrors(result)
    # Query the resulting values for m and b
    a = result.atDouble(ka)
    b = result.atDouble(kb)
    c = result.atDouble(kc)
    d = result.atDouble(kd)
    print("a: ", a, "b: ", b, "c: ", c, "d: ", d)

    # Print the data for plotting.
    # Should be further tested that the resulting m, b actually fit the data
    for i in range(T):
        print(i, GT[i], Z[i])

    # plot
    plot(T, GT, Z, a, b, c, d)