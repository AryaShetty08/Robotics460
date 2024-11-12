import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from matplotlib.patches import Rectangle
import time

# define the potetntials
# attractive
# repulsive
# calculate and use gradient to find force for robot to move this is in 2d no rotation 
# and implement

def main():
    print("hello")
    parser = argparse.ArgumentParser(description="Potential Function")

    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)

    args = parser.parse_args()
    
    # Convert arguments to numpy arrays
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)
    
if __name__ == "__main__":
    main()