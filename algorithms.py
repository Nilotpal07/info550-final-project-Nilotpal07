# Import the NumPy library for numerical operations
import numpy as np

# Define a class named RandomAgent
class RandomAgent:

    # Initialize the RandomAgent with the number of possible actions
    def __init__(self, action_size):
        self.action_size = action_size

    # Method to choose an action randomly from the available actions
    def choose_action(self, state):
        return np.random.randint(self.action_size)