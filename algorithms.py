import numpy as np  #Import the NumPy library for numerical operations
import random

# Define a class named RandomAgent
class RandomAgent:

    # Initialize the RandomAgent with the number of possible actions
    def __init__(self, action_size):
        self.action_size = action_size

    # Method to choose an action randomly from the available actions
    def choose_action(self, state):
        return np.random.randint(self.action_size)

class QLearningAgent:
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def set_q_value(self, state, action, value):
        self.q_table[(tuple(state), action)] = value

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        future_q = 0 if done else max(self.get_q_value(next_state, a) for a in range(self.action_size))
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * future_q - current_q)
        self.set_q_value(state, action, new_q)