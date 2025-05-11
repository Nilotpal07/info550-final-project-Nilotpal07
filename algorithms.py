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


def get_state_key(state):
    # Convert state to a hashable key
    if isinstance(state, np.ndarray):
        return tuple(state.flatten())
    elif isinstance(state, (list, tuple)):
        return tuple(state)
    else:
        return state


class QLearningAgent:
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1, epsilon_decay=0.995, epsilon_min=0.05):
        self.action_size = action_size
        self.q_table = {}  # Dictionary to store Q-values
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        state_key = get_state_key(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            self.q_table.setdefault(state_key, np.zeros(self.action_size))
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state, action, reward, next_state, done):
        state_key = get_state_key(state)
        next_state_key = get_state_key(next_state)

        self.q_table.setdefault(state_key, np.zeros(self.action_size))
        self.q_table.setdefault(next_state_key, np.zeros(self.action_size))

        best_next_action = np.max(self.q_table[next_state_key])
        td_target = reward + self.gamma * best_next_action * (0 if done else 1)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)