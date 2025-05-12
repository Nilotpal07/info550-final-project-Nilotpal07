import numpy as np
import random

# ─── Random Agent ───────────────────────────────────────────────────────────
class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def choose_action(self, state):
        return random.randrange(self.action_size)

# ─── Generic Tabular Q-Learning ─────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_size  = state_size
        self.action_size = action_size
        self.lr          = learning_rate
        self.gamma       = discount_factor
        self.epsilon     = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table     = np.zeros((state_size, action_size), dtype=float)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ─── Approximate Q-Learning with One-Hot Features ───────────────────────────
def one_hot_feature(state, action, n_states, n_actions):
    phi = np.zeros(n_states * n_actions, dtype=float)
    idx = state * n_actions + action
    phi[idx] = 1.0
    return phi

class ApproxQLearningAgent:
    def __init__(self, action_size, feature_extractor,
                 learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.action_size = action_size
        self.fe          = feature_extractor
        self.lr          = learning_rate
        self.gamma       = discount_factor
        self.epsilon     = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # infer feature dim
        self.weights = np.zeros_like(self.fe(0, 0), dtype=float)

    def q_value(self, state, action):
        phi = self.fe(state, action)
        return float(np.dot(self.weights, phi))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        q_vals = [self.q_value(state, a) for a in range(self.action_size)]
        return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            q_next = [self.q_value(next_state, a) for a in range(self.action_size)]
            target = reward + self.gamma * max(q_next)
        current = self.q_value(state, action)
        error = target - current
        phi = self.fe(state, action)
        self.weights += self.lr * error * phi
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ─── Tic-Tac-Toe Tabular Q-Learning ──────────────────────────────────────────
def board_to_key(state):
    return ''.join(map(str, state.tolist()))

class QLearningTTTAgent:
    def __init__(self, action_size,
                 learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.action_size = action_size
        self.lr          = learning_rate
        self.gamma       = discount_factor
        self.epsilon     = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table     = {}  # key: board string → np.zeros(9)

    def choose_action(self, state):
        key = board_to_key(state)
        self.q_table.setdefault(key, np.zeros(self.action_size))
        empties = np.where(state == 0)[0]
        if random.random() < self.epsilon:
            return random.choice(empties)
        q = self.q_table[key].copy()
        q[state != 0] = -np.inf
        return int(np.argmax(q))

    def update(self, state, action, reward, next_state, done):
        sk = board_to_key(state)
        nk = board_to_key(next_state)
        self.q_table.setdefault(sk, np.zeros(self.action_size))
        self.q_table.setdefault(nk, np.zeros(self.action_size))
        q_sa = self.q_table[sk][action]
        q_next = self.q_table[nk].copy()
        q_next[next_state != 0] = -np.inf
        target = reward if done else reward + self.gamma * np.max(q_next)
        self.q_table[sk][action] += self.lr * (target - q_sa)
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ─── Pong Tabular & Approx Features ─────────────────────────────────────────
def pong_key(state):
    return ','.join(map(str, state))

class QLearningPongAgent:
    def __init__(self, action_size,
                 learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.action_size = action_size
        self.lr          = learning_rate
        self.gamma       = discount_factor
        self.epsilon     = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table     = {}

    def choose_action(self, state):
        key = pong_key(state)
        self.q_table.setdefault(key, np.zeros(self.action_size))
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, done):
        sk = pong_key(state)
        nk = pong_key(next_state)
        self.q_table.setdefault(sk, np.zeros(self.action_size))
        self.q_table.setdefault(nk, np.zeros(self.action_size))
        q_sa = self.q_table[sk][action]
        best_next = np.max(self.q_table[nk])
        target = reward if done else reward + self.gamma * best_next
        self.q_table[sk][action] += self.lr * (target - q_sa)
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def pong_features(state, action):
    # hand-crafted 7-dim features
    bx, by, lp, rp = state
    return np.array([
        bx, by,
        lp, rp,
        bx*by,
        abs(lp - rp),
        action
    ], dtype=float)
