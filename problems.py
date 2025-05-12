import numpy as np # Import the NumPy library for numerical operations

class TicTacToeGame:
    # Initialize a 3x3 Tic-Tac-Toe board with zeros (empty cells)
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        # Player 1 starts the game
        self.player = 1

    def reset(self):
        # Reset the board and set the player back to 1
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1
        return self.board.flatten() # Return board as a flattened 1D array

    def step(self, action):
        # Convert 1D action to 2D coordinates
        x, y = divmod(action, 3)
        if self.board[x, y] != 0:
            return self.board.flatten(), -1, True  # Invalid move, opponent wins

        # Mark the player's move on the board
        self.board[x, y] = self.player
        if self.check_win(self.player):
            return self.board.flatten(), 1, True  # Player wins
        # If board is full
        if np.all(self.board != 0):
            return self.board.flatten(), -1, True  # Draw, opponent wins
        self.player = 2 # Switch to opponent
        empty = np.where(self.board.flatten() == 0)[0]
        if len(empty) > 0:
            opp_move = np.random.choice(empty)
            x, y = divmod(opp_move, 3)
            self.board[x, y] = 2
            if self.check_win(2):
                return self.board.flatten(), -1, True  # Opponent wins

        self.player = 1 # Switch back to player
        return self.board.flatten(), 0, False

    def check_win(self, player):
        # Check rows and columns for a win
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
            # Check diagonals for a win
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

class PongGame:
    def __init__(self):
        self.width = 10 # Width of the game area
        self.height = 10 # Height of the game area
        self.paddle_length = 2 # Length of paddles
        self.max_paddle_y = self.height - self.paddle_length # Max paddle position

        # Ball position
        self.ball_x = 2
        self.ball_y = 2

        # Paddle positions
        self.left_paddle_y = 1
        self.right_paddle_y = 1

        # Ball movement direction
        self.ball_dx = np.random.choice([-1, 1])
        self.ball_dy = np.random.choice([-1, 1])

    def reset(self):
        # Reset ball and paddles to initial positions
        self.ball_x = 2
        self.ball_y = 2
        self.left_paddle_y = 1
        self.right_paddle_y = 1
        self.ball_dx = np.random.choice([-1, 1])
        self.ball_dy = np.random.choice([-1, 1])
        return self.get_state()

    def get_state(self):
        return self.ball_x, self.ball_y, self.left_paddle_y, self.right_paddle_y

    def step(self, left_action, right_action):
        # Move left paddle
        if left_action == 0 and self.left_paddle_y > 0:
            self.left_paddle_y -= 1
        elif left_action == 2 and self.left_paddle_y < self.max_paddle_y:
            self.left_paddle_y += 1

        # Move right paddle
        if right_action == 0 and self.right_paddle_y > 0:
            self.right_paddle_y -= 1
        elif right_action == 2 and self.right_paddle_y < self.max_paddle_y:
            self.right_paddle_y += 1

        # Update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball bounces off the top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height - 1:
            self.ball_dy *= -1

        reward_left = 0
        reward_right = 0
        done = False

        # Ball reaches the left side
        if self.ball_x == 0:
            if self.left_paddle_y <= self.ball_y < self.left_paddle_y + self.paddle_length:
                reward_left = 1
                self.ball_dx *= -1
            else:
                reward_right = -1  # Right player wins
                done = True

        # Ball reaches the right side
        elif self.ball_x == self.width - 1:
            if self.right_paddle_y <= self.ball_y < self.right_paddle_y + self.paddle_length:
                reward_right = 1
                self.ball_dx *= -1
            else:
                reward_left = -1  # Left player wins
                done = True

        return self.get_state(), (reward_left, reward_right), done

class FrozenLakeGame:
    def __init__(self):
        # 9x9 grid world
        self.grid = np.array([
            ['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'F', 'H', 'H', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'F', 'F', 'H', 'F', 'F', 'F'],
            ['F', 'H', 'H', 'F', 'H', 'F', 'F', 'F', 'F'],
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'G']
        ])
        self.rows = 9
        self.cols = 9
        self.agent_row = 0
        self.agent_col = 0

    def reset(self):
        # Reset agent position to start
        self.agent_row = 0
        self.agent_col = 0
        return self.agent_row * self.cols + self.agent_col

    def step(self, action):
        # Move agent based on action
        if action == 0 and self.agent_row > 0:         # Up
            self.agent_row -= 1
        elif action == 1 and self.agent_row < self.rows - 1:  # Down
            self.agent_row += 1
        elif action == 2 and self.agent_col > 0:        # Left
            self.agent_col -= 1
        elif action == 3 and self.agent_col < self.cols - 1:  # Right
            self.agent_col += 1

        reward = 0
        done = False
        tile = self.grid[self.agent_row, self.agent_col]

        if tile == 'G':  # Goal reached
            reward = 1
            done = True
        elif tile == 'H':  # Fell into hole
            reward = -1
            done = True

        state = self.agent_row * self.cols + self.agent_col
        return state, reward, done

    def get_position(self):
        return self.agent_row, self.agent_col