# Import the NumPy library for numerical operations
import numpy as np
# OpenCV for rendering visualizations
import cv2
from problems import TicTacToeGame, PongGame, FrozenLakeGame  # Importing custom game environments
from algorithms import RandomAgent # Importing a random agent to play the games

def render_state(env, state, step):
    """
        Render the current state of the game visually using OpenCV.
        :param env: The game environment.
        :param state: The current state of the game.
        :param step: The current step in the game sequence.
    """
    # If the game is Tic-Tac-Toe
    if isinstance(env, TicTacToeGame): # Create a white image
        img = np.ones((150, 150, 3), dtype=np.uint8) * 255
        # Draw grid lines
        for i in range(4):
            cv2.line(img, (i * 50, 0), (i * 50, 150), (0, 0, 0), 2)
            cv2.line(img, (0, i * 50), (150, i * 50), (0, 0, 0), 2)
        # Draw X and O on the board
        board = state.reshape(3, 3)
        for i in range(3):
            for j in range(3):
                if board[i, j] == 1:
                    cv2.putText(img, "X", (j * 50 + 15, i * 50 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif board[i, j] == 2:
                    cv2.putText(img, "O", (j * 50 + 15, i * 50 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif isinstance(env, PongGame): # If the game is Pong
        img = np.zeros((250, 250, 3), dtype=np.uint8) # Create a black background
        ball_x, ball_y, left_paddle_y, right_paddle_y = state
        # Draw grid lines
        for i in range(6):
            cv2.line(img, (i * 50, 0), (i * 50, 250), (50, 50, 50), 1)
            cv2.line(img, (0, i * 50), (250, i * 50), (50, 50, 50), 1)
        cv2.circle(img, (int(ball_x * 50 + 25), int(ball_y * 50 + 25)), 10, (0, 255, 0), -1)
        cv2.rectangle(img, (0, int(left_paddle_y * 50)), (50, int((left_paddle_y + 2) * 50)), (255, 0, 0), -1)
        cv2.rectangle(img, (200, int(right_paddle_y * 50)), (250, int((right_paddle_y + 2) * 50)), (0, 0, 255), -1)
    elif isinstance(env, FrozenLakeGame): # If the game is Frozen Lake
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255 # Create a white background
        # Draw grid lines
        for i in range(5):
            cv2.line(img, (i * 50, 0), (i * 50, 200), (0, 0, 0), 2)
            cv2.line(img, (0, i * 50), (200, i * 50), (0, 0, 0), 2)
            # Draw tiles based on environment grid
        for i in range(4):
            for j in range(4):
                tile = env.grid[i, j]
                if tile == 'S':
                    color = (0, 255, 0) # Start tile (Green)
                elif tile == 'F':
                    color = (255, 255, 255) # Frozen tile (White)
                elif tile == 'H':
                    color = (255, 0, 0) # Hole tile (Red)
                elif tile == 'G':
                    color = (0, 0, 255) # Goal tile (Blue)
                cv2.rectangle(img, (j * 50, i * 50), ((j + 1) * 50, (i + 1) * 50), color, -1)
                cv2.putText(img, tile, (j * 50 + 15, i * 50 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        row, col = env.get_position() # Get player's current position
        cv2.circle(img, (int(col * 50 + 25), int(row * 50 + 25)), 15, (255, 255, 0), -1) # Draw player circle
    cv2.imshow(f"Project Demo - Step {step}", img)
    cv2.waitKey(500)

def demo_random_agent(env, action_size):
    """
        Runs a demonstration of a random agent playing the given game environment.
        :param env: The game environment.
        :param action_size: Number of possible actions in the environment.
    """
    agent1 = RandomAgent(action_size) # Create first random agent
    agent2 = RandomAgent(action_size) if isinstance(env, PongGame) else None # Create second random agent
    max_steps = 50 # Set maximum steps per game
    winner = None # Variable to store winner

    if isinstance(env, FrozenLakeGame):
        while True:
            state = env.reset() # Reset the environment
            done = False
            steps = 0
            score = 0
            print(f"\n=== Demo: Random Agent on {type(env).__name__} ===")
            while not done and steps < max_steps:
                action = agent1.choose_action(state)
                row, col = env.get_position()
                next_state, reward, done = env.step(action)
                score += reward
                print(f"Move {steps}:")
                print(f"State={state} (Row={row}, Col={col}), Action={action}, Reward={reward}")
                render_state(env, state, steps + 1)
                state = next_state
                steps += 1
            if score > 0:  # Win condition
                winner = "Random Agent"
                break
            print(f"Episode failed (Reward={score}), restarting...")
            cv2.destroyAllWindows()
    else:
        state = env.reset()
        done = False
        steps = 0
        score = 0
        score_left = 0
        score_right = 0
        print(f"\n=== Demo: Random Agent on {type(env).__name__} ===")
        while not done and steps < max_steps:
            if isinstance(env, PongGame):
                action1 = agent1.choose_action(state)
                action2 = agent2.choose_action(state)
                next_state, (reward_left, reward_right), done = env.step(action1, action2)
                score_left += reward_left
                score_right += reward_right
                print(f"Move {steps}:")
                print(f"Ball=({state[0]}, {state[1]}), Left PaddleY={state[2]}, Right PaddleY={state[3]}, "
                      f"Left Action={action1}, Right Action={action2}, Reward Left={reward_left}, Reward Right={reward_right}")
            else:  # TicTacToe
                action = agent1.choose_action(state)
                next_state, reward, done = env.step(action)
                score += reward
                print(f"Move {steps}:")
                print(state.reshape(3, 3))
            render_state(env, state, steps + 1)
            state = next_state
            steps += 1
        if isinstance(env, PongGame):
            winner = "Left Paddle Agent" if score_left < 0 else "Right Paddle Agent"
        else:  # TicTacToe
            winner = "Random Agent (1)" if score > 0 else "Opponent Agent (2)"

    print(f"Demo ended after {steps} steps (Done={done})")
    print(f"The Winner is {winner}!")
    cv2.destroyAllWindows()

def run_project_demo():
    """
        Runs the demo for each game environment.
    """
    games = {
        "TicTacToe": (TicTacToeGame(), 9),
        "Pong": (PongGame(), 3),
        "FrozenLake": (FrozenLakeGame(), 4)
    }
    for game_name, (env, action_size) in games.items():
        demo_random_agent(env, action_size)

if __name__ == "__main__":
    print("Project Report: Random Agent Demonstrations")
    run_project_demo()