import numpy as np
import cv2
from problems import TicTacToeGame, PongGame, FrozenLakeGame
from algorithms import RandomAgent
import time

def render_state(env, state, step):
    if isinstance(env, TicTacToeGame):
        tile_size = 100
        img = np.ones((3 * tile_size, 3 * tile_size, 3), dtype=np.uint8) * 255
        for i in range(1, 3):
            cv2.line(img, (i * tile_size, 0), (i * tile_size, 3 * tile_size), (0, 0, 0), 2)
            cv2.line(img, (0, i * tile_size), (3 * tile_size, i * tile_size), (0, 0, 0), 2)
        board = state.reshape(3, 3)
        for i in range(3):
            for j in range(3):
                center = (j * tile_size + tile_size // 2, i * tile_size + tile_size // 2)
                if board[i, j] == 1:
                    cv2.putText(img, "X", (center[0] - 25, center[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                elif board[i, j] == 2:
                    cv2.putText(img, "O", (center[0] - 25, center[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

    elif isinstance(env, PongGame):
        tile_size = 100
        img = np.zeros((env.height * tile_size, env.width * tile_size, 3), dtype=np.uint8)
        ball_x, ball_y, left_paddle_y, right_paddle_y = state
        center_x = int(ball_x * tile_size + tile_size // 2)
        center_y = int(ball_y * tile_size + tile_size // 2)
        cv2.circle(img, (center_x, center_y), tile_size // 4, (0, 255, 0), -1, cv2.LINE_AA)
        lp_y1 = int(left_paddle_y * tile_size)
        lp_y2 = int((left_paddle_y + env.paddle_length) * tile_size)
        cv2.rectangle(img, (10, lp_y1), (30, lp_y2), (255, 100, 100), -1, cv2.LINE_AA)
        rp_y1 = int(right_paddle_y * tile_size)
        rp_y2 = int((right_paddle_y + env.paddle_length) * tile_size)
        cv2.rectangle(img, (img.shape[1] - 30, rp_y1), (img.shape[1] - 10, rp_y2), (100, 100, 255), -1, cv2.LINE_AA)

    elif isinstance(env, FrozenLakeGame):
        tile_size = 60
        grid_size = env.grid.shape[0]
        img = np.ones((grid_size * tile_size, grid_size * tile_size, 3), dtype=np.uint8) * 255
        for i in range(grid_size):
            for j in range(grid_size):
                tile = env.grid[i, j]
                color = {
                    'S': (0, 255, 0),
                    'F': (245, 245, 245),
                    'H': (60, 60, 255),
                    'G': (0, 100, 255)
                }.get(tile, (200, 200, 200))
                top_left = (j * tile_size, i * tile_size)
                bottom_right = ((j + 1) * tile_size, (i + 1) * tile_size)
                cv2.rectangle(img, top_left, bottom_right, color, -1)
                cv2.putText(img, tile, (top_left[0] + 20, top_left[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        row, col = env.get_position()
        center = (col * tile_size + tile_size // 2, row * tile_size + tile_size // 2)
        cv2.circle(img, center, tile_size // 3, (255, 255, 0), -1, cv2.LINE_AA)

    cv2.imshow(f"Project Demo - Step {step}", img)
    cv2.waitKey(300)

def demo_agent(env, action_size, agent_class):
    agent1 = agent_class(action_size)
    agent2 = agent_class(action_size) if isinstance(env, PongGame) else None
    max_steps = 50
    winner = None

    if isinstance(env, FrozenLakeGame):
        while True:
            state = env.reset()
            done = False
            steps = 0
            score = 0
            print(f"\n=== Demo: {agent_class.__name__} on {type(env).__name__} ===")
            while not done and steps < max_steps:
                action = agent1.choose_action(state)
                row, col = env.get_position()
                next_state, reward, done = env.step(action)
                if hasattr(agent1, 'update'):
                    agent1.update(state, action, reward, next_state, done)
                score += reward
                print(f"Move {steps}: State={state} (Row={row}, Col={col}), Action={action}, Reward={reward}")
                render_state(env, state, steps + 1)
                state = next_state
                steps += 1
            if score > 0:
                winner = agent_class.__name__
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
        print(f"\n=== Demo: {agent_class.__name__} on {type(env).__name__} ===")
        while not done and steps < max_steps:
            if isinstance(env, PongGame):
                action1 = agent1.choose_action(state)
                action2 = agent2.choose_action(state)
                next_state, (reward_left, reward_right), done = env.step(action1, action2)
                if hasattr(agent1, 'update'):
                    agent1.update(state, action1, reward_left, next_state, done)
                score_left += reward_left
                score_right += reward_right
                print(f"Move {steps}: Ball=({state[0]}, {state[1]}), Left PaddleY={state[2]}, Right PaddleY={state[3]}, Left Action={action1}, Right Action={action2}, Reward Left={reward_left}, Reward Right={reward_right}")
            else:
                action = agent1.choose_action(state)
                next_state, reward, done = env.step(action)
                if hasattr(agent1, 'update'):
                    agent1.update(state, action, reward, next_state, done)
                score += reward
                print(f"Move {steps}:")
                print(state.reshape(3, 3))
                time.sleep(0.5)  # Delay between X and O moves
            render_state(env, state, steps + 1)
            state = next_state
            steps += 1
        if isinstance(env, PongGame):
            winner = "Left Paddle Agent" if score_left < 0 else "Right Paddle Agent"
        else:
            winner = f"{agent_class.__name__} (1)" if score > 0 else "Opponent Agent (2)"

    print(f"Demo ended after {steps} steps (Done={done})")
    print(f"The Winner is {winner}!")
    cv2.destroyAllWindows()

def run_project_demo():
    games = {
        "TicTacToe": (TicTacToeGame(), 9),
        "Pong": (PongGame(), 3),
        "FrozenLake": (FrozenLakeGame(), 8)
    }
    for game_name, (env, action_size) in games.items():
        demo_agent(env, action_size)

if __name__ == "__main__":
    print("Project Report: Random Agent Demonstrations")
    run_project_demo()
