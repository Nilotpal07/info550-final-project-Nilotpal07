import time
from problems import TicTacToeGame, PongGame, FrozenLakeGame
from algorithms import RandomAgent

def run_episode(env, agent1, agent2=None, max_steps=50):
    """
        Runs a single episode of a game with a random agent.
        :param env: The game environment.
        :param agent1: The first agent (Random Agent).
        :param agent2: The second agent (only for Pong).
        :param max_steps: Maximum number of steps allowed per episode.
        :return: A dictionary containing episode metrics (time taken, score, steps, and winner).
    """
    score1 = 0 # Score for agent1
    score2 = 0 # Score for agent2
    state = env.reset()
    done = False
    steps = 0
    start_time = time.time()

    while not done and steps < max_steps:
        if isinstance(env, PongGame):
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)
            next_state, (reward1, reward2), done = env.step(action1, action2)
            score1 += reward1
            score2 += reward2
        else:
            action = agent1.choose_action(state)
            next_state, reward, done = env.step(action)
            score1 += reward
        state = next_state
        steps += 1

    # Calculate total execution time
    total_time = time.time() - start_time
    if isinstance(env, PongGame):
        winner = "Left Paddle Agent" if score1 < 0 else "Right Paddle Agent"
        return {"time": total_time, "score_left": score1, "score_right": score2, "steps": steps, "winner": winner}
    elif isinstance(env, TicTacToeGame):
        winner = "Random Agent (1)" if score1 > 0 else "Opponent Agent (2)"
        return {"time": total_time, "efficiency": score1, "steps": steps, "winner": winner}
    else:  # FrozenLakeEnv
        winner = "Random Agent" if score1 > 0 else None
        return {"time": total_time, "efficiency": score1, "steps": steps, "winner": winner}


def run_comparisons():
    """
        Runs comparisons of the Random Agent on different game environments.
        :return: A dictionary with performance results for each game.
    """
    games = {
        "TicTacToe": (TicTacToeGame(), 9),
        "Pong": (PongGame(), 3),
        "FrozenLake": (FrozenLakeGame(), 4)
    }
    results = {}
    for game_name, (env, action_size) in games.items():
        agent1 = RandomAgent(action_size)
        agent2 = RandomAgent(action_size) if game_name == "Pong" else None
        print(f"Running RandomAgent on {game_name}...")
        if game_name == "FrozenLake":
            while True:
                metrics = run_episode(env, agent1)
                if metrics["efficiency"] > 0:
                    break
                env.reset()
        else:
            metrics = run_episode(env, agent1, agent2)
        results[game_name] = {"Random": metrics}

    # Print results for each game
    for game_name, game_results in results.items():
        print(f"\nResults for {game_name}:")
        for agent_name, metrics in game_results.items():
            if game_name == "Pong":
                print(
                    f"  {agent_name}: Time={metrics['time']:.2f}s, Score Left={metrics['score_left']}, Score Right={metrics['score_right']}, Steps={metrics['steps']}, Winner={metrics['winner']}")
            else:
                print(
                    f"  {agent_name}: Time={metrics['time']:.2f}s, Efficiency={metrics['efficiency']:.2f}, Steps={metrics['steps']}, Winner={metrics['winner']}")
    # Return performance results
    return results

if __name__ == "__main__":
    run_comparisons()