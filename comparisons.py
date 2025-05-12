import time
import numpy as np

from problems import TicTacToeGame, PongGame, FrozenLakeGame
from algorithms import RandomAgent, QLearningAgent, ApproxQLearningAgent, one_hot_feature

def run_episode(env, agent1, agent2=None, max_steps=500):
    """
    Runs one episode in `env` using agent1 (and agent2 for Pong).
    Returns (reward1, reward2_or_None, steps, elapsed_time).
    """
    state = env.reset()
    done = False
    steps = 0
    total1 = 0
    total2 = 0
    start = time.time()

    while not done and steps < max_steps:
        if isinstance(env, PongGame):
            a1 = agent1.choose_action(state)
            a2 = agent2.choose_action(state)
            (ns, (r1, r2), done) = env.step(a1, a2)
            if hasattr(agent1, 'update'):
                agent1.update(state, a1, r1, ns, done)
                agent2.update(state, a2, r2, ns, done)
            total1 += r1
            total2 += r2
        else:
            a = agent1.choose_action(state)
            ns, r, done = env.step(a)
            if hasattr(agent1, 'update'):
                agent1.update(state, a, r, ns, done)
            total1 += r
        state = ns
        steps += 1

    elapsed = time.time() - start
    return total1, (total2 if isinstance(env, PongGame) else None), steps, elapsed

def run_comparisons():
    games = {
        "TicTacToe": (TicTacToeGame(), 9),
        "Pong":      (PongGame(),      3),
        "FrozenLake":(FrozenLakeGame(), 4)
    }

    # hyperparameters
    train_eps = 1500
    test_eps  = 200
    max_steps = 500

    for name, (env, action_size) in games.items():
        print(f"\n=== {name} ===")
        results = {}

        # Run RandomAgent
        rand = RandomAgent(action_size)
        r1, r2, s, t = [], [], [], []
        for _ in range(test_eps):
            rr, rr2, ss, tt = run_episode(env, rand, RandomAgent(action_size) if isinstance(env, PongGame) else None, max_steps)
            r1.append(rr);
            if rr2 is not None: r2.append(rr2)
            s.append(ss); t.append(tt)
            env.reset()
        results["Random"] = {
            "avg_reward1": np.mean(r1),
            "avg_reward2": np.mean(r2) if r2 else None,
            "avg_steps":   np.mean(s),
            "avg_time":    np.mean(t)
        }

        # For TicTacToe and Pong
        if name == "FrozenLake":
            # Tabular Q-Learning
            state_size = env.rows * env.cols
            ql = QLearningAgent(
                state_size, action_size,
                learning_rate=0.1, discount_factor=0.99,
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05
            )
            # train
            for _ in range(train_eps):
                run_episode(env, ql, max_steps=max_steps)
                env.reset()
            # test
            r1, s, t = [], [], []
            for _ in range(test_eps):
                rr, _, ss, tt = run_episode(env, ql, max_steps=max_steps)
                r1.append(rr); s.append(ss); t.append(tt)
                env.reset()
            results["Q learning"] = {
                "avg_reward1": np.mean(r1),
                "avg_steps":   np.mean(s),
                "avg_time":    np.mean(t)
            }

            # Approximate Q-Learning
            fe = lambda s,a: one_hot_feature(s, a, state_size, action_size)
            aq = ApproxQLearningAgent(
                action_size,
                feature_extractor=fe,
                learning_rate=0.1, discount_factor=0.99,
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05
            )
            # train
            for _ in range(train_eps):
                run_episode(env, aq, max_steps=max_steps)
                env.reset()
            # test
            r1, s, t = [], [], []
            for _ in range(test_eps):
                rr, _, ss, tt = run_episode(env, aq, max_steps=max_steps)
                r1.append(rr); s.append(ss); t.append(tt)
                env.reset()
            results["Approx Q-learning"] = {
                "avg_reward1": np.mean(r1),
                "avg_steps":   np.mean(s),
                "avg_time":    np.mean(t)
            }

        # print
        if name == "Pong":
            print(f"{'Agent':<12} {'R1':>6} {'R2':>6} {'Steps':>6} {'Time(s)':>8}")
            for agent, m in results.items():
                print(f"{agent:<12} {m['avg_reward1']:6.2f} {m['avg_reward2']:6.2f} "
                      f"{m['avg_steps']:6.1f} {m['avg_time']:8.3f}")
        else:
            print(f"{'Agent':<12} {'Reward':>8} {'Steps':>6} {'Time(s)':>8}")
            for agent, m in results.items():
                print(f"{agent:<12} {m['avg_reward1']:8.3f} {m['avg_steps']:6.1f} {m['avg_time']:8.3f}")

if __name__ == "__main__":
    run_comparisons()
