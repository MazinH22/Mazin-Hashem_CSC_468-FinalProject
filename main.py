import matplotlib.pyplot as plt
import numpy as np

from state import MatchState, Position
from environment import WrestlingEnvironment
from minimax_agent import MinimaxAgent
from qlearning_agent import QLearningAgent
from baseline_agents import RandomAgent, GreedyAgent
from simulation import simulate_match


def collect_results(env, agent1, agent2, num_matches=100):
    wins = 0
    losses = 0
    ties = 0
    total_margin = 0

    for _ in range(num_matches):
        final_state = simulate_match(env, agent1, agent2)
        total_margin += final_state.score_diff

        if final_state.score_diff > 0:
            wins += 1
        elif final_state.score_diff < 0:
            losses += 1
        else:
            ties += 1

    return wins, losses, ties, total_margin / num_matches


def plot_q_learning_rewards(rewards):
    window = 100
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure()
    plt.plot(smoothed)
    plt.xlabel("Training Episode")
    plt.ylabel("Average Reward")
    plt.title("Q-Learning Agent Improvement Over Time")
    plt.show()


def plot_win_loss_tie(matchup_names, results):
    wins = [r[0] for r in results]
    losses = [r[1] for r in results]
    ties = [r[2] for r in results]

    x = np.arange(len(matchup_names))
    width = 0.25

    plt.figure()
    plt.bar(x - width, wins, width, label="Wins")
    plt.bar(x, losses, width, label="Losses")
    plt.bar(x + width, ties, width, label="Ties")

    plt.xticks(x, matchup_names, rotation=30, ha="right")
    plt.ylabel("Number of Matches")
    plt.title("Win/Loss/Tie Comparison Across Agents")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_score_margins(matchup_names, results):
    margins = [r[3] for r in results]

    plt.figure()
    plt.bar(matchup_names, margins)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average Score Margin")
    plt.title("Average Score Margin by Agent Matchup")
    plt.tight_layout()
    plt.show()


def main():
    env = WrestlingEnvironment(step_time=10)

    print("Training Q-learning agent...")
    q_agent = QLearningAgent(env)
    rewards = q_agent.train(episodes=3000)

    minimax_agent = MinimaxAgent(env, depth=2)
    random_agent = RandomAgent(env)
    greedy_agent = GreedyAgent(env)

    sample_state = MatchState(Position.BOTTOM, -1, 20)
    best_action, value = minimax_agent.choose_action(sample_state)

    print("\nSample decision from Minimax:")
    print(f"State: {sample_state}")
    print(f"Recommended action: {best_action.value}")
    print(f"Estimated value: {value:.2f}")

    matchup_names = [
        "Minimax vs Random",
        "Q-Learning vs Random",
        "Minimax vs Greedy",
        "Q-Learning vs Greedy",
        "Random vs Greedy",
        "Random vs Random"
    ]

    matchups = [
        (minimax_agent, random_agent),
        (q_agent, random_agent),
        (minimax_agent, greedy_agent),
        (q_agent, greedy_agent),
        (random_agent, greedy_agent),
        (random_agent, random_agent)
    ]

    results = []

    for name, (agent1, agent2) in zip(matchup_names, matchups):
        print(f"\n{name}")
        result = collect_results(env, agent1, agent2, num_matches=100)
        results.append(result)

        wins, losses, ties, avg_margin = result
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Ties: {ties}")
        print(f"Average score margin: {avg_margin:.2f}")

    plot_q_learning_rewards(rewards)
    plot_win_loss_tie(matchup_names, results)
    plot_score_margins(matchup_names, results)


if __name__ == "__main__":
    main()