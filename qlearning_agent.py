from collections import defaultdict
import random
from state import MatchState, Position, Action
from environment import WrestlingEnvironment


class QLearningAgent:
    """
    Reinforcement learning agent using Q-learning.
    The agent learns action values from repeated simulated matches.
    """

    def __init__(self, env: WrestlingEnvironment, alpha=0.05, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha          # slower learning → smoother graph
        self.gamma = gamma
        self.epsilon = epsilon      # less randomness during training
        self.q_table = defaultdict(float)

    def state_key(self, state: MatchState):
        return (state.position.value, state.score_diff, state.time_left)

    def get_q(self, state: MatchState, action: Action) -> float:
        return self.q_table[(self.state_key(state), action.value)]

    def choose_action(self, state: MatchState, training: bool = False) -> Action:
        actions = self.env.get_legal_actions(state)

        if training and random.random() < self.epsilon:
            return random.choice(actions)

        best_action = actions[0]
        best_value = self.get_q(state, best_action)

        for action in actions:
            q_value = self.get_q(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def get_reward(self, old_state: MatchState, new_state: MatchState) -> float:
        reward = (new_state.score_diff - old_state.score_diff) * 2

        if new_state.position == Position.TOP:
            reward += 0.5
        elif new_state.position == Position.BOTTOM:
            reward -= 0.5

        if new_state.is_terminal():
            if new_state.score_diff > 0:
                reward += 10
            elif new_state.score_diff < 0:
                reward -= 10

        return reward

    def update_q(self, state: MatchState, action: Action, reward: float, next_state: MatchState):
        old_q = self.get_q(state, action)

        if next_state.is_terminal():
            max_future_q = 0
        else:
            future_actions = self.env.get_legal_actions(next_state)
            max_future_q = max(self.get_q(next_state, a) for a in future_actions)

        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[(self.state_key(state), action.value)] = new_q

    def train(self, episodes: int = 3000):
        rewards_per_episode = []

        for _ in range(episodes):
            # RANDOM starting state (important improvement)
            state = MatchState(
                random.choice([Position.NEUTRAL, Position.TOP, Position.BOTTOM]),
                random.randint(-4, 4),
                random.choice([20, 30, 40, 50, 60])
            )

            total_reward = 0

            while not state.is_terminal():
                action = self.choose_action(state, training=True)
                next_state = self.env.apply_action(state, action, for_us=True)

                # opponent move
                if not next_state.is_terminal():
                    opponent_actions = self.env.get_opponent_actions(next_state)
                    if opponent_actions:
                        opponent_action = random.choice(opponent_actions)
                        next_state = self.env.apply_action(next_state, opponent_action, for_us=False)

                reward = self.get_reward(state, next_state)
                total_reward += reward

                self.update_q(state, action, reward, next_state)
                state = next_state

            rewards_per_episode.append(total_reward)

        return rewards_per_episode