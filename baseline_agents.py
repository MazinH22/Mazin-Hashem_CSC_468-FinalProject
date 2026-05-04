import random
import math
from state import MatchState, Action
from environment import WrestlingEnvironment


class RandomAgent:
    def __init__(self, env: WrestlingEnvironment):
        self.env = env

    def choose_action(self, state: MatchState) -> Action:
        return random.choice(self.env.get_legal_actions(state))


class GreedyAgent:
    def __init__(self, env: WrestlingEnvironment):
        self.env = env

    def choose_action(self, state: MatchState) -> Action:
        actions = self.env.get_legal_actions(state)
        best_action = actions[0]
        best_score = -math.inf

        for action in actions:
            next_state = self.env.apply_action(state, action, for_us=True)
            score = self.env.evaluate_state(next_state)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action