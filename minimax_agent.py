import math
from state import MatchState, Action
from environment import WrestlingEnvironment


class MinimaxAgent:
    def __init__(self, env: WrestlingEnvironment, depth: int = 2):
        self.env = env
        self.depth = depth

    def choose_action(self, state: MatchState) -> tuple[Action, float]:
        best_action = None
        best_value = -math.inf

        for action in self.env.get_legal_actions(state):
            next_state = self.env.apply_action(state, action, for_us=True)
            value = self._min_value(next_state, self.depth - 1)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action, best_value

    def _max_value(self, state: MatchState, depth: int) -> float:
        if depth == 0 or state.is_terminal():
            return self.env.evaluate_state(state)

        value = -math.inf

        for action in self.env.get_legal_actions(state):
            next_state = self.env.apply_action(state, action, for_us=True)
            value = max(value, self._min_value(next_state, depth - 1))

        return value

    def _min_value(self, state: MatchState, depth: int) -> float:
        if depth == 0 or state.is_terminal():
            return self.env.evaluate_state(state)

        value = math.inf

        for action in self.env.get_opponent_actions(state):
            next_state = self.env.apply_action(state, action, for_us=False)
            value = min(value, self._max_value(next_state, depth - 1))

        return value