from state import MatchState, Position, Action


class WrestlingEnvironment:
    def __init__(self, step_time: int = 10):
        self.step_time = step_time

    def get_legal_actions(self, state: MatchState) -> list[Action]:
        if state.position == Position.NEUTRAL:
            return [Action.SHOOT, Action.DEFEND]
        if state.position == Position.BOTTOM:
            return [Action.ESCAPE, Action.REVERSAL]
        if state.position == Position.TOP:
            return [Action.RIDE]
        return []

    def get_opponent_actions(self, state: MatchState) -> list[Action]:
        if state.position == Position.NEUTRAL:
            return [Action.SHOOT, Action.DEFEND]
        if state.position == Position.TOP:
            return [Action.ESCAPE, Action.REVERSAL]
        if state.position == Position.BOTTOM:
            return [Action.RIDE]
        return []

    def apply_action(self, state: MatchState, action: Action, for_us: bool = True) -> MatchState:
        new_time = max(0, state.time_left - self.step_time)
        score = state.score_diff
        position = state.position

        if for_us:
            if position == Position.NEUTRAL:
                if action == Action.SHOOT:
                    score += 2
                    position = Position.TOP

            elif position == Position.BOTTOM:
                if action == Action.ESCAPE:
                    score += 1
                    position = Position.NEUTRAL
                elif action == Action.REVERSAL:
                    score += 2
                    position = Position.TOP

            elif position == Position.TOP:
                if action == Action.RIDE:
                    position = Position.TOP

        else:
            if position == Position.NEUTRAL:
                if action == Action.SHOOT:
                    score -= 2
                    position = Position.BOTTOM

            elif position == Position.TOP:
                if action == Action.ESCAPE:
                    score -= 1
                    position = Position.NEUTRAL
                elif action == Action.REVERSAL:
                    score -= 2
                    position = Position.BOTTOM

            elif position == Position.BOTTOM:
                if action == Action.RIDE:
                    position = Position.BOTTOM

        return MatchState(position, score, new_time)

    def evaluate_state(self, state: MatchState) -> float:
        position_bonus = 0

        if state.position == Position.TOP:
            position_bonus = 1
        elif state.position == Position.BOTTOM:
            position_bonus = -1

        time_pressure_bonus = 0

        if state.time_left <= 20:
            time_pressure_bonus = 0.5 * state.score_diff

        return (3 * state.score_diff) + position_bonus + time_pressure_bonus