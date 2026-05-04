from dataclasses import dataclass
from enum import Enum


class Position(Enum):
    NEUTRAL = "Neutral"
    TOP = "Top"
    BOTTOM = "Bottom"


class Action(Enum):
    SHOOT = "Shoot"
    DEFEND = "Defend"
    ESCAPE = "Escape"
    REVERSAL = "Reversal"
    RIDE = "Ride"


@dataclass(frozen=True)
class MatchState:
    """
    A simplified wrestling match state.

    position: our current position relative to the opponent
    score_diff: our score - opponent score
    time_left: seconds left in the match
    """
    position: Position
    score_diff: int
    time_left: int

    def is_terminal(self) -> bool:
        return self.time_left <= 0