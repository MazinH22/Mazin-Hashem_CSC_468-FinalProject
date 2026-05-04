import random
from state import MatchState, Position
from environment import WrestlingEnvironment


def random_start_state() -> MatchState:
    position = random.choice([Position.NEUTRAL, Position.TOP, Position.BOTTOM])
    score_diff = random.randint(-4, 4)
    time_left = random.choice([20, 30, 40, 50, 60])

    return MatchState(position, score_diff, time_left)


def _get_agent_action(agent, state: MatchState):
    result = agent.choose_action(state)
    if isinstance(result, tuple):
        return result[0]
    return result


def simulate_match(env: WrestlingEnvironment, our_agent, opponent_agent, verbose: bool = False) -> MatchState:
    state = random_start_state()

    if verbose:
        print(f"Starting state: {state}")

    while not state.is_terminal():
        our_action = _get_agent_action(our_agent, state)
        state = env.apply_action(state, our_action, for_us=True)

        if verbose:
            print(f"Our action: {our_action.value} -> {state}")

        if state.is_terminal():
            break

        opponent_action = _get_agent_action(opponent_agent, state)
        state = env.apply_action(state, opponent_action, for_us=False)

        if verbose:
            print(f"Opponent action: {opponent_action.value} -> {state}")

    return state


def evaluate_agent(env: WrestlingEnvironment, our_agent, opponent_agent, num_matches: int = 100):
    wins = 0
    losses = 0
    ties = 0
    total_margin = 0

    for _ in range(num_matches):
        final_state = simulate_match(env, our_agent, opponent_agent)
        total_margin += final_state.score_diff

        if final_state.score_diff > 0:
            wins += 1
        elif final_state.score_diff < 0:
            losses += 1
        else:
            ties += 1

    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Ties: {ties}")
    print(f"Average score margin: {total_margin / num_matches:.2f}")