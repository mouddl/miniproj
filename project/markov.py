"""
markov.py - Markov Chain Construction and Analysis
Phase 3 & 4: Stochastic Modeling
"""


from typing import Dict, List, Tuple
from tools import get_neighbors, is_valid_state, get_all_states
import numpy as np

def build_transition_matrix(
        policy: Dict[Tuple[int, int], str],
        grid: Dict,
        epsilon: float = 0.1,
        goal_state: Tuple[int, int] = None,
        fail_state: Tuple[int, int] = None
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build the stochastic transition matrix P based on policy and uncertainty.

    Args:
        policy: Dict mapping state -> action.
        epsilon: Probability of slippage/deviation.
        goal_state: Absorbing goal state.
        fail_state: Absorbing fail state (for collisions).

    Returns:
        Tuple (P_matrix, state_list) where state_list maps indices to states.
    """
    # Define all states including special absorbing states if not in grid
    states = get_all_states(grid)

    # Add Goal and Fail to state list if they are not already in free states
    # For this implementation, we assume Goal is in free states, Fail is virtual
    # We will map Fail to a specific index at the end
    state_list = list(states)

    # Ensure goal is in state_list (it should be if it's not an obstacle)
    if goal_state and goal_state not in state_list:
        state_list.append(goal_state)

    # Add a virtual FAIL state index
    fail_idx = len(state_list)
    state_list.append(('FAIL', 'FAIL'))  # Placeholder representation

    n_states = len(state_list)
    P = np.zeros((n_states, n_states))

    # Map state to index
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    goal_idx = state_to_idx.get(goal_state, -1)

    # Direction mappings
    action_to_vec = {
        'right': (0, 1), 'left': (0, -1),
        'down': (1, 0), 'up': (-1, 0),
        'stay': (0, 0), 'goal': (0, 0)
    }

    # Lateral deviations for each action
    lateral_map = {
        'right': ['up', 'down'], 'left': ['up', 'down'],
        'up': ['left', 'right'], 'down': ['left', 'right'],
        'stay': ['up', 'down', 'left', 'right'], 'goal': []
    }

    for i, state in enumerate(state_list):
        # Absorbing states (Goal and Fail)
        if i == goal_idx:
            P[i, i] = 1.0
            continue
        if i == fail_idx:
            P[i, i] = 1.0
            continue

        # If state not in policy (unreachable), stay put
        if state not in policy:
            P[i, i] = 1.0
            continue

        action = policy[state]
        intended_vec = action_to_vec.get(action, (0, 0))

        # Intended transition
        intended_next = (state[0] + intended_vec[0], state[1] + intended_vec[1])

        # Probability distribution
        p_success = 1.0 - epsilon
        p_slip = epsilon / 2.0

        # Handle Success
        if is_valid_state(intended_next, grid):
            next_idx = state_to_idx.get(intended_next, -1)
            if next_idx != -1:
                P[i, next_idx] += p_success
            else:
                # Should not happen if intended_next is valid and in state_list
                P[i, fail_idx] += p_success
        else:
            # Collision -> Fail
            P[i, fail_idx] += p_success

        # Handle Slips (Lateral)
        for lat_action in lateral_map.get(action, []):
            lat_vec = action_to_vec[lat_action]
            lat_next = (state[0] + lat_vec[0], state[1] + lat_vec[1])

            if is_valid_state(lat_next, grid):
                next_idx = state_to_idx.get(lat_next, -1)
                if next_idx != -1:
                    P[i, next_idx] += p_slip
                else:
                    P[i, fail_idx] += p_slip
            else:
                # Collision during slip -> Fail
                P[i, fail_idx] += p_slip

    return P, state_list


def compute_state_evolution(pi0: np.ndarray, P: np.ndarray, steps: int) -> np.ndarray:
    """
    Compute pi(n) = pi(0) * P^n.

    Returns:
        Matrix of shape (steps+1, n_states) containing distribution at each step.
    """
    n_states = P.shape[0]
    history = np.zeros((steps + 1, n_states))
    history[0] = pi0

    current_pi = pi0.copy()
    for t in range(1, steps + 1):
        current_pi = current_pi @ P
        history[t] = current_pi

    return history


def calculate_absorption_metrics(P: np.ndarray, goal_idx: int, fail_idx: int) -> Dict:
    """
    Calculate absorption probabilities and mean time using Fundamental Matrix.
    Decomposes P into Q (transient) and R (absorbing).
    """
    n_states = P.shape[0]
    absorbing_indices = [goal_idx, fail_idx]
    transient_indices = [i for i in range(n_states) if i not in absorbing_indices]

    if not transient_indices:
        return {'prob_goal': 1.0 if goal_idx in absorbing_indices else 0.0, 'mean_time': 0}

    # Reorder matrix for analysis (Transient first, then Absorbing)
    # However, for simple calculation without reordering:
    # Q is submatrix of P restricted to transient states
    Q = P[np.ix_(transient_indices, transient_indices)]
    R = P[np.ix_(transient_indices, absorbing_indices)]

    # Fundamental Matrix N = (I - Q)^-1
    try:
        I = np.eye(len(transient_indices))
        N = np.linalg.inv(I - Q)

        # Absorption probabilities B = N * R
        B = N @ R

        # Mean time to absorption t = N * 1_vector
        ones = np.ones((len(transient_indices), 1))
        t = N @ ones

        # Map back to original indices if needed, here we return aggregate
        return {
            'absorption_matrix': B,
            'mean_time_vector': t.flatten(),
            'transient_indices': transient_indices,
            'absorbing_indices': absorbing_indices
        }
    except np.linalg.LinAlgError:
        return {'error': 'Singular matrix, cannot compute fundamental matrix'}


def simulate_trajectories(
        P: np.ndarray,
        start_idx: int,
        n_trials: int,
        max_steps: int,
        goal_idx: int,
        fail_idx: int
) -> Dict:
    """
    Monte-Carlo simulation of Markov trajectories.
    """
    n_states = P.shape[0]
    success_count = 0
    fail_count = 0
    steps_to_absorb = []

    for _ in range(n_trials):
        current_idx = start_idx
        for step in range(max_steps):
            if current_idx == goal_idx:
                success_count += 1
                steps_to_absorb.append(step)
                break
            if current_idx == fail_idx:
                fail_count += 1
                steps_to_absorb.append(step)
                break

            # Sample next state
            probs = P[current_idx]
            next_idx = np.random.choice(n_states, p=probs)
            current_idx = next_idx
        else:
            # Did not absorb within max_steps
            fail_count += 1
            steps_to_absorb.append(max_steps)

    return {
        'prob_goal_empirical': success_count / n_trials,
        'prob_fail_empirical': fail_count / n_trials,
        'avg_steps': np.mean(steps_to_absorb),
        'std_steps': np.std(steps_to_absorb)
    }