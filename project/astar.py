"""
astar.py - Heuristic Search Algorithms (A*, UCS, Greedy, Weighted)
Phase 2: Deterministic Planning

Compliance:
- Supports Phase 2.1 (A* Implementation)
- Supports Phase 2.2 (UCS, Greedy, A* comparison)
- Supports Experiment E.3 (Heuristic comparison via heuristic argument)
- Supports Experiment E.4 (Weighted A* via weight argument)
"""

import heapq
import time
from typing import List, Tuple, Dict, Optional, Callable

# Dependencies from tools.py (Ensure these exist in tools.py)
from tools import get_neighbors, is_valid_state


def heuristic_manhattan(state: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    Admissible heuristic: Manhattan distance.
    Suitable for 4-directional grid movement with uniform cost.
    """
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def heuristic_zero(state: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    Zero heuristic: Reduces A* to Uniform Cost Search (Dijkstra).
    Admissible but not informative.
    """
    return 0


def heuristic_euclidean(state: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """
    Euclidean distance heuristic.
    Admissible if movement cost allows diagonals or cost >= geometric distance.
    """
    return ((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2) ** 0.5


def run_search(
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: Dict,
        algorithm: str = 'astar',
        heuristic: Callable = heuristic_manhattan,
        weight: float = 1.0
) -> Optional[Dict]:
    """
    Generic search function supporting A*, UCS, Greedy, and Weighted A*.

    Args:
        start: Starting coordinate (row, col).
        goal: Target coordinate (row, col).
        grid: Dictionary containing grid metadata (rows, cols, obstacles).
        algorithm: 'astar', 'ucs', 'greedy', 'weighted_astar'.
        heuristic: Function computing h(n).
        weight: Weight for weighted A* (w >= 1). Ignored for non-weighted algos.

    Returns:
        Dictionary containing path, cost, and performance metrics compatible
        with experiments.py expectations.
    """
    if not is_valid_state(start, grid) or not is_valid_state(goal, grid):
        return {
            'success': False,
            'path': None,
            'cost': float('inf'),
            'nodes_expanded': 0,
            'algorithm': algorithm
        }

    # Priority Queue: (f_score, counter, state)
    # Counter ensures FIFO behavior for ties in f_score
    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, counter, start))

    # Track costs and parents
    g_score = {start: 0}
    came_from = {}
    closed_set = set()

    # Metrics
    nodes_expanded = 0
    max_open_size = 0
    start_time = time.time()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_expanded += 1
        max_open_size = max(max_open_size, len(open_set))

        if current == goal:
            end_time = time.time()
            path = reconstruct_path(came_from, current)
            return {
                'success': True,
                'path': path,
                'cost': g_score[goal],
                'nodes_expanded': nodes_expanded,
                'max_open_size': max_open_size,
                'execution_time': end_time - start_time,
                'algorithm': algorithm
            }

        for neighbor in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1  # Uniform cost = 1 per step

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # Calculate f_score based on algorithm type
                h_val = heuristic(neighbor, goal)

                if algorithm == 'ucs':
                    f_score = tentative_g
                elif algorithm == 'greedy':
                    f_score = h_val
                elif algorithm == 'astar':
                    f_score = tentative_g + h_val
                elif algorithm == 'weighted_astar':
                    f_score = tentative_g + weight * h_val
                else:
                    # Default to A*
                    f_score = tentative_g + h_val

                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    return {
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'algorithm': algorithm
    }


def reconstruct_path(came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from goal to start using parent pointers.
    Returns list of states from start to goal.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def extract_policy(path: List[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
    """
    Extract a deterministic policy from a planned path.
    Maps state -> action ('up', 'down', 'left', 'right').
    Required for Phase 3 (Markov Chain Construction).
    """
    policy = {}
    if not path:
        return policy

    direction_map = {
        (0, 1): 'right', (0, -1): 'left',
        (1, 0): 'down', (-1, 0): 'up'
    }

    for i in range(len(path) - 1):
        curr = path[i]
        next_state = path[i + 1]
        action_vec = (next_state[0] - curr[0], next_state[1] - curr[1])
        policy[curr] = direction_map.get(action_vec, 'stay')

    # Define action at goal as 'goal' (absorbing)
    if path:
        policy[path[-1]] = 'goal'

    return policy