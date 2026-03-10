"""
utils.py - Utilitaires de grille et fonctions d'aide
"""

import numpy as np
from typing import List, Tuple, Set, Dict

def create_grid(rows: int, cols: int, obstacles: List[Tuple[int, int]]) -> Dict:
    """Créer un environnement de grille."""
    return {
        'rows': rows,
        'cols': cols,
        'obstacles': set(obstacles),
        'shape': (rows, cols)
    }

def is_valid_state(state: Tuple[int, int], grid: Dict) -> bool:
    """Vérifier si un état est valide."""
    r, c = state
    rows, cols = grid['rows'], grid['cols']
    return (0 <= r < rows) and (0 <= c < cols) and (state not in grid['obstacles'])

def get_neighbors(state: Tuple[int, int], grid: Dict) -> List[Tuple[int, int]]:
    """Obtenir les voisins 4-directionnels valides."""
    r, c = state
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for dr, dc in directions:
        new_state = (r + dr, c + dc)
        if is_valid_state(new_state, grid):
            neighbors.append(new_state)
    return neighbors

def get_all_states(grid: Dict) -> List[Tuple[int, int]]:
    """Récupérer tous les états libres valides."""
    states = []
    for r in range(grid['rows']):
        for c in range(grid['cols']):
            if (r, c) not in grid['obstacles']:
                states.append((r, c))
    return states

def generate_test_case(difficulty: str = 'moyenne') -> Dict:
    """Générer des cas de test prédéfinis selon la difficulté."""
    if difficulty == 'facile':
        rows, cols = 10, 10
        obstacles = [(2, 2), (2, 3), (5, 5), (5, 6), (5, 7)]
        start, goal = (0, 0), (9, 9)
    elif difficulty == 'moyenne':
        rows, cols = 15, 15
        obstacles = [
            (3, 3), (3, 4), (3, 5), (3, 6),
            (7, 7), (7, 8), (7, 9), (7, 10),
            (10, 2), (10, 3), (10, 4),
            (12, 10), (12, 11), (12, 12)
        ]
        start, goal = (0, 0), (14, 14)
    elif difficulty == 'difficile':
        rows, cols = 20, 20
        obstacles = []
        for i in range(0, 20, 4):
            for j in range(2, 18):
                if (i, j) not in [(5, 10), (13, 10)]:
                    obstacles.append((i, j))
        start, goal = (0, 0), (19, 19)
    else:
        rows, cols = 10, 10
        obstacles = []
        start, goal = (0, 0), (9, 9)

    return {
        'grid': create_grid(rows, cols, obstacles),
        'start': start,
        'goal': goal
    }