# -*- coding: utf-8 -*-
"""
Module: astar.py
Description: Implémentation des algorithmes de recherche heuristique (A*, UCS, Greedy)
Auteur: Mini-Projet Planification Robuste
Date: Mars 2026
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass(order=True)
class Noeud:
    """Représentation d'un nœud dans la recherche"""
    f: float
    g: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional['Noeud'] = field(compare=False, default=None)


class Grille:
    """Classe représentant la grille de navigation"""

    def __init__(self, taille: Tuple[int, int], obstacles: List[Tuple[int, int]] = None):
        self.lignes, self.colonnes = taille
        self.obstacles = set(obstacles) if obstacles else set()
        self.grille = np.zeros(taille, dtype=int)

        # Marquer les obstacles
        for obs in self.obstacles:
            self.grille[obs] = 1

    def est_valide(self, position: Tuple[int, int]) -> bool:
        """Vérifie si une position est valide (dans la grille et non obstacle)"""
        x, y = position
        return (0 <= x < self.lignes and
                0 <= y < self.colonnes and
                position not in self.obstacles)

    def voisins(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Retourne les voisins valides (4-connexité)"""
        x, y = position
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        voisins_valides = []

        for dx, dy in directions:
            nouvelle_pos = (x + dx, y + dy)
            if self.est_valide(nouvelle_pos):
                voisins_valides.append(nouvelle_pos)

        return voisins_valides

    def cout_deplacement(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Retourne le coût de déplacement (uniforme = 1)"""
        return 1.0


def heuristique_manhattan(position: Tuple[int, int],
                          objectif: Tuple[int, int]) -> float:
    """
    Heuristique de Manhattan (admissible pour 4-connexité)
    h(n) = |x - x_goal| + |y - y_goal|
    """
    return abs(position[0] - objectif[0]) + abs(position[1] - objectif[1])


def heuristique_euclidienne(position: Tuple[int, int],
                            objectif: Tuple[int, int]) -> float:
    """Heuristique euclidienne"""
    return np.sqrt((position[0] - objectif[0]) ** 2 +
                   (position[1] - objectif[1]) ** 2)


def recherche_astar(grille: Grille,
                    depart: Tuple[int, int],
                    objectif: Tuple[int, int],
                    heuristique=heuristique_manhattan) -> Dict:
    """
    Algorithme A* avec f(n) = g(n) + h(n)

    Retourne:
        dict avec: chemin, cout, noeuds_explores, open_final, closed_final
    """
    open_set = []
    closed_set = set()

    # Initialisation
    noeud_depart = Noeud(
        f=heuristique(depart, objectif),
        g=0,
        position=depart
    )
    heapq.heappush(open_set, noeud_depart)

    open_dict = {depart: noeud_depart}
    came_from = {}
    g_score = {depart: 0}

    noeuds_explores = []

    while open_set:
        noeud_courant = heapq.heappop(open_set)
        pos_courante = noeud_courant.position

        # Vérifier si objectif atteint
        if pos_courante == objectif:
            chemin = reconstruire_chemin(came_from, pos_courante)
            return {
                'succes': True,
                'chemin': chemin,
                'cout': g_score[objectif],
                'noeuds_explores': noeuds_explores,
                'open_size': len(open_set),
                'closed_size': len(closed_set)
            }

        closed_set.add(pos_courante)
        noeuds_explores.append(pos_courante)

        # Explorer les voisins
        for voisin in grille.voisins(pos_courante):
            if voisin in closed_set:
                continue

            cout_temp = g_score[pos_courante] + grille.cout_deplacement(pos_courante, voisin)

            if voisin not in g_score or cout_temp < g_score[voisin]:
                came_from[voisin] = pos_courante
                g_score[voisin] = cout_temp
                f_score = cout_temp + heuristique(voisin, objectif)

                nouveau_noeud = Noeud(
                    f=f_score,
                    g=cout_temp,
                    position=voisin,
                    parent=noeud_courant
                )

                if voisin in open_dict:
                    open_dict[voisin] = nouveau_noeud
                    heapq.heappush(open_set, nouveau_noeud)
                else:
                    open_dict[voisin] = nouveau_noeud
                    heapq.heappush(open_set, nouveau_noeud)

    return {
        'succes': False,
        'chemin': [],
        'cout': float('inf'),
        'noeuds_explores': noeuds_explores,
        'open_size': len(open_set),
        'closed_size': len(closed_set)
    }


def recherche_ucs(grille: Grille,
                  depart: Tuple[int, int],
                  objectif: Tuple[int, int]) -> Dict:
    """
    Uniform Cost Search (UCS) - f(n) = g(n)
    Cas particulier de A* avec h(n) = 0
    """
    return recherche_astar(grille, depart, objectif,
                           heuristique=lambda p, g: 0)


def recherche_greedy(grille: Grille,
                     depart: Tuple[int, int],
                     objectif: Tuple[int, int]) -> Dict:
    """
    Greedy Best-First Search - f(n) = h(n)
    """
    return recherche_astar(grille, depart, objectif,
                           heuristique=heuristique_manhattan)


def reconstruire_chemin(came_from: Dict,
                        position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruit le chemin depuis l'objectif jusqu'au départ"""
    chemin = [position]
    while position in came_from:
        position = came_from[position]
        chemin.append(position)
    return chemin[::-1]


def generer_grille_aleatoire(taille: Tuple[int, int],
                             taux_obstacles: float = 0.2,
                             seed: int = None) -> Grille:
    """Génère une grille avec obstacles aléatoires"""
    if seed is not None:
        np.random.seed(seed)

    lignes, colonnes = taille
    obstacles = []

    for i in range(lignes):
        for j in range(colonnes):
            if np.random.random() < taux_obstacles:
                obstacles.append((i, j))

    return Grille(taille, obstacles)