"""
experiments.py - Exécution des expériences avec diagrammes ESSENTIELS uniquement
"""

import numpy as np
from tools import generate_test_case
from astar import run_search, heuristic_manhattan, extract_policy
from markov import build_transition_matrix, compute_state_evolution, simulate_trajectories
from visualisation import (
    plot_chemin_grille_clair,
    plot_performance_par_difficulte,
    plot_probabilites_absorption_essentiel,
    plot_evolution_distribution_essentiel,
    export_tableau_resultats
)

def run_experiment_1_comparaison_algorithmes():
    """E.1 : Comparer UCS vs Greedy vs A* sur 3 grilles"""
    print("=== Expérience E.1 : Comparaison des algorithmes ===")
    difficulties = ['facile', 'moyenne', 'difficile']
    algorithms = ['ucs', 'greedy', 'astar']

    all_results = {}
    paths_for_plot = {}

    for diff in difficulties:
        print(f"\n--- Grille : {diff.upper()} ---")
        case = generate_test_case(diff)
        grid, start, goal = case['grid'], case['start'], case['goal']

        results_diff = []
        paths_diff = {}

        for algo in algorithms:
            print(f"Exécution de {algo.upper()}...")
            res = run_search(start, goal, grid, algorithm=algo, heuristic=heuristic_manhattan)

            if res and res.get('success'):
                result_entry = {
                    'algorithm': algo,
                    'cost': res['cost'],
                    'nodes_expanded': res['nodes_expanded'],
                    'max_open_size': res.get('max_open_size', 0),
                    'execution_time': res.get('execution_time', 0),
                    'success': True,
                    'path': res['path']
                }
                results_diff.append(result_entry)
                paths_diff[algo] = res['path']
                print(f"  ✓ {algo.upper()}: Coût={res['cost']}, Développés={res['nodes_expanded']}, "
                      f"Testés={res.get('max_open_size', 0)}, Temps={res.get('execution_time', 0)*1000:.1f}ms")
            else:
                print(f"  ✗ {algo.upper()}: ÉCHEC")
                results_diff.append({'algorithm': algo, 'success': False})

        all_results[diff] = results_diff
        paths_for_plot[diff] = paths_diff

        # Diagramme CLAIR des chemins superposés
        if paths_diff:
            plot_chemin_grille_clair(grid, paths_diff, start, goal, diff)

    # Diagramme de performance (4 métriques séparées)
    plot_performance_par_difficulte(all_results)

    return all_results

def run_experiment_2_markov_essentiel():
    """
    E.2 : Diagrammes Markov ESSENTIELS pour le rapport (seulement 2)
    1. Probabilités d'absorption vs ε
    2. Évolution de π(n) vers absorption
    """
    print("\n=== Expérience E.2 : Diagrammes Markov essentiels ===")
    grid_name = 'moyenne'
    case = generate_test_case(grid_name)
    grid, start, goal = case['grid'], case['start'], case['goal']

    # Chemin de référence avec A*
    res_ref = run_search(start, goal, grid, algorithm='astar', heuristic=heuristic_manhattan)
    if not res_ref or not res_ref['success']:
        print("ÉCHEC : A* n'a pas trouvé de chemin.")
        return {}

    policy = extract_policy(res_ref['path'])
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.4]

    markov_results = {grid_name: {}}
    goal_probs, fail_probs = [], []

    # Préparer indices Markov
    P_ref, state_list = build_transition_matrix(policy, grid, epsilon=0.1, goal_state=goal)
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    start_idx = state_to_idx[start]
    goal_idx = state_to_idx[goal]
    fail_idx = len(state_list) - 1

    for eps in epsilons:
        P, _ = build_transition_matrix(policy, grid, epsilon=eps, goal_state=goal)

        # Calcul théorique π(n) = π(0) P^n
        pi0 = np.zeros(P.shape[0])
        pi0[start_idx] = 1.0
        history = compute_state_evolution(pi0, P, steps=100)
        prob_goal = history[-1, goal_idx]
        prob_fail = history[-1, fail_idx]

        # Simulation Monte-Carlo pour validation
        sim_res = simulate_trajectories(P, start_idx, n_trials=1000, max_steps=100,
                                       goal_idx=goal_idx, fail_idx=fail_idx)

        goal_probs.append(prob_goal)
        fail_probs.append(prob_fail)

        markov_results[grid_name][eps] = {
            'prob_goal_theoretical': prob_goal,
            'prob_goal_empirical': sim_res['prob_goal_empirical'],
            'prob_fail': prob_fail
        }

        print(f"ε={eps:.1f} → P(GOAL) théorique={prob_goal:.4f}, simulation={sim_res['prob_goal_empirical']:.4f}")

    # DIAGRAMME ESSENTIEL 1/2 : Probabilités d'absorption vs ε
    plot_probabilites_absorption_essentiel(goal_probs, fail_probs, epsilons, grid_name)

    # DIAGRAMME ESSENTIEL 2/2 : Évolution de π(n) pour ε=0.2 (valeur intermédiaire représentative)
    P_eps2, _ = build_transition_matrix(policy, grid, epsilon=0.2, goal_state=goal)
    pi0 = np.zeros(P_eps2.shape[0])
    pi0[start_idx] = 1.0
    history_eps2 = compute_state_evolution(pi0, P_eps2, steps=100)
    plot_evolution_distribution_essentiel(history_eps2, goal_idx, fail_idx, grid_name, epsilon=0.2)

    return markov_results

def run_all_experiments():
    """Exécuter toutes les expériences et exporter les résultats"""
    print("Mini-Projet : Planification robuste sur grille (A* + Chaînes de Markov)")
    print("Date : 3 mars 2026")
    print("="*70)

    # E.1 : Comparaison algorithmes (3 grilles)
    all_results = run_experiment_1_comparaison_algorithmes()

    # E.2 : Diagrammes Markov essentiels
    markov_results = run_experiment_2_markov_essentiel()

    # Export CSV pour le rapport
    export_tableau_resultats(all_results, markov_results)

    print("\n" + "="*70)
    print("✓ Projet terminé avec succès.")
    print(f"✓ Tous les diagrammes sont dans le dossier :'")
    print("✓ Fichier CSV prêt pour le rapport : 'tableau_resultats.csv'")
    print("\nDiagrammes générés (prêts pour le rapport 6-10 pages) :")
    print("  📊 comparaison_chemins_*.png : Chemins A*/UCS/Greedy superposés (clair)")
    print("  📊 performance_par_difficulte.png : 4 métriques séparées (coût inclus)")
    print("  📊 probabilites_absorption_moyenne.png : Robustesse vs ε (ESSENTIEL)")
    print("  📊 evolution_distribution_moyenne_eps0.2.png : Convergence Markov (ESSENTIEL)")
    print("  📄 tableau_resultats.csv : Toutes les données pour tableaux du rapport")