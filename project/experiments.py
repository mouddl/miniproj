"""
experiments.py - Exécution des expériences COMPLÈTES (conforme miniproj.pdf)
"""
import numpy as np
from tools import generate_test_case
from astar import run_search, heuristic_manhattan, heuristic_zero, extract_policy
from markov import (
    build_transition_matrix,
    compute_state_evolution,
    simulate_trajectories,
    calculate_absorption_metrics
)
from visualisation import (
    plot_chemin_grille_clair,
    plot_performance_par_difficulte,
    plot_probabilites_absorption_essentiel,
    plot_evolution_distribution_essentiel,
    plot_temps_absorption_vs_epsilon,
    plot_comparaison_heuristiques,
    export_tableau_resultats
)

OUTPUT_DIR = "resultats"

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
            print(f"Exécution de {algo.upper()}... ")
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
                print(f"  ✓ {algo.upper()}: Coût={res['cost']}, Développés={res['nodes_expanded']},  "
                      f"Testés={res.get('max_open_size', 0)}, Temps={res.get('execution_time', 0)*1000:.1f}ms ")
            else:
                print(f"  ✗ {algo.upper()}: ÉCHEC ")
                results_diff.append({'algorithm': algo, 'success': False})

        all_results[diff] = results_diff
        paths_for_plot[diff] = paths_diff

        if paths_diff:
            plot_chemin_grille_clair(grid, paths_diff, start, goal, diff)

    plot_performance_par_difficulte(all_results)
    return all_results


def run_experiment_3_comparaison_heuristiques():
    """E.3 : Comparer deux heuristiques admissibles (h=0 vs Manhattan)"""
    print("\n=== Expérience E.3 : Comparaison des heuristiques ===")
    grid_name = 'moyenne'
    case = generate_test_case(grid_name)
    grid, start, goal = case['grid'], case['start'], case['goal']

    heuristics = [
        ('heuristic_zero (UCS)', heuristic_zero),
        ('heuristic_manhattan (A*)', heuristic_manhattan)
    ]

    results = []
    for h_name, h_func in heuristics:
        print(f"Exécution avec {h_name}... ")
        res = run_search(start, goal, grid, algorithm='astar', heuristic=h_func)
        if res and res.get('success'):
            results.append({
                'heuristique': h_name,
                'cost': res['cost'],
                'nodes_expanded': res['nodes_expanded'],
                'execution_time': res.get('execution_time', 0)
            })
            print(f"  ✓ {h_name}: Coût={res['cost']}, Développés={res['nodes_expanded']}")

    plot_comparaison_heuristiques(results, grid_name)
    return results


def run_experiment_2_markov_essentiel():
    """E.2 : Analyse Markov complète avec fallback simulation"""
    print("\n=== Expérience E.2 : Analyse Markov complète ===")
    grid_name = 'moyenne'
    case = generate_test_case(grid_name)
    grid, start, goal = case['grid'], case['start'], case['goal']

    res_ref = run_search(start, goal, grid, algorithm='astar', heuristic=heuristic_manhattan)
    if not res_ref or not res_ref['success']:
        print("ÉCHEC : A* n'a pas trouvé de chemin.")
        return {}

    policy = extract_policy(res_ref['path'])
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.4]

    markov_results = {grid_name: {}}
    goal_probs, fail_probs = [], []
    mean_times = []

    P_ref, state_list = build_transition_matrix(policy, grid, epsilon=0.1, goal_state=goal)
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    start_idx = state_to_idx[start]
    goal_idx = state_to_idx[goal]
    fail_idx = len(state_list) - 1

    for eps in epsilons:
        P, _ = build_transition_matrix(policy, grid, epsilon=eps, goal_state=goal)

        pi0 = np.zeros(P.shape[0])
        pi0[start_idx] = 1.0
        history = compute_state_evolution(pi0, P, steps=100)
        prob_goal = history[-1, goal_idx]
        prob_fail = history[-1, fail_idx]

        absorption_metrics = calculate_absorption_metrics(P, goal_idx, fail_idx)
        mean_time = 0.0

        if 'error' not in absorption_metrics and 'mean_time_vector' in absorption_metrics:
            transient_indices = absorption_metrics.get('transient_indices', [])
            if start_idx in transient_indices:
                local_idx = transient_indices.index(start_idx)
                candidate_time = absorption_metrics['mean_time_vector'][local_idx]
                if candidate_time > 0 and np.isfinite(candidate_time):
                    mean_time = candidate_time
                else:
                    sim_res = simulate_trajectories(P, start_idx, n_trials=1000, max_steps=100,
                                                   goal_idx=goal_idx, fail_idx=fail_idx)
                    mean_time = sim_res['avg_steps']
            else:
                sim_res = simulate_trajectories(P, start_idx, n_trials=1000, max_steps=100,
                                               goal_idx=goal_idx, fail_idx=fail_idx)
                mean_time = 0.0 if start_idx == goal_idx else sim_res['avg_steps']
        else:
            sim_res = simulate_trajectories(P, start_idx, n_trials=1000, max_steps=100,
                                           goal_idx=goal_idx, fail_idx=fail_idx)
            mean_time = sim_res['avg_steps']

        mean_times.append(mean_time)

        sim_res = simulate_trajectories(P, start_idx, n_trials=1000, max_steps=100,
                                       goal_idx=goal_idx, fail_idx=fail_idx)

        goal_probs.append(prob_goal)
        fail_probs.append(prob_fail)

        markov_results[grid_name][eps] = {
            'prob_goal_theoretical': prob_goal,
            'prob_goal_empirical': sim_res['prob_goal_empirical'],
            'prob_fail': prob_fail,
            'mean_time_absorption': mean_time
        }

        print(f"ε={eps:.1f} → P(GOAL)={prob_goal:.4f}, Temps moyen={mean_time:.2f} étapes")

    plot_probabilites_absorption_essentiel(goal_probs, fail_probs, epsilons, grid_name)
    plot_temps_absorption_vs_epsilon(mean_times, epsilons, grid_name)

    P_eps2, _ = build_transition_matrix(policy, grid, epsilon=0.2, goal_state=goal)
    pi0 = np.zeros(P_eps2.shape[0])
    pi0[start_idx] = 1.0
    history_eps2 = compute_state_evolution(pi0, P_eps2, steps=100)
    plot_evolution_distribution_essentiel(history_eps2, goal_idx, fail_idx, grid_name, epsilon=0.2)

    return markov_results


def run_all_experiments():
    """Exécuter TOUTES les expériences (E.1, E.2, E.3)"""
    print("Mini-Projet : Planification robuste sur grille (A* + Chaînes de Markov)")
    print("Date : 3 mars 2026")
    print("="*70)

    all_results = run_experiment_1_comparaison_algorithmes()
    heuristic_results = run_experiment_3_comparaison_heuristiques()
    markov_results = run_experiment_2_markov_essentiel()

    export_tableau_resultats(all_results, markov_results, heuristic_results)

    print("\n" + "="*70)
    print("✓ Projet terminé avec succès. ")
    print(f"✓ Tous les diagrammes sont dans le dossier : '{OUTPUT_DIR}/' ")
    print(f"✓ Fichier CSV prêt pour le rapport : '{OUTPUT_DIR}/tableau_resultats.csv' ")
    print("\nDiagrammes générés (prêts pour le rapport 6-10 pages) : ")
    print("  📊 comparaison_chemins_*.png")
    print("  📊 performance_par_difficulte.png")
    print("  📊 comparaison_heuristiques.png")
    print("  📊 probabilites_absorption_*.png")
    print("  📊 temps_absorption_*.png")
    print("  📊 evolution_distribution_*.png")
    print("  📄 tableau_resultats.csv")