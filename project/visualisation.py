"""
visualization.py - Diagrammes essentiels pour le rapport (titres en français)
Tous les fichiers sauvegardés dans: resultats/
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple
from matplotlib.patches import Circle

OUTPUT_DIR = "resultats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'

def save_figure(fig, filename: str, dpi: int = 300) -> None:
    """Sauvegarder et fermer la figure"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)





def plot_chemin_grille_clair(grid: Dict, paths: Dict[str, List[Tuple[int, int]]],
                              start: Tuple[int, int], goal: Tuple[int, int],
                              grid_name: str) -> None:
    """
    Diagramme CLAIR: 3 algorithmes superposés sur même grille
    - Chemins en couleurs distinctes avec marqueurs
    - START = cercle vert, GOAL = cercle rouge (bien visibles)
    - Légende explicite
    """
    rows, cols = grid['rows'], grid['cols']
    grid_map = np.zeros((rows, cols))

    # Obstacles en noir
    for (r, c) in grid['obstacles']:
        grid_map[r, c] = 1

    plt.figure(figsize=(11, 11))
    ax = sns.heatmap(grid_map, cmap='Greys', cbar=False, linewidths=0.8,
                    linecolor='gray', xticklabels=False, yticklabels=False)

    # Couleurs et styles pour chaque algorithme
    styles = {
        'astar': {'color': '#E63946', 'label': 'A*', 'marker': 'o', 'linestyle': '-'},
        'ucs': {'color': '#1D3557', 'label': 'UCS', 'marker': 's', 'linestyle': '--'},
        'greedy': {'color': '#2A9D8F', 'label': 'Greedy', 'marker': '^', 'linestyle': '-.'}
    }

    # Tracer chaque chemin avec style distinct
    for algo, path in paths.items():
        if path and algo in styles:
            style = styles[algo]
            path_y = [p[1] + 0.5 for p in path]
            path_x = [p[0] + 0.5 for p in path]
            plt.plot(path_y, path_x,
                    color=style['color'],
                    linewidth=2.5,
                    marker=style['marker'],
                    markersize=4,
                    markevery=2,  # Afficher un marqueur sur 2 pour lisibilité
                    linestyle=style['linestyle'],
                    label=f"Chemin {style['label']}",
                    markerfacecolor='white',
                    markeredgecolor=style['color'],
                    markeredgewidth=1.5)

    # START et GOAL bien visibles (cercles avec bordure blanche)
    start_circle = Circle((start[1] + 0.5, start[0] + 0.5), radius=0.45,
                         facecolor='limegreen', edgecolor='white',
                         linewidth=3, label='Départ (s₀)', zorder=10)
    goal_circle = Circle((goal[1] + 0.5, goal[0] + 0.5), radius=0.45,
                        facecolor='crimson', edgecolor='white',
                        linewidth=3, label='But (g)', zorder=10)
    ax.add_patch(start_circle)
    ax.add_patch(goal_circle)

    # Légende claire
    plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)

    plt.title(f'Comparaison des chemins - Grille {grid_name.capitalize()}',
              fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    save_figure(plt.gcf(), f"comparaison_chemins_{grid_name}.png")
    plt.show()




def plot_performance_par_difficulte(all_results: Dict[str, List[Dict]]) -> None:
    """
    4 sous-graphes SÉPARÉS : coût, nœuds développés, nœuds testés, temps
    Chaque métrique dans son propre subplot, couleurs cohérentes, grille en abscisse
    """
    difficulties = ['facile', 'moyenne', 'difficile']
    algorithms = ['ucs', 'greedy', 'astar']
    algo_labels = {'ucs': 'UCS', 'greedy': 'Greedy', 'astar': 'A*'}
    algo_colors = {'ucs': '#1D3557', 'greedy': '#2A9D8F', 'astar': '#E63946'}

    metrics = [
        ('cost', 'Coût du chemin'),
        ('nodes_expanded', 'Nœuds développés'),
        ('max_open_size', 'Nœuds testés (OPEN max)'),
        ('execution_time', "Temps d'exécution (ms)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    fig.suptitle('Performance des algorithmes selon la difficulté',
                 fontsize=17, fontweight='bold', y=1.01)

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        data_list = []

        for diff in difficulties:
            for res in all_results.get(diff, []):
                val = res.get(metric, 0)
                if metric == 'execution_time':
                    val *= 1000  # ms
                data_list.append({
                    'Grille': diff.capitalize(),
                    'Algorithme': algo_labels[res['algorithm']],
                    'Valeur': val,
                    'algo_key': res['algorithm']
                })

        df = pd.DataFrame(data_list)
        if df.empty:
            continue

        # Barres groupées par difficulté, couleurs par algorithme
        x = np.arange(len(difficulties))
        width = 0.25

        for i, algo in enumerate(algorithms):
            subset = df[df['algo_key'] == algo]
            values = [subset[subset['Grille'] == d.capitalize()]['Valeur'].values[0]
                     if not subset[subset['Grille'] == d.capitalize()].empty else 0
                     for d in difficulties]
            ax.bar(x + i*width, values, width,
                  label=algo_labels[algo],
                  color=algo_colors[algo],
                  edgecolor='black', linewidth=0.8)

        ax.set_xlabel('Difficulté', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.legend(title='Algorithme', fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Ajouter les valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f' if metric != 'execution_time' else '%.1f',
                        fontsize=8, padding=2)

    plt.tight_layout()
    save_figure(fig, "performance_par_difficulte.png")
    plt.show()

def plot_probabilites_absorption_essentiel(goal_probs: List[float], fail_probs: List[float],
                                          epsilons: List[float], grid_name: str) -> None:
    """
    DIAGRAMME ESSENTIEL 1/2 : Probabilités d'atteindre GOAL ou FAIL selon ε
    Clair, lisible, directement utilisable dans le rapport
    """
    plt.figure(figsize=(11, 7))

    plt.plot(epsilons, goal_probs, marker='o', markersize=9, linewidth=3, color='#2A9D8F',
            label='P(atteindre GOAL)', markerfacecolor='white', markeredgecolor='#2A9D8F', markeredgewidth=2)
    plt.plot(epsilons, fail_probs, marker='s', markersize=9, linewidth=3, color='#E63946',
            label="P(échec → FAIL)", markerfacecolor='white', markeredgecolor='#E63946', markeredgewidth=2)

    plt.xlabel("Taux d'incertitude ε", fontsize=13, fontweight='bold')
    plt.ylabel('Probabilité', fontsize=13, fontweight='bold')
    plt.title(f'Robustesse du plan face à l\'incertitude\nGrille {grid_name.capitalize()}',
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.xticks(epsilons, [f'{e:.1f}' for e in epsilons], fontsize=11)
    plt.yticks(fontsize=11)

    # Zone de sécurité visuelle
    plt.axhline(y=0.9, color='green', linestyle=':', alpha=0.3, label='_nolegend_')
    plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.3, label='_nolegend_')

    plt.tight_layout()
    save_figure(plt.gcf(), f"probabilites_absorption_{grid_name}.png")
    plt.show()

def plot_evolution_distribution_essentiel(history: np.ndarray, goal_idx: int, fail_idx: int,
                                         grid_name: str, epsilon: float, max_steps: int = 40) -> None:
    """
    DIAGRAMME ESSENTIEL 2/2 : Évolution de π(n) vers absorption
    Montre clairement la convergence vers GOAL ou FAIL
    """
    steps = min(history.shape[0], max_steps)
    t_range = list(range(steps))

    p_goal = history[:steps, goal_idx]
    p_fail = history[:steps, fail_idx]

    plt.figure(figsize=(11, 7))

    # Courbes avec remplissage pour lisibilité
    plt.plot(t_range, p_goal, label='P(être dans GOAL)', color='#2A9D8F', linewidth=3, marker='o', markersize=4)
    plt.fill_between(t_range, p_goal, alpha=0.25, color='#2A9D8F')

    plt.plot(t_range, p_fail, label="P(être dans FAIL)", color='#E63946', linewidth=3, marker='s', markersize=4)
    plt.fill_between(t_range, p_fail, alpha=0.25, color='#E63946')

    plt.xlabel('Itération n (étapes)', fontsize=13, fontweight='bold')
    plt.ylabel('Probabilité πₙ(état)', fontsize=13, fontweight='bold')
    plt.title(f'Convergence vers l\'absorption\nGrille {grid_name.capitalize()}, ε={epsilon}',
              fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Ligne de convergence visuelle
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, label='_nolegend_')

    plt.tight_layout()
    save_figure(plt.gcf(), f"evolution_distribution_{grid_name}_eps{epsilon}.png")
    plt.show()

def export_tableau_resultats(all_results: Dict, markov_results: Dict,
                            filename: str = "tableau_resultats.csv") -> None:
    """
    Exporter TOUS les résultats dans un CSV formaté pour le rapport
    Séparateur ';', décimale ',' pour compatibilité Excel/LibreOffice
    """
    rows = []

    # Résultats algorithmes (E.1)
    for grid_name, results in all_results.items():
        for res in results:
            if not res.get('success', False):
                continue
            rows.append({
                'Expérience': 'E.1 - Comparaison algorithmes',
                'Grille': grid_name.capitalize(),
                'Algorithme': res['algorithm'].upper(),
                'Coût du chemin': res.get('cost', 'N/A'),
                'Nœuds développés': res.get('nodes_expanded', 'N/A'),
                'Nœuds testés (OPEN)': res.get('max_open_size', 'N/A'),
                "Temps d'exécution (ms)": f"{res.get('execution_time', 0)*1000:.2f}",
                'Succès': 'Oui'
            })

    # Résultats Markov (E.2)
    for grid_name, data in markov_results.items():
        for eps, metrics in data.items():
            rows.append({
                'Expérience': 'E.2 - Impact de ε (Markov)',
                'Grille': grid_name.capitalize(),
                'Algorithme': 'A* + Markov',
                'ε': f"{eps:.1f}",
                'P(GOAL) théorique': f"{metrics.get('prob_goal_theoretical', 0):.4f}",
                'P(GOAL) simulation': f"{metrics.get('prob_goal_empirical', 0):.4f}",
                'P(FAIL)': f"{metrics.get('prob_fail', 0):.4f}",
                'Écart théorie/simulation': f"{abs(metrics.get('prob_goal_theoretical', 0) - metrics.get('prob_goal_empirical', 0)):.4f}",
                'Succès': 'Oui'
            })

    df = pd.DataFrame(rows)
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print(f"✓ Tableau exporté : {filepath}")