"""
main.py - Project Entry Point
"""

from experiments import run_all_experiments

if __name__ == "__main__":
    print("Starting Mini-Projet: Robust Planning on Grid (A* + Markov)")
    print("="*60)
    try:
        run_all_experiments()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()