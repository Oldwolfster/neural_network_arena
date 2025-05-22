import snakeviz

from src.ArenaSettings import *
#from src.engine.Engine import run_a_match, run_batch_of_matches, run_all_matchups
from engine.SQL import list_runs


import cProfile

from src.engine.StoreHistory import list_snapshots
from src.engine.NeuroEngine import NeuroEngine

def main():
    shared_hyper = HyperParameters()
    neuro_engine = NeuroEngine(shared_hyper)
    results = []

    if single_match:
        result = neuro_engine.run_a_match(gladiators, training_pit)
        print(f"Result of single match ({gladiators[0]}): {result[0]:.2f}%")
    else:
        shared_hyper.record = False
        arena_scores = []

        for arena in batch_arenas:
            print(f"Running arena: {arena}")
            relative_errors = neuro_engine.run_a_match(gladiators, arena)

            for gladiator_name, error in zip(gladiators, relative_errors):
                label = f"{arena} / {gladiator_name}"
                arena_scores.append((label, error))

        print("\n--- Relative MAE by Arena and Gladiator ---")
        for name, error in arena_scores:
            print(f"{name:<50} {error:.2f}%")

        if arena_scores:
            average = sum(score for _, score in arena_scores) / len(arena_scores)
            print(f"\nAverage Relative MAE across all runs: {average:.2f}%")

        results.extend(arena_scores)

def mainOrig():
    #if instead_of_run_show_past_runs and len(run_previous_training_data) == 0:
    shared_hyper = HyperParameters()
    neuro_engine = NeuroEngine(shared_hyper)
    results = []
    if single_match:
        results = neuro_engine.run_a_match(gladiators, training_pit)
        print(f"results of single={results}")
    else:
        shared_hyper.record = False
        for arena in batch_arenas:
            print(f"running arena: {arena}")
            results.extend( neuro_engine.run_a_match(gladiators, arena))
    print(f"results of batch={results}")
    print(f"")


if __name__ == '__main__':
    main() #Normal run

    #cProfile.run('main()', 'profile_stats.prof')
    # CMD LINE RUN snakeviz src\profile_stats.prof