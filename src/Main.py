import snakeviz

from src.ArenaSettings import *
#from src.engine.Engine import run_a_match, run_batch_of_matches, run_all_matchups
from engine.SQL import list_runs


import cProfile

from src.Legos.LegoLister import LegoLister
from src.engine.StoreHistory import list_snapshots
from src.engine.NeuroEngine import NeuroEngine


def main():
    shared_hyper = HyperParameters()
    neuro_engine = NeuroEngine(shared_hyper)

    if single_match:
        result = neuro_engine.run_a_match(gladiators, training_pit)
        print(f"Result of single match ({gladiators[0]}): {result[0]:.2f}%")
    else:
        print(f"test_attribute={test_attribute}")
        if test_attribute:
            lister = LegoLister()
            test_legos = lister.list_legos(test_attribute)
            #print("Testing lookup:", list(test_legos.keys()))
            for name, strategy in test_legos.items():
                print(f"\n➤ Running with {test_attribute} = {name}")
                run_a_set(shared_hyper, neuro_engine, test_attribute, strategy)
        else:
            run_a_set(shared_hyper, neuro_engine)

def run_a_set(shared_hyper, neuro_engine, test_attribute=None, test_strategy=None):
    results = []
    shared_hyper.record = False
    arena_scores = []

    for arena in batch_arenas:
        print(f"Running arena: {arena}")
        relative_errors = neuro_engine.run_a_match(gladiators, arena, test_attribute, test_strategy)

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






if __name__ == '__main__':
    main() #Normal run

    #cProfile.run('main()', 'profile_stats.prof')
    # CMD LINE RUN snakeviz src\profile_stats.prof