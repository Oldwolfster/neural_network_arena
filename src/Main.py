
import time
from src.ArenaSettings import *
#from src.engine.Engine import run_a_match, run_batch_of_matches, run_all_matchups
from engine.SQL import list_runs
import datetime

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
        start_time                  = time.time()

        start_str = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        if test_attribute:
            lister = LegoLister()
            test_legos = lister.list_legos(test_attribute)
            #print("Testing lookup:", list(test_legos.keys()))
            for name, strategy in test_legos.items():
                print(f"\n➤ Batch Mode on   {test_attribute} = {name}")
                run_a_set(shared_hyper, neuro_engine, test_attribute, strategy)
            end_time = time.time()
            elapsed = end_time - start_time
            end_str = datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n✅ Attribute testing completed")
            print(f"   ├ Attribute : {test_attribute}")
            print(f"   ├ Start     : {start_str}")
            print(f"   ├ End       : {end_str}")
            print(f"   └ Duration  : {elapsed:.2f} seconds")
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