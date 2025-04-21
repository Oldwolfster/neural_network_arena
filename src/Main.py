import snakeviz

from src.ArenaSettings import *
from src.engine.Engine import run_a_match, run_batch_of_matches, run_all_matchups
from engine.SQL import list_runs


import cProfile

from src.engine.StoreHistory import list_snapshots
from src.engine.TurboForge import NeuroEngine


def main():
    #if instead_of_run_show_past_runs and len(run_previous_training_data) == 0:
    shared_hyper = HyperParameters()
    if history_to_show > 0:
        list_snapshots(history_to_show)
        #list_runs()
    else:
        #run_batch_of_matches(gladiators, training_pit)        #
        #run_a_match(gladiators, training_pit, shared_hyper)
        neuro_engine = NeuroEngine()
        neuro_engine.run_a_match(gladiators)
        #run_batch_of_matches(gladiators, training_pit, shared_hyper)
        #run_all_matchups(training_pit,shared_hyper)
        #from src.backprop.tutorial.nn_xor import run_tutorial
        #run_tutorial()


if __name__ == '__main__':
    main() #Normal run

    #cProfile.run('main()', 'profile_stats.prof')
    # CMD LINE RUN snakeviz src\profile_stats.prof