import snakeviz

from src.ArenaSettings import *
#from src.engine.Engine import run_a_match, run_batch_of_matches, run_all_matchups
from engine.SQL import list_runs


import cProfile

from src.engine.StoreHistory import list_snapshots
from src.engine.NeuroEngine import NeuroEngine


def main():
    #if instead_of_run_show_past_runs and len(run_previous_training_data) == 0:
    shared_hyper = HyperParameters()
    if single_match:

        neuro_engine = NeuroEngine()
        neuro_engine.run_a_match(gladiators, training_pit)
    else:
        pass


if __name__ == '__main__':
    main() #Normal run

    #cProfile.run('main()', 'profile_stats.prof')
    # CMD LINE RUN snakeviz src\profile_stats.prof