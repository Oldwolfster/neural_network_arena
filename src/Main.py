
import time
from src.ArenaSettings import *
#from src.engine.Engine import run_a_match, run_batch_of_matches, run_all_matchups
from engine.SQL import list_runs
import datetime

import cProfile

from src.Legos.LegoLister import LegoLister

from src.engine.NeuroEngine import NeuroEngine


def main():
    shared_hyper = HyperParameters()
    neuro_engine = NeuroEngine(shared_hyper)
    neuro_engine.run_a_batch()


def main2():
    test_attribute ="initializer"
    lister = LegoLister()
    test_legos = lister.list_legos(test_attribute)
    #print("Testing lookup:", list(test_legos.keys()))
    for name, strategy in test_legos.items():
        print(f"\n➤ Found  {test_attribute} = {name}\tStrategy = {strategy}")




if __name__ == '__main__':
    main() #Normal run

    #cProfile.run('main()', 'profile_stats.prof')
    # CMD LINE RUN snakeviz src\profile_stats.prof