import os
import sys

# Add the parent directory of this file to sys.path - Makes compatible with python inside unity.
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


from src.ArenaSettings import *
from src.NNA.Legos.LegoLister import LegoLister
from src.NNA.engine.NeuroEngine import NeuroEngine


def main():
    shared_hyper = HyperParameters()
    neuro_engine = NeuroEngine(shared_hyper)
    neuro_engine.run_a_batch()
    print("end")


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