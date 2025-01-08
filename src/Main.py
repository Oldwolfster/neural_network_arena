from src.ArenaSettings import *
from src.engine.Engine import run_a_match
from engine.SQL import list_runs


def main():
    if instead_of_run_show_past_runs and len(run_previous_training_data) == 0:
        list_runs()
    else:
        #run_a_match(gladiators, training_pit)
        from src.backprop.tutorial.nn_xor import run_tutorial
        run_tutorial()


if __name__ == '__main__':
    main()