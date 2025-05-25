from datetime import datetime
from typing import List
from typing import Dict
from typing import Any
from itertools import product


# test = lister.list_legos("hidden_activation")
# test = lister.list_legos("output_activation")
# test = lister.list_legos("optimizer")
# test = lister.list_legos("scaler") # works a bit different....
# test = lister.list_legos("initializer")

class TrainingBatchInfo:
    def __init__(self, gladiators, arenas, dimensions: Dict[str, List[Any]] = None):
        self.gladiators = gladiators
        self.arenas = arenas
        self.dimensions = dimensions or {"__noop__": [None]}  # placeholder
        self.ATAMs = []
        self.build_run_instructions()

    def build_run_instructions(self):
        config_keys = list(self.dimensions.keys())
        config_values = list(self.dimensions.values())

        combos = list(product(*config_values))

        for gladiator in self.gladiators:
            for arena in self.arenas:
                for combo in combos:
                    config_dict = dict(zip(config_keys, combo))
                    config_dict["gladiator"] = gladiator
                    config_dict["arena"] = arena
                    self.ATAMs.append(config_dict)


    def get_unique_run_id():
        now = datetime.now()
        return int(now.strftime("%Y%m%d%H%M%S%f"))  # microsecond precision
