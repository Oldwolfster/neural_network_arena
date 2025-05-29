from datetime import datetime
from typing import List
from typing import Dict
from typing import Any
from itertools import product
import os

# test = lister.list_legos("hidden_activation")
# test = lister.list_legos("output_activation")
# test = lister.list_legos("optimizer")
# test = lister.list_legos("scaler") # works a bit different....
# test = lister.list_legos("initializer")

class TrainingBatchInfo:
    def __init__(self, gladiators, arenas, dimensions: Dict[str, List[Any]]):
        self.gladiators = gladiators
        self.arenas = arenas
        self.dimensions = dimensions
        self.setups = []
        self.build_run_instructions()
        self.remove_lists_from_setups()

    def remove_lists_from_setups(self):
        for setup in self.setups:
            for key, value in setup.items():
                if isinstance(value, list) and len(value) == 1:
                    setup[key] = value[0]



    def build_run_instructions(self):
        config_keys = list(self.dimensions.keys())
        config_values = list(self.dimensions.values())
        combos = list(product(*config_values))

        for gladiator in self.gladiators:
            lr_flag = self.model_explicitly_sets_lr(gladiator)
            for arena in self.arenas:
                for combo in combos:
                    config_dict = dict(zip(config_keys, combo))
                    config_dict["gladiator"] = gladiator
                    config_dict["arena"] = arena
                    config_dict["lr_specified"] = lr_flag
                    self.setups.append(config_dict)
    def one_option_for_get_unique_run_id():
        now = datetime.now()
        return int(now.strftime("%Y%m%d%H%M%S%f"))  # microsecond precision

    def model_explicitly_sets_lr(self, gladiator_name: str) -> bool:
        for root, _, files in os.walk("coliseum/gladiators"):
            for file in files:
                if file == f"{gladiator_name}.py":
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        for line in f:
                            if "config.learning_rate" in line and not line.strip().startswith("#"):
                                return True
                    return False
        raise FileNotFoundError(f"❌ Could not find file for gladiator '{gladiator_name}' (expected '{gladiator_name}.py') in 'coliseum/gladiators'.")
