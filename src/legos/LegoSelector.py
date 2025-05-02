from dataclasses import asdict
from typing import Any


class LegoSelector:
    def __init__(self):
        self.applied_rules = []

    def apply(self, config, rules):
        # track which rule‐indices we’ve applied (and to what value)
        seen: dict[int, Any] = {}
        # keep running until no rule makes an actual change
        while self._apply_for_real(config, rules, seen):
            pass

        #print(self.pretty_print_applied_rules())
        # if you ever re‐use this selector again, you can clear for next time
        seen.clear()

    def _apply_for_real(self, config, rules, seen: dict[int, Any]) -> bool:
        """
        Returns True if it applied one rule (and mutated config).
        Uses 'seen' to skip any rule whose action wouldn't actually change anything.
        """
        # sort by priority
        sorted_rules = sorted(rules, key=lambda rule: rule[1])

        for idx, (override, priority, action, condition) in enumerate(sorted_rules):
            try:
                if eval(condition, {}, config.__dict__):
                    for key, value in action.items():
                        current = getattr(config, key)
                        # if we've already set this rule to the same value, skip it
                        if seen.get(idx) == value:
                            continue
                        # only apply if:
                        #  - override=True and value is actually different, or
                        #  - override=False and the config[key] is still None
                        should_apply = (
                            (override and current != value) or
                            (not override and current is None)
                        )
                        if should_apply:
                            setattr(config, key, value)
                            self.applied_rules.append({
                                "priority": priority,
                                "setting": key,
                                "value": value,
                                "condition": condition
                            })
                            # remember that this rule idx set this exact value
                            seen[idx] = value
                            print(f"  ➤ Rule applied: {priority}: {key} = {value} from condition: '{condition}'")
                            return True
            except Exception as e:
                print(f"⚠️ Rule failed: {priority} {condition} -> {e}")

        return False

"""
***************** BELOW HERE IS THE OLD UNUSED ORIG *****************
***************** BELOW HERE IS THE OLD UNUSED ORIG *****************
***************** BELOW HERE IS THE OLD UNUSED ORIG *****************
***************** BELOW HERE IS THE OLD UNUSED ORIG *****************

class LegoSelector:
    def __init__(self, rules):
        self.rules = rules  # List of rule dicts

    def apply(self, config):
        changed = True
        applied_rules = set()

        while changed:
            changed = False
            for rule in self.rules:
                key = rule['setting']
                condition = rule.get('condition', 'True')
                value = rule['value']
                rule_id = rule.get('id')

                if key in config and config[key] is not None:
                    continue  # Skip already set values

                context = {**config}
                try:
                    if eval(condition, {}, context):
                        config[key] = value
                        changed = True
                        applied_rules.add(rule_id)
                except Exception as e:
                    print(f"⚠️ Condition eval failed for rule {rule_id}: {condition} -> {e}")

        return config, applied_rules


# Example rules for auto-configuration
rules = [
    {"id": 1, "setting": "loss_function", "value": "Loss_MAE", "condition": "problem_type == 'Regression'"},
    {"id": 2, "setting": "loss_function", "value": "Loss_BCEWithLogits", "condition": "problem_type == 'Binary' and threshold is None"},
    {"id": 3, "setting": "threshold", "value": 0.5, "condition": "loss_function == 'Loss_BCEWithLogits'"},
    {"id": 4, "setting": "output_activation", "value": "Activation_Linear", "condition": "loss_function == 'Loss_MAE'"},
    {"id": 5, "setting": "output_activation", "value": "Activation_Sigmoid", "condition": "loss_function == 'Loss_BCEWithLogits'"},
    {"id": 6, "setting": "learning_rate", "value": "auto_from_optimizer", "condition": "learning_rate is None and optimizer == 'Simplex'"},

    # 🧱 Architecture rules added
    {"id": 7, "setting": "architecture", "value": [4], "condition": "training_data.problem_type == 'Binary' and training_data.feature_count <= 3"},
    {"id": 8, "setting": "architecture", "value": [8, 4], "condition": "training_data.problem_type == 'Binary' and 3 < training_data.feature_count <= 10"},
    {"id": 9, "setting": "architecture", "value": [16, 8, 4], "condition": "training_data.problem_type == 'Binary' and training_data.feature_count > 10"},
    {"id": 10, "setting": "architecture", "value": [3], "condition": "training_data.problem_type == 'Regression' and training_data.feature_count <= 2"},
    {"id": 11, "setting": "architecture", "value": [4, 4], "condition": "training_data.problem_type == 'Regression' and 2 < training_data.feature_count <= 5"},
    {"id": 12, "setting": "architecture", "value": [8, 4], "condition": "training_data.problem_type == 'Regression' and training_data.feature_count > 5"},
    {"id": 13, "setting": "architecture", "value": [4], "condition": "True"},  # Fallback
]


# Example usage
# config = {
#     "problem_type": "Regression",
#     "threshold": None,
#     "optimizer": "Simplex",
#     "learning_rate": None,
#     "loss_function": None,
#     "output_activation": None,
# }

# selector = LegoSelector(rules)
# final_config, applied = selector.apply(config)
# print("Final Config:", final_config)
# print("Applied Rules:", applied)

"""