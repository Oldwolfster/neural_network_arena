from dataclasses import asdict

class LegoSelector:
    def __init__(self):

        self.applied_rules = []

    def apply(self, config, rules):
        while self.apply_for_real(config, rules):
            pass
        print(self.pretty_print_applied_rules())

    def apply_for_real(self, config, rules):
        changed = True

        sorted_rules = sorted(rules, key=lambda rule: rule[1])

        while changed:
            changed = False
            for override, priority, action, condition in sorted_rules:
                try:
                    # 🧠 Evaluate directly with config.__dict__ as the variable context
                    if eval(condition, {}, config.__dict__):
                    ##if eval(condition, globals(), config.__dict__):
                    #context = {**globals(), **config.__dict__}
                    #if eval(condition, context):
                        for key, value in action.items():
                            if override or getattr(config, key) is None:
                                setattr(config, key, value)
                                self.applied_rules.append({
                                    "priority": priority,
                                    "setting": key,
                                    "value": value,
                                    "condition": condition
                                })
                                return  True
                except Exception as e:
                    print(f"⚠️ Rule failed: {priority} {condition} -> {e}")
        return False




    def pretty_print_applied_rules(self):
        if not self.applied_rules:
            return ("🚫 No rules were applied.")

        return_value = f"\n🧩 Applied Rules:"
        for rule in sorted(self.applied_rules, key=lambda r: r['priority']):
            return_value += (f"  ➤ [{rule['priority']}] Set '{rule['setting']}' = {rule['value']} (because {rule['condition']}\n)")



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