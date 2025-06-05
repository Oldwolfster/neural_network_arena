from dataclasses import asdict
from typing import Any

from src.engine.Utils_DataClasses import RecordLevel


class LegoSelector:
    def __init__(self):
        self.applied_rules = []
        #if tri.should_record(RecordLevel.SUMMARY ):
        #    print(f"\t🧠 Welcome to 'Smart Configuration' Anything not set in your model will be set to optimal conditions(hopefully) 🧠\n\t🧠 ", end=""),

    def apply(self, config, rules, tri):
        # track which rule‐indices we’ve applied (and to what value)
        # keep running until no rule makes an actual change
        while self._apply_single_rule(config, rules, tri):
            pass


    def _apply_single_rule(self, config, rules,  tri) -> bool:
        """
        Returns True if it applied one rule (and mutated config).
        """
        # sort by priority
        sorted_rules = sorted(rules, key=lambda rule: rule[1])

        for idx, (override, priority, action, condition) in enumerate(sorted_rules):
            try:
                #if eval(condition, {}, config.__dict__):
                if self._safe_eval(condition, config):
                    for key, value in action.items():
                        current = getattr(config, key)

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
                            if tri.should_record(RecordLevel.SUMMARY ): #print(f"  ➤ Rule applied: {priority}: {key} = {value} from condition: '{condition}'")
                                print(f"{priority}: {key} = {value}", end="") # from condition: '{condition}'")
                            return True
            except Exception as e:
                print(f"⚠️ Rule failed: {priority} {condition} -> {e}")

        return False

    def _safe_eval(self, condition: str, config: object) -> bool:
        tokens = condition.strip().split()
        if not tokens:
            return False

        root_var = tokens[0].split(".")[0]

        # If the root var looks like a literal (e.g., '1', '"text"', 'True'), go ahead
        if root_var.isdigit() or root_var in {"True", "False"} or root_var.startswith(("'", '"')):
            return bool(eval(condition, {}, config.__dict__))

        # Otherwise, check if the referenced attribute exists
        if getattr(config, root_var, None) is None:
            return False

        return bool(eval(condition, {}, config.__dict__))



