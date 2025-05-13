from dataclasses import asdict
from typing import Any




class LegoSelector:
    def __init__(self):
        self.applied_rules = []

    def apply(self, config, rules, output_to_log: bool):
        # track which rule‐indices we’ve applied (and to what value)
        seen: dict[int, Any] = {}
        # keep running until no rule makes an actual change
        while self._apply_single_rule(config, rules, seen, output_to_log):
            pass

        #print(self.pretty_print_applied_rules())
        # if you ever re‐use this selector again, you can clear for next time
        seen.clear()

    def _apply_single_rule(self, config, rules, seen: dict[int, Any], output_to_log: bool) -> bool:
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
                            if output_to_log:
                                print(f"  ➤ Rule applied: {priority}: {key} = {value} from condition: '{condition}'")
                            return True
            except Exception as e:
                print(f"⚠️ Rule failed: {priority} {condition} -> {e}")

        return False

