from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from src.Legos.LossFunctions import LossFunction
import statistics

class TrainingData:
    """
    Manages machine learning training data with support for various normalization methods and metrics.

    This class handles data storage, normalization, segmentation, and batching operations
    for training neural networks. Normalization methods (original, z-score, min-max) are
    computed lazily when first requested.

    Attributes:
        td_original: List of original, unmodified training samples
        td_zscore: List of z-score normalized samples (computed when first requested)
        td_minmax: List of min-max normalized samples (computed when first requested)
        current_data_list: Reference to currently active data list based on normalization setting
    """

    def __init__(self, raw_data: List[Tuple[float, ...]], feature_labels: Optional[List[str]] = None, target_labels: Optional[List[str]] = None) -> None:
        """
        Initialize the TrainingData instance with raw input data.

        Args:
            raw_data: List of tuples where each tuple contains input features and a target value.
                 The last element of each tuple is treated as the target value.
            feature_labels: List of what each element of the tuple represents
            target_labels: For binary decision, two classes the targets represent

        Raises:
            ValueError: If data is empty
            ValueError: If data contains inconsistent feature dimensions
            ValueError: If any numeric values are invalid (NaN or infinite)
        """
        self.arena_name     = "Unknown"
        self._raw           = [tuple(r) for r in raw_data]      # Store the original data
        self._current       = [tuple(r) for r in raw_data]      # Holds modified data.
        self._cache         = {}                    # Private dictionary for caching values
        self.feature_labels = feature_labels        # Optional feature labels (e.g., ["Credit Score", "Income"])
        self.target_labels  = target_labels         # Optional outcome labels (e.g., ["Repaid", "Defaulted"])
        self.norm_scheme    = "Not applied"

        self.problem_type   = self.determine_problem_type()

        if self.problem_type == "Binary Decision" and not self.target_labels: # If it's BD and no labels were provided, assign default labels
            self.target_labels = ["Class Alpha", "Class Beta"]

    def reset_to_default(self):
        """Restore self.current to the original raw data."""
        self._current = [tuple(r) for r in self._raw]
        self._cache = {}
        self.norm_scheme = None

    def get_list(self):
        """Get the latest version of the data (scaled or unscaled)."""
        return self._current

    def set_normalization_min_max(self):
        """
        1. Computes min/max from self._raw
        2. Builds a brand-new scaled list into self._current
        3. Sets norm_scheme = "Min-Max"
        """
        min_values, max_values = self.calculate_min_max()
        denominators = [(mx - mn) or 1 for mn, mx in zip(min_values, max_values)]
        self.norm_scheme = "Min-Max"

        scaled = []
        for record in self._raw:
            feats, label = record[:-1], record[-1]
            norm_feats = [
                (feats[i] - min_values[i]) / denominators[i]
                for i in range(len(feats))
            ]
            scaled.append((*norm_feats, label))

        self._current = scaled
        # clear any cached stats that depended on the old _current
        self._cache.pop("max_input",   None)
        self._cache.pop("max_output",  None)
        self._cache.pop("everything_max", None)
        self._cache.pop("input_count", None)
        self._cache.pop("sample_count", None)
        return self._current


    def calculate_target_min_max(self) -> Tuple[float, float]:
        """
        Returns the (min, max) of the target column in the raw data.
        """
        targets = [rec[-1] for rec in self._raw]
        return min(targets), max(targets)

    def set_target_normalization_min_max(self) -> List[Tuple[float, ...]]:
        """
        1. Computes target min/max from self._raw
        2. Scales only the target in self._current to [0,1]
        3. Stores the (min, max) in cache for later inversion
        4. Updates norm_scheme
        """
        t_min, t_max = self.calculate_target_min_max()
        denom = (t_max - t_min) or 1.0
        # cache for invert
        self._cache['target_minmax'] = (t_min, t_max)

        scaled = []
        for record in self._current:
            *feats, y = record
            y_scaled = (y - t_min) / denom
            scaled.append((*feats, y_scaled))

        self._current = scaled
        self.norm_scheme = "Min-Max (targets only)"
        return self._current

    def invert_min_max_target(self, y_scaled: float) -> float:
        """
        Given a normalized target y_scaled in [0,1], returns it in the original scale.
        """
        # fetch from cache or recompute if missing
        t_min, t_max = self._cache.get('target_minmax', self.calculate_target_min_max())
        return y_scaled * (t_max - t_min) + t_min



    def determine_problem_type(self) -> str:
        """
        Examine training data to determine if it's binary decision or regression
        """

        unique_values = set(item[-1] for item in self._current)  # Use last element for targets
        if len(unique_values) == 2:
            return "Binary Decision"
        elif len(unique_values) > 2:
            return "Regression"
        else:
            return "Inconclusive"

    def apply_binary_decision_targets_for_specific_loss_function(self, loss_function: LossFunction) ->  tuple[float, float, float]:
        """
        Updates targets in `td_current` for Binary Decision problems based on the loss function's BD rules.


        Args:
            loss_function: The loss function being used.

        Raises:
            ValueError: If unexpected target values are found.
        """
        if self.problem_type != "Binary Decision":
            return  # No need to modify targets if it's not BD

        target_alpha, target_beta, threshold = self.get_binary_decision_settings(loss_function)


        # Extract existing unique targets while preserving their original order
        seen = {}
        for sample in self._current:
            if sample[-1] not in seen:
                seen[sample[-1]] = None  # Preserve insertion order

        existing_targets = list(seen.keys())

        if len(existing_targets) != 2:
            raise ValueError(f"Binary Decision dataset expected 2 distinct target values, but found: {existing_targets}")

        # Map old targets to new targets (Preserving Arena's order)
        target_map = {existing_targets[0]: target_alpha, existing_targets[1]: target_beta}
        #print(f"DEBUG: Before applying BD rules, targets: {set(sample[-1] for sample in self.td_current)}")


        # Replace targets in training data (without modifying input order)
        self._current = [
            (*sample[:-1], target_map[sample[-1]]) for sample in self._current
        ]
        #print(f"DEBUG: After applying BD rules, targets: {set(sample[-1] for sample in self.td_current)}")
        #print(f"DEBUG: Target Mapping in TrainingData: {target_map}")
        return target_alpha, target_beta, threshold
        #print(f"✅ Binary Decision targets updated: {target_map}")

    def get_binary_decision_settings(self, loss_function: LossFunction) -> Tuple[float, float, float]:
        """
        Determines the appropriate targets and decision threshold for Binary Decision tasks.

        Args:
            loss_function: The loss function being used.

        Returns:
            Tuple containing:
            - target_alpha: The numerical target for Class Alpha (e.g., 0.0 or -1.0)
            - target_beta: The numerical target for Class Beta (e.g., 1.0)
            - threshold: The decision boundary
        """
        if self.problem_type != "Binary Decision":
            return "N/A", "N/A", "N/A"
            #raise ValueError("get_bd_settings() was called, but the problem type is not Binary Decision.")

        # Directly access bd_rules (assuming it's always set in LossFunction)
        bd_rules = loss_function.bd_rules

        return bd_rules[0], bd_rules[1], (bd_rules[0] + bd_rules[1]) / 2  # Auto-calculate threshold

    def calculate_min_max(self):
        """
        Calculates global min/max for each feature across the *raw* data.
        Returns two lists: [min1, min2, …], [max1, max2, …].
        """
        if not self._raw:
            raise ValueError("No raw data available for normalization.")
        num_features = len(self._raw[0]) - 1
        min_values = [float('inf')] * num_features
        max_values = [float('-inf')] * num_features

        for record in self._raw:
            for i in range(num_features):
                v = record[i]
                if v < min_values[i]:
                    min_values[i] = v
                if v > max_values[i]:
                    max_values[i] = v

        return min_values, max_values

    def set_normalization_min_max(self):
        """
        1. Computes min/max from self._raw
        2. Builds a brand-new scaled list into self._current
        3. Sets norm_scheme = "Min-Max"
        """
        min_values, max_values = self.calculate_min_max()
        denominators = [(mx - mn) or 1 for mn, mx in zip(min_values, max_values)]
        self.norm_scheme = "Min-Max"

        scaled = []
        for record in self._raw:
            feats, label = record[:-1], record[-1]
            norm_feats = [
                (feats[i] - min_values[i]) / denominators[i]
                for i in range(len(feats))
            ]
            scaled.append((*norm_feats, label))

        self._current = scaled
        # clear any cached stats that depended on the old _current
        self._cache.pop("max_input",   None)
        self._cache.pop("max_output",  None)
        self._cache.pop("everything_max", None)
        self._cache.pop("input_count", None)
        self._cache.pop("sample_count", None)
        return self._current

    def set_normalization_z_score(self):
        """
        1. Computes μ and σ (population) from self._raw
        2. Builds a brand-new z-scored list into self._current
        3. Sets norm_scheme = "Z-Score"
        """
        if not self._raw:
            raise ValueError("No raw data available for normalization.")

        # transpose only features
        cols = list(zip(*(r[:-1] for r in self._raw)))
        means = [statistics.mean(c) for c in cols]
        stds  = [statistics.pstdev(c) or 1 for c in cols]

        self.norm_scheme = "Z-Score"
        scaled = []
        for record in self._raw:
            feats, label = record[:-1], record[-1]
            z_feats = [
                (feats[i] - means[i]) / stds[i]
                for i in range(len(feats))
            ]
            scaled.append((*z_feats, label))

        self._current = scaled
        # reset cached stats
        self._cache.pop("max_input",   None)
        self._cache.pop("max_output",  None)
        self._cache.pop("everything_max", None)
        self._cache.pop("input_count", None)
        self._cache.pop("sample_count", None)
        return self._current

    @property
    def input_max(self) -> float:
        """
        Returns:
            float: The largest input value across all training data tuples.
        """
        if "max_input" not in self._cache:
            if not self._current:
                raise ValueError("Training data is empty; cannot compute max input.")
            # Find the max *feature* value in each sample, then take the overall max
            self._cache["max_input"] = max(
                max(record[:-1])  # max feature in this record
                for record in self._current
            )
        return self._cache["max_input"]

    @property
    def output_max(self) -> float:
        if "max_output" not in self._cache:
            if not self._current:
                raise ValueError("Training data is empty; cannot compute max output.")
            self._cache["max_output"] = max(t[-1] for t in self._current)
        return self._cache["max_output"]

    def everything_max_magnitude(self) -> float:
        if "everything_max" not in self._cache:
            # pick the largest absolute value across all features+labels
            self._cache["everything_max"] = max(
                abs(item)
                for tup in self._current
                for item in tup
            )
        return self._cache["everything_max"]

    @property
    def sample_count(self) -> int:
        if "sample_count" not in self._cache:
            self._cache["sample_count"] = len(self._current)
        return self._cache["sample_count"]

    @property
    def input_count(self) -> int:
        if "input_count" not in self._cache:
            if not self._current:
                raise ValueError("No data; cannot determine input count.")
            # number of features = tuple length minus one label
            self._cache["input_count"] = len(self._current[0]) - 1
        return self._cache["input_count"]



