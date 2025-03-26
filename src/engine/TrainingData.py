from dataclasses import dataclass
from typing import List, Tuple, Optional

from src.Legos.LossFunctions import LossFunction


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

    def __init__(self, data: List[Tuple[float, ...]], feature_labels: Optional[List[str]] = None, target_labels: Optional[List[str]] = None) -> None:
        """
        Initialize the TrainingData instance with raw input data.

        Args:
            data: List of tuples where each tuple contains input features and a target value.
                 The last element of each tuple is treated as the target value.
            feature_labels: List of what each element of the tuple represents
            target_labels: For binary decision, two classes the targets represent

        Raises:
            ValueError: If data is empty
            ValueError: If data contains inconsistent feature dimensions
            ValueError: If any numeric values are invalid (NaN or infinite)
        """
        self.arena_name     = "Unknown"
        self.td_original    = [Tuple[float, ...]]
        self.td_original    = data                  # Store the original data
        self.td_z_score     = [Tuple[float, ...]]   # List in case model requests zscore
        self.td_min_max     = []                    # List in case model request minmax
        self.td_current     = self.td_original      # pointer to the "selected list" defaults to original
        self._cache         = {}                    # Private dictionary for caching values
        self.feature_labels = feature_labels        # Optional feature labels (e.g., ["Credit Score", "Income"])
        self.target_labels  = target_labels         # Optional outcome labels (e.g., ["Repaid", "Defaulted"])
        self.norm_scheme    = "Not applied"

        self.problem_type   = self.determine_problem_type(data)
        if self.problem_type == "Binary Decision" and not self.target_labels: # If it's BD and no labels were provided, assign default labels
            self.target_labels = ["Class Alpha", "Class Beta"]


    def determine_problem_type(self, data: List[Tuple[float, ...]]) -> str:
        """
        Examine training data to determine if it's binary decision or regression
        """

        unique_values = set(item[-1] for item in data)  # Use last element for targets
        if len(unique_values) == 2:
            return "Binary Decision"
        elif len(unique_values) > 2:
            return "Regression"
        else:
            return "Inconclusive"

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
        for sample in self.td_current:
            if sample[-1] not in seen:
                seen[sample[-1]] = None  # Preserve insertion order

        existing_targets = list(seen.keys())

        if len(existing_targets) != 2:
            raise ValueError(f"Binary Decision dataset expected 2 distinct target values, but found: {existing_targets}")

        # Map old targets to new targets (Preserving Arena's order)
        target_map = {existing_targets[0]: target_alpha, existing_targets[1]: target_beta}
        #print(f"DEBUG: Before applying BD rules, targets: {set(sample[-1] for sample in self.td_current)}")


        # Replace targets in training data (without modifying input order)
        self.td_current = [
            (*sample[:-1], target_map[sample[-1]]) for sample in self.td_current
        ]
        #print(f"DEBUG: After applying BD rules, targets: {set(sample[-1] for sample in self.td_current)}")
        #print(f"DEBUG: Target Mapping in TrainingData: {target_map}")
        return target_alpha, target_beta, threshold
    #print(f"✅ Binary Decision targets updated: {target_map}")

    @property
    def input_max(self) -> float:
        """
        Returns:
            float: The largest input value across all training data tuples.
        """
        if "max_input" not in self._cache:
            if not self.td_original:
                raise ValueError("Training data is empty; cannot compute max input.")
            self._cache["max_input"] = max(max(t[:-1]) for t in self.td_original)
        return self._cache["max_input"]

    def everything_max_magnitude(self) -> float:
        """
        Returns:
            float: The largest value across all training data tuples (inputs and outputs).
        """
        if not self.td_original:
            raise ValueError("Training data is empty; cannot compute maximum value.")
        # Iterate over each tuple, then each element in the tuple, and compute the max.
        self._cache["everything_max"] = abs(max(item for t in self.td_original for item in t))
        return self._cache["everything_max"]

    @property
    def output_max(self) -> float:
        """
        Returns:
            float: The largest input value across all training data tuples.
        """
        if "max_output" not in self._cache:
            if not self.td_original:
                raise ValueError("Training data is empty; cannot compute max output.")
            self._cache["max_output"] = max(max(t[:-1]) for t in self.td_original)
        return self._cache["max_output"]

    @property
    def sum_of_targets(self) -> int:
        """
        Returns:
            int: The sum of the targets for the entire training data
        """
        if "sum_targets" not in self._cache:
            if not self.td_original:
                raise ValueError("Training data is empty; cannot compute sum of targets.")
            self._cache["sum_targets"] = sum(tuple[-1] for tuple in self.td_original)
        return self._cache["sum_targets"]

    @property
    def input_count(self) -> int:
        """
        Returns:
            int: The number of inputs  in the training data
        """
        if "input_count" not in self._cache:
            self._cache["input_count"] = len(self.td_original[0])  - 1   #len(self.training_samples[0]) - 1
        return self._cache["input_count"]

    @property
    def sample_count(self) -> int:
        """
        Returns:
            int: The number of samples in the training data
        """
        if "sample_count" not in self._cache:
            self._cache["sample_count"] = len(self.td_original)
        return self._cache["sample_count"]

    def get_list(self) -> List[Tuple[float, ...]]:
        """
        Get all samples as a list of tuples based on current normalization selection.

        Returns:
            List[Tuple[float, ...]]: List where each tuple contains feature values(inputs) followed by target value
        """
        return self.td_current

    def reset_to_default(self) -> None:
        """
        Reset the current data list to use the original, unnormalized data.
        This operation is always safe and does not require computation.
        """
        self.td_current = self.td_original
        self.norm_scheme= "Not applied"

    def calculate_min_max(self):
        """
        Calculates the global min and max for each feature across all data tuples.

        :return: Two lists containing min and max values for each feature.
        """
        if not self.td_current:
            raise ValueError("No data available for normalization.")

        num_features = len(self.td_current[0]) - 1  # Exclude label
        min_values = [float('inf')] * num_features
        max_values = [float('-inf')] * num_features

        for tuple in self.td_current:
            for i in range(num_features):
                feature = tuple[i]
                if feature < min_values[i]:
                    min_values[i] = feature
                if feature > max_values[i]:
                    max_values[i] = feature

        return min_values, max_values

    def set_normalization_min_max(self):
        """
        1) populates list td_minmax with the features (all elements except the last) using min-max scaling.
        2) points the "current list" to td_min_max

        Returns:
            td_min_max
        """
        min_values, max_values  = self.calculate_min_max()
        denominators            = []
        self.norm_scheme        = "Min-Max"

        # Calculate denominators and handle division by zero
        for min_val, max_val in zip(min_values, max_values):
            denominator = max_val - min_val
            if denominator == 0:
                denominator = 1  # To avoid division by zero; alternatively, set normalized value to 0
            denominators.append(denominator)

        for idx, tuple in enumerate(self.td_current):
            #print(f"Unnormalized tuple #{idx + 1}: {tuple}")
            norm_tuple = []
            for x, feature in enumerate(tuple[:-1]):
                #print(f"  Feature #{x + 1}: {feature}")
                norm_value = (feature - min_values[x]) / denominators[x]
                norm_tuple.append(norm_value)
                #print(f"    Normalized Feature #{x + 1}: {norm_value}")
            norm_tuple.append(tuple[-1])  # Append the label without normalization
            self.td_min_max.append(norm_tuple)
            #print(f"  Normalized tuple #{idx + 1}: {tuple(norm_tuple)}\n")
        #print(f"Normalized data{self.td_min_max}")
        self.td_current = self.td_min_max # Point "Current mode" to the min max list

    @property
    def sum_of_inputs(self) -> List[float]:
        """
        Computes and caches the sum of each input feature across all training samples.

        Returns:
            List[float]: A list where each element is the sum of the corresponding input feature
                         across all training samples.
        """
        if "sum_inputs" not in self._cache:
            if not self.td_original:
                raise ValueError("Training data is empty; cannot compute sum of inputs.")

            num_features = len(self.td_original[0]) - 1  # Exclude the target value
            sums = [0] * num_features

            for sample in self.td_original:
                for i in range(num_features):
                    sums[i] += sample[i]  # Sum each input feature

            self._cache["sum_inputs"] = sums

            return self._cache["sum_inputs"]
    @property
    def normalizers(self) -> List[float]:
        """
        Computes and caches the normalizers for each input feature, adjusting for the global impact on updates.

        Returns:
            List[float]: A list of normalized values for each input feature.
        """
        if "normalizers" not in self._cache:
            sum_inputs = self.sum_of_inputs  # Get the sum of each input feature
            total_sum = sum(sum_inputs)     # Total sum of all input feature sums

            if total_sum == 0:
                raise ValueError("Total sum of inputs is zero; cannot compute normalizers.")

            raw_normalizers = [s / total_sum for s in sum_inputs]

            # Calculate adjustment to preserve update magnitude
            scaling_factor = len(raw_normalizers) / sum(raw_normalizers)

            # Apply scaling to ensure updates retain original magnitude
            adjusted_normalizers = [n * scaling_factor for n in raw_normalizers]
            self._cache["normalizers"] = adjusted_normalizers

        return self._cache["normalizers"]


"""
        # create a list of min and max values for each input
        min_values = [min(tuple[:-1]) for tuple in self.td_current]
        max_values = [max(tuple[:-1]) for tuple in self.td_current]
        denominators = max_values - min)values

        for tuple in self.td_current:
            print(f"unnormalized features:{tuple}")
            for x, input in  enumerate(tuple[:-1]):
                norm_tuple=[]
                print(f"#{x} input:{input}")
                norm_value = input - min_values[x]
                norm_tuple.append(norm_value)
            norm_tuple.append(tuple[-1])
            print(f"#{x} input:{input}")
            #self.td_min_max.append()
            #normalized_features = [(    feature - min_val) /
            #                       (    max_val - min_val)
            #                            for feature, min_val, max_val
            #                            in zip(tuple[:-1], min_values, max_values)]

            # normalized_tuple    =       tuple(normalized_features) + (tuple[-1],)  # Add the original target
            # Above line gives error: TypeError: 'tuple' object is not callable
            # normalized_tuple = zip(normalized_features,tuple[-1])
            # normalized_tuple = tuple(normalized_features + (tuple[-1],))
            # Above line gives error: TypeError: can only concatenate list (not "tuple") to list
            print (f"notmalized features {normalized_features}")
            #normalized_tuple = tuple(normalized_features + [tuple[-1]])
            # Above line gives error: TypeError: 'tuple' object is not callable
            self.td_min_max     .       append(normalized_tuple)

        self.td_current         =       self.td_min_max # Point "Current mode" to the min max list













    "" "

    def z_scale(self):
        "" "Normalizes the features (all elements except the last) using z-score scaling.

        Returns:
            A new list of tuples with normalized features and the original target.
        "" "

        # Calculate the mean and standard deviation for each feature
        means = [sum(feature) / len(self.td_current) for feature in zip(*self.td_current[:-1])]
        stds = [((feature - mean) ** 2).sum() / (len(self.td_current) - 1) for feature, mean in zip(zip(*self.td_current[:-1]), means)]

        normalized_data = []
        for tuple in self.td_current:
            normalized_features = [(feature - mean) / std for feature, mean, std in zip(tuple[:-1], means, stds)]
            normalized_tuple = tuple(normalized_features) + (tuple[-1],)  # Add the original target
            normalized_data.append(normalized_tuple)

        return normalized_data

    """