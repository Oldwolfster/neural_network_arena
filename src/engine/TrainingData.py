from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from src.Legos.LossFunctions import StrategyLossFunction
import statistics
import numpy as np

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
        #self.raw_data       = [tuple(r) for r in raw_data]      # Store the original data
        self._raw_data = None  # initialize private backing field
        self.raw_data = raw_data  # use setter (it will freeze the input)
        self._fingerprint   = [row[-1] for row in self.raw_data[:10]]

        self._cache         = {}                    # Private dictionary for caching values
        self.feature_labels = feature_labels        # Optional feature labels (e.g., ["Credit Score", "Income"])

        if not target_labels:
            target_labels = ["Beta","Alpha"]
        self.target_labels  = target_labels         # Optional outcome labels (e.g., ["Repaid", "Defaulted"])
        #print(f"target_labels {target_labels}")

        self.problem_type   = self.determine_problem_type()
        self.is_binary_decision = False
        if self.problem_type == "Binary Decision": # If it's BD and no labels were provided, assign default labels
            self.is_binary_decision = True



    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        if self._raw_data is not None:
            raise AttributeError("raw_data is immutable and has already been set.")

        # deep freeze: convert inner sequences to tuples
        self._raw_data = tuple(tuple(row) for row in value)

    def verify_targets_unchanged(self, label=""):
        current = [row[-1] for row in self.raw_data[:10]]
        if current != self._fingerprint:
            print(f"🧨 TARGET CORRUPTION DETECTED during {label}!\nBefore: {self._fingerprint}\nNow:    {current}")
            assert False, "Training data was mutated!"


    def get_list(self):
        """Get the latest version of the data """
        return self.raw_data

    def get_inputs(self)-> int:
        return  [sample[:-1] for sample in self.raw_data]

    def get_targets(self):
        return  [sample[-1] for sample in self.raw_data]

    def determine_problem_type(self) -> str:
        """
        Examine training data to determine if it's binary decision or regression
        """

        unique_values = set(item[-1] for item in self.raw_data)  # Use last element for targets
        if len(unique_values) == 2:
            return "Binary Decision"
        elif len(unique_values) > 2:
            return "Regression"
        else:
            return "Inconclusive"


    @property
    def input_max(self) -> float:
        """
        Returns:
            float: The largest input value across all training data tuples.
        """
        if "max_input" not in self._cache:
            if not self.raw_data:
                raise ValueError("Training data is empty; cannot compute max input.")
            # Find the max *feature* value in each sample, then take the overall max
            self._cache["max_input"] = max(
                max(record[:-1])  # max feature in this record
                for record in self.raw_data
            )
        return self._cache["max_input"]

    @property
    def output_max(self) -> float:
        if "max_output" not in self._cache:
            if not self.raw_data:
                raise ValueError("Training data is empty; cannot compute max output.")
            self._cache["max_output"] = max(t[-1] for t in self.raw_data)
        return self._cache["max_output"]

    @property
    def mean_absolute_target(self) -> float:
        """
        Returns the mean of the absolute target values in the dataset.
        Useful for regression accuracy metrics like 1 - (MAE / mean).
        """
        if "mean_absolute_target" not in self._cache:
            if not self.raw_data:
                raise ValueError("Training data is empty; cannot compute mean target.")
            #self._cache["mean_absolute_target"] = np.mean([abs(t[-1]) for t in self.raw_data])
            self._cache["mean_absolute_target"] = np.mean(np.abs([t[-1] for t in self.raw_data]))
        return self._cache["mean_absolute_target"]


    def everything_max_magnitude(self) -> float:
        if "everything_max" not in self._cache:
            # pick the largest absolute value across all features+labels
            self._cache["everything_max"] = max(
                abs(item)
                for tup in self.raw_data
                for item in tup
            )
        return self._cache["everything_max"]

    @property
    def sample_count(self) -> int:
        if "sample_count" not in self._cache:
            self._cache["sample_count"] = len(self.raw_data)
        return self._cache["sample_count"]

    @property
    def input_count(self) -> int:
        if "input_count" not in self._cache:
            if not self.raw_data:
                raise ValueError("No data; cannot determine input count.")
            # number of features = tuple length minus one label
            self._cache["input_count"] = len(self.raw_data[0]) - 1
        return self._cache["input_count"]

    def has_outliers(values, threshold_ratio=0.05):
        import numpy as np
        arr = np.array(values)
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (arr < lower) | (arr > upper)
        return np.mean(outliers) > threshold_ratio

    def is_bounded(values, std_threshold=3):
        import numpy as np
        arr = np.array(values)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        upper_bound = mean + std_threshold * std
        lower_bound = mean - std_threshold * std
        return np.all((arr >= lower_bound) & (arr <= upper_bound))

    @property
    def target_values(self):
        return [row[-1] for row in self.raw_data]

    @property
    def target_mean(self) -> float:
        if "target_mean" not in self._cache:
            self._cache["target_mean"] = float(np.mean(self.target_values))
        return self._cache["target_mean"]

    @property
    def target_stdev(self) -> float:
        if "target_stdev" not in self._cache:
            self._cache["target_stdev"] = float(np.std(self.target_values))
        return self._cache["target_stdev"]

    @property
    def target_min(self) -> float:
        if "target_min" not in self._cache:
            self._cache["target_min"] = float(min(self.target_values))
        return self._cache["target_min"]

    @property
    def target_max(self) -> float:
        if "target_max" not in self._cache:
            self._cache["target_max"] = float(max(self.target_values))
        return self._cache["target_max"]

    @property
    def target_min_label(self) -> Optional[str]:
        if not self.target_labels or len(set(self.target_values)) != 2:
            return None
        return self.target_labels[0]

    @property
    def target_max_label(self) -> Optional[str]:
        if not self.target_labels or len(set(self.target_values)) != 2:
            return None
        return self.target_labels[1]
