from dataclasses import dataclass
from typing import List, Tuple, Optional

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

    def __init__(self, data: List[Tuple[float, ...]]) -> None:
        """
        Initialize the TrainingData instance with raw input data.

        Args:
            data: List of tuples where each tuple contains input features and a target value.
                 The last element of each tuple is treated as the target value.

        Raises:
            ValueError: If data is empty
            ValueError: If data contains inconsistent feature dimensions
            ValueError: If any numeric values are invalid (NaN or infinite)
        """
        self.td_original    = [Tuple[float, ...]]
        self.td_original    = data                  # Store the original data
        self.td_z_score     = [Tuple[float, ...]]   # List in case model requests zscore
        self.td_min_max     = []   # List in case model request minmax
        self.td_current     = self.td_original      # pointer to the "selected list" defaults to original
        self._cache         = {}  # Private dictionary for caching values

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
        if "sample_count" not in self._cache:
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
        min_values, max_values = self.calculate_min_max()
        denominators = []

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
                print(f"    Normalized Feature #{x + 1}: {norm_value}")
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
        Computes and caches the normalizers for each input feature based on their relative magnitudes.

        Returns:
            List[float]: A list of normalizers for each input feature.
        """
        if "normalizers" not in self._cache:
            sum_inputs = self.sum_of_inputs  # Get the sum of each input feature
            total_sum = sum(sum_inputs)     # Total sum of all input feature sums

            if total_sum == 0:
                raise ValueError("Total sum of inputs is zero; cannot compute normalizers.")

            self._cache["normalizers"] = [s / total_sum for s in sum_inputs]

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