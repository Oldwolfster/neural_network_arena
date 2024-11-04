from typing import List, Tuple, Optional
from dataclasses import dataclass
import statistics
import random
import math

@dataclass
class Sample:
    """
    Represents a single training sample with input features and target value.

    Attributes:
        inputs: A tuple of feature values for the sample
        target: The target/label value for the sample
        is_outlier: Flag indicating whether this sample has been marked as an outlier
    """
    inputs: Tuple[float, ...]
    target: float
    is_outlier: bool = False


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
        if not data:
            raise ValueError("Cannot initialize TrainingData with empty data")

        # Validate data consistency
        feature_dim = len(data[0]) - 1  # -1 for target value
        if any(len(sample) - 1 != feature_dim for sample in data):
            raise ValueError(f"Inconsistent feature dimensions. Expected {feature_dim} features for all samples")

        # Validate numeric values
        for i, sample in enumerate(data):
            if any(not math.isfinite(x) for x in sample):
                raise ValueError(f"Invalid numeric value (NaN or infinite) found in sample at index {i}")

        self.td_original: List[Sample] = [
            Sample(inputs=sample[:-1], target=sample[-1]) for sample in data
        ]
        self.td_zscore: List[Sample] = []
        self.td_minmax: List[Sample] = []
        self._sample_count: int = len(self.td_original)
        self.current_data_list: List[Sample] = self.td_original

        # Normalization parameters
        self._zscore_means: Optional[List[float]] = None
        self._zscore_stdevs: Optional[List[float]] = None

    @property
    def sample_count(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset
        """
        return self._sample_count

    @property
    def sum_targets(self) -> float:
        """
        Calculate the sum of all target values in the original dataset.

        Returns:
            float: Sum of all target values

        Raises:
            OverflowError: If the sum exceeds floating-point limits
        """
        try:
            return sum(sample.target for sample in self.td_original)
        except OverflowError as e:
            raise OverflowError("Sum of targets exceeded floating-point limits") from e

    def set_normalization_original(self) -> None:
        """
        Reset the current data list to use the original, unnormalized data.
        This operation is always safe and does not require computation.
        """
        self.current_data_list = self.td_original

    def set_normalization_zscore(self) -> None:
        """
        Set the current data list to use Z-score normalized data.
        Z-score normalization transforms features to have mean=0 and standard deviation=1.
        Computation is performed lazily on first call.

        Raises:
            ValueError: If normalization fails due to zero standard deviation in any feature
            RuntimeError: If computation fails for any other reason
        """
        if not self.td_zscore:
            try:
                self._compute_zscore_normalization()
            except Exception as e:
                raise RuntimeError(f"Failed to compute Z-score normalization: {str(e)}") from e
        self.current_data_list = self.td_zscore

    def set_normalization_minmax(self, min_value: float = 0.0, max_value: float = 1.0) -> None:
        """
        Set the current data list to use Min-Max normalized data.
        Min-Max normalization scales features to a specified range.
        Computation is performed lazily on first call.

        Args:
            min_value: Desired minimum value after normalization (default: 0.0)
            max_value: Desired maximum value after normalization (default: 1.0)

        Raises:
            ValueError: If min_value >= max_value
            ValueError: If any feature has zero range (max = min)
            RuntimeError: If computation fails for any other reason
        """
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")

        if not self.td_minmax:
            try:
                self._compute_minmax_normalization(min_value, max_value)
            except Exception as e:
                raise RuntimeError(f"Failed to compute Min-Max normalization: {str(e)}") from e
        self.current_data_list = self.td_minmax

    def get_list(self) -> List[Tuple[float, ...]]:
        """
        Get all samples as a list of tuples based on current normalization.

        Returns:
            List[Tuple[float, ...]]: List where each tuple contains feature values followed by target value
        """
        return [(*sample.inputs, sample.target) for sample in self.current_data_list]

    def get_sample(self, index: int) -> Sample:
        """
        Retrieve a single sample based on current normalization.

        Args:
            index: Index of the desired sample

        Returns:
            Sample: The requested sample

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < self._sample_count:
            raise IndexError(f"Sample index {index} out of range [0, {self._sample_count - 1}]")
        return self.current_data_list[index]

    def _compute_zscore_normalization(self) -> None:
        """
        Compute Z-score normalization parameters and normalized data.
        Results are stored in td_zscore list.

        Raises:
            ValueError: If any feature has zero standard deviation
            RuntimeError: If computation fails for any other reason
        """
        num_features = len(self.td_original[0].inputs)
        feature_values = [[] for _ in range(num_features)]

        # Collect values for each feature
        for sample in self.td_original:
            for i, value in enumerate(sample.inputs):
                feature_values[i].append(value)

        try:
            # Compute mean and std for each feature
            self._zscore_means = [statistics.mean(values) for values in feature_values]
            self._zscore_stdevs = [statistics.stdev(values) for values in feature_values]

            # Check for zero standard deviations
            zero_std_features = [i for i, std in enumerate(self._zscore_stdevs) if std == 0]
            if zero_std_features:
                raise ValueError(f"Zero standard deviation in features at indices: {zero_std_features}")

            # Normalize inputs
            for sample in self.td_original:
                normalized_inputs = tuple(
                    (value - self._zscore_means[i]) / self._zscore_stdevs[i]
                    for i, value in enumerate(sample.inputs)
                )
                self.td_zscore.append(Sample(inputs=normalized_inputs, target=sample.target))
        except Exception as e:
            self.td_zscore = []  # Reset on failure
            raise RuntimeError(f"Z-score normalization failed: {str(e)}") from e

    def _compute_minmax_normalization(self, min_value: float, max_value: float) -> None:
        """
        Compute Min-Max normalization parameters and normalized data.
        Results are stored in td_minmax list.

        Args:
            min_value: Desired minimum value after normalization
            max_value: Desired maximum value after normalization

        Raises:
            ValueError: If any feature has zero range (max = min)
            RuntimeError: If computation fails for any other reason
        """
        num_features = len(self.td_original[0].inputs)
        feature_values = [[] for _ in range(num_features)]

        # Collect values for each feature
        for sample in self.td_original:
            for i, value in enumerate(sample.inputs):
                feature_values[i].append(value)

        try:
            # Compute min and max for each feature
            mins = [min(values) for values in feature_values]
            maxs = [max(values) for values in feature_values]

            # Check for zero ranges
            zero_range_features = [i for i, (min_val, max_val)
                                 in enumerate(zip(mins, maxs))
                                 if min_val == max_val]
            if zero_range_features:
                raise ValueError(f"Zero range (max = min) in features at indices: {zero_range_features}")

            # Normalize inputs
            for sample in self.td_original:
                normalized_inputs = tuple(
                    ((value - mins[i]) / (maxs[i] - mins[i]) * (max_value - min_value) + min_value)
                    for i, value in enumerate(sample.inputs)
                )
                self.td_minmax.append(Sample(inputs=normalized_inputs, target=sample.target))
        except Exception as e:
            self.td_minmax = []  # Reset on failure
            raise RuntimeError(f"Min-Max normalization failed: {str(e)}") from e

    @property
    def zscore_means(self) -> Optional[List[float]]:
        """
        Get the means used in Z-score normalization, if computed.

        Returns:
            Optional[List[float]]: List of means for each feature, or None if Z-score
                                 normalization hasn't been computed
        """
        return self._zscore_means

    @property
    def zscore_stdevs(self) -> Optional[List[float]]:
        """
        Get the standard deviations used in Z-score normalization, if computed.

        Returns:
            Optional[List[float]]: List of standard deviations for each feature,
                                 or None if Z-score normalization hasn't been computed
        """
        return self._zscore_stdevs

    def segment_data(self, validation_ratio: float = 0.2) -> Tuple['TrainingData', 'TrainingData']:
        """
        Split data into training and validation sets.

        Args:
            validation_ratio: Proportion of data to use for validation (default: 0.2)

        Returns:
            Tuple[TrainingData, TrainingData]: Tuple containing (training_data, validation_data)

        Raises:
            ValueError: If validation_ratio is not in range (0, 1)
            ValueError: If dataset is too small to split given the ratio
        """
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1")

        min_samples = 2  # Minimum samples needed in each split
        if self._sample_count < min_samples * 2:
            raise ValueError(f"Dataset too small to split: need at least {min_samples * 2} samples")

        validation_size = int(len(self.current_data_list) * validation_ratio)
        if validation_size < min_samples:
            raise ValueError(f"validation_ratio too small: would result in only {validation_size} validation samples")

        data_copy = self.current_data_list[:]
        random.shuffle(data_copy)
        split_index = len(data_copy) - validation_size

        # Convert samples back to tuples for initialization
        training_samples = [(*sample.inputs, sample.target) for sample in data_copy[:split_index]]
        validation_samples = [(*sample.inputs, sample.target) for sample in data_copy[split_index:]]

        return TrainingData(training_samples), TrainingData(validation_samples)

    def batch_data(self, batch_size: int) -> List[List[Sample]]:
        """
        Divide the data into batches.

        Args:
            batch_size: Number of samples per batch

        Returns:
            List[List[Sample]]: List of batches, where each batch is a list of Sample instances

        Raises:
            ValueError: If batch_size is less than 1
            ValueError: If batch_size is larger than dataset size
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > self._sample_count:
            raise ValueError(f"batch_size ({batch_size}) cannot be larger than dataset size ({self._sample_count})")

        data_copy = self.current_data_list[:]
        random.shuffle(data_copy)
        return [
            data_copy[i:i + batch_size]
            for i in range(0, len(data_copy), batch_size)
        ]