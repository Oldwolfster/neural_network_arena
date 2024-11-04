from typing import List, Tuple, Optional
from dataclasses import dataclass
import statistics
import random
#TODO  in the future add robust and decimal normaliztaion methods

@dataclass
class Sample:
    inputs: Tuple[float, ...]
    target: float
    is_outlier: bool = False

class TrainingData:
    def __init__(self, data: List[Tuple[float, ...]]):
        self.td_original: List[Sample] = [
            Sample(inputs=sample[:-1], target=sample[-1]) for sample in data
        ]
        self.td_zscore: List[Sample] = []
        self.td_minmax: List[Sample] = []
        self._sample_count: int = len(self.td_original)
        self.current_data_list: List[Sample] = self.td_original  # Default to original data

        # Normalization parameters
        self._zscore_means: Optional[List[float]] = None
        self._zscore_stdevs: Optional[List[float]] = None

    @property
    def sample_count(self) -> int:
        """Returns the total number of samples."""
        return self._sample_count

    @property
    def sum_targets(self) -> float:
        """Returns the sum of all target values."""
        return sum(sample.target for sample in self.td_original)

    def set_normalization_original(self):
        """Set the current data list to original data."""
        self.current_data_list = self.td_original

    def set_normalization_zscore(self):
        """
        Set the current data list to Z-score normalized data.
        Normalization parameters are computed from the data.
        """
        if not self.td_zscore:
            self._compute_zscore_normalization()
        self.current_data_list = self.td_zscore

    def set_normalization_minmax(self, min_value: float = 0.0, max_value: float = 1.0):
        """
        Set the current data list to Min-Max normalized data.
        Allows specifying the desired scaling range.
        """
        if not self.td_minmax:
            self._compute_minmax_normalization(min_value, max_value)
        self.current_data_list = self.td_minmax

    def get_list(self) -> List[Tuple[float, ...]]:
        """Return the data as a list of tuples based on the current normalization."""
        return [(*sample.inputs, sample.target) for sample in self.current_data_list]

    def get_sample(self, index: int) -> Sample:
        """Retrieve a single sample based on the current normalization."""
        return self.current_data_list[index]

    def _compute_zscore_normalization(self):
        """Compute Z-score normalization and store the normalized data."""
        num_features = len(self.td_original[0].inputs)
        feature_values = [[] for _ in range(num_features)]

        # Collect values for each feature
        for sample in self.td_original:
            for i, value in enumerate(sample.inputs):
                feature_values[i].append(value)

        # Compute mean and std for each feature
        self._zscore_means = [statistics.mean(values) for values in feature_values]
        self._zscore_stdevs = [statistics.stdev(values) for values in feature_values]

        # Normalize inputs
        for sample in self.td_original:
            normalized_inputs = tuple(
                (value - self._zscore_means[i]) / self._zscore_stdevs[i] if self._zscore_stdevs[i] > 0 else 0.0
                for i, value in enumerate(sample.inputs)
            )
            self.td_zscore.append(Sample(inputs=normalized_inputs, target=sample.target))

    def _compute_minmax_normalization(self, min_value: float, max_value: float):
        """Compute Min-Max normalization and store the normalized data."""
        num_features = len(self.td_original[0].inputs)
        feature_values = [[] for _ in range(num_features)]

        # Collect values for each feature
        for sample in self.td_original:
            for i, value in enumerate(sample.inputs):
                feature_values[i].append(value)

        # Compute min and max for each feature
        mins = [min(values) for values in feature_values]
        maxs = [max(values) for values in feature_values]

        # Normalize inputs
        for sample in self.td_original:
            normalized_inputs = tuple(
                ((value - mins[i]) / (maxs[i] - mins[i]) * (max_value - min_value) + min_value)
                if (maxs[i] - mins[i]) > 0 else min_value
                for i, value in enumerate(sample.inputs)
            )
            self.td_minmax.append(Sample(inputs=normalized_inputs, target=sample.target))

    @property
    def zscore_means(self) -> Optional[List[float]]:
        """Returns the means used in Z-score normalization."""
        return self._zscore_means

    @property
    def zscore_stdevs(self) -> Optional[List[float]]:
        """Returns the standard deviations used in Z-score normalization."""
        return self._zscore_stdevs

    def segment_data(self, validation_ratio: float = 0.2) -> Tuple['TrainingData', 'TrainingData']:
        """
        Split data into training and validation sets.
        Returns two TrainingData instances.
        """
        data_copy = self.current_data_list[:]
        random.shuffle(data_copy)
        split_index = int(len(data_copy) * (1 - validation_ratio))
        training_samples = data_copy[:split_index]
        validation_samples = data_copy[split_index:]

        # Convert samples back to tuples for initialization
        training_data = TrainingData(
            [(*sample.inputs, sample.target) for sample in training_samples]
        )
        validation_data = TrainingData(
            [(*sample.inputs, sample.target) for sample in validation_samples]
        )

        return training_data, validation_data

    def batch_data(self, batch_size: int) -> List[List[Sample]]:
        """
        Divide the data into batches.
        Returns a list of batches, each batch is a list of Sample instances.
        """
        data_copy = self.current_data_list[:]
        random.shuffle(data_copy)
        return [
            data_copy[i:i + batch_size] for i in range(0, len(data_copy), batch_size)
        ]
"""
def _compute_adjusted_zscore_normalization(self, std_dev_multiplier=1.0, mean_adjustment=0.0):
    # Compute means and standard deviations
    # ... [same as before]
    
    # Normalize inputs with adjustments
    for sample in self.td_original:
        normalized_inputs = tuple(
            ((value - self._zscore_means[i]) / (self._zscore_stdevs[i] * std_dev_multiplier) + mean_adjustment)
            if self._zscore_stdevs[i] > 0 else mean_adjustment
            for i, value in enumerate(sample.inputs)
        )
        self.td_zscore.append(Sample(inputs=normalized_inputs, target=sample.target))


"""