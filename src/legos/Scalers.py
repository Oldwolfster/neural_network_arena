from src.engine.TrainingData import TrainingData

class MultiScaler:
    def __init__(self, training_data: TrainingData):
        self.training_data = training_data
        self.scalers = [Scaler_NONE for _ in range(training_data.input_count + 1)]  # +1 for target
        self.scaled_data = []
        self.scaled_samples = []  # ← Packed [inputs..., target]
        self.unscaled_samples = training_data.get_list()

    def set_input_scaler(self, scaler, index):
        self.scalers[index] = scaler

    def set_target_scaler(self, scaler):
        self.scalers[-1] = scaler

    def scale_all(self):
        """
        Scales training data using the assigned scalers. Populates scaled_samples and returns inputs + targets.
        """
        raw_samples = self.unscaled_samples
        transposed = list(zip(*raw_samples))
        self.scaled_data = [
            scaler.scale(list(feature)) for scaler, feature in zip(self.scalers, transposed)
        ]
        # Split back into inputs and targets
        inputs = list(zip(*self.scaled_data[:-1]))  # all but last column
        targets = self.scaled_data[-1]
        self.scaled_samples = [list(inp) + [target] for inp, target in zip(inputs, targets)]
        return inputs, targets

    def unscale_input(self, input_index: int, value):
        scaler = self.scalers[input_index]
        return scaler.unscale(scaler.params, value)

    def unscale_target(self, value):
        scaler = self.scalers[-1]
        return scaler.unscale(value)

    @property
    def inputs_are_unscaled(self):
        return all(scaler is Scaler_NONE for scaler in self.scalers[:-1])

    @property
    def target_is_unscaled(self):
        return self.scalers[-1] is Scaler_NONE

    @property
    def all_unscaled(self):
        return self.inputs_are_unscaled and self.target_is_unscaled

    def get_scaling_labels(self):
        labels = list(self.training_data.feature_labels)
        #labels.append("Target")  # Prevent extra line mismatch
        return [f"scaling: {label}" for label in labels]

    def get_scaling_names(self):
        return  ["-------------"]+ [scaler.name for scaler in self.scalers[:-1]] + ["Coming Soon"]  # blank for placeholder

    def __repr__(self):
        return f"MultiScaler({[s.name for s in self.scalers]})"


class Scaler:
    def __init__(self, method, name="Scaler", desc="", when_to_use="", best_for=""):
        self.method = method
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for
        self.params = {}
        self.scaled_data = []  # ← store scaled inputs for inspection

    def scale(self, values):
        """
        Fits the scaler to the flat list of values and returns the scaled result.
        Also stores internal params and scaled data.
        """
        self.params.clear()
        self.scaled_data = self.method.scale(self.params, values)
        return self.scaled_data

    def unscale(self, scaled):
        """
        Un-scales a single value or list of values back to the original space.
        """
        return self.method.unscale(self.params, scaled)

    def __repr__(self):
        return f"Scaler(name={self.name})"

class MinMaxMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1
        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            params['min'] = min(vals)
            params['max'] = max(vals)
            return [(v - params['min']) / (params['max'] - params['min']) for v in vals]
        else:
            params['min'] = [min(v[i] for v in values) for i in range(dim)]
            params['max'] = [max(v[i] for v in values) for i in range(dim)]
            return [
                [(v[i] - params['min'][i]) / (params['max'][i] - params['min'][i]) for i in range(dim)]
                for v in values
            ]

    def unscale(self, params, x):
        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [
                [xv[i] * (params['max'][i] - params['min'][i]) + params['min'][i] for i in range(len(xv))]
                for xv in x
            ]
        elif isinstance(x, list):
            return [
                xv * (params['max'] - params['min']) + params['min'] for xv in x
            ]
        else:
            return x * (params['max'] - params['min']) + params['min']


Scaler_MinMax = Scaler(
    method=MinMaxMethod(),
    name="Min-Max",
    desc="Scales data to a fixed range [0,1] using min and max.",
    when_to_use="Targets or inputs with known range and no severe outliers.",
    best_for="Tanh, Sigmoid, Linear and the kitchen sink."
)
class NoScalingMethod:
    def scale(self, params, values):
        return values  # Pass-through

    def unscale(self, params, x):
        return x  # Pass-through


Scaler_NONE = Scaler(
    method=NoScalingMethod(),
    name="No Scaling",
    desc="Pass-through scaler that performs no transformation.",
    when_to_use="Use when raw values are already appropriately scaled or scaling is undesirable.",
    best_for="Debugging, baselines, or models insensitive to scale."
)