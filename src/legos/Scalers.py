class Scaler:
    def __init__(self, method, name="Scaler", desc="", when_to_use="", best_for=""):
        self.method = method
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for
        self.params = {}
        self.scaled_data = []  # ← store scaled inputs for inspection

    def scale(self, data_list):
        """
        Fits the scaler to the data and returns the scaled result.
        Also stores internal params and scaled data.
        """
        self.params.clear()
        self.scaled_data = self.method.scale(self.params, data_list)
        return self.scaled_data

    def unscale(self, x):
        return self.method.unscale(self.params, x)

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


class ZScoreMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1
        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
            params['mean'] = mean
            params['std'] = std
            return [(v - mean) / std for v in vals]
        else:
            means = [sum(v[i] for v in values) / len(values) for i in range(dim)]
            stds = [
                (sum((v[i] - means[i]) ** 2 for v in values) / len(values)) ** 0.5 or 1.0
                for i in range(dim)
            ]
            params['mean'] = means
            params['std'] = stds
            return [
                [(v[i] - means[i]) / stds[i] for i in range(dim)] for v in values
            ]

    def unscale(self, params, x):
        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [
                [xv[i] * params['std'][i] + params['mean'][i] for i in range(len(xv))]
                for xv in x
            ]
        elif isinstance(x, list):
            return [xv * params['std'] + params['mean'] for xv in x]
        else:
            return x * params['std'] + params['mean']


Scaler_ZScore = Scaler(
    method=ZScoreMethod(),
    name="Z-Score",
    desc="Scales data to zero mean and unit variance using standard score.",
    when_to_use="Data with unknown range or needing normalization for gradient stability.",
    best_for="Most ML algorithms, especially gradient-based methods."
)


class MaxAbsMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1
        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            max_abs = max(abs(v) for v in vals) or 1.0
            params['max_abs'] = max_abs
            return [v / max_abs for v in vals]
        else:
            max_abs = [max(abs(v[i]) for v in values) or 1.0 for i in range(dim)]
            params['max_abs'] = max_abs
            return [[v[i] / max_abs[i] for i in range(dim)] for v in values]

    def unscale(self, params, x):
        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [[xv[i] * params['max_abs'][i] for i in range(len(xv))] for xv in x]
        elif isinstance(x, list):
            return [xv * params['max_abs'] for xv in x]
        else:
            return x * params['max_abs']


Scaler_MaxAbs = Scaler(
    method=MaxAbsMethod(),
    name="Max-Abs",
    desc="Scales data to [-1, 1] range using the maximum absolute value.",
    when_to_use="Inputs centered around zero or signed data without strong outliers.",
    best_for="Linear models or cases where sign preservation is important."
)


class RobustScalerMethod:

    def _sanitize_iqr(self, iqr):
        """
        Ensure that no dimension of `iqr` is zero.
        Works whether `iqr` is a scalar or an array.
        """
        import numpy as np

        if np.isscalar(iqr):
            return 1.0 if iqr == 0 else iqr
        else:
            # make sure it’s an array
            iqr = np.array(iqr)
            iqr[iqr == 0] = 1.0
            return iqr

    def scale(self, params, values):
        import numpy as np
        arr = np.array(values)

        # Compute median and (raw) IQR
        params['median'] = np.median(arr, axis=0)
        raw_iqr         = np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)

        # Sanitize: avoid division by zero
        params['iqr'] = self._sanitize_iqr(raw_iqr)

        # Do the scaling
        return ((arr - params['median']) / params['iqr']).tolist()

    def unscale(self, params, x):
        import numpy as np
        x = np.array(x)
        return (x * params['iqr'] + params['median']).tolist()


Scaler_Robust = Scaler(
    method=RobustScalerMethod(),
    name="Robust",
    desc="Uses median and IQR to reduce the influence of outliers.",
    when_to_use="Data with heavy outliers or skewed distributions.",
    best_for="Linear models or datasets with rare extreme values."
)


class LogScalerMethod:
    def scale(self, params, values):
        import numpy as np
        arr = np.array(values)
        params['offset'] = np.abs(np.min(arr)) + 1 if np.min(arr) <= 0 else 0
        return np.log(arr + params['offset']).tolist()

    def unscale(self, params, x):
        import numpy as np
        return (np.exp(x) - params['offset']).tolist()


Scaler_Log = Scaler(
    method=LogScalerMethod(),
    name="Log",
    desc="Applies natural log transform to reduce skew.",
    when_to_use="Data with exponential growth or multiplicative relationships.",
    best_for="Financial, biological, or power-law data."
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
