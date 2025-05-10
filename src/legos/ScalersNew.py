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
    def scale(self, params, vals):
        # Compute min and max
        params['min'] = min(vals)
        params['max'] = max(vals)
        span = params['max'] - params['min'] or 1.0
        # Scale each value into [0,1]
        return [(v - params['min']) / span for v in vals]

    def unscale(self, params, scaled_vals):
        span = params['max'] - params['min']
        return [(v * span) + params['min'] for v in scaled_vals]


class ZScoreMethod:
    def scale(self, params, vals):
        # Compute mean and (non-zero) std
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        params['mean'], params['std'] = mean, std
        return [(v - mean) / std for v in vals]

    def unscale(self, params, scaled_vals):
        mean, std = params['mean'], params['std']
        return [(v * std) + mean for v in scaled_vals]


class MaxAbsMethod:
    def scale(self, params, vals):
        max_abs = max(abs(v) for v in vals) or 1.0
        params['max_abs'] = max_abs
        return [v / max_abs for v in vals]

    def unscale(self, params, scaled_vals):
        return [v * params['max_abs'] for v in scaled_vals]


class RobustScalerMethod:
    def _sanitize_iqr(self, iqr):
        # Avoid zero IQR
        return iqr or 1.0

    def scale(self, params, vals):
        import numpy as np
        arr = np.array(vals)
        median = float(np.median(arr))
        raw_iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        iqr = self._sanitize_iqr(raw_iqr)
        params['median'], params['iqr'] = median, iqr
        return ((arr - median) / iqr).tolist()

    def unscale(self, params, scaled_vals):
        import numpy as np
        arr = np.array(scaled_vals)
        return (arr * params['iqr'] + params['median']).tolist()


class LogScalerMethod:
    def scale(self, params, vals):
        import numpy as np
        arr = np.array(vals)
        offset = abs(arr.min()) + 1 if arr.min() <= 0 else 0
        params['offset'] = float(offset)
        return np.log(arr + offset).tolist()

    def unscale(self, params, scaled_vals):
        import numpy as np
        return (np.exp(scaled_vals) - params['offset']).tolist()


class NoScalingMethod:
    def scale(self, params, vals):
        return list(vals)  # Pass-through

    def unscale(self, params, scaled_vals):
        return list(scaled_vals)  # Pass-through


# Example instantiation for a single list
Scaler_MinMax = Scaler(
    method=MinMaxMethod(),
    name="Min-Max",
    desc="Scales data to [0,1] using min and max.",
    when_to_use="Known range, no severe outliers.",
    best_for="Tanh, Sigmoid, Linear, etc."
)
