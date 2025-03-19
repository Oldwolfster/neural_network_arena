import pygame
from src.NeuroForge.EZForm import EZForm
from src.engine.Utils import smart_format
from src.NeuroForge import Const
class DisplayPanelPrediction(EZForm):
    __slots__ = ("model_id", "problem_type", "loss_function")

    def __init__(self, model_id: str, problem_type: str, loss_function, width_pct: int, height_pct: int, left_pct: int, top_pct: int):
        self.model_id       = model_id
        self.loss_function  = loss_function
        self.problem_type   = problem_type
        self.target_name    = Const.configs[0].hyper.data_labels[-1].strip()

        # Define the fields and default values for the form
        fields = {
            self.target_name: "0.000",
            "Prediction": "0.000",
            "Error": "0.000 / 0.000",
            "Epoch Avg Error": "0.000",
            "Loss Function": self.loss_function.short_name,
            "Loss Gradient": "0.0",
        }

        # Initialize EZForm with updated fields
        super().__init__(
            fields=fields,
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            banner_text="Prediction", #shortened to fit in banner.
            banner_color=(Const.COLOR_BLUE)  # Matching neuron colors
        )

    def update_me(self):
        # Extract data from query result (mock data for now)
        rs_iteration    = Const.dm.get_model_iteration_data (self.model_id)
        rs_epoch        = Const.dm.get_model_epoch_data     (self.model_id)
        loss_gradient   = smart_format(rs_iteration.get("loss_gradient", 0.0))
        error           = rs_iteration.get("error", 0.0)
        avg_error       = rs_epoch.get("mean_absolute_error", 0.0)
        target          = rs_iteration.get("target", 0.0)
        prediction      = rs_iteration.get("prediction", 0.0)
        prediction_raw  = rs_iteration.get("prediction_raw", 0.0)        #print(f"Prediction = {prediction}\t Prediction Raw={prediction_raw}")

        if self.problem_type == "Binary Decision":
            predictions = f"{smart_format(prediction_raw)} => {smart_format(prediction)} "
            if abs(prediction - target) < 0.000001:
                self.banner_text = "Correct"
                self.set_colors(1)
            else:
                self.banner_text = "Wrong"
                self.set_colors(0)
        else:                                               # not binary decision
            predictions = f"{smart_format(prediction_raw)}"

        # Update the form fields
        self.fields["Prediction"] = predictions
        #self.fields["Error/Avg Epoch"] = f"{smart_format(error)} / {smart_format(avg_error)}"
        self.fields["Error"] = smart_format(error)
        self.fields[self.target_name] = smart_format(target)
        self.fields["Loss Gradient"] = smart_format(loss_gradient)

        if avg_error >= 1000:
            self.fields["Epoch Avg Error"] = f"{avg_error:,.0f}"
        else:
            self.fields["Epoch Avg Error"] = f"{avg_error:.9f}"
