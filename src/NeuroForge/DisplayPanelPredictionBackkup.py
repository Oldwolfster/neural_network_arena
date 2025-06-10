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
        self.target_name    = Const.TRIs[0].hyper.data_labels[-1].strip()

        # Define the fields and default values for the form
        fields = {
            self.target_name: "0.000",
            "Prediction": "0.000",
            "Error / Avg": "0.000 / 0.000",
            f"{loss_function.short_name} Gradient": "0.0",
        }

        super().__init__(
            fields=fields,
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            banner_text="Prediction",
            banner_color=Const.COLOR_BLUE
        )

    def update_me(self):
        rs_iteration    = Const.dm.get_model_iteration_data(self.model_id)
        rs_epoch        = Const.dm.get_model_epoch_data(self.model_id)

        # Extract values
        target          = rs_iteration.get("target_unscaled", 0.0)
        prediction      = rs_iteration.get("prediction", 0.0)
        prediction_raw  = rs_iteration.get("prediction_unscaled", 0.0)
        loss_gradient   = rs_iteration.get("loss_gradient", 0.0)
        error           = rs_iteration.get("error_unscaled", 0.0)
        avg_error       = rs_epoch.get("mean_absolute_error_unscaled", 0.0)

        # 🧠 Binary Decision Special Case
        if self.problem_type == "Binary Decision":
            predictions = f"{smart_format(prediction_raw)} / {smart_format(prediction)}"
            if abs(prediction - target) < 1e-6:
                self.banner_text = "Correct"
                self.set_colors(1)
            else:
                self.banner_text = "Wrong"
                self.set_colors(0)
        else:
            predictions = smart_format(prediction)

        # Update field values
        self.fields["Prediction"] = predictions
        self.fields[self.target_name] = smart_format(target)
        self.fields["Error / Avg"] = f"{smart_format(error)} / {smart_format(avg_error)}"
        self.fields[f"{self.loss_function.short_name} Gradient"] = smart_format(loss_gradient)
