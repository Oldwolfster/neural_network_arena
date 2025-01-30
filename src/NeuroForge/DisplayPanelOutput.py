import pygame
from src.NeuroForge.EZForm import EZForm
from src.engine.Utils import smart_format


class DisplayPanelOutput(EZForm):
    def __init__(self, screen: pygame.Surface,problem_type: str, width_pct, height_pct, left_pct, top_pct):
            self.problem_type = problem_type
            # Define the fields and default values for the form
            fields = {
                "Target": "0.000",
                "Raw Prediction": "0.000",
                "Error (Targ-Raw)": "0.000",
                "Relative Error" : "0.000",
                "Loss Function": "MSE",
                "Loss Result": "0.000",
                "Loss Gradient": "0.000",
                "Threshold": "x >= .5",
                "Final Prediction": "0,000"

            }
            self.screen = screen
            # Pass the calculated absolute dimensions to the parent class
            super().__init__(screen=self.screen, fields=fields,width_pct=width_pct,height_pct=height_pct,left_pct=left_pct,top_pct=top_pct,banner_text="Prediction",banner_color=(0, 0, 255))  # Matching neuron colorsbg_color=(240, 240, 240),font_color=(0, 0, 0)        )

    def update_me(self, rs: dict):
        # Extract data from query result
        prediction = float(rs.get("prediction", 0.0))
        prediction_raw = float(rs.get("prediction_raw", 0.0))
        target = rs.get("target", 0.0)
        raw_sum = float(rs.get("raw_sum", 0.0))  # Needed for activation gradient
        activation_name = rs.get("activation_name", "Unknown")
        #print(f"rs in output::: {rs}")
        error = float(rs.get("error", 0.0))
        loss_result = rs.get("loss", 0.0)
        loss_gradient = float(rs.get("loss_gradient"))


        # Compute Relative Error (Avoid divide by zero)
        rel_error = 0 if target == 0 else (error / target) * 100

        if self.problem_type == "Binary Decision":
            self.fields["Threshold"] = "x >= .5"

                # Update the form fields
        self.fields["Target"] = smart_format(target)
        self.fields["Raw Prediction"] = smart_format(prediction_raw)
        self.fields["Error (Targ-Raw)"] = smart_format(error)
        self.fields["Relative Error"] = f'{rel_error:4.2f}%'
        self.fields["Loss Function"] = "MSE"
        self.fields["Loss Result"] = smart_format(loss_result)
        self.fields["Loss Gradient"] = smart_format(loss_gradient)
        self.fields["Final Prediction"] = smart_format(prediction)
        """
        print(f"Activation Name = {activation_name}")
        # Compute Activation Gradient (∂Activation/∂z)
        if activation_name == "Sigmoid":
            activation_gradient = prediction_raw * (1 - prediction_raw)  # Sigmoid derivative
        elif activation_name == "Tanh":
            activation_gradient = 1 - prediction_raw**2  # Tanh derivative
        elif activation_name == "ReLU":
            activation_gradient = 1 if raw_sum > 0 else 0  # ReLU derivative
        else:
            activation_gradient = 6969  # Default to silly value that is clearly wrong

        # Compute Error Signal (δ)
        error_signal = error * activation_gradient


        
        # New Fields for Backprop
        self.fields["Activation Gradient"] = smart_format(activation_gradient)
        self.fields["Error Signal (δ)"] = smart_format(error_signal)
        """


