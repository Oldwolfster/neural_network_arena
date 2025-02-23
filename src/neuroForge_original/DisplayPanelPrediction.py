import pygame
from src.neuroForge_original.EZForm import EZForm
from src.engine.Utils import smart_format
from src.neuroForge_original import mgr
class DisplayPanelPrediction(EZForm):
    def __init__(self, screen: pygame.Surface,problem_type: str, width_pct, height_pct, left_pct, top_pct):
            self.problem_type = problem_type
            # Define the fields and default values for the form
            if problem_type == "Binary Decision":
                fields = {
                    "Target Value": "0.000",
                    "Prediction(Raw)": "0.000",
                    "With Threshold" : "0.000",

                    "Error (Targ-Raw)": "0.000",
                    "Avg for Epoch": "0.000"
                }
            else:
                fields = {
                    "Target Value": "0.000",
                    "Prediction(Raw)": "0.000",
                    "Error (Targ-Raw)": "0.000"
                }
            self.screen = screen

            super().__init__(screen=self.screen, fields=fields,width_pct=width_pct,height_pct=height_pct,left_pct=left_pct,top_pct=top_pct,banner_text="Prediction",banner_color=(0, 0, 255))  # Matching neuron colorsbg_color=(240, 240, 240),font_color=(0, 0, 0)        )

    def update_me(self, rs: dict, epoch_data: dict):
        # Extract data from query result
        #print(f"Prob {self.problem_type}")
        prediction = float(rs.get("prediction", 0.0))
        prediction_raw = float(rs.get("prediction_raw", 0.0))
        target = rs.get("target", 0.0)
        raw_sum = float(rs.get("raw_sum", 0.0))  # Needed for activation gradient
        error = float(rs.get("error", 0.0))


        if self.problem_type == "Binary Decision":
            prediction_thd = 1 if prediction > .5 else 0
            self.banner_text = "Correct" if abs(prediction_thd - target) <.000001 else "Wrong" #Handle fp errors

        # Compute Relative Error (Avoid divide by zero)
        rel_error = 0 if target == 0 else (error / target) * 100

        # Update the form fields
        self.fields["Prediction(Raw)"] = smart_format(prediction_raw)
        self.fields["Error (Targ-Raw)"] = smart_format(error)
        self.fields["Target Value"] = smart_format(target)

        avg_error =  epoch_data.get("mean_absolute_error", 0.0)
        if avg_error >= 1000:
            self.fields["Avg for Epoch"] = f"{avg_error:,.0f}"
        else:
            self.fields["Avg for Epoch"] = f"{avg_error:.9f}"



        #self.fields["Relative Error"] = f'{rel_error:4.2f}%'
        if self.problem_type == "Binary Decision":
            self.fields["With Threshold"] = smart_format(prediction_thd)