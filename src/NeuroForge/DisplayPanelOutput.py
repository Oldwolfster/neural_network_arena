import pygame

from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


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
                "Threshold": "x >= .5",
                "Final Prediction": "0,000"

            }
            self.screen = screen
            # Pass the calculated absolute dimensions to the parent class
            super().__init__(screen=self.screen, fields=fields,width_pct=width_pct,height_pct=height_pct,left_pct=left_pct,top_pct=top_pct,banner_text="Prediction",banner_color=(0, 0, 255))  # Matching neuron colorsbg_color=(240, 240, 240),font_color=(0, 0, 0)        )

    def update_me(self, rs: dict):
        # Extract data from query result
        prediction = float(rs.get("prediction", 0.0))
        prediction_raw  = float(rs.get("prediction_raw", 0.0))
        target = rs.get("target", 0.0)

        error = float(rs.get("error", 0.0))
        if target == 0: #can't divide by zero
            rel_error = 0
        else:
            rel_error = error/target*100
        loss_result = rs.get("loss", 0.0)

        # Update the form fields
        self.fields["Target"] = smart_format(target)
        self.fields["Raw Prediction"] = smart_format(prediction_raw)
        self.fields["Error (Targ-Raw)"] = smart_format(error)
        self.fields["Relative Error"] = f'{rel_error:4.2f}%'
        self.fields["Loss Function"] = "MSE"
        self.fields["Loss Result"] = smart_format(loss_result)
        self.fields["Final Prediction"] = smart_format(prediction)


        if self.problem_type == "Binary Decision":
            self.fields["Threshold"] = "x >= .5"




        # Debugging
        #print(f"Updated Output Panel: {self.fields}")
