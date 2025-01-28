import pygame

from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


from src.NeuroForge.EZForm import EZForm
from src.engine.Utils import smart_format


class DisplayPanelOutput(EZForm):
    def __init__(self, screen: pygame.Surface,problem_type: str, width_pct, height_pct, left_pct, top_pct):
            # Calculate absolute dimensions from percentages
            screen_width, screen_height = screen.get_size()
            #width = int(screen_width * (width_pct / 100))
            #height = int(screen_height * (height_pct / 100))
            #left = int(screen_width * (left_pct / 100))
            #top = int(screen_height * (top_pct / 100))
            self.problem_type = problem_type
            # Define the fields and default values for the form
            fields = {
                "Target": "0.000",
                "Prediction": "0.000",
                "Error (Target-Pred)": "0.000",
                "Step Function": "N/A",
                "Step Result": "N/A",
                "Loss Function": "MSE",
                "Loss Result": "0.000"

            }
            self.screen = screen
            # Pass the calculated absolute dimensions to the parent class
            #super().__init__(screen, fields, width_pct, height_pct, left_pct, top_pct, bg_color=(173, 216, 230))  # Light blue background
            super().__init__(
                screen=self.screen,
                fields=fields,
                width_pct=width_pct,
                height_pct=height_pct,
                left_pct=left_pct,
                top_pct=top_pct,
                banner_text="Output",
                banner_color=(0, 0, 255),  # Matching neuron colors
                bg_color=(240, 240, 240),
                font_color=(0, 0, 0)
        )

    def update_me(self, rs: dict):
        # Extract data from query result
        prediction = float(rs.get("prediction", 0.0))
        target = float(rs.get("target", 0.0))
        error = float(rs.get("error", 0.0))
        loss_result = float(rs.get("loss", 0.0))

        # Update the form fields
        self.fields["Prediction"] = smart_format(prediction)
        self.fields["Error (Target-Pred)"] = smart_format(error)
        self.fields["Loss Result"] = smart_format(loss_result)
        self.fields["Target"] = smart_format(target)

        if self.problem_type == "Binary Decision":
            step_result = 1 if prediction >= 0.5 else 0
            self.fields["Step Result"] = str(step_result)
            self.fields["Step Function"] = "x >= 0.5"
        else:
            self.fields["Step Result"] = "N/A"
            self.fields["Step Function"] = "N/A"

        self.fields["Loss Function"] = "MSE"

        # Debugging
        #print(f"Updated Output Panel: {self.fields}")
