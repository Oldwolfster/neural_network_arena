import pygame
from src.neuroForge.EZForm import EZForm
from src.engine.Utils import smart_format


class DisplayPanelLoss(EZForm):
    def __init__(self, screen: pygame.Surface,problem_type: str, width_pct, height_pct, left_pct, top_pct):
            self.problem_type = problem_type
            # Define the fields and default values for the form
            fields = {
                "Loss Function": "MSE",
                "Loss Value": "Who cares?",
                "Gradient Factor":"2",
                "Gradient Formula" : "Factor * Error",
                "Loss Gradient": "0.0"
            }
            self.screen = screen
            # Pass the calculated absolute dimensions to the parent class
            super().__init__(screen=self.screen, fields=fields,width_pct=width_pct,height_pct=height_pct,left_pct=left_pct,top_pct=top_pct,banner_text="Gradient",banner_color=(0, 0, 255))  # Matching neuron colorsbg_color=(240, 240, 240),font_color=(0, 0, 0)        )

    def update_me(self, rs: dict, epoch_data: dict):
        loss_function = str(rs.get("loss_function",""))
        loss_value = float(rs.get("loss", 0.0))
        loss_gradient = float(rs.get("loss_gradient", 0.0))


        # Update the form fields
        self.fields["Loss Function"] = loss_function
        self.fields["Loss Value"] = smart_format(loss_value)
        #self.fields["Gradient Formula"] = smart_format(loss_gradient)
        self.fields["Loss Gradient"] = smart_format(loss_gradient)



