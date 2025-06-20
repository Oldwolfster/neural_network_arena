import json
from typing import List
import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZFormLEFT import EZForm
from src.engine.Utils import smart_format

class DisplayPanelInput(EZForm):
    def __init__(self,  width_pct: int, height_pct: int, left_pct: int, top_pct: int, bg_color=Const.COLOR_WHITE, banner_color=Const.COLOR_BLUE):#data_labels: List[str]
        """Dynamically creates an input form for displaying model input values."""
        data_labels = Const.TRIs[0].training_data.feature_labels
        self.target_name = data_labels[-1]

        # Construct fields: Input fields first, then target value last
        fields = {label: "0.000" for label in data_labels[:-1]}  # Default value for inputs
        fields[self.target_name] = ""  # Add target value last

        #print(f"In DisplayPanelInput,width_pct= {width_pct}")
        # Initialize the parent class with dynamically created fields
        super().__init__(
            fields=fields,
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            banner_text="Sample",
            banner_color=banner_color,
            bg_color=bg_color,
            font_color=Const.COLOR_BLACK
        )

    def update_me(self):
        """Update input fields dynamically based on retrieved values."""
        rs = Const.dm.get_model_iteration_data()

        raw_inputs = rs.get("inputs_unscaled", "[]")
        try:
            inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
        except json.JSONDecodeError:
            inputs = []  # Handle malformed input gracefully

        # Iterate over form fields and update them dynamically
        input_index = 0
        for label in self.fields.keys():
            if label == self.target_name:  # Skip target value for now
                continue
            if input_index < len(inputs):
                self.fields[label] = smart_format(float(inputs[input_index]))
            else:
                self.fields[label] = "N/A"  # Handle missing inputs gracefully
            input_index += 1

        # Update the target value explicitly
        self.fields[self.target_name] = smart_format(rs.get("target_unscaled", ""))
