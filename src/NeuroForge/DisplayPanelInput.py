import json
from typing import List

import pygame

from src.neuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


from src.neuroForge.EZForm import EZForm
from src.engine.Utils import smart_format


class DisplayPanelInput(EZForm):
    def __init__(self, screen: pygame.Surface, data_labels: List[str], width_pct: int, height_pct: int, left_pct: int, top_pct: int, bg_color=(255, 255, 255), banner_color=(0, 0, 255)):
        # Calculate absolute dimensions from percentages
        screen_width, screen_height = screen.get_size()

        # Dynamically create fields for all input labels
        fields = {}

        for label in data_labels[:-1]:  # Exclude the last label (assuming it's the target)
            fields[label] = "0.000"  # Default value for each input
        # Add the target field explicitly
        fields["Target"] = data_labels[-1]


        # Initialize the parent class with dynamically created fields
        super().__init__(
            screen=screen,
            fields=fields,
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            banner_text="Inputs",
            banner_color=banner_color,  # Matching neuron colors
            bg_color=bg_color,
            font_color=(0, 0, 0)
        )

    def update_me(self, rs: dict):
        """Update the form fields using values from the provided dictionary."""
        # Update input fields dynamically based on the keys in the dictionary
        raw_inputs = rs.get("inputs", "[]")  # Retrieve raw inputs as a JSON-like string
        inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs

        for i, label in enumerate(self.fields.keys()):
            if label == "Target":  # Skip updating "Target" here
                continue
            if i < len(inputs):  # Update inputs only if they exist in the list
                self.fields[label] = smart_format(float(inputs[i]))
            else:
                self.fields[label] = "N/A"  # Set to "N/A" if no input exists

        # Update the target explicitly
        #target = rs.get("target", 0.0)
        #self.fields["Target"] = smart_format(float(target))

        # Debugging
        #print(f"Updated Input Panel: {self.fields}")