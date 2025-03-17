import json
from typing import List

import pygame

from src.neuroForge_original.EZSurface import EZSurface
from src.engine.RamDB import RamDB

from src.neuroForge_original.EZForm import EZForm
from src.engine.Utils import smart_format


class DisplayPanelInput(EZForm):
    def __init__(self, screen: pygame.Surface, data_labels: List[str], width_pct: int, height_pct: int, left_pct: int, top_pct: int, bg_color=(255, 255, 255), banner_color=(0, 0, 255)):
        # Calculate absolute dimensions from percentages
        screen_width, screen_height = screen.get_size()
        self.target_name = data_labels[-1]

        # Dynamically create fields for all input labels.
        # Move "Target Value" to the top by adding it first.
        fields = {}

         # Then add all other input fields (excluding the target label, which is assumed to be the last element)
        for label in data_labels[:-1]:
            fields[label] = "0.000"  # Default value for each input

        # Add the target field explicitly (this one is not updated during back pass)
        fields[self.target_name] = ""

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

    def update_me(self, rs: dict, epoch_data: dict):
        """Update the form fields using values from the provided dictionary."""
        # Update input fields dynamically based on the keys in the dictionary
        raw_inputs = rs.get("inputs", "[]")  # Retrieve raw inputs as a JSON-like string
        inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs

        # Use a separate counter for input fields since the ordering now includes "Target Value" at the top.
        input_index = 0
        for label in self.fields.keys():
            # Skip fields that are not part of the raw inputs
            if label in ("Target", "Target Value"):
                continue
            if input_index < len(inputs):  # Update inputs only if they exist in the list
                self.fields[label] = smart_format(float(inputs[input_index]))
            else:
                self.fields[label] = "N/A"  # Set to "N/A" if no input exists
            input_index += 1

        # Update the target value explicitly (this field is now at the bottom)
        self.fields[self.target_name] =  smart_format(rs.get("target", ""))
        #self.fields["Target Value"] = smart_format(rs.get("target", ""))
