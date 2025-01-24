import json
from typing import List
import pygame

from src.NeuroForge.DisplayModel import DisplayModel
from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


class DisplayInputs(EZSurface):
    def __init__(self, screen : pygame.Surface, width_pct  , height_pct, left_pct, top_pct):
        super().__init__(screen, width_pct, height_pct, left_pct, top_pct, bg_color=(200, 200, 200))
        self.input_count = 2
        self.input_values = ["69", "69"]  # Default values for inputs

    def render(self):
        """Draw inputs on the surface."""
        self.clear()  # Clear surface with background color

        font = pygame.font.Font(None, 24)
        input_height = self.height // (self.input_count + 1)

        for i in range(self.input_count):
            input_rect = pygame.Rect(20, (i + 1) * input_height, self.width - 40, 30)
            pygame.draw.rect(self.surface, (255, 255, 255), input_rect)  # Input box
            pygame.draw.rect(self.surface, (0, 0, 0), input_rect, 2)  # Border

            label = font.render(f"Input {i + 1}", True, (0, 0, 0))  #labels above the box
            self.surface.blit(label, (input_rect.x, input_rect.y - 20))

            # Render the value inside the box
            #value_text = font.render(self.input_values[i], True, (0, 0, 0))
            value_text = font.render(f"{self.input_values[i]:.3f}", True, (0, 0, 0))
            value_text_rect = value_text.get_rect(center=input_rect.center)  # Center text in the box
            self.surface.blit(value_text, value_text_rect)

    def update_me(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        sql = """  
            SELECT * FROM Iteration 
            WHERE  epoch = ? AND iteration = ?  
        """
        params = (epoch, iteration)

        # Debugging SQL and parameters
        #print(f"SQL in update_me for Inputs: {sql}")
        #print(f"Params: {params}")

        # Execute query
        rs = db.query(sql, params)
        #print(f"Query result: {rs}")

        # Update attributes based on query result
        if rs:
            # Extract the inputs value and parse it
            raw_inputs = rs[0].get("inputs", self.input_values)
            try:
                # Try parsing as JSON
                self.input_values = json.loads(raw_inputs)
            except json.JSONDecodeError:
                # Fallback to safely evaluating the string as a Python object
                self.input_values = literal_eval(raw_inputs)

            #print(f"self.input_values= {self.input_values}")

