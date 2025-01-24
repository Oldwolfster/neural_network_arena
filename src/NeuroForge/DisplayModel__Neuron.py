import json
from typing import List
import pygame

from src.NeuroForge import mgr
from src.NeuroForge.EZPrint import EZPrint
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format
from src.NeuroForge.mgr import * # Imports everything into the local namespace


class DisplayNeuron:
    def __init__(self, nid: int):
        self.location_left=0
        self.location_top=0
        self.location_width=0
        self.location_height = 0
        self.nid = 0
        self.label="" #need to define, try to use existing standard
        self.layer = 0
        self.weights = []
        self.bias = 0
        self.weight_count = []
        self.weight_formula_txt = ""
        self.raw_sum = 0
        self.activation_function = ""
        self.activation_value =0
        self.weight_text = ""
        # Create EZPrint instance
        self.ez_printer = EZPrint(mgr.font, color=(0, 0, 0), max_width=200, max_height=100, sentinel_char="\n")


    def draw_me(self, screen):
        # Draw the neuron rectangle
        pygame.draw.rect(
            screen,
            (0, 0, 255),  # Blue color
            (self.location_left, self.location_top, self.location_width, self.location_height),
            3  # Border width
        )

        #print(f"nid={self.nid}\tself.weight_text= {self.weight_text}")

        # Draw the weight text in the top-left corner of the rectangle
        if hasattr(self, "weight_text") and self.weight_text:  # Check if weight_text exists and is not empty
            #font = pygame.font.Font(None, 18)  # Font and size
            #rendered_text = mgr.font.render(self.weight_text, True, (0, 0, 0))  # Render text in black
            #screen.blit(rendered_text, (self.location_left + 5, self.location_top + 5))  # Add slight padding
            self.ez_printer.render(screen,text= self.weight_text,x=30,y=11)


    def update_me(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        # Parameterized query with placeholders
        SQL =   """
            SELECT  *
            FROM    Iteration I
            JOIN    Neuron N
            ON      I.model_id  = N.model 
            AND     I.epoch     = N.epoch_n
            AND     I.iteration = N.iteration_n
            WHERE   model = ? AND iteration_n = ? AND epoch_n = ? AND nid = ?
            ORDER BY epoch, iteration, model, nid 
        """
        params = (model_id, iteration, epoch, self.nid)

        # Debugging SQL and parameters
        #print(f"SQL in update_me: {SQL}")
        #print(f"Params: {params}")

        # Execute query
        rs = db.query(SQL, params)
        self.weight_text = self.neuron_report_build_prediction_logic(rs[0])
        #print(f"Query result: {rs}")
        #print(f"PREDICTIONS: {self.weight_text}")



    def neuron_report_build_prediction_logic(self,row):
        """
        Build prediction logic for a single neuron (row).
        Loops through weights and inputs, generating labeled calculations.
        """
        nid = row.get('nid')  # Get neuron ID
        weights = json.loads(row.get('weights_before', '[]'))  # Deserialize weights
        inputs = json.loads(row.get('inputs', '[]'))  # Deserialize inputs

        # Validate lengths of weights and inputs
        if len(weights) != len(inputs):
            raise ValueError(f"Mismatch in length of weights ({len(weights)}) and inputs ({len(inputs)})")

        # Generate prediction logic
        predictions = []
        self.raw_sum = 0
        for i, (w, inp) in enumerate(zip(weights, inputs), start=1):
            #label = f"W{i}I{i}"  # Update label to match new specs
            linesum= (w*inp)
            calculation = f"{smart_format(w)} * {smart_format(inp)} = {smart_format(w * inp)}"
            predictions.append(calculation)
            self.raw_sum += linesum

        # Combine multi-line predictions into a single string
        #return "\n".join(predictions)
        return f"{'\n'.join(predictions)}\n{self.format_bias_and_raw_sum(row)}"

    def format_bias_and_raw_sum(self, row):
        """
        Format the bias and raw sum for display.

        Args:
            row (dict): A dictionary representing a single neuron instance.

        Returns:
            str: Nicely formatted string with bias and raw sum.
        """
        # Extract bias and raw sum from the dictionary
        bias = row.get('bias', 0)
        self.raw_sum +=bias

        # Format the values
        bias_str = f"Bias: {smart_format(bias)}"
        raw_sum_str = f"Raw Sum (z): {smart_format(self.raw_sum)}"

        # Combine them into a single string
        return f"{bias_str}\n{raw_sum_str}"

