import json
from typing import List

import pygame

from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


class DisplayModel(EZSurface):
    def __init__(self, screen, width_pct=60, height_pct=80, left_pct=10, top_pct=10):
        super().__init__(screen, width_pct, height_pct, left_pct, top_pct, bg_color=(240, 240, 240))
        self.neurons = []
        self.connections = []
        self.model_id = None

    def initialize_with_model_info(self, model_info):
        self.model_id = model_info.model_id

        # Calculate layer and neuron spacing
        layer_spacing = self.width // len(model_info.full_architecture)
        neuron_size, vertical_spacing = self.calculate_dynamic_neuron_layout(model_info.full_architecture, self.height)

        self.neurons = []
        for layer_index, neuron_count in enumerate(model_info.full_architecture):
            layer_neurons = []

            # Horizontal position for the current layer
            layer_x = layer_spacing * layer_index + 20

            # Calculate vertical centering
            total_layer_height = (neuron_size + vertical_spacing) * neuron_count - vertical_spacing
            vertical_offset = (self.height - total_layer_height) // 2

            for neuron_index in range(neuron_count):
                neuron = DisplayNeuron(nid=f"{layer_index}-{neuron_index}")
                neuron.layer = layer_index

                # Set position and size
                neuron.location_left = layer_x
                neuron.location_top = vertical_offset + (neuron_size + vertical_spacing) * neuron_index
                neuron.location_width = neuron_size
                neuron.location_height = neuron_size

                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)


        # Create connections
        self.connections = []
        for layer_index in range(len(model_info.full_architecture) - 1):
            current_layer = self.neurons[layer_index]
            next_layer = self.neurons[layer_index + 1]
            for from_neuron in current_layer:
                for to_neuron in next_layer:
                    connection = DisplayConnection(from_neuron=from_neuron, to_neuron=to_neuron)
                    self.connections.append(connection)

    def render(self):
        """Draw neurons and connections on the model's surface."""
        self.clear()  # Clear the surface before rendering
        for connection in self.connections:
            connection.draw_me(self.surface)
        for layer in self.neurons:  # Iterate through each layer
            for neuron in layer:  # Iterate through each neuron in the layer
                neuron.draw_me(self.surface)

    def update_me(self, db: RamDB, iteration : int, epoch : int):
        for neuron in self.neurons:
            neuron.update_me(db, iteration, epoch, self.id)

    def calculate_dynamic_neuron_layout(self, architecture, surface_height, margin=5, max_neuron_size=1500, spacing_ratio=0.25):
        """
        Calculate neuron size and spacing to fit within the surface height.

        Parameters:
            architecture (list[int]): Number of neurons in each layer.
            surface_height (int): Height of the surface in pixels.
            margin (int): Margin around the edges of the surface.
            max_neuron_size (int): Maximum size of a single neuron.
            spacing_ratio (float): Ratio of spacing to neuron size (e.g., 0.5 means spacing is half the size).

        Returns:
            tuple: (neuron_size, vertical_spacing)
        """
        max_neurons = max(architecture)
        available_height = surface_height - (2 * margin)  # Deduct margins

        # Calculate tentative neuron size
        tentative_neuron_size = available_height // (max_neurons + (max_neurons - 1) * spacing_ratio)

        # Clamp neuron size to the maximum allowed
        neuron_size = min(tentative_neuron_size, max_neuron_size)

        # Calculate spacing based on the neuron size
        vertical_spacing = int(neuron_size * spacing_ratio)

        return neuron_size, vertical_spacing



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

    def draw_me(self, screen):
        #print(f"self.location_left{self.location_left}\tself.location_top{self.location_top}\tself.location_height{self.location_height}\tself.location_width{self.location_width}\t")
        pygame.draw.rect(
            screen,
            (0, 0, 255),  # Blue
            (self.location_left, self.location_top, self.location_width, self.location_height),
            3 #width
        )

    def update_me(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        # Parameterized query with placeholders
        sql = """
            SELECT * FROM Neuron 
            WHERE model = ? AND iteration_n = ? AND epoch_n = ? AND nid = ?
        """
        params = (model_id, iteration, epoch, self.nid)

        # Debugging SQL and parameters
        print(f"SQL in update_me: {sql}")
        print(f"Params: {params}")

        # Execute query
        rs = db.query(sql, params)
        print(f"Query result: {rs}")

        # Update attributes based on query result
        if rs:
            # Assuming rs is a dictionary or object with keys matching the database columns
            self.activation_value = rs[0].get("activation_value", self.activation_value)
            self.bias = rs[0].get("bias", self.bias)

import math

class DisplayConnection:
    def __init__(self, from_neuron, to_neuron, weight=0):
        self.from_neuron = from_neuron  # Reference to DisplayNeuron
        self.to_neuron = to_neuron      # Reference to DisplayNeuron
        self.weight = weight
        self.color = (0, 0, 0)          # Default to black, could be dynamic
        self.thickness = 1             # Line thickness, could vary by weight
        self.arrow_size = 10           # Size of the arrowhead

    def draw_me(self, screen):
        # Calculate start and end points
        start_x = self.from_neuron.location_left + self.from_neuron.location_width
        start_y = self.from_neuron.location_top + self.from_neuron.location_height // 2
        end_x = self.to_neuron.location_left
        end_y = self.to_neuron.location_top + self.to_neuron.location_height // 2

        # Draw the main connection line
        pygame.draw.line(screen, self.color, (start_x, start_y), (end_x, end_y), self.thickness)

        # Calculate arrowhead points
        angle = math.atan2(end_y - start_y, end_x - start_x)  # Angle of the connection line
        arrow_point1 = (
            end_x - self.arrow_size * math.cos(angle - math.pi / 12),
            end_y - self.arrow_size * math.sin(angle - math.pi / 12)
        )
        arrow_point2 = (
            end_x - self.arrow_size * math.cos(angle + math.pi / 12),
            end_y - self.arrow_size * math.sin(angle + math.pi / 12)
        )

        # Draw the arrowhead
        pygame.draw.polygon(screen, self.color, [(end_x, end_y), arrow_point1, arrow_point2])

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
        print(f"SQL in update_me for Inputs: {sql}")
        print(f"Params: {params}")

        # Execute query
        rs = db.query(sql, params)
        print(f"Query result: {rs}")

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

            print(f"self.input_values= {self.input_values}")

def update_all(db : RamDB, iteration : int, epoch: int, display_models : List[DisplayModel], screen : pygame.Surface):
    db.list_tables()
    for model in display_models:
        print(model.id)
        model.update_me(db, iteration, epoch)

    #if not display_globals:
    #    display_globals = DisplayGlobals(screen)