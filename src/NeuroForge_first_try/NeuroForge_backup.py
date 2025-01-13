import pygame
import sys
from src.ArenaSettings import HyperParameters
from typing import List
from src.engine.Utils_DataClasses import ModelInfo
from src.engine.Neuron import Neuron
from src.engine.RamDB import RamDB
#from src.NeuroForge.mgr import screen
from src.NeuroForge.mgr import * # Imports everything into the local namespace
from src.NeuroForge import mgr # Keeps the module reference for assignments

class ModelVisualization:
    def __init__(self, info: ModelInfo, canvas_width=400, canvas_height=300):
        self.info = info
        self.neuron_positions = {}  # To store neuron positions, e.g., {layer: [(x, y), ...]}
        print(f"Neuron positions for model {self.info.model_id}: {self.neuron_positions}")
        self.connections = []  # To store connections, e.g., [((x1, y1), (x2, y2)), ...]
        self.canvas = pygame.Surface((canvas_width, canvas_height))  # Dedicated canvas for this model
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def calculate_neuron_positions(self):
        """
        Calculate positions for neurons in each layer based on the architecture.
        """
        layer_spacing = self.canvas_width // (len(self.info.full_architecture) + 1)
        neuron_spacing = self.canvas_height // max(self.info.full_architecture)

        for layer_index, neuron_count in enumerate(self.info.full_architecture):
            layer_x = layer_spacing * (layer_index + 1)
            self.neuron_positions[layer_index] = [
                (layer_x, neuron_spacing * (i + 1)) for i in range(neuron_count)
            ]

    def calculate_connections(self):
        """
        Generate connection positions between neurons in adjacent layers.
        """
        self.connections = []  # Clear existing connections
        for layer_index in range(len(self.neuron_positions) - 1):
            current_layer = self.neuron_positions[layer_index]
            next_layer = self.neuron_positions[layer_index + 1]
            for neuron_start in current_layer:
                for neuron_end in next_layer:
                    self.connections.append((neuron_start, neuron_end))



def populate_model_info(model_info_list: List[ModelInfo]) -> List[ModelVisualization]:
    rendering_details_per_model = []
    for info in model_info_list:
        single_model_rendering_details = ModelVisualization(info)
        single_model_rendering_details.calculate_neuron_positions()
        single_model_rendering_details.calculate_connections()
        rendering_details_per_model.append(single_model_rendering_details)


    return rendering_details_per_model

def draw_model_canvas(model: ModelVisualization, font, screen):
    """
    Draw the neurons and connections for a single model on the main screen.
    """
    # Draw connections
    for start, end in model.connections:
        pygame.draw.line(screen, (0, 0, 0), start, end, 2)  # Black lines for connections

    # Draw neurons
    for layer_index, neurons in model.neuron_positions.items():
        for neuron_index, (x, y) in enumerate(neurons):
            neuron_rect = pygame.Rect(x - 20, y - 20, 40, 40)  # Square centered at (x, y)
            pygame.draw.rect(screen, (135, 206, 235), neuron_rect, 0)  # Filled blue square
            pygame.draw.rect(screen, (0, 0, 0), neuron_rect, 3)  # Black border

            # Add details inside neuron
            activation = model.activations.get(layer_index, [])[neuron_index]
            weight = model.weights.get(layer_index, [])[neuron_index]
            text_surface = font.render(f"A: {activation:.2f}", True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=neuron_rect.center)
            screen.blit(text_surface, text_rect)


def draw_neuron():
    # Draw Neuron (border only)
    neuron_rect = pygame.Rect(150, 100, 50, 50)  # x, y, width, height
    pygame.draw.rect(mgr.screen, sky_blue, neuron_rect, width=3)
    # Add Text Inside Neuron
    text_surface = mgr.font.render("0.87", True, black)  # Activation value as text
    text_rect = text_surface.get_rect(center=neuron_rect.center)  # Center text in the neuron
    mgr.screen.blit(text_surface, text_rect)

def NeuroForge(db: RamDB, training_data, hyper: HyperParameters, model_info_list: List[ModelInfo]):
    models = populate_model_info(model_info_list)
    print("Populated Models:")
    for model in models:
        print(f"Model ID: {model.info.model_id}, Architecture: {model.info.full_architecture}")
    neuro_forge_init()

    # Clock for controlling the frame rate
    clock = pygame.time.Clock()

    # Game Loop
    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the main screen
        mgr.screen.fill((255, 255, 255))  # White background

        # Render each model
        for i, model in enumerate(models):
            model.canvas.fill((220, 220, 220))  # Light gray background
            draw_model_canvas(model)

            # Blit model canvas to the main screen
            x_position = (mgr.screen.get_width() - model.canvas.get_width()) // 2
            y_position = i * (model.canvas.get_height() + 20)  # Stack vertically
            mgr.screen.blit(model.canvas, (x_position, y_position))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Quit Pygame
    pygame.quit()
    sys.exit()


def neuro_forge_init():
    pygame.init()
    # Font Setup
    mgr.font = pygame.font.Font(None, 24)  # Default font, size 24
    # Screen Settings
    screen_width, screen_height = 800, 600
    mgr.screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Neural Network Visualization")



"""
pygame.draw.circle(screen, (0, 0, 255), (200, 300), 20)  # Input Neuron
pygame.draw.line(screen, (0, 255, 0), (220, 300), (380, 300), 2)  # Weight Line
pygame.draw.circle(screen, (255, 0, 0), (400, 300), 20)  # Perceptron

THINKNESS:
    weight = 0.8
    color = (0, 0, int(255 * weight)) if weight > 0 else (int(-255 * weight), 0, 0)
    thickness = int(5 * abs(weight))  # Scale thickness with magnitude
    pygame.draw.line(screen, color, (220, 300), (380, 300), thickness)

DROPDOWN FOR TIME SCALE
    granularity_options = ["Iteration", "Epoch", "Layer", "Neuron"]
    current_option = "Iteration"
    
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if dropdown_clicked:  # Detect click on dropdown
                current_option = selected_option  # Update based on user choice
    
    
    ZOOMING NN AND NOT ZOOMING CONTROLS
"""