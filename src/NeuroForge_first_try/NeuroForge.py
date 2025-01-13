import pygame
import sys
from src.ArenaSettings import HyperParameters
from typing import List

from src.NeuroForge.ObjectPlacer import ObjectPlacer, populate_model_info
from src.engine.Utils_DataClasses import ModelInfo
from src.engine.Neuron import Neuron
from src.engine.RamDB import RamDB
#from src.NeuroForge.mgr import screen
from src.NeuroForge.mgr import * # Imports everything into the local namespace
from src.NeuroForge import mgr # Keeps the module reference for assignments


def NeuroForge(db : RamDB, training_data, hyper : HyperParameters,  model_info_list: List[ModelInfo]):
    neuro_forge_init()
    models = populate_model_info(model_info_list)


    # Clock for controlling the frame rate
    clock = pygame.time.Clock()
    db.list_tables()


    # Game Loop
    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update Game State (later you'll update weights, activations, etc.)

        # Clear the screen
        mgr.screen.fill((255, 255, 255))  # White background

        # Draw Elements (neurons, lines, etc.)
        pygame.draw.circle(mgr.screen, (0, 0, 255), (400, 300), 20)  # Example neuron



        # Draw each model
        for i, model in enumerate(models):

            #sql = f"Select * from Iteration  and model_id='{model.info.model_id}'"
            sql = "Select * from neuron Where epoch_n =1 and iteration_n=1"
            print(f"Model={model}\tSQL={sql}")

            db.query_print(sql)
            # Draw connections
            for start, end in model.connections:
                pygame.draw.line(mgr.screen, (0, 0, 0), start, end, 2)  # Black lines for connections

            # Draw neurons
            for layer_index, neurons in model.neuron_positions.items():
                for x, y in neurons:
                    pygame.draw.circle(mgr.screen, (135, 206, 235), (x, y), 15, 3)  # Blue neuron outline
                    # Add labels (activation values or identifiers)
                    text_surface = mgr.font.render(f"N{layer_index}", True, (0, 0, 0))  # Black text
                    text_rect = text_surface.get_rect(center=(x, y))
                    mgr.screen.blit(text_surface, text_rect)


        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Quit Pygame
    pygame.quit()
    sys.exit()

def draw_inputs(inputs, screen, font, spacing=50, x_offset=100, y_start=100):
    """
    Draw the shared input layer.
    """
    for i, value in enumerate(inputs):
        neuron_rect = pygame.Rect(x_offset, y_start + i * spacing, 40, 40)  # 40x40 square
        pygame.draw.rect(screen, (200, 200, 200), neuron_rect, 0)  # Gray filled square
        pygame.draw.rect(screen, (0, 0, 0), neuron_rect, 3)  # Black border
        text_surface = font.render(str(value), True, (0, 0, 0))  # Input value
        text_rect = text_surface.get_rect(center=neuron_rect.center)
        screen.blit(text_surface, text_rect)


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

def draw_sample():
    # Draw Neuron (border only)
    neuron_rect = pygame.Rect(150, 100, 50, 50)  # x, y, width, height
    pygame.draw.rect(mgr.screen, sky_blue, neuron_rect, width=3)
    # Add Text Inside Neuron
    text_surface = mgr.font.render("0.87", True, black)  # Activation value as text
    text_rect = text_surface.get_rect(center=neuron_rect.center)  # Center text in the neuron
    mgr.screen.blit(text_surface, text_rect)

"""