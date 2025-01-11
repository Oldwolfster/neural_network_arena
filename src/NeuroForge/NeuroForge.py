import pygame
import sys
from src.ArenaSettings import HyperParameters

from src.engine.Neuron import Neuron
from src.engine.RamDB import RamDB






#from src.NeuroForge.mgr import screen
from src.NeuroForge.mgr import * # Imports everything into the local namespace
from src.NeuroForge import mgr # Keeps the module reference for assignments


def NeuroForge(db : RamDB, training_data, hyper : HyperParameters):
    # Initialize Pygame

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

        # Update Game State (later you'll update weights, activations, etc.)

        # Clear the screen
        mgr.screen.fill((255, 255, 255))  # White background

        # Draw Elements (neurons, lines, etc.)
        pygame.draw.circle(mgr.screen, (0, 0, 255), (400, 300), 20)  # Example neuron
        draw_neuron()

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Quit Pygame
    pygame.quit()
    sys.exit()

def draw_neuron():
    # Draw Neuron (border only)
    neuron_rect = pygame.Rect(150, 100, 50, 50)  # x, y, width, height
    pygame.draw.rect(mgr.screen, sky_blue, neuron_rect, width=3)
    # Add Text Inside Neuron
    text_surface = mgr.font.render("0.87", True, black)  # Activation value as text
    text_rect = text_surface.get_rect(center=neuron_rect.center)  # Center text in the neuron
    mgr.screen.blit(text_surface, text_rect)

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

def neuro_forge_init():
    pygame.init()
    # Font Setup
    mgr.font = pygame.font.Font(None, 24)  # Default font, size 24
    # Screen Settings
    screen_width, screen_height = 800, 600
    mgr.screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Neural Network Visualization")
