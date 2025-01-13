import pygame
import sys
from src.ArenaSettings import HyperParameters
from typing import List

from src.NeuroForge.DisplayClasses import update_all
from src.NeuroForge.DisplayManager import DisplayManager
#from src.NeuroForge.ObjectPlacer import create_display_models
from src.engine.Utils_DataClasses import ModelInfo

from src.engine.RamDB import RamDB
#from src.NeuroForge.mgr import screen
from src.NeuroForge.mgr import * # Imports everything into the local namespace
from src.NeuroForge import mgr # Keeps the module reference for assignments

def NeuroForge(db: RamDB, training_data, hyper: HyperParameters, model_info_list: List[ModelInfo]):
    neuro_forge_init()
    #display_models = create_display_models(model_info_list)
    display_manager = DisplayManager(mgr.screen)
    display_manager.initialize(model_info_list)  # Set up all components
    #screen_width, screen_height = pygame.display.get_surface().get_size()

    # Initialize tracking variables
    last_iteration = mgr.iteration
    last_epoch = mgr.epoch
    update_all(db, mgr.iteration, mgr.epoch, display_models, mgr.screen)
    # Pygame main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:  # Advance frame
                    mgr.iteration +=1;

                if event.key == pygame.K_SEMICOLON:
                    mgr.epoch +=1;

                if event.key == pygame.K_j:  # reverse
                    mgr.iteration -=1;

                if event.key == pygame.K_h:
                    mgr.epoch -=1;


        # Check if iteration or epoch has changed
        if mgr.iteration != last_iteration or mgr.epoch != last_epoch:
            # Query and update data
            for model in display_models:
                pass#model.update_data(db, mgr.iteration, mgr.epoch)

            # Update tracking variables
            last_iteration = mgr.iteration
            last_epoch = mgr.epoch

        # Render models
        mgr.screen.fill((255, 255, 255))  # Clear screen
        display_manager.render()
        for model in display_models:
            model.draw_me(mgr.screen)  # Draw each DisplayModel

        pygame.display.flip()


def neuro_forge_init():
    pygame.init()
    # Font Setup
    mgr.font = pygame.font.Font(None, 24)  # Default font, size 24
    # Screen Settings
    screen_width, screen_height = 800, 600
    mgr.screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Neural Network Visualization")
