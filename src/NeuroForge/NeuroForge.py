import pygame
import sys
from src.ArenaSettings import HyperParameters
from typing import List

from src.NeuroForge.Display_Mgr import DisplayManager
from src.engine.Utils_DataClasses import ModelInfo

from src.engine.RamDB import RamDB
#from src.NeuroForge.mgr import screen
from src.NeuroForge.mgr import * # Imports everything into the local namespace
from src.NeuroForge import mgr # Keeps the module reference for assignments
import tkinter.messagebox as mb
def NeuroForge(db: RamDB, training_data, hyper: HyperParameters, model_info_list: List[ModelInfo]):
    neuro_forge_init()
    #display_models = create_display_models(model_info_list)
    display_manager = DisplayManager(mgr.screen, hyper, db)
    display_manager.initialize(model_info_list)  # Set up all components
    #screen_width, screen_height = pygame.display.get_surface().get_size()

    # Initialize tracking variables
    last_iteration = mgr.iteration -1 # -1 makes it trigger it the first time.
    last_epoch = mgr.epoch
    #update_all(db, mgr.iteration, mgr.epoch, display_models, mgr.screen)

    display_manager.update( db, mgr.iteration, mgr.epoch,'global no model')
    # Pygame main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                respond_to_UI(event)

        #print (f"mgr.epoch{mgr.epoch}\tmgr.iteration{mgr.iteration}\t")
        # Check if iteration or epoch has changed
        if mgr.iteration != last_iteration or mgr.epoch != last_epoch:
            # Query and update data
            display_manager.update( db, mgr.iteration, mgr.epoch,'global no model')

            # Update tracking variables
            last_iteration = mgr.iteration
            last_epoch = mgr.epoch

        # Render models
        mgr.screen.fill((255, 255, 255))  # Clear screen
        display_manager.render()
        pygame.display.flip()

def respond_to_UI(event):
    if event.key == pygame.K_l:  # Advance frame
        mgr.iteration += 1
        if mgr.iteration > mgr.max_iteration:
            mgr.epoch += 1
            mgr.iteration = 1

    if event.key == pygame.K_j:  # Reverse frame
        mgr.iteration -= 1
        if mgr.iteration == 0:
            mgr.epoch -= 1
            mgr.iteration = mgr.max_iteration

    if event.key == pygame.K_o:  # Advance epoch
        mgr.epoch += 1
        mgr.iteration = 1  # Reset to the first iteration of the new epoch

    if event.key == pygame.K_u:  # Reverse epoch
        mgr.epoch -= 1
        mgr.iteration = 1  # Reset to the first iteration of the new epoch

    # Check for out of range conditions
    if mgr.epoch < 1:  # Check if trying to move past the beginning
        mb.showinfo("Out of Range", "You cannot go past the first epoch!")
        mgr.epoch = 1
        mgr.iteration = 1

    if mgr.epoch > mgr.max_epoch:  # Check if trying to move past the end
        mb.showinfo("Out of Range", "You are at the end!")
        mgr.epoch = mgr.max_epoch
        mgr.iteration = mgr.max_iteration

def neuro_forge_init():
    pygame.init()
    # Font Setup
    mgr.font = pygame.font.Font(None, 24)  # Default font, size 24
    # Screen Settings
    screen_width, screen_height = 1200, 900
    mgr.screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Neural Network Visualization")
