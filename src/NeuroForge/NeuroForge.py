import pygame
import pygame_gui
from src.NeuroForge import Const

#from src.DisplayManager import DisplayManager
#from src.VCR import VCR
from src.UI.Menus import create_menu
from src.engine.RamDB import RamDB
from src.ArenaSettings import HyperParameters
from src.engine.Utils_DataClasses import ModelInfo
from typing import List

def neuroForge(db: RamDB, training_data, hyper: HyperParameters, model_info_list: List[ModelInfo]):
    """Initialize NeuroForge and run the visualization loop."""
    pygame.init()
    Const.SCREEN = pygame.display.set_mode((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Network Visualization")

    ui_manager = pygame_gui.UIManager((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))
    #display_manager = DisplayManager(Const.SCREEN, hyper, db, model_info_list, ui_manager)
    #vcr = VCR()  # Handles event processing
    menu = create_menu(Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT, db)  # Create UI menu

    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(60) / 1000.0  # Convert to seconds
        events = pygame.event.get()

        # Process user inputs
        #running = vcr.process_events(events)
        #ui_manager.process_events(events)

        # Update display components
        #display_manager.update()
        ui_manager.update(time_delta)

        # Render
        Const.SCREEN.fill(Const.COLOR_WHITE)
        #display_manager.render()
        ui_manager.draw_ui(Const.SCREEN)

        if menu.is_enabled():
            menu.update(events)
            menu.draw(Const.SCREEN)

        pygame.display.flip()
