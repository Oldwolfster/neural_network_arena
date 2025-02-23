import pygame
import pygame_gui
from src.NeuroForge import Const
from src.NeuroForge.DisplayManager import DisplayManager
from src.NeuroForge.VCR import VCR

from src.UI.Menus import create_menu
from src.engine.ModelConfig import ModelConfig
from src.engine.RamDB import RamDB
from src.ArenaSettings import HyperParameters
from src.engine.Utils_DataClasses import ModelInfo
from typing import List

def neuroForge(config: ModelConfig, model_info_list: List[ModelInfo]):
    """Initialize NeuroForge and run the visualization loop."""
    pygame.init()
    Const.SCREEN = pygame.display.set_mode((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Network Visualization")

    Const.UI_MANAGER = pygame_gui.UIManager((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))
    display_manager = DisplayManager(config, model_info_list)
    vcr = VCR()  # Handles event processing
    menu = create_menu(Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT, config.db)  # Create UI menu

    clock = pygame.time.Clock()
    running = True

    while running:
        print("runinng")
        time_delta = clock.tick(60) / 1000.0  # Convert to seconds
        events = pygame.event.get()

        # Process user inputs
        for event in events:
            running = vcr.process_event(event)
            Const.UI_MANAGER.process_events(event)

        # Update display components
        display_manager.update()
        Const.UI_MANAGER.update(time_delta)

        # Render
        Const.SCREEN.fill(Const.COLOR_WHITE)
        display_manager.render()
        Const.UI_MANAGER.draw_ui(Const.SCREEN)

        if Const.MENU_ACTIVE:
            menu.update(events)
            menu.draw(Const.SCREEN)

        pygame.display.flip()
