import pygame
import pygame_gui
from src.NeuroForge import Const
from src.NeuroForge.Display_Manager import Display_Manager
from src.NeuroForge.VCR import VCR

from src.UI.Menus import create_menu
from src.engine.Config import Config
from src.engine.RamDB import RamDB
from src.ArenaSettings import HyperParameters
from src.engine.Utils_DataClasses import ModelInfo
from typing import List
from src.NeuroForge.ButtonMenu import ButtonMenu, ButtonInfo


def neuroForge(configs:  List[Config]):
    """Initialize NeuroForge and run the visualization loop."""
    pygame.init()
    Const.SCREEN = pygame.display.set_mode((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Network Visualization")

    Const.UI_MANAGER = pygame_gui.UIManager((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))
    Const.vcr = VCR()  # Handles event processing
    Const.set_vcr_instance(Const.vcr)

    Const.dm = Display_Manager(configs)  # Assign `DisplayManager` directly

    menu = create_menu(Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT, configs[0].db)  # Create UI menu
    menu_button = ButtonMenu()
    info_button = ButtonInfo()

    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(120) / 1000.0  # Convert to seconds        #print(clock.get_fps())  # See FPS impact capped at 60
        events = pygame.event.get()

        # Process user inputs
        for event in events:
            Const.UI_MANAGER.process_events(event)
            Const.dm.process_events(event)
            menu_button.handle_event(event)

        # Update display components
        Const.dm.update()
        Const.UI_MANAGER.update(time_delta)

        # Render
        Const.SCREEN.fill(Const.COLOR_WHITE)
        Const.dm.render()
        Const.UI_MANAGER.draw_ui(Const.SCREEN)
        menu_button.draw()
        info_button.draw()
        Const.dm.render_pop_up_window()

        if Const.MENU_ACTIVE and menu.is_enabled():
            menu.update(events)
            menu.draw(Const.SCREEN)

        pygame.display.flip()

