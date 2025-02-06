from typing import List

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UITextEntryLine, UIDropDownMenu
from src.neuroForge.EZForm import EZForm


class DisplayPanelCtrl(EZForm):
    """
    UI Control Panel for managing playback, jumping to epochs, and speed control.
    Inherits from EZForm to maintain consistent UI styling.
    """
    def __init__(self, screen: pygame.Surface, data_labels: List[str], width_pct: int, height_pct: int, left_pct: int, top_pct: int):
        # Define UI fields for display purposes
        fields = {
            "Playback Speed": "1x",
            "Jump to Epoch": "",
            "Current Mode": "Paused"
        }

        # Initialize the parent class (EZForm) to maintain styling
        super().__init__(screen, ui_manager, fields, width_pct, height_pct, left_pct, top_pct, banner_text="Controls")

        # Create interactive UI elements
        self.play_button = None
        self.pause_button = None
        self.epoch_input = None
        self.speed_dropdown = None
        self.create_ui_elements()

    def create_ui_elements(self):
        """
        Initializes the UI elements (buttons, text boxes, dropdowns) inside the control panel.
        """
        panel_x, panel_y = self.left, self.top  # Panel's position on screen

        # Play button
        self.play_button = UIButton(
            relative_rect=pygame.Rect((panel_x + 10, panel_y + 50), (80, 30)),
            text="Play",
            manager=self.ui_manager
        )

        # Pause button
        self.pause_button = UIButton(
            relative_rect=pygame.Rect((panel_x + 100, panel_y + 50), (80, 30)),
            text="Pause",
            manager=self.ui_manager
        )

        # Epoch Jump Input Box
        self.epoch_input = UITextEntryLine(
            relative_rect=pygame.Rect((panel_x + 10, panel_y + 100), (170, 30)),
            manager=self.ui_manager
        )

        # Speed Dropdown (1x, 2x, 4x, etc.)
        self.speed_dropdown = UIDropDownMenu(
            options_list=["0.5x", "1x", "2x", "4x", "10x"],
            starting_option="1x",
            relative_rect=pygame.Rect((panel_x + 10, panel_y + 150), (170, 30)),
            manager=self.ui_manager
        )

    def update_me(self, rs: dict):
        """
        Updates the display panel UI fields dynamically.
        """
        # Example of updating displayed values dynamically
        self.fields["Playback Speed"] = self.speed_dropdown.selected_option
        self.fields["Current Mode"] = "Playing" if self.is_playing() else "Paused"

    def process_an_event(self, event: pygame.event.Event):
        """
        Processes UI events (button clicks, text entry, dropdown selection).

        Args:
            event (pygame.event.Event): The event being processed.
        """
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.play_button:
                self.toggle_playback(True)
            elif event.ui_element == self.pause_button:
                self.toggle_playback(False)

        elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == self.epoch_input:
            self.jump_to_epoch()

        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self.speed_dropdown:
            self.set_playback_speed(self.speed_dropdown.selected_option)

    def toggle_playback(self, play: bool):
        """
        Toggles between play and pause modes.

        Args:
            play (bool): If True, play; otherwise, pause.
        """
        self.fields["Current Mode"] = "Playing" if play else "Paused"
        # Implement logic to control playback in DisplayManager

    def jump_to_epoch(self):
        """
        Reads the epoch number from the input box and jumps to that epoch.
        """
        try:
            epoch_number = int(self.epoch_input.get_text())
            print(f"Jumping to epoch {epoch_number}")  # Replace with actual function call
        except ValueError:
            print("Invalid epoch number")

    def set_playback_speed(self, speed: str):
        """
        Updates playback speed based on dropdown selection.

        Args:
            speed (str): The selected speed (e.g., "1x", "2x").
        """
        print(f"Setting playback speed to {speed}")  # Replace with actual logic
