from typing import List

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UITextEntryLine, UIDropDownMenu

from src.neuroForge import mgr
from src.neuroForge.EZForm import EZForm


class DisplayPanelCtrl(EZForm):
    """
    UI Control Panel for managing playback, jumping to epochs, and speed control.
    Inherits from EZForm to maintain consistent UI styling.
    """
    def __init__(self, screen: pygame.Surface, ui_manager: pygame_gui.ui_manager, width_pct: int, height_pct: int, left_pct: int, top_pct: int):
        self.ui_manager = ui_manager
        # Define UI fields for display purposes
        fields = {
            "Playback Speed": "1x",
            "Jump to Epoch": "",
            "Current Mode": "Paused"
        }

        # Initialize the parent class (EZForm) to maintain styling
        super().__init__(screen,  fields, width_pct, height_pct, left_pct, top_pct, banner_text="Controls")

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
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 100), (68, 30)),
            text="Play",
            manager=self.ui_manager
        )

        # Pause button
        self.pause_button = UIButton(
            relative_rect=pygame.Rect((panel_x + 71, panel_y + 100), (68, 30)),
            text="Pause",
            manager=self.ui_manager
        )

        # Epoch Jump Input Box
        self.epoch_input = UITextEntryLine(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 169), (138, 36)),
            manager=self.ui_manager
        )

        # Speed Dropdown (1x, 2x, 4x, etc.)
        self.speed_dropdown = UIDropDownMenu(
            options_list=["0.5x", "1x", "2x", "4x", "10x"],
            starting_option="1x",
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 62), (138, 36)),
            manager=self.ui_manager
        )

    def update_me(self, rs: dict):
        """
        Updates the display panel UI fields dynamically.
        """
        # Example of updating displayed values dynamically
        #self.fields["Playback Speed"] = self.speed_dropdown.selected_option
        #self.fields["Current Mode"] = "Playing" if self.is_playing() else "Paused"

    def process_an_event(self, event: pygame.event.Event):
        """
        Processes UI events (button clicks, text entry, dropdown selection).
        Also ensures pygame_gui receives events.
        """
        #print(f"in process_an_event, event={event}")


        self.handle_enter_key(event)

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.play_button:
                self.toggle_playback(True)
            elif event.ui_element == self.pause_button:
                self.toggle_playback(False)

        elif event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED and event.ui_element == self.epoch_input:
            print(f"Text Changed: {self.epoch_input.get_text()}")  # Debugging

        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self.speed_dropdown:
            self.set_playback_speed(self.speed_dropdown.selected_option)

    def handle_enter_key(self, event: pygame.event.Event):
        """
        Handles 'Enter' key press for the epoch input box.
        - If empty: Moves focus to the input box.
        - If valid number: Jumps to that epoch.
        - If invalid: Displays an error message.
        """
        #check if enter key was pressed
        #if not (event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == self.epoch_input):
        if not(event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
            return      #
        epoch_text = self.epoch_input.get_text().strip()  # Get text from input box

        if epoch_text == "":
            print("Text box is empty. Moving focus to input box.")
            self.epoch_input.focus()  # ✅ Move focus to text box (no clicking needed!)
            return

        try:
            epoch_number = int(epoch_text)  # ✅ Check if it's a valid number
            #print(f"Jumping to epoch {epoch_number}")
            self.jump_to_epoch() # 🔥 Jump to epoch
            self.epoch_input.set_text("")  # ✅ Clear text box after processing

        except ValueError:
            mgr.jump_to_epoch = -1  # 🚨 Handle non-numeric input - will happen in main loop

    def jump_to_epoch(self):
        """
        Reads the epoch number from the input box and jumps to that epoch.
        """
        try:
            mgr.jump_to_epoch = int(self.epoch_input.get_text())
        except ValueError:
            pass


    def toggle_playback(self, play: bool):
            """
            Toggles between play and pause modes.

            Args:
                play (bool): If True, play; otherwise, pause.
            """
            self.fields["Current Mode"] = "Playing" if play else "Paused"
            # Implement logic to control playback in DisplayManager


    def toggle_playback(self, play: bool):
        """
        Toggles between play and pause modes.

        Args:
            play (bool): If True, play; otherwise, pause.
        """
        self.fields["Current Mode"] = "Playing" if play else "Paused"
        # Implement logic to control playback in DisplayManager



    def set_playback_speed(self, speed: str):
        """
        Updates playback speed based on dropdown selection.

        Args:
            speed (str): The selected speed (e.g., "1x", "2x").
        """
        try:
            if isinstance(speed, tuple):
                speed = speed[0]  # ✅ Extract first element if tuple
            remove_x = speed.replace("x", "")  # ✅ Remove 'x' safely
            new_speed = int(remove_x)  # ✅ Check if it's a valid number
            mgr.vcr_rate = new_speed # 🔥 Set rate to selected speed
            self.epoch_input.set_text("")  # ✅ Clear text box after processing

        except ValueError:
            pass
            #mgr.jump_to_epoch = -1  # 🚨 Handle non-numeric input - will happen in main loop

