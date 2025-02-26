from typing import List
import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UITextEntryLine, UIDropDownMenu
from src.NeuroForge import Const
from src.NeuroForge.EZForm import EZForm

class DisplayPanelCtrl(EZForm):
    """
    UI Control Panel for managing playback, jumping to epochs, and speed control.
    Inherits from EZForm to maintain consistent UI styling.
    """
    def __init__(self, width_pct: int, height_pct: int, left_pct: int, top_pct: int):
        fields = {
            "Playback Speed": "1x",
            "Jump to Epoch": "",
            "Current Mode": "Playing"
        }

        super().__init__(fields, width_pct, height_pct, left_pct, top_pct, banner_text="Controls")

        self.is_playing = True  # Track playback state
        self.is_reversed = False  # Track playback direction

        # Create interactive UI elements
        self.play_button = None
        self.reverse_button = None
        self.step_forward = None
        self.step_back = None
        self.epoch_input = None
        self.speed_dropdown = None
        self.create_ui_elements()

    def process_an_event(self, event):
        """Handles UI events and sends commands to VCR.
        Also ensures pygame_gui receives events.
        """
        self.handle_enter_key(event)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.play_button:
                self.toggle_playback()
            elif event.ui_element == self.reverse_button:
                self.toggle_reverse()
            elif event.ui_element == self.step_forward:
                Const.vcr.step_x_iteration(1)
            elif event.ui_element == self.step_back:
                Const.vcr.step_x_iteration(-1)
            elif event.ui_element == self.step_forward_big:
                Const.vcr.step_x_epochs(100)
            elif event.ui_element == self.step_back_big:
                Const.vcr.step_x_epochs(-100)

        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self.speed_dropdown:
            self.set_playback_speed(self.speed_dropdown.selected_option)

    def toggle_reverse(self):
        """Toggles the playback direction (Reverse/Forward)."""
        self.is_reversed = not self.is_reversed
        Const.vcr.reverse()
        self.reverse_button.set_text("Forward" if self.is_reversed else "Reverse")

    def set_playback_speed(self, speed: str):
        """
        Updates playback speed based on dropdown selection.
        """
        try:
            new_speed = int(speed.replace("x", ""))  # Remove 'x' and convert to int
            Const.vcr.set_speed(new_speed)
            self.epoch_input.set_text("")  # Clear input box after processing
        except ValueError:
            pass  # Ignore invalid inputs

    def handle_enter_key(self, event: pygame.event.Event):
        """
        Handles 'Enter' key press for the epoch input box.
        """
        if not (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
            return
        epoch_text = self.epoch_input.get_text().strip()

        if epoch_text == "":
            self.epoch_input.focus()
        else:
            print(f"Jumping to epoch {epoch_text}")
            self.epoch_input.set_text("")  # Clear text box after processing
            Const.vcr.jump_to_epoch(epoch_text)

    def toggle_playback(self):
        """Toggles between play and pause modes."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            Const.vcr.play()
            self.play_button.set_text("Pause")
            self.fields["Current Mode"] = "Playing"
        else:
            Const.vcr.pause()
            self.play_button.set_text("Play")
            self.fields["Current Mode"] = "Paused"

    def create_ui_elements(self):
        """Initializes the UI elements (buttons, text boxes, dropdowns) inside the control panel."""
        panel_x, panel_y = self.left, self.top

        # Speed Dropdown (1x, 2x, 4x, etc.)
        self.speed_dropdown = UIDropDownMenu(
            options_list=["0.5x", "1x", "2x", "4x", "10x", "25x", "50x"],
            starting_option="1x",
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 52), (138, 35)),
            manager=Const.UI_MANAGER
        )

        # Play button
        self.play_button = UIButton(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 90), (68, 25)),
            text="Pause",
            manager=Const.UI_MANAGER
        )

        # Reverse button
        self.reverse_button = UIButton(
            relative_rect=pygame.Rect((panel_x + 71, panel_y + 90), (68, 25)),
            text="Reverse",
            manager=Const.UI_MANAGER
        )

        # Step Forward button
        self.step_forward = UIButton(
            relative_rect=pygame.Rect((panel_x + 71, panel_y + 117), (68, 26)),
            text=">",
            manager=Const.UI_MANAGER
        )

        # Step Back button
        self.step_back = UIButton(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 117), (68, 26)),
            text="<",
            manager=Const.UI_MANAGER
        )

        # Step Forward BIG button
        self.step_forward_big = UIButton(
            relative_rect=pygame.Rect((panel_x + 71, panel_y + 146), (68, 26)),
            text=">>>>",
            manager=Const.UI_MANAGER
        )

        # Step Back BIG button
        self.step_back_big = UIButton(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 146), (68, 26)),
            text="<<<<",
            manager=Const.UI_MANAGER
        )

        # Epoch Jump Input Box
        self.epoch_input = UITextEntryLine(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 169), (138, 36)),
            manager=Const.UI_MANAGER
        )

    def update_me(self):
        """
        Updates the display panel UI fields dynamically.
        """


        self.fields["Current Mode"] = "Playing" if self.is_playing else "Paused"
