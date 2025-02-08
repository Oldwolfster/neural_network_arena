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
    def __init__(self, screen: pygame.Surface, ui_manager: pygame_gui.ui_manager,  width_pct: int, height_pct: int, left_pct: int, top_pct: int):
        fields = {            "Playback Speed": "1x",            "Jump to Epoch": "",            "Current Mode": "Playing"        }
        super().__init__(screen,  fields, width_pct, height_pct, left_pct, top_pct, banner_text="Controls")
        self.ui_manager = ui_manager

        # Create interactive UI elements
        self.play_button = None
        self.reverse_button = None
        self.step_forward = None
        self.step_back = None
        self.epoch_input = None
        self.speed_dropdown = None
        self.create_ui_elements()

    def process_an_event(self, event):
        """Handles UI events and sends commands to VNR_Controller.
        Also ensures pygame_gui receives events."""
        self.handle_enter_key(event)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.play_button:
                self.toggle_playback()
            elif event.ui_element == self.reverse_button:
                self.toggle_reverse()
            elif event.ui_element == self.step_forward:
                mgr.VCR.step_x_iteration(1)
            elif event.ui_element == self.step_back:
                mgr.VCR.step_x_iteration(-1)
            elif event.ui_element == self.step_forward_big:
                mgr.VCR.step_x_epochs(100)
            elif event.ui_element == self.step_back_big:
                mgr.VCR.step_x_epochs(-100)


        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self.speed_dropdown:
            self.set_playback_speed(self.speed_dropdown.selected_option)



    def toggle_reverse(self):
        if self.reverse_button.text == "Reverse":
            mgr.VCR.reverse()  # Start playing
            self.reverse_button.set_text("Forward")  # Update button text
        else:
            mgr.VCR.reverse()  # Pause playback
            self.reverse_button.set_text("Reverse")  # Update button text


    def set_playback_speed(self, speed: str):
        """
        Updates playback speed based on dropdown selection.

        Args:
            speed (str): The selected speed (e.g., "1x", "2x").
        """
        try:
            if isinstance(speed, tuple):
                speed = speed[0]                            # ✅ Extract first element if tuple
                remove_x = speed.replace("x", "")           # ✅ Remove 'x' safely
                new_speed = int(remove_x)                   # ✅ Check if it's a valid number
            mgr.VCR.set_speed(new_speed)                   #  🔥 Set rate to selected speed)
            self.epoch_input.set_text("")                   # ✅ Clear text box after processing
        except ValueError:
            pass

    def handle_enter_key(self, event: pygame.event.Event):
        """
        Handles 'Enter' key press for the epoch input box.
        - If empty: Moves focus to the input box.
        - If valid number: Jumps to that epoch.
        """
        #check if enter key was pressed
        if not(event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
            return      #
        epoch_text = self.epoch_input.get_text().strip()  # Get text from input box

        if epoch_text == "":
            self.epoch_input.focus()  # ✅ Move focus to text box (no clicking needed!)
        else:
            print(f"Jumping to epoch {epoch_text}")
            self.epoch_input.set_text("")  # ✅ Clear text box after processing
            mgr.VCR.jump_to_epoch(epoch_text) # 🔥 Jump to epoch

    def toggle_playback(self):
        """
        Toggles between play and pause modes.

        Args:
            play (bool): If True, play; otherwise, pause.
        """
        if self.play_button.text == "Play":
            mgr.VCR.play()  # Start playing
            self.play_button.set_text("Pause")  # Update button text
            self.fields["Current Mode"] = "Playing"
        else:
            mgr.VCR.pause()  # Pause playback
            self.play_button.set_text("Play")  # Update button text
            self.fields["Current Mode"] = "Paused"

    def create_ui_elements(self):#Initializes the UI elements (buttons, text boxes, dropdowns) inside the control panel.
        panel_x, panel_y = self.left, self.top  # Panel's position on screen

        # Speed Dropdown (1x, 2x, 4x, etc.)
        self.speed_dropdown = UIDropDownMenu(
            options_list=["0.5x", "1x", "2x", "4x", "10x", "25x", "50x"],
            starting_option="1x",
            relative_rect=pygame.Rect((panel_x + 4,  panel_y + 62), (138, 30)),
            manager=self.ui_manager
        )

        # Play button
        self.play_button = UIButton(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 90), (68, 25)),
            text="Pause",
            manager=self.ui_manager
        )

        # Pause button
        self.reverse_button = UIButton(
            relative_rect=pygame.Rect((panel_x + 71, panel_y + 90), (68, 25)),
            text="Reverse",
            manager=self.ui_manager
        )

        # 'Step Forward' button
        self.step_forward = UIButton(
            relative_rect=pygame.Rect((panel_x + 71, panel_y + 117), (68, 26)),
            text=">",
            manager=self.ui_manager
        )

        # 'Step back' button
        self.step_back = UIButton(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 117), (68, 26)),
            text="<",
            manager=self.ui_manager
        )

        # 'Step Forward BIG' button
        self.step_forward_big = UIButton(
            relative_rect=pygame.Rect((panel_x + 71, panel_y + 146), (68, 26)),
            text=">>>>",
            manager=self.ui_manager
        )

        # 'Step back BIG' button
        self.step_back_big = UIButton(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 146), (68, 26)),
            text="<<<<",
            manager=self.ui_manager
        )

        # Epoch Jump Input Box
        self.epoch_input = UITextEntryLine(
            relative_rect=pygame.Rect((panel_x + 4, panel_y + 169), (138, 36)),
            manager=self.ui_manager
        )


    def update_me(self, rs: dict, epoch_data: dict):
        """
        Updates the display panel UI fields dynamically.
        """
        # Example of updating displayed values dynamically
        #self.fields["Playback Speed"] = self.speed_dropdown.selected_option
        #self.fields["Current Mode"] = "Playing" if self.is_playing() else "Paused"
