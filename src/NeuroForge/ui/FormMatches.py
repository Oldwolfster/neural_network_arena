from typing import List
from enum import Enum, auto
import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UITextEntryLine, UIDropDownMenu
from src.NeuroForge import Const
from src.NeuroForge.EZForm import EZForm
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.ui.HoloPanel import HoloPanel

class Action(Enum): #This goes at top of file
    STEP_BACK_SAMPLE    = auto()
    STEP_FORWARD_SAMPLE = auto()
    STEP_BACK_EPOCH     = auto()
    STEP_FORWARD_EPOCH  = auto()
    STEP_BACK_BIG       = auto()
    STEP_FORWARD_BIG    = auto()
    TOGGLE_PLAY         = auto()
    TOGGLE_REVERSE      = auto()
    EXIT_NEUROFORGE     = auto()







class FormMatches(EZForm):
    def __init__(self, width_pct: int, height_pct: int, left_pct: int, top_pct: int,
                    banner_text=None,
                    background_image_path=None
                    ):
        fields = { }
        super().__init__(fields, width_pct, height_pct, left_pct, top_pct,shadow_offset_x=0, banner_text=banner_text, background_image_path=background_image_path)
        self.holo_gladiators = HoloPanel(
            parent_surface=self.surface,
            title="🏛️ Gladiators",
            left_pct=5,
            top_pct=10,
            width_pct=40,
            height_pct=60
        )

    def update_me(self):
        pass


############################## BELOW HERE IS ORIGINAL CTRL FORM##############################
############################## BELOW HERE IS ORIGINAL CTRL FORM##############################
############################## BELOW HERE IS ORIGINAL CTRL FORM##############################
############################## BELOW HERE IS ORIGINAL CTRL FORM##############################




    """
        self.is_playing             = True
        self.is_reversed            = False
        self.buttons                = {}
        self.epoch_input            = None
        self.speed_dropdown         = None
        #self.create_ui_elements()


    def process_an_event(self, event):
        self.process_mouse_events(event)
        if event.type == pygame.KEYDOWN:
            self.process_keyboard_events(event)
    
    def perform_action(self, action: Action):
        if action == Action.TOGGLE_PLAY:
            self.toggle_playback()
        elif action == Action.TOGGLE_REVERSE:
            self.toggle_reverse()
        elif action == Action.STEP_BACK_SAMPLE:
            Const.vcr.step_x_iteration(-1, True)
        elif action == Action.STEP_FORWARD_SAMPLE:
            Const.vcr.step_x_iteration(1, True)
        elif action == Action.STEP_BACK_EPOCH:
            Const.vcr.step_x_epochs(-1, True)
        elif action == Action.STEP_FORWARD_EPOCH:
            Const.vcr.step_x_epochs(1, True)
        elif action == Action.STEP_BACK_BIG:
            Const.vcr.step_x_epochs(-100, True)
        elif action == Action.STEP_FORWARD_BIG:
            Const.vcr.step_x_epochs(100, True)
        elif action == Action.EXIT_NEUROFORGE:
            Const.IS_RUNNING = False

    def handle_key(self, key_function=None):
        if self.is_playing:
            self.toggle_playback()
        elif key_function:
            key_function()

    def process_keyboard_events(self, event):
        self.handle_enter_key(event)

        key_map = {
            pygame.K_q:         Action.STEP_BACK_SAMPLE,
            pygame.K_w:         Action.STEP_FORWARD_SAMPLE,
            pygame.K_a:         Action.STEP_BACK_EPOCH,
            pygame.K_s:         Action.STEP_FORWARD_EPOCH,
            pygame.K_z:         Action.STEP_BACK_BIG,
            pygame.K_x:         Action.STEP_FORWARD_BIG,
            pygame.K_TAB:         Action.TOGGLE_PLAY,
            pygame.K_e:       Action.TOGGLE_REVERSE,
            pygame.K_ESCAPE:    Action.EXIT_NEUROFORGE
        }

        action = key_map.get(event.key)
        if action:
            self.perform_action(action)

    def process_mouse_events(self, event):
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            for action, button in self.buttons.items():
                if event.ui_element == button:
                    self.perform_action(action)
                    break

        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == self.speed_dropdown:
            self.set_playback_speed(self.speed_dropdown.selected_option)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            Const.vcr.play()
            self.buttons[Action.TOGGLE_PLAY].set_text("Pause")
            self.fields["Current Mode"] = "Playing"
        else:
            Const.vcr.pause()
            self.buttons[Action.TOGGLE_PLAY].set_text("Play")
            self.fields["Current Mode"] = "Paused"

    def toggle_reverse(self):
        self.is_reversed = not self.is_reversed
        Const.vcr.reverse()
        self.buttons[Action.TOGGLE_REVERSE].set_text("Forward" if self.is_reversed else "Reverse")

    def set_playback_speed(self, speed: str):
        Const.vcr.advance_by_epoch = 1
        if speed[0] == "Iteration":
            Const.vcr.advance_by_epoch = 0
            Const.vcr.set_speed(1)
            return
        try:
            new_speed = int(speed[0].replace("x", ""))
            Const.vcr.set_speed(new_speed)
            self.epoch_input.set_text("")
        except ValueError:
            pass
    
    def handle_enter_key(self, event):
        if not (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
            return
        epoch_text = self.epoch_input.get_text().strip()
        if epoch_text:
            Const.vcr.jump_to_epoch(epoch_text)
            self.epoch_input.set_text("")
        else:
            self.epoch_input.focus()

    def create_ui_elements(self):
        panel_x, panel_y = self.left, self.top
        button_height, button_width = 25, 69
        row_offsets = [80, 102, 124, 146]
        x_offsets = [8, 77]

        tooltips = {
            Action.TOGGLE_PLAY: "(TAB) Play/Pause",
            Action.TOGGLE_REVERSE: "(E) Reverse Playback",
            Action.STEP_BACK_SAMPLE: "(Q) Step Back Sample",
            Action.STEP_FORWARD_SAMPLE: "(W) Step Forward Sample",
            Action.STEP_BACK_EPOCH: "(A) Step Back Epoch",
            Action.STEP_FORWARD_EPOCH: "(S) Step Forward Epoch",
            Action.STEP_BACK_BIG: "(Z) Back 100 Epochs",
            Action.STEP_FORWARD_BIG: "(X) Forward 100 Epochs",
        }

        self.speed_dropdown = UIDropDownMenu(
            options_list=["Iteration", "0.5x", "1x", "2x", "4x", "10x", "25x", "50x"],
            starting_option="1x",
            relative_rect=pygame.Rect((panel_x + x_offsets[0], panel_y + 49), (138, 32)),
            manager=Const.UI_MANAGER,
        )

        def add_button(action, label, x_index, row):
            self.buttons[action] = UIButton(
                relative_rect=pygame.Rect(
                    (panel_x + x_offsets[x_index], panel_y + row_offsets[row]),
                    (button_width, button_height),
                ),
                text=label,
                manager=Const.UI_MANAGER,
                tool_tip_text=tooltips.get(action, "")
            )

        add_button(Action.TOGGLE_PLAY, "Pause", 0, 0)
        add_button(Action.TOGGLE_REVERSE, "Reverse", 1, 0)
        add_button(Action.STEP_BACK_SAMPLE, "<", 0, 1)
        add_button(Action.STEP_FORWARD_SAMPLE, ">", 1, 1)
        add_button(Action.STEP_BACK_EPOCH, "<<", 0, 2)
        add_button(Action.STEP_FORWARD_EPOCH, ">>", 1, 2)
        add_button(Action.STEP_BACK_BIG, "<<<<", 0, 3)
        add_button(Action.STEP_FORWARD_BIG, ">>>>", 1, 3)

        self.epoch_input = UITextEntryLine(
            relative_rect=pygame.Rect((panel_x + x_offsets[0], panel_y + 169), (138, 36)),
            manager=Const.UI_MANAGER,
        )
        return
    """
