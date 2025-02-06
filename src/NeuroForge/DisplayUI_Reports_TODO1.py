import pygame
import pygame_gui

from src.neuroForge.EZSurface import EZSurface


class DisplayUI_Reports(EZSurface): # !!!WOULD BE NICE TO GIVE QUICKER ACCESS TO REPORTS
    def __init__(self, screen : pygame.Surface,  max_epoch : int, max_iteration : int, width_pct  , height_pct, left_pct, top_pct):
        super().__init__    (screen, width_pct, height_pct, left_pct, top_pct, bg_color=(0, 0, 0))
        self.banner_text    = "Loading..."
        self.max_epoch      = max_epoch
        self.max_iteration  = max_iteration
        self.manager = pygame_gui.UIManager((screen.get_width(),  screen.get_height()))  # Adjust for your window size

        # Create dropdown for selecting reports
        self.report_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["Select A Report","Evolution - Parameters vs Error", "Training Summary", "Loss Graph"],
            starting_option="Select A Report",
            relative_rect=pygame.Rect((0, 0), (200, 30)),  # Position and size
            manager=self.manager
        )

    def handle_event(self, event):
        if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            selected_report = event.text
            print(f"Selected Report: {selected_report}")  # Debugging
            self.display_report(selected_report)  # Call function to display report

    def display_report(self, report_name):
        if report_name == "Neuron Report":
            print("Displaying Neuron Report...")
            # Call NF function to show the neuron report
        elif report_name == "Training Summary":
            print("Displaying Training Summary...")
            # Call NF function to show the training summary
        elif report_name == "Loss Graph":
            print("Displaying Loss Graph...")
            # Call NF function to show the loss graph

    def update_me(self, time_delta):
        #print(f"time_delta type: {type(time_delta)}, value: {time_delta}")  # Debugging
        if isinstance(time_delta, dict):
            time_delta = 0.016  # Default to 16ms (approx. 60 FPS) if invalid
        self.manager.update(time_delta)

    def render(self):
        # 🚀 Ensure the dropdown UI is also drawn
        self.manager.draw_ui(self.surface)
