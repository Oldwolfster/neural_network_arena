import pygame

from src.NeuroForge.DisplayInputs import DisplayInputs
from src.NeuroForge.DisplayModel import DisplayModel
from src.engine.RamDB import RamDB


class DisplayManager:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.components = []  # List for general EZSurface-based components
        self.models = []  # List specifically for display models

    def initialize(self, model_info_list):
        """Initialize and configure all display components."""
        # Add inputs panel
        inputs_panel = DisplayInputs(self.screen, width_pct=15, height_pct=80, left_pct=5, top_pct=10)
        self.components.append(inputs_panel)

        # Add global display panel
        #globals_panel = DisplayGlobals(self.screen, width_pct=100, height_pct=10, left_pct=0, top_pct=0)
        #self.components.append(globals_panel)

        # Create and add models
        self.models = create_display_models(self.screen, model_info_list)
        #print(f"Model list (in displaymanager==============={self.models[0].model_id}")

    def render(self):
        """Render all components on the screen."""
                # Render models
        for model in self.models:
            model.draw_me()

        # Render general components
        for component in self.components:
            #print (f"DEBUG IN DM - Component = {component}")
            component.draw_me()

    def update(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        """Render all components on the screen."""
                # Render models
        for model in self.models:
            model.update_me(db, iteration, epoch, model_id)

        # Render general components
        for component in self.components:
            #print (f"DEBUG IN DM - Component = {component}")
            component.update_me( db, iteration, epoch, model_id)
        self.render()





def create_display_models(screen: pygame.Surface, model_info_list):
    """Create DisplayModel instances based on the provided model information."""
    models = []
    for index, model_info in enumerate(model_info_list):
        # Example: Adjust positions for multiple models
        model_left = 10 + index * 300  # Spacing models horizontally
        model_top = 50

        display_model = DisplayModel(
            screen=screen,
            width_pct=80,
            height_pct=80,
            left_pct=20,
            top_pct=10
        )
        display_model.initialize_with_model_info(model_info)  # Populate model details
        models.append(display_model)

    return models
