from typing import List
import pygame
from src.NeuroForge import Const
from src.NeuroForge.DisplayBanner import DisplayBanner
from src.NeuroForge.DisplayPanelCtrl import DisplayPanelCtrl
from src.NeuroForge.DisplayPanelInput import DisplayPanelInput
from src.NeuroForge.GeneratorModel import ModelGenerator
from src.engine.ModelConfig import ModelConfig
from src.NeuroForge.Metrics import get_max_epoch, get_max_iteration, get_max_weight, get_max_error

class DisplayManager:

    def __init__(self, configs: List[ModelConfig]):
        Const.configs       = configs  # Store all model configs
        self.components     = []  # List for EZSurface-based components
        self.eventors       = []  # Components that need event handling
        self.event_runners  = []
        self.models         = []  # List for display models
        self.db             = configs[0].db  # Temporary shortcut
        self.iteration_data = None
        self.last_iteration = 0
        self.last_epoch     = 0
        Const.dm = self

        # Compute global max values across all models using Metrics module
        Const.MAX_EPOCH     = get_max_epoch(self.db)
        Const.MAX_ITERATION = get_max_iteration(self.db)
        Const.MAX_WEIGHT    = get_max_weight(self.db)
        Const.MAX_ERROR     = get_max_error(self.db)

        # Initialize UI Components
        self.get_iteration_dict()
        self.initialize_components()

        #print(self.get_model_iteration_data("NeuroForge_Template"))
        #print(self.get_model_iteration_data("HayabusaTwoWeights"))

    def initialize_components(self):
        """Initialize UI components like EZForm-based input panels and model displays."""

        problem_type = Const.configs[0].training_data.problem_type
        display_banner = DisplayBanner(problem_type, Const.MAX_EPOCH, Const.MAX_ITERATION)
        self.components.append(display_banner)

        # Add Input Panel
        input_panel = DisplayPanelInput(width_pct=12, height_pct=42, left_pct=2, top_pct=10)
        self.components.append(input_panel)

        # Add Control Panel
        panel = DisplayPanelCtrl( width_pct=12, height_pct=42, left_pct=2, top_pct=54)
        self.components.append(panel)
        self.eventors.append(panel)

        self.components.extend(ModelGenerator.create_models())  # This will process all layout calculations #create models
        positions = ModelGenerator.model_positions   #
        #print(positions)

    def update(self):
        if self.last_iteration == Const.CUR_ITERATION and self.last_epoch == Const.CUR_EPOCH:
            return #No change so no need to update
        self.last_iteration = Const.CUR_ITERATION   # Set them to current values
        self.last_epoch     = Const.CUR_EPOCH       # Set them to current values
        for component in self.components:
            component.update_me()

    def render(self):
        """Render all registered components."""
        for component in self.components:            #print(f"Rendering: {component.child_name}")  # Print the subclass name
            component.draw_me()

    def process_events(self, event):
        for component in self.eventors:            #print(f"Display Manager: event={event} ")
            component.process_an_event(event)

    def get_iteration_dict(self):
        """Retrieve iteration data from the database and return it as a nested dictionary indexed by model_id."""
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """
        params = (Const.CUR_EPOCH, Const.CUR_ITERATION)
        rs = self.db.query(sql, params)

        self.iteration_data = {}
        for row in rs:
            model_id = row["model_id"]
            self.iteration_data[model_id] = row  # Store each model's data separately

    def get_model_iteration_data(self, model_id: str = None) -> dict:
        """Retrieve iteration data for a specific model from the cached dictionary."""
        if model_id:
            return self.iteration_data.get(model_id, {})

        # If no model_id is provided, return the first available model's data
        for model in self.iteration_data.values():
            return model  # Return the first entry found
        return {}

