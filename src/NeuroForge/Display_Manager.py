from typing import List
import pygame
from src.NeuroForge import Const
from src.NeuroForge.DisplayArrowsOutsideNeuron import DisplayArrowsOutsideNeuron
from src.NeuroForge.DisplayBanner import DisplayBanner
from src.NeuroForge.DisplayModel__Graph import DisplayModel__Graph
from src.NeuroForge.DisplayPanelCtrl import DisplayPanelCtrl
from src.NeuroForge.DisplayPanelInput import DisplayPanelInput
from src.NeuroForge.DisplayPanelPrediction import DisplayPanelPrediction
from src.NeuroForge.GeneratorModel import ModelGenerator
from src.engine.Config import Config
from src.engine.RamDB import RamDB
from src.engine.Utils import ez_debug
class Display_Manager:
    """
    This class is the heart of NeuroForge.  It does the following.
    1) Initializes all components including Neurons, UI Panels and Controls, Activations(Outputs)
    2) Runs the main "Loops" such as update, render, and event processing
    3) Is a central location for retrieving and making available global data such as iteration and epoch queries
    It receives all initial needed information in the constructors parameter List[Configs].
    Changing information is retrieved from the in-ram SQL Lite DB that stores the Models states (think VCR Tape of the models training)
    NOTE: The underscore in the class name is deliberate to separate it from the classes it manages (which all are prefixed with 'Display'
    """
    def __init__(self, configs: List[Config]):
        Const.configs       = configs  # Store all model configs
        self.hovered_neuron = None  # ✅ Store the neuron being hovered over
        self.components     = []  # List for EZSurface-based components
        self.eventors       = []  # Components that need event handling
        self.event_runners  = []
        self.models         = []  # List for display models
        self.db             = configs[0].db  # Temporary shortcut
        self.data_iteration = None
        self.data_epoch     = None
        self.last_iteration = 0
        self.last_epoch     = 0
        self.input_panel    = None
        Const.dm = self

        # Compute global max values across all models using Metrics module
        Const.MAX_EPOCH     = self.get_max_epoch(self.db)
        Const.MAX_ITERATION = self.get_max_iteration(self.db)
        Const.MAX_WEIGHT    = self.get_max_weight(self.db)
        Const.MAX_ERROR     = self.get_max_error(self.db)

        # Initialize UI Components
        self.query_dict_iteration()
        self.query_dict_epoch()
        self.initialize_components()

    def update(self):
        Const.vcr.play_the_tape()
        if self.last_iteration == Const.CUR_ITERATION and self.last_epoch == Const.CUR_EPOCH:
            return #No change so no need to update
        self.last_iteration = Const.CUR_ITERATION   # Set them to current values
        self.last_epoch     = Const.CUR_EPOCH       # Set them to current values
        for component in self.components:
            component.update_me()


    def process_events(self, event):
        for component in self.eventors:            #print(f"Display Manager: event={event} ")
            component.process_an_event(event)

    def render(self):
        """Render all registered components. (Except pop up window"""
        for component in self.components:            #print(f"Rendering: {component.child_name}")  # Print the subclass name
            component.draw_me()


    def render_pop_up_window(self):
        """
        This is rendered separately to ensure it is last and is not overwritten by anything
        such as UI Controls.
        """
        self.update_hover_state()
        if self.hovered_neuron is not None:
            self.hovered_neuron.render_tooltip()

    def initialize_components(self):
        """Initialize UI components like EZForm-based input panels and model displays."""
        display_banner = DisplayBanner(Const.configs[0].training_data, Const.MAX_EPOCH, Const.MAX_ITERATION)
        self.components.append(display_banner)

        # Add Input Panel  # Storing reference for arrows from input to first layer of neurons
        self.input_panel = DisplayPanelInput(width_pct=12, height_pct=39, left_pct=1, top_pct=10)
        self.components.append(self.input_panel)

        # Add Control Panel
        panel = DisplayPanelCtrl( width_pct=12, height_pct=44, left_pct=1, top_pct=51)
        self.components.append(panel)
        self.eventors.append(panel)

        # Add Prediction Panels for each model
        self.create_prediction_panels()

        # Create Models
        #self.components.extend(ModelGenerator.create_models())  # This will process all layout calculations #create models
        self.models = ModelGenerator.create_models()
        self.components.extend(self.models) #add models to component list

        # Add Input and output Arrows (Spans multiple surfaces) - will be full area and not clear)
        arrows = DisplayArrowsOutsideNeuron(self.models[0])
        self.components.append(arrows)

        # Add Graph  # MAE over epoch
        #moved to display model
        #self.Graph = DisplayGraph(width_pct=40, height_pct=30, left_pct=60, top_pct=70)
        #self.components.append(self.Graph)


    def create_prediction_panels(self): #one needed per model
        for idx, model_config in enumerate(Const.configs):
            model_id = model_config.gladiator_name  # Assuming Config has a `model_id` attribute
            problem_type = model_config.training_data.problem_type
            #For now, this will show 2 and write the rest over the top of each other.
            top = 10 #Assume 1 model
            if idx == 1:    #move 2nd box down (0 based)
                top = 52
            if idx <2:      #Only show two prediction panels
                panel = DisplayPanelPrediction(model_id, problem_type, model_config.loss_function, width_pct=12, height_pct=39, left_pct=86, top_pct=top)
                self.components.append(panel)

    def query_dict_iteration(self):
        """Retrieve iteration data from the database and return it as a nested dictionary indexed by model_id."""
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """
        params = (Const.CUR_EPOCH, Const.CUR_ITERATION)
        rs = self.db.query(sql, params)

        self.data_iteration = {}
        for row in rs:
            model_id = row["model_id"]
            self.data_iteration[model_id] = row  # Store each model's data separately

    def query_dict_epoch(self ):  #Retrieve iteration data from the database."""
        # db.query_print("PRAGMA table_info(Iteration);")
        sql = """  
            SELECT * FROM EpochSummary            
            WHERE epoch=? and 1=?
        """
        params = (Const.CUR_EPOCH, 1)

        rs = self.db.query(sql, params)
        self.data_epoch = {}
        for row in rs:
            model_id = row["model_id"]
            self.data_epoch[model_id] = row  # Store each model's data separately

    def get_model_iteration_data(self, model_id: str = None) -> dict:
        """Retrieve iteration data for a specific model from the cached dictionary."""
        if model_id:
            return self.data_iteration.get(model_id, {})

        # If no model_id is provided, return the first available model's data
        for model in self.data_iteration.values():
            return model  # Return the first entry found
        return {}

    def get_model_epoch_data(self, model_id: str = None) -> dict:
        """Retrieve iteration data for a specific model from the cached dictionary."""
        if model_id:
            return self.data_epoch.get(model_id, {})

        # If no model_id is provided, return the first available model's data
        for model in self.data_epoch.values():
            return model  # Return the first entry found
        return {}

    def get_max_error(self, db: RamDB) -> int:
        """Retrieve highest abs(error)"""
        sql = "SELECT MAX(abs(error_signal)) as error_signal FROM Neuron"
        rs = db.query(sql)
        return rs[0].get("error_signal")

    def get_max_epoch(self, db: RamDB) -> int:
        """Retrieve highest epoch."""
        sql = "SELECT MAX(epoch) as max_epoch FROM Iteration"
        rs = db.query(sql)
        return rs[0].get("max_epoch")

    def get_max_weight(self, db: RamDB) -> float:
        """Retrieve highest weight magnitude."""
        sql = "SELECT MAX(ABS(value)) AS max_weight FROM Weight"
        rs = db.query(sql)
        return rs[0].get("max_weight")

    def get_max_iteration(self, db: RamDB) -> int:
        """Retrieve highest iteration"""
        sql = "SELECT MAX(iteration) as max_iteration FROM Iteration"
        rs = db.query(sql)
        return rs[0].get("max_iteration")

    def update_hover_state(self):
        """
        Check which neuron is being hovered over, prioritizing the topmost model.
        """
        self.hovered_neuron = None  # Reset each frame
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for model in reversed(self.models):  # ✅ Start with the topmost model
            for layer in model.neurons:
                for neuron in layer:
                    if neuron.is_hovered(model.left, model.top, mouse_x, mouse_y):
                        #print(f"hovering over {model.config.gladiator_name} { neuron.label}")
                        self.hovered_neuron = neuron  # ✅ Store the first neuron found
                        return  # ✅ Stop checking once we find one (avoids conflicts)





