from typing import List
import pygame

from src.NeuroForge.ButtonBase import  Button_Base
from src.NeuroForge.PopupInfoButton import PopupInfoButton
from src.NeuroForge.PopupTrainingData import PopupTrainingData
from src.engine.Config import Config
from src.engine.RamDB import RamDB
from src.NeuroForge import Const
from src.engine.Utils import draw_rect_with_border, draw_text_with_background, ez_debug, check_label_collision, get_text_rect, beautify_text
from src.NeuroForge.DisplayArrowsOutsideNeuron import DisplayArrowsOutsideNeuron
from src.NeuroForge.DisplayBanner import DisplayBanner
from src.NeuroForge.DisplayPanelCtrl import DisplayPanelCtrl
from src.NeuroForge.DisplayPanelInput import DisplayPanelInput
from src.NeuroForge.DisplayPanelPrediction import DisplayPanelPrediction
from src.NeuroForge.GeneratorModel import ModelGenerator
from src.NeuroForge.ui.WindowMatches import WindowMatches


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
        # MOVED TO EACH MODEL self.hoverlings: list     = None  # ✅ Store the neuron and objects that react when being hovered over
        self.the_hovered    = None # if an object is being hovered.
        self.hoverlings     = []
        self.t_data_popup   = PopupTrainingData(configs[0])
        self.info_popup     = PopupInfoButton()
        self.components     = []  # List for EZSurface-based components
        self.eventors       = []  # Components that need event handling
        #self.each_framers  = []
        #self.draw_lasters   = []
        self.models         = []  # List for display models
        self.db             = configs[0].db  # Temporary shortcut
        self.data_iteration = None
        self.data_epoch     = None
        self.last_iteration = 0
        self.last_epoch     = 0
        self.input_panel    = None
        self.base_window    = None
        Const.dm            = self

        # Compute global max values across all models using Metrics module
        self.get_max_epoch_per_model(self.db)
        self.populate_list_of_avaiable_frames()
        Const.MAX_EPOCH     = self.get_max_epoch(self.db)
        Const.MAX_ITERATION = self.get_max_iteration(self.db)
        Const.MAX_WEIGHT    = self.get_max_weight(self.db)
        Const.MAX_ERROR     = self.get_max_error(self.db)

        # Initialize UI Components
        self.query_dict_iteration()
        self.query_dict_epoch()
        self.initialize_components()
        
        ############# Make list of anything that reacts when hovered.
        #self.the_hovered = None  # Reset each frame
        #mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Add Neurons to each models "hoverlings"
        for model in reversed(self.models):  # ✅ Start with the topmost model
            for layer in model.neurons:
                for neuron in layer:
                    model.hoverlings.append(neuron)
        #############
    def populate_list_of_avaiable_frames(self):
        Const.vcr. recorded_frames = self.db.query("SELECT epoch, iteration from Weight group by epoch, iteration order by epoch, iteration",as_dict=False)

    def update(self):
        Const.vcr.play_the_tape()
        if self.last_iteration == Const.vcr.CUR_ITERATION and self.last_epoch == Const.vcr.CUR_EPOCH_MASTER:
            return #No change so no need to update
        self.last_iteration = Const.vcr.CUR_ITERATION   # Set them to current values
        self.last_epoch     = Const.vcr.CUR_EPOCH_MASTER       # Set them to current values
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

        #if Const.tool_tip_to_show is not None:
        #    print(f"Showing popup - Const.tool_tip_to_show = {Const.tool_tip_to_show }")
        #    Const.tool_tip_to_show()
        #    #Const.tool_tip_to_show = None


        #print(f"Const.tool_tip_to_show = {Const.tool_tip_to_show}")
        #if Const.tool_tip_to_show is not None:
        #    Const.tool_tip_to_show()   # now calls render()
        #    Const.tool_tip_to_show = None

        self.update_hover_state()
        if self.the_hovered is not None:
            self.the_hovered.render_tooltip()


    def initialize_components(self):
        """Initialize UI components like EZForm-based input panels and model displays."""
        display_banner = DisplayBanner(Const.configs[0].training_data, Const.MAX_EPOCH, Const.MAX_ITERATION)
        self.components.append(display_banner)
        panel_width = 8

        # Render behind the input button
        button_td=Button_Base(text=beautify_text("Training Data"),
                      width_pct=panel_width-1, height_pct=39, left_pct=2, top_pct=10, on_click=self.show_info,
                      on_hover=lambda: self.t_data_popup.show_me(),
                      shadow_offset=-5, auto_size=False, my_surface=Const.SCREEN,
                      text_line2=f"(click for details)", surface_offset=(0, 0))
        self.components.append(button_td)
        self.hoverlings.append(button_td)

        button_info = Button_Base(my_surface= Const.SCREEN, text="Info",
                                  width_pct=panel_width, height_pct=4, left_pct=1, top_pct=5,
                                  on_click=self.show_info,
                                  on_hover=lambda: self.info_popup.show_me(),
                                  shadow_offset=-5)
        self.components.append(button_info)
        self.hoverlings.append(button_info)

        # Add Input Panel  # Storing reference for arrows from input to first layer of neurons
        self.input_panel = DisplayPanelInput(width_pct=panel_width, height_pct=39, left_pct=1, top_pct=10)
        self.components.append(self.input_panel)

        # Add Control Panel
        panel = DisplayPanelCtrl( width_pct=panel_width, height_pct=44, left_pct=1, top_pct=51)
        self.components.append(panel)
        self.eventors.append(panel)

        # Add Prediction Panels for each model
        self.create_prediction_panels(panel_width)

        # Create Models
        #self.components.extend(ModelGenerator.create_models())  # This will process all layout calculations #create models
        self.models = ModelGenerator.create_models()
        self.components.extend(self.models) #add models to component list
        #self.eventors.extend(self.models)

        # Add Input and output Arrows (Spans multiple surfaces) - will be full area and not clear)
        arrows = DisplayArrowsOutsideNeuron(self.models[0])
        self.components.append(arrows)

        # Add window Match
        #self.base_window = BaseWindow(width_pct=60, height_pct=60, left_pct=20, top_pct=15, banner_text="Configure Match",
        #                              background_image_path="assets/form_backgrounds/coliseum_glow.png")
        #win_matches = WindowMatches()
        #self.components.append(win_matches)
        #self.eventors.append(win_matches)

    def show_info(self):
        print("Show info")

    def create_prediction_panels(self, panel_width): #one needed per model
        for idx, model_config in enumerate(Const.configs):
            model_id = model_config.gladiator_name  # Assuming Config has a `model_id` attribute
            problem_type = model_config.training_data.problem_type
            #For now, this will show 2 and write the rest over the top of each other.
            top = 10 #Assume 1 model
            if idx == 1:    #move 2nd box down (0 based)
                top = 52
            if idx <2:      #Only show two prediction panels
                panel = DisplayPanelPrediction(model_id, problem_type, model_config.loss_function, width_pct=panel_width, height_pct=39, left_pct=99-panel_width, top_pct=top)
                self.components.append(panel)

    def query_dict_iterationOld(self):
        """Retrieve iteration data from the database and return it as a nested dictionary indexed by model_id."""
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """
        params = (Const.vcr.CUR_EPOCH_MASTER, Const.vcr.CUR_ITERATION)
        rs = self.db.query(sql, params)

        self.data_iteration = {}
        for row in rs:
            model_id = row["model_id"]
            self.data_iteration[model_id] = row  # Store each model's data separately

    def query_dict_iteration(self):
        """Retrieve iteration data for each model from the latest valid epoch."""
        sql = """
            SELECT i.*
            FROM Iteration i
            JOIN (
                SELECT model_id, MAX(epoch) AS latest_epoch
                FROM Iteration
                WHERE epoch <= ?
                GROUP BY model_id
            ) latest ON i.model_id = latest.model_id AND i.epoch = latest.latest_epoch
            WHERE i.iteration = ?
        """
        params = (Const.vcr.CUR_EPOCH_MASTER, Const.vcr.CUR_ITERATION)
        rs = self.db.query(sql, params)

        self.data_iteration = {row["model_id"]: row for row in rs}

    def query_dict_epoch(self):
        sql = """
            SELECT e.*
            FROM EpochSummary e
            JOIN (
                SELECT model_id, MAX(epoch) AS latest_epoch
                FROM EpochSummary
                WHERE epoch <= ?
                GROUP BY model_id
            ) latest ON e.model_id = latest.model_id AND e.epoch = latest.latest_epoch
        """
        rs = self.db.query(sql, (Const.vcr.CUR_EPOCH_MASTER,))
        self.data_epoch = {row["model_id"]: row for row in rs}

    def query_dict_epoch_OLD(self ):  #failed for a model that converged and had no data for later epochs.
        # db.query_print("PRAGMA table_info(Iteration);")
        sql = """  
            SELECT * FROM EpochSummary            
            WHERE epoch=? and 1=?
        """
        params = (Const.vcr.CUR_EPOCH_MASTER, 1)

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

    def get_max_epoch_per_model(self, db: RamDB) -> None:
        """
        Retrieves the highest epoch per model from EpochSummary and stores it in each config's final_epoch.
        """
        for config in Const.configs:
            #model_id =   # Or whatever uniquely identifies this model in EpochSummary
            sql = "SELECT MAX(epoch) as max_epoch FROM EpochSummary WHERE model_id = ?"
            result = db.query(sql, (config.gladiator_name,))
            max_epoch = result[0].get("max_epoch") or 0
            config.final_epoch = max_epoch

    def get_max_epoch(self, db: RamDB) -> int:
        """Retrieve highest epoch for all models."""

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
        self.the_hovered = None  # Reset each frame
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for obj in self.hoverlings:
            if obj.is_hovered(0,0,mouse_x,mouse_y):
                self.the_hovered = obj
                return

        for model in reversed(self.models):  # ✅ Start with the topmost model
            for potential in model.hoverlings:
                if potential.is_hovered(model.left, model.top, mouse_x, mouse_y):
                    self.the_hovered = potential    # We found one, store it
                    return                                     # ✅ Stop checking once we find one (avoids conflicts)
