from typing import List

import pygame
import pygame_gui
import time
from src.ArenaSettings import HyperParameters
from src.neuroForge_original import mgr
from src.neuroForge_original.DisplayBanner import DisplayBanner
from src.neuroForge_original.DisplayModel import DisplayModel
from src.neuroForge_original.DisplayPanelCtrl import DisplayPanelCtrl
from src.neuroForge_original.DisplayPanelInput import DisplayPanelInput
from src.neuroForge_original.DisplayPanelLoss import DisplayPanelLoss
from src.neuroForge_original.DisplayPanelPrediction import DisplayPanelPrediction
from src.engine.RamDB import RamDB
from src.engine.Utils_DataClasses import ModelInfo
from src.neuroForge_original.VCR import VCR
from collections import namedtuple
class DisplayManager:
    def __init__(self, screen: pygame.Surface, hyper: HyperParameters, db: RamDB, model_info_list: List[ModelInfo], ui_manager):
        self.screen         = screen
        self.hyper          = hyper
        self.data_labels    = hyper.data_labels
        self.components     = []  # List for general EZSurface-based components
        self.eventors       = [] # components that need to process events.
        self.event_runners  = []
        self.models         = []  # List specifically for display models
        self.db = db
        self.model_info     = model_info_list
        mgr.max_epoch       = self.get_max_epoch()
        mgr.max_weight      = self.get_max_weight()
        mgr.max_iteration   = self.get_max_iteration()
        mgr.max_error       = self.get_max_error()
        mgr.color_neurons   = hyper.color_neurons
        self.neurons        = None
        mgr.VCR            = VCR()
        self.last_epoch = 0 #to get loop started
        self.get_iteration_dict(1, 1, model_info_list[0].model_id)
        self.get_epoch_dict(1, model_info_list[0].model_id)
        self.initialize_components(ui_manager)

    def initialize_components(self,  ui_manager):
        """Initialize and configure all display components."""

        # print(f"Model list (in displaymanager==============={ model_info_list[0].}")

        # Add Banner for EPoch and Iteration
        problem_type = self.model_info[0].problem_type
        banner = DisplayBanner(self.screen, problem_type, mgr.max_epoch, mgr.max_iteration, 96, 4, 2, 0)
        self.components.append(banner)

        # Add Input Panel
        input_panel = DisplayPanelInput(self.screen, data_labels=self.data_labels
                                        , width_pct=12, height_pct=42, left_pct=2, top_pct=10)
        self.components.append(input_panel)

        # Add Control Panel
        panel = DisplayPanelCtrl(self.screen, ui_manager, width_pct=12, height_pct=42, left_pct=2, top_pct=54)
        self.components.append(panel)
        self.eventors.append(panel)

        # Add Prediction panel
        prediction_panel = DisplayPanelPrediction(self.screen, problem_type
                                                  , width_pct=12, height_pct=42, left_pct=86, top_pct=10)
        self.components.append(prediction_panel)

        # Add Loss breakdown panel
        loss_panel = DisplayPanelLoss(self.screen, problem_type
                                      , width_pct=12, height_pct=42, left_pct=86, top_pct=54)
        self.components.append(loss_panel)

        # Create and add models
        self.models = self.create_display_models(72, 91, 14, 5, self.screen, self.hyper.data_labels)


    def update(self, iteration: int, epoch: int, model_id: str):
        """Render all components on the screen."""
        # db.list_tables()

        iteration_dict = self.get_iteration_dict(epoch, iteration, model_id)
        epoch_dict = self.get_epoch_dict(epoch, model_id)
        # db.query_print("Select * from iteration")
        #print(f"iteration_dict:::{iteration_dict}")
        # Render models
        # db.query_print("SELECT typeof(target), target FROM Iteration LIMIT 5;")
           #print(f"Retrieved iteration data: {rs[0]}")
        error=iteration_dict.get("error", 0.0)
        loss=iteration_dict.get("loss", 0.0)
        loss_grd=iteration_dict.get("loss_gradient", 0.0)
        #self.summarize_epoch(error,1.2,1.3 )

        # Update general components
        for component in self.components:
            component.update_me(iteration_dict, epoch_dict)
        self.render()

        for model in self.models:
            model.update_me(self.db, iteration, epoch, model_id)


    def process_events(self, event):
        for component in self.eventors:
            #print (f"DEBUG IN DM - Component = {component}")
            component.process_an_event(event)


    def get_epoch_dict(self, epoch: int, model_id: str ) -> dict:  #Retrieve iteration data from the database."""
        # db.query_print("PRAGMA table_info(Iteration);")
        sql = """  
            SELECT * FROM EpochSummary 
            -- WHERE epoch = ? and model_id = ?
            WHERE epoch=? and 1=?
        """
        #print(f"epoch={epoch}\tmodel_id={model_id}")    #TODO fix model_id
        #params = (epoch, model_id)
        params = (epoch,1)
        #params = (1,1)
        rs = self.db.query(sql, params)
        #print(f"EPOCH DICT: {rs}")
        if rs:            #
            return rs[0]  # Return the first row as a dictionary

        #print(f"No data found for epoch={epoch}")
        return {}  # Return an empty dictionary if no results


    def get_iteration_dict(self, epoch: int, iteration: int, model_id: str ) -> dict:  #Retrieve iteration data from the database."""
        # db.query_print("PRAGMA table_info(Iteration);")
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """#TODO ADD MODEL TO CRITERIIA
        params = (epoch, iteration)
        rs = self.db.query(sql, params)

        if rs:            #
            return rs[0]  # Return the first row as a dictionary
        #print(f"No data found for epoch={epoch}, iteration={iteration}")
        return {}  # Return an empty dictionary if no results

    def summarize_epochs(self, error: float, loss: float):
        pass
    def gameloop_hook(self):
        mgr.VCR.play_the_tape()

    def render(self):
        """
        Render all components on the screen."""
        for component in self.components:  # Render general components
            component.draw_me()

        # Render models
        for model in self.models:
            model.draw_me()

        if mgr.tool_tip is not None:
            mgr.tool_tip.render_tooltip(self.screen)
            mgr.tool_tip = None

    def get_max_error(self) -> int:
        """Retrieve highest abs(error) """
        sql = "SELECT MAX(abs(error_signal)) as error_signal FROM Neuron"
        rs = self.db.query(sql)
        # print(f"Max epoch{rs}")
        return rs[0].get("error_signal")

    def get_max_epoch(self) -> int:
        """Retrieve highest epoch."""
        sql = "SELECT MAX(epoch) as max_epoch FROM Iteration"
        rs = self.db.query(sql)
        # print(f"Max epoch{rs}")
        return rs[0].get("max_epoch")

    def get_max_weight(self) -> float:
        sql ="""
                SELECT MAX(ABS(value)) AS max_weight
                FROM (  SELECT json_each.value AS value
                        FROM Neuron, json_each(Neuron.weights))    
        """
        rs = self.db.query(sql)
        max_weight = rs[0].get("max_weight")
        print(f"max weight: {max_weight}")
        return max_weight


    def get_max_iteration(self) -> int:
        """Retrieve highest iteration"""
        sql = "SELECT MAX(iteration) as max_iteration FROM Iteration"
        rs = self.db.query(sql)
        return rs[0].get("max_iteration")

    def create_display_models(self, width_pct: int, height_pct: int, left_pct: int, top_pct: int,
            screen: pygame.Surface, labels):
        """Create DisplayModel instances based on the provided model information."""
        models = []
        for index, model_info in enumerate(self.model_info):
            # Example: Adjust positions for multiple models
            model_left = 10 + index * 300  # Spacing models horizontally
            model_top = 50
            display_model = DisplayModel(
                screen=screen,
                data_labels=labels,
                width_pct=width_pct,
                height_pct=height_pct,
                left_pct=left_pct,
                top_pct=top_pct,
                db=self.db
            )
            display_model.initialize_with_model_info(model_info)  # Populate model details
            models.append(display_model)
        return models

