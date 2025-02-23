from typing import List

import pygame

from src.engine.Utils_DataClasses import ModelInfo


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


    def update(self):
        pass

    def render(self):
        pass
