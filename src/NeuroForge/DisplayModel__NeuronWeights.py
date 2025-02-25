import pygame
from src.engine.RamDB import RamDB
import json
from src.engine.Utils import draw_rect_with_border, draw_text_with_background


class DisplayModel__NeuronWeights:
    def __init__(self, neuron):
        #Adjustable settings
        self.neuron = neuron  # ✅ Store reference to parent neuron
        self.global_max = 0
        self.num_weights = 0
        self.bar_height = 0
        self.neuron_height = 0
        self.max_activation = 0
        """
        self.font_size_weight = 24
        self.padding_top = 3
        self.padding_bottom = 3
        self.gap_between_weight_bars = 1
        self.gap_between_weights = 2
        self.right_margin = 40  # SET IN ITITALNew: Space reserved for activation visualization
        self.BANNER_HEIGHT = 29  # 4 pixels above + 26 pixels total height

        self.neuron = neuron  # ✅ Store reference to parent neuron
        self.model_id = model_id
        self.min_weight = float('inf')  # Track min/max for scaling
        self.max_weight = float('-inf')
        self.max_per_weight = []
        self.my_fcking_labels=[]
        self.need_label_coord = True #track if we recorded the label positions for the arrows to point from
        
        
        
        self.previous_weights = None  # Store last weights for comparison
        
        #TODO consolidate below to above
        self.global_max_activation = 0
        """
    def initialize(self, screen, ez_printer, body_y_start):
        print(f"intitialize 1 {len(self.neuron.weights)}")
        if len(self.neuron.weights) == 0:
            return
        print("intitialize 1")
        self.global_max, self.max_per_weight = self.get_weight_min_max(self.neuron.db, self.model_id, self.neuron.nid)
        self.num_weights = len(self.neuron.weights)
        self.neuron_height = self.neuron.location_height
        #self.max_activation = self.get_max_activation_for_run(self.neuron.db, self.model_id)
        print("intitialize 2")
        if self.num_weights > 0:
            self.bar_height= self.calculate_bar_height(
                num_weights=self.num_weights,neuron_height=self.neuron_height
                ,padding_top=self.padding_top,padding_bottom=self.padding_bottom
                ,gap_between_weight_bars= self.gap_between_weight_bars,gap_between_weights=self.gap_between_weights
            )
        #print(f"INITIALIZING//////////   Max per weight{self.max_per_weight}\tGlobal max: {self.global_max}")
        #self.debug_bar_placement()

    def render(self, screen, ez_printer, body_y_start):
        if self.global_max == 0:
            self.initialize(screen, ez_printer, body_y_start)

        #self.debug_weight_changes()
        #self.draw_weight_bars(screen)
        #self.draw_activation_bar(screen)
        #self.draw_activation_value(screen)
