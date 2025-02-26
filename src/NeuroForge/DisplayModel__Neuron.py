from ast import literal_eval
from typing import List
import pygame
from src.engine.ActivationFunction import get_activation_derivative_formula
from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights
from src.NeuroForge.EZPrint import EZPrint
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format, draw_gradient_rect, ez_debug
from src.NeuroForge import Const

class DisplayModel__Neuron:
    """
    DisplayModel__Neuron is created by DisplayModel.
    This class has the following primary purposes:
    1) Store all information related to the neuron
    2) Update that information when the iteration or epoch changes.
    3) Draw the "Standard" components of the neuron.  (Body, Banner, and Banner Text)
    4) Invoke the appropriate "Visualizer" to draw the details of the Neuron
    """
    __slots__ = ("max_per_weight", "max_activation", "global_weight_max", "model_id", "screen", "db", "rs", "nid", "layer", "position", "output_layer", "label", "location_left", "location_top", "location_width", "location_height", "weights", "weights_before", "neuron_inputs", "raw_sum", "activation_function", "activation_value", "activation_gradient", "banner_text", "tooltip_columns", "weight_adjustments", "error_signal_calcs", "avg_err_sig_for_epoch", "loss_gradient", "ez_printer", "neuron_visualizer", "neuron_build_text", "weight_before" )
    input_values = []   # Class variable to store inputs

    def __init__(self, left: int, top: int, width: int, height:int, nid: int, layer: int, position: int, output_layer: int, text_version: str,  model_id: str, screen: pygame.surface, max_activation: float):
        self.model_id               = model_id
        self.screen                 = screen
        self.db                     = Const.dm.db
        self.rs                     = None  # Store result of querying Iteration/Neuron table for this iteration/epoch
        self.nid                    = nid
        self.layer                  = layer
        self.position               = position
        self.output_layer           = output_layer
        self.max_activation         = max_activation
        self.label                  = f"{layer}-{position}"

        # Positioning
        self.location_left          = left
        self.location_top           = top
        self.location_width         = width
        self.location_height        = height

        # Neural properties
        self.weights                = []
        self.weight_before          = []
        self.neuron_inputs          = []
        self.max_per_weight         = []
        self.activation_function    = ""
        self.raw_sum                = 0.0
        self.activation_value       = 0.0
        self.activation_gradient    = 0.0
        self.global_weight_max      = 0.0

        # Visualization properties
        self.banner_text            = ""
        self.tooltip_columns        = []
        self.weight_adjustments     = ""
        self.error_signal_calcs     = ""
        self.avg_err_sig_for_epoch  = 0.0
        self.loss_gradient          = 0.0
        self.neuron_build_text      = "fix me"
        self.ez_printer             = EZPrint(pygame.font.Font(None, 24), color=Const.COLOR_BLACK, max_width=200, max_height=100, sentinel_char="\n")

        # Conditional visualizer
        self.update_neuron()        # must come before selecting visualizer
        self.neuron_visualizer      = DisplayModel__NeuronWeights(self, self.ez_printer)
        #self.neuron_build_text = self.neuron_build_text_large if text_version == "Verbose" else self.neuron_build_text_small


    def draw_neuron(self):
        """Draw the neuron visualization."""
        # Define colors
        #TODO add Gradient body_color = self.get_color_gradient(self.avg_err_sig_for_epoch, mgr.max_error)

        # Font setup
        font = pygame.font.Font(None, 24) #TODO remove and use EZ_Print

        # Banner text
        label_surface = font.render(f"ID: {self.label}", True, Const.COLOR_FOR_NEURON_TEXT)
        output_surface = font.render(self.activation_function, True, Const.COLOR_FOR_NEURON_TEXT)
        label_strip_height = label_surface.get_height() + 8  # Padding

        # Draw the neuron body below the label
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        pygame.draw.rect(self.screen,  Const.COLOR_FOR_NEURON_BODY, (self.location_left, body_y_start, self.location_width, body_height), border_radius=6, width=7)

        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        draw_gradient_rect(self.screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)
        self.screen.blit(label_surface, (self.location_left + 5, self.location_top + (label_strip_height - label_surface.get_height()) // 2))
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5
        self.screen.blit(output_surface, (right_x, self.location_top + (label_strip_height - output_surface.get_height()) // 2))

        # Render visual elements
        if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer:
            self.neuron_visualizer.render() #, self, body_y_start)

    def update_neuron(self):
        #print(f"updating neuron {self.nid}")
        if not self.update_avg_error():
            return #no record found so exit early
        self.update_rs()
        self.update_weights()
        self.get_weight_min_max()

    def get_weight_min_max(self):
        """
        Retrieves:
        1. The global maximum absolute weight across all epochs and neurons.
        2. The maximum absolute weight for each individual weight index across all epochs.
        """

        # ✅ Query 1: Get the highest absolute weight overall
        SQL_GLOBAL_MAX = """
            SELECT MAX(ABS(value)) AS global_max
            FROM Weight
            WHERE model_id = ? AND nid = ?
        """
        global_max_result = self.db.query(SQL_GLOBAL_MAX, (self.model_id, self.nid))
        self.global_weight_max = global_max_result[0]['global_max'] if global_max_result and global_max_result[0]['global_max'] is not None else 1.0

        # ✅ Query 2: Get the max absolute weight per weight index
        SQL_MAX_PER_WEIGHT = """
            SELECT MAX(ABS(value)) AS max_weight
            FROM Weight
            WHERE model_id = ? AND nid = ?
            GROUP BY weight_id
            ORDER BY weight_id ASC
        """
        max_per_weight = self.db.query_scalar_list(SQL_MAX_PER_WEIGHT, (self.model_id, self.nid))
        self.max_per_weight = max_per_weight if max_per_weight != 0 else 1
    def update_rs(self):
        # Parameterized query with placeholders
        SQL =   """
            SELECT  *
            FROM    Iteration I
            JOIN    Neuron N
            ON      I.model_id  = N.model 
            AND     I.epoch     = N.epoch_n
            AND     I.iteration = N.iteration_n
            WHERE   model = ? AND iteration_n = ? AND epoch_n = ? AND nid = ?
            ORDER BY epoch, iteration, model, nid 
        """
        rs = self.db.query(SQL, (self.model_id, Const.CUR_ITERATION, Const.CUR_EPOCH, self.nid)) # Execute query
        self.rs = rs[0]
        self.loss_gradient =  float(rs[0].get("loss_gradient", 0.0))
        self.error_signal_calcs = rs[0].get("error_signal_calcs")

        # Activation function details
        self.activation_function    = rs[0].get('activation_name', 'Unknown')
        self.activation_value       = rs[0].get('activation_value', None)        #THE OUTPUT
        self.activation_gradient    = rs[0].get('activation_gradient', None)  # From neuron


        self.banner_text = f"{self.label}  Output: {smart_format( self.activation_value)}"

    def update_avg_error(self):
        SQL = """
        SELECT AVG(ABS(error_signal)) AS avg_error_signal            
        FROM Neuron
        WHERE 
        model   = ? and
        epoch_n = ? and  -- Replace with the current epoch(ChatGPT is trolling us)
        nid     = ?      
        """
        params = (self.model_id,  Const.CUR_EPOCH, self.nid)
        rs = self.db.query(SQL, params)  # Execute query

        # ✅ Check if `rs` is empty before accessing `rs[0]`
        if not rs:
            return False  # No results found

        # ✅ Ensure `None` does not cause an error
        self.avg_err_sig_for_epoch = float(rs[0].get("avg_error_signal") or 0.0)
        #print("in update_avg_error returning TRUE")
        return True
    def update_weights(self):
        """Fetches weights from the Weight table instead of JSON and populates self.weights and self.weights_before."""
        SQL = """
            SELECT weight_id, value, value_before
            FROM Weight
            WHERE model_id = ? AND nid = ? AND epoch = ? AND iteration = ?
            ORDER BY weight_id ASC
        """
        weights_data = self.db.query(SQL, (self.model_id, self.nid, Const.CUR_EPOCH, Const.CUR_ITERATION), False)

        if weights_data:
            self.weights = [column[1] for column in weights_data]  # Extract values
            self.weights_before = [column[2] for column in weights_data]  # Extract previous values
        else:
            # TODO: Handle case where no weights are found for the current epoch/iteration
            self.weights = []
            self.weights_before = []