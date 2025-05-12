import pygame
from src.NeuroForge import Const
from src.NeuroForge.ButtonBase import Button_Base
from src.NeuroForge.DisplayArrow import DisplayArrow
from src.NeuroForge.DisplayModel__Graph import DisplayModel__Graph
from src.NeuroForge.DisplayModel__Connection import DisplayModel__Connection
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.GeneratorNeuron import GeneratorNeuron
from src.NeuroForge.PopupArchitecture import ArchitecturePopup
from src.engine.Config import Config
from src.engine.Utils import draw_rect_with_border, draw_text_with_background, ez_debug, check_label_collision, get_text_rect, beautify_text
from src.engine.Utils import smart_format

class DisplayModel(EZSurface):
    __slots__ = ("last_epoch", "input_scaler_neuron", "hoverlings", "arch_popup","buttons", "config", "neurons", "threshold", "arrows_forward", "model_id", "graph_holder", "graph")
    def __init__(self, config: Config, position: dict )   :
        """Initialize a display model using pixel-based positioning."""
        super().__init__(
            width_pct=0, height_pct=0, left_pct=0, top_pct=0,  # Ignore percent-based positioning
            pixel_adjust_width  = position["width"],
            pixel_adjust_height = position["height"],
            pixel_adjust_left   = position["left"],
            pixel_adjust_top    = position["top"],
            bg_color            = Const.COLOR_FOR_BACKGROUND
        )
        self.hoverlings: list   = []  # ✅ Store the neuron and objects that react when being hovered over
        self.buttons        = []
        self.config         = config
        self.arch_popup     = ArchitecturePopup(self, config)

        self.last_epoch     = config.final_epoch
        #print(f"self.last_epoch = {self.last_epoch}")
        self.graph          = None
        self.model_id       = config.gladiator_name
        self.neurons        = [[] for _ in range(len(self.config.architecture))]  # Nested list by layers
        self.arrows_forward = []  # List of neuron connections
        _, _,self.threshold = config.training_data.get_binary_decision_settings(config.loss_function)

        btn = Button_Base(
                 text=f"{beautify_text(self.config.gladiator_name)}",
                 width_pct=10, height_pct=4, left_pct=1, top_pct=1,
                 on_click=self.show_info,
                 on_hover=lambda: self.arch_popup.show_me(),
                 shadow_offset=-5, auto_size=True, my_surface=self.surface,text_line2=f"Error: {self.lowest_error} ", surface_offset=(self.left, self.top))

        self.buttons.append(btn)


        self.hoverlings.extend(self.buttons)

    @property
    def lowest_error(self):
        return f"{smart_format(self.config.lowest_error)}({smart_format(self.config.percent_off)}%) at {self.config.lowest_error_epoch}"
    @property
    def display_epoch(self):
        """
        Returns the appropriate epoch to display based on the global VCR state.
        If the model converged early, it freezes at its last recorded epoch.
        """
        if self.last_epoch is None:
            return Const.vcr.CUR_EPOCH_MASTER
        return min(Const.vcr.CUR_EPOCH_MASTER, self.last_epoch)

    @property
    def display_iteration(self):
        """
        Same logic as display_epoch but for iteration display.
        You could optionally track last_iteration if you want extra granularity.
        """
        return min(Const.vcr.CUR_ITERATION, self.last_iteration or Const.vcr.CUR_ITERATION)

    def update_last_epoch(self, epoch):
        self.last_epoch = epoch

    def update_last_iteration(self, iteration):
        self.last_iteration = iteration

    def initialize_with_model_info(self):
        """Create neurons and connections based on architecture."""
        max_activation = self.get_max_activation_for_model(self.model_id)
        GeneratorNeuron.create_neurons(self, max_activation)
        for layer in self.neurons:
            for neuron in layer:
                neuron.my_model = self
        # Create error over epochs graph
        self.graph = self.create_graph(self.graph_holder)# Add Graph  # MAE over epoch
        Const.dm.eventors.append(self.graph)

        self.render()   #Run once so everything is created
        self.create_neuron_to_neuron_arrows(True)  # Forward pass arrows


    def create_graph(self, gh):
        return DisplayModel__Graph(left=gh.location_left, width= gh.location_width, top=gh.location_top, height=gh.location_height, model_surface=self.surface, model_id=self.model_id, my_model=self)

    def render(self):
        """Draw neurons and connections."""
        self.clear()
        self.draw_border()
        self.graph.render()
#        for connection in self.connections:
#            connection.draw_connection(self.surface)
        if self.input_scaler_neuron is not None:
            self.input_scaler_neuron.draw_neuron()
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron()

        for arrow in self.arrows_forward:
            arrow.draw()

        for button in self.buttons:
            button.draw_me()



    def update_me(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron()
        #for arrow in self.arrows_forward:
        #    arrow.update_connection()


    def draw_border(self):
        """Draw a rectangle around the perimeter of the display model."""
        pygame.draw.rect(
            self.surface, Const.COLOR_FOR_NEURON_BODY,
            (0, 0, self.width, self.height), 3
        )

    def get_max_activation_for_model(self,  model_id: str):
        """
        Retrieves the highest absolute activation value across all epochs and iterations for the given model.

        :param model_id: The model identifier
        :return: The maximum absolute activation value in the run
        """

        SQL_MAX_ACTIVATION = """
            SELECT MAX(abs_activation) AS max_activation
            FROM (
                SELECT ABS(activation_value) AS abs_activation
                FROM Neuron
                WHERE model = ?
                ORDER BY abs_activation ASC
                LIMIT (SELECT CAST(COUNT(*) * 0.95 AS INT) 
                       FROM Neuron WHERE model = ?)
            ) AS FilteredActivations;
        """

        result = self.config.db.query(SQL_MAX_ACTIVATION, (model_id, model_id))
        #print(f"Max activation for run {result}")
        # Return the max activation or a default value to prevent division by zero
        return result[0]['max_activation'] if result and result[0]['max_activation'] is not None else 1.0

    def create_neuron_to_neuron_arrows(self, forward: bool):
        """Creates neuron-to-neuron arrows using DisplayArrow."""

        self.arrows_forward = []
        y_offset = 10
        for layer_index in range(1, len(self.config.architecture)):  # Start from the first hidden layer
            current_layer = self.neurons[layer_index - 1]
            next_layer = self.neurons[layer_index]
            for weight_index, from_neuron in enumerate(current_layer):
                for to_neuron in next_layer:
                    start_x = from_neuron.location_left + from_neuron.location_width  # Right edge of from_neuron
                    end_x = to_neuron.location_left  # Left edge of to_neuron

                    # ✅ Ensure we don’t exceed `my_fcking_labels` length
                    max_index_from = min(weight_index + 1, len(from_neuron.neuron_visualizer.my_fcking_labels) - 1)
                    max_index_to = min(weight_index + 1, len(to_neuron.neuron_visualizer.my_fcking_labels) - 1)

                    start_y = from_neuron.location_top+ from_neuron.location_height//2 + y_offset
                    end_y = to_neuron.neuron_visualizer.my_fcking_labels[max_index_to][1] + y_offset # Get Y for correct weight

                    self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))

    #def process_an_event(self, event):
    #    #print(f"my left={self.left} my top = {self.top}")
    #    self.model_button.process_an_event(event)


    def show_info(self):#onclick model button
        print("copy to clipboard")
        """
        import pygame.scrap

        pygame.scrap.init()
        pygame.scrap.set_mode(pygame.SCRAP_CLIPBOARD)
        
        def copy_info_to_clipboard():
            data = your_info_str.encode("utf-8")
            pygame.scrap.put(pygame.SCRAP_TEXT, data)
        """
