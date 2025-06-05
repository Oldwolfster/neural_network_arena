import pygame
from src.NeuroForge import Const
from src.NeuroForge.DisplayArrow import DisplayArrow
from src.NeuroForge.DisplayModel__Connection import DisplayModel__Connection
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.GeneratorNeuron import GeneratorNeuron
from src.engine.Config import Config
from src.engine.Utils import draw_rect_with_border, draw_text_with_background, ez_debug, check_label_collision, get_text_rect, beautify_text

class DisplayArrowsOutsideNeuron(EZSurface):
    #__slots__ = ("config", "neurons", "arrows_forward", "model_id")
    def __init__(self, first_model)   :
        """Initialize a display model using pixel-based positioning."""
        super().__init__(
            width_pct=0, height_pct=0, left_pct=0, top_pct=0,  # Ignore percent-based positioning
            pixel_adjust_width  = Const.SCREEN_WIDTH,
            pixel_adjust_height = Const.SCREEN_HEIGHT,
            pixel_adjust_left   = 0,
            pixel_adjust_top    = 0,
            transparent         = True
        )
        self.first_model= first_model
        self.arrows_forward = []  # List of neuron connections
        if self.first_model.input_scaler_neuron:
            self.add_input_connections_through_scaler(True)
        else:
            self.add_input_connections(True)
        self.add_output_connections()

    def add_input_connections_through_scaler (self, forward: bool):
        self.step_1_input_to_scaler(forward)
        self.step_2_scaler_to_hidden(forward)

    def step_2_scaler_to_hidden(self, forward: bool):
        model = self.first_model
        first_layer = model.neurons[0]  # First hidden layer
        neuron_visualizer = model.input_scaler_neuron.neuron_visualizer
        input_positions = neuron_visualizer.label_y_positions#   Const.dm.input_panel.label_y_positions  # 🔹 Reference input positions
        for target_neuron in first_layer:  # 🔹 Loop through all neurons in first layer
            #print(f"neuron_id={target_neuron.nid}")
            neuron_visualizer = target_neuron.neuron_visualizer
            for input_index, (start_x, start_y) in enumerate(input_positions):  # 🔹 Track input index

                # ✅ Use input_index to ensure correct weight label mapping
                end_x = model.left + neuron_visualizer.my_fcking_labels[input_index+1][0]   # Get global X position
                end_y = model.top + neuron_visualizer.my_fcking_labels[input_index+1][1]  +9  # Get global Y position

                # ✅ Adjust a bit to center arrows.
                start_x += model.left-6.9
                start_y += model.top+16.9
                #end_y   += 6.9

                # ✅ Now create an arrow using the correct X and Y values
                #print(f"to neuron start_x={start_x}, start_y={start_y}, end_x={end_x}, end_y={end_y}")
                self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))


    def step_1_input_to_scaler(self, forward: bool):
        """
        Creates connections from input panel to the first layer of neurons. (In the first model only - otherwise too much clutter)
        """
        #return # TODO temp for test, remove me
        model = self.first_model

        input_positions = Const.dm.input_panel.label_y_positions  # 🔹 Reference input positions
        #for target_neuron in first_layer:  # 🔹 Loop through all neurons in first layer
        neuron_visualizer = model.input_scaler_neuron.neuron_visualizer
        for input_index, (start_x, start_y) in enumerate(input_positions[:-1]):  # 🔹 Track input index
            #ez_debug(input_index=neuron_visualizer.my_fcking_labels)
            end_x = model.left + neuron_visualizer.my_fcking_labels[input_index][0]   # Get global X position
            end_y = model.top + neuron_visualizer.my_fcking_labels[input_index][1]    # Get global Y position
            start_x += -20
            end_y   += 6.9
            #print(f"to neuron start_x={start_x}, start_y={start_y}, end_x={end_x}, end_y={end_y}")
            self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))



    def add_input_connections(self, forward: bool):
        """
        Creates connections from input panel to the first layer of neurons. (In the first model only - otherwise too much clutter)
        """
        model = self.first_model

        first_layer = model.neurons[0]  # First hidden layer

        input_positions = Const.dm.input_panel.label_y_positions  # 🔹 Reference input positions
        for target_neuron in first_layer:  # 🔹 Loop through all neurons in first layer
            neuron_visualizer = target_neuron.neuron_visualizer
            for input_index, (start_x, start_y) in enumerate(input_positions[:-1]):  # 🔹 Track input index

                # ✅ Use input_index to ensure correct weight label mapping
                end_x = model.left + neuron_visualizer.my_fcking_labels[input_index+1][0]   # Get global X position
                end_y = model.top + neuron_visualizer.my_fcking_labels[input_index+1][1]    # Get global Y position

                # ✅ Adjust a bit to center arrows.
                start_x += -20
                end_y   += 6.9

                # ✅ Now create an arrow using the correct X and Y values
                #print(f"to neuron start_x={start_x}, start_y={start_y}, end_x={end_x}, end_y={end_y}")
                self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))


    def add_output_connections(self):
        """
        Creates connections from the last layer of neurons to the prediction panel.
        """
        model = self.first_model
        last_layer = model.neurons[-1]  # Last hidden layer
        for target_neuron in last_layer:  # 🔹 Loop through all neurons in the last layer
            start_x = model.left + target_neuron.location_left + target_neuron.location_width  # Right edge of neuron
            start_y = model.top + target_neuron.location_top + (target_neuron.location_height // 2) + 6.9  # Center Y

            # ✅ Fixed prediction panel coordinates
            end_x = Const.SCREEN_WIDTH *.91
            end_y = 239.96

            # ✅ Now create an arrow using the correct X and Y values
            #self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))

            #From output layer to prediction scaler - coming from same place so no change to start_x and start_y
            if model.config.scaler.target_is_scaled:
                end_x = model.left + model.prediction_scaler_neuron.neuron_visualizer.my_fcking_labels[0][0]
                end_y = model.top + model.prediction_scaler_neuron.neuron_visualizer.my_fcking_labels[0][1]
                self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))


    def render(self):
        #ez_debug(inputarrows =len(self.arrows_forward))
        for arrow in self.arrows_forward:
            arrow.draw()
        # this works pygame.draw.rect(self.surface, (255, 0, 0), (100, 100, 400, 300), 3)  # Red outline

    def update_me(self):
        pass