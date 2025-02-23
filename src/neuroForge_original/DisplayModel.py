import pygame
from typing import List
from src.neuroForge_original import mgr
from src.neuroForge_original.DisplayModel__ConnectionForward import DisplayModel__ConnectionForward
from src.neuroForge_original.DisplayModel__Neuron import DisplayModel__Neuron
from src.neuroForge_original.EZSurface import EZSurface
from src.engine.RamDB import RamDB



class DisplayModel(EZSurface):
    def __init__(self, screen, data_labels, width_pct, height_pct, left_pct, top_pct, db: RamDB, architecture=None):
        #print(f"IN DISPLAYMODEL -- left_pct = {left_pct}")
        #chg below bg color to do bo
        super().__init__(screen, width_pct, height_pct, left_pct, top_pct, bg_color=(255, 255, 255))
        self.db = db
        self.neurons = [[] for _ in range(len(architecture))] if architecture else []  # Nested list by layers
        self.connections = []  # List of connections between neurons
        self.model_id = None
        self.data_labels = data_labels
        self.architecture = architecture or []  # Architecture to define layers and neurons

    def initialize_with_model_info(self, model_info):
        """
        Populate neurons and connections based on the provided model information.
        """
        self.model_id = model_info.model_id
        self.architecture = model_info.full_architecture
        self.create_neurons()
        self.create_arrows(True) #false indicates back prop

    def create_arrows(self, forward: bool):
        # Create connections
        self.connections = []
        for layer_index in range(1, len(self.architecture) - 1):  # Start from the first hidden layer
            current_layer = self.neurons[layer_index - 1]  # Adjust to skip the input layer
            next_layer = self.neurons[layer_index]
            for weight_index,from_neuron in enumerate( current_layer):
                for to_neuron in next_layer:
                    #print(f"Weight index={weight_index}")
                    if forward: #forward prop arrows
                        connection = DisplayModel__ConnectionForward(from_neuron=from_neuron, to_neuron=to_neuron, weight_index=weight_index)
                    else:   #back prop (reversed)
                        connection = DisplayModel__ConnectionForward(from_neuron=to_neuron, to_neuron=from_neuron, weight_index=weight_index)
                    self.connections.append(connection)

        # *** Add Input-to-First-Hidden-Layer Connections ***
        self.add_input_connections(forward)
        self.add_output_connections(forward)
        return self.neurons

    def add_input_connections(self, forward: bool):
        """
        Creates connections from a single fixed point on the left edge of the model area
        to the first hidden layer neurons.
        """

        first_hidden_layer = self.neurons[0]  # First hidden layer
        if not first_hidden_layer:
            return  # Edge case: No neurons in first hidden layer

        # 🔹 Fixed single origin point (Middle of the left edge of the grey box)
        origin_x = self.left
        origin_x = 0        # Left edge of the grey box
        origin_y =  self.height // 2  # Vertical center of the grey box
        origin_point = (origin_x, origin_y)  # Define as coordinate tuple
        #print(f"origin_point = {origin_point}")
        for neuron in first_hidden_layer:
            # Connect from the single origin point to each neuron in the first hidden layer
            self.connections.append(DisplayModel__ConnectionForward(from_neuron=origin_point, to_neuron=neuron))

    def add_output_connections(self, forward: bool): # Creates connections from last output neuron to prediction box
        destination_x = self.width
        destination_y = 20
        dest_point = (self.width,144)
        output_neuron = self.neurons[-1][0]
        self.connections.append(DisplayModel__ConnectionForward(from_neuron=self.neurons[-1][0], to_neuron=dest_point))

    def render(self):
        """
        Draw neurons and connections on the model's surface.
        """
        self.clear()  # Clear the surface before rendering
        # Draw connections
        for connection in self.connections:
            connection.draw_connection(self.surface)

        neuron_for_tooltip = None
        # Draw neurons
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron(self.surface)
                #print(f"self.left={self.left}self.screen.size={self.screen.size}")
                if neuron.is_hovered  (self.left, self.top):
                    mgr.tool_tip = neuron
                    #print(f"{mgr.tool_tip}")

    def update_me(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        """
        Update neuron and connection information based on the current state in the database.
        """
        DisplayModel__Neuron.retrieve_inputs(db, iteration, epoch, model_id)
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron(db, iteration, epoch, self.model_id)

        # (Optional) If connections have dynamic properties, update them too
        for connection in self.connections:
            connection.update_connection()


    def calculate_neuron_size(self, margin, gap, max_neuron_size):
        """
        Calculate the largest neuron size possible while ensuring all neurons fit
        within the given surface dimensions.

        Parameters:

            margin (int): Space around the entire neuron visualization.
            gap (int): Minimum space between neurons (both horizontally & vertically).
            max_neuron_size (int): Max allowable size per neuron.

        Returns:
            int: Optimized neuron size.
        """
        architecture=self.architecture[1:] #do not include input layer
        max_neurons = max(architecture)  # Determine the layer with the most neurons
        max_layers = len(architecture)  # Total number of layers

        # 🔹 Compute maximum available height and width
        available_height = self.height - (2 * margin)   - ( max_neurons - 1) * gap
        available_width = self.width - (2 * margin)        - ( max_layers - 1) * gap
        #print(f"available_width ={available_width}")
        width_per_cell = available_width // max_layers
        #print (f"width_per_cell={width_per_cell}")
        height_per_cell = available_height // max_neurons
        # 🔹 Take the minimum size that fits both width and height constraints
        optimal_neuron_size = min(width_per_cell, height_per_cell, max_neuron_size)
        #print (f"optimal_neuron_size={optimal_neuron_size}")
        return optimal_neuron_size

    def create_neurons(self, margin=20, gap=60, max_neuron_size=400):
    #def create_neurons(self, margin=00, gap=0, max_neuron_size=2000):
        """
        Create neuron objects, dynamically positioning them based on architecture.

        Parameters:
            margin (int): Space around the entire neuron visualization.
            gap (int): Minimum space between neurons (both horizontally and vertically).
            max_neuron_size (int): Maximum allowed size for a single neuron.
        """
        architecture=self.architecture[1:] #do not include input layer

        # 🔹 Compute neuron size and spacing
        size = self.calculate_neuron_size( margin, gap, max_neuron_size)

        max_neurons = max(architecture)  # Determine the layer with the most neurons
        max_layers = len(architecture)  # Total number of layers
        text_version = "Concise" if max(max_neurons,max_layers)  > 3 else "Verbose" # Choose appropriate text method based on network size
        width_needed = size * max_layers + (max_layers -1) * gap + margin * 2
        extra_width = self.width - width_needed
        extra_width_to_center = extra_width / 2

        #print(f"height_needed={height_needed}\tself.height={self.height}\textra_height={extra_height}\textra_height_to_center={extra_height_to_center}")
        # 🔹 Compute the horizontal centering offset (adjust for EZSurface width)
        offset = (self.screen_width - self.width)
        #print(f"self.screen_width={self.screen_width}\tself.width={self.width}\toffset={offset}")
        # 🔹 Compute horizontal spacing between layers
        num_layers = len(architecture)

        # 🔹 Create neurons
        self.neurons = []
        nid = -1

        for layer_index, neuron_count in enumerate(architecture):
            layer_neurons = []

            # 🔹 Compute X coordinate for neurons in this layer
            x_coord = size * layer_index + layer_index * gap  + margin + extra_width_to_center


            for neuron_index in range(neuron_count):
                nid += 1  # Increment neuron ID
                height_needed = size * neuron_count + (neuron_count -1) * gap
                extra_height = self.height - height_needed
                extra_height_to_center = extra_height  / 2

                neuron = DisplayModel__Neuron(nid=nid, layer=layer_index, position=neuron_index, output_layer=len(architecture)-1, text_version=text_version, db=self.db, model_id= self.model_id   )
                y_coord = size * neuron_index + gap * neuron_index + margin + extra_height_to_center

                # 🔹 Assign calculated position & size
                #print(f"size= {size}\tcoord = {x_coord},{y_coord}")
                neuron.location_left = x_coord   # Add offset for full screen centering
                neuron.location_top = y_coord
                neuron.location_width = size
                neuron.location_height = size
                layer_neurons.append(neuron)

            self.neurons.append(layer_neurons)



