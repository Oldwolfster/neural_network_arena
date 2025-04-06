import pygame
from src.Legos.ActivationFunctions import get_activation_derivative_formula
from src.Legos.Optimizers import *

from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights
from src.NeuroForge.EZPrint import EZPrint
from src.engine.Neuron import Neuron
from src.engine.Utils import smart_format, draw_gradient_rect, is_numeric
from src.NeuroForge import Const
import json

from src.engine.Utils_DataClasses import ez_debug


class DisplayModel__Neuron:
    """
    DisplayModel__Neuron is created by DisplayModel.
    Note: DisplayModel inherits from EzSurface, DisplayModel__Neuron does not!
    This class has the following primary purposes:
    1) Store all information related to the neuron
    2) Update that information when the iteration or epoch changes.
    3) Draw the "Standard" components of the neuron.  (Body, Banner, and Banner Text)
    4) Invoke the appropriate "Visualizer" to draw the details of the Neuron
    """
    __slots__ = ("my_model", "config", "column_widths","cached_tooltip", "text_version", "last_epoch","last_iteration", "font_header", "header_text", "font_body", "max_per_weight", "max_activation",  "model_id", "screen", "db", "rs", "nid", "layer", "position", "output_layer", "label", "location_left", "location_top", "location_width", "location_height", "weights", "weights_before", "neuron_inputs", "raw_sum", "activation_function", "activation_value", "activation_gradient", "banner_text", "tooltip_columns", "weight_adjustments", "blame_calculations", "avg_err_sig_for_epoch", "loss_gradient", "ez_printer", "neuron_visualizer", "neuron_build_text", )

    input_values = []   # Class variable to store inputs #TODO Delete me
    def __repr__(self):
        """Custom representation for debugging."""
        return f"Neuron {self.label})"
    def __init__(self, my_model, left: int, top: int, width: int, height:int, nid: int, layer: int, position: int, output_layer: int, text_version: str,  model_id: str, screen: pygame.surface, max_activation: float):
        self.model_id               = model_id
        self.my_model               = my_model
        self.config                 = self.my_model.config
        #print (f"In DM_Neuron {self.my_model} ")
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
        self.neuron_inputs          = []
        self.max_per_weight         = []
        self.activation_function    = ""
        self.raw_sum                = 0.0
        self.activation_value       = 0.0
        self.activation_gradient    = 0.0

        # Visualization properties
        self.banner_text            = ""
        self.tooltip_columns        = []
        self.weight_adjustments     = ""
        self.blame_calculations     = ""
        self.avg_err_sig_for_epoch  = 0.0
        self.loss_gradient          = 0.0
        self.neuron_build_text      = "fix me"
        self.ez_printer             = EZPrint(pygame.font.Font(None, 24), color=Const.COLOR_BLACK, max_width=200, max_height=100, sentinel_char="\n")
        self.get_max_val_per_wt()
        self.initialize_fonts()
        # Conditional visualizer
        self.update_neuron()        # must come before selecting visualizer
        self.neuron_visualizer      = DisplayModel__NeuronWeights(self, self.ez_printer)
        self.text_version           = text_version
        if self.layer == Neuron.output_neuron.layer_id:
            self.banner_text = "Out"
            if self.text_version == "Verbose":
                self.banner_text = "Output Neuron"
        else:
            self.banner_text = self.label
            if self.text_version == "Verbose":
                self.banner_text = f"Hidden Neuron {self.label}"
        #self.neuron_build_text = self.neuron_build_text_large if text_version == "Verbose" else self.neuron_build_text_small

    def is_hovered(self, model_x, model_y, mouse_x, mouse_y):
        """
        Check if the mouse is over this neuron.
        """
        neuron_x = model_x + self.location_left
        neuron_y = model_y + self.location_top
        return (neuron_x <= mouse_x <= neuron_x + self.location_width) and (neuron_y <= mouse_y <= neuron_y + self.location_height)

    def draw_neuron(self):
        """Draw the neuron visualization."""

        # Font setup
        font = pygame.font.Font(None, 30) #TODO remove and use EZ_Print

        # Banner text
        label_surface = font.render(self.banner_text, True, Const.COLOR_FOR_NEURON_TEXT)
        output_surface = font.render(self.activation_function, True, Const.COLOR_FOR_NEURON_TEXT)
        label_strip_height = label_surface.get_height() + 8  # Padding

        # Draw the neuron body below the label
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        pygame.draw.rect(self.screen,  Const.COLOR_FOR_NEURON_BODY, (self.location_left, body_y_start, self.location_width, body_height), border_radius=6, width=7)

        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        draw_gradient_rect(self.screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)
        self.screen.blit(label_surface, (self.location_left + 5, self.location_top + 5 + (label_strip_height - label_surface.get_height()) // 2))
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5
        self.screen.blit(output_surface, (right_x, self.location_top + 5 + (label_strip_height - output_surface.get_height()) // 2))

        # Render visual elements
        if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer:
            self.neuron_visualizer.render() #, self, body_y_start)

    def update_neuron(self):
        #print(f"updating neuron {self.nid}")
        if not self.update_avg_error():
            return #no record found so exit early
        self.update_rs()
        self.update_weights()

    def get_max_val_per_wt(self):
        """Retrieves:The maximum absolute weight for each individual weight index across all epochs."""

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
        rs = self.db.query(SQL, (self.model_id, Const.vcr.CUR_ITERATION, self.my_model.display_epoch, self.nid)) # Execute query
        # ✅ Check if `rs` is empty before accessing `rs[0]`
        if not rs:
            return False  # No results found
        self.rs = rs[0]
        self.loss_gradient =  float(rs[0].get("loss_gradient", 0.0))
        self.blame_calculations = rs[0].get("blame_calculations")
        self.neuron_inputs = json.loads( rs[0].get("neuron_inputs"))
        #ez_debug(selfneuinp= self.neuron_inputs)

        # Activation function details
        self.activation_function    = rs[0].get('activation_name', 'Unknown')
        self.activation_value       = rs[0].get('activation_value', None)        #THE OUTPUT
        self.activation_gradient    = rs[0].get('activation_gradient', None)  # From neuron
        #self.banner_text = f"{self.label}  Output: {smart_format( self.activation_value)}"

    def update_avg_error(self):
        SQL = """
        SELECT AVG(ABS(error_signal)) AS avg_error_signal            
        FROM Neuron
        WHERE 
        model   = ? and
        epoch_n = ? and  -- Replace with the current epoch(ChatGPT is trolling us)
        nid     = ?      
        """
        #print(f"In update_avg_error  self.my_model={self.my_model}")
        params = (self.model_id,  self.my_model.display_epoch, self.nid)
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
        weights_data = self.db.query(SQL, (self.model_id, self.nid, self.my_model.display_epoch, Const.vcr.CUR_ITERATION), False)

        if weights_data:
            self.weights = [column[1] for column in weights_data]  # Extract values
            self.weights_before = [column[2] for column in weights_data]  # Extract previous values
        else:
            # TODO: Handle case where no weights are found for the current epoch/iteration
            self.weights = []
            self.weights_before = []

    def initialize_fonts(self):
        self.font_header            = pygame.font.Font(None, Const.TOOLTIP_FONT_HEADER)
        self.font_body              = pygame.font.Font(None, Const.TOOLTIP_FONT_BODY)
        self.header_text            = self.font_header.render("Prediction               Adjust Weights To Improve", True, Const.COLOR_BLACK)

############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################

################### All popup divider lines #############################
    def draw_all_popup_dividers(self):
        self.draw_lines_for_header(0)
        self.draw_lines_for_header(1)
        self.draw_lines_for_weighted_sum(0)
        self.draw_lines_for_weighted_sum(1)
        self.draw_lines_forward_pass_only(0)
        self.draw_lines_forward_pass_only(2)
        self.draw_popup_vertical_divider_between_forward_and_backprop()
        self.draw_highlighted_popup_cell(len(self.weights)+2, 5)
        blame_y = 10 if self.layer == self.output_layer else 12
        self.draw_highlighted_popup_cell(len(self.weights*2)+1, blame_y)

    def draw_lines_for_header(self, extra_row : int):
        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_BLACK,
            (Const.TOOLTIP_PADDING, self.y_coord_for_row(Const.TOOLTIP_LINE_OVER_HEADER_Y + extra_row)),
            (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING, self.y_coord_for_row(Const.TOOLTIP_LINE_OVER_HEADER_Y+ extra_row)),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )

    def draw_lines_for_weighted_sum(self, extra_row : int):
        num_weights = len(self.weights)  # Includes bias and weights
        row_index = 1 + num_weights  # +1 for the header row
        y = self.y_coord_for_row(row_index + extra_row)

        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_GRAY_DARK,
            (Const.TOOLTIP_PADDING, y),
            (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING, y),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )

    def draw_lines_forward_pass_only(self, extra_row : int):
        num_weights = len(self.weights)  # Includes bias and weights
        row_index = 4 + num_weights  # +1 for the header row
        y = self.y_coord_for_row(row_index + extra_row)

        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_GRAY_DARK,
            (Const.TOOLTIP_PADDING, y),
            (self.x_coord_for_col(Const.TOOLTIP_LINE_BEFORE_BACKPROP), y),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )
    def draw_popup_vertical_divider_between_forward_and_backprop(self):
        x = self.x_coord_for_col(Const.TOOLTIP_LINE_BEFORE_BACKPROP)
        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_BLACK,
            (x, Const.TOOLTIP_HEADER_PAD),
            (x, Const.TOOLTIP_HEIGHT - Const.TOOLTIP_PADDING),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS+2
        )

    def draw_highlighted_popup_cell(self, row_index: int, col_index: int):
        x = self.x_coord_for_col(col_index)
        y = self.y_coord_for_row(row_index)
        width = self.column_widths[col_index]
        height = Const.TOOLTIP_ROW_HEIGHT

        # Draw shaded background
        pygame.draw.rect(
            self.cached_tooltip,
            Const.COLOR_HIGHLIGHT_FILL,
            pygame.Rect(x, y, width, height)
        )

        # Draw border
        pygame.draw.rect(
            self.cached_tooltip,
            Const.COLOR_HIGHLIGHT_BORDER,
            pygame.Rect(x, y, width, height),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )


    def y_coord_for_row(self, row_index: int) -> int:
        return Const.TOOLTIP_HEADER_PAD + (row_index * Const.TOOLTIP_ROW_HEIGHT)
    def x_coord_for_col(self, index: int) -> int:
        return Const.TOOLTIP_PADDING + sum(self.column_widths[:index])

    def render_tooltip(self):
        """
        Render the tooltip with neuron details.
        Cache the rendered tooltip and only update if epoch or iteration changes.
        """
                # ✅ Define dynamic column widths (adjust per column)
        #TODO check correct model
        #if Const.configs[0].optimizer != Optimizer_SGD:
        self.column_widths = [45, 50, 10, 69, 15, 69,   #This ends the forward prop columns
                              69, 10, 69, 15, 69, 15, 69, 69, 60, 100]
        #else:
        #self.column_widths = [45, 50, 10, 60, 15, 69, 60, 10, #ends on first op  in backprop
        #                 69, 15, 155, 115, 160, 60, 60, 100]


        # ✅ Check if we need to redraw the tooltip
        if not hasattr(self, "cached_tooltip") or self.last_epoch != self.my_model.display_epoch or self.last_iteration != Const.vcr.CUR_ITERATION:
            self.last_epoch = self.my_model.display_epoch  # ✅ Update last known epoch
            self.last_iteration = Const.vcr.CUR_ITERATION  # ✅ Update last known iteration

            # ✅ Tooltip dimensions
            tooltip_width = Const.TOOLTIP_WIDTH
            tooltip_height = Const.TOOLTIP_HEIGHT

            # ✅ Create a new surface for the tooltip
            self.cached_tooltip = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)

            # ✅ Fill background and draw border
            self.cached_tooltip.fill(Const.COLOR_CREAM)
            pygame.draw.rect(self.cached_tooltip, Const.COLOR_BLACK, (0, 0, tooltip_width, tooltip_height), 2)

            # ✅ Draw header
            self.cached_tooltip.blit(self.header_text, (Const.TOOLTIP_PADDING, Const.TOOLTIP_PADDING))

            # ✅ Populate content
            self.tooltip_generate_text()

            # ✅ Ensure column widths match number of columns
            #if len(column_widths) < len(self.tooltip_columns):
            #    column_widths.extend([Const.TOOLTIP_COL_WIDTH] * (len(self.tooltip_columns) - len(column_widths)))
            self.draw_all_popup_dividers()
            # ✅ Draw each column with dynamic spacing
            x_offset = Const.TOOLTIP_PADDING
            for col_index, (column, col_width) in enumerate(zip(self.tooltip_columns, self.column_widths)):
                for row_index, text in enumerate(column):
                    text_color = self.get_text_color(col_index, row_index, text)
                    text = self.smart_format_for_popup(text)
                    #label = self.font_body.render(str(text), True, text_color)
                    #self.cached_tooltip.blit(label, (                        x_offset,                        Const.TOOLTIP_HEADER_PAD + row_index * Const.TOOLTIP_ROW_HEIGHT + Const.TOOLTIP_PADDING                    ))
                    label = self.font_body.render(str(text), True, text_color)
                    text_rect = label.get_rect()
                    y_pos = Const.TOOLTIP_HEADER_PAD + row_index * Const.TOOLTIP_ROW_HEIGHT + Const.TOOLTIP_PADDING
                    x_pos = x_offset

                    if self.is_right_aligned(text):
                        text_rect.topright = (x_offset + col_width - Const.TOOLTIP_PADDING, y_pos)
                    else:
                        text_rect.topleft = (x_offset + Const.TOOLTIP_PADDING, y_pos)

                    self.cached_tooltip.blit(label, text_rect)

                x_offset += col_width  # ✅ Move X position based on column width



        # ✅ Get mouse position and adjust tooltip placement
        mouse_x, mouse_y = pygame.mouse.get_pos()
        tooltip_x = self.adjust_position(mouse_x + Const.TOOLTIP_PLACEMENT_X, Const.TOOLTIP_WIDTH, Const.SCREEN_WIDTH)
        tooltip_y = self.adjust_position(mouse_y + Const.TOOLTIP_PLACEMENT_Y, Const.TOOLTIP_HEIGHT, Const.SCREEN_HEIGHT)

        # ✅ Draw cached tooltip onto the screen
        Const.SCREEN.blit(self.cached_tooltip, (tooltip_x, tooltip_y))

    def get_text_color(self,col_index, row_index, text):
        #if col_index == 7 and row_index > 0 and text:
        if  row_index > 11111 and text:
            if is_numeric(text):
                value = float(text.replace(",", ""))
                return Const.COLOR_GREEN_FOREST if value >= 0 else Const.COLOR_CRIMSON
        return Const.COLOR_BLACK

    def adjust_position(self, position, size, screen_size):
        # If the tooltip would overflow to the right
        if position + size > screen_size:
            position = screen_size - size - Const.TOOLTIP_ADJUST_PAD

        # If the tooltip would overflow to the left
        if position < Const.TOOLTIP_ADJUST_PAD:
            position = Const.TOOLTIP_ADJUST_PAD

        return position


    def tooltip_generate_text(self):
        """Clears and regenerates tooltip text columns."""
        #print("Generating text")
        self.tooltip_columns.clear()
        self.tooltip_columns.extend(self.tooltip_columns_for_forward_pass())
        self.tooltip_columns.extend(self.tooltip_columns_for_backprop())  # 🔹 Uncomment when backprop data is ready

################### Gather Values for Forward Pass #############################
################### Gather Values for Forward Pass #############################
################### Gather Values for Forward Pass #############################
    def tooltip_columns_for_forward_pass_row_labels(self, inputs):
        labels = ["Cog", "Bias"]
        for i,inp in enumerate(inputs[:-2]):
            labels.append(f"Wt {i+1}")
        labels.append("Raw Sum")
        return labels

    def tooltip_columns_for_forward_pass(self):

        #Next we need the actual inputs.
        iteration_data = Const.dm.get_model_iteration_data(self.model_id)
        all_columns = []
        inputs          = self.tooltip_column_forward_pass_one_inputs(iteration_data) #first item on the list is the literal "Input"

        row_labels = self.tooltip_columns_for_forward_pass_row_labels(inputs)
        all_columns.append(row_labels)
        all_columns.append(inputs)

        # Multiply signs
        multiply_signs = ["*", " "]   # the equals is for bias
        multiply_signs.extend(["*"] * (len(inputs)-2))
        all_columns.append(multiply_signs)
        weights=["Weight"]
        weights.extend(self.weights_before)
        #ez_debug(wt_with_lbl = weights)
        all_columns.append(weights)
        all_columns.append(["="] * (len(inputs) + 2) ) # col_op1
        #all_columns.extend([["="] * (len(inputs) + 1)," ", "=" ) # col_op1


        # weighted product
        # Slice inpts to start from the 3rd item (index 2) and wt_before to start from the 2nd item (index 1)
        inputs_sliced = inputs[2:]  # Slices from index 2 to the end
        wt_before_sliced = weights[2:]  # Slices from index 1 to the end
        products = [inp * wt for inp, wt in zip(inputs_sliced, wt_before_sliced)]
        product_col = ["Product", weights[1]]    #Label and bias
        product_col.extend(products)
        weighted_sum = sum(product_col[1:])     # Sums everything except the first element - calculate weighted sum
        product_col.append(weighted_sum)
        row_labels.append(f"{self.activation_function}({smart_format(weighted_sum)})")

        product_col.append(self.activation_value)
        all_columns.append(product_col)
        #ez_debug(wt_before = self.weights_before)
        #print(products)
        #ez_debug(all_columns_after_inputs=all_columns)

        row_labels.append("") #Blank row after output
        product_col.append("") #Blank row after output
        row_labels.append("Act Gradient")
        product_col.append(self.activation_gradient)
        row_labels.append( get_activation_derivative_formula(f"{self.activation_function}"))
        #TODO only add below if space permits
        row_labels.extend(["(How much 'Raw Sum' contri-","butes to final prediction)"])        #So, for hidden neurons, a better description might be something like:ow much the neuron's raw sum, after being transformed by its activation function, contributes to the propagation of error through the network."
        inputs[1] = " "  #"N/A" # remove the 1 for bias
        return all_columns

    def tooltip_column_forward_pass_one_inputs(self,iteration_data):
        input_col =["Input","1"]
        input_col.extend(self.neuron_inputs)
        return input_col

################### Gather Values for Back Pass #############################
################### Gather Values for Back Pass #############################
################### Gather Values for Back Pass #############################

    def tooltip_columns_for_backprop(self):
        all_columns = self.tooltip_columns_for_backprop_error_distribution()
        weights=  ["    Orig"]
        weights.extend(self.weights_before)
        all_columns.append(weights)
        weights = ["    New"]
        weights.extend(self.weights)
        all_columns.append(weights)
        all_columns = self.tooltip_columns_for_error_signal_calculation(all_columns)
        return all_columns

    def tooltip_columns_for_backprop_error_distribution(self):
        """
        Populates tooltip columns with backprop calculations for the hovered neuron.
        Queries the database using nid, model_id, epoch, iteration and orders by weight_id.
        """

        # Query weight updates from DB
        sql = """
        SELECT weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result 
        FROM DistributeErrorCalcs 
        WHERE  epoch = ? AND iteration = ? AND model_id = ? AND nid = ? 
        ORDER BY weight_index ASC
        """

        sql2 = """
        SELECT A.weight_index, A.arg_1, A.op_1, A.arg_2,A.op_2,A.arg_3,A.op_3, A.result--, B.arg_3 AS batch_arg_3         -- 🧠 Pulled from last sample in batch
        FROM DistributeErrorCalcs A
        JOIN DistributeErrorCalcs B
          ON A.model_id = B.model_id
          AND A.epoch = B.epoch
          AND A.nid = B.nid
          AND A.weight_index = B.weight_index
          AND A.batch_id = B.batch_id
          AND B.iteration = (SELECT MAX(iteration)
              FROM ErrorSignalCalcs
              WHERE epoch = A.epoch
                AND model_id = A.model_id
                AND nid = A.nid
                AND weight_id = A.weight_index
                AND B.batch_id = A.batch_id       )
        WHERE A.epoch = ?          AND A.iteration = ?          AND A.model_id = ?          AND A.nid = ?        ORDER BY A.weight_index ASC
        """


        #ez_debug(epoch=self.my_model.display_epoch, iter=Const.vcr.CUR_ITERATION,model= self.model_id, nid=self.nid)
        #Const.dm.db.query_print("SELECT epoch, iteration,  count(1) FROM DistributeErrorCalcs GROUP BY epoch, iteration ")
        #Const.dm.db.query_print("SELECT * FROM DistributeErrorCalcs WHERE iteration = 2 and nid = 0")
        results = Const.dm.db.query(sql, (self.my_model.display_epoch, Const.vcr.CUR_ITERATION, self.model_id, self.nid ), as_dict=False)
        if not results:
            return []  # ✅ No data found, exit early

        #ez_debug(results=results)

        # ✅ Initialize columns for backpropagation
        print (f"self.config.backprop_headers={self.config.backprop_headers}")
        col_input   = [self.config.backprop_headers[0]] #col_input = ["Input1"]
        col_op1     = [self.config.backprop_headers[1]] #col_op1 = ["*"]
        col_err_sig = [self.config.backprop_headers[2]] # col_err_sig = ["Accp Blm"]
        col_op2     = [self.config.backprop_headers[3]] # col_op2 = ["*"]
        col_lrate   = [self.config.backprop_headers[4]] # col_lrate = ["LRate"]
        col_op3     = [self.config.backprop_headers[5]] # col_op3 = ["="]
        col_adj     = [self.config.backprop_headers[6]] # col_adj = ["      Adj"]




        # ✅ Loop through results and populate columns
        for row in results:
            weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result_value = row
            col_input.append(arg_1)  # Input
            col_op1.append(op_1)  # ×
            col_err_sig.append(arg_2)  # Blame
            col_op2.append(op_2)  # ×
            col_lrate.append(arg_3)  # Learning Rate
            col_op3.append(op_3)  # =
            col_adj.append(result_value)   # Prev Activation

        # ✅ Append all columns in the correct order
        all_columns = []
        all_columns.append(col_input)
        all_columns.append(col_op1)
        all_columns.append(col_err_sig)
        all_columns.append(col_op2)
        all_columns.append(col_lrate)
        all_columns.append(col_op3)
        all_columns.append(col_adj)
        col_input[1] = " "     #"N/A"    #  remove the 1 for bias
        col_op1[1] = " "    #  remove the * for bias
        return all_columns

    def tooltip_columns_for_error_signal_calculation(self, all_cols):
        # Row in the box between adj and blame
        #print(f"len(all_cols)={len(all_cols)}")
        for i in range(len(all_cols)):  #Prints blank row, empty space in each cell
            if i == 0:
                all_cols[0].append("Why I'm to Blame??? (My Responsibility)")
            else:
                all_cols[i].append(" ")



        if self.layer == self.output_layer: # This is an output neuron
            return self.tooltip_columns_for_error_sig_outputlayer(all_cols)
        else:
            return self.tooltip_columns_for_error_sig_hiddenlayer(all_cols)

    # Better Terms for backprob... blame, yelling, accepted blame

    def tooltip_columns_for_error_sig_outputlayer(self, all_cols):
        all_cols[0].append("Accepted Blame Calculation Below")
        all_cols[0].append( f"Accepted Blame = Loss Gradient * Activation Gradient")
        all_cols[0].extend([f"Accepted Blame = {smart_format( self.loss_gradient)} * {smart_format(self.activation_gradient)} = {smart_format(self.loss_gradient * self.activation_gradient)}"])
        return all_cols

    def tooltip_columns_for_error_sig_hiddenlayer(self, all_cols):
        col_weight = 0
        col_errsig = 3
        col_contri = 6
        all_cols[col_weight].append("Accepted Blame Calculation Below")
        all_cols[col_errsig-1].append(" ")
        all_cols[col_errsig].append(" ")
        all_cols[col_contri-1].append(" ")
        all_cols[col_contri].append(" ")
        # IWFM = It's weight From Me
        all_cols[col_weight].append("IWFM")
        all_cols[col_errsig].append("It's Blame")
        all_cols[col_contri].append("My Share of it's blame")
        arg_1, arg_2 = self.get_elements_of_backproped_error()
        contributions = [a * b for a, b in zip(arg_1, arg_2)]
        all_cols[col_weight].extend(arg_1)
        all_cols[col_errsig-1].extend("*" * (len(arg_1)+1))
        all_cols[col_errsig].extend(arg_2)
        all_cols[col_contri-1].extend("=" * (len(arg_1)+3))
        all_cols[col_contri].extend(contributions)
        bpe=sum(contributions)
        all_cols[col_weight].append("My Blame from All")
        all_cols[col_weight].append("Accp Blame = MBFA * Act Grad")
        all_cols[col_contri].append(bpe)
        all_cols[col_contri].append(bpe*self.activation_gradient)
        #all_cols[col_weight].append(f"BackPropped Error = {smart_format(bpe)}")
        #all_cols[col_weight].append(f"Blame = {smart_format(bpe*self.activation_gradient)}")
        return all_cols

    def get_elements_of_backproped_error(self):
        """Fetches elements required to calculate backpropogated error for a hidden neuron"""
        SQL = """
            SELECT arg_1, arg_2
            FROM ErrorSignalCalcs
            WHERE model_id = ? AND nid = ? AND epoch = ? AND iteration = ?
            ORDER BY weight_id ASC
        """
        backprop_error_elements = self.db.query(SQL, (self.model_id, self.nid, self.my_model.display_epoch, Const.vcr.CUR_ITERATION), False)

        if backprop_error_elements:             # Split the elements into two lists using the helper function
            list1, list2 = self.split_error_elements(backprop_error_elements)
            return list1, list2
        else:
            return [],[]

    def split_error_elements(self,elements):
        """
        Splits a list of tuples into two lists.
        Each tuple is expected to have two elements.

        Parameters:
            elements (list of tuple): List of tuples to split.

        Returns:
            tuple: A tuple containing two lists:
                   - The first list contains the first element of each tuple.
                   - The second list contains the second element of each tuple.
        """
        if elements:
            # Use zip to transpose the list of tuples and then convert each tuple to a list
            first_list, second_list = map(list, zip(*elements))
            return first_list, second_list
        return [], []

    def smart_format_for_popup(self, num):
        try:
            num = float(num)  # Ensure input is a number
        except (ValueError, TypeError):

            return str(num)  # If conversion fails, return as is

        if num == 0:
            return "0"
        #elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
        #    return f"{num:.2e}"
        elif abs(num) < 0.001:  # Use 6 decimal places for small numbers
            #formatted = f"{num:,.6f}"
            return f"{num:.1e}"
        elif abs(num) < 1:  # Use 3 decimal places for numbers less than 1
            formatted = f"{num:,.3f}"
        elif abs(num) > 1e6:  # Use 6 decimal places for small numbers
            return f"{num:.1e}"
        elif abs(num) > 1000:  # Use no decimal places for large numbers
            formatted = f"{num:,.0f}"

        else:  # Default to 2 decimal places
            formatted = f"{num:,.2f}"

        # Remove trailing zeros and trailing decimal point if necessary
        return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

    def is_right_aligned(self, text):
        if isinstance(text, (int, float)):
            return True
        if isinstance(text, str):
            cleaned = text.replace(",", "").strip()
            return cleaned.upper() in ["N/A", "NONE"] or cleaned.replace(".", "", 1).replace("-", "", 1).isdigit()
        return False
