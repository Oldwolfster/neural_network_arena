import json
from ast import literal_eval
from typing import List
import pygame

from src.engine.ActivationFunction import get_activation_derivative_formula
from src.neuroForge import mgr
from src.neuroForge.EZPrint import EZPrint
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format, draw_gradient_rect
from src.neuroForge.mgr import * # Imports everything into the local namespace


class DisplayModel__Neuron:
    input_values = []   # Class variable to store inputs
    def __init__(self, nid:int, layer: int, position: int, output_layer: int, text_version: str):
        #print(f"Instantiating neuron Pnid={nid}\tlabel={label}")


        # Attach Display Strategy
        self.display_strategy = display_strategy

        self.location_left=0
        self.location_top=0
        self.location_width=0
        self.location_height = 0
        self.nid = nid
        self.layer = layer
        self.position = position
        self.output_layer = output_layer
        #print(f"OUTPUT LAYER{output_layer}")
        self.label = f"{layer}-{position}" #need to define, try to use existing standard
        self.weights = []
        self.neuron_inputs = []
        self.error_signal = 1
        self.bias = 0
        self.weight_count = []
        self.raw_sum = 0
        self.activation_function = ""
        self.activation_value =0
        self.activation_gradient = 0
        self.weight_text = ""
        self.banner_text = ""
        self.mouse_x = 0
        self.mouse_y = 0
        self.tooltip_columns = []
        self.weight_adjustments = ""
        self.error_signal_calcs = ""
        self.avg_err_sig_for_epoch = 0.0
        self.loss_gradient = 0.0    #Same for all neurons
        self.neuron_build_text = self.neuron_build_text_large if text_version == "Verbose" else self.neuron_build_text_small
        # Create EZPrint instance
        self.ez_printer = EZPrint(pygame.font.Font(None, 24)
                                  , color=(0, 0, 0), max_width=200, max_height=100, sentinel_char="\n")
    def is_hovered(self, offset_x : int, offset_y : int): #"""Check if the mouse is over this neuron."""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.mouse_x =  mouse_x-offset_x
        self.mouse_y = mouse_y - offset_y  #print(f"left={self.location_left}\twidth={self.location_left + self.location_width  }\tmouse={mouse_x}tmouse2={self.mouse_x}")
        return (self.location_left <= self.mouse_x <= self.location_left + self.location_width and self.location_top <= self.mouse_y <= self.location_top + self.location_height)



    def get_contrasting_text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        """
        Given a background RGB color, this function returns an RGB tuple for either black or white text,
        whichever offers better readability.

        The brightness is computed using the formula:
            brightness = (R * 299 + G * 587 + B * 114) / 1000
        which is a standard formula for perceived brightness. If the brightness is greater than 128,
        the background is considered light and black text is returned; otherwise, white text is returned.

        Parameters:
            rgb (tuple[int, int, int]): A tuple representing the background color (R, G, B).

        Returns:
            tuple[int, int, int]: An RGB tuple for the text color (either (0, 0, 0) for black or (255, 255, 255) for white).
        """
        r, g, b = rgb
        # Calculate the perceived brightness of the background color.
        brightness = (r * 299 + g * 587 + b * 114) / 1000

        # Choose black text for light backgrounds and white text for dark backgrounds.
        if brightness > 128:
            return (0, 0, 0)  # Black text for lighter backgrounds.
        else:
            return (255, 255, 255)  # White text for darker backgrounds.

    """
    # Example usage:
    if __name__ == "__main__":
        # Example background colors:
        examples = [
            (255, 255, 255),  # white background -> should use black text
            (0, 0, 0),        # black background -> should use white text
            (100, 150, 200)   # medium background -> decision based on brightness
        ]
        
        for bg in examples:
            text_color = get_contrasting_text_color(bg)
            print(f"Background color {bg} -> Contrasting text color {text_color}")
    """


    @classmethod
    def retrieve_inputs(cls, db: RamDB, iteration: int, epoch: int, modelID: str):
        """
        Retrieve inputs from the database and store in the class variable.
        """
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """
        params = (epoch, iteration)

        # Execute query
        rs = db.query(sql, params)

        # Parse and store inputs
        if rs:
            raw_inputs = rs[0].get("inputs", cls.input_values)
            try:
                cls.input_values = json.loads(raw_inputs)
            except json.JSONDecodeError:
                cls.input_values = literal_eval(raw_inputs)


    def update_neuron(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        if not self.update_avg_error(db, iteration, epoch, model_id):
            return #no record found so exit early
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


        params = (model_id, iteration, epoch, self.nid)
        # print(f"SQL in update_me: {SQL}")
        # print(f"Params: {params}")

        rs = db.query(SQL, params) # Execute query
        try:
            self.weight_text = self.neuron_build_text(rs[0])
            self.loss_gradient =  float(rs[0].get("loss_gradient", 0.0))
            self.error_signal_calcs = rs[0].get("error_signal_calcs")
            #print(f"calcsforerror{self.error_signal_calcs}")
            self.banner_text = f"{self.label}  Output: {smart_format( self.activation_value)}"
            #print(f"Query result: {rs}")
            #print(f"PREDICTIONS: {self.weight_text}")
        except:
            pass

    def update_avg_error(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        #print(f"model_id======={model_id}")
        SQL ="""
        SELECT AVG(ABS(error_signal)) AS avg_error_signal            
        FROM Neuron
        WHERE 
        model   = ? and
        epoch_n = ? and  -- Replace with the current epoch(ChatGPT is trolling us)
        nid     = ?        
        """
        params = (model_id,  epoch, self.nid)
        rs = db.query(SQL, params)  # Execute query

        # ✅ Check if `rs` is empty before accessing `rs[0]`
        if not rs:
            #print("in update_avg_error returning false")
            return False  # No results found

        # ✅ Ensure `None` does not cause an error
        self.avg_err_sig_for_epoch = float(rs[0].get("avg_error_signal") or 0.0)
        #print("in update_avg_error returning TRUE")
        return True


    def get_color_gradient(self, error_signal, max_error: float, gamma: float = 2.0):
        """
        Maps an absolute error signal to a color gradient from red (high error) to green (low error)
        using gamma correction to better utilize the color spectrum. This is a variant of get_color_gradient
        that increases sensitivity in the lower error range.

        Parameters:
            error_signal (float): The error signal for the neuron.
            max_error (float): The maximum error signal used for normalization.
            gamma (float): Gamma correction factor. A value > 1 will enhance differences at the low end.

        Returns:
            tuple: An (R, G, B) color where red intensity increases with error.
        """
        if max_error != 0:
            norm_error = min(abs(error_signal) / max_error, 1)  # Normalize to [0, 1]
            # Apply gamma correction to increase sensitivity for lower error values.
            # With gamma > 1, even small errors get amplified.
            norm_error = norm_error ** (1 / gamma)
            red = int(255 * norm_error)          # High error → More red
            green = int(255 * (1 - norm_error))    # Low error → More green
        else:
            red = 255
            green = 0
        return (red, green, 0)  # RGB format

    def get_color_gradient_shows1_red(self, error_signal, max_error: float, gamma: float = 2.0):
        if max_error != 0:
            norm_error = min(abs(error_signal) / max_error, 1)
            norm_error = norm_error ** (1/gamma)  # Apply gamma correction
            red = int(255 * norm_error)
            green = int(255 * (1 - norm_error))
        else:
            red = 255
            green = 0
        return (red, green, 0)

    def get_color_gradient(self, error_signal, max_error : float):
        """Maps an absolute error signal to a color gradient from red (high) to green (low)."""

        #print(f"error_signal={error_signal} and max_error={max_error} and {mgr.max_error}")
        if max_error != 0:
            norm_error = min(abs(error_signal) / max_error, 1)  # Normalize to [0, 1]
            red = int(255 * norm_error)   # High error → More red
            green = int(255 * (1 - norm_error))  # Low error → More green
        else:
            red =255
            green = 0
        return (red, green, 0)  # RGB format


    def draw_neuron(self, screen):
        # Define colors
        body_color = (0, 0, 255)  # Blue for the neuron body
        #label_color = (70, 130, 180)  # Steel blue for the label strip
        text_color = (255, 255, 255)  # White for text on the label

        if color_neurons:
            body_color = self.get_color_gradient(self.avg_err_sig_for_epoch,mgr.max_error)

        # Font setup
        font = pygame.font.Font(None, 24)

        # **Split banner text into two parts:**
        label_text = f"ID: {self.label}"  # Left side (Neuron ID)
        output_text = f"{smart_format(self.activation_value)}"  # Right side

        # Render both texts separately
        label_surface = font.render(label_text, True, text_color)
        output_surface = font.render(output_text, True, text_color)

        # Get text dimensions
        text_height = label_surface.get_height()
        label_strip_height = text_height + 8  # Padding (8px)

        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top+4, self.location_width, label_strip_height)
        draw_gradient_rect(screen, banner_rect, (25, 25, 112), (70, 130, 180))
        text_height -= 6
        screen.blit(label_surface, (self.location_left + 5, self.location_top + (label_strip_height - text_height) // 2)) # **Blit Label on the Left**
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5  # Align to right  # **Blit Output on the Right**
        screen.blit(output_surface, (right_x, self.location_top + (label_strip_height - text_height) // 2))


        # Draw the neuron body below the label
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        pygame.draw.rect(
            screen,
            body_color,
            (self.location_left, body_y_start, self.location_width, body_height),
            border_radius=6,
            width= 5  # Border width
        )


        # Render neuron details inside the body
        body_text_y_start = body_y_start + 5
        self.ez_printer.render(
            screen,
            text=self.weight_text,
            x=self.location_left + 11,
            y=body_text_y_start + 7
        )

    def neuron_build_text_small(self, row): #less info so it still fits
        self.neuron_build_text_large(row) # Ensures values are saved for the pop up window.
        error_signal = row.get('error_signal', None)  # From neuron
        return f"{smart_format(self.activation_value)}\n{self.activation_function}\nδ={smart_format(error_signal)}"

    def neuron_build_text_large(self, row): #lots of info
        """
        Generate a formatted report for a single neuron.
        Includes weighted sum calculations, bias, activation details,
        and backpropagation details (activation gradient, error signal).
        """
        prediction_logic = self.build_prediction_logic(row)
        bias_activation_info = self.format_bias_and_activation(row)
        backprop_details = self.format_backpropagation_details(row)  # 🔥 New Function!
        #print(row)
        self.weight_adjustments =  row.get('weight_adjustments')
        #return f"{prediction_logic}\n{bias_activation_info}\n{backprop_details}\n{self.weight_adjustments}"
        return f"{prediction_logic}\n{bias_activation_info}\n{backprop_details}"

    # ---------------------- Logic around metrics ---------------------- #

    def build_prediction_logic(self, row):
        """
        Compute weighted sum calculations and format them.
        """
        nid = row.get('nid')  # Get neuron ID
        self.weights = json.loads(row.get('weights_before', '[]'))  # Deserialize weights
        self.neuron_inputs = json.loads(row.get('neuron_inputs', '[]'))  # Deserialize inputs

        # Generate weighted sum calculations
        predictions = []
        self.raw_sum = 0

        for i, (w, inp) in enumerate(zip(self.weights, self.neuron_inputs), start=1):
            linesum = (w * inp)
            calculation = f"{smart_format(inp)} * {smart_format(w)} = {smart_format(linesum)}"
            predictions.append(calculation)
            self.raw_sum += linesum  # Accumulate weighted sum

        return '\n'.join(predictions)

    def format_bias_and_activation(self, row):
        """
        Format the bias, raw sum, and activation function for display.
        """
        self.bias = row.get('bias_before', 0)
        self.raw_sum += self.bias

        # Activation function details
        self.activation_function = row.get('activation_name', 'Unknown')
        self.activation_value = row.get('activation_value', None)        #THE OUTPUT
        self.activation_gradient = row.get('activation_gradient', None)  # From neuron

        # Format strings
        bias_str = f"Bias: {smart_format(self.bias)}"
        raw_sum_str = f"Raw Sum: {smart_format(self.raw_sum)}"
        #activation_str = f"{activation_name}: {smart_format(activation_value)}" if activation_value is not None else ""
        activation_str = f"{self.activation_function} Gradient: {smart_format(self.activation_gradient)}"
        return f"{bias_str}\n{raw_sum_str}\n{activation_str}"

    # ---------------------- 🔥 New Function! 🔥 ---------------------- #

    def format_backpropagation_details(self, row):
        """
        Format and display backpropagation details:
        - Activation Gradient (A')
        - Error Signal (δ)
        """
        self.error_signal = row.get('error_signal', None)  # From neuron
        error_signal_str = f"Error Signal (δ): {smart_format(self.error_signal)}"
        return f"{error_signal_str}"

    def parse_weight_adjustments(self,text: str):
        lines = text.strip().splitlines()
        parameters = []
        weights = []
        learning_rates = []
        error_signals = []
        inputs = []
        adjustments = []
        new_weights = []

        for line in lines:
            # Split on '=' to separate the parameter name from the rest.
            param_part, rest = line.split('=', 1)
            parameters.append(param_part.strip())
            weight_float=0.0
            # Split on '+' to isolate the weight.
            if '+' in rest:
                weight_str, rest = rest.split('+', 1)
                weights.append(smart_format(float(weight_str.strip())))

            else:
                weights.append(smart_format(float(rest.strip())))
                rest = ""

            # Now, the rest should contain the learning rate, error signal, and possibly the input.
            parts = [part.strip() for part in rest.split('*') if part.strip()]
            learning_rates.append(smart_format(float( parts[0])) if len(parts) >= 1 else "")
            error_signals.append(smart_format(float( parts[1])) if len(parts) >= 2 else "")
            inputs.append(smart_format(float(parts[2])) if len(parts) >= 3 else "")

            if len(parts) >= 3:
                adjustments.append(  smart_format( float(parts[0]) * float(parts[1]) * float(parts[2])))
            else:
                adjustments.append(smart_format(float(parts[0].replace(",", "")) * float(parts[1].replace(",", ""))))

            new_weights.append((float(adjustments[-1].replace(",", "")) + float(weights[-1].replace(",", ""))))
            new_weights.append((adjustments[-1] + weights[-1]))
        backprop_data = [parameters, weights, learning_rates, error_signals, inputs, adjustments, new_weights]

        return backprop_data

    def parse_error_signal(self, input_string):
        """
        this input  "0.1!-0.01@0.2!-0.02@0.3!0.03@"
        should generate [[0.1, 0.2, 0.3], [-0.01, -0.02, 0.03]]
        """
        # Split the string into pairs using '@' as the delimiter
        pairs = input_string.strip('@').split('@')

        # Initialize two empty lists to store the first and second numbers of each pair
        err_sigs = []
        weights = []
        for pair in pairs:
            if not pair or '!' not in pair:
                print(f"⚠️ Skipping invalid pair: {repr(pair)}", flush=True)
                continue  # Skip this iteration

            #print(f"Pair={pair}", flush=True)
            try:
                num1, num2 = pair.split('!')
            except Exception as e:
                print(f"Error while splitting {repr(pair)}", flush=True)
                raise


        # Iterate through each pair
        #for pair in pairs:
            # Split the pair into two numbers using '!' as the delimiter
        #    print(f"Pair={pair}")
        #    try:
        #        num1, num2 = pair.split('!')
        #    except:
        #        print(f"Pair2={pair}")
        #        raise

            # Convert the strings to floats and append to the respective lists
            weights.append(float(num1))
            err_sigs.append(float(num2))
        contributions = [w * e for w, e in zip(weights, err_sigs)]
        return [weights, err_sigs, contributions] # Return the three lists as a list of lists

    def tooltip_generate_text(self):
        self.tooltip_columns.clear()
        self.tooltip_columns_for_forward_pass()
        self.tooltip_columns_for_backprop()


    def tooltip_columns_for_error_sig(self):
        if self.layer == self.output_layer: # This is an output neuron
            self.tooltip_columns_for_error_sig_outputlayer()
        else:
            self.tooltip_columns_for_error_sig_hiddenlayer()

    def tooltip_columns_for_error_sig_outputlayer(self):
        self.tooltip_columns[4].extend(["Error Signal = Loss Gradient * Activation Gradient"])
        self.tooltip_columns[4].extend([f"Error Signal = {smart_format( self.loss_gradient)} * {smart_format(self.activation_gradient)} = {smart_format(self.loss_gradient * self.activation_gradient)}"])

    def tooltip_columns_for_error_sig_hiddenlayer(self):
        weights, err_sig, contributions = self.parse_error_signal(self.error_signal_calcs)
        from_neurons = self.generate_from_neuron_labels(len(weights), self.layer + 1) #the +1 gets the next layer
        string_ver_weights = [str(w) for w in weights]
        string_ver_err_sig = [str(w) for w in err_sig]
        string_ver_contributions = [str(w) for w in contributions]


        self.tooltip_columns[4].extend(["Neuron"] + from_neurons + ["Total Sum:", "Err Signal: Sum * Act Gradient ="]   )
        self.tooltip_columns[5].extend(["", "", "Wt Val"] + string_ver_weights )
        self.tooltip_columns[6].extend(["", "", "Err Sig"] + string_ver_err_sig)
        self.tooltip_columns[7].extend(["", "", "Weighted Contribution"])
        self.tooltip_columns[8].extend(["", "", "", *string_ver_contributions, smart_format(sum(contributions)), smart_format(sum(contributions)* self.activation_gradient)]) # +  + sum(contributions))



        #print(f"{self.tooltip_columns[5]}")

    def generate_from_neuron_labels(self, num_of_neurons: int, layer_id: int):
        # Generate a list of neuron labels in the format "layer_id,neuron_index"
        return [f"{layer_id},{i}" for i in range(num_of_neurons)]

    def tooltip_columns_for_backprop(self):
        temp_list = [""]    #Blank column dividing forward and back

        self.tooltip_columns.append(temp_list)
        temp_list = ["Input"]
        self.tooltip_columns.append(temp_list)
        temp_list = ["Err Sig"]
        self.tooltip_columns.append(temp_list)
        temp_list = ["L Rate"]
        self.tooltip_columns.append(temp_list)
        temp_list = ["ADJ"]
        self.tooltip_columns.append(temp_list)
        temp_list = ["Old Wt"]
        self.tooltip_columns.append(temp_list)
        temp_list = ["New Wt"]
        self.tooltip_columns.append(temp_list)
        #names = list(map(lambda name: "hi " + name, names))
        bp_info = self.parse_weight_adjustments (self.weight_adjustments)
        #self.tooltip_columns[3].extend(bp_info[0]) # Paramter name i.e.WW1,W2, B

        self.tooltip_columns[4].extend(bp_info[4]) # LR
        self.tooltip_columns[5].extend(bp_info[3]) # Err sig
        self.tooltip_columns[6].extend(bp_info[2]) # Input Magnitude
        self.tooltip_columns[7].extend(bp_info[5]) # Adjustment
        self.tooltip_columns[8].extend(bp_info[1]) # original Weight
        self.tooltip_columns[9].extend(bp_info[6]) # new Weight
        self.tooltip_columns[4].extend(["Input * Error Signal * Learning Rate = Adjustment",""]) #Includes blank line
        self.tooltip_columns_for_error_sig()

    def tooltip_columns_for_forward_pass(self):
        temp_list = ["Input"]
        self.tooltip_columns.append(temp_list)
        temp_list = ["* Weight"]
        self.tooltip_columns.append(temp_list)
        temp_list = [""]
        self.tooltip_columns.append(temp_list)
        self.raw_sum = 0
        for i, (w, inp) in enumerate(zip(self.weights, self.neuron_inputs), start=1):
            linesum = (w * inp)
            self.tooltip_columns[0].append(f"{smart_format(inp)}")
            #self.col_1.append(f"#{i}: {smart_format(inp)}")
            self.tooltip_columns[1].append(f"* {smart_format(w)}")
            self.tooltip_columns[2].append(f"= {smart_format(linesum)}")
            self.raw_sum += linesum  # Accumulate weighted sum
        self.raw_sum += self.bias
        self.tooltip_columns[0].append("Add Bias")
        self.tooltip_columns[1].append("")
        self.tooltip_columns[2].append(f"+ {smart_format(self.bias)}")

        self.tooltip_columns[0].append(f"Raw Sum")
        self.tooltip_columns[1].append("")
        self.tooltip_columns[2].append(f"= {smart_format(self.raw_sum)}")
        self.tooltip_columns[0].append("")
        self.tooltip_columns[1].append("")
        self.tooltip_columns[2].append("")
        self.tooltip_columns[0].append(f"Act Function")
        self.tooltip_columns[1].append("")
        self.tooltip_columns[2].append(f"= {self.activation_function}")
        self.tooltip_columns[0].append(f"OUTPUT")
        self.tooltip_columns[1].append("")
        self.tooltip_columns[2].append(f"= {smart_format(self.activation_value)}")
        self.tooltip_columns[0].append("")

        self.tooltip_columns[2].append("")
        self.tooltip_columns[0].append(f"Act Gradient")
        self.tooltip_columns[2].append(f"= {smart_format(self.activation_gradient)}")
        self.tooltip_columns[0].append( get_activation_derivative_formula(f"{self.activation_function}"))
        self.tooltip_columns[0].extend(["(How much 'Raw Sum' contri-","butes to final prediction)"])        #So, for hidden neurons, a better description might be something like:ow much the neuron's raw sum, after being transformed by its activation function, contributes to the propagation of error through the network."

    def render_tooltip(self, screen):   #"""Render the tooltip with neuron details."""
        tooltip_width = 619
        tooltip_height = 300
        tooltip_x = self.mouse_x + 10
        tooltip_y = self.mouse_y + 10

        # Ensure tooltip doesn't go off screen
        if tooltip_x + tooltip_width > screen.get_width():
            tooltip_x -= tooltip_width + 20
        if tooltip_y + tooltip_height > screen.get_height():
            tooltip_y -= tooltip_height + 20

        # Draw background box
        pygame.draw.rect(screen, (255, 255, 200), (tooltip_x, tooltip_y, tooltip_width, tooltip_height))
        pygame.draw.rect(screen, (0, 0, 0), (tooltip_x, tooltip_y, tooltip_width, tooltip_height), 2)

        #Header
        font2 = pygame.font.Font(None, 40)
        head1 = font2.render("Forward Prop       Back Prop", True, mgr.color_black)
        screen.blit(head1 , (tooltip_x + 5, tooltip_y + 5))

        # Font setup
        font = pygame.font.Font(None, 22)
        self.tooltip_generate_text()
        col_size = 60
        header_spac = 39
        for x, text_col in enumerate(self.tooltip_columns):  # loop through columns
            for y, text_cell in enumerate(text_col): #print(f"x={x}\ty={y}\ttext_cell='{text_cell}'")
                # Set a default color. You might define a normal_color if needed.
                this_color = mgr.color_black  # or some default color
                if x == 7 and y > 0 and len(text_cell) > 0:  # Adjustment column
                    try:
                        # Attempt to convert the text_cell to a float
                        val = float(text_cell.replace(",", ""))
                        this_color = mgr.color_greenforest if val >= 0 else mgr.color_crimson
                    except ValueError as e:
                        print(f"Error converting text_cell to float: {text_cell}. Error: {e}")
                        # Optionally, set this_color to a fallback (here normal_color) if conversion fails
                        this_color = mgr.color_black
                label = font.render(str(text_cell), True, this_color)
                screen.blit(label, (tooltip_x + x * col_size + 5,  header_spac + (tooltip_y + 5 + y * 20)))



