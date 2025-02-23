
class DisplayModel__NeuronText:
    def __init__(self, neuron):
        self.neuron = neuron  # ✅ Store reference to parent neuron
        self.min_weight = float('inf')  # Track min/max for scaling
        self.max_weight = float('-inf')

    def render(self, screen, ez_printer, body_y_start, weight_text,location_left):
        # Render neuron details inside the body
        print(f"In TEXT RENDERER")
        body_text_y_start = body_y_start + 5
        ez_printer.render(
            screen,
            text=weight_text,
            x=location_left + 11,
            y=body_text_y_start + 7
        )