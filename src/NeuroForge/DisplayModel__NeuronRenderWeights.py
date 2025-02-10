from src.neuroForge.DisplayModel__NeuronBase import DisplayModel__NeuronBase


class DisplayModel__NeuronRenderWeights(DisplayModel__NeuronBase):
    """Visualizes weights inside neurons instead of along arrows."""
    def render(self, neuron, screen):
        # Draw neuron outline
        pygame.draw.rect(screen, (255, 255, 255),
                         (neuron.location_left, neuron.location_top, neuron.location_width, neuron.location_height), 2)

        # Draw weight rectangles inside the neuron
        weight_spacing = neuron.location_width // max(1, len(neuron.weights))
        for i, weight in enumerate(neuron.weights):
            weight_color = (0, 255, 0) if weight >= 0 else (255, 0, 0)  # Green for positive, red for negative
            bar_x = neuron.location_left + (i * weight_spacing)
            bar_width = weight_spacing - 2
            bar_height = abs(int(weight * 10))  # Scale for visualization
            bar_y = neuron.location_top + neuron.location_height // 2 - (bar_height // 2)

            pygame.draw.rect(screen, weight_color, (bar_x, bar_y, bar_width, bar_height))
