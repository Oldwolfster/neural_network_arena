from src.neuroForge.DisplayModel__NeuronBase import DisplayModel__NeuronBase


class DisplayModel__NeuronRenderText(DisplayModel__NeuronBase):
    """Traditional text-based neuron visualization."""
    def render(self, neuron, screen):
        # Render weight text inside the neuron
        screen.blit(neuron.ez_printer.render(neuron.weight_text), (neuron.location_left, neuron.location_top))
