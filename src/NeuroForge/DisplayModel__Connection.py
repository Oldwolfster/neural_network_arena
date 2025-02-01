import math
import pygame

class DisplayModel__Connection:
    def __init__(self, from_neuron, to_neuron, weight=0):
        self.from_neuron = from_neuron  # Can be a neuron or (x, y) coordinates
        self.to_neuron = to_neuron      # Reference to DisplayNeuron
        self.weight = weight
        self.color = (0, 0, 0)          # Default to black, could be dynamic
        self.thickness = 2              # Line thickness, could vary by weight
        self.arrow_size = 18            # Size of the arrowhead

    def update_connection(self):
        pass

    def _get_start_point(self):

        if isinstance(self.from_neuron, tuple): # If from_neuron is a tuple, assume it's (x, y) coordinates
            return self.from_neuron
        # Otherwise, assume it's a neuron object
        start_x = self.from_neuron.location_left + self.from_neuron.location_width
        banner_height = self.from_neuron.location_height * 0.2  # Assume banner is 20% of neuron height
        start_y = self.from_neuron.location_top + (banner_height * 0.5)  # Place in middle of banner
        return (start_x, start_y)

    def _get_end_point(self):

        if isinstance(self.to_neuron, tuple):   # If from_neuron is a tuple, assume it's (x, y) coordinates
            return self.to_neuron
        # Otherwise assume to_neuron is a neuron object
        end_x = self.to_neuron.location_left
        end_y = self.to_neuron.location_top + self.to_neuron.location_height // 2
        return (end_x, end_y)


    def _get_end_point_old(self):
        # Always assume to_neuron is a neuron object
        end_x = self.to_neuron.location_left
        end_y = self.to_neuron.location_top + self.to_neuron.location_height // 2
        return (end_x, end_y)

    def draw_connection(self, screen):
        # Get start and end points
        start_x, start_y = self._get_start_point()
        end_x, end_y = self._get_end_point()

        # Draw the main connection line
        pygame.draw.line(screen, self.color, (start_x, start_y), (end_x, end_y), self.thickness)

        # Calculate arrowhead points
        angle = math.atan2(end_y - start_y, end_x - start_x)  # Angle of the connection line
        arrow_point1 = (
            end_x - self.arrow_size * math.cos(angle - math.pi / 12),
            end_y - self.arrow_size * math.sin(angle - math.pi / 12)
        )
        arrow_point2 = (
            end_x - self.arrow_size * math.cos(angle + math.pi / 12),
            end_y - self.arrow_size * math.sin(angle + math.pi / 12)
        )

        # Draw the arrowhead
        pygame.draw.polygon(screen, self.color, [(end_x, end_y), arrow_point1, arrow_point2])