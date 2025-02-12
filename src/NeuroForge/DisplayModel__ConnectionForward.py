import math
import pygame

from src.neuroForge import mgr


class DisplayModel__ConnectionForward:
    def __init__(self, from_neuron, to_neuron, weight=0):
        self.from_neuron = from_neuron  # Can be a neuron or (x, y) coordinates
        self.to_neuron = to_neuron      # Reference to DisplayNeuron
        self.weight = 0
        self.color = (0, 0, 0)          # Default to black, could be dynamic
        self.thickness = 2              # Line thickness, could vary by weight
        self.arrow_size = 18            # Size of the arrowhead
        self.is_really_a_weight = True  # not include lines from inputs or from output neuron to prediction box
    def update_connection(self):
        """
        Updates the connection's thickness and color based on weight magnitude.
        Uses mgr.max_weight for normalization.
        """
        self.weight = self.get_my_weight()
        if mgr.max_weight == 0:  # Avoid division by zero
            weight_ratio = 0
        else:
            weight_ratio = abs(self.weight) / mgr.max_weight  # Normalize weight

        # 🔹 Adjust thickness (minimum 1, max 8 for visibility)
        self.thickness = max(1, int(1 + weight_ratio * 12))     #Scale thickness here!!!
        #print(f"mgr.max_weight = {self.thickness}\tmgr.max_weight = {mgr.max_weight}\tself.thickness = {self.thickness}")
        # 🔹 Adjust color based on weight sign
        base_intensity = int(255 * weight_ratio)  # Scale color by magnitude
        if self.weight > 0:
            self.color = (0, base_intensity, 0)  # Green for positive weights
        else:
            self.color = (base_intensity, 0, 0)  # Red for negative weights

    def get_my_weight(self):
        if not self.is_really_a_weight:
            return 0
        return self.to_neuron.weights[self.from_neuron.position]


    def _get_start_point(self):
        if isinstance(self.from_neuron, tuple): # If from_neuron is a tuple, assume it's (x, y) coordinates
            self.is_really_a_weight = False
            return self.from_neuron
        # Otherwise, assume it's a neuron object
        #Forward prop
        start_x = self.from_neuron.location_left + self.from_neuron.location_width
        #Back prop
        #start_x = self.from_neuron.location_left
        banner_height = self.from_neuron.location_height * 0.2  # Assume banner is 20% of neuron height

        start_y = self.from_neuron.location_top + (self.from_neuron.location_height * 0.5)  # Place in middle of banner
        return (start_x, start_y)

    def _get_end_point(self):
        if isinstance(self.to_neuron, tuple):   # If from_neuron is a tuple, assume it's (x, y) coordinates
            self.is_really_a_weight = False
            return self.to_neuron
        # Otherwise assume to_neuron is a neuron object
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