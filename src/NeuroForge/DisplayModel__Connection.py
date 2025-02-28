import math
import pygame
from src.engine.Utils import draw_rect_with_border, draw_text_with_background, ez_debug, check_label_collision, get_text_rect
from src.NeuroForge import Const
class DisplayModel__Connection:
    def __init__(self, from_neuron, to_neuron, my_screen, weight_index=0):
        self.my_screen          = my_screen
        self.from_neuron        = from_neuron   # Can be a neuron or (x, y) coordinates
        self.to_neuron          = to_neuron     # Reference to DisplayNeuron
        self.weight_index       = weight_index
        self.arrow_size         = 18            # Size of the arrowhead
        self.color              = (0, 0, 0)     # Default black (will be updated dynamically)
        self.thickness          = 4             # Default thickness
        self.label_offset       = 10
        self.weight             = 0             # Will be updated dynamically
        self.is_really_a_weight = not isinstance(from_neuron, tuple) and not isinstance(to_neuron, tuple)

    def draw(self):
        """ Draws the connection line and arrowhead on the screen. """
        start_x, start_y = self._get_start_point()
        end_x, end_y = self._get_end_point()

        # Draw the main connection line
        ez_debug( x1=end_x,y1=end_y,x2=start_x,y2=start_y)
        pygame.draw.line(self.my_screen, self.color, (start_x, start_y), (end_x, end_y), self.thickness)

        # Draw the arrowhead
        arrow_points = self._calculate_arrowhead(end_x, end_y, start_x, start_y)
        pygame.draw.polygon(self.my_screen , self.color, [(end_x, end_y)] + arrow_points)

        # Draw shadow
        #shadow_offset = 2
        #pygame.draw.line(self.get_real_neuron().screen, (0, 0, 0), (start_x + shadow_offset, start_y + shadow_offset), (end_x + shadow_offset, end_y + shadow_offset), self.thickness+1)

    def get_my_weight(self):
        """ Retrieves the weight associated with this connection. """
        return 0 if not self.is_really_a_weight else self.to_neuron.weights[self.from_neuron.position]

    def _normalize_weight(self, weight):
        """ Normalizes the weight using the max weight from the manager. """
        return 0 if Const.MAX_WEIGHT == 0 else abs(weight) / Const.MAX_WEIGHT

    def _get_start_point(self):
        """ Determines the start point for the connection. """
        if isinstance(self.from_neuron, tuple):
            return self.from_neuron  # Coordinates directly provided

        start_x = self.from_neuron.location_left + self.from_neuron.location_width
        start_y = self.from_neuron.location_top + (self.from_neuron.location_height * 0.5) + self.label_offset
        return start_x, start_y

    def _get_end_point(self):
        """ Determines the endpoint for the connection. """
        if isinstance(self.to_neuron, tuple):
            return self.to_neuron

        end_x = self.to_neuron.location_left
        end_y = self.point_arrow_at_weight(self.to_neuron.location_top + self.to_neuron.location_height // 2) + self.label_offset
        return end_x, end_y

    def get_real_neuron(self):
        """ One of the other is a neuron, this returns the real neuron . """
        if isinstance(self.from_neuron, tuple):
            return self.to_neuron  # Coordinates directly provided
        else:
            return self.from_neuron
    def point_arrow_at_weight(self, original_y):
        """ Adjusts the endpoint Y coordinate to align with the weight label if applicable. """
        if isinstance(self.to_neuron, tuple):
            return original_y  # If pointing to coordinates, do nothing
        if len(self.to_neuron.neuron_visualizer.my_fcking_labels) == 0:
            return original_y
        return self.to_neuron.neuron_visualizer.my_fcking_labels[self.weight_index]

    def _calculate_arrowhead(self, end_x, end_y, start_x, start_y):
        """ Calculates the arrowhead coordinates based on the line direction. """
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_point1 = (
            end_x - self.arrow_size * math.cos(angle - math.pi / 10),  # Sharpened
            end_y - self.arrow_size * math.sin(angle - math.pi / 10)
        )
        arrow_point2 = (
            end_x - self.arrow_size * math.cos(angle + math.pi / 10),
            end_y - self.arrow_size * math.sin(angle + math.pi / 10)
        )

        return [arrow_point1, arrow_point2]

    def update_connection(self):
        """ Updates the connection's thickness and color based on weight magnitude. """
        return #not sure how best to leverage so not using it atm.
        self.weight = self.get_my_weight()
        weight_ratio = self._normalize_weight(self.weight)

        # 🔹 Adjust thickness dynamically
        self.thickness = max(1, int(1 + weight_ratio * 12))

        # 🔹 Adjust color based on weight sign
        base_intensity = int(255 * weight_ratio)
        self.color = (0, base_intensity, 0) if self.weight > 0 else (base_intensity, 0, 0)
