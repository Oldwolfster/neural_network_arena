import pygame
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from src.engine.Utils import draw_gradient_rect


class DisplayGraph():
    #def __init__(self, width_pct=98, height_pct=4.369, left_pct=1, top_pct=0):
    def __init__(self, width, height, left, top, model_surface, model_id):
        """Creates a Graph showing MAE over epoch"""
        #super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)
        #super().__init__(width_pct=0, left_pct=0, top_pct=0, height_pct=0,                         pixel_adjust_width=width, pixel_adjust_left=left,                         pixel_adjust_top=top, pixel_adjust_height=height)
        self.model_surface          = model_surface

        # Positioning
        self.location_left          = left
        self.location_top           = top
        self.location_width         = width
        self.location_height        = height
        SQL = "SELECT epoch, mean_absolute_error FROM EpochSummary WHERE model_id = ? ORDER BY epoch"
        self.results = Const.dm.db.query(SQL, (model_id,))

        # Build and store the plot surface once.
        self.plot_surface = self.create_plot_surface_from_results()

    def create_plot_surface_from_results(self):
        # Prepare the data
        epochs = [r['epoch'] for r in self.results]
        mae_values = [r['mean_absolute_error'] for r in self.results]

        dpi = 100
        figsize = (self.location_width / dpi, self.location_height / dpi)

        # 🔹 Create the figure and axis
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("none")  # Transparent figure background

        ax = fig.add_subplot(111)
        ax.set_facecolor("none")  # Transparent plot background

        # 🔹 Plot the error line
        ax.plot(epochs, mae_values, marker='o', linestyle='-', color=(1,0,1))

        # 🔹 Remove padding, ticks, labels, and borders
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.subplots_adjust(left=0, right=1, top=.97, bottom=0)

        # 🔹 Force y-axis to start at 0
        ax.set_ylim(bottom=0)

        # 🔹 Render to ARGB and convert to RGBA for Pygame
        canvas = FigureCanvas(fig)
        canvas.draw()
        raw_data = canvas.tostring_argb()
        width, height = canvas.get_width_height()

        # 🔁 Convert ARGB to RGBA for Pygame
        import numpy as np
        argb_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))
        rgba_array = argb_array[:, :, [1, 2, 3, 0]]  # Reorder channels

        # 🔹 Create Pygame surface
        plot_surface = pygame.image.frombuffer(rgba_array.tobytes(), (width, height), "RGBA")

        plt.close(fig)
        return plot_surface

    def render(self):
        """Called once per pygame loop"""
        # Blit the pre-rendered plot onto the EZSurface's surface.

        self.model_surface.blit(self.plot_surface, (self.location_left, self.location_top + 30))
        self.draw_graph_frame()

    def update_me(self):
        """Graph will not change"""
        pass

    def draw_graph_frame(self, label_text="Error History"):
        """Draws a frame styled like a neuron with a banner and border."""
        font = pygame.font.Font(None, 30)

        # === Banner ===
        banner_height = font.get_height() + 8
        banner_rect = pygame.Rect(self.location_left, self.location_top+ 3, self.location_width, banner_height)
        # === Border (below banner) ===
        body_top = self.location_top + banner_height
        body_height = self.location_height - banner_height
        pygame.draw.rect(
            self.model_surface,
            Const.COLOR_FOR_NEURON_BODY,
            (self.location_left, body_top, self.location_width, body_height),
            border_radius=6,
            width=7
        )
        draw_gradient_rect(self.model_surface, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)
        label_surface = font.render(label_text, True, Const.COLOR_FOR_NEURON_TEXT)
        self.model_surface.blit(
            label_surface,
            (self.location_left + 5, self.location_top + 5 + (banner_height - label_surface.get_height()) // 2)
        )