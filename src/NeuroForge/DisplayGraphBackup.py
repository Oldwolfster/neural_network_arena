import pygame
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

class DisplayGraph():
    #def __init__(self, width_pct=98, height_pct=4.369, left_pct=1, top_pct=0):
    def __init__(self, width, height, left, top):
        """Creates a Graph showing MAE over epoch"""
        #super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)
        #super().__init__(width_pct=0, left_pct=0, top_pct=0, height_pct=0,                         pixel_adjust_width=width, pixel_adjust_left=left,                         pixel_adjust_top=top, pixel_adjust_height=height)

        SQL = "SELECT epoch, mean_absolute_error FROM EpochSummary ORDER BY epoch"
        self.results = Const.dm.db.query(SQL)

        # Build and store the plot surface once.
        self.plot_surface = self.create_plot_surface_from_results()

    def create_plot_surface_from_results(self):
        """
        Helper method to create a pygame surface from a matplotlib plot of the query results,
        removing all labels/ticks to show only the line.
        """
        # Convert query results to lists for plotting
        epochs = [r['epoch'] for r in self.results]
        mae_values = [r['mean_absolute_error'] for r in self.results]

        # Determine the size of the plot based on the EZSurface dimensions.
        width, height = self.surface.get_width(), self.surface.get_height()
        dpi = 100
        figsize = (width / dpi, height / dpi)

        # Create a matplotlib figure (no interactive backend changes needed).
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        # Plot the data
        ax.plot(epochs, mae_values, marker='o', linestyle='-')

        # Remove all text, ticks, and spines (so only the line is shown).
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("Error History")

        # First let Matplotlib auto-scale
        ax.autoscale()
        # Then force the lower bound to be 0, keeping the auto upper bound
        ax.set_ylim(bottom=0)

        #for spine in ax.spines.values():
        #    spine.set_visible(False)

        # (Optional) Force a wide y-range so tiny changes look flat:
        # ax.set_ylim(0, 300000)  # Example for a ~300k baseline
        # or auto-scale (matplotlib default) to zoom in on tiny differences:
        # ax.autoscale()

        # Render the figure off-screen
        canvas = FigureCanvas(fig)
        canvas.draw()
        raw_data = canvas.tostring_rgb()
        size = canvas.get_width_height()

        # Convert the off-screen buffer to a pygame surface
        plot_surface = pygame.image.frombuffer(raw_data, size, "RGB")
        plt.close(fig)  # Free memory
        return plot_surface

    def render(self):
        """Called once per pygame loop"""
        # Blit the pre-rendered plot onto the EZSurface's surface.
        self.surface.blit(self.plot_surface, (0, 0))

    def update_me(self):
        """Graph will not change"""
        pass
