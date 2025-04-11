# src/NeuroForge/ui/WindowMatches.py
from src.NeuroForge.ui.BaseWindow import BaseWindow
from src.NeuroForge.ui.HoloPanel import HoloPanel
from src.NeuroForge.ui.TreePanel import TreePanel
from src.engine import BaseArena
from src.engine.BaseGladiator import Gladiator


class WindowMatches(BaseWindow):
    def __init__(self):
        super().__init__(
            title_text="Configure Match",
            background_image_path="assets/form_backgrounds/coliseum_glow.png"
        )

        # Example panels
        #self.panel_gladiators = self.add_panel("🏛️ Gladiators", left_pct=4, top_pct=15, width_pct=40, height_pct=70, panel_id="glads")
        #self.panel_arenas = self.add_panel("🧪 Arena", left_pct=52, top_pct=15, width_pct=40, height_pct=70, panel_id="arena")

        # In WindowMatches __init__:
        #self.gladiator_browser = HoloPanel(
        #    parent_surface=self.surface,
        #    title="Select Gladiators",
        #    left_pct=52,
        #    top_pct=12,
        #    width_pct=17,
        #    height_pct=90
        #)

        import os

        base_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "coliseum", "gladiators"))
        #print(f"Base_Path={base_path}")

        self.gladiator_browser = TreePanel(
            parent_surface=self.surface,
            title="Available Gladiators",
            path=base_path,
            superclass=Gladiator,
            left_pct=2,
            top_pct=2,
            width_pct=30,
            height_pct=70
        )
        self.selected_gladiators = HoloPanel(
            parent_surface=self.surface,
            title="Selected Gladiators",
            left_pct=2,
            top_pct=75,
            width_pct=30,
            height_pct=23
        )
        base_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "coliseum", "arenas"))
        self.arena_browser = TreePanel(
            parent_surface=self.surface,
            title="Available Arenas",
            path=base_path,
            superclass=BaseArena,
            left_pct=40,
            top_pct=25,
            width_pct=25,
            height_pct=70
        )

        self.children.extend([self.gladiator_browser, self.selected_gladiators, self.arena_browser])

    def process_an_event(self, event):
        self.gladiator_browser.handle_events(event, self.left, self.top)
        self.arena_browser.handle_events(event, self.left, self.top)

    def update_me(self):
        super().update_me()  # Handles any dynamic logic for panels
