# src/NeuroForge/ui/WindowMatches.py
from src.NeuroForge.ui.BaseWindow import BaseWindow

class WindowMatches(BaseWindow):
    def __init__(self):
        super().__init__(
            title_text="Configure Match",
            background_image_path="assets/form_backgrounds/coliseum_glow.png"
        )

        # Example panels
        #self.panel_gladiators = self.add_panel("🏛️ Gladiators", left_pct=4, top_pct=15, width_pct=40, height_pct=70, panel_id="glads")
        #self.panel_arenas = self.add_panel("🧪 Arena", left_pct=52, top_pct=15, width_pct=40, height_pct=70, panel_id="arena")

    def update_me(self):
        super().update_me()  # Handles any dynamic logic for panels
