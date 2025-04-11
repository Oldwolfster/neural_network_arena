# src/NeuroForge/ui/WindowMatches.py
from src.NeuroForge.ui.BaseWindow import BaseWindow
from src.NeuroForge.ui.GlowButton import GlowButton
from src.NeuroForge.ui.HoloPanel import HoloPanel
from src.NeuroForge.ui.InputPanel import LabelInputPanel
from src.NeuroForge.ui.TreePanel import TreePanel
from src.engine import BaseArena
from src.engine.BaseGladiator import Gladiator
from src.NeuroForge import Const

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

        self.selected_gladiators = TreePanel(
            parent_surface=self.surface,
            title="Selected Gladiators",
            path=None,
            superclass=None,
            left_pct=2,
            top_pct=75,
            width_pct=24,
            height_pct=23
        )

        self.gladiator_browser = TreePanel(
            parent_surface=self.surface,
            title="Available Gladiators",
            path=base_path,
            superclass=Gladiator,
            left_pct=2,
            top_pct=2,
            width_pct=24,
            height_pct=70,
            on_file_selected=self.selected_gladiators.add_file
        )

        self.selected_arenas = TreePanel(
            parent_surface=self.surface,
            title="Selected Arena",
            path=None,
            superclass=None,
            left_pct=38,
            top_pct=62,
            width_pct=24,
            height_pct=23
        )

        base_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "coliseum", "arenas"))
        self.arena_browser = TreePanel(
            parent_surface=self.surface,
            title="Available Arenas",
            path=base_path,
            superclass=BaseArena,
            left_pct=38,
            top_pct=14,
            width_pct=24,
            height_pct=47,
            on_file_selected=self.selected_arenas.add_file
        )

        print(f"TSS = {Const.configs[0].hyper.training_set_size}")
        print(f"TSS = {Const.configs[0].hyper.epochs_to_run}")
        print(f"TSS = {Const.configs[0].hyper.min_no_epochs}")

        self.params_panel = LabelInputPanel(
            parent_surface=self.surface,
            title="Core Battle Parameters",
            left_pct=74,
            top_pct=2,
            width_pct=24,
            height_pct=50,
            fields=["Random Seed","Default Learning Rate", "Max Epochs", "Training set size", "Min Epochs"],
            banner_color=Const.COLOR_BLUE,
            initial_values={
                "Random Seed": str(Const.configs[0].hyper.random_seed),
                "Default Learning Rate": str(Const.configs[0].hyper.default_learning_rate),
                "Max Epochs": str(Const.configs[0].hyper.epochs_to_run),
                "Training set size": str(Const.configs[0].hyper.training_set_size),
                "Min Epochs": str(Const.configs[0].hyper.min_no_epochs),
            }
        )

        self.thresholds_panel = LabelInputPanel(
            parent_surface=self.surface,
            title="Thresholds/Convergence",
            left_pct=74,
            top_pct=55,
            width_pct=24,
            height_pct=42,
            fields=["accuracy_threshold","converge_threshold","converge_epochs","etc."],
            banner_color=Const.COLOR_BLUE,
            initial_values={
                "Max Epochs": str(Const.configs[0].hyper.epochs_to_run),
                "Training set size": str(Const.configs[0].hyper.training_set_size),
                "Min Epochs": str(Const.configs[0].hyper.min_no_epochs),
            }
        )

        self.battle_button = GlowButton(
            parent_surface=self.surface,
            text="Let the Battle Begin",
            left_pct=36,
            top_pct=88,
            width_pct=28,
            height_pct=9,
            on_click=self.launch_battle
        )

        self.children.extend([self.gladiator_browser, self.selected_gladiators, self.arena_browser, self.selected_arenas, self.params_panel, self.battle_button, self.thresholds_panel])

    def process_an_event(self, event):
        self.gladiator_browser.handle_events(event, self.left, self.top)
        self.arena_browser.handle_events(event, self.left, self.top)
        self.params_panel.handle_events(event,self.left, self.top)
        self.battle_button.handle_events(event, self.left, self.top)
        self.thresholds_panel.handle_events(event,self.left, self.top)

    def update_me(self):
        super().update_me()  # Handles any dynamic logic for panels

    def launch_battle(self):
        selected_glads = self.selected_gladiators.get_selected_files()  # We'll implement this
        selected_arenas = self.selected_arenas.get_selected_files()     # "
        params = self.params_panel.inputs

        print("🧠 Launching battle with:")
        print(f"Gladiators: {selected_glads}")
        print(f"Arenas: {selected_arenas}")
        print(f"Params: {params}")

        # TODO: hand this off to your core battle engine

