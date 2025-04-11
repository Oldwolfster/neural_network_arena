from src.NeuroForge.ui.HoloPanel import HoloPanel
from src.NeuroForge import Const
import pygame

class TreePanel(HoloPanel):
    def __init__(
        self,
        parent_surface,
        title,
        data,
        left_pct,
        top_pct,
        width_pct,
        height_pct,
        banner_color=Const.COLOR_BLUE,
    ):
        super().__init__(
            parent_surface=parent_surface,
            title=title,
            left_pct=left_pct,
            top_pct=top_pct,
            width_pct=width_pct,
            height_pct=height_pct,
            fields=[],
            banner_color=banner_color,
        )
        self.data = data
        self.font = pygame.font.Font(None, 22)
        self.expanded = {k: True for k in data.keys()}
        self.row_rects = []

    def draw_me(self):
        self.render()

    def rebuild_row_rects(self):
        self.row_rects = []
        y = 40
        for category, models in self.data.items():
            folder_rect = pygame.Rect(10, y, self.width - 20, 26)
            self.row_rects.append((folder_rect, True, category))
            y += 30
            if self.expanded.get(category, False):
                for model in models:
                    model_rect = pygame.Rect(30, y, self.width - 40, 26)
                    self.row_rects.append((model_rect, False, model))
                    y += 26

    def render(self):
        if not hasattr(self, "data") or self.data is None:
            print("⏭️ Skipping TreePanel render — self.data not yet set.")
            return

        super().render()
        self.surface.fill((0, 0, 0, 0))
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.surface.blit(overlay, (0, 0))

        pygame.draw.rect(self.surface, Const.COLOR_BLACK, self.surface.get_rect(), 5, border_radius=6)

        y = 40
        for category, models in self.data.items():
            folder_label = f"[{'-' if self.expanded.get(category, False) else '+'}] {category}"
            folder_surf = self.font.render(folder_label, True, Const.COLOR_WHITE)
            self.surface.blit(folder_surf, (10, y))
            y += 30

            if self.expanded.get(category, False):
                for model in models:
                    model_label = f"⚔️ {model}"
                    model_surf = self.font.render(model_label, True, Const.COLOR_WHITE)
                    self.surface.blit(model_surf, (30, y))
                    y += 26

        self.blit_to_parent()
        self.rebuild_row_rects()

    def debug_print_row_map(self):
        print("🗺️ Current row layout:")
        for i, (rect, is_folder, key) in enumerate(self.row_rects):
            kind = "📁" if is_folder else "⚔️"
            print(f"  {i:>2}: {kind} {key.ljust(20)} → {rect}")


    def handle_click(self, event, parent_offset_x=0, parent_offset_y=0):
        if event.type != pygame.MOUSEBUTTONDOWN:
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Convert to TreePanel-local coordinates by subtracting both parents
        local_x = mouse_x - self.left - parent_offset_x
        local_y = mouse_y - self.top - parent_offset_y

        print(f"🖱️ Global: ({mouse_x}, {mouse_y}) | TreePanel Local: ({local_x}, {local_y})")

        for rect, is_folder, key in self.row_rects:
            if rect.collidepoint(local_x, local_y):
                print(f"🎯 Clicked {key}")
                if is_folder:
                    self.expanded[key] = not self.expanded.get(key, False)
                    self.render()
                break

    def debug_print_row_map(self):
        print("\n=== Row Rect Debug ===")
        for i, (rect, is_folder, key) in enumerate(self.row_rects):
            tag = "📁" if is_folder else "⚔️"
            print(f"{i:2}: {tag} {key:<20} → {rect}")
        print("======================\n")
BasicModels