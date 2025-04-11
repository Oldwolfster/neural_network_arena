from src.NeuroForge.ui.HoloPanel import HoloPanel
from src.NeuroForge import Const
import pygame
import os

class TreePanel(HoloPanel):
    def __init__(
        self,
        parent_surface,
        title,
        path,
        superclass,
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
        self.path = path
        self.superclass = superclass
        self.font = pygame.font.Font(None, 22)
        self.data = {}  # Initialize early to prevent attribute errors
        self.data = self.load_tree_data(path)
        self.expanded = set()
        self.row_rects = []
        self.scroll_offset_y = 0
        self.scroll_speed = 30
        self.SCROLL_START_Y = 40

    def draw_me(self):
        self.render()

    def load_tree_data(self, base_path):
        tree = {}

        for root, dirs, files in os.walk(base_path):
            rel_root = os.path.relpath(root, base_path)
            path_parts = rel_root.split(os.sep) if rel_root != "." else []
            current = tree
            for part in path_parts:
                current = current.setdefault(part, {})

            for f in sorted(files):
                if (
                    f.endswith(".py")
                    and not f.startswith("__")
                    and "__pycache__" not in f
                    and not os.path.isdir(os.path.join(root, f))
                ):
                    current[f] = None  # File leaf

        return tree

    def rebuild_row_rects(self):
        self.row_rects = []
        self._rebuild_row_rects_recursive(self.data, self.SCROLL_START_Y - self.scroll_offset_y, 0)

    def _rebuild_row_rects_recursive(self, subtree, y, indent):
        for key, value in subtree.items():
            is_folder = isinstance(value, dict)
            rect = pygame.Rect(10 + indent * 20, y, self.width - 20, 26)
            self.row_rects.append((rect, is_folder, key))
            y += 30
            if is_folder and key in self.expanded:
                y = self._rebuild_row_rects_recursive(value, y, indent + 1)
        return y

    def render(self):
        if not hasattr(self, "data") or not self.data:
            print("⏭️ Skipping TreePanel render — self.data not yet set or is empty.")
            return

        super().render()
        self.surface.fill((0, 0, 0, 0))
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))  # Less opaque so background shows through more
        self.surface.blit(overlay, (0, 0))
        pygame.draw.rect(self.surface, Const.COLOR_BLACK, self.surface.get_rect(), 5, border_radius=6)

        self._render_recursive(self.data, self.SCROLL_START_Y - self.scroll_offset_y, 0)
        self.blit_to_parent()
        self.rebuild_row_rects()

    def _render_recursive(self, subtree, y, indent):
        for key, value in subtree.items():
            is_folder = isinstance(value, dict)
            label = f"[{'-' if key in self.expanded else '+'}] {key}" if is_folder else f"⚔️ {key}"
            text_surf = self.font.render(label, True, Const.COLOR_WHITE)
            self.surface.blit(text_surf, (10 + indent * 20, y))
            y += 30
            if is_folder and key in self.expanded:
                y = self._render_recursive(value, y, indent + 1)
        return y

    def handle_events(self, event, parent_offset_x=0, parent_offset_y=0):
        self.handle_click(event, parent_offset_x, parent_offset_y)
        self.handle_scroll(event)

    def handle_click(self, event, parent_offset_x=0, parent_offset_y=0):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        local_x = mouse_x - self.left - parent_offset_x
        local_y = mouse_y - self.top - parent_offset_y

        for rect, is_folder, key in self.row_rects:
            if rect.collidepoint(local_x, local_y):
                if is_folder:
                    if key in self.expanded:
                        self.expanded.remove(key)
                    else:
                        self.expanded.add(key)
                    self.render()
                break

    def handle_scroll(self, event):
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_offset_y -= event.y * self.scroll_speed
            self.scroll_offset_y = max(0, self.scroll_offset_y)
            self.render()

    def debug_print_row_map(self):
        print("\n=== Row Rect Debug ===")
        for i, (rect, is_folder, key) in enumerate(self.row_rects):
            tag = "📁" if is_folder else "⚔️"
            print(f"{i:2}: {tag} {key:<20} → {rect}")
        print("======================\n")
