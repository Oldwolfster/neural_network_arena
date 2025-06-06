from src.NeuroForge.ui.HoloPanel import HoloPanel
from src.NeuroForge import Const
import pygame
import os
import time

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
        on_file_selected=None,
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
        self.on_file_selected = on_file_selected
        self.font = pygame.font.Font(None, 22)
        self.data = {}  # Initialize early to prevent attribute errors
        self.data = {} if path is None else self.load_tree_data(path)
        #print(f"Tree{self.data}")
        self.expanded = set()
        self.row_rects = []
        self.scroll_offset_y = 0
        self.scroll_speed = 30
        self.SCROLL_START_Y = 40

        self.last_click_time = 0
        self.last_clicked_path = None
        self.double_click_threshold = 0.3  # seconds

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

    def _rebuild_row_rects_recursive(self, subtree, y, indent, parent_path=()):
        for key, value in subtree.items():
            is_folder = isinstance(value, dict)
            full_path = parent_path + (key,)
            rect = pygame.Rect(10 + indent * 20, y, self.width - 20, 26)
            self.row_rects.append((rect, is_folder, full_path))
            y += 30
            if is_folder and full_path in self.expanded:
                y = self._rebuild_row_rects_recursive(value, y, indent + 1, full_path)
        return y

    def render(self):
        if not hasattr(self, "data"): # or not self.data:
            #print("⏭️ Skipping TreePanel render — self.data not yet set or is empty.")
            return

        super().render()

        # Only fill below the banner to allow background and title to shine through
        scroll_area = pygame.Rect(0, self.SCROLL_START_Y, self.width, self.height - self.SCROLL_START_Y)
        overlay = pygame.Surface((scroll_area.width, scroll_area.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))  # Soft translucent black
        self.surface.blit(overlay, scroll_area.topleft)

        # Optional: draw inner border just around scroll area (or remove this)
        pygame.draw.rect(self.surface, Const.COLOR_BLACK, scroll_area, 3, border_radius=4)

        self._render_recursive(self.data, self.SCROLL_START_Y - self.scroll_offset_y, 0)
        self.blit_to_parent()
        self.rebuild_row_rects()


    def _render_recursive(self, subtree, y, indent, parent_path=()):
        for key, value in subtree.items():
            is_folder = isinstance(value, dict)
            full_path = parent_path + (key,)

            # ⬇️ Strip .py from file labels, but leave folder names unchanged
            display_key = key[:-3] if not is_folder and key.endswith(".py") else key
            label = f"[{'-' if full_path in self.expanded else '+'}] {display_key}" if is_folder else f"⚔️ {display_key}"

            text_surf = self.font.render(label, True, Const.COLOR_WHITE)
            self.surface.blit(text_surf, (10 + indent * 20, y))
            y += 30

            if is_folder and full_path in self.expanded:
                y = self._render_recursive(value, y, indent + 1, full_path)

        return y

    def handle_events(self, event, parent_offset_x=0, parent_offset_y=0):
        self.handle_click(event, parent_offset_x, parent_offset_y)
        self.handle_scroll(event, parent_offset_x, parent_offset_y)


    def handle_click(self, event, parent_offset_x=0, parent_offset_y=0):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        local_x = mouse_x - self.left - parent_offset_x
        local_y = mouse_y - self.top - parent_offset_y

        for rect, is_folder, full_path in self.row_rects:
            if rect.collidepoint(local_x, local_y):
                if is_folder:
                    if full_path in self.expanded:
                        self.expanded.remove(full_path)
                    else:
                        self.expanded.add(full_path)
                    self.render()
                else:
                    now = time.time()
                    if full_path == self.last_clicked_path and now - self.last_click_time <= self.double_click_threshold:
                        if self.on_file_selected:
                            self.on_file_selected(full_path[-1])  # Safely pass only the filename string
                    else:
                        self.last_click_time = now
                        self.last_clicked_path = full_path
                break


    def handle_scroll(self, event, parent_offset_x=0, parent_offset_y=0):
        if event.type == pygame.MOUSEWHEEL:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            local_x = mouse_x - self.left - parent_offset_x
            local_y = mouse_y - self.top - parent_offset_y

            # Only scroll if the mouse is actually over this panel
            if 0 <= local_x <= self.width and 0 <= local_y <= self.height:
                self.scroll_offset_y -= event.y * self.scroll_speed
                self.scroll_offset_y = max(0, self.scroll_offset_y)
                self.render()

    def add_file(self, filename: str):
        """Adds a file to the root level of the TreePanel (flat list only)."""
        # Ensure .py is stripped (we store keys as display-ready)
        if filename.endswith(".py"):
            filename = filename[:-3]

        self.data[filename] = None
        self.render()

    def get_selected_files(self):
        return list(self.data.keys())


    def debug_print_row_map(self):
        print("\n=== Row Rect Debug ===")
        for i, (rect, is_folder, key) in enumerate(self.row_rects):
            tag = "📁" if is_folder else "⚔️"
            print(f"{i:2}: {tag} {key:<20} → {rect}")
        print("======================\n")
