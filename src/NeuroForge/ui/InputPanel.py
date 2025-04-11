from src.NeuroForge.ui.HoloPanel import HoloPanel
import pygame

class LabelInputPanel(HoloPanel):
    def __init__(
        self,
        parent_surface,
        title,
        left_pct,
        top_pct,
        width_pct,
        height_pct,
        fields,
        banner_color,
        input_width_pct=45,  # 🆕 percentage of panel width used for input box
        initial_values=None,  # 🆕 optional dict of label -> prefilled value
    ):
        super().__init__(
            parent_surface=parent_surface,
            title=title,
            left_pct=left_pct,
            top_pct=top_pct,
            width_pct=width_pct,
            height_pct=height_pct,
            fields=fields,
            banner_color=banner_color,
        )
        initial_values = initial_values or {}
        self.inputs = {label: initial_values.get(label, "") for label in self.fields}
        self.active_field = None
        self.font = pygame.font.Font(None, 22)
        self.input_width_pct = input_width_pct
        self.render()

    def render(self):
        super().render()  # Draw background, title, etc.
        if not hasattr(self, "font"):
            print("⏭️ Skipping TreePanel render — self.data not yet set or is empty.")
            return

        for i, label in enumerate(self.fields):
            label_rect = self.get_label_rect(i)
            input_rect = self.get_input_rect(i)

            # Render label
            label_surf = self.font.render(f"{label}:", True, (255, 255, 255))
            self.parent.blit(label_surf, label_rect)

            # Render input box
            box_color = (0, 255, 0) if label == self.active_field else (255, 255, 255)
            pygame.draw.rect(self.parent, box_color, input_rect, 2)

            # Render text
            text = self.inputs[label]
            text_surf = self.font.render(text, True, (255, 255, 255))
            self.parent.blit(text_surf, (input_rect.x + 5, input_rect.y + 5))

    def handle_events(self, event, parent_offset_x=0, parent_offset_y=0):
        self.handle_click(event, parent_offset_x, parent_offset_y)
        self.handle_key(event)

    def handle_click(self, event, parent_offset_x=0, parent_offset_y=0):
        if event.type == pygame.MOUSEBUTTONDOWN:
            local_x = event.pos[0] - self.left - parent_offset_x
            local_y = event.pos[1] - self.top - parent_offset_y

            for i, label in enumerate(self.fields):
                rect = pygame.Rect(
                    self.get_input_rect(i).x - self.left,
                    self.get_input_rect(i).y - self.top,
                    self.get_input_rect(i).width,
                    self.get_input_rect(i).height,
                )
                if rect.collidepoint(local_x, local_y):
                    self.active_field = label
                    return
            self.active_field = None

    def handle_key(self, event):
        if self.active_field and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.inputs[self.active_field] = self.inputs[self.active_field][:-1]
            elif event.key == pygame.K_TAB:
                labels = list(self.fields)
                idx = labels.index(self.active_field) if self.active_field in labels else 0
                if event.mod & pygame.KMOD_SHIFT:
                    self.active_field = labels[(idx - 1) % len(labels)]
                else:
                    self.active_field = labels[(idx + 1) % len(labels)]
            else:
                char = event.unicode
                if char.isprintable():
                    self.inputs[self.active_field] += char

            self.render()

    def get_input_rect(self, index):
        row_height = 40
        y = self.top + 40 + index * row_height
        return pygame.Rect(
            self.left + int(self.width * (1 - self.input_width_pct / 100.0)),
            y,
            int(self.width * (self.input_width_pct / 100.0)),
            30,
        )
