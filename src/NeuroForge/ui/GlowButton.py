import pygame

class GlowButton:

    def __init__(self, parent_surface, text, left_pct, top_pct, width_pct, height_pct, on_click):
        self.surface = parent_surface
        self.text = text
        self.on_click = on_click
        self.font = pygame.font.Font(None, 32)

        # Calculate pixel values from percentages
        pw, ph = parent_surface.get_size()
        self.width = int(pw * (width_pct / 100))
        self.height = int(ph * (height_pct / 100))
        self.x = int(pw * (left_pct / 100))
        self.y = int(ph * (top_pct / 100))

        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def render(self):
        # Draw opaque background
        pygame.draw.rect(self.surface, (0, 0, 0), self.rect, border_radius=8)

        # Draw border
        pygame.draw.rect(self.surface, (255, 255, 255), self.rect, 2, border_radius=8)

        # Render glow text and center it
        glow = self.font.render(self.text, True, (255, 100, 0))
        label = self.font.render(self.text, True, (255, 255, 255))
        text_rect = label.get_rect(center=self.rect.center)

        self.surface.blit(glow, (text_rect.x + 2, text_rect.y + 2))
        self.surface.blit(label, text_rect)

    def handle_events(self, event, parent_offset_x=0, parent_offset_y=0):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Convert global to local coordinates
            #print("In button click")
            local_x = event.pos[0] - self.x - parent_offset_x
            local_y = event.pos[1] - self.y - parent_offset_y
            if pygame.Rect(0, 0, self.rect.width, self.rect.height).collidepoint(local_x, local_y):
                #print("CLICK IS ON BUTTON")
                self.on_click()
