import pygame

class GlowText:
    def __init__(self, text, x , y, font, surface,  color=(255, 50, 0), glow_color=(255, 150, 50), glow_strength=5):
        self.text = text
        self.surface=surface
        self.font = font
        self.color = color
        self.glow_color = glow_color
        self.x = x
        self.y = y
        self.glow_strength = glow_strength
        self.glow_surface, self.text_surface = self.create_glow()

    def create_glow(self):
        base = self.font.render(self.text, True, self.color)
        glow = self.font.render(self.text, True, self.glow_color)
        for _ in range(self.glow_strength):
            glow = pygame.transform.smoothscale(glow, (glow.get_width() + 1, glow.get_height() + 1))
        return glow, base

    def draw_me(self):
        self.surface.blit(self.glow_surface, (self.x - 5, self.y - 5))
        self.surface.blit(self.text_surface, (self.x, self.y))

    def update_me(self):
        pass