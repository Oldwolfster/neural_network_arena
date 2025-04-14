import pygame

class GlowText:
    def __init__(self, text, x , y, font,  surface, font2=None, color=(255, 50, 0), glow_color=(255, 150, 50), glow_strength=5, center=False):
        self.text = text
        self.surface=surface
        self.font = font
        self.font2= font2
        self.color = color
        self.glow_color = glow_color
        self.x = x
        self.y = y
        self.glow_strength = glow_strength
        self.glow_surface, self.text_surface = self.create_glow()

        # If centering is requested, override x
        if center:
            self.x = (self.surface.get_width() - self.text_surface.get_width()) // 2

    def create_glow(self):
        base = self.font.render(self.text, True, self.color)
        if self.font2 is not None:
            glow = self.font.render(self.text, True, self.glow_color)
        else:
            glow = self.font.render(self.text, True, self.glow_color)
        for _ in range(self.glow_strength):
            glow = pygame.transform.smoothscale(glow, (glow.get_width() + 1, glow.get_height() + 1))
        return glow, base

    def draw_me(self):
        self.surface.blit(self.glow_surface, (self.x - 3, self.y - 3))
        self.surface.blit(self.text_surface, (self.x, self.y))

    def update_me(self):
        pass