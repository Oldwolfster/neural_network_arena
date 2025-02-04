import pygame
import pygame_menu

class SimpleMenu:
    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 36)
        self.menu_items = []
        self.selected_index = 0
        self.running = False

        # Colors
        self.BACKGROUND = (50, 50, 50)
        self.TEXT_COLOR = (255, 255, 255)
        self.SELECTED_COLOR = (255, 255, 0)

    def add.button(self, text, action=None):
        self.menu_items.append({
            'text': text,
            'action': action
        })

    def draw(self):
        # Semi-transparent background
        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        s.fill((0, 0, 0, 128))
        self.screen.blit(s, (0, 0))

        # Calculate spacing
        item_height = 50
        start_y = (self.height - len(self.menu_items) * item_height) // 2

        # Draw menu items
        for i, item in enumerate(self.menu_items):
            color = self.SELECTED_COLOR if i == self.selected_index else self.TEXT_COLOR
            text_surface = self.font.render(item['text'], True, color)
            text_rect = text_surface.get_rect(center=(self.width // 2, start_y + i * item_height))
            self.screen.blit(text_surface, text_rect)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected_index = (self.selected_index - 1) % len(self.menu_items)
                elif event.key == pygame.K_DOWN:
                    self.selected_index = (self.selected_index + 1) % len(self.menu_items)
                elif event.key == pygame.K_RETURN:
                    # Execute the selected item's action
                    action = self.menu_items[self.selected_index]['action']
                    if action:
                        action()
                    else:
                        self.running = False
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

        return True

    def run(self):
        self.running = True
        self.selected_index = 0

        while self.running:
            if not self.handle_events():
                break

            self.draw()
            pygame.display.flip()

        return self.running

def run_menu(WIDTH, HEIGHT, screen):
    # Create menu
    menu = SimpleMenu(screen, WIDTH, HEIGHT)

    # Define actions (these are placeholders)
    def forge_action():
        print("The Forge selected")

    def gladiators_action():
        print("Gladiators selected")

    def arenas_action():
        print("Arenas selected")

    def reporting_action():
        print("Reporting selected")

    def settings_action():
        print("Settings selected")

    # Add menu items
    menu.add.button('The Forge', forge_action)
    menu.add.button('Gladiators', gladiators_action)
    menu.add.button('Arenas', arenas_action)
    menu.add.button('Reporting', reporting_action)
    menu.add.button('Settings', settings_action)
    menu.add.button('Quit')

    # Run the menu
    return menu.run()