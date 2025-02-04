import pygame
import pygame_menu
from src.NeuroForge import mgr
# A simple callback to use for the menu items
show_menu = False

def some_callback():
    print("Callback executed!")
def close_menu():
    mgr.menu_active = False


def create_menu(WIDTH, HEIGHT):
    # Create the main menu
    menu = pygame_menu.Menu('Main Menu', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)

    # --- Submenu: The Forge ---
    forge_menu = pygame_menu.Menu('The Forge', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    forge_menu.add.button('New Model', some_callback)
    forge_menu.add.button('Load Template', some_callback)
    forge_menu.add.button('Edit Existing', some_callback)
    forge_menu.add.button('Export/Save', some_callback)
    forge_menu.add.button('Back', pygame_menu.events.BACK)

    # --- Submenu: Gladiators ---
    gladiators_menu = pygame_menu.Menu('Gladiators', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    gladiators_menu.add.button('Train a Gladiator', some_callback)
    gladiators_menu.add.button('View Gladiators', some_callback)
    gladiators_menu.add.button('Compare Gladiators', some_callback)
    gladiators_menu.add.button('Fine-tune Training', some_callback)
    gladiators_menu.add.button('Back', pygame_menu.events.BACK)

    # --- Submenu: Arenas ---
    arenas_menu = pygame_menu.Menu('Arenas', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    arenas_menu.add.button('Choose Arena', some_callback)
    arenas_menu.add.button('Create Custom Arena', some_callback)
    arenas_menu.add.button('Modify Arena', some_callback)
    arenas_menu.add.button('Test Gladiators in Arena', some_callback)
    arenas_menu.add.button('Back', pygame_menu.events.BACK)

    # --- Submenu: Reporting ---
    reporting_menu = pygame_menu.Menu('Reporting', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    reporting_menu.add.button('Training History', some_callback)
    reporting_menu.add.button('Visual Debugger', some_callback)
    reporting_menu.add.button('Performance Charts', some_callback)
    reporting_menu.add.button('Export Reports', some_callback)
    reporting_menu.add.button('Back', pygame_menu.events.BACK)

    # --- Submenu: Settings ---
    settings_menu = pygame_menu.Menu('Settings', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    settings_menu.add.button('Appearance', some_callback)
    settings_menu.add.button('Computation', some_callback)
    settings_menu.add.button('Data & Storage', some_callback)
    settings_menu.add.button('Developer Mode', some_callback)
    settings_menu.add.button('Back', lambda: menu.disable())

    # Add main menu items linking to each submenu
    menu.add.button('The Forge', forge_menu)
    menu.add.button('Gladiators', gladiators_menu)
    menu.add.button('Arenas', arenas_menu)
    menu.add.button('Reporting', reporting_menu)
    menu.add.button('Settings', settings_menu)
    menu.add.button('Quit', close_menu )

    return menu



""" Below code does bring up the damn menu
def draw_button(screen):    # Draw the "Open Menu" button
    WHITE = (255, 255, 255)
    BLUE = (50, 50, 255)
    top = 40
    width = 150
    left = 40
    height = 40
    menu_button_rect = pygame.Rect(left,top, width,  height)
    pygame.draw.rect( screen, BLUE, menu_button_rect)
    font = pygame.font.SysFont(None, 36)
    text_surface = font.render("Open Menu", True, WHITE)
    text_rect = text_surface.get_rect(center=menu_button_rect.center)



def pygame_loop():
    menu = None
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Menu Standalone Test")
    clock = pygame.time.Clock()

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            # Toggle the menu with the 'M' key
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    print("M has been pressed")
                    if menu is None:
                        menu =create_menu(400,400)
                        print("Here 2")
                        print(f"menu.is_enabled()={menu.is_enabled()}")
                    else:
                        if menu.is_enabled():
                            print("Here 3")
                            print(f"menu.is_enabled()={menu.is_enabled()}")
                            menu.disable()
                        else:
                            menu.enable()
                            print("Here 4")
                            print(f"menu.is_enabled()={menu.is_enabled()}")

        # Clear the screen
        screen.fill((0, 0, 0))
        draw_button(screen)
        # If the menu is enabled, update and draw it on top of the screen.
        if menu and menu.is_enabled():
            menu.update(events)
            menu.draw(screen)
        else:
            # Optionally, display instructions when the menu is not active.
            font = pygame.font.SysFont(None, 36)
            text_surface = font.render("Press M to open the menu", True, (255, 255, 255))
            screen.blit(text_surface, (50, 50))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    pygame_loop()

"""