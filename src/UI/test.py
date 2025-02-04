import pygame
import pygame_menu

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pygame Menu Example")

# A simple callback to use for the menu items
def some_callback():
    print("Callback executed!")

# Function to create and return the menu
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
    settings_menu.add.button('Back', pygame_menu.events.BACK)

    # Add main menu items linking to each submenu
    menu.add.button('The Forge', forge_menu)
    menu.add.button('Gladiators', gladiators_menu)
    menu.add.button('Arenas', arenas_menu)
    menu.add.button('Reporting', reporting_menu)
    menu.add.button('Settings', settings_menu)
    menu.add.button('Quit', pygame_menu.events.EXIT)

    return menu

# Create the menu
menu = create_menu(WIDTH, HEIGHT)

# Main game loop
running = True
show_menu = False  # Controls whether the menu is displayed

while running:
    # Handle Pygame events
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Toggle menu on ESC key
                show_menu = not show_menu

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw your game here (if not showing the menu)
    if not show_menu:
        # Example: Draw something on the screen
        pygame.draw.rect(screen, (255, 0, 0), (100, 100, 200, 200))

    # Update the menu if it's visible
    if show_menu:
        # Update the menu and check if it should close
        menu.update(events)
        menu.draw(screen)

        # Check if the menu is requesting to close (e.g., via the close button or "Quit" option)
        if menu.is_enabled() and not menu.is_active():
            show_menu = False  # Hide the menu

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()