import pygame
import pygame_menu

# A simple callback to use for the menu items
show_menu = False

def some_callback():
    print("Callback executed!")

def get_menu(WIDTH: int, HEIGHT: int):
    # Create the main menu
    menu = pygame_menu.Menu('Main Menu', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)

    # --- Submenu: The Forge ---
    forge_menu = pygame_menu.Menu('The Forge', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    forge_menu.add.button('New Model', some_callback)
    forge_menu.add.button('Load Template', some_callback)
    forge_menu.add.button('Edit Existing', some_callback)
    forge_menu.add.button('Export/Save', some_callback)
    forge_menu.add.button('Back', lambda: menu.disable())

    # --- Submenu: Gladiators ---
    gladiators_menu = pygame_menu.Menu('Gladiators', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    gladiators_menu.add.button('Train a Gladiator', some_callback)
    gladiators_menu.add.button('View Gladiators', some_callback)
    gladiators_menu.add.button('Compare Gladiators', some_callback)
    gladiators_menu.add.button('Fine-tune Training', some_callback)
    gladiators_menu.add.button('Back', lambda: menu.disable())

    # --- Submenu: Arenas ---
    arenas_menu = pygame_menu.Menu('Arenas', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    arenas_menu.add.button('Choose Arena', some_callback)
    arenas_menu.add.button('Create Custom Arena', some_callback)
    arenas_menu.add.button('Modify Arena', some_callback)
    arenas_menu.add.button('Test Gladiators in Arena', some_callback)
    arenas_menu.add.button('Back', lambda: menu.disable())

    # --- Submenu: Reporting ---
    reporting_menu = pygame_menu.Menu('Reporting', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    reporting_menu.add.button('Training History', some_callback)
    reporting_menu.add.button('Visual Debugger', some_callback)
    reporting_menu.add.button('Performance Charts', some_callback)
    reporting_menu.add.button('Export Reports', some_callback)
    reporting_menu.add.button('Back', lambda: menu.disable())

    # --- Submenu: Settings ---
    settings_menu = pygame_menu.Menu('Settings', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    settings_menu.add.button('Appearance', some_callback)
    settings_menu.add.button('Computation', some_callback)
    settings_menu.add.button('Data & Storage', some_callback)
    settings_menu.add.button('Developer Mode', some_callback)
    settings_menu.add.button('Back', lambda: show_menu == True)

    # Add main menu items linking to each submenu
    menu.add.button('The Forge', forge_menu)
    menu.add.button('Gladiators', gladiators_menu)
    menu.add.button('Arenas', arenas_menu)
    menu.add.button('Reporting', reporting_menu)
    menu.add.button('Settings', settings_menu)
    menu.add.button('Exit Menu', lambda: menu.disable())

    return menu
