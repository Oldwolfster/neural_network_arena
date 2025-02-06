import pygame
import pygame_menu
from src.neuroForge import mgr
from src.engine.RamDB import RamDB

# A simple callback to use for the menu items
show_menu = False

def some_callback():
    print("Callback executed!")
def close_menu():
    mgr.menu_active = False


def create_menu(WIDTH : int, HEIGHT : int, db: RamDB):
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
    # Use the separate function to generate the dynamic reporting menu
    reporting_menu = create_reporting_menu(WIDTH, HEIGHT, db)

    # --- Submenu: Settings ---
    settings_menu = pygame_menu.Menu('Settings', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)
    settings_menu.add.button('Appearance', some_callback)
    settings_menu.add.button('Computation', some_callback)
    settings_menu.add.button('Data & Storage', some_callback)
    settings_menu.add.button('Developer Mode', some_callback)
    settings_menu.add.button('Back', lambda: menu.disable())

    # Add main menu items linking to each submenu
    menu.add.button('Reporting', reporting_menu)
    menu.add.button('The Forge', forge_menu)
    menu.add.button('Gladiators', gladiators_menu)
    menu.add.button('Arenas', arenas_menu)
    menu.add.button('Settings', settings_menu)
    menu.add.button('Quit', close_menu )
    return menu

from src.engine.Utils import dynamic_instantiate
def load_report(report_name, db: RamDB):
    """Callback function for opening a report.

    The closure happens in the lambda function that captures `db` and `report_name`
    when creating the menu buttons.
    """
    #print(f"Loading report: {report_name}")
    report = dynamic_instantiate(report_name, "reports", db)
    report.run_report() # db is now inside the report instance (from the constructor)



def create_reporting_menu(WIDTH: int, HEIGHT: int, db : RamDB):
    """Creates the Reporting menu dynamically based on available reports."""
    reporting_menu = pygame_menu.Menu('Reporting', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_DARK)

    reports = list_reports("Reports")  # Scan the reports folder

    if reports:
        for report in reports:
            reporting_menu.add.button(report, lambda r=report: load_report(r, db))
    else:
        reporting_menu.add.label("No Reports Found")

    reporting_menu.add.button('Back', pygame_menu.events.BACK)

    return reporting_menu

import os

def list_reports(directory="reports"):
    """Returns a list of valid report files (.py) from the reports folder (one level up from UI)."""

    # Navigate up one level (from UI to src)
    base_dir = os.path.dirname(__file__)  # Current directory (UI/)
    search_directory = os.path.join(os.path.dirname(base_dir), directory)  # Moves up to src/, then into reports/

    reports = []

    if os.path.exists(search_directory):
        #print(f"Searching directory: {search_directory}")  # Debugging output

        for file in os.listdir(search_directory):
            #print(f"Found file: {file}")  # Debugging output

            if file.endswith(".py") and not file.startswith("_"):  # Only .py files, ignore _ prefixed
                reports.append(file[:-3])
                #print(f"Added to report list: {file}")  # Debugging output

    else:
        print(f"Directory not found: {search_directory}")  # Debugging output

    #print(f"Final reports list: {reports}")  # Debugging output
    return reports
