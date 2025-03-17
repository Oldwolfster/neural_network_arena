import pygame
import sys
import pygame_gui
from src.ArenaSettings import HyperParameters
from typing import List
from src.neuroForge_original.Display_Manager import DisplayManager
from src.UI.Menus import create_menu
from src.engine.Utils_DataClasses import ModelInfo
from src.engine.RamDB import RamDB
from src.neuroForge_original import mgr # Keeps the module reference for assignments


def neuroForge(db: RamDB, training_data, hyper: HyperParameters, model_info_list: List[ModelInfo]):
    neuro_forge_init()
    #TODO get theme working
    #ui_manager = pygame_gui.UIManager((mgr.screen.get_width(), mgr.screen.get_height()),"ui_theme.json")
    ui_manager = pygame_gui.UIManager((mgr.screen.get_width(), mgr.screen.get_height()))
    display_manager = DisplayManager(mgr.screen, hyper, db, model_info_list, ui_manager)
    #display_manager.initialize(model_info_list)  # Set up all components

    # Initialize tracking variables
    last_iteration = mgr.iteration -1 # -1 makes it trigger it the first time.
    last_epoch = mgr.epoch
    menu_button_rect = create_menu_button_rect()
    colr_button_rect = create_colr_button_rect()
    menu = create_menu(mgr.screen_width, mgr.screen_height, db)
    #colr = create_colr(mgr.screen_width, mgr.screen_height, db)

    clock = pygame.time.Clock()

    # Pygame main loop
    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0  # Convert to seconds
        events=handle_events(menu_button_rect, display_manager, ui_manager)
        display_manager.gameloop_hook()
        # Check if iteration or epoch has changed
        if mgr.iteration != last_iteration or mgr.epoch != last_epoch:
            display_manager.update(mgr.iteration, mgr.epoch,'global no model') # Query and update data
            last_iteration = mgr.iteration  # Update tracking variables
            last_epoch = mgr.epoch          # Update tracking variables
        # finish pygame tasks
        ui_manager.update(time_delta)  # ✅ Ensure pygame_gui updates every frame
        mgr.screen.fill((255, 255, 255))  # Clear screen
        display_manager.render() # Render models
        draw_button(menu_button_rect, "Menu", 5)
        draw_button(colr_button_rect, "Colors", -5)
        ui_manager.draw_ui(mgr.screen)
        if mgr.menu_active and menu.is_enabled() :
            menu.update(events)
            menu.draw(mgr.screen)

        pygame.display.flip()


def handle_events(menu_button_rect, display_manager, ui_manager):
    print("Checking events")
    events = pygame.event.get()
    for event in events:
        print(f"NeuroForge: event={event} ")
        ui_manager.process_events(event)  # ✅ Fix: Ensure pygame_gui gets events
        check_menu_button(event, menu_button_rect)
        display_manager.process_events(event)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            respond_to_UI(event)
    return events

def respond_to_UI(event):
    if event.key == pygame.K_l:  # Advance frame
        mgr.iteration += 1
        if mgr.iteration > mgr.max_iteration:
            mgr.epoch += 1
            mgr.iteration = 1

    if event.key == pygame.K_j:  # Reverse frame
        mgr.iteration -= 1
        if mgr.iteration == 0:
            mgr.epoch -= 1
            mgr.iteration = mgr.max_iteration

    if event.key == pygame.K_o:  # Advance epoch
        mgr.epoch += 1
        mgr.iteration = 1  # Reset to the first iteration of the new epoch

    if event.key == pygame.K_u:  # Reverse epoch
        mgr.epoch -= 1
        mgr.iteration = 1  # Reset to the first iteration of the new epoch
    #validate_epoch_change() # Check for out of range conditions
"""
    # Check for out of range conditions
    if mgr.epoch < 1:  # Check if trying to move past the beginning
        mb.showinfo("Out of Range", "You cannot go past the first epoch!")
        mgr.epoch = 1
        mgr.iteration = 1

    if mgr.epoch > mgr.max_epoch:  # Check if trying to move past the end
        mb.showinfo("Out of Range", "You are at the end!")
        mgr.epoch = mgr.max_epoch
        mgr.iteration = mgr.max_iteration
"""
def create_menu_button_rect():
    top = 40
    width = 140
    left = mgr.screen_width - 30 - width
    height = 40
    menu_button_rect = pygame.Rect(left,top, width,  height)
    return menu_button_rect

def create_colr_button_rect():
    top = 40
    width = 140
    left = 30  # mgr.screen_width - 30 - width
    height = 40
    colors_button_rect = pygame.Rect(left,top, width,  height)
    return colors_button_rect
def check_menu_button(event, menu_button_rect):
    if event.type == pygame.MOUSEBUTTONDOWN:
        if menu_button_rect.collidepoint(event.pos):
            #print(f"I've been clicked,{mgr.menu_active} ")
            mgr.menu_active = True

def draw_button(menu_button_rect, button_text :str, shadow_offset: int):    # Draw the "Open Menu" button
    main_button_color = ( 0, 0, 255)
    border_radius = 10  # Adjust for roundness

    # Draw shadow
    shadow_color = (30, 30, 100)  # Darker red for depth
    shadow_rect = menu_button_rect.move(shadow_offset, abs( shadow_offset))
    pygame.draw.rect(mgr.screen, shadow_color, shadow_rect, border_radius=border_radius)

    # Draw main button
    pygame.draw.rect(mgr.screen, main_button_color, menu_button_rect, border_radius=border_radius)

    # Draw text
    font = pygame.font.SysFont(None, 32)
    text_surface = font.render(button_text, True, mgr.white)
    text_rect = text_surface.get_rect(center=menu_button_rect.center)
    mgr.screen.blit(text_surface, text_rect)


def neuro_forge_init():
    pygame.init()
    # Font Setup
    mgr.font = pygame.font.Font(None, 24)  # Default font, size 24
    # Screen Settings
    mgr.screen = pygame.display.set_mode((mgr.screen_width, mgr.screen_height))
    pygame.display.set_caption("Neural Network Visualization")
