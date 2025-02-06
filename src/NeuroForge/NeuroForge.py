import pygame
import sys
from src.ArenaSettings import HyperParameters
from typing import List
from src.neuroForge.Display_Manager import DisplayManager
from src.UI.Menus import create_menu

from src.engine.Utils_DataClasses import ModelInfo
from src.engine.RamDB import RamDB
#from src.neuroForge.mgr import screen
from src.neuroForge.mgr import * # Imports everything into the local namespace
from src.neuroForge import mgr # Keeps the module reference for assignments
import tkinter.messagebox as mb

#def minimize_window():
#    hwnd = pygame.display.get_wm_info()["window"]  # Get the window handle
#    ctypes.windll.user32.ShowWindow(hwnd, 6)  # 6 is the command for minimizing the window
def neuroForge(db: RamDB, training_data, hyper: HyperParameters, model_info_list: List[ModelInfo]):

    neuro_forge_init()
    display_manager = DisplayManager(mgr.screen, hyper, db)
    display_manager.initialize(model_info_list)  # Set up all components

    # Initialize tracking variables
    last_iteration = mgr.iteration -1 # -1 makes it trigger it the first time.
    last_epoch = mgr.epoch
    menu_button_rect= create_menu_button_rect()
    menu = create_menu(mgr.screen_width, mgr.screen_height, db)

    # Pygame main loop
    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            check_menu_button(event, menu_button_rect)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                respond_to_UI(event)

        # Check if iteration or epoch has changed
        if mgr.iteration != last_iteration or mgr.epoch != last_epoch:
            display_manager.update( db, mgr.iteration, mgr.epoch,'global no model') # Query and update data
            last_iteration = mgr.iteration  # Update tracking variables
            last_epoch = mgr.epoch          # Update tracking variables
        # finish pygame tasks
        mgr.screen.fill((255, 255, 255))  # Clear screen
        display_manager.render() # Render models
        draw_button(menu_button_rect)
        if mgr.menu_active and menu.is_enabled() :
            menu.update(events)
            menu.draw(mgr.screen)

        pygame.display.flip()

def check_menu_button(event, menu_button_rect):
    if event.type == pygame.MOUSEBUTTONDOWN:
        if menu_button_rect.collidepoint(event.pos):
            print(f"I've been clicked,{mgr.menu_active} ")
            mgr.menu_active = True


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

    # Check for out of range conditions
    if mgr.epoch < 1:  # Check if trying to move past the beginning
        mb.showinfo("Out of Range", "You cannot go past the first epoch!")
        mgr.epoch = 1
        mgr.iteration = 1

    if mgr.epoch > mgr.max_epoch:  # Check if trying to move past the end
        mb.showinfo("Out of Range", "You are at the end!")
        mgr.epoch = mgr.max_epoch
        mgr.iteration = mgr.max_iteration

def create_menu_button_rect():
    top = 40
    width = 140
    left = mgr.screen_width - 30 - width
    height = 40
    menu_button_rect = pygame.Rect(left,top, width,  height)
    return menu_button_rect

def draw_button(menu_button_rect):    # Draw the "Open Menu" button
    main_button_color = ( 0, 0, 255)
    border_radius = 10  # Adjust for roundness
    shadow_offset = 4  # Depth effect

    # Draw shadow
    shadow_color = (30, 30, 100)  # Darker red for depth
    shadow_rect = menu_button_rect.move(shadow_offset, shadow_offset)
    pygame.draw.rect(mgr.screen, shadow_color, shadow_rect, border_radius=border_radius)

    # Draw main button
    pygame.draw.rect(mgr.screen, main_button_color, menu_button_rect, border_radius=border_radius)

    # Draw text
    font = pygame.font.SysFont(None, 32)
    text_surface = font.render("Open Menu", True, mgr.white)
    text_rect = text_surface.get_rect(center=menu_button_rect.center)
    mgr.screen.blit(text_surface, text_rect)


def neuro_forge_init():
    pygame.init()
    # Font Setup
    mgr.font = pygame.font.Font(None, 24)  # Default font, size 24
    # Screen Settings
    mgr.screen = pygame.display.set_mode((mgr.screen_width, mgr.screen_height))
    pygame.display.set_caption("Neural Network Visualization")
