import pygame
import pygame_gui

def create_report_parameters_form(window_surface, manager):
    # Create a panel to hold the form elements
    panel_rect = pygame.Rect(50, 50, 400, 500)
    panel = pygame_gui.elements.UIPanel(relative_rect=panel_rect,
                                        manager=manager)

    # Text Entry Field
    text_entry_rect = pygame.Rect(10, 10, 380, 40)
    text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=text_entry_rect,
                                                     manager=manager,
                                                     container=panel,
                                                     placeholder_text="Enter Report Name")

    # Drop-down Menu
    drop_down_rect = pygame.Rect(10, 60, 380, 40)
    drop_down = pygame_gui.elements.UIDropDownMenu(options_list=['Option 1', 'Option 2', 'Option 3'],
                                                   starting_option='Option 1',
                                                   relative_rect=drop_down_rect,
                                                   manager=manager,
                                                   container=panel)

    # Slider
    slider_rect = pygame.Rect(10, 120, 380, 40)
    slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=slider_rect,
                                                    start_value=50,
                                                    value_range=(0, 100),
                                                    manager=manager,
                                                    container=panel)



    # Radio Button Group
    radio_button_rect1 = pygame.Rect(10, 240, 380, 40)
    radio_button1 = pygame_gui.elements.UIButton(relative_rect=radio_button_rect1,
                                                 text="Option A",
                                                 manager=manager,
                                                 container=panel,
                                                 object_id="#option_a")

    radio_button_rect2 = pygame.Rect(10, 290, 380, 40)
    radio_button2 = pygame_gui.elements.UIButton(relative_rect=radio_button_rect2,
                                                 text="Option B",
                                                 manager=manager,
                                                 container=panel,
                                                 object_id="#option_b")

    radio_button_rect2 = pygame.Rect(10, 340, 380, 40)
    radio_button2 = pygame_gui.elements.UIButton(relative_rect=radio_button_rect3,
                                                 text="Option C",
                                                 manager=manager,
                                                 container=panel,
                                                 object_id="#option_b")

    # Submit Button
    submit_button_rect = pygame.Rect(10, 350, 380, 40)
    submit_button = pygame_gui.elements.UIButton(relative_rect=submit_button_rect,
                                                 text="Generate Report",
                                                 manager=manager,
                                                 container=panel)

    return panel, text_entry, drop_down, slider,  radio_button1, radio_button2, submit_button

def main():
    pygame.init()

    # Set up the window and UI manager
    window_surface = pygame.display.set_mode((800, 600))
    manager = pygame_gui.UIManager((800, 600))

    # Create the form
    form_elements = create_report_parameters_form(window_surface, manager)

    clock = pygame.time.Clock()
    is_running = True

    while is_running:
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == form_elements[-1]:  # Submit button
                    print("Report Parameters:")
                    print(f"Report Name: {form_elements[1].get_text()}")
                    print(f"Selected Option: {form_elements[2].selected_option}")
                    print(f"Slider Value: {form_elements[3].get_current_value()}")
                    print(f"Feature Enabled: {form_elements[4].checked}")
                    if form_elements[5].check_pressed():
                        print("Selected Option: A")
                    elif form_elements[6].check_pressed():
                        print("Selected Option: B")

            manager.process_events(event)

        manager.update(time_delta)

        window_surface.fill((255, 255, 255))
        manager.draw_ui(window_surface)

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()