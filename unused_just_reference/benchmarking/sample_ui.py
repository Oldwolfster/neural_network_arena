import pygame
import pygame_gui

pygame.init()

# 🪟 Window setup
pygame.display.set_caption('🏛️ Coliseum Match Config')
window_size = (800, 700)
window_surface = pygame.display.set_mode(window_size)

# 🧠 UI Manager
ui_manager = pygame_gui.UIManager(window_size)

# 🎮 Dropdowns
training_pit_dropdown = pygame_gui.elements.UIDropDownMenu(
    options_list=["XOR", "Predict_Income_2_Inputs", "Predict_Income_Piecewise_Growth"],
    starting_option="Predict_Income_2_Inputs",
    relative_rect=pygame.Rect((50, 50), (300, 30)),
    manager=ui_manager
)

gladiator_dropdown = pygame_gui.elements.UIDropDownMenu(
    options_list=["NeuroForge_Template", "TestBatch"],
    starting_option="NeuroForge_Template",
    relative_rect=pygame.Rect((50, 100), (300, 30)),
    manager=ui_manager
)

# 📥 Text Inputs
epoch_input = pygame_gui.elements.UITextEntryLine(pygame.Rect((50, 150), (300, 30)), ui_manager)
epoch_input.set_text("22")

train_size_input = pygame_gui.elements.UITextEntryLine(pygame.Rect((50, 200), (300, 30)), ui_manager)
train_size_input.set_text("33")

learning_rate_input = pygame_gui.elements.UITextEntryLine(pygame.Rect((50, 250), (300, 30)), ui_manager)
learning_rate_input.set_text(".01")

# 🟢 Run Button
run_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((50, 310), (150, 40)),
    text='Run Match',
    manager=ui_manager
)

# ⏱️ Main Loop
clock = pygame.time.Clock()
is_running = True

while is_running:
    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

        if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == run_button:
            print("Running Match With Settings:")
            print("Arena:", training_pit_dropdown.selected_option)
            print("Gladiator:", gladiator_dropdown.selected_option)
            print("Epochs:", epoch_input.get_text())
            print("Training Set Size:", train_size_input.get_text())
            print("Learning Rate:", learning_rate_input.get_text())

        ui_manager.process_events(event)

    ui_manager.update(time_delta)
    window_surface.fill((30, 30, 30))
    ui_manager.draw_ui(window_surface)
    pygame.display.update()

pygame.quit()
