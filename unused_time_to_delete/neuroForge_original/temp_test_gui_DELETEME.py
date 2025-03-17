import pygame
import pygame_gui

# Initialize pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 400, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Standalone Text Box Example")

# Create UI manager
ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT))

# Create a text input box
text_box = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((100, 100), (200, 30)),  # (x, y), (width, height)
    manager=ui_manager
)

# Clock for frame rate control
clock = pygame.time.Clock()
running = True

while running:
    time_delta = clock.tick(30) / 1000.0  # Convert to seconds

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        ui_manager.process_events(event)  # Pass events to pygame_gui

    # Update UI Manager
    ui_manager.update(time_delta)

    # Clear screen and draw UI
    screen.fill((30, 30, 30))  # Dark background
    ui_manager.draw_ui(screen)

    pygame.display.update()

pygame.quit()
