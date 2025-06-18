import pygame

def smart_format(num):
    try:
        num = float(num)  # Ensure input is a number
    except (ValueError, TypeError):

        return str(num)  # If conversion fails, return as is

    if num == 0:
        return "0"
    #elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
    #    return f"{num:.2e}"

    elif abs(num) >= 1e8:  # Very large → scientific
        return f"{num:.1e}"
    elif abs(num) < 0.001:  # Use 6 decimal places for small numbers
        #formatted = f"{num:,.6f}"
        return f"{num:.1e}"
    elif abs(num) < 1:  # Use 3 decimal places for numbers less than 1
        formatted = f"{num:,.3f}"
#    elif abs(num) > 1e5:  # Use 6 decimal places for small numbers
#        return f"{num:.1e}"
    elif abs(num) > 1000:  # Use no decimal places for large numbers
        formatted = f"{num:,.0f}"

    else:  # Default to 2 decimal places
        formatted = f"{num:,.2f}"

    # Remove trailing zeros and trailing decimal point if necessary
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def get_text_rect(text: str, font_size: int):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, (0,0,0))
    return text_surface.get_rect()

def draw_text_with_background(screen, value_to_print, x, y, font_size, text_color=(255, 255, 255), bg_color=(0, 0, 0), right_align=False, border_color=None):
    """
    Draws text with a background rectangle for better visibility.

    :param right_align: If True, the text is right-aligned to x; otherwise, x is the left edge.
    :param border_color: If True, adds a black border
    """
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(smart_format(value_to_print), True, text_color)
    text_rect = text_surface.get_rect()

    # Original logic if right_align is False
    if not right_align:
        text_rect.topleft = (x, y)
    else:
        # If right_align is True, place the text so its right edge is at x
        text_rect.topright = (x, y)

    if not border_color is None:
        pygame.draw.rect(screen, border_color, text_rect.inflate(9, 7))  # Slight padding around text
        screen.blit(text_surface, text_rect)

    # Draw background rectangle
    pygame.draw.rect(screen, bg_color, text_rect.inflate(6, 4))  # Slight padding around text
    screen.blit(text_surface, text_rect)
def get_darker_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Given a background RGB color, this function returns an RGB tuple for a darker color,


    Parameters:
        rgb (tuple[int, int, int]): A tuple representing the background color (R, G, B).

    Returns:
        tuple[int, int, int]: An RGB tuple darker color
    """
    r, g, b = rgb
    towards_color = 11

    return (min(r+ towards_color, 255) / 2,min(g+ towards_color, 255) / 2,min(b+ towards_color, 255) / 2,)


def get_contrasting_text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Given a background RGB color, this function returns an RGB tuple for either black or white text,
    whichever offers better readability.

    The brightness is computed using the formula:
        brightness = (R * 299 + G * 587 + B * 114) / 1000
    which is a standard formula for perceived brightness. If the brightness is greater than 128,
    the background is considered light and black text is returned; otherwise, white text is returned.

    Parameters:
        rgb (tuple[int, int, int]): A tuple representing the background color (R, G, B).

    Returns:
        tuple[int, int, int]: An RGB tuple for the text color (either (0, 0, 0) for black or (255, 255, 255) for white).
    """
    r, g, b = rgb
    # Calculate the perceived brightness of the background color.
    brightness = (r * 299 + g * 587 + b * 114) / 1000

    # Choose black text for light backgrounds and white text for dark backgrounds.
    if brightness > 128:
        return (0, 0, 0)  # Black text for lighter backgrounds.
    else:
        return (255, 255, 255)  # White text for darker backgrounds.


def draw_gradient_rect( surface, rect, color1, color2):
    #print(f"rect.height={rect.height}")
    safe_height = min(rect.height, 1500)  # Clamp height to prevent hanging if height explodes. 2E31 lines drawn
    for i in range(safe_height):
        ratio = i / safe_height
        blended_color = [
            int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
        ]
        pygame.draw.line(surface, blended_color, (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))


def check_label_collision(new_label_rect, existing_labels_rects):
    """
    Checks if the new label's rect collides with any of the existing label rectangles.

    :param new_label_rect: A pygame.Rect representing the new label's boundaries.
    :param existing_labels_rects: A list of pygame.Rect objects for already placed labels.
    :return: True if there is a collision with any existing label, False otherwise.
    """
    for rect in existing_labels_rects:
        if new_label_rect.colliderect(rect):
            return True
    return False

def draw_rect_with_border(screen, rect, color, border_width, border_color=(0,0,0)):
    """
    Draws a rectangle with a border on the given Pygame surface.

    Parameters:
        screen (pygame.Surface): The surface to draw on.
        rect (pygame.Rect): The rectangle defining the position and size.
        color (tuple): The RGB color of the inner rectangle.
        border_color (tuple): The RGB color of the border.
        border_width (int): The thickness of the border.
    """
    # Draw the outer rectangle (border)
    pygame.draw.rect(screen, border_color, rect)

    # Calculate the dimensions of the inner rectangle
    inner_rect = rect.inflate(-2*border_width, -2*border_width)

    # Draw the inner rectangle
    #pygame.draw.rect(screen, color, inner_rect)
    draw_gradient_rect(screen, inner_rect, color, get_darker_color(color))

def draw_gradient_rect(screen, rect, color_start, color_end_before_avg):
    """
    Draws a gradient rectangle from color_start to color_end.
    - screen: Pygame surface
    - rect: Pygame.Rect object defining position and size
    - color_start: RGB color for the top
    - color_end: RGB color for the bottom
    """

    color_end = average_rgb([color_start, color_end_before_avg])
    # Split the height into gradient steps
    num_steps = rect.height
    for i in range(num_steps):
        # Interpolate color
        r = color_start[0] + (color_end[0] - color_start[0]) * i // num_steps
        g = color_start[1] + (color_end[1] - color_start[1]) * i // num_steps
        b = color_start[2] + (color_end[2] - color_start[2]) * i // num_steps
        pygame.draw.line(screen, (r, g, b), (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))


def average_rgb(rgb_colors):
  """Calculates the average RGB color from a list of RGB tuples.

  Args:
    rgb_colors: A list of RGB tuples, where each tuple contains three integers
      representing the red, green, and blue values (0-255).

  Returns:
    A tuple representing the average RGB color, or None if the input list is empty.
  """
  if not rgb_colors:
    return None

  r_sum = 0
  g_sum = 0
  b_sum = 0

  for r, g, b in rgb_colors:
    r_sum += r
    g_sum += g
    b_sum += b

  num_colors = len(rgb_colors)
  r_avg = r_sum / num_colors
  g_avg = g_sum / num_colors
  b_avg = b_sum / num_colors

  return (int(r_avg), int(g_avg), int(b_avg))

def beautify_text(text: str) -> str:
    """
    Turn things_likeThis_andThat into:
      'Things Like This And That'
    """
    # First pass: mark every position where we need a space
    breaks = [False] * len(text)
    for i in range(1, len(text)):
        if text[i] == "_":
            breaks[i] = True
        elif text[i].isupper() and text[i-1].islower():
            breaks[i] = True

    out = []
    new_word = True
    for i, ch in enumerate(text):
        if ch == "_":
            out.append(" ")
            new_word = True
            continue

        if breaks[i]:
            out.append(" ")
            new_word = True

        # Title-case logic
        if new_word:
            out.append(ch.upper())
        else:
            out.append(ch.lower())
        new_word = False

    return "".join(out)

def is_numeric(text):
    """Validate if text can be safely converted to a number without exceptions."""
    if not isinstance(text, str) or not text:
        return False

    # Handle commas in number format
    text = text.replace(",", "")

    # Check for decimal numbers
    if text.count(".") <= 1:
        # Remove one decimal point if it exists
        text = text.replace(".", "", 1)

    # Check for sign character at beginning
    if text and text[0] in "+-":
        text = text[1:]

    # If we're left with only digits, it's numeric
    return text.isdigit()

def ez_debug(**kwargs):
    """
    Print debug information for each provided variable.

    For every keyword argument passed in, this function prints:
    1) The variable name
    2) An equal sign
    3) The variable's value
    4) A tab character for separation

    Example:
        a = 1
        b = 2
        c = 3
        ez_debug(a=a, b=b, c=c)
        # Output: a=1    b=2    c=3
    """
    debug_output = ""
    for name, value in kwargs.items():
        debug_output += f"{name}={value}\t"
    print(debug_output)
