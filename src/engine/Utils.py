import random

import numpy as np


import pygame

from src.reports._BaseReport import BaseReport
from src.engine.BaseArena import BaseArena
from src.engine.BaseGladiator import Gladiator


import pygame
import re

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

# Example usage:
#colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
#average_color = average_rgb(colors)
#print(average_color)

def chunk_list(lst: list, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]




def print_call_stack():
    stack = inspect.stack()
    for frame in stack:
        print(f"Function: {frame.function}, Line: {frame.lineno}, File: {frame.filename}")

def smart_format(num):
    try:
        num = float(num)  # Ensure input is a number
    except (ValueError, TypeError):

        return str(num)  # If conversion fails, return as is

    if num == 0:
        return "0"
    #elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
    #    return f"{num:.2e}"
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

def store_num(number):
    formatted = f"{number:,.6f}"
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def smart_format_Feb03(number):
    # Define the formatting for a single number
    def format_single(num):
        if num == 0:
            return "0"
        elif abs(num) < .001:
            return f"{num:.1e}"
        elif abs(num) < 1:
            return f"{num:,.3f}"
        elif abs(num) > 1000:
            return f"{num:,.0f}"
        else:
            return f"{num:,.2f}"

    # Check if `number` is an array or iterable, format each element if so
    if isinstance(number, (np.ndarray, list, tuple)):
        # Apply `format_single` to each element in the array or list
        vectorized_format = np.vectorize(format_single)
        return vectorized_format(number)
    else:
        # If it's a single number, just format it
        return format_single(number)

def set_seed(seed) -> int:
    """ Sets random seed for numpy & Python's random module.
        If hyperparameters has seed value uses it for repeatabilty.
        IF not, generates randomly
    """
    if seed == 0:
        seed = random.randint(1, 999999)
    np.random.seed(seed)
    random.seed(seed)
    print(f"🛠️ Using Random Seed: {seed}")
    return  seed



def draw_gradient_rect( surface, rect, color1, color2):
    #print(f"rect.height={rect.height}")
    safe_height = min(rect.height, 1500)  # Clamp height to prevent hanging if height explodes. 2E31 lines drawn
    for i in range(safe_height):
        ratio = i / safe_height
        blended_color = [
            int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
        ]
        pygame.draw.line(surface, blended_color, (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))
import os
import importlib
import inspect

def dynamic_instantiate(class_name, base_path='arenas', *args):
    """
    Dynamically instantiate an object of any class inheriting from BaseArena
    or BaseGladiator in the specified file, avoiding class name mismatches.

    Args:
        class_name (str): The name of the file to search within (file must end in .py).
        base_path (str): The base module path to search within.
        *args: Arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.

    Raises:
        ImportError: If the file or class is not found.
        ValueError: If the same file is found in multiple subdirectories or no matching class found.
    """
    # Set up the directory to search
    search_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_path.replace('.', os.sep))
    matched_module = None
    matched_file   = None   # For returing the code of the arena.

    # Walk through directories to find the file
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file == f"{class_name}.py":
                # Calculate relative path from src folder and clean it up
                relative_path = os.path.relpath(root, os.path.dirname(os.path.dirname(__file__)))
                # Clean up extra ".." and slashes for importlib
                relative_path = relative_path.strip(os.sep).replace(os.sep, '.')
                module_path = f"{relative_path}.{class_name}"

                # Debugging output to verify paths
                #print(f"Found file: {file}")
                #print(f"Module path: {module_path}")

                # Check for duplicates
                if matched_module:
                    raise ValueError(f"Duplicate module found for {class_name}: {matched_module} and {module_path}")

                # Set matched module path
                matched_module = module_path
                matched_file   = os.path.join(root, file)

    if not matched_module:
        raise ImportError(f"Module {class_name} not found in {base_path} or any subdirectories.")

    # Import module and instantiate class
    module = importlib.import_module(matched_module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (issubclass(obj, BaseReport) or issubclass(obj, BaseArena) or issubclass(obj, Gladiator)) and obj.__module__ == module.__name__:
            #return obj(*args)
            instance = obj(*args)
            # if it's an Arena, load its source as well
            if issubclass(obj, BaseArena) and matched_file:
                try:
                    with open(matched_file, 'r', encoding='utf-8') as f:
                        instance.source_code = f.read()
                except IOError:
                    instance.source_code = None
            return instance

    raise ImportError(f"No class inheriting from BaseArena or BaseGladiator found in {class_name}.py")

def clean_multiline_string(input_string):
    # Split the input string into lines
    lines = input_string.splitlines()

    # Strip leading and trailing spaces from each line
    cleaned_lines = [line.strip() for line in lines]

    # Join the cleaned lines back together with newlines
    cleaned_string = '\n'.join(cleaned_lines)

    return cleaned_string


import time

class PlaybackController:
    def __init__(self):
        """Initialize a dictionary to store playback states for different keys."""
        self.playback_states = {}  # {key: {"rate": int, "last_update": float}}

    def move_tape(self, key: str, rate: int, step_function):
        """
        Controls playback speed and frame advancement for any process.

        Args:
            key (str): Unique identifier for the process being controlled.
            rate (int): Playback speed (e.g., 3 = forward 3 FPS, -2 = reverse 2 FPS, 0 = paused).
            step_function (callable): Function to execute each step.
        """

        if rate == 0:
            return  # No movement if playback is paused

        # Ensure we have a state record for this key
        if key not in self.playback_states:
            self.playback_states[key] = {"rate": rate, "last_update": time.monotonic()}

        state = self.playback_states[key]

        # If the rate has changed, update it
        if state["rate"] != rate:
            state["rate"] = rate

        # Time-based stepping logic
        seconds_per_frame = 1.0 / abs(rate)  # Convert FPS to step interval
        current_time = time.monotonic()

        if current_time - state["last_update"] >= seconds_per_frame:
            step_function(1 if rate > 0 else -1)  # Execute step in the correct direction
            state["last_update"] = current_time  # Update time record

            print(f"[{key}] Moved at {rate} FPS")  # Debug output

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

import inspect
import ast

import inspect
import ast
import sys

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

# Example usage:
if __name__ == "__main__":
    a = 1
    b = 2
    c = 3
    ez_debug(a=a, b=b, c=c)

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



def get_absolute_position(surface, local_x, local_y):
    """Converts a coordinate from a surface's local space to absolute screen space."""
    abs_x = surface.get_abs_x() + local_x
    abs_y = surface.get_abs_y() + local_y
    return abs_x, abs_y


def clean_value(value):
    """Ensure values are correctly formatted for SQLite storage."""
    if isinstance(value, (int, float, np.float64, np.int64)):
        return float(value)  # Convert to standard float
    elif value == 0:
        return 0.0  # Explicitly ensure zero is stored correctly
    elif value is None or value == "":
        return None  # Ensure NULL values are handled properly
    return value  # Return unchanged for text or other valid values
