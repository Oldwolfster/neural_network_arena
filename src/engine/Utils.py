
import random

import numpy as np


import pygame

from src.reports._BaseReport import BaseReport
from src.engine.BaseArena import BaseArena
from src.engine.BaseGladiator import Gladiator


def chunk_list(lst: list, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]




def print_call_stack():
    stack = inspect.stack()
    for frame in stack:
        print(f"Function: {frame.function}, Line: {frame.lineno}, File: {frame.filename}")

def smart_format(num):
    if num == 0:
        return "0"
    elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
        return f"{num:.2e}"
    elif abs(num) < 0.001:  # Use 6 decimal places for small numbers
        formatted = f"{num:,.6f}"
    elif abs(num) < 1:  # Use 3 decimal places for numbers less than 1
        formatted = f"{num:,.3f}"
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
    for i in range(rect.height):
        ratio = i / rect.height
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

    if not matched_module:
        raise ImportError(f"Module {class_name} not found in {base_path} or any subdirectories.")

    # Import module and instantiate class
    module = importlib.import_module(matched_module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (issubclass(obj, BaseReport) or issubclass(obj, BaseArena) or issubclass(obj, Gladiator)) and obj.__module__ == module.__name__:
            return obj(*args)

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

    """
    # Example usage:
    if __name__ == "__main__":
        # Example background colors:
        examples = [
            (255, 255, 255),  # white background -> should use black text
            (0, 0, 0),        # black background -> should use white text
            (100, 150, 200)   # medium background -> decision based on brightness
        ]
        
        for bg in examples:
            text_color = get_contrasting_text_color(bg)
            print(f"Background color {bg} -> Contrasting text color {text_color}")
    """
