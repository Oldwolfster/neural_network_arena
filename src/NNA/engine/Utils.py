import random
import numpy as np
from src.reports._BaseReport import BaseReport
from src.NNA.engine.BaseArena import BaseArena
from src.NNA.engine.BaseGladiator import Gladiator

def ensure_src_in_path():
    """Ensure the 'src' directory is in sys.path once per process."""
    import sys, os
    this_file = os.path.abspath(__file__)
    src_root = os.path.abspath(os.path.join(this_file, "..", "..", ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def format_percent(x: float, decimals: int = 2) -> str:
        """
        Format a fraction x (e.g. 0.9999) as a percentage string:
          • two decimal places normally → "99.99%"
          • no trailing .00 → "100%"
        """
        # 1) turn into a fixed-decimal string, e.g. "100.00" or " 99.99"
        s = f"{x * 100:.{decimals}f}"
        # 2) drop any trailing zeros and then a trailing dot
        s = s.rstrip("0").rstrip(".")
        # 3) tack on the percent sign
        return s + "%"





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
    return  seed




import os
import importlib


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
    #search_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_path.replace('.', os.sep))
    ensure_src_in_path()
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search_directory = os.path.join(src_root, base_path.replace('.', os.sep))
    search_directory = os.path.abspath(search_directory)

    matched_module = None
    matched_file   = None   # For returing the code of the arena.

    # Walk through directories to find the file
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file == f"{class_name}.py":
                # Calculate relative path from src folder and clean it up
                #relative_path = os.path.relpath(root, os.path.dirname(os.path.dirname(__file__)))
                relative_path = os.path.relpath(root, src_root)

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




import inspect

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
