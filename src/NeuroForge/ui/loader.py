import os
import importlib.util
import inspect
from src.NNA.engine.BaseGladiator import Gladiator  # or wherever your base class is

def load_gladiators(directory):
    gladiators = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                full_path = os.path.join(root, file)

                module_name = os.path.splitext(os.path.basename(file))[0]
                spec = importlib.util.spec_from_file_location(module_name, full_path)
                module = importlib.util.module_from_spec(spec)

                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"⚠️ Failed to import {module_name}: {e}")
                    continue

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Gladiator) and obj is not Gladiator:
                        doc = inspect.getdoc(obj)
                        preview = doc.splitlines()[0] if doc else "(No description)"
                        gladiators.append((name, preview))

    return gladiators
