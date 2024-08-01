# Package-wide variables
__version__ = "1.0.0"
CONFIG = {
    "setting1": "value1",
    "setting2": "value2",
}


# Initialization logic
def initialize_package():
    global CONFIG
    # Modify the CONFIG dictionary as needed during initialization
    CONFIG["initialized"] = True
    print("Package initialized")


# Automatically run the initialization logic when the package is imported
initialize_package()