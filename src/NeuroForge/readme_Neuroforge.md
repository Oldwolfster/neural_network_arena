# neuroForge Documentation

## Overview
The **neuroForge** module is the core visualization and management system for the Neural Network Arena (NNA). It handles the representation of models, neurons, connections, and inputs in a way that is both visually clear and dynamically adjustable. The goal is to make neural networks more intuitive to understand and debug by providing a flexible and modular structure for visualization.

### Key Features:
- **Visualization Framework:** Displays neurons, connections, inputs, and activations using Pygame.
- **Dynamic Adjustments:** Components like neuron size, spacing, and scaling adapt to the architecture.
- **Modular Design:** Highly decoupled classes for better organization and reusability.
- **Customization:** Easily adjustable to include additional data like gradients or predictions.

---

## Architecture
- Should we have a difference between models and components?
- DisplayInputs
  - render
  - update_me
  - draw_me **In Parent**
- DisplayModel
  - render
  - update_me
  - draw_me **In Parent**
- DisplayModel__Connection
  - draw_me
- DisplayModel__Neuron
  - draw_me
  - update_me

## File Structure and Responsibilities
The following sections outline the roles of each file in the `neuroForge` module:

### 1. **`Display_Mgr.py`**
- **Purpose:** Manages all display components and orchestrates rendering.
- **Key Responsibilities:**
  - Holds a list of all display objects (e.g., models, inputs, neurons).
  - Iterates through and invokes rendering for all components.
  - Provides an interface for initializing and updating visuals.
- **Key Methods:**
  - `initialize_components`: Initializes all components (models, inputs, etc.).
  - `render_all`: Loops through all components and calls their `draw_me` method.

---

### 2. **`DisplayConnection.py`**
- **Purpose:** Represents the connections between neurons in different layers.
- **Key Responsibilities:**
  - Stores references to the source and destination neurons.
  - Draws lines between neurons, optionally annotating weights or gradients.
- **Key Attributes:**
  - `from_neuron`: The source neuron object.
  - `to_neuron`: The destination neuron object.
  - `weight`: The weight value associated with the connection.
- **Key Methods:**
  - `draw_me`: Draws the connection line and renders the weight text at the midpoint.

---

### 3. **`DisplayInputs.py`**
- **Purpose:** Handles the display of input values for the network.
- **Key Responsibilities:**
  - Displays input values as squares (or another shape, like circles, if customized).
  - Updates input values dynamically.
- **Key Methods:**
  - `update_me`: Queries the database to fetch the latest input values.
  - `draw_me`: Renders input boxes and their values on the screen.

---

### 4. **`DisplayModel.py`**
- **Purpose:** Represents the entire neural network model.
- **Key Responsibilities:**
  - Organizes neurons and connections for a single model.
  - Handles the layout, rendering, and updates for neurons and connections.
- **Key Attributes:**
  - `neurons`: A nested list where each sublist contains neurons in a single layer.
  - `connections`: A list of `DisplayConnection` objects.
- **Key Methods:**
  - `initialize_with_model_info`: Sets up the neurons and connections based on the model architecture.
  - `render`: Draws neurons and connections onto its surface.

---

### 5. **`DisplayNeuron.py`**
- **Purpose:** Represents an individual neuron in the network.
- **Key Responsibilities:**
  - Displays neuron properties like activation and bias.
  - Updates its state dynamically based on model calculations.
- **Key Attributes:**
  - `activation`: The activation value of the neuron.
  - `bias`: The bias value for the neuron.
- **Key Methods:**
  - `update_me`: Updates activation, bias, and other properties using data from the database.
  - `draw_me`: Renders the neuron rectangle and associated text (e.g., activation value, bias).

---

### 6. **`EZPrint.py`**
- **Purpose:** Simplifies the rendering of multi-line text with boundaries and line spacing.
- **Key Responsibilities:**
  - Handles splitting text into lines.
  - Ensures text fits within specified dimensions.
- **Key Methods:**
  - `render`: Renders text line by line onto a given surface.

---

### 7. **`EZSurface.py`**
- **Purpose:** Provides a base class for managing rectangular surfaces.
- **Key Responsibilities:**
  - Acts as a wrapper for Pygame surfaces.
  - Handles positioning and scaling.
- **Key Attributes:**
  - `surface`: The Pygame surface to render onto.
  - `width_pct`/`height_pct`: Percentage dimensions relative to the screen.
- **Key Methods:**
  - `draw_me`: Draws the surface onto the screen.

---

### 8. **`mgr.py`**
- **Purpose:** Stores global variables and shared resources.
- **Key Responsibilities:**
  - Manages global screen and font objects.
  - Acts as a central hub for configuration.
- **Key Attributes:**
  - `screen`: The main Pygame screen.
  - `font`: Default font for text rendering.

---

### 9. **`neuroForge.py`**
- **Purpose:** The entry point for the neuroForge visualization system.
- **Key Responsibilities:**
  - Initializes Pygame.
  - Manages the main event loop.
  - Calls the `Display_Mgr` to render components.
- **Key Methods:**
  - `neuro_forge_init`: Sets up Pygame and initializes the `Display_Mgr`.
  - `run`: Main loop for rendering and handling user input.

---

### 10. **`ObjectPlacer.py`**
- **Purpose:** Handles positioning of neurons and connections based on the model architecture.
- **Key Responsibilities:**
  - Calculates neuron positions for each layer.
  - Generates connections between neurons in adjacent layers.
- **Key Methods:**
  - `calculate_neuron_positions`: Determines the positions of neurons.
  - `calculate_connections`: Creates `DisplayConnection` objects based on neuron positions.

---

## Example Workflow
### 1. Initialize neuroForge
- Call `neuro_forge_init` to initialize Pygame and set up the display.

### 2. Add Models and Inputs
- Use `DisplayModel` and `DisplayInputs` to define models and their inputs.

### 3. Render Components
- Call `render_all` in `Display_Mgr` during the main loop to draw all components on the screen.

### 4. Update Components
- Periodically update neurons, connections, and inputs by querying the database and calling `update_me` methods.

---

## Planned Enhancements
- **Dynamic Scaling:** Automatically adjust neuron size and spacing based on the screen dimensions and model complexity.
- **Toggle Views:** Allow users to switch between "Standard" and "Custom" visualizations.
- **Gradient Visualization:** Add gradient values to connections and neurons during backpropagation.

---

## Notes
This documentation provides a high-level overview. For specific implementation details, refer to the code in each module.

---

## Questions or Feedback?
Feel free to reach out with questions about this structure or suggestions for improvement!

## Thoughts on interface components for point and click.

Edit button brings up tools.
Add/remove layers.
add neurons to layers.
Select neurons for mass change.
activation function
loss function
