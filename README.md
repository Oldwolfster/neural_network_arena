# Neural Network Arena (NNA) - README

---

## 🌍 Overview

Neural Network Arena (NNA) is a framework for building, training, and **comparing neural network models** ("Gladiators") against controlled **data generators** ("Arenas").

The goal: **Create a fully autonomous system that builds the right model** with **zero manual tuning** if desired, while still allowing full control for experts.

At every step, the philosophy is: **Simplify > Understand > Expand**.

---

## 🔄 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neural_network_arena.git
cd neural_network_arena
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Make sure your Python environment is activated and correct (Python 3.12+ recommended).

### 3. Run the Application
```bash
python src/main.py
```

---

## 🔹 Main Components

### 1. Gladiators (Models)
- **Simple to Complex Neural Networks**.
- Each inherits from a base `Gladiator` class.
- Configurations can be:
  - **Fully Manual**
  - **Fully Automatic** via the "LegoSelector" system.

### 2. Arenas (Training Data)
- Algorithms that generate **reproducible training sets**.
- Types include:
  - Linear Regression
  - Piecewise Regression
  - Binary Decision Problems (Classification)
  - Nonlinear Problems
- Consistent data ensures fair model comparisons.

### 3. Engine
- Orchestrates:
  - Data generation
  - Gladiator initialization
  - Training
  - Result collection
- Produces detailed reports:
  - **Iteration View**: Inputs, Targets, Predictions, Weights
  - **Epoch Summary**
  - **Final Run Summary**

### 4. NeuroForge (Visualizer)
- Play back training like a **VCR**.
- Step through epochs and iterations.
- See models learn in real time.
- Show details of Neurons including weights, bias, and activation

### 5. LegoSelector (Autonomous Configurator)
- **Dynamic rules engine** that sets hyperparameters automatically.
- Based on:
  - Problem type (regression vs classification)
  - Output activation functions
  - Input feature count
  - Defaults when uncertain
- Ensures that even a **blank config** produces a usable model.

---

## 🔹 Project Philosophy

- **Simplify Neural Networks**: Understand every piece by stripping away unnecessary complexity.
- **Transparent Testing**: All models face the exact same data under identical conditions.
- **Skeptical of Gradient Descent (GBS)**: Gradient-based methods are treated critically and tested rather than assumed to be optimal.
- **Prefer Manual First, Automate Second**: If a rule improves stability, add it; if not, remove it.
- **Zero Hidden Magic**: Every piece is visible, understandable, and modifiable.

---

## 🔹 Current Status

- Core systems (Engine, Gladiators, Arenas, LegoSelector) are functional.
- Blank configurations now succeed in most cases.
- Arena-based testing is starting to expose where additional rules or strategies are needed.

---

## 💪 Future Directions

- Expand Arena coverage:
  - Add noisy regression
  - Add MNIST-style small image tasks
- Build smarter LegoSelector rules based on observed failures.
- Incorporate lightweight batch-based hyperparameter exploration.
- Continue documenting the learning process via YouTube tutorials.

---

## 🎥 YouTube Channel

Follow the journey as each component is explored and explained step-by-step.
Focus on true **learning through simplification**.

---

## ✨ License and Contributions

Open to suggestions, contributions, collaborations, and new ideas.
Let's make it the simplest, most intuitive NN Arena ever built.

---

## 💭 Final Note

NNA isn't about building the most complex neural networks.
It's about building **the clearest mental model**.
From there, anything is possible.

---

Ready to enter the Arena? 🏀

