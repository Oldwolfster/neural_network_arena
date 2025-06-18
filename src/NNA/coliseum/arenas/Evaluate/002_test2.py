import random
import math
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Assuming your existing test data generator is available.
# For completeness, here is your provided generator class:
class Predict_Income_2_Inputs__HighlyNonlinear:
    """
    Very challenging nonlinear regression task that includes squared terms,
    interaction terms, and sine waves to introduce oscillatory behavior.

    This will definitely require at least 2 hidden layers to learn properly.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float, float]], list]:
        training_data = []
        for _ in range(self.num_samples):
            years_exp = random.uniform(0, 40)
            college = random.uniform(0, 8)

            base_salary = 15000
            coeff_exp = 9000
            coeff_col = 7000
            coeff_sq_exp = 15
            coeff_interact = 1200
            coeff_sin_exp = 3000  # causes local minima
            coeff_sin_col = 1000  # causes local minima

            noise = random.gauss(0, 0)  # Disable noise for now

            salary = (
                base_salary
                + coeff_exp * years_exp
                + coeff_col * college
                + coeff_sq_exp * (years_exp ** 2)
                + coeff_interact * years_exp * college
                + coeff_sin_exp * math.sin(years_exp / 2.0)
                + coeff_sin_col * math.sin(college * 1.5)
                + noise
            )
            training_data.append((years_exp, college, salary))

        return training_data, ["Years on Job", "Years College", "Salary"]

# Helper function to generate data
def generate_data(num_samples: int = 1000):
    generator = Predict_Income_2_Inputs__HighlyNonlinear(num_samples)
    data, feature_names = generator.generate_training_data()
    data = np.array(data)
    X = data[:, :2]  # Two input features: years_exp and college
    y = data[:, 2]   # Output: salary
    return X, y, feature_names

# Helper function to build the baseline model
def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Main function to train and evaluate the model
def train_and_evaluate(num_samples: int = 1000, epochs: int = 100, batch_size: int = 32):
    X, y, feature_names = generate_data(num_samples)
    model = build_model((X.shape[1],))
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, verbose=1)

    # Plot training and validation loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Evaluate model performance on the full dataset
    loss, mae = model.evaluate(X, y, verbose=0)
    print(f"Final training loss: {loss:.3f}, MAE: {mae:.3f}")
    return model, history

if __name__ == '__main__':
    # Run training with the baseline model
    model, history = train_and_evaluate(num_samples=1000, epochs=100, batch_size=32)
