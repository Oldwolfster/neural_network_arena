import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Provided data
data = np.array([
    [0.6865318267106588, 0.0, 283560.5205867383],
    [0.08045030305379103, 0.15156160325413248, 65446.06245354115],
    [0.9872870357222059, 0.813036153930387, 954895.3777621521],
    [0.22681795645179026, 0.9969918029932361, 305971.05672009196],
    [0.11048188935340655, 0.944093176495698, 182197.85502807715],
    [0.7641447422034934, 1.0, 852529.9758417873],
    [0.0, 0.1883824785879859, 29626.868897424665],
    [0.9045877015361623, 0.8195280783707949, 883570.9632620066],
    [1.0, 0.3169444055456317, 626347.3020094554],
    [0.8933458322619753, 0.06101524695002256, 403800.31872490206],
    [0.9326379264622002, 0.3320099687160382, 595497.8776557705],
    [0.8281550841805015, 0.6977467842642556, 743340.5700267805]
])

# Separate input and output
X = data[:, :2]
y = data[:, 2]

# Normalize input for stability
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Normalize output for better training performance
y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std

# Define a strong architecture using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mae'])

# Train the model
history = model.fit(X_norm, y_norm, epochs=500, verbose=0)

# Denormalize predictions for interpretability
predictions = model.predict(X_norm).flatten() * y_std + y_mean
true_vals = y

# Store loss history for visualization
loss = history.history['loss']
mae = history.history['mae']

# Prepare comparison output
comparison = list(zip(true_vals, predictions))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from src.ace_tools import display_dataframe_to_user

df = pd.DataFrame(comparison, columns=["True Salary", "Predicted Salary"])
df["Error"] = df["Predicted Salary"] - df["True Salary"]
df["Absolute Error"] = df["Error"].abs()
print(df.to_string())
# Show to user
#display_dataframe_to_user("TensorFlow Regression Results", df)

# Return model performance metrics for review
final_mae = mae[-1]
final_loss = loss[-1]
final_mae, final_loss
