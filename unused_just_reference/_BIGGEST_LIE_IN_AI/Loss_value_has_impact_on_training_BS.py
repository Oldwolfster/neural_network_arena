import tensorflow as tf
import numpy as np
import random

# 🔹 Ensure repeatability
SEED = 66
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 🔹 Generate synthetic regression dataset
X_train = np.random.rand(1000, 1) * 10  # Inputs: Random values between 0-10
y_train = 3 * X_train + 5 + np.random.randn(1000, 1) * 2  # Linear function + noise

# 🔹 Create identical models
def build_model(loss_function, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=None, input_shape=(1,))
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss=loss_function)
    return model

base_lr = 0.001

# 🔹 Model A (MSE, LR=0.001)
model_a = build_model(loss_function="mse", learning_rate=base_lr)

# 🔹 Model B (MAE, LR=0.002)
model_b = build_model(loss_function="mae", learning_rate=base_lr * 2)

# 🔹 Get initial weights and bias before training
weights_a_init, bias_a_init = model_a.layers[0].get_weights()
weights_b_init, bias_b_init = model_b.layers[0].get_weights()

print("\n🔹 Initial Weights and Biases BEFORE Training:")
print(f"Model A (MSE) - Initial Weight: {weights_a_init.flatten()[0]:.4f}, Initial Bias: {bias_a_init[0]:.4f}")
print(f"Model B (MAE) - Initial Weight: {weights_b_init.flatten()[0]:.4f}, Initial Bias: {bias_b_init[0]:.4f}")

# 🔹 Custom callback to log weight updates
class WeightLogger(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        self.model_name = model_name
        self.weights_log = []
        self.bias_log = []
        self.loss_log = []

    def on_epoch_end(self, epoch, logs=None):
        weights, bias = self.model.layers[0].get_weights()
        self.weights_log.append(weights[0][0])
        self.bias_log.append(bias[0])
        self.loss_log.append(logs["loss"])

        # Log only key epochs to reduce clutter
        if epoch in [0, 9, 49, 99]:  # Log at Epoch 1, 10, 50, 100
            print(f"{self.model_name} - Epoch {epoch+1}: Loss={logs['loss']:.4f}, Weight={weights[0][0]:.4f}, Bias={bias[0]:.4f}")

# 🔹 Train models with logging
logger_a = WeightLogger("MSE")
logger_b = WeightLogger("MAE")

history_a = model_a.fit(X_train, y_train, epochs=100, verbose=2, callbacks=[logger_a])
history_b = model_b.fit(X_train, y_train, epochs=100, verbose=2, callbacks=[logger_b])

# 🔹 Compare final loss values
final_loss_a = history_a.history['loss'][-1]
final_loss_b = history_b.history['loss'][-1]

print(f"\nFinal Loss (MSE, LR={base_lr}): {final_loss_a:.4f}")
print(f"Final Loss (MAE, LR={base_lr * 2}): {final_loss_b:.4f}")

# 🔹 Compare final weights & biases
weights_a, bias_a = model_a.layers[0].get_weights()
weights_b, bias_b = model_b.layers[0].get_weights()

print(f"\nFinal Weights and Biases AFTER Training:")
print(f"Model A (MSE) - Final Weight: {weights_a.flatten()[0]:.4f}, Final Bias: {bias_a[0]:.4f}")
print(f"Model B (MAE) - Final Weight: {weights_b.flatten()[0]:.4f}, Final Bias: {bias_b[0]:.4f}")

# 🔹 Print initial 10 weight updates for both models
print("\n🔹 First 10 Weight Updates:")
for i in range(10):
    print(f"Epoch {i+1}: MSE Weight={logger_a.weights_log[i]:.4f}, MAE Weight={logger_b.weights_log[i]:.4f}")

print("\n🔹 First 10 Bias Updates:")
for i in range(10):
    print(f"Epoch {i+1}: MSE Bias={logger_a.bias_log[i]:.4f}, MAE Bias={logger_b.bias_log[i]:.4f}")
