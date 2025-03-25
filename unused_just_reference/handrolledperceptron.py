import random

# Arena ground truth
BASE_SALARY = 14000
COEFF_EXP   = 12000
COEFF_COL   =8000

# Generate raw training data
def generate_data(n_samples=100):
    data = []
    for _ in range(n_samples):
        exp = random.uniform(0, 40)
        col = random.uniform(0, 8)
        noise = 0  # No noise for now
        salary = BASE_SALARY + COEFF_EXP * exp + COEFF_COL * col + noise
        data.append((exp, col, salary))
    return data

# Training loop
def train_perceptron(data, epochs=1000, lr=1e-6):
    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    b  = random.uniform(-1, 1)

    for epoch in range(epochs):
        total_error = 0
        for x1, x2, y in data:
            # Forward pass
            y_hat = w1 * x1 + w2 * x2 + b
            error = y_hat - y
            total_error += abs(error)

            # Gradient Descent (manual)
            w1 -= lr * error * x1
            w2 -= lr * error * x2
            b  -= lr * error

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: MAE = {total_error / len(data):,.2f}\tw1 = {w1}\tw2 = {w2}\tb = {b}")

    print(f"\nLearned weights:")
    print(f"  w1 (Exp):     {w1:.2f}\tvs {COEFF_EXP}\t(delta {w1-COEFF_EXP})")
    print(f"  w2 (College): {w2:.2f}\t\tvs {COEFF_COL}\t\t(delta {w2-COEFF_COL})")
    print(f"  Bias:         {b:.2f}\t\tvs {BASE_SALARY}\t(delta {b-BASE_SALARY})")

# Run test
if __name__ == "__main__":
    data = generate_data(n_samples=5)
    train_perceptron(data, lr=.001)
