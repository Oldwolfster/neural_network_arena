from src.backprop.tutorial.neuralnetwork import NeuralNetwork
import numpy as np
def run_tutorial():
    # import the necessary packages

    # construct the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    print("yo")
    nn = NeuralNetwork([2, 2, 1], alpha=2)
    print("BEFORE WEIGHTS:")
    nn.print_weights()
    nn.fit(X, y, epochs=2222)
    print("After WEIGHTS:")
    nn.print_weights()
    # now that our network is trained, loop over the XOR data points
    for (x, target) in zip(X, y):
        # make a prediction on the data point and display the result
        # to our console
        pred = nn.predict(x)[0][0]

        step = 1 if pred > 0.5 else 0
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}, result={}".format(
            x, target[0], pred, step, "Correct" if target[0] == step else "Not even close"))

