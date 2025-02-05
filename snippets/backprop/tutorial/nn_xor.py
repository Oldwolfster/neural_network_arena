from src.backprop.tutorial.neuralnetwork2 import NeuralNetwork
import numpy as np
def run_tutorial():
    # import the necessary packages

    # construct the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    print("yo")
    nn = NeuralNetwork([2, 2, 1], alpha=.5)
    print("BEFORE WEIGHTS:")
    nn.print_weights()
    nn.fit(X, y, epochs=2)
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


run_tutorial()
"""

       
3 hidden 
array([[ 311, 321, 331, 341],
       [ 312, 322, 332, 342]
       [ 313, 323, 433, 343 ]])
       , array([[ 0.45463082],       [-0.34625484],       [-0.34208892],       [ 0.13332949]])]
       
2 hidden
[array([[ 211, 221,  231],
       [ 212, 222, 232]
       [ 213, 223, 233 ]])
       , array([[ 0.29197288],       [-0.26514266],       [-0.31561029]])]
       
1 hidden  
[array([[111, 121],
       [112, 122],
       [113, 123]]) 
,array([[ 0.46236008],     [-0.27699637]])] 

0 hidden - error makes sense but the pattern would have one...
Weights:[array([[nan],
       [nan],
       [nan]]), array([[nan]])] 

the first array always contains 3 rows, 
but the length of the row varies based on number of hidden neurons specified 
so i believe that proves the columns correspond to Inp1, inp2, and bias
It seems though the length is always one to long... i.e. it's adding an extra neuron

The 2nd array has an element for each hidden neuron(including the extra one)

"""