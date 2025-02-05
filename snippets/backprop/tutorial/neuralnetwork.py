import numpy as np
#From https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers    # [2, 2, 1] would imply that our first input layer has two nodes, our hidden layer has two nodes, and our final output layer has one node.
        self.alpha = alpha
        seed = 42
        np.random.seed(seed)

        #Loop through layers but stop before reaching last two layers
        for i in np.arange(0, len(layers)-2):   #arange causes it to return a numpy array
            #randomly initialize a weight matrix connecting the number of nodes
            #in each respective layer together
            #adding an extra node for the bias
            w = np.random.randn(layers[i]+1,layers[i+1]+1)
            self.W.append(w / np.sqrt(layers[i]))  #sqrt to normalize variance

            #last two layers are special case where where the input connections
            #need a bias term but the output does not.
            w = np.random.randn(layers[-2]+1,layers[-1])
            self.W.append(w/np.sqrt(layers[-2]))

    def print_weights(self):
        print (f"Weights:{self.W} ")
    def __repr__(self): #magic method for debugging???
        #construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x): # compute sigmoid
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self,x):
        #compute the deriv of sigmoid assuming x has already gone through
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate = 10):
        # insert a column of 1s as the last entry in the feature matrix
        # this allows us to treat the bias as a trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):  #loop over epochs
            for (x, target) in zip (X, y):      #loop through samples
                #if epoch < 1000000:
                    #print(f"epoch {epoch}\t inputs{x}")
                self.fit_partial(x, target)

            #check for update display
            if epoch == 0 or (epoch + 1) % displayUpdate ==0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch = {}, loss = {:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        # construct our list of output activations for each layer
        # as our data point flows through the network
        # activation is a special case, it's just the input feature vector itself.
        A = [np.atleast_2d(x)] #responsible for storing output activations

        #FEED FORWARD
        #loop over the layers of the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer
            # by taking the dot product between the activation
            # and the weight matrix - this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])

            #net output is simply applying sigmoid
            out = self.sigmoid(net)

            #once we have the net output, add it t oour list of activations
            A.append(out)


            #BACKPROP:
            # The first phase of backprop is to compute error
            # (the final output activation in the activations list) and the true target value
            error = A[-1] -y

            #from here, we need to apply chain rule and build list of deltas'D'
            #first entry in deltas is simply the eorror of the output layer times derivated of activation function
            D = [error* self.sigmoid_deriv(A[-1])]

            # start applying the chain rule to build our list of deltas

            #once you understand the chain rule it becomes super easy to implement with a for
            #just loop over layers in reverse order(ignoring the last two)
            for layer in np.arange(len(A) - 2, 0, -1):
                # the delta for the current layer is equal to the delta
                # of the *previous layer# dotted with the weight matrix
                # of the current layer, follwed by multiplying the delta
                # by the derivative of the nonlinear activation function for the activations of the current layer

                #we are simply taking the delta from the previous layer, dotting it with the weights of the current layer, and then multiplying by the derivative of the activation. This process is repeated until we reach the first layer in the network.
                delta = D[-1].dot(self.W[layer].T)
                delta = delta * self.sigmoid_deriv(A[layer])
                D.append(delta)

                # since we looped over layers in revers order we need to reverse the deltas
                D = D[::-1]

                #WEIGHT UPDATE PHASE
                # Loop over layers
                for layer in np.arange(0, len (self.W)):
                    #update weights by taking the dot product of the layer
                    # with their respective deltas, then multiplying
                    # this value by some small learning rate and adding to our
                    # weight matrix
                    self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
                                # initialize the output prediction as the input features -- this
                                # value will be (forward) propagated through the network to
                                # obtain the final prediction
        p = np.atleast_2d(X)
        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature
            # matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]
        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking
            # the dot product between the current activation value 'p'
            # and the weight matrix associated with the current layer,
            # then passing this value through a nonlinear activation
            # function
            p = self.sigmoid(np.dot(p, self.W[layer]))
        # return the predicted value
        return p

    def calculate_loss(self, X, targets): # make predictions for the input data points then compute then loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss # return the loss













