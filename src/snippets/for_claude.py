# Claude, below is the simplest example of the Simpletron, a NN model we've designed for the purpose of education and comparing NN techniques.
# Another class, Arena instantiates the below along with any other versions (as long as it inherits from Gladiator).   Generates
# training data (function for that is below the Simpletron class).  Trains the model with the identical data and that reports things like accuarcy, precision, convergence, etc.
# Note, the training data is normally run with anomalies (which i define as the outcome that is not statistically most likely)  this is
# because if there are no anomalies it will get 100% correct, so there isn't much to learn there.



############################################################
# Model Parameters are set here as global variables.       #
############################################################
neuron_weight   = 10        # Any value works as the training data will adjust it
learning_rate   = .001       # Reduces impact of adjustment to avoid overshooting


class Simpletron(Gladiator):

    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics

    def train(self, training_data):
        global neuron_weight
        for epoch in range(self.number_of_epochs):
            neuron_weight = self.run_a_epoch(training_data, neuron_weight, epoch)

    def run_a_epoch(self, train_data, weight, epoch_num):
        for i, (credit_score, result) in enumerate(train_data):
            weight = self.training_iteration(i, epoch_num, credit_score, result, weight)
        self.metrics.record_epoch()
        return weight

    def training_iteration(self, i, epoch, credit_score, result, weight):
        prediction  = self.predict(credit_score, weight)
        loss        = self.compare(prediction, result)
        adjustment  = self.adjust_weight(loss, credit_score, learning_rate)
        new_weight  = weight + adjustment
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, weight, new_weight, self.metrics)
        return new_weight

    def predict(self, credit_score, weight):
        #return 1 if credit_score * weight >= 0.5 else 0
        #product = credit_score * weight
        #result = 1 if product >= 0.5 else 0
        #print(f"Credit Score: {credit_score}, Weight: {weight}, Product: {product}, Result: {result}")
        #return result
        return 1 if round(credit_score * weight, 7) >= 0.5 else 0  # NOTE: Credit score of 50 was incorrectly  predicting "no pay" due to fp precision.. Credit Score: 50, Weight: 0.009999999999999842, Product: 0.4999999999999921, Result: 0

    def compare(self, prediction, result):
        return result - prediction  # This remains the same

    def adjust_weight(self, loss, score, learning_rate):
        return loss * learning_rate

# Claude, the below function is in a different class but it is used to generate training data to be sent to all the models being compared
# it first calculates a credit score between 0-100.  If include_anomolies is false and the credit is 50 or greater the output is 1 (repayment)
# if include_anomalies is true, it uses the credit score as the percent chance the loan was repaid
# for example a score of 90 would normally repay, but there is a 10% chance it will not.

def generate_random_linear_data(include_anomalies):
    training_data = []
    for _ in range(qty_rand_data):
        score = random.randint(1, 100)
        if include_anomalies:
            second_number = 1 if random.random() < (score / 100) else 0
        else:
            second_number = 1 if score >=50 else 0
        training_data.append((score, second_number))
    return training_data