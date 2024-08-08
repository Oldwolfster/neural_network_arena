# Claude, below is the simplest example of the Simpletron, a NN model we've designed for the purpose of education and comparing NN techniques.
# Note:  The template is meant to be as bare bones as possible, so we deliberately do not want things such as bias or activation functions.  This will allow us to compare with and without them.
# Another class, Arena instantiates the below along with any other versions (as long as it inherits from Gladiator).   Generates
# training data (function for that is below the Simpletron class).  Trains the model with the identical data and that reports things like accuarcy, precision, convergence, etc.
# Note, the training data is normally run with anomalies (which i define as the outcome that is not statistically most likely)  this is
# because if there are no anomalies it will get 100% correct, so there isn't much to learn there.

class _Template_Simpletron(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    In order to utilize the metrics there are two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args, **kwargs):
        super().__init__(number_of_epochs, metrics, *args, **kwargs)
        # Ideally avoid overriding these, but specific models may need, so must be free to do so
        # It keeps comparisons straight if respected
        # self.weight = override_weight
        # self.learning_rate = override_learning_rate

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):
            if self.run_an_epoch(training_data, epoch):
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, epoch_num, credit_score, result)
        return self.metrics.record_epoch()

    def training_iteration(self, i: int, epoch: int, credit_score: float, result: int) -> None:
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        adjustment = self.adjust_weight(loss)
        new_weight = self.weight + adjustment
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        self.weight = new_weight

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight, 7) >= 0.5 else 0

    def compare(self, prediction: int, result: int) -> float:
        return result - prediction

    def adjust_weight(self, loss: float) -> float:
        return loss * self.learning_rate

def run_a_match(gladiators, arena):
    metrics_list = []
    for gladiator in gladiators:    # Loop through the NNs competing.
        metrics = Metrics(gladiator)  # Create a new Metrics instance with the name as a string
        metrics_list.append(metrics)
        nn = dynamic_instantiate(gladiator, 'gladiators', epochs_to_run, metrics, default_neuron_weight, default_learning_rate)
        start_time = time.time()  # Start timing
        nn.train(arena)
        end_time = time.time()  # End timing
        metrics.run_time = end_time - start_time

    print_results(metrics_list, arena, display_graphs)


def dynamic_instantiate(class_name, path, *args):
    """
    Dynamically instantiate object without needing an include statement

    Args:
        class_name (str): The name of the file AND class to instantiate.
                            THE NAMES MUST MATCH EXACTLY
        path (str): The path to the module containing the class.
        *args: Arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.
    """
    module_path = f'{path}.{class_name}'
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    return class_(*args)


def generate_random_linear_data(include_anomalies):
    """
    it first calculates a credit score between 0-100.  If include_anomolies is false and the credit is 50 or greater the output is 1 (repayment)
    if include_anomalies is true, it uses the credit score as the percent chance the loan was repaid
    for example a score of 90 would normally repay, but there is a 10% chance it will not.
    """
    training_data = []
    for _ in range(qty_rand_data):
        score = random.randint(1, 100)
        if include_anomalies:
            second_number = 1 if random.random() < (score / 100) else 0
        else:
            second_number = 1 if score >=50 else 0
        training_data.append((score, second_number))
    return training_data

