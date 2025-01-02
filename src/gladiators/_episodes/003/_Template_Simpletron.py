from src.engine.BaseGladiator import Gladiator


class _Template_Simpletron(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def training_iteration(self, training_data) -> float:
        input = training_data[0]
        target = training_data[1]
        prediction  = input * self.weights[0]                       # Step 1) Guess
        prediction = 1 if round(prediction,7) >= 0.5 else 0     #         Step Function to make it 0 or 1
        error        = target - prediction                      # Step 2) Check guess,
        adjustment  = error * self.learning_rate                #         if wrong, how much
        new_weight  = self.weights[0] + adjustment                  # Step 3) Adjust(Apply)


        self.weights[0] = new_weight
        print(f"input={input}\ttarget={target}\tprediction={prediction}\terror={error}\tadjustment={adjustment}\tnew_weight={new_weight}\tself.weights[0]={self.weights[0]}\t")
        return prediction


