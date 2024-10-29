# Quick and dirty place to put notes regarding testing.
**10/1/2024**
Regression_RegularizationChatGPT
In Arenas, CreditScoreRegressionNeedsBias and CreditScoreRegression it performs comparable, but when switching to SalaryExperienceRegressionNeedsBias it shits the bed
**10/3/2024**
For CreditScoreRegression
BlackBird (no bias) does best
Hayabusa_MSE_ close if bias starts at 0, if bias is 5 it's bad.
Regression_Bias_ChatGPT2 Good 

For CreditScoreRegressionNeedsBias
BB is bad
Busa good if bias init=.5 bad if init =0
Regression_Bias_ChatGPT2 good either way

**10/28/2024**
    Refactored to store and allow rerun of training data
    * What about a metric comparing total error as percent of Sum of targets
      * Original input had college as multiplicative which made it non linear..
      * FOIL
      * Capturing Non-Linearity:
      By including the interaction term, we transform the problem into a linear one in terms of the inputs.
      Perceptron Limitations:
      A perceptron without activation functions can only model linear relationships between inputs and outputs.
      Feature Engineering:
      Helps to extend the capabilities of linear models by transforming the input space.
      n your perceptron model, the bias term corresponds to the base salary.
      The bias represents the expected salary when both years of experience and college years are zero (assuming noise is zero).
      Therefore, the optimum bias learned by the perceptron should approximate the base salary value used in the data generation (e.g., $30,000).

    Adding Noise to the Target Variable vs Inputs
        Adding Noise to the Target Variable (Salary):
        
        Simulates measurement errors or inherent variability in the salary.
        The model learns to predict the noisy target, which can make learning more challenging but reflects real-world scenarios.
        The underlying relationship between inputs and outputs remains the same.
        Adding Noise to the Input Features:
        
        Simulates measurement errors in the input data (years of experience and college).
        Alters the input data distribution and can affect the model's ability to learn the true relationship.
        May introduce additional variability that the model needs to account for.

        ** .001 ran perfect but 500 epochs,  .001 exploding gradient
          * Binary search
          * Feature Scaling