# Training Data Below
td = [(1, 1), (0, 0), (1, 1), (0, 0), (0, 1)]                # This data isn't exciting but makes it easy to get it working.

# Simpletron Neural Network
weight = .2                                         # Initial value of weight.
for epoch in range(30):                             # Run 5 Epochs (Full set of training data)
    for i, (credit, result) in enumerate(td):       # Loop through each set of training data
        predict = 1 if credit*weight >= 0.5 else 0  # 1) Guess    - Make a Prediction
        loss = result - predict                     # 2) Check    - CCalculate Loss - How far is our guess from actual outcome..
        weight += loss * .1                         # 3) Tweak    - Adjust weight by loss times Learning Rate - NOT GRADIENT DESCENT!

        print(f"epoch: {epoch+1} iteration: {i}\tcredit score: {credit:2}\tresult: {result}\tprediction: {predict}\tloss: {loss:.2f}\tweight: {weight:.2f}\tresult: {'CORRECT' if loss == 0 else 'WRONG'}")










#import random
##################################################################################################################################################################
### Function to Generate RAndom Data
##################################################################################################################################################################
#simplest_data = generate_random_linear_data(30,False)
def generate_random_linear_data(qty_rand_data, include_anomalies): # TODO rename these variables):
    training_data = []
    for _ in range(qty_rand_data):
        score = random.randint(1, 100)
        if include_anomalies:
            second_number = 1 if random.random() < (score / 100) else 0
        else:
            second_number = 1 if score >=50 else 0
        training_data.append((score, second_number))
    return training_data





