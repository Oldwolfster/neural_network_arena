"""
Neural Network Arena - README

Overview:
----------
The Neural Network Arena is a project designed to simplify the process of understanding and evaluating neural networks. It began with a curiosity about Large Language Models (LLMs) and evolved into a comprehensive framework for building, testing, and comparing neural network models—referred to as "Gladiators"—against various test data generation algorithms known as "Training Pits." The core philosophy is to simplify each component, examine each machine learning technique individually, and gain a clear understanding of its impact.

Components:
------------
1. **Gladiators**:
   - These are the neural network models that compete against each other in the arena.
   - The goal is to isolate various ML techniques and examine their effects on the models.
   - Each Gladiator inherits from a base class that defines their shared behavior.

2. **Training Pits**:
   - Training Pits are the algorithms responsible for producing the test and training data.
   - Various types of pits are available, including linear, quadratic, binary decision, and regression models.
   - Each pit inherits from a base class that defines their shared behavior.

3. **Engine**:
   - The Engine runs the entire arena, taking the specified Training Pit, a list of Gladiators, and hyperparameters.
   - It generates a single set of training data, feeds it to all the Gladiators, and collects detailed results.
   - The Engine produces comprehensive reports comparing the Gladiators’ performance based on key metrics.

Approach and Philosophy:
-------------------------
The Neural Network Arena was created with two main objectives:
1. **Simplify Neural Networks**: 
   - The starting point was the development of the Simpletron—a neural network model stripped down to its simplest form. The goal was to make the entire process easy to understand, even at the expense of some accuracy.
   - One at a time, various ML techniques are added to the Simpletron to see their isolated effect.

2. **Transparent Testing**:
   - The Arena provides a clear comparison of neural network models by testing them against the same data under identical conditions. 
   - Randomness in training data generation was controlled by creating the Training Pits, which allows the models to be compared on the same set of training data.
   
Critical View on Gradient Descent (GBS):
----------------------------------------
- From day one, gradient descent (referred to as GBS - Gradient Bull $hit) has felt like a flawed technique in this context.
- The belief is that gradient descent "double counts" the input since the input is already part of the error calculation.
- Through rigorous testing, it has been consistently more accurate to exclude gradient descent in favor of simpler methods.
- The flexibility of the learning rate, as long as it remains small enough, invalidates the benefit of computationally expensive adjustments, further questioning the usefulness of gradient descent.
- The author would prefer to have IBS than GBS!

Project Inspiration and Motivation:
-----------------------------------
The project was born from the user's experience learning neural networks after experimenting with LLMs. Initially aiming to refresh MVC skills, the simplicity and power of LLMs sparked a deeper dive into neural networks and machine learning. Rather than accepting complex implementations as black boxes, the user chose to simplify, explore, and build everything from the ground up.

YouTube Channel:
-----------------
In addition to building the arena, the user is creating a series of YouTube videos documenting the exploration and explaining each ML technique applied to the Simpletron. The videos emphasize learning through simplification, breaking down each technique, and analyzing its impact.

Future Directions:
-------------------
- The next steps may involve splitting data into training and test sets, expanding the collection of Gladiators, and introducing more complex neural network architectures to the arena.
- Additional metrics, such as cross-entropy (log loss), will be incorporated as the project continues to evolve.

License:
---------
The project is open to contributions, and any improvements, suggestions, or collaborations are welcome!

"""
