### Double Trouble: Neural Networks Learn to Juggle Two Inputs
# Introduction and Context Setting

**Standard Greeting:**
Greetings and welcome back to Neural Network Arena, where we're leveling up our NN one feature at a time! 
In today's episode, we’re doubling down on complexity by adding a 2nd input to our model.
I'm your host, Simpleton, here to keep it simple with the simpletron  and 

# Hook the Audience

> <<Visual of a juggler trying to handle one ball vs two balls>>

Imagine juggling one ball versus two — keeping it simple has allowed us to easily and clearly see how every detail
affects accuracy. But to see the impact of some new features like loss functions activation functinos and
feature scaling, we will need to inputs to see the impact.

The example we will use today is predicting income from
1) Years of Experience
2) Years of College

# Contender Introductions

> <<Visual of Arena>>

In today's showdown, we've got a three-way face-off
In the left corner, we have the simpleton, just a single input

> <<Visual of Simpleton>>

In the left corner we are going back to our simpletron, the blackbird,
it will continue to have a single input

In the right corner, we have Hayabusa - our champion from previous
battles, which we will upgrade to handle two inputs! 
> <<Visual of Hayabusa>>

And in the center, we've got the infamous GBS, AKA Gradient Descent. 
Sure, it's rocking parallel processing and 
dynamic input handling — too bad it also brings all the BS of GBS!"
It will serve as a great benchmark as we search for a better approach.

# NNA Tour

Before we plan out this enhancement, the NNA has been back to SPA getting some upgrades,
making it even easier to make the models now:

> <<Visual of refactoring benefits>>
- The parent class of the gladiators will automatically create a weight for each input, removing this burden from 
the individual model classes.
- We removed even more boiler plate from the model..  We realized the parent class has access to most of 
the info the child class was returning, in fact everything except the models prediction.
Instead of the model needing to 
create a "gladiator output" object, simply return the prediction.  This helps in ease of building models and
keeping the code concise.
- It used to be we had to name the model's class the same as the file so the engine could find it.
  - Now, name it anything you like, the engine will find it by file name regardless of what you name the class.
- Also, to get organized we set it up so we can have subfolders in Arenas and Models. 
The engine does a recursive scan through them all.  This allows us to grow our model library without chaos
- Finally, we decided to give the engine a folder of it's own to keep everything more organized.

With that update, let's make a plan so we know where we are going!

> <<Visual animation of basic NN loop>>

The blackbird is following out basic NN loop.
Let's use predicting house value from square feet as an example.

1. Predict: Take the input(sq feet), multiply by weight, add bias
   (note - they often start weight and bias as random numbers - i don't quite love that
    but  i do feel once we get this model right, you should be able to start them at any value
    and the training process will keep adjusting them until they get to their optimum values)
2. Check: Compare prediction with actual answer
3. Adjust: Tweak the weight and bias based on our error

But now...

> <<Visual split screen - old vs new prediction>>

The second input means our neuron isn’t just working harder—
it’s working smarter. Three small changes:

1. **Second Weight:** Our neuron now has TWO weights to manage
2. **Prediction:** Instead of just one multiplication, we're doing TWO:
   ```python
   prediction = input1 * weight1 + input2 * weight2 + bias
   ```
3. **Adjustment:** We need to update BOTH weights during learning

Here is a diagram

Here's the plan:
1) Test our existing single input Blackbird
2) Copy it and add 2nd input handling (details on next slides)
3) Include existing GBS

Let's copy our single-input Simpleton and convert it to the two-input Hayabusa. 
We'll do this step by step:
1. First, we'll capture both inputs
2. Next - add a 2nd weight
2. Then update the prediction logic
3. Finally, adjust the update(learning) mechanism to update new weight

Last thing before the battle, let's take a closer look at the arena.

1) Start with base Salary of 30k
2) Add 2k per year of exp
3) Add 3k per year of college
4) Add some noise
Keep in mind this Arena is more about providing test data for developing or multiple input handling
opposed to trying to match exactly how it is in the real world
> <<Visual split screen - old vs new prediction>>

# Code Forge

> <<Visual of code editor split screen>>

When upgrading, even a tiny tweak can make or break the whole model. 
Here, we're carefully evolving the Simpletron to handle a new layer of complexity.

Let's dive into the code! I'll show you how we're upgrading our Simpletron to handle multiple inputs.

> <<Live coding isn't scripted>>


# Battle Royale

> <<Visual of real-time competition>>

Now it's time to see these neural networks in action! Let's take a look at our Arena
Predict salary from years of experience and college

Today, we're going to do a simple linear approach, on the next episode 
we will add a curveball by making it multiplicative

30k base salary
1-40 years of experience (4k / year)
1-8 years of college     (5k / year)

Now let's see if Hayabusa’s upgrade makes it the new champion, or
if GBS will show us that multiple inputs demand Gradient Descent’s stinky touch.

[Live competition runs]

# Results Analysis

> <<Visual of learning curves>>

Let's break down what we just saw:

# Lessons from the Arena

Key takeaways from today's battle:
1. In the real world, Real-world applications often need multiple inputs
2. All you need for the additional input is an additional weight.
2. Standard GBS update works on the new just like the old..
3. Optimum weights correspond 
# Next Battle Preview

Next time, we're tackling feature scaling! 
Scaling inputs is key—otherwise, GBS could end up swimming in more BS than it can handle!

# Audience Engagement

That's all for today's episode of Neural Network Arena! Don't forget to like and subscribe, 
and let me know in the comments - what real-world problems would YOU solve with a two-input neural network?

# Peace

As always... PEACE!!!