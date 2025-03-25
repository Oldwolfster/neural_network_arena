Mission Statement (Working Draft)
"Simplex is a self-regulating optimizer that eliminates
the need for manual learning rate tuning, 
dynamically prevents gradient explosions,
can add new Loss functions, activations, and initializers like legos, but calls out 
incompatible components (for example Hinge and Sigmoid).
Designed for clarity, speed, and flexibility,
it works out of the box for regression and binary tasks,
and adapts to architecture, data, and noise—so you can focus on results, not tweaking knobs.
Simplex delivers smooth, fast convergence while making fragile networks 'Hard to get wrong'" 

Simplex is a highly performant self-regulating optimizer that eliminates
the need for manual tuning.  The key to outperforming all other optimizers to date,
lies in the self regulation as it sets monitors and updates optimum values for hyper
parameters.

In addition to performance, the self optimizing mechanism makes it "Hard to get wrong"
with the flexibility to override architecture (in the event an override would prevent 
success, rather than most optimizers which run and then surprise you with dismal results
,wasting time and compute, if an override is fatal, will terminate before running and clearly 
explain incompatibility.   If suboptimal, will warn and allow user to make an informed decision

Ideas to improve it:
Momentum and Its Coefficient:
Yes, when you use a momentum coefficient of 0.9, it means 90% of the previous update (the “momentum” term) is carried over and only 10% comes from the current gradient. This strong momentum can smooth out oscillations, though it does mean that historical gradients have a very heavy influence. The balance is crucial—too high, and you risk overshooting if the direction of the gradient changes abruptly.

Explosion Threshold, Growth Rate, and Correction Rate:
Absolutely—these hyperparameters are critical and currently arbitrary. It’s important to fine-tune (or even adapt dynamically) the explosion threshold (the multiplier on max sample magnitude), the growth rate (+5% per iteration), and the explosion correction (halving the learning rate on an “explosion”). Their values could be set relative to the characteristics of the data and the weight scales rather than as fixed constants.

Adapting Hyperparameters During Training:
Using a decay schedule to adjust the base learning rate over time can be very effective. Early on, a higher learning rate helps make rapid progress, but as you get closer to convergence, a slower, more precise adjustment is needed.

Reversing the Learning Rate Growth in Later Stages:
This is a key insight. Initially, increasing the learning rate can speed up training ("getting close"), but once the MAE begins to plateau, continuing to increase the learning rate can prevent fine-tuning. In the later stages, reducing the learning rate (or even applying a decay) can help the optimizer settle into a minimum.

Error-Based Learning Rate Adjustment:
Incorporating the behavior of the error (like MAE or MSE) directly into your adaptive mechanism can create a more holistic feedback loop. If the error increases or plateaus, then that should prompt a reduction in the learning rate, ensuring that adjustments are aligned with the loss landscape rather than just the gradient magnitude.

Tiered or Granular Correction Rates:
Instead of a binary approach (e.g., +5% or halving), a tiered strategy that adjusts the correction based on the severity of the update can prevent over-penalizing the learning rate. For instance, a mild overshoot might warrant a small reduction, while a major explosion could trigger a larger cut.

Learning from Rprop:
Rprop’s approach—focusing solely on the sign of the gradient rather than its magnitude—offers a robust mechanism to adjust weight updates. Studying and incorporating ideas from Rprop (like using sign changes to determine when to reduce the update magnitude) could add stability to your optimizer.

Cyclical Learning Rates and “One Cycle” Policy:
These techniques introduce controlled, predictable oscillations in the learning rate that are known to help in escaping local minima and ensuring convergence. They offer a structured way to ramp up the learning rate initially and then lower it systematically, which aligns with your goals of reducing manual tuning.

Dropping the Learning Rate in the Final Stages:
Your insight here is spot on. When the MAE levels off, reducing the learning rate further can help the optimizer fine-tune the weights, leading to a larger reduction in error per update. This “cooling down” phase is common practice and can help avoid the persistent oscillations that come with a high learning rate.

Leveraging Weight Movements:
Considering not just the gradient magnitudes but also the actual weight updates (their direction and magnitude) can provide additional context for adjusting the learning rate. This may help in dynamically balancing aggressive updates with necessary stability, and it’s a concept closely related to momentum-based methods.

Explosion Cooldown Period:
Introducing a “cooldown” period after an explosion event—where the same parameter isn’t allowed to trigger another explosion immediately—could be beneficial. It’s a trade-off: while it might allow one more large update, it could also prevent overreactive adjustments. Tuning the length of this cooldown is key.

Short Bursts to Escape Plateaus:
Occasionally raising the learning rate in a controlled burst can help escape local minima or overcome stagnation. This is a strategy seen in cyclical learning rates and can be useful when the error appears stuck. You could even differentiate between using such bursts for escaping shallow local minima versus breaking out of a plateau.

Red Flags in Late Training:
An increase in the learning rate or a sudden spike in MAE late in training is indeed a strong signal that the optimizer needs intervention. Whether that means further reducing the learning rate or re-evaluating the strategy for that parameter, monitoring these “red flags” can be critical for maintaining convergence.

Rotating Through Strategies When MAE Stalls:
Implementing a meta-strategy that cycles through different remedies (e.g., adjusting momentum, applying a decay, issuing a burst, etc.) when a plateau is detected is an innovative idea. By testing multiple approaches sequentially, you can identify which adjustment most effectively scrubs the error down again. This dynamic, automated approach could be a cornerstone of a new optimization paradigm that minimizes manual intervention.

In Regression, Tanh is super saturated from the hidden weights

**** INVESTIGATE  iRprop+ variant

Your summary and these points underscore a holistic approach to not only eliminate manual LR tuning but also to dynamically adapt the entire optimization process. This strategy would make your optimizer both robust and flexible across various tasks—from classification (like XOR) to more complex regression problems.