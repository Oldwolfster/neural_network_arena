Backpropogation is based on the calculus chain rule.... think of it as a chain of links...


Link 1, derivative of loss with respect to prediction.
how much did the prediction impact the loss.   ALL OF IT?


🔗 Link 1: Loss Gradient (Anchor of the Chain)
📌 Formula:

∂
𝐿
𝑜
𝑠
𝑠
∂
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
=
𝑇
𝑎
𝑟
𝑔
𝑒
𝑡
−
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
∂Prediction
∂Loss
​
 =Target−Prediction
💡 Interpretation:
This is the gradient of the loss function with respect to the model’s output.

In MSE, this is simply the error:
(
𝑇
𝑎
𝑟
𝑔
𝑒
𝑡
−
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
)
=
(
−
0.981
)
(Target−Prediction)=(−0.981)
🔗 Link 2: Activation Gradient (Sigmoid Derivative)
📌 Formula:

∂
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
∂
𝑅
𝑎
𝑤
𝑆
𝑢
𝑚
=
𝜎
(
𝑧
)
⋅
(
1
−
𝜎
(
𝑧
)
)
∂RawSum
∂Prediction
​
 =σ(z)⋅(1−σ(z))
💡 Interpretation:
This tells us how much the raw sum (pre-activation) contributes to the prediction.

In this case, since the output neuron uses Sigmoid, we use its derivative:
0.981
⋅
(
1
−
0.981
)
=
0.0183
0.981⋅(1−0.981)=0.0183
🔗 Link 3: Blame (δ)
📌 Formula:

𝛿
=
∂
𝐿
𝑜
𝑠
𝑠
∂
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
×
∂
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
∂
𝑅
𝑎
𝑤
𝑆
𝑢
𝑚
δ= 
∂Prediction
∂Loss
​
 × 
∂RawSum
∂Prediction
​
 
💡 Interpretation:

This combines the loss gradient with the activation gradient, showing the total influence of error on the neuron.
(
−
0.981
)
×
(
0.0183
)
=
−
0.0179
(−0.981)×(0.0183)=−0.0179
🔗 Link 4: Weight Gradients (Final Adjustments)
Each weight update follows:

Δ
𝑊
=
learning rate
×
𝛿
×
input
ΔW=learning rate×δ×input
💡 Interpretation:

We now distribute the error signal to each weight by multiplying it with the input value.
If W1 = 0.5, and input = 0.76, update rule is:
0.5
+
(
0.1
×
−
0.0179
×
0.76
)
=
0.4986
0.5+(0.1×−0.0179×0.76)=0.4986
(Similarly for W2)
Final Chain Recap
Step	Formula	Value in Example
Link 1: Loss Gradient	
∂
𝐿
𝑜
𝑠
𝑠
∂
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
=
(
𝑇
𝑎
𝑟
𝑔
𝑒
𝑡
−
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
)
∂Prediction
∂Loss
​
 =(Target−Prediction)	
−
0.981
−0.981
Link 2: Activation Gradient	
𝜎
(
𝑧
)
⋅
(
1
−
𝜎
(
𝑧
)
)
σ(z)⋅(1−σ(z))	
0.0183
0.0183
Link 3: Blame (δ)	
𝛿
=
Loss Gradient
×
Activation Gradient
δ=Loss Gradient×Activation Gradient	
−
0.0179
−0.0179
Link 4: Weight Updates	
𝑊
=
𝑊
+
𝜂
⋅
𝛿
⋅
Input
W=W+η⋅δ⋅Input	
0.5
→
0.4986
0.5→0.4986


