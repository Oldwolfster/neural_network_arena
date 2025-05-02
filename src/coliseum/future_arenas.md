# Plans for Future Arenas

---

## ⚔️ Classic Competitions (Skill Showcases)

| Arena                        | Description                                         | Goal                                      |
|------------------------------|-----------------------------------------------------|-------------------------------------------|
| **Circle-Inside-a-Square**   | Intuitive nonlinear classification                  | Tests ability to learn curved boundaries  |
| **Parity Check**             | Count even/odd number of 1s; input size grows       | Requires a memory-like structure          |
| **Bit-Flipping Memory Task** | Input: binary string; Output: same string with bit N flipped | Tracks long-term dependencies   |

---

## 📊 Real-World Mini-Arenas (Tuned for Speed)

| Arena                                    | Task Type                | Notes                                    |
|------------------------------------------|--------------------------|------------------------------------------|
| **California Housing** (downsampled)     | Regression               | Recognizable yet simplified              |
| **Titanic Survivors**                    | Binary classification    | Benchmarkable and human-interest driven  |
| **MNIST Subset** (e.g., 0s vs 1s)        | Binary classification    | Visual + performance challenge           |
| **Iris Dataset**                         | Multi-class classification | Quick to train, non-trivial            |

---

## 🧠 Strategy-Based Arenas

| Arena                   | Description                                    | Why It’s Good                                |
|-------------------------|------------------------------------------------|-----------------------------------------------|
| **Adversarial Noise**   | Inject controlled Gaussian noise mid-training  | Tests optimizer robustness under pressure     |
| **One Giant Outlier**   | All targets ≈10, one target = 10,000           | MAE vs MSE vs Huber showdown                  |
| **Sparse Inputs**       | Many zeros, a few key features                 | Forces reliance on signal over brute force    |

---

## 🧪 Experimental & Research-ish

| Arena                         | Hook                                               |
|-------------------------------|----------------------------------------------------|
| **Chaotic Function Prediction** | Predict sin(x) + sin(x²)                          |
| **Custom Function Recovery**    | y = 3x + noise; ground-truth known                |
| **AutoNormalize Challenge**     | Inputs differ wildly in magnitude                 |
| **Multi-Gladiator Ensembling**  | Each gladiator sees only part of the data         |

---

## 🧩 Advanced Concepts (Future)

- **Genetic Arena**  
  Evolution-style selection: gladiators evolve over time.

- **Curriculum Arena**  
  Controls sample difficulty across epochs.

- **Reinforcement Arena**  
  Simulate a simple control task (for future exploration).

---

## 🏆 Scoring Ideas (Coliseum Style)

| Metric               | Meaning                                                      |
|----------------------|--------------------------------------------------------------|
| **Time to Converge** | Fewest epochs to hit an accuracy or loss threshold           |
| **Generalization**   | Train vs test loss ratio                                     |
| **Consistency**      | Stability of performance across multiple random seeds        |
| **Resilience**       | Degradation under noise or outliers                          |

---

## Regression Arenas

| Arena                                     | Description                                                         |
|-------------------------------------------|---------------------------------------------------------------------|
| **Car Value Prediction**                 | From age & miles; inverse relationship, Dead-ReLU trap              |
| **House Price Prediction**               | Square feet (linear) + location score (step-graded nonlinear)       |
| **Income Prediction**                    | Education & experience; outliers add richness                      |
| **Temperature Prediction**               | Day of year; sinusoidal (tests seasonality)                         |
| **Advertising Spend → Sales**            | TV, online, print; diminishing returns nonlinear stress test        |

---

## Binary Decision Arenas

| Arena                                      | Features                                     |
|--------------------------------------------|----------------------------------------------|
| **Loan Repayment**                         | Credit score & income; classic logistic test |
| **Email Spam Detection**                   | Word count & URL count; NLP-style separability |
| **Plane Takeoff Decision**                 | Weight vs runway length; toy physics domain  |
| **XOR & Logic Gates**                      | Tests need for hidden layers                |
| **Hire vs Reject**                         | GPA & interview score; combining signals     |

---

## Nonlinearity & Hidden-Dependency Arenas

| Arena                                         | Description                                             |
|-----------------------------------------------|---------------------------------------------------------|
| **Noisy XOR**                                 | Classic XOR with added noise; breaks perceptrons        |
| **Circle vs Outside**                         | (x,y) points inside vs outside a circle; geometry trap  |
| **Final Grade Prediction**                    | Homework, midterm, effort multiplier (nonlinear weight) |
| **Room Brightness Determination**             | #Windows & shade position (practical nonlinearity)      |
| **Rocket Altitude Prediction**                | Fuel rate & angle; physics-curve challenge              |
