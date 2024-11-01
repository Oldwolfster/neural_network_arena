import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def analyze_salary_model(num_samples=1000):
    # Generate sample data
    years_experience = np.random.uniform(0, 40, num_samples)
    college = np.random.uniform(0, 8, num_samples)
    noise = np.random.normal(0, 5, num_samples)

    # True relationship
    intermediate_salary = 30 + (4 * years_experience) + noise
    true_salary = intermediate_salary * (college + 0.5)

    # Linear approximation components
    component1 = years_experience  # w1 term
    component2 = college          # w2 term
    component3 = years_experience * college  # w3 term (interaction)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))

    # 1. 3D surface of true relationship
    ax1 = fig.add_subplot(231, projection='3d')
    X, Y = np.meshgrid(np.linspace(0, 40, 20), np.linspace(0, 8, 20))
    Z = (30 + 4*X) * (Y + 0.5)
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Experience')
    ax1.set_ylabel('College')
    ax1.set_zlabel('Salary')
    ax1.set_title('True Relationship\n(Multiplicative)')

    # 2. Component contributions
    ax2 = fig.add_subplot(232)
    ax2.scatter(years_experience, component1, alpha=0.5, label='Experience Term')
    ax2.scatter(years_experience, component3, alpha=0.5, label='Interaction Term')
    ax2.set_xlabel('Years Experience')
    ax2.set_ylabel('Component Value')
    ax2.set_title('Component Terms vs Experience')
    ax2.legend()

    # 3. Residuals analysis
    ax3 = fig.add_subplot(233)
    # Fit a simple linear model
    X = np.column_stack((years_experience, college, years_experience*college))
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, true_salary)
    predictions = model.predict(X)
    residuals = true_salary - predictions

    ax3.scatter(predictions, residuals, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Predicted Salary')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs Predicted\nValues')

    # 4. Interaction effect visualization
    ax4 = fig.add_subplot(234)
    for college_level in [2, 4, 6]:
        mask = (college > college_level - 0.5) & (college < college_level + 0.5)
        ax4.scatter(years_experience[mask], true_salary[mask],
                   alpha=0.5, label=f'College ≈ {college_level}')
    ax4.set_xlabel('Years Experience')
    ax4.set_ylabel('Salary')
    ax4.set_title('Salary vs Experience\nfor Different College Levels')
    ax4.legend()

    # 5. Weight contribution analysis
    ax5 = fig.add_subplot(235)
    contribution1 = model.coef_[0] * years_experience
    contribution2 = model.coef_[1] * college
    contribution3 = model.coef_[2] * (years_experience * college)

    ax5.boxplot([contribution1, contribution2, contribution3],
                labels=['Exp', 'College', 'Interaction'])
    ax5.set_ylabel('Contribution to Prediction')
    ax5.set_title('Distribution of\nWeight Contributions')

    # Print coefficients
    print(f"Weight 1 (Experience): {model.coef_[0]:.2f}")
    print(f"Weight 2 (College): {model.coef_[1]:.2f}")
    print(f"Weight 3 (Interaction): {model.coef_[2]:.2f}")
    print(f"Bias: {model.intercept_:.2f}")

    plt.tight_layout()
    return fig, model.coef_

# Example usage:

fig, coefficients = analyze_salary_model()
plt.show()
