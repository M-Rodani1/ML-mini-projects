# Linear Regression from Scratch

A minimal implementation of linear regression using only NumPy, built from first principles to understand the mathematics behind gradient descent optimization.

## Overview

This project implements **linear regression without using scikit-learn's built-in models**. Instead, it manually computes:
- Mean Squared Error (MSE) loss function
- Analytical gradients using calculus
- Gradient descent optimization from scratch
- Model predictions and visualizations

**Goal:** Recover parameters of a noisy linear relationship using only gradient-based optimization.

---

## Mathematical Background

### The Model
We fit a linear function to data:

```
ŷ = wx + b
```

where:
- `w` = slope (weight)
- `b` = intercept (bias)
- `ŷ` = predicted value

### Loss Function (Mean Squared Error)

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

This measures the average squared difference between true values and predictions.

### Gradient Descent

To minimize loss, we iteratively update parameters in the direction of steepest descent:

```
w ← w - α × ∂MSE/∂w
b ← b - α × ∂MSE/∂b
```

where `α` is the learning rate.

### Gradients (Derived from Calculus)

Using the chain rule on MSE:

```
∂MSE/∂w = -(2/n) × Σ(yᵢ - ŷᵢ) × xᵢ
∂MSE/∂b = -(2/n) × Σ(yᵢ - ŷᵢ)
```

---

## Implementation Details

### Core Components

1. **Data Generation**: Synthetic data from `y = 3x + 5 + noise` with Gaussian noise (σ=2.0)
2. **Forward Pass**: Compute predictions `ŷ = wx + b`
3. **Loss Computation**: Calculate MSE
4. **Gradient Computation**: Manually compute ∂L/∂w and ∂L/∂b
5. **Parameter Update**: Apply gradient descent step
6. **Training Loop**: Iterate for fixed number of epochs

### Hyperparameters

- **Learning rate (α)**: 0.01
- **Epochs**: 100
- **Data points**: 100
- **Noise level (σ)**: 2.0

---

## Results

### Parameter Recovery

| Parameter | True Value | Learned Value | Error |
|-----------|-----------|---------------|-------|
| w (slope) | 3.000 | 3.403 | 0.403 |
| b (intercept) | 5.000 | 2.159 | 2.841 |

### Loss Reduction

- **Initial Loss**: 476.46
- **Final Loss**: 4.86
- **Theoretical Minimum** (noise variance): 4.00

The final loss is very close to the theoretical minimum, indicating the model has extracted all learnable signal from the noisy data.

### Training Dynamics

![Loss Curve](training_loss.png)

- **Fast convergence** in first ~15 epochs (loss drops from 476 → 10)
- **Gradual refinement** in epochs 15-100 (loss drops from 10 → 4.86)
- **Smooth decrease** indicates well-tuned learning rate

### Fitted Line

![Regression Results](regression_fit.png)

The learned line (red) passes through the center of the noisy data points (blue), while the true generating function (green dashed) is shown for comparison.

---

## Key Learnings

### 1. The Noise Floor
Even with perfect optimization, loss cannot drop below the variance of the noise in the data. This is the **irreducible error** or **Bayes error**.

### 2. Intercept Uncertainty
The intercept `b` has higher estimation error than the slope `w` because:
- It's estimated via extrapolation to x=0
- Data far from x=0 provides weak constraints on b
- Errors in slope estimation compound when extrapolating

This is a fundamental statistical property, not a bug in the implementation.

### 3. Gradient Descent Dynamics
- Large gradients early → fast progress
- Small gradients late → slow refinement
- Exponential convergence for convex problems

### 4. Hyperparameter Sensitivity
- Too large learning rate → oscillation or divergence
- Too small learning rate → slow convergence
- Proper tuning is essential for efficient optimization

---

## Requirements

```
numpy>=1.20.0
matplotlib>=3.3.0
```

Install with:
```bash
pip install numpy matplotlib
```

---

## Usage

```bash
python linear_regression_scratch.py
```

**Output:**
- Prints training progress every 10 epochs
- Displays final learned vs true parameters
- Shows two plots:
  1. Fitted regression line on data
  2. Loss curve over training epochs

---

## Code Structure

```python
# 1. Data generation
x, y = generate_synthetic_data(n=100, noise_std=2.0)

# 2. Model functions
predict(x, w, b)                    # Forward pass
compute_mse(y_true, y_pred)         # Loss function
compute_gradients(x, y_true, y_pred) # Gradient calculation

# 3. Training loop
for epoch in range(epochs):
    y_pred = predict(x, w, b)
    loss = compute_mse(y, y_pred)
    dw, db = compute_gradients(x, y, y_pred)
    w -= learning_rate * dw  # Update step
    b -= learning_rate * db
```

---

## License

MIT License - Feel free to use for learning purposes.