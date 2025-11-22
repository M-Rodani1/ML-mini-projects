import numpy as np
import matplotlib.pyplot as plt
# Set seed
np.random.seed(42)
# True parameters
w_true =3
b_true = 5
# Generate x values: use np.linspace(start, stop, num_points)
x = np.linspace(0, 10, 100)
# Perfect y (no noise)
y_perfect = w_true * x + b_true
# Add noise: use np.random.randn(n) * std_dev
noise = np.random.randn(100) * 2
y = y_perfect + noise

# Print first 5 pairs
print("First 5 data points:")
for i in range(5):
    print(f"x={x[i]:.2f}, y={y[i]:.2f}")

# Initialize parameters
w = np.random.randn() * 0.01
b = 0.0
print(f"\nInitial parameters: w={w:.4f}, b={b:.4f}")
print(f"True parameters:    w={w_true}, b={b_true}")

def predict(x, w, b):
    return w * x + b

# Test it with initial parameters
y_pred = predict(x, w, b)
print("\nPredictions:")
for i in range(5):
    print(f"x={x[i]:.2f}, y={y_pred[i]:.2f}")

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

initial_loss = compute_mse(y, y_pred)
print(f"\nInitial loss: {initial_loss:.2f}")

def compute_gradients(x,y_true,y_pred):
    n = len(y_true)
    error = y_true - y_pred
    # Gradient w.r.t. w: -(2/n) * sum(error * x)
    dw = -(2/n) * np.sum(error*x)
    # Gradient w.r.t. b: -(2/n) * sum(error)
    db = -(2/n) * np.sum(error)
    return dw, db
dw,db = compute_gradients(x,y_true=y,y_pred=y_pred)
print(f"\nGradients:\ndw={dw:.2f}, db={db:.2f}")

# Hyperparameters
learning_rate = 0.01
epochs = 100
# Track loss history for plotting later
loss_history = []

for epoch in range(epochs):
    # Forward pass
    y_pred = predict(x, w, b)
    # Compute loss
    loss = compute_mse(y, y_pred)
    loss_history.append(loss)
    # Compute gradients
    dw, db = compute_gradients(x,y_true=y,y_pred=y_pred)
    # Update parameters (gradient descent step)
    w = w - learning_rate * dw
    b =b-learning_rate * db
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")
print(f"\nFinal parameters: w={w:.4f}, b={b:.4f}")
print(f"True parameters:  w={w_true}, b={b_true}")
print(f"Final loss: {loss_history[-1]:.4f}")

# Create predictions with learned parameters
y_pred_final = predict(x, w, b)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Noisy data', s=30)
plt.plot(x, y_pred_final, 'r-', linewidth=2, label=f'Learned: y={w:.2f}x+{b:.2f}')
plt.plot(x, y_perfect, 'g--', linewidth=2, label=f'True: y={w_true}x+{b_true}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression from Scratch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("TRAINING COMPLETE")
print(f"Learned parameters: w={w:.4f}, b={b:.4f}")
print(f"True parameters:    w={w_true}, b={b_true}")
print(f"Parameter errors:   Δw={abs(w-w_true):.4f}, Δb={abs(b-b_true):.4f}")
print(f"\nFinal MSE loss: {loss_history[-1]:.4f}")
print(f"Noise variance: {2.0**2:.4f} (theoretical minimum loss)")

plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()
