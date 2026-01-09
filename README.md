# ML Mini-Projects: From Scratch Implementations

Building machine learning algorithms from the ground up using NumPy to understand the mathematical foundations behind modern ML.

---

## Purpose

This repository documents my journey learning machine learning as a first-year physics student. Rather than relying on high-level libraries like scikit-learn, I implement each algorithm from scratch using only NumPy. This approach forces me to:

- Understand the mathematical derivations (loss functions, gradients, optimization)
- Connect theory to code (matrix operations, numerical stability, vectorization)
- Build intuition through hands-on implementation

Each project includes comprehensive theory documentation (LaTeX), clean NumPy implementations, and visualizations.

---

##  Projects

### Completed

| Algorithm | Key Concepts |
|-----------|--------------|
| **Linear Regression** | Gradient descent, MSE loss, closed-form solution | 
| **Classification Pipeline** | Scikit-learn workflow (preprocessing, pipelines, evaluation) | 

### In Progress / Planned

| Algorithm | Key Concepts |
|-----------|--------------|
| **K-Means Clustering** | Expectation-Maximization, centroid updates, convergence |
| **Multinomial Naive Bayes** | Bayes' theorem, likelihood estimation, Laplace smoothing |
| **PCA** | Eigendecomposition, dimensionality reduction, variance explained |
| **Ridge/Lasso Regression** | L1/L2 regularization, feature selection, shrinkage |
| **Logistic Regression** | Binary classification, sigmoid, cross-entropy loss |
| **Softmax Regression** | Multiclass classification, softmax function, categorical CE |
| **Multi-Layer Perceptron** | Backpropagation, activation functions, hidden layers |
| **Support Vector Machine** | Margin maximization, kernel trick, dual formulation |
| **Autoencoder** | Dimensionality reduction, encoder-decoder, reconstruction loss |
| **Recurrent Neural Network** | Sequential data, hidden states, backpropagation through time |
| **Gaussian Mixture Model + EM** | Probabilistic clustering, expectation-maximization algorithm |
| **Gradient Boosting** | Ensemble methods, weak learners, boosting |

---

## Tech Stack

**Core Implementation:**
- **NumPy**: All mathematical operations (matrix algebra, gradients, optimization)
- **Matplotlib**: Visualizations (loss curves, decision boundaries, clusters)
- **Pandas**: Data handling and preprocessing

**Development:**
- **Python 3.x**: Standard library only for utilities
- **LaTeX**: Theory documentation (mathematical derivations, worked examples)

**Note**: Scikit-learn is used only for:
- Loading standard datasets (breast cancer, iris, etc.)
- Comparison/validation of results
- One dedicated project on production ML pipelines

```

All projects use synthetic datasets by default to focus on algorithm mechanics rather than data preprocessing.

---

##Background

I'm a first-year physics student learning machine learning through hands-on implementation. This repository serves both as:
- A learning tool for building deep understanding
- A portfolio demonstrating ability to translate mathematical concepts into working code

---
