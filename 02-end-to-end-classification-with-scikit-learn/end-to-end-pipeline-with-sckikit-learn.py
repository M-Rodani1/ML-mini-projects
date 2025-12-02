from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.shape)
print(df.head())

print(data.target_names)
print('target distribution:')
print(pd.Series(data.target).value_counts())
print(df.describe())

from sklearn.model_selection import train_test_split
# First, separate features (X) and target (y)
X = df
y = data.target
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

from sklearn.preprocessing import StandardScaler
# Create the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Fit on training data only, then transform training data
# Transform test data using the SAME scaler (already fitted)
X_test_scaled = scaler.transform(X_test)
print("Before scaling (first feature):")
print(f"  Mean: {X_train.iloc[:, 0].mean():.2f}, Std: {X_train.iloc[:, 0].std():.2f}")
print("After scaling (first feature):")
print(f"  Mean: {X_train_scaled[:, 0].mean():.2f}, Std: {X_train_scaled[:, 0].std():.2f}")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000,random_state=42)
model.fit(X_train_scaled, y_train)
print("Model trained successfully!")
print(f"Number of features: {model.n_features_in_}")
print(f"Classes: {model.classes_}")  # Should be [0, 1]

# Make predictions on the test set (scaled data!)
y_pred = model.predict(X_test_scaled)
y_ped_probability = model.predict_proba(X_test_scaled)
print("First 10 predictions vs true labels:")
print("Predicted:", y_pred[:10])
print("True:     ", y_test[:10])
print("\nFirst 5 probability predictions [P(class=0), P(class=1)]:")
print(y_ped_probability[:5])

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
print(f"  True Negatives (TN): {cm[0,0]}")
print(f"  False Positives (FP): {cm[0,1]}")
print(f"  False Negatives (FN): {cm[1,0]}")
print(f"  True Positives (TP): {cm[1,1]}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
import matplotlib.pyplot as plt
import seaborn as sns

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved as 'confusion_matrix.png'")