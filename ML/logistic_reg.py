# Classification (Logistic Regression)
# The goal it to Classify data into two classes (0 or 1).

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# 1. Create dataset
X = np.array([
    [20], [22], [25], [30], [35],
    [40], [45], [50], [55], [60]
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # binary labels

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
# Accuracy: percentage of correct predictions
accuracy = accuracy_score(y_test, y_pred)
# The confusion matrix is a detailed breakdown of the classification results
# Shows how many samples were classified correctly or incorrectly for each class.
cm = confusion_matrix(y_test, y_pred)


# Calculate precision, recall, and F1-score
# how many were actually positive?
# It measures the accuracy of positive predictions
# false positives are costly, e.g., spam filters
precision = precision_score(y_test, y_pred)

# how many did the model correctly identify?
# It measures the model’s ability to capture all positive cases
recall = recall_score(y_test, y_pred)

# It balances both precision and recall.
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# Test
user_input = float(input("Enter an age value to predict class (0 or 1): "))

# Reshape input for prediction (1 sample, 1 feature)
user_X = np.array([[user_input]])

# Predict class
user_pred = model.predict(user_X)
print(f"Predicted class for age {user_input}: {user_pred[0]}")
