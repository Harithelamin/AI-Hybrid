# Linear Regression(Supervised Learning)
# The goal is to Learn a relationship between input X and output y, then make predictions.

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)   # feature
y = np.array([2, 4, 6, 8, 10])                 # target


# split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# creat and train the model
model = LinearRegression()
model.fit(X ,y)

# predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# evaluate
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)


print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)


# 5. Plot
plt.scatter(X_train, y_train, color="blue", label="Train data")
plt.scatter(X_test, y_test, color="red", label="Test data")
plt.plot(X, model.predict(X), linestyle="--", label="Regression line")
plt.legend()
plt.show()

# Test
value = np.array([[6]])
prediction = model.predict(value)
print("Prediction for X=6:", prediction[0])


# user input
user_value = float(input("Enter a value for X: "))
# convert to 2D array
user_X = np.array([[user_value]])
prediction = model.predict(user_X)
print(f"Predicted value for X = {user_value} is Y = {prediction[0]}")