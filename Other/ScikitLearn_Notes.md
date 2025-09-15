# 📘 Scikit-Learn: Supervised & Unsupervised Learning

## 🔍 Supervised Learning
- The **target values** are known ahead of time.
- **Types:**
  - **Classification** – Predict categories (e.g., spam vs. not spam)
  - **Regression** – Predict continuous values (e.g., house prices)

## 📊 Unsupervised Learning
- No labeled data (targets unknown)
- Model learns structure from data alone

## 🔁 Model Structure (General Workflow)
```python
from sklearn.module import Model

model = Model()
model.fit(X, y)           # Train with labeled data
predictions = model.predict(X_new)  # Predict new data
```
- `X`: array of features
- `y`: array of labels/targets

### Steps for Classification
1. Build a model
2. Train model with labeled data
3. Pass in **unlabeled data**
4. Model predicts labels

## 👥 K-Nearest Neighbors (KNN) Classifier

### Fit a KNN Classifier
```python
from sklearn.neighbors import KNeighborsClassifier

X = df[[column1, column2]].values
y = df['target_column'].values

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)

X_new = [[val1, val2], [val3, val4]]
predictions = knn.predict(X_new)
print(f"Predictions: {predictions}")
```

### Measuring Performance
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))  # Accuracy
```

### Model Complexity
- Larger `k` → Less complex → **Avoids overfitting**
- Smaller `k` → More complex → **Risk of overfitting**
- Try different `k` values to find the best one.

## 📈 Intro to Regression

### Dataset Example
```python
import pandas as pd
df = pd.read_csv("diabetes.csv")
print(df.head())

X = df.drop("glucose", axis=1).values
y = df["glucose"].values

X_bmi = X[:, 3].reshape(-1, 1)  # Make 2D for scikit-learn
```

### Plotting Data
```python
import matplotlib.pyplot as plt

plt.scatter(X_bmi, y)
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.show()
```

## 🔧 Fitting a Linear Regression Model
```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_bmi, y)

predictions = reg.predict(X_bmi)

plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.show()
```

### Key Concepts
- Model: `y = ax + b`
- `a` and `b` are **coefficients** (slope and intercept)
- Goal: minimize **loss function** (aka error or cost function)
- Residual: vertical distance from data point to line
- **Ordinary Least Squares (OLS)**: minimizes sum of squared residuals (RSS)

## 📉 Multiple Linear Regression
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
```

## 📐 Model Metrics

### R-squared (R²)
- Quantifies **variance in target explained by model**
- Ranges from **0 to 1**: higher is better
```python
reg_all.score(X_test, y_test)
```

### RMSE (Root Mean Squared Error)
- RMSE is in **same units** as target variable
- Derived from MSE (which is squared)
```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
```
