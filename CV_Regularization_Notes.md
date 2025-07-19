# 📘 Cross Validation & Regularization

## 🔁 Cross Validation

- Model performance can vary depending on how the data is split.
- **Cross-validation (CV)** helps generalize model performance by:
  - Splitting data into multiple "folds"
  - Using each fold as a test set once, and training on the remaining folds
  - Aggregating performance metrics across folds

### 🔹 Common Terms
- **k-fold CV**: k separate folds (e.g., 5-fold CV → 5 rounds)
- More folds = more accurate estimate, but more computation

### 🔧 Code Example
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import numpy as np

kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()

cv_results = cross_val_score(reg, X, y, cv=kf)

print(cv_results)
print(np.mean(cv_results), np.std(cv_results))
print(np.quantile(cv_results, [0.025, 0.975]))
```

---

## 🔐 Regularized Regression

### 📉 Why Regularize?
- Linear regression minimizes a loss function to fit `y = ax + b`
- Large coefficients can lead to **overfitting**
- **Regularization** adds a penalty term to discourage large coefficients

---

## 🧮 Ridge Regression
- **Penalty**: `alpha * sum(a_i^2)` (L2 norm)
- Penalizes large *positive or negative* coefficients
- `alpha` is a **hyperparameter** that controls model complexity
  - `alpha = 0`: behaves like OLS (can overfit)
  - High `alpha`: can underfit

### 🔧 Code Example
```python
from sklearn.linear_model import Ridge

scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))

print(scores)
```

---

## ✂️ Lasso Regression (Feature Selection)
- **Penalty**: `alpha * sum(|a_i|)` (L1 norm)
- Shrinks **less important coefficients** to **zero**
- Helps with **feature selection**

### 🔧 Lasso Example
```python
from sklearn.linear_model import Lasso

scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    scores.append(lasso.score(X_test, y_test))

print(scores)
```

### 🧪 Lasso for Feature Selection
```python
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

X = df.drop("glucose", axis=1).values
y = df["glucose"].values
names = df.drop("glucose", axis=1).columns

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_

plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.title("Lasso Coefficients (Feature Importance)")
plt.show()
```

---

## 🧠 How Good is Your Model?

### ⚖️ Classification Metrics
- Accuracy = Correct predictions / Total predictions
- Not always reliable — especially with **class imbalance**

#### ⚠️ Example:
- Fraud detection: 99% legit, 1% fraud
- Classifier that always predicts "legit" is 99% accurate... but useless

---

## 🔍 Confusion Matrix
|                | **Predicted: No** | **Predicted: Yes** |
|----------------|-------------------|--------------------|
| **Actual: No** | True Negative     | False Positive     |
| **Actual: Yes**| False Negative    | True Positive      |

### 🧮 Common Metrics:
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP) → How many predicted positives are actually correct?
- **Recall** = TP / (TP + FN) → How many actual positives were identified?
- **F1 Score** = Harmonic mean of precision and recall

---

## 🔧 Confusion Matrix in scikit-learn
```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
