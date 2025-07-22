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
from sklearn.metrics import root_mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
```

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
- Penalizes large _positive or negative_ coefficients
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

|                 | **Predicted: No** | **Predicted: Yes** |
| --------------- | ----------------- | ------------------ |
| **Actual: No**  | True Negative     | False Positive     |
| **Actual: Yes** | False Negative    | True Positive      |

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

### Logistic Regression and the ROC Curve

- Logistic regression is used for classification problems
- Logistic regression outputs probabilities
- If the probability, `p` > 0.5:
  - Data is labeled 1
- If the probability `p` < 0.5:
  - Data is labeled 0

```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

y_pred_probs = logreg.predict_proba(X_test)[:, 1]
print(y_pred_probs[0])
```

## Probability thresholds

- By default, logistic regression threshold = 0.5
- Not specified to logistic regression
  - KNN classifiers also have thresholds
- What happens if we vary the threshold?

# ROC Curve

# Plotting ROC curve

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

# ROC AUC

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

# Hyperparameter tuning

- Ridge/lasso regression: Choosing `alpha`
- KNN: Choosing `n_neighbors`
- Hyperparameters: Parameters we specify before fitting the model
  - Like `alpha` and `n_neighbors`

# Choosing the correct hyperparameters

- Try lots of different hyperparameter values
- Fit all of them seperately
- See how well they perform
- Choose the best performing values

- This is called hyperparameter tuning
- It is essential to use cross-validation to avoid overfitting to the test set
- We can still split the data and perform cross-validation on the training set
- We withhold the test set for final evaluation

## Grid search cross-validation

```python
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": np.arrange(0.0001, 1, 10),
              "solver": ["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

## Limitations and an alternative approach

- scales badly

### RandomizedSearchCV

```python
from sklearn.model_selection import Randomized


 d
```
