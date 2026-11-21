# Beyond Accuracy: Model Evaluation on Imbalanced Data

Evaluating classification and regression models with scikit-learn, and choosing metrics that fit the problem instead of defaulting to accuracy.

- **Classification:** predicting customer churn in a telecom dataset where only ~15% of customers churn.
- **Regression:** predicting median house values for California districts.

## Project structure

```
.
├── lab1.ipynb          # main notebook
├── data/
│   ├── bigml_59c28831336c6604c800002a.csv   # telecom churn data
│   └── california_housing.csv               # California housing data
├── pyproject.toml
└── uv.lock
```

## How to run

```bash
uv sync
uv run jupyter lab lab1.ipynb
```

## Results

### Churn classification (5-fold CV, validation scores)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| DummyClassifier | 0.850 | 0.000 | 0.000 | 0.000 |
| LogisticRegression | 0.856 | 0.549 | 0.209 | 0.303 |
| SVC | 0.904 | 0.928 | 0.390 | 0.545 |
| LogisticRegression (balanced) | 0.761 | 0.351 | 0.699 | 0.467 |
| **SVC (balanced)** | 0.883 | 0.593 | **0.711** | **0.646** |

### Housing regression (5-fold CV, validation scores)

| Model | R² | RMSE | MAPE |
|---|---|---|---|
| DummyRegressor | 0.000 | – | 0.623 |
| **KNeighborsRegressor** | **0.689** | **0.643** | **0.240** |
| Ridge | 0.482 | 0.810 | 0.320 |

## Key learnings

### Accuracy lies on imbalanced data
- A `DummyClassifier` that always predicts the majority class gets **85% accuracy** and catches **zero** churners.
- `1 - accuracy` of the dummy model = the proportion of the minority class.
- Always compare against a baseline before trusting a score.

### DummyClassifier strategies
- `most_frequent` and `prior` make the **same predictions** (always the majority class).
- They differ in `predict_proba`: `most_frequent` gives `[1.0, 0.0]`, `prior` gives the class proportions (`[0.85, 0.15]`).

### Classification metrics
```
              pred False   pred True
real False  [[   TN          FP   ],
real True    [   FN          TP   ]]
```
- **Precision** = TP / (TP + FP): of the customers I flagged, how many really churned?
- **Recall** = TP / (TP + FN): of the customers who churned, how many did I catch?
- **F1** = harmonic mean of precision and recall.
- Precision is undefined (0/0) when the model never predicts the positive class, which causes `UndefinedMetricWarning`.

### Pipelines prevent data leakage
- Models that use the features (LogisticRegression, SVC) need preprocessing: one-hot encoding and scaling.
- Put the preprocessor **inside** a pipeline so that each CV fold fits the scaler only on its own training data.
- The dummy model doesn't need it because it ignores the features.

### `cross_validate` vs `cross_val_predict`
- `cross_validate` returns **a score per fold**, to compare models (mean ± std). Use `return_train_score=True` to get train scores too.
- `cross_val_predict` returns **a prediction per example**, each made by a model that didn't see it. Use it for confusion matrices.
- `cross_val_predict` has no `scoring` argument.
- In `cross_validate`, `test_*` means the **validation fold**, not the test set.

### `class_weight="balanced"`
- It reweights errors during training: weight = n_samples / (n_classes × n_class_samples). Here, missing a churner costs ~5.7× more than misclassifying a customer who stays.
- Effect on SVC: TP 136 → 248, FN 213 → 101, FP 11 → 171.
- It's the **precision/recall trade-off**: recall goes up and precision goes down. Whether it's worth it depends on the business cost of each error type.

### Regression metrics
- **MSE**: mean squared error. Squaring removes the sign and punishes large errors more.
- **RMSE**: √MSE, back in the target's units. RMSE ≥ MAE, and the gap grows when errors are uneven.
- **MAPE**: average absolute error as a percentage of the true value. It is relative and has no sign.
- **R²**: improvement over always predicting the mean. 1 = perfect, 0 = same as the mean (the `DummyRegressor`), < 0 = worse than the mean. It has no units, so it can be compared across problems.
- Raw errors can't simply be averaged because positive and negative errors cancel out.

### Why `neg_` scorers?
- scikit-learn always treats **bigger as better**, so error metrics are negated (`neg_mean_squared_error`).
- Closer to 0 is better.

### Overfitting vs underfitting
- **SVC (balanced)**: train recall 0.929 vs validation 0.711, which is overfitting. Next step: tune `C` and `gamma` with `RandomizedSearchCV`.
- **KNN**: train R² 0.795 vs validation 0.689, a small amount of overfitting. Next step: tune `n_neighbors`.
- **Ridge**: low R² even on train (0.606), which is underfitting. Next step: tune `alpha` or add features.
- Reducing overfitting mainly **closes the gap**, often by lowering the train score, not by raising validation to the train level.

### Python gotchas
- A function needs an explicit `return`. Only the last line of a notebook cell is displayed automatically.
- Use the function's parameters (`X`, `y`), not global variables (`X_train`).
- A function that calls itself with no stopping point causes a `RecursionError`.
