# Yacht Hydrodynamics

## Dataset and Problem
The Yacht Hydrodynamics dataset (UCI / OpenML, 308 samples) contains six hull design
parameters — longitudinal position of the centre of buoyancy, prismatic coefficient,
length-displacement ratio, beam-draught ratio, length-beam ratio, and Froude number —
with residuary resistance per unit weight of displacement as the regression target.
The goal is to predict hydrodynamic resistance from hull geometry, which is useful
early in vessel design before physical testing is feasible.

## Methodology
- Loaded and explored the dataset with descriptive statistics, feature distributions,
  a correlation matrix, and a Froude number vs. resistance scatter plot.
- Preprocessed with an 80/20 train/test split and StandardScaler (fitted on train only).
- Trained and compared nine regression models: Linear Regression, Ridge, Lasso,
  ElasticNet, KNN, Decision Tree, Random Forest, Gradient Boosting, and SVR.
- Evaluated each on RMSE, MAE, R², and 5-fold cross-validated R².
- Saved the best model and scaler with joblib for deployment.

## Key Decisions
- **Gradient Boosting as the final model:** it had the highest CV-R² (0.9980) and
  lowest MAE (0.2623), making it the most generalisable despite marginally higher
  RMSE than the tree models.
- **Cross-validated R² as the tiebreaker:** with only 308 samples, test-set metrics
  alone are sensitive to the split. CV-R² gives a more reliable view of generalisation.
- **StandardScaler fitted on train only:** prevents data leakage into validation and test.
- **joblib over pickle:** more efficient for sklearn estimators, especially tree ensembles.

## Implementation
Nine models were trained under identical conditions using sklearn pipelines where
scaling was required. Tree-based models were trained directly on unscaled data since
they are scale-invariant. All models were evaluated on the held-out test set and via
5-fold cross-validation. The scaler and best model were bundled into a single `.pkl`
file. A standalone `predict.py` handles inference from a feature dictionary, and
`GUI.py` provides a Tkinter interface with labelled input fields and natural value
ranges drawn from the dataset.

## Results

| Model | RMSE | MAE | R² | CV-R² |
|---|---|---|---|---|
| Decision Tree | 0.5392 | 0.3277 | 0.9980 | 0.9921 |
| Random Forest | 0.5440 | 0.3060 | 0.9980 | 0.9941 |
| **Gradient Boosting** | **0.6025** | **0.2623** | **0.9976** | **0.9980** |
| SVR | 2.9625 | 1.5491 | 0.9409 | 0.9226 |
| ElasticNet | 8.0496 | 6.5303 | 0.5639 | 0.6170 |
| Lasso | 8.1838 | 6.7886 | 0.5493 | 0.6158 |
| Ridge | 8.2071 | 6.7847 | 0.5467 | 0.6111 |
| Linear Regression | 8.2221 | 6.8075 | 0.5450 | 0.6090 |
| KNN | 8.7679 | 3.8437 | 0.4826 | 0.5566 |

The Froude number was the dominant predictor, as expected from hydrodynamic theory.
Linear models performed poorly, indicating strong non-linearity in the relationship
between hull geometry and resistance. Tree-based ensembles handled this naturally
without feature engineering.
