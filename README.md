# Car Price Prediction with Neural Networks

Predicting used-car prices from the **London Cars 2014** dataset (9,080 listings) using feed-forward neural networks, framed both as a regression problem (predicting the exact price) and a classification problem (predicting a price band). Built with PyTorch and PyTorch Lightning, the project covers the full pipeline: preprocessing, model design, regularisation, cross-validated hyperparameter tuning, and a comparison against classical machine-learning baselines.

## Overview

The dataset contains 9,080 rows and 11 columns (Make, Model, Year, Mileage, Body Style, Engine, Transmission, colours, etc.). The same underlying problem is approached two ways:

- **Regression** — predict the numerical price directly.
- **Classification** — bucket prices into low / medium / high bands and predict the band.

## What's inside

**Preprocessing**
- Label encoding for the high-cardinality `Model` column (1,060 unique values) and one-hot encoding for the remaining categoricals, with `drop_first=True` to avoid the dummy-variable trap.
- Feature standardisation with `StandardScaler`, fit on training data only to prevent leakage.
- A 70/30 split, stratified for the classification task to preserve the heavy class imbalance.
- **Duplicate detection:** around 10% of the rows turned out to be exact duplicates. Because duplicates leak between train and test and inflate scores, the whole pipeline was rerun on a deduplicated dataset, which is used as the primary basis for all reported results.

**Regression network**
- Architecture: `Input → Dense(50, ReLU) → Dense(20, ReLU) → Dense(1)`, MSE loss, RMSprop.
- A baseline version (no regularisation) and an improved version with dropout + L2 weight decay.
- Hyperparameters (dropout rate and L2 strength) chosen via 5-fold cross-validated grid search.

**Classification networks**
- A **shallow** network (one wide hidden layer) and a **deep** network (four tapered hidden layers).
- 5-fold cross-validation used to select the activation function (ReLU, LeakyReLU, Sigmoid, Tanh, Swish) and regularisation strength for each.
- Evaluation with accuracy, precision, recall, F1 (macro-averaged for fairness under imbalance), and confusion matrices.

**Comparison with classical ML**
Results are benchmarked against tuned Linear/Ridge Regression, Random Forest, and Logistic Regression models on the same data.

## Key findings

- **Regularisation is essential, not optional.** When duplicates were removed, the unregularised regression model's R² dropped by 0.15, while the regularised model dropped by only 0.02 — strong evidence that the baseline was memorising duplicated rows rather than generalising.
- **Depth gave only marginal gains.** The deep classifier (≈31k parameters) only slightly outperformed the shallow one (≈14k parameters), consistent with the idea that extra depth offers limited benefit on moderate-sized tabular data.
- **Neural networks handled class imbalance far better than classical models.** On the minority "high" price band, the networks achieved ~70% recall versus 18–33% for tuned Logistic Regression and Random Forest.
- **Cross-validation validated default choices** while revealing small available improvements — the best activation/regularisation combinations were within ~1% of the obvious defaults.

## Tech stack

- Python, pandas, NumPy
- PyTorch + PyTorch Lightning (model definition and training)
- scikit-learn (preprocessing, cross-validation, metrics)
- matplotlib for training curves and confusion matrices
- Jupyter Notebook

## Running it

```bash
pip install torch pytorch-lightning scikit-learn pandas numpy matplotlib jupyter
jupyter notebook
```

Open the notebook and run the cells in order. Note that the cross-validation grid searches train many models and can take a while on CPU; a GPU (CUDA or Apple MPS) speeds this up considerably.

## Repository contents

- `*_Code.ipynb` — main notebook (regression + classification, with deduplication)
- `Part_B_Report.md` — written report covering methodology, results, and discussion

## Note on data

The `LondonCars2014.csv` dataset is not included in this repository. [Add a line on where you obtained it, or remove this section if you decide to include the file.]
