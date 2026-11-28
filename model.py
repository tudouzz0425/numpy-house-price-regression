"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X_clean = np.array(X, dtype=float, copy=True)

    for col in range(X_clean.shape[1]):
        column = X_clean[:, col]
        missing = np.isnan(column)
        observed = column[~missing]

        if observed.size > 0:
            fill_value = np.mean(observed)
        else:
            fill_value = 0.0

        X_clean[missing, col] = fill_value

    return X_clean

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    X = np.asarray(X, dtype=float)

    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return lower, upper

# Step 3 - clip_columns
import numpy as np
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
import numpy as np
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    return numerator / (denominator + eps)

# Step 5 - append_column
import numpy as np

def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack((X,col))

# Step 6 - one_hot_encode
import numpy as np
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    categories, inverse = np.unique(labels, return_inverse=True)

    result = np.zeros((len(inverse), len(categories)), dtype=float)
    result[np.arange(len(inverse)), inverse] = 1.0

    return result

# Step 7 - fit_standardizer
import numpy as np

def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0

    return mean, std

# Step 8 - apply_standardizer
import numpy as np

def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    return (X - mean) / std

# Step 9 - add_bias_column
import numpy as np

def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    X = np.asarray(X, dtype=float)
    ones = np.ones(X.shape[0])
    return np.column_stack((ones,X))

# Step 10 - make_shuffled_indices
import numpy as np

def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.default_rng(seed)
    return rng.permutation(n_samples)

# Step 11 - partition_indices
import numpy as np

def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    indices = np.asarray(indices, dtype=int)
    n = len(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx,val_idx,test_idx

# Step 12 - subset_xy
import numpy as np

def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    return X[indices],y[indices]

# Step 13 - ols_fit
import numpy as np

def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X@theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    n = len(y_pred)
    mae = sum(abs(y_pred-y_true))/n
    
    return mae

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    n = len(y_pred)
    rmse = (sum((y_pred-y_true)**2)/n)**0.5

    return rmse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...

    ss_res = sum((y_true - y_pred) ** 2)
    ss_tot = sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r = y_true - y_pred

    return {
        'mean': float(np.mean(r)),
        'std': float(np.std(r, ddof=0)),
        'median_abs': float(np.median(np.abs(r))),
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X_imputed = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_imputed, k=iqr_k)
    X_clean = clip_columns(X_imputed, lower, upper)

    return X_clean

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

