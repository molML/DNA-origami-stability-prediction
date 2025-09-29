# -------------------------------------------------- #
# Experiment varaibles/definitions for running  
# the DNA regression experiments
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# -- scaler

SCALER_MODELS = {
    'StandardScaler': StandardScaler,
    'Normalizer': Normalizer,
    'MinMaxScaler': MinMaxScaler,
}


# -------------------------------------------------------------------
# -- scoring func

SCORING_FUNCS = {
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),

    'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
    
    'NRMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(np.concatenate([y_true,y_pred])) - np.min(np.concatenate([y_true,y_pred]))), 
                        greater_is_better=False),

    'MAE': make_scorer(mean_absolute_error, greater_is_better=False),

    'R2': 'r2'
}

