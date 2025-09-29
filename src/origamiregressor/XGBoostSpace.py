# -------------------------------------------------------------------
# -- GPR searchspace
from xgboost import XGBRegressor

PARAM_GRID = [
    {
        'n_estimators': [25, 50, 100, 200],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.1, 0.2, 0.5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0]
    }
]

# -- GPR search space dict

SearchSpaceCollection = {
    'Default' : PARAM_GRID,
}

Regressor = XGBRegressor()