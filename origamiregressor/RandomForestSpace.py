# -------------------------------------------------------------------
# -- Random Forest searchspace
from sklearn.ensemble import RandomForestRegressor

PARAM_GRID = [
    {
    'n_estimators': [25, 50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
    }
]

# -- GPR search space dict

SearchSpaceCollection = {
    'Default' : PARAM_GRID,
}

Regressor = RandomForestRegressor()