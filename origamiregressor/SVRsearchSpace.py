# -------------------------------------------------------------------
# -- GPR searchspace
from sklearn.svm import SVR

PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

# -- GPR search space dict

SearchSpaceCollection = {
    'Default' : PARAM_GRID,
}

Regressor = SVR()