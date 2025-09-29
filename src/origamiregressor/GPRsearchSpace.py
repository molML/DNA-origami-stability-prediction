# -------------------------------------------------------------------
# -- GPR searchspace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

PARAM_GRID_3k = [
    {
        'kernel' : [
            C(1.0) * RBF(lc1) + C(1.0) * RBF(lc2) + C(1.0) * RBF(lc3) + WhiteKernel(noise_level=noise)
            for lc1 in [0.1, 0.01, 0.001, 0.0001]
            for lc2 in [0.1, 1.0, 5.0]
            for lc3 in [10.0, 25.0, 50.0]
            for noise in [1.0, .5, .25, 0.1]
        ],
        'alpha' : [1e-3, 1e-5, 1e-7, 1e-9]
    },
]

PARAM_GRID_3k_noNoise = [
    {
        'kernel' : [
            C(1.0) * RBF(lc1) + C(1.0) * RBF(lc2) + C(1.0) * RBF(lc3)
            for lc1 in [0.1, 0.01, 0.001, 0.0001]
            for lc2 in [0.1, 1.0, 5.0]
            for lc3 in [10.0, 25.0, 50.0]
        ],
        'alpha' : [1e-3, 1e-5, 1e-7, 1e-9]
    },
]

PARAM_GRID_2k = [
    {
        'kernel' : [
            C(1.0) * RBF(lc1) + C(1.0) * RBF(lc2) + WhiteKernel(noise_level=noise)
            for lc1 in [0.01, 0.001, 0.0001]
            for lc2 in [0.1, 1.0, 5.0, 10.]
            for noise in [1.0, .5, .25, 0.1]
        ],
        'alpha' : [1e-3, 1e-5, 1e-7, 1e-9]
    },
]

PARAM_GRID_2k_noNoise = [
    {
        'kernel' : [
            C(1.0) * RBF(lc1) + C(1.0) * RBF(lc2)
            for lc1 in [0.01, 0.001, 0.0001]
            for lc2 in [0.1, 1.0, 5.0, 10.]
        ],
        'alpha' : [1e-3, 1e-5, 1e-7, 1e-9]
    },
]

PARAM_GRID_1k = [
    {
        'kernel' : [
            C(1.0) * RBF(lc1) + WhiteKernel(noise_level=noise)
            for lc1 in [10., 5., 1., 0.1, 0.01, 0.001]
            for noise in [1.0, .5, .25, 0.1]
        ],
        'alpha' : [1e-3, 1e-5, 1e-7, 1e-9]
    },
]

PARAM_GRID_1k_noNoise = [
    {
        'kernel' : [
            C(1.0) * RBF(lc1)
            for lc1 in [10., 5., 1., 0.1, 0.01]
        ],
        'alpha' : [1e-3, 1e-5, 1e-7, 1e-9]
    },
]

# -- GPR search space dict

SearchSpaceCollection = {
    '1k_noise' : PARAM_GRID_1k,
    '1k': PARAM_GRID_1k_noNoise,
    '2k_noise' : PARAM_GRID_2k,
    '2k': PARAM_GRID_2k_noNoise,
    '3k_noise' : PARAM_GRID_3k,
    '3k': PARAM_GRID_3k_noNoise,
    'None': None
}

Regressor = GaussianProcessRegressor(n_restarts_optimizer=5)