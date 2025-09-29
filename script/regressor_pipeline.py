# -------------------------------------------------- #
# Script for running the prediction routine
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import yaml
import joblib
import pprint
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score

from origamiregressor.experimentManager import pre_process_pipeline, create_strict_folder
from origamiregressor.experimentVariables import SCORING_FUNCS

from pathlib import Path
from typing import Callable, Any

# -------------------------------------------------------------------
# -- MAIN

def regression_experiment(pp_config_dict: dict,
                          regressor_model: Callable,
                          parameter_searchspace: Any,
                          ncv: int,
                          repetitions: int):

    # extract the output dir from the preprocessin
    BASE_EXPERIMENT_DIR = str(Path(__file__).resolve().parents[1]) + '/experiments/'
    output_dir_path = pp_config_dict['output_dir']
    overwrite_output_dir = pp_config_dict.get('overwrite_output_dir', False)
    output_dir = create_strict_folder(str(BASE_EXPERIMENT_DIR / output_dir_path), overwrite=overwrite_output_dir)

    # set up the dataframes with the metrics
    df_metrics = pd.DataFrame(columns=['MSE', 'RMSE', 'r2'])

    for rep in range(repetitions):
        print(f'Repetition: {rep+1}')

        X_train_scaled, y_train, X_test_scaled, y_test = pre_process_pipeline(**pp_config_dict, repetition=rep)

        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)

        # print(X_train_scaled.head(20), X_test_scaled.head(20))

        if rep == 0 and parameter_searchspace is not None:
            print('Best model not computed ...')

            rgr = regressor_model
            print('Parameter space:')
            print(parameter_searchspace)
        
            grid_search_model = GridSearchCV(estimator=rgr,
                                             param_grid=parameter_searchspace,
                                             scoring=SCORING_FUNCS['RMSE'],
                                             cv=ncv,
                                             n_jobs=-1,
                                             verbose=1)
    
            grid_search_model.fit(X_train_scaled, y_train)

            grid_search_df = pd.DataFrame(grid_search_model.cv_results_)
            grid_search_df.to_csv(output_dir+'gridsearchCV_results.csv')
            print(f'best estimator: {grid_search_model.best_estimator_}')
            print(f'best params: {grid_search_model.best_params_}')
            print(f'best score: {grid_search_model.best_score_*-1}')
            print(f'best index: {grid_search_model.best_index_}')
            best_model = grid_search_model.best_estimator_

            print('Found best model')
            print(best_model)
            print(f'Saving it to folder: {output_dir}')
            joblib.dump(best_model, output_dir+'best_model.pkl')

        else:
            best_model = joblib.load(output_dir+'best_model.pkl')
            print(f'Loaded best model\n{best_model}')
        
        print('Computing CV: scores on best model')
        cv = ShuffleSplit(n_splits=ncv, test_size=0.3, random_state=None)
        cv_score = cross_val_score(best_model, 
                                   X_train_scaled, y_train, 
                                   cv=cv, 
                                   scoring=SCORING_FUNCS['RMSE'], 
                                   n_jobs=-1, 
                                   verbose=False)
        print(f'CV scores: {cv_score*-1}')
        print("%0.2f RMSE with a standard deviation of %0.2f" % (-cv_score.mean(), cv_score.std()))
        np.save(output_dir+f'RMSE_cvscores_rep{rep+1}.npy', cv_score*-1)

        best_model.fit(X_train_scaled, y_train)
        try:
            print(f'Rep{rep} best kernel: {best_model.kernel_}')
            y_pred, y_pred_std = best_model.predict(X_test_scaled, return_std=True)
            np.save(output_dir+f'y_predict_std_rep{rep+1}.npy', y_pred_std)
        except:
            y_pred = best_model.predict(X_test_scaled)

        np.save(output_dir+f'y_predict_rep{rep+1}.npy', y_pred)
        np.save(output_dir+f'y_test_rep{rep+1}.npy', y_test)
        np.save(output_dir+f'X_train_rep{rep+1}.npy', X_train_scaled)
        np.save(output_dir+f'X_test_rep{rep+1}.npy', X_test_scaled)

        scores_tmp = {
            'MSE' : mean_squared_error(y_true=y_test, y_pred=y_pred),
            'RMSE' : np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)),
            'r2' : r2_score(y_true=y_test, y_pred=y_pred)
        }

        print(f'Rep{rep} MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred)}')
        print(f'Rep{rep} RMSE: {np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))}')
        print(f'Rep{rep} R2: {r2_score(y_true=y_test, y_pred=y_pred)}')
        df_metrics.loc[rep] = scores_tmp

    df_metrics.to_csv(output_dir+'scores.csv')

    print('# END Regression Experiment')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ppc', dest='PPconfig', type=str, help='Config file.')
    parser.add_argument('-exc', dest='EXPconfig', type=str, help='Config file.')
    args = parser.parse_args()

    pp_config_dict = yaml.load(open(args.PPconfig, 'r'), Loader=yaml.FullLoader)
    print('Preprocessing config:')
    pprint.pprint(pp_config_dict)

    exp_config_dict = yaml.load(open(args.EXPconfig, 'r'), Loader=yaml.FullLoader)

    if exp_config_dict['regressor_model'] == 'GPR':
        import origamiregressor.GPRsearchSpace as expModel
        space_label = exp_config_dict['parameter_searchspace']
        parameter_searchspace = expModel.SearchSpaceCollection[space_label]

    elif exp_config_dict['regressor_model'] == 'SVR':
        import origamiregressor.SVRsearchSpace as expModel
        parameter_searchspace = expModel.SearchSpaceCollection['Default']

    elif exp_config_dict['regressor_model'] == 'RandomForest':
        import origamiregressor.RandomForestSpace as expModel
        parameter_searchspace = expModel.SearchSpaceCollection['Default']

    elif exp_config_dict['regressor_model'] == 'XGBoost':
        import origamiregressor.XGBoostSpace as expModel
        parameter_searchspace = expModel.SearchSpaceCollection['Default']

    else:
        raise ValueError('Regressor not yet implemented!')

    print('\nPreprocessing config:')
    pprint.pprint(exp_config_dict)
    print(f'Regressor: {expModel.Regressor}')
    
    number_of_cv = exp_config_dict['ncv']

    regression_experiment(pp_config_dict=pp_config_dict,
                          regressor_model=expModel.Regressor,
                          parameter_searchspace=parameter_searchspace,
                          ncv=number_of_cv,
                          repetitions=exp_config_dict['repetitions'])