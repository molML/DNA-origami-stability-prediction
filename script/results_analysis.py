# Results Analysis Script

import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from origamiregressor.beauty import plot_correlations_A, plot_correlations_B
from origamiregressor.experimentManager import (create_strict_folder, 
                                                get_files_from,
                                                combine_means_stds)


def get_mean_std(arr: np.ndarray) -> tuple[float, float]:
    return np.mean(arr), np.std(arr, ddof=1)


def get_hyperCVscores(report: pd.DataFrame, n_split: int) -> np.ndarray:
    hyper_test_score = []
    for i in range(n_split):
        hyper_test_score.append(report[f'split{i}_test_score'].values[0])
    return np.array(hyper_test_score)


def analyze_results(model_name: str):
    print(f'Analyzing results for model: {model_name}')

    # Configuration parameters
    EXPERIMENT_DIR = str(Path(__file__).resolve().parents[1]) + '/experiments/' + model_name + '/'
    OUTPUT_DIR = EXPERIMENT_DIR + '/0.results_analysis/'
    create_strict_folder(OUTPUT_DIR, overwrite=True)
    print(f'Output directory for analysis: {OUTPUT_DIR}')
    FIG_FILE_NAME = f'{model_name}_outputs'
    FONTSIZE = 8
    XLABEL = 'Experimental'
    YLABEL = 'Predicted'

    # Training step correlation plots
    scores_df = pd.read_csv(EXPERIMENT_DIR + '/scores.csv')
    mse_scores = scores_df['MSE'].to_numpy()
    rmse_scores = scores_df['RMSE'].to_numpy()
    r2_scores = scores_df['r2'].to_numpy()

    # Collect the files
    y_test_file_list = get_files_from(folder=EXPERIMENT_DIR, sw='y_test', verbose=False)
    y_pred_file_list = get_files_from(folder=EXPERIMENT_DIR, sw='y_pred', verbose=False)[:len(y_test_file_list)]
    y_pred_std_file_list = get_files_from(folder=EXPERIMENT_DIR, sw='y_pred', verbose=False)[len(y_test_file_list):]

    if len(y_pred_std_file_list) == 0:
        STD_MODEL = False
    else:
        STD_MODEL = True

    y_test_list = [np.load(EXPERIMENT_DIR+f) for f in y_test_file_list]
    y_pred_list = [np.load(EXPERIMENT_DIR+f) for f in y_pred_file_list]
    if STD_MODEL:
        y_pred_std_list = [np.load(EXPERIMENT_DIR+f) for f in y_pred_std_file_list]
    else:
        y_pred_std_list = [None]*len(y_test_file_list)

    # Plot the correlations
    fig, ax = plot_correlations_A(
        y_test_list=y_test_list, 
        y_pred_list=y_pred_list, 
        y_pred_std_list=y_pred_std_list, 
        Xlabel=XLABEL,
        Ylabel=YLABEL,
        std_model=STD_MODEL,
        font_size=FONTSIZE
    )
    fig.savefig(OUTPUT_DIR + FIG_FILE_NAME + '_1.png', bbox_inches='tight')

    # Collect the scores
    rmse_cvscores = []
    hyper_param_df = pd.read_csv(EXPERIMENT_DIR + 'gridsearchCV_results.csv')
    best_hyperparam_scores = get_hyperCVscores(
        report=hyper_param_df.loc[hyper_param_df['rank_test_score']==1].head(1), 
        n_split=5
        )
    rmse_train_cv_score_files = get_files_from(folder=EXPERIMENT_DIR, sw='RMSE_cvscores', verbose=False)

    rmse_cvscores.append(best_hyperparam_scores*-1)
    for f in rmse_train_cv_score_files:
        rmse_cvscores.append(np.load(EXPERIMENT_DIR+f))

    fig, ax = plot_correlations_B(
        y_test_list=y_test_list, 
        y_pred_list=y_pred_list,
        rmse_scores=rmse_scores,
        rmse_cv_scores=rmse_cvscores,
        Xlabel=XLABEL,
        Ylabel=YLABEL,
        model_name=model_name,
        font_size=FONTSIZE
    )
    fig.savefig(OUTPUT_DIR + FIG_FILE_NAME + '_2.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', dest='model', type=str, help='Model name.')
    args = parser.parse_args()
    
    analyze_results(args.model)

