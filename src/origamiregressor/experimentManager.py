# -------------------------------------------------- #
# Functions for running the DNA regression 
# experiments
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import os
import re
import json
import shutil
import joblib
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from origamiregressor.experimentVariables import SCALER_MODELS

from pathlib import Path
from datetime import datetime
from typing import Any, List


# -------------------------------------------------------------------
# -- func

def is_standardscaler_fitted(scaler):
    return hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_')


def is_scaler_fitted(scaler):
    # Common attributes for StandardScaler and MinMaxScaler after fitting
    common_fitted_attributes = ['scale_', 'min_', 'mean_', 'data_min_']
    
    # Check if any of these attributes exist in the scaler object
    return any(hasattr(scaler, attr) for attr in common_fitted_attributes)


def data_scaler(df: pd.DataFrame, 
                scaler: Any,
                held_out_columns: List[str]):

    df_copy = df.copy()
    df_held_out = df[held_out_columns]
    df_copy.drop(columns=held_out_columns, axis=1, inplace=True)
    df_copy_columns = df_copy.columns

    # -
    if not is_scaler_fitted(scaler=scaler):
        print('Scaler not trained ..\nTraining')
        scaler_model = scaler()
        # scaler_model.fit(df_copy)
        scaled_data = scaler_model.fit_transform(df_copy)
        scaled_df = pd.DataFrame(data=scaled_data, columns=df_copy_columns)
        # problem here
        return pd.concat([scaled_df.reset_index(drop=True), 
                          df_held_out.reset_index(drop=True)], axis=1), scaler_model
    
    elif is_scaler_fitted(scaler=scaler):
        print('Scaler trained, applying scaling')
        scaled_data = scaler.transform(df_copy)
        scaled_df = pd.DataFrame(data=scaled_data, columns=df_copy_columns)

        return pd.concat([scaled_df.reset_index(drop=True), 
                          df_held_out.reset_index(drop=True)], axis=1)


def save_to_json(dictonary: dict, fout_name: str) -> None:
    timestamp = datetime.now().strftime('%b_%d_%Y-%H:%M:%S')
    dictonary['execution'] = timestamp
    with open(fout_name+'.json', 'w', encoding='utf-8') as f:
        json.dump(dictonary, 
                  f, 
                  ensure_ascii=False, 
                  indent=4)
        

def combine_means_stds(means: List[float], stds: List[float], n_folds: int=5):
    # Calculate the combined mean
    combined_mean = np.mean(means)
    
    # Calculate the within-experiment variance
    within_variance = np.mean([std**2 for std in stds])
    
    # Calculate the between-experiment variance
    between_variance = np.var(means, ddof=1)
    
    # Total variance (accounting for n_folds)
    total_variance = within_variance / n_folds + between_variance
    
    # Combined standard deviation
    combined_std = np.sqrt(total_variance)
    
    return combined_mean, combined_std        

# -------------------------------------------------------------------
# -- Pre process pipeline

def pre_process_pipeline(data_path: str,
                       output_dir: str,
                       scaler: str,
                       input_variables: List[str],
                       heldout_scaler_colums: List[str],
                       target_variable: List[str],
                       test_split: float=.3,
                       repetition: int=None):
    
    save_to_json(dictonary=locals(), 
                 fout_name=output_dir+f'preprocess_config_rep{repetition}')

    print('# \t\t Init Pre-Processing \n')
    print(f'Reading: {data_path}')
    df = pd.read_csv(data_path)

    print(f'Input variables: {input_variables}')
    X = df[input_variables]

    print(f'Target variable: {target_variable}')
    y = df[target_variable]

    print('Creating [train/test]/[validation] split ...')
    print(f'Training/Test split: {(1-test_split)*100}/{test_split*100}')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_split, 
                                                        random_state=None)

    X_train_scaled, scaler_model = data_scaler(df=X_train,
                                               scaler=SCALER_MODELS[scaler],
                                               held_out_columns=heldout_scaler_colums)
    
    print(f'Using scaler: {scaler_model}')

    X_test_scaled = data_scaler(df=X_test, 
                                scaler=scaler_model,
                                held_out_columns=heldout_scaler_colums)

    joblib.dump(scaler_model, output_dir + f'{scaler}_rep{repetition}.pkl')

    # return the X train scaled data and the X test scaled data using
    # the train fit scaler
    # the y is not scaled
    return X_train_scaled, y_train, X_test_scaled, y_test


# -------------------------------------------------------------------
# -- Point sampling

def sampling_fps(X: np.ndarray, 
                 n: int, 
                 start_idx: int=None, 
                 return_distD: bool=False) -> List[int] | np.ndarray:

    if isinstance(X, pd.DataFrame):
        X = np.array(X)

    # init the output quantities
    fps_ndxs = np.zeros(n, dtype=int)
    distD = np.zeros(n)

    # check for starting index
    if not start_idx:
        # the b limits has to be decreaed because of python indexing
        # start from zero
        start_idx = random.randint(a=0, b=X.shape[0]-1)
    # inset the first idx of the sampling method
    fps_ndxs[0] = start_idx

    # compute the distance from selected point vs all the others
    dist1 = np.linalg.norm(X - X[start_idx], axis=1)

    # loop over the distances from selected starter
    # to find the other n points
    for i in range(1, n):
        # get and store the index for the max dist from the point chosen
        fps_ndxs[i] = np.argmax(dist1)
        distD[i - 1] = np.amax(dist1)

        # compute the dists from the newly selected point
        dist2 = np.linalg.norm(X - X[fps_ndxs[i]], axis=1)
        # takes the min from the two arrays dist1 2
        dist1 = np.minimum(dist1, dist2)

        # little stopping condition
        if np.abs(dist1).max() == 0.0:
            print(f"Only {i} iteration possible")
            return fps_ndxs[:i], distD[:i]
        
    if return_distD:
        return list(fps_ndxs), distD
    else:
        return list(fps_ndxs)
    
    
# -------------------------------------------------- #
# --- file and folders handling

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_files_from(folder, ew=None, sw=None, verbose=True):
    """Simple function to select files from a location
    with possible restraints.
    folder : file str location
    ew : "endswith" str selection
    sw : "startswith" str selection
    """
    file_list = list()
    # file selection following the constraints
    for entry in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, entry)):
            if ew:
                if entry.endswith(ew):
                    file_list.append(entry)
            elif sw:
                if entry.startswith(sw):
                    file_list.append(entry)
            else:
                file_list.append(entry)
    # sorting of the files :)
    file_list.sort(key=natural_keys)
    if verbose:
        print(f"Files:\n{file_list}, ({len(file_list)})")
    return file_list


def create_strict_folder(path_str: str, overwrite: bool = False) -> None:
    """
    Create a folder from a path string, with optional overwrite.
    
    Args:
        path_str: str - Path to the folder to create
        overwrite: bool - If True, allows overwriting existing folder (default: False)
    """
    path = Path(path_str)
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Directory '{path}' already exists.")
    path.mkdir(parents=True)

# -------------------------------------------------- #
# --- points grid

def create_meshgrid(arrays: List[np.ndarray]) -> np.ndarray:
    grid = np.meshgrid(*arrays)
    return np.asarray(grid)


def recover_points(meshgrid: np.ndarray) -> np.ndarray:
    num_dim = len(meshgrid)
    num_points = meshgrid[0].size
    positions = np.empty((num_dim, num_points), dtype=meshgrid[0].dtype)

    for i in range(num_dim):
        positions[i] = meshgrid[i].ravel()
    
    return positions
