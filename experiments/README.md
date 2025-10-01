<!-- Regression experiment results-->
<h1 id="Title">Regression experiment results</h1>

Each folder refers to a model trained on the same dataset:
-   `./GPR_1k_noise/` $\rightarrow$ **G**aussian **P**rocess **R**egressor, 1 RBF Kernel + WhiteNoise Kernel
-   `./GPR_2k_noise/` $\rightarrow$ **G**aussian **P**rocess **R**egressor, 2 RBF Kernel (linear combination) + WhiteNoise Kernel
-   `./SVR/` $\rightarrow$ **S**upport **V**ector **R**egressor
-   `./RandForest/` $\rightarrow$ **Rand**om **Forest** regressor
-   `./XGBoost/` $\rightarrow$ e**X**treeme **G**radient **B**oosting regressor
-   `./Consensus_model_A` $\rightarrow$ Consensus model built merging the predictive power of three models: RF, XGBoost, and GRP2k
-   `./Consensus_model_B` $\rightarrow$ $\rightarrow$ Consensus model built merging the predictive power of the two best performing models: RF, and GRP2k

The specific implementation and (hyper)parameters for all the models is detailed in the paper's methods section.
Each folder contains the results for the application of the regression pipeline to the training dataset using a specific model.
The generated files are listed below.

<!-- Files organization-->
<h2 id="filesoraganization">Files organization</h2>

    ./$model_name/
    ├── best_model.pkl
    ├── gridsearchCV_results.csv
    ├── RMSE_cvscores_rep*.npy
    ├── MinMaxScaler.pkl
    ├── X_train_rep*.npy
    ├── X_test_rep*.npy
    ├── y_test_rep*.npy
    ├── y_predict_rep*.npy
    └── y_predict_std_rep*.npy [*]
        0.results_analysis/
        1.validation_on_heldout/ [+]
        ├── y_predict_heldout.npy
        ├── y_predict_std_heldout.npy [*]
        ├── y_predict_newexperiments.npy
        ├── y_predict_std_newexperiments.npy [*]
---

[*] Only present when the method outputs an estimation of the uncertainty, i.e., GPR1k/GPR2k.\
[+] Only present for the three best performing methods, i.e., RF, XGBoost, GPR2k_noise

The files represents all the outputs obtained from the current version of the script.
The label `_train` and `_test` represent the corresponding split taken from the training data.
The label `_predict` references the prediction on the test split using the best (hyper-param tuned) model.
All the other files are considered self-explainatory.

<!-- Running Experiments-->
<h2 id="runexperiment">Running Experiments</h2>

A specific experiment can be run using the script stored in the `./script/` folder.
The script takes two arguments that fully define the experiment.

The first input file must contain the information about the pre-processing to be done on the dataset:
```yaml
# Path for the training data
data_path: 'raw_data/raw_data_origami_TRAIN.csv'

# Output dir name
output_dir: 'Test_model'
overwrite_output_dir: true

# Variables
scaler: 'MinMaxScaler'
input_variables: ['Temperature', 'MgCl2_concentration', 'pH_f', 'Incubation_time', 'DNase_I_concentration', 'Rectangles', 'Rods', 'Spheres']
heldout_scaler_colums: ['Rectangles', 'Rods', 'Spheres']
target_variable: ['Diffusion_Coefficient']
test_split: 0.3
```

The second input file must contai inforation regardin the experiment model:
```yaml
# ['GPR', 'RandomForest', 'SVR', 'XGBoost']
regressor_model: 'RandomForest'

# Unique to each regressors
# GPR : ['1k', '1k_noise', '2k', '2k_noise', '3k', '3k_noise']
# SVR / RandomForest / XGBoost: 'Defaut'
parameter_searchspace: 'Defaut'

# Number of cross validations
ncv: 5

# Repetitions
repetitions: 5
```

From the script directory run:
```bash
python regressor_pipeline.py -ppc preprocess_config.yaml -exc experiment_config.yaml
```

If the command run correctly a series of output statements will appear showcasing the model performances.\
Example:
```
Preprocessing config:
{'data_path': 'raw_data/raw_data_origami_TRAIN.csv',
 'heldout_scaler_colums': ['Rectangles', 'Rods', 'Spheres'],
 'input_variables': ['Temperature',
                     'MgCl2_concentration',
                     'pH_f',
                     'Incubation_time',
                     'DNase_I_concentration',
                     'Rectangles',
                     'Rods',
                     'Spheres'],
 'output_dir': 'Test_model',
 'overwrite_output_dir': True,
 'scaler': 'MinMaxScaler',
 'target_variable': ['Diffusion_Coefficient'],
 'test_split': 0.3}

Experiment config:
{'ncv': 5,
 'parameter_searchspace': 'Defaut',
 'regressor_model': 'RandomForest',
 'repetitions': 5}
Regressor: RandomForestRegressor()
Output dir: path/to/DNA-origami-stability-prediction/experiments/Test_model/
Using training data from: path/to/DNA-origami-stability-prediction/datasets/raw_data/raw_data_origami_TRAIN.csv

Repetition: 1
# 		 Init Pre-Processing 

Reading: /home/andreag/Work/1.main_project/molML_git_repo/DNA-origami-stability-prediction/datasets/raw_data/raw_data_origami_TRAIN.csv
Input variables: ['Temperature', 'MgCl2_concentration', 'pH_f', 'Incubation_time', 'DNase_I_concentration', 'Rectangles', 'Rods', 'Spheres']
Target variable: ['Diffusion_Coefficient']
Creating [train/test]/[validation] split ...
Training/Test split: 70.0/30.0
Scaler not trained ..
Training
Using scaler: MinMaxScaler()
Scaler trained, applying scaling
Best model not computed ...
Parameter space:
[{'n_estimators': [25, 50, 100, 200], 'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 4, 8]}]
Fitting 5 folds for each of 320 candidates, totalling 1600 fits
best estimator: RandomForestRegressor(max_depth=20, n_estimators=200)
best params: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
best score: 0.7117325205636654
best index: 131
Found best model
RandomForestRegressor(max_depth=20, n_estimators=200)
Saving it to folder: /home/andreag/Work/1.main_project/molML_git_repo/DNA-origami-stability-prediction/experiments/Test_model/
Computing CV: scores on best model
CV scores: [0.68990802 0.76609411 0.76370315 0.87610472 0.71454181]
0.76 RMSE with a standard deviation of 0.06
Rep0 MSE: 0.4107277905954231
Rep0 RMSE: 0.6408804807414742
Rep0 R2: 0.7644073249189764

Repetition: 2
# 		 Init Pre-Processing 
...
...
...
```

With the current provided setup of config files a new folder called `Test_model` will be created in the `/experimnet` folder containing the outputs of this test experiment.


<!-- Generating plots -->
<h2 id="genplots">Generating plots</h2>

To generate the plots that summarize the training results (train, validation, test steps) a script is provided.
The script can be run by adding to the flag `-model` the name of the model's folder as saved in the experiments folder.
```bash
python results_analysis.py -model Test_model
```
