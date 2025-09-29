# Regression experiment results

Each folder refers to a model trained on the same dataset:
-   `./GPR_1k_noise/` $\rightarrow$ **G**aussian **P**rocess **R**egressor, 1 RBF Kernel + WhiteNoise Kernel
-   `./GPR_2k_noise/` $\rightarrow$ **G**aussian **P**rocess **R**egressor, 2 RBF Kernel (linear combination) + WhiteNoise Kernel
-   `./SVR/` $\rightarrow$ **S**upport **V**ector **R**egressor
-   `./RandForest/` $\rightarrow$ **Rand**om **Forest** regressor
-   `./XGBoost/` $\rightarrow$ e**X**treeme **G**radient **B**oosting regressor
-   `./Consensus_model_A` $\rightarrow$ Consensus model built merging the predictive power of three models: RF, XGBoost, and GRP2k
-   `./Consensus_model_B` $\rightarrow$ $\rightarrow$ Consensus model built merging the predictive power of the two best performing models: RF, and GRP2k

The specific implementation and (hyper)parameters for all the models is detailed in the paper methods section.

Each folder contains the results for the application of the regression pipeline to the training dataset using a specific model.
The generated files are listed below.

Files organization:
------------
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

[*] Only present when the method outputs an estimation of the uncertainty, i.e., GPR1k/GPR2k.

[+] Only present for the three best performing methods, i.e., RF, XGBoost, GPR2k_noise

