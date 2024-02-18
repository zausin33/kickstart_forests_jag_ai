import numpy as np
import optuna
import config
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    forests_data = pd.read_csv(config.csv_data_path('RtmSimulation_kickstart.csv'), index_col= 0)

    # define target and features
    X = forests_data.drop(['lai'], axis=1)
    y = forests_data['lai']
    print('Xshape: \n{}'.format(X.shape))

    """
        Train-Test Split
        Train-test split is performed with a 60-20-20 ratio.
        Feature engineering steps are applied separately to the train test and validation sets to avoid data leakage.
        Shuffle is left to True to ensure that all sets have the same distribution.
        To check this, we plot the distribution of the target variable.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Run some plots
    sns.kdeplot(y_train, label='y_train')
    sns.kdeplot(y_val, label='y_val')
    sns.kdeplot(y_test, label='y_test')
    plt.legend()
    plt.show()

    """
    Feature Engineering
        We fill missing values with the median of each column.
        We perform PCA on the wavelength features to reduce the dimensionality. We retain 95% of the variance.
        Categories are encoded with OrdinalEncoder and then passed to LightGBM as categorical features. LightGBM will handle the encoding internally.
        Numerical features are scaled with StandardScaler.
    Explaining Feature Selection
        We keep the treeSpecies and wetness features and only perform dimensionality reduction on the wavelength features.
        This is because the wavelength features are highly correlated and we want to reduce the dimensionality of the dataset.
    """

    # Identify categorical and numerical columns
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_cols = [col for col in X.columns if col not in categorical_cols + ['id', 'lai']]

    # Fill missing values
    X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].median())
    X_val[numerical_cols] = X_val[numerical_cols].fillna(X_train[numerical_cols].median())
    X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train[numerical_cols].median())

    # feature selection and expansion methods
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train.iloc[:, 2:])
    X_val_pca = pca.transform(X_val.iloc[:, 2:])
    X_test_pca = pca.transform(X_test.iloc[:, 2:])

    # Convert the PCA results into a DataFrame
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PCA_{i}' for i in range(X_train_pca.shape[1])],
                                  index=X_train.index)
    X_val_pca_df = pd.DataFrame(X_val_pca, columns=[f'PCA_{i}' for i in range(X_val_pca.shape[1])], index=X_val.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PCA_{i}' for i in range(X_test_pca.shape[1])],
                                 index=X_test.index)

    X_train = pd.concat([X_train.iloc[:, :2], X_train_pca_df], axis=1)
    X_val = pd.concat([X_val.iloc[:, :2], X_val_pca_df], axis=1)
    X_test = pd.concat([X_test.iloc[:, :2], X_test_pca_df], axis=1)

    # Identify new categorical and numerical columns
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols + ['id', 'lai']]

    print(X_train)

    # encode categorical features
    ordinal_encoder = OrdinalEncoder()
    X_train[categorical_cols] = ordinal_encoder.fit_transform(X_train[categorical_cols])
    X_val[categorical_cols] = ordinal_encoder.transform(X_val[categorical_cols])
    X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols])

    # scale numerical features
    scl = StandardScaler()
    X_train[numerical_cols] = scl.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = scl.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scl.transform(X_test[numerical_cols])

    # modelling
        # lightGBM
    gbm_model = LGBMRegressor(random_state=42, verbose=-1)
    gbm_model.fit(X_train, y_train, categorical_feature=set(categorical_cols))

    y_pred_train = gbm_model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred_train)

    print("Train MSE:", mse, "Train r2:", r2_score(y_train, y_pred_train))

    y_pred_val = gbm_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)

    print("Validation MSE:", mse, "Validation r2:", r2_score(y_val, y_pred_val))

    y_pred_test = gbm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)

    print("Test MSE:", mse, "Test r2:", r2_score(y_test, y_pred_test))

    # hyperparameters optimization
    def param_bounds(trial: optuna.Trial):
        return {
            # Sample an integer between 10 and 100
            "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 300, 400, 500, 700, 1000]),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 10, 100),
            # Sample a categorical value from the list provided
            "objective": trial.suggest_categorical(
                "objective", ["regression", "regression_l1", "huber"]
            ),
            "random_state": [42],
            # Sample from a uniform distribution between 0.3 and 1.0
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            # Sample from a uniform distribution between 0 and 10
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 100),
            # Sample from a uniform distribution between 0 and 10
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 100),
        }

    def objective(trial: optuna.Trial):
        gbm_model = LGBMRegressor(verbose=-1)
        params = param_bounds(trial)
        gbm_model.set_params(**params)

        gbm_model.fit(X_train, y_train, categorical_feature=set(categorical_cols))

        return gbm_model.score(X_val, y_val)


    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)
    # Create a study
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Start the optimization run
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

    bo_search_trials = study.trials_dataframe()
    best_params = study.best_params
    best_score = study.best_value
    print(bo_search_trials.sort_values("value").head())
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    # train with best parameters
    gbm_model = LGBMRegressor(random_state=42, verbose=-1, **best_params)
    start = time.time()
    gbm_model.fit(X_train, y_train, categorical_feature=set(categorical_cols))
    end = time.time()
    print(f"Training time: {end - start} seconds")

    start = time.time()
    y_pred_train = gbm_model.predict(X_train)
    end = time.time()
    mse = mean_squared_error(y_train, y_pred_train)
    print("Train MSE:", mse, "Train r2:", r2_score(y_train, y_pred_train), f"Prediction time: {end - start} seconds")

    start = time.time()
    y_pred_val = gbm_model.predict(X_val)
    end = time.time()
    mse = mean_squared_error(y_val, y_pred_val)
    print("Validation MSE:", mse, "Validation r2:", r2_score(y_val, y_pred_val),
          f"Prediction time: {end - start} seconds")

    start = time.time()
    y_pred_test = gbm_model.predict(X_test)
    end = time.time()
    mse = mean_squared_error(y_test, y_pred_test)
    print("Test MSE:", mse, "Test r2:", r2_score(y_test, y_pred_test), f"Prediction time: {end - start} seconds")

    """
    Cross validation
    We concatenate the train and validation sets to perform cross-validation on the whole dataset.
    """
    gbm_model = LGBMRegressor(random_state=42, verbose=-1, **best_params)

    X_train_cross_val, y_train_cross_val = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])

    # Define the cross-validation strategy
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation for R² score
    r2_scores = cross_val_score(gbm_model, X_train_cross_val, y_train_cross_val, cv=cv_strategy,
                                fit_params={'categorical_feature': set(categorical_cols)})

    # Calculate mean and standard deviation of R² scores
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)

    # Perform cross-validation for MSE
    mse_scores = cross_val_score(gbm_model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv_strategy,
                                 fit_params={'categorical_feature': set(categorical_cols)})

    # Calculate mean and standard deviation of MSE scores
    mse_mean = np.mean(-mse_scores)  # Negate the scores back to positive
    mse_std = np.std(-mse_scores)  # Negate the scores back to positive

    print(f"Mean cross-validation R²: {r2_mean:.3f} +/- {r2_std:.3f}")
    print(f"Mean cross-validation MSE: {mse_mean:.3f} +/- {mse_std:.3f}")

    """
    Final evaluation
    We train the model on the whole dataset and evaluate it on the test set, to get the final performance metrics
    """
    gbm_model.fit(X_train_cross_val, y_train_cross_val, categorical_feature=set(categorical_cols))
    y_pred_test = gbm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)

    print("Test MSE:", mse, "Test r2:", r2_score(y_test, y_pred_test))

    """
    Explainability
    Gradient boosting models have a feature_importances_ attribute that can be used to get the relative importance of 
    each feature. When using PCA, the interpretability is more limited, as the features are linear 
    combinations of the original features.
    """
    feat_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": gbm_model.feature_importances_.ravel(),
        }
    )

    feat_df["_abs_imp"] = np.abs(feat_df.importance)
    feat_df = feat_df.sort_values("_abs_imp", ascending=False).drop(
        columns="_abs_imp"
    )

    feat_df = feat_df.sort_values(by="importance", ascending=False).head(15)
    feat_df.plot(x="feature", y="importance", kind="bar", color="blue", )