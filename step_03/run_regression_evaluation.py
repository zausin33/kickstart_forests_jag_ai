import warnings
from typing import Tuple, Optional, Union, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import config

warnings.filterwarnings('ignore', category=UserWarning)

COL_LEAF_AREA_INDEX = 'lai'
COL_ID = 'id'
COL_WETNESS = 'wetness'
COL_TREE_SPECIES = 'treeSpecies'
COLS_SENTINEL = ["Sentinel_2A_492.4", "Sentinel_2A_559.8", "Sentinel_2A_664.6", "Sentinel_2A_704.1",
                 "Sentinel_2A_740.5", "Sentinel_2A_782.8", "Sentinel_2A_832.8", "Sentinel_2A_864.7",
                 "Sentinel_2A_1613.7", "Sentinel_2A_2202.4"]
COLS_WAVELENGTH = [f"w{wavelength}" for wavelength in range(400, 2501)]

COLS_CATEGORICAL = [COL_TREE_SPECIES]
COLS_NUMERICAL = [COL_WETNESS] + COLS_SENTINEL + COLS_WAVELENGTH


class Dataset:
    def __init__(self, num_samples: Optional[int] = None, random_seed: int = 42,
                 data_filename: str = 'RtmSimulation_kickstart.csv'):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param random_seed: the random seed to use when sampling data points
        :param data_filename: the filename of file containing the dataset
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.data_filename = data_filename

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        df = pd.read_csv(config.csv_data_path(self.data_filename), index_col=0)
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        return df

    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresponding
        series of class values
        """
        df = self.load_data_frame()
        return df.drop(columns=[COL_LEAF_AREA_INDEX]), df[COL_LEAF_AREA_INDEX]


class DataFrameColumnTransformer(ColumnTransformer):
    def __init__(self, transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None,
                 verbose=False):
        super().__init__(transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
                         transformer_weights=transformer_weights, verbose=verbose)
        self.output_columns = None

    def fit_transform(self, X, y=None, **params):
        result = super().fit_transform(X, y, **params)
        self.output_columns = self.get_feature_names_out()

        # if without the prefix, there are no duplicates, remove the prefix
        if len(set(self.output_columns)) == len(
                set([col.split('__')[1] if '__' in col else col for col in self.output_columns])):
            self.output_columns = [col.split('__')[1] if '__' in col else col for col in self.output_columns]

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=self.output_columns, index=X.index)
        return result

    def transform(self, X, **params):
        result = super().transform(X, **params)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=self.output_columns, index=X.index)
        return result


class DataFrameTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformer, columns=None):
        """
        transformer: The sklearn transformer to wrap
        columns: The column names to apply to the output DataFrame. If None, attempt to use the input column names.
        """
        self.transformer = transformer
        self.columns = columns

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        # Attempt to capture input column names if columns are not provided
        if self.columns is None and hasattr(X, 'columns'):
            self.columns = X.columns
        return self

    def transform(self, X, y=None):
        transformed = self.transformer.transform(X)
        # If transformer output is not a DataFrame, convert it to DataFrame with the captured or provided column names
        if not isinstance(transformed, pd.DataFrame):
            if self.columns is not None:
                transformed = pd.DataFrame(transformed, columns=self.columns, index=X.index)
            else:
                transformed = pd.DataFrame(transformed, index=X.index)
        return transformed

    def get_feature_names_out(self, input_features=None):
        if self.columns is not None:
            return np.array(self.columns)
        if input_features is not None:
            return np.array(input_features)
        raise ValueError("Feature names could not be determined.")


class CustomPCA(TransformerMixin):
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.cols = None

    def fit(self, X, y=None):
        transformed_data = self.pca.fit_transform(X)
        self.cols = [f"PCA_{i + 1}" for i in range(transformed_data.shape[1])]
        return self

    def transform(self, X, y=None):
        transformed_data = self.pca.transform(X)
        return pd.DataFrame(transformed_data, index=X.index, columns=self.cols)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols)


class CustomLGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, categorical_feature: Union[str, List[str]] = 'auto', **kwargs):
        self.model = LGBMRegressor(**kwargs)
        self.categorical_feature = categorical_feature

    def fit(self, X, y, **fit_params):
        # If X is a DataFrame, we can use column names
        if isinstance(X, pd.DataFrame) and self.categorical_feature != 'auto':
            cat_features = [col for col in self.categorical_feature if col in X.columns]
        else:
            cat_features = 'auto'  # Let LightGBM handle it automatically

        self.model.fit(X, y, categorical_feature=cat_features, **fit_params)
        return self

    def predict(self, X, **predict_params):
        return self.model.predict(X, **predict_params)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class ModelFactory:
    @classmethod
    def __scaleEncodePipeline(cls) -> BaseEstimator:
        numeric_transformer = Pipeline(
            steps=[("imputer", DataFrameTransformer(SimpleImputer(strategy="median"), columns=COLS_NUMERICAL))]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("encoder", DataFrameTransformer(OrdinalEncoder(), columns=COLS_CATEGORICAL)),
            ]
        )
        preprocessor = DataFrameColumnTransformer(
            transformers=[
                ("num", numeric_transformer, COLS_NUMERICAL),
                ("cat", categorical_transformer, COLS_CATEGORICAL),
            ]
        )
        return preprocessor

    @classmethod
    def __dimensionalityReductionPipeline(cls) -> BaseEstimator:
        pca_transformer = Pipeline(
            steps=[("pca", CustomPCA(n_components=0.95))]
        )
        preprocessor = DataFrameColumnTransformer(
            transformers=[
                ("pca", pca_transformer, COLS_SENTINEL + COLS_WAVELENGTH),
            ], remainder='passthrough'
        )
        return preprocessor

    @classmethod
    def lgbm_regressor(cls, model_params=None) -> Pipeline:
        preprocessor = cls.__scaleEncodePipeline()
        pca = cls.__dimensionalityReductionPipeline()
        return Pipeline([
            ("preprocessor", preprocessor),
            ("dim_reduction", pca),
            ("model", CustomLGBMRegressor(categorical_feature=COLS_CATEGORICAL, random_state=42, verbose=-1,
                                          **(model_params or {})))])

    @classmethod
    def lgbmRegressorCV(cls) -> Pipeline:
        pass

    @classmethod
    def explain(cls):
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


if __name__ == '__main__':
    data = Dataset()

    # define target and features
    X, y = data.load_xy()
    print('Xshape: \n{}'.format(X.shape))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    gbm_model = ModelFactory.lgbm_regressor()
    cat_features_indices = [X_train.columns.get_loc(c) for c in COLS_CATEGORICAL if c in X_train]
    gbm_model.fit(X_train, y_train)
    y_pred_train = gbm_model.predict(X_train)

    mse = mean_squared_error(y_train, y_pred_train)

    print("Train MSE:", mse, "Train r2:", r2_score(y_train, y_pred_train))

    y_pred_val = gbm_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)

    print("Validation MSE:", mse, "Validation r2:", r2_score(y_val, y_pred_val))

    y_pred_test = gbm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)

    print("Test MSE:", mse, "Test r2:", r2_score(y_test, y_pred_test))
