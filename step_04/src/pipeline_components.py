from typing import Union, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


class DataFrameColumnTransformer(ColumnTransformer):
    def __init__(self, transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None,
                 verbose=False):
        """
        Wrapper for sklearn ColumnTransformer that reformat the output to a DataFrame

        :param transformers: The transformers to apply to the data
        :param remainder: The strategy to use for the remaining columns
        :param sparse_threshold: The threshold for the number of non-zero entries in the transformed output
        :param n_jobs: The number of jobs to use for the transformation
        :param transformer_weights: The weights to apply to the transformers
        :param verbose: Whether to print out information during the transformation
        """
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
        Wrapper for sklearn Transformer that reformat the output to a DataFrame

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
        """
        Wrapper for sklearn PCA that reformat the output to a DataFrame

        :param n_components:
        """
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
        """
        Wrapper for LightGBM Regressor that handles categorical features

        :param categorical_feature: The categorical features to use
        :param kwargs: Additional keyword arguments to pass to the LightGBM Regressor
        """
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