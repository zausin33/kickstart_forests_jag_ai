import warnings

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import COLS_NUMERICAL, COLS_CATEGORICAL, COLS_SENTINEL, COLS_WAVELENGTH
from .pipeline_components import DataFrameColumnTransformer, DataFrameTransformer, CustomPCA, CustomLGBMRegressor

warnings.filterwarnings('ignore', category=UserWarning)


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

    """"@classmethod
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
        feat_df.plot(x="feature", y="importance", kind="bar", color="blue", )"""
