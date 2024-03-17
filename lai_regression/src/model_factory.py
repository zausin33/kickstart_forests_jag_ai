import warnings
from pathlib import Path
from typing import Sequence, Optional, List

from sensai.catboost import CatBoostVectorRegressionModel
from sensai.featuregen import FeatureCollector
from sensai.lightgbm import LightGBMVectorRegressionModel
from sensai.sklearn.sklearn_regression import SkLearnSVRVectorRegressionModel, SkLearnDummyVectorRegressionModel, \
    SkLearnLinearRegressionVectorRegressionModel, SkLearnRandomForestVectorRegressionModel

from .features import registry, FeatureName, PcaWavelengthSentinelFeatureGenerator
from .data import Dataset

warnings.filterwarnings('ignore', category=UserWarning)


def best_regression_model_storage_path(dataset: Dataset) -> Path:
    return Path("results") / "models" / "regression" / dataset.tag() / "best_model.pickle"


class ModelFactory:
    DEFAULT_FEATURES = [FeatureName.WETNESS, FeatureName.TREE_SPECIES, FeatureName.PCA_WAVELENGTHS_SENTINEL]

    @classmethod
    def create_mean_model(cls):
        return SkLearnDummyVectorRegressionModel(strategy="mean").with_name("Mean")

    @classmethod
    def create_linear_regression(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnLinearRegressionVectorRegressionModel() \
            .with_feature_collector(fc) \
            .with_feature_transformers(
            fc.create_feature_transformer_one_hot_encoder(),
            fc.create_feature_transformer_normalisation()) \
            .with_name("Linear")

    @classmethod
    def create_random_forest(cls):
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        return SkLearnRandomForestVectorRegressionModel() \
            .with_feature_collector(fc) \
            .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder()) \
            .with_name("RandomForest")

    @classmethod
    def create_lgbm_regressor(cls, name_suffix="", features: Optional[List[FeatureName]] = None,
                              **model_params) -> LightGBMVectorRegressionModel:
        if features is None:
            features = cls.DEFAULT_FEATURES
        fc = FeatureCollector(*features, registry=registry)
        name_suffix = f"_{name_suffix}" if name_suffix else ""

        return LightGBMVectorRegressionModel(random_state=42, verbose=-1,
                                             categorical_feature_names=fc.get_categorical_feature_name_regex(),
                                             **model_params) \
            .with_feature_collector(fc) \
            .with_name(f"LightGBM{name_suffix}")

    @classmethod
    def create_catboost_regressor(cls, name_suffix="", features: Optional[List[FeatureName]] = None,
                                  pca_n_components=0.96, **model_params) -> CatBoostVectorRegressionModel:
        if features is None:
            features = cls.DEFAULT_FEATURES

        if FeatureName.PCA_WAVELENGTHS_SENTINEL in features:
            features.remove(FeatureName.PCA_WAVELENGTHS_SENTINEL)
            fc = FeatureCollector(*features,
                                  PcaWavelengthSentinelFeatureGenerator(n_components=pca_n_components),
                                  registry=registry)
        else:
            fc = FeatureCollector(*features, registry=registry)

        name_suffix = f"_{name_suffix}" if name_suffix else ""

        return CatBoostVectorRegressionModel(categorical_feature_names=fc.get_categorical_feature_name_regex(),
                                             **model_params) \
            .with_feature_collector(fc) \
            .with_name(f"CatBoost{name_suffix}")

    @classmethod
    def create_catboost_regressor_hyperopt(cls):
        return cls.create_catboost_regressor(
            name_suffix="hyperopt",
            features=[FeatureName.PCA_WAVELENGTHS_SENTINEL],
            pca_n_components=0.9836,
            depth=4, learning_rate=0.0641, iterations=634, l2_leaf_reg=8.544
        )

    @classmethod
    def create_svr(cls, name_suffix="", features: Optional[List[FeatureName]] = None,
                   model_params=None) -> SkLearnSVRVectorRegressionModel:
        if features is None:
            features = cls.DEFAULT_FEATURES
        fc = FeatureCollector(*features, registry=registry)
        name_suffix = f"_{name_suffix}" if name_suffix else ""

        return SkLearnSVRVectorRegressionModel(**(model_params or {})) \
            .with_feature_collector(fc) \
            .with_feature_transformers(
            fc.create_feature_transformer_one_hot_encoder(),
            fc.create_feature_transformer_normalisation()) \
            .with_name(f"SVR{name_suffix}")
