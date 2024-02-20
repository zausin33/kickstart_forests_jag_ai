import warnings
from typing import Sequence

from sensai.data_transformation import DFTSkLearnTransformer
from sensai.featuregen import FeatureCollector
from sensai.lightgbm import LightGBMVectorRegressionModel
from sensai.sklearn.sklearn_regression import SkLearnSVRVectorRegressionModel
from sklearn.preprocessing import StandardScaler

from .data import COLS_NUMERICAL, COLS_CATEGORICAL
from .features import registry, FeatureName

warnings.filterwarnings('ignore', category=UserWarning)


class ModelFactory:
    DEFAULT_FEATURES = (FeatureName.WETNESS, FeatureName.CATEGORICAL, FeatureName.WAVELENGTH_DIM_REDUCTION)


    @classmethod
    def create_lgbm_regressor(cls, name_suffix="", features: Sequence[FeatureName] = DEFAULT_FEATURES, model_params=None) -> LightGBMVectorRegressionModel:
        fc = FeatureCollector(*features, registry=registry)
        name_suffix = f"_{name_suffix}" if name_suffix else ""

        return LightGBMVectorRegressionModel(random_state=42, verbose=-1, categorical_feature_names=COLS_CATEGORICAL,
                                             **(model_params or {})) \
            .with_feature_collector(fc).with_feature_transformers(
            fc.create_feature_transformer_normalisation(require_all_handled=False),
            DFTSkLearnTransformer(StandardScaler())) \
            .with_name(f"LightGBM{name_suffix}")

    @classmethod
    def create_svr(cls, name_suffix="", features: Sequence[FeatureName] = DEFAULT_FEATURES, model_params=None) -> SkLearnSVRVectorRegressionModel:
        fc = FeatureCollector(*features, registry=registry)
        name_suffix = f"_{name_suffix}" if name_suffix else ""

        return SkLearnSVRVectorRegressionModel(**(model_params or {})) \
            .with_feature_collector(fc) \
            .with_feature_transformers(
            fc.create_feature_transformer_one_hot_encoder(ignore_unknown=True),
            fc.create_feature_transformer_normalisation(require_all_handled=False)) \
            .with_name(f"SVR{name_suffix}")
