import warnings
from typing import Sequence, Optional

from sensai.data_transformation import DFTSkLearnTransformer
from sensai.featuregen import FeatureCollector
from sensai.lightgbm import LightGBMVectorRegressionModel
from sklearn.preprocessing import StandardScaler

from .data import COLS_NUMERICAL, COLS_CATEGORICAL
from .features import registry, FeatureName

warnings.filterwarnings('ignore', category=UserWarning)


class ModelFactory:
    DEFAULT_FEATURES = (FeatureName.WETNESS, FeatureName.CATEGORICAL, FeatureName.WAVELENGTH)


    @classmethod
    def create_lgbm_regressor(cls, name_suffix="", features: Sequence[FeatureName] = DEFAULT_FEATURES,
                   add_features: Sequence[FeatureName] = (),
                   min_child_weight: Optional[float] = None, **kwargs) -> LightGBMVectorRegressionModel:
        # fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
        fc = FeatureCollector(*features, *add_features, registry=registry)

        return LightGBMVectorRegressionModel(min_child_weight=min_child_weight, **kwargs) \
            .with_feature_collector(fc).with_feature_transformers(
            fc.create_dft_one_hot_encoder(),
            fc.create_feature_transformer_normalisation(),
            DFTSkLearnTransformer(StandardScaler())) \
            .with_name(f"LightGBM{name_suffix}")

    @classmethod
    def create_lgbm_wavelength_opt(cls):
        params = {
            'colsample_bytree': 0.9428047884327909,
            'gamma': 0.23886800503364314,
            'max_depth': 10,
            'min_child_weight': 15,
            'num_leaves': 37,
            'reg_lambda': 0.6598013723009393
        }
        return cls.create_lgbm_regressor("-wavelength-opt", add_features=[FeatureName.WAVELENGTH], **params)

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
