import warnings

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
    def create_lgbm_regressor(cls, model_params=None) -> LightGBMVectorRegressionModel:
        fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)

        return LightGBMVectorRegressionModel(random_state=42, verbose=-1,
                                             **(model_params or {})) \
            .with_feature_collector(fc).with_feature_transformers(
            fc.create_dft_one_hot_encoder(),
            fc.create_feature_transformer_normalisation(),
            DFTSkLearnTransformer(StandardScaler())) \
            .with_name("LightGBM")


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
