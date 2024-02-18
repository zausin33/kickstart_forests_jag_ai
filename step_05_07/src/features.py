from enum import Enum

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import *
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory, DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns, FeatureGeneratorFromDFT, \
    FeatureGenerator, ChainedFeatureGenerator


class FeatureName(Enum):
    WAVELENGTH = "wavelength"
    CATEGORICAL = "categorical"
    WETNESS = "wetness"


class FeatureGeneratorWavelengthDimensionalityReduction(FeatureGenerator):
    def __init__(self, n_components=0.96):
        super().__init__(normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True))
        self.pca = PCA(n_components=n_components)
        self.cols = None

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        transformed_data = self.pca.fit_transform(x)
        self.cols = [f"PCA_{i + 1}" for i in range(transformed_data.shape[1])]

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        transformed_data = self.pca.transform(df)

        return pd.DataFrame(transformed_data, index=df.index, columns=self.cols)


numeric_transformer = FeatureGeneratorFromDFT(
    DFTSkLearnTransformer(Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])),
    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)
)


registry = FeatureGeneratorRegistry()
registry.register_factory(FeatureName.WAVELENGTH,
                          lambda: ChainedFeatureGenerator([
                                FeatureGeneratorTakeColumns(COLS_SENTINEL + COLS_WAVELENGTH, normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)),
                                numeric_transformer,
                                FeatureGeneratorWavelengthDimensionalityReduction()
                          ]))

registry.register_factory(FeatureName.CATEGORICAL, lambda: FeatureGeneratorTakeColumns(
    COLS_CATEGORICAL,
    categorical_feature_names=COLS_CATEGORICAL
))

registry.register_factory(FeatureName.WETNESS, lambda: FeatureGeneratorTakeColumns(COL_WETNESS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))