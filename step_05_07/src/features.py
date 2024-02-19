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
        # NOTE: self.pca.fit(x) would have been sufficient, in conjunction with self.pca.get_feature_names_out() to determine shape
        transformed_data = self.pca.fit_transform(x)
        self.cols = [f"PCA_{i + 1}" for i in range(transformed_data.shape[1])]

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        transformed_data = self.pca.transform(df)

        return pd.DataFrame(transformed_data, index=df.index, columns=self.cols)


# NOTE: This uses a single instance, i.e. if multiple models are trained in succession, they all share the same instance, i.e.
#   the models that were trained earlier will get their component retrained by the later learning processes, which can be highly
#   problematic, especially if we were to work with multiple splits of the data.
numeric_transformer = FeatureGeneratorFromDFT(
    DFTSkLearnTransformer(Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])),
    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)
)

# We could have used this instead:
class FeatureGeneratorImputeMedian(FeatureGeneratorFromDFT):
    def __init__(self):
        super().__init__(DFTSkLearnTransformer(SimpleImputer(strategy="median")),
            normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True))


registry = FeatureGeneratorRegistry()
# NOTE: Name `WAVELENGTH` is misleading, as sentinel values are also included
registry.register_factory(FeatureName.WAVELENGTH,
                          lambda: ChainedFeatureGenerator([
                                # NOTE: normalisation rule for generator which is not the last in the chain has no effect.
                                FeatureGeneratorTakeColumns(COLS_SENTINEL + COLS_WAVELENGTH, normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)),
                                numeric_transformer,  # FeatureGeneratorImputeMedian()
                                FeatureGeneratorWavelengthDimensionalityReduction()
                          ]))

registry.register_factory(FeatureName.CATEGORICAL, lambda: FeatureGeneratorTakeColumns(
    COLS_CATEGORICAL,
    categorical_feature_names=COLS_CATEGORICAL
))

registry.register_factory(FeatureName.WETNESS, lambda: FeatureGeneratorTakeColumns(COL_WETNESS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))