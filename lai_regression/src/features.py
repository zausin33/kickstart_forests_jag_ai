from enum import Enum

from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory, DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns, FeatureGeneratorFromDFT, \
    FeatureGenerator, ChainedFeatureGenerator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from .data import *

log = logging.getLogger(__name__)


class FeatureName(Enum):
    PCA_WAVELENGTHS_SENTINEL = "pca_wavelengths_sentinel"
    SENTINEL = "sentinel"
    WAVELENGTHS = "wavelengths"
    TREE_SPECIES = "tree_species"
    WETNESS = "wetness"


class FeatureGeneratorWavelengthDimensionalityReduction(FeatureGenerator):
    def __init__(self, n_components=0.96):
        """
        A feature generator that reduces the dimensionality of the wavelength features using PCA.

        :param n_components: The number of components to keep. If 0 < n_components < 1, it will be the ratio of the
        variance explained by the components.
        """
        self.rule = DFTNormalisation.Rule(skip=True, regex=None)
        super().__init__(normalisation_rules=[self.rule])
        self.pca = PCA(n_components=n_components)
        self.cols = None

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        """
        Fit the PCA model to the data.
        :param x: The input data
        :param y: The target data
        :param ctx: The context
        :return: None
        """

        self.pca.fit(x)
        feature_names_out = self.pca.get_feature_names_out()

        self.cols = [f"PCA_{i + 1}" for i in range(len(feature_names_out))]
        log.info(f"PCA: {x.shape[1]} -> {len(self.cols)}")
        self.rule.set_regex(f"PCA_.*")

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        """
        Transform the input data using the PCA model.
        :param df: The input data
        :param ctx: The context
        :return: The transformed data
        """
        transformed_data = self.pca.transform(df)

        return pd.DataFrame(transformed_data, index=df.index, columns=self.cols)


class FeatureGeneratorImputeMedian(FeatureGeneratorFromDFT):
    def __init__(self):
        super().__init__(DFTSkLearnTransformer(SimpleImputer(strategy="median")),
                         normalisation_rule_template=DFTNormalisation.RuleTemplate(
                             transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler(),
                             independent_columns=False))


class PcaWavelengthSentinelFeatureGenerator(ChainedFeatureGenerator):
    def __init__(self, n_components=0.96):
        super().__init__(
            [
                FeatureGeneratorTakeColumns(COLS_SENTINEL + COLS_WAVELENGTH),
                FeatureGeneratorImputeMedian(),
                FeatureGeneratorWavelengthDimensionalityReduction(n_components=n_components)
            ]
        )


registry = FeatureGeneratorRegistry()
registry.register_factory(FeatureName.PCA_WAVELENGTHS_SENTINEL,
                          lambda: PcaWavelengthSentinelFeatureGenerator())

registry.register_factory(FeatureName.SENTINEL,
                          lambda: ChainedFeatureGenerator([
                              FeatureGeneratorTakeColumns(COLS_SENTINEL),
                              FeatureGeneratorImputeMedian(),
                          ]))

registry.register_factory(FeatureName.WAVELENGTHS,
                          lambda: ChainedFeatureGenerator([
                              FeatureGeneratorTakeColumns(COLS_WAVELENGTH),
                              FeatureGeneratorImputeMedian(),
                          ]))

registry.register_factory(FeatureName.TREE_SPECIES, lambda: FeatureGeneratorTakeColumns(
    COLS_CATEGORICAL,
    categorical_feature_names=COLS_CATEGORICAL
))

registry.register_factory(FeatureName.WETNESS, lambda: FeatureGeneratorTakeColumns(
    COL_WETNESS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
