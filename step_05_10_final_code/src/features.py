from enum import Enum

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import *
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory, DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns, FeatureGeneratorFromDFT, \
    FeatureGenerator, ChainedFeatureGenerator

log = logging.getLogger(__name__)


class FeatureName(Enum):
    WAVELENGTH_DIM_REDUCTION = "wavelength_dim_reduction"
    ALL_WAVELENGTH = "all_wavelength"
    SENTINEL = "sentinel"
    CATEGORICAL = "categorical"
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

        transformed_data = self.pca.fit_transform(x)
        self.cols = [f"PCA_{i + 1}" for i in range(transformed_data.shape[1])]
        log.info(f"PCA: {transformed_data.shape[1]} -> {len(self.cols)}")
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


numeric_transformer = lambda: FeatureGeneratorFromDFT(
    DFTSkLearnTransformer(Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]))
)

ordinal_transformer = FeatureGeneratorFromDFT(
    DFTSkLearnTransformer(Pipeline(steps=[("ordinal", OrdinalEncoder())])),
    add_categorical_default_rules=False,
    categorical_feature_names=COLS_CATEGORICAL
)


registry = FeatureGeneratorRegistry()
registry.register_factory(FeatureName.WAVELENGTH_DIM_REDUCTION,
                          lambda: ChainedFeatureGenerator([
                                FeatureGeneratorTakeColumns(COLS_SENTINEL + COLS_WAVELENGTH),
                                numeric_transformer(),
                                FeatureGeneratorWavelengthDimensionalityReduction()
                          ]))

registry.register_factory(FeatureName.SENTINEL,
                          lambda: ChainedFeatureGenerator([
                              FeatureGeneratorTakeColumns(COLS_SENTINEL),
                              numeric_transformer(),
                          ]))

registry.register_factory(FeatureName.ALL_WAVELENGTH,
                          lambda: ChainedFeatureGenerator([
                              FeatureGeneratorTakeColumns(COLS_WAVELENGTH),
                              numeric_transformer(),
                          ]))

registry.register_factory(FeatureName.CATEGORICAL, lambda: ChainedFeatureGenerator([
    FeatureGeneratorTakeColumns(COLS_CATEGORICAL),
    ordinal_transformer
]))

registry.register_factory(FeatureName.WETNESS, lambda: FeatureGeneratorTakeColumns(COL_WETNESS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))