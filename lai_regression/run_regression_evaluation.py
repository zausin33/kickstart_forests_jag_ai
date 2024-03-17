import os
import warnings

from sensai.evaluation import RegressionModelEvaluation, RegressionEvaluatorParams, VectorModelCrossValidatorParams
from sensai.evaluation.eval_stats import RegressionMetricR2
from sensai.feature_importance import FeatureImportanceProvider
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.string import TagBuilder

from lai_regression.src.data import Dataset, COL_LEAF_AREA_INDEX
from lai_regression.src.features import FeatureName
from lai_regression.src.model_factory import ModelFactory
from lai_regression.src.resources.mlflow_session import MlflowSession
from lai_regression.src.constants import (
    DATA_BASE_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_PASSWORD,
    MLFLOW_TRACKING_URL,
    MLFLOW_USERNAME,
)

warnings.filterwarnings('ignore', category=UserWarning)

log = logging.getLogger(__name__)

def main():
    # define & load dataset
    dataset = Dataset()
    use_cross_validation = True
    do_plot_feature_importance = True

    experiment_name = TagBuilder("lai_regression_", dataset.tag()).with_conditional(use_cross_validation, "CV").build()
    run_id = datetime_tag()
    result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
    logging.add_file_logger(result_writer.path("log.txt"))

    io_data = dataset.load_io_data()

    # setup mlflow session
    mlflow_session = MlflowSession(
        tracking_url=MLFLOW_TRACKING_URL,
        username=MLFLOW_USERNAME,
        password=MLFLOW_PASSWORD,
        experiment=MLFLOW_EXPERIMENT,
    )
    mlflow_session.setup_for_execution()
    with mlflow_session.get_run():
        # define models to be evaluated
        models = [
            ModelFactory.create_mean_model(),
            ModelFactory.create_linear_regression(),
            ModelFactory.create_random_forest(),

            ModelFactory.create_lgbm_regressor(name_suffix="-no-pca", features=[FeatureName.WETNESS, FeatureName.TREE_SPECIES, FeatureName.WAVELENGTHS, FeatureName.SENTINEL]),
            ModelFactory.create_lgbm_regressor(name_suffix="-wet-species-sentinel", features=[FeatureName.WETNESS, FeatureName.TREE_SPECIES, FeatureName.SENTINEL]),
            ModelFactory.create_lgbm_regressor(name_suffix="-pca-only", features=[FeatureName.PCA_WAVELENGTHS_SENTINEL]),
            ModelFactory.create_lgbm_regressor(name_suffix="-default(pca,wetness,treeSpecies)"),

            ModelFactory.create_catboost_regressor(),
            ModelFactory.create_catboost_regressor(name_suffix="-pca-only", features=[FeatureName.PCA_WAVELENGTHS_SENTINEL]),
            ModelFactory.create_catboost_regressor_hyperopt(),

            ModelFactory.create_svr("wavelength-dim-reduction", model_params={'C': 100, 'epsilon': 0.001, 'kernel': 'rbf'}),
            ModelFactory.create_svr(
                "all-wavelengths-and-sentinel",
                features=[FeatureName.WETNESS, FeatureName.TREE_SPECIES, FeatureName.SENTINEL, FeatureName.WAVELENGTHS],
                model_params={'C': 100, 'epsilon': 0.001, 'kernel': 'rbf'}),
        ]

        # declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
        evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=42)
        cross_validator_params = VectorModelCrossValidatorParams(folds=3)

        # use a high-level utility class for evaluating the models based on these parameters
        ev = RegressionModelEvaluation(io_data, evaluator_params=evaluator_params, cross_validator_params=cross_validator_params)
        result = ev.compare_models(models, fit_models=True, result_writer=result_writer, use_cross_validation=use_cross_validation)

        # feature importance of best model
        if do_plot_feature_importance and not use_cross_validation:
            best_model = result.get_best_model(RegressionMetricR2.name)
            if isinstance(best_model, FeatureImportanceProvider):
                log.info(f"Plotting importance of best model '{best_model.get_name()}")
                fi = best_model.get_feature_importance()
                fig = fi.plot(predicted_var_name=COL_LEAF_AREA_INDEX)
                fig.savefig(result_writer.path("feature_importance.png"))



if __name__ == '__main__':
    logging.run_main(main)
