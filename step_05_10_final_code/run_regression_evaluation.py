import os
import warnings

from sensai.evaluation import RegressionModelEvaluation, RegressionEvaluatorParams, VectorModelCrossValidatorParams
from sensai.evaluation.eval_stats import RegressionMetricR2
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.string import TagBuilder
from sensai.feature_importance import plot_feature_importance, FeatureImportanceProvider

from step_05_10_final_code.src.data import Dataset, COL_LEAF_AREA_INDEX
from step_05_10_final_code.src.features import FeatureName
from step_05_10_final_code.src.model_factory import ModelFactory

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

    # define models to be evaluated
    models = [
        ModelFactory.create_lgbm_regressor("wavelength-dim-reduction"),
        ModelFactory.create_lgbm_regressor("only-sentinal-no-wavelength",
                                           features=[FeatureName.WETNESS, FeatureName.CATEGORICAL, FeatureName.SENTINEL]),
        ModelFactory.create_lgbm_regressor(
            "all-wavelengths-and-sentinel",
            features=[FeatureName.WETNESS, FeatureName.CATEGORICAL, FeatureName.SENTINEL, FeatureName.ALL_WAVELENGTH]
        ),
        ModelFactory.create_svr("wavelength-dim-reduction", model_params={'C': 100, 'epsilon': 0.001, 'kernel': 'rbf'}),
        ModelFactory.create_svr(
            "all-wavelengths-and-sentinel",
            features=[FeatureName.WETNESS, FeatureName.CATEGORICAL, FeatureName.SENTINEL, FeatureName.ALL_WAVELENGTH],
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
