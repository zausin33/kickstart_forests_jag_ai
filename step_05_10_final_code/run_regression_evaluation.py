import os
import warnings

from sensai.evaluation import RegressionModelEvaluation, RegressionEvaluatorParams, VectorModelCrossValidatorParams
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.string import TagBuilder

from step_05_10_final_code.src.data import Dataset
from step_05_10_final_code.src.model_factory import ModelFactory

warnings.filterwarnings('ignore', category=UserWarning)

log = logging.getLogger(__name__)

def main():
    # define & load dataset
    dataset = Dataset()
    use_cross_validation = True

    experiment_name = TagBuilder("lai_regression_", dataset.tag()).with_conditional(use_cross_validation, "CV").build()
    run_id = datetime_tag()
    result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
    logging.add_file_logger(result_writer.path("log.txt"))

    io_data = dataset.load_io_data()

    # define models to be evaluated
    models = [
        ModelFactory.create_lgbm_regressor()
    ]

    # declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
    evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=42)
    cross_validator_params = VectorModelCrossValidatorParams(folds=3)

    # use a high-level utility class for evaluating the models based on these parameters
    ev = RegressionModelEvaluation(io_data, evaluator_params=evaluator_params, cross_validator_params=cross_validator_params)
    ev.compare_models(models, fit_models=True, result_writer=result_writer, use_cross_validation=use_cross_validation)

if __name__ == '__main__':
    logging.run_main(main)
