import os
import warnings
from typing import Literal, Dict, Any

import hyperopt
from hyperopt import hp

from sensai.evaluation import RegressionEvaluatorParams, RegressionModelEvaluation
from sensai.evaluation.eval_stats import RegressionMetricRRSE
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.pickle import load_pickle

from step_05_10_final_code.src.data import Dataset
from step_05_10_final_code.src.model_factory import ModelFactory

log = logging.getLogger(__name__)

def run_hyperopt(dataset: Dataset, model: Literal["lgbm"] = "lgbm", hours=2):
    experiment_name = f"{datetime_tag()}-{model}-{dataset.tag()}"
    result_writer = ResultWriter(os.path.join("results", "hyperopt", experiment_name))
    logging.add_file_logger(result_writer.path("log.txt"))

    if model == "lgbm":
        initial_space = [
            {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': -1,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'min_split_gain': 0,
            }
        ]
        search_space = {
            'num_leaves': hp.uniformint("num_leaves", 20, 40),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'n_estimators': hp.uniformint('n_estimators', 80, 200),
            'max_depth': hp.uniformint('max_depth', -1, 15),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'min_split_gain': hp.uniform('min_split_gain', 0, 0.5),
        }

        def create_model(search_space_element: Dict[str, Any]):
            model_params = {
            'verbosity': 0,
            'num_leaves': search_space_element['num_leaves'],
            'learning_rate': search_space_element['learning_rate'],
            'n_estimators': search_space_element['n_estimators'],
            'max_depth': search_space_element['max_depth'],
            'reg_alpha': search_space_element['reg_alpha'],
            'reg_lambda': search_space_element['reg_lambda'],
            'min_split_gain': search_space_element['min_split_gain']
            }
            return ModelFactory.create_lgbm_regressor(model_params=model_params)


        warnings.filterwarnings("ignore")
    else:
        # Handle different models here
        raise ValueError(model)

    io_data = dataset.load_io_data()
    metric = RegressionMetricRRSE()
    evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=21)
    ev = RegressionModelEvaluation(io_data, evaluator_params=evaluator_params)

    def objective(search_space_element: Dict[str, Any]):
        log.info(f"Evaluating {search_space_element}")
        model = create_model(search_space_element)
        loss = ev.perform_simple_evaluation(model).get_eval_stats().compute_metric_value(metric)
        log.info(f"Loss[{metric.name}]={loss}")
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    trials_file = result_writer.path("trials.pickle")
    logging.getLogger("sensai").setLevel(logging.WARN)
    log.info(f"Starting hyperparameter optimisation for {model} and {dataset}")
    hyperopt.fmin(objective, search_space, algo=hyperopt.tpe.suggest, timeout=hours*3600, show_progressbar=False,
        trials_save_file=trials_file, points_to_evaluate=initial_space)
    logging.getLogger("sensai").setLevel(logging.INFO)
    trials: hyperopt.Trials = load_pickle(trials_file)
    log.info(f"Best trial: {trials.best_trial}")

if __name__ == '__main__':
    logging.run_main(lambda: run_hyperopt(Dataset(is_classification=False), hours=10))
