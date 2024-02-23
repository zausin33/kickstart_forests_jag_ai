import os
import warnings
from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, List

import hyperopt
from hyperopt import hp

from sensai.evaluation import RegressionEvaluatorParams, RegressionModelEvaluation
from sensai.evaluation.eval_stats import RegressionMetricRRSE
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.pickle import load_pickle

from lai_regression.src.data import Dataset
from lai_regression.src.features import FeatureName
from lai_regression.src.model_factory import ModelFactory

log = logging.getLogger(__name__)


class HyperoptModelConfig(ABC):

    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def create_model(self, search_space_element: Dict[str, Any]):
        pass

    @abstractmethod
    def initial_space(self) -> List[Dict[str, int | float]]:
        pass

    @abstractmethod
    def search_space(self) -> Dict[str, Any]:
        pass


class LGBMHyperoptConfig(HyperoptModelConfig):

    def model_name(self) -> str:
        return "lgbm"

    def create_model(self, search_space_element: Dict[str, Any]):
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

    def initial_space(self) -> List[Dict[str, int | float]]:
        return [
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

    def search_space(self) -> Dict[str, Any]:
        return {
            'num_leaves': hp.uniformint("num_leaves", 20, 40),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'n_estimators': hp.uniformint('n_estimators', 80, 200),
            'max_depth': hp.uniformint('max_depth', -1, 15),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'min_split_gain': hp.uniform('min_split_gain', 0, 0.5),
        }


class CatBoostHyperoptConfig(HyperoptModelConfig):
    def model_name(self) -> str:
        return "catboost"

    def create_model(self, search_space_element: Dict[str, Any]):
        model_params = {
            'depth': search_space_element['depth'],
            'learning_rate': search_space_element['learning_rate'],
            'iterations': search_space_element['iterations'],
            'l2_leaf_reg': search_space_element['l2_leaf_reg'],
            'verbose': False,
            "pca_n_components": search_space_element["pca_n_components"]
        }
        return ModelFactory.create_catboost_regressor(name_suffix="-pca-only",
                                                      features=[FeatureName.PCA_WAVELENGTHS_SENTINEL],
                                                      **model_params)

    def initial_space(self) -> List[Dict[str, int | float]]:
        return [
            {
                'depth': 6,
                'learning_rate': 0.1,
                'iterations': 1000,
                'l2_leaf_reg': 3,
                "pca_n_components": 0.96
            }
        ]

    def search_space(self) -> Dict[str, Any]:
        return {
            'depth': hp.uniformint('depth', 4, 10),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'iterations': hp.uniformint('iterations', 500, 1500),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
            "pca_n_components": hp.uniform("pca_n_components", 0.8, 0.99)
        }


def run_hyperopt(dataset: Dataset, model_config: HyperoptModelConfig, minutes=30):
    experiment_name = f"{datetime_tag()}-{model_config.model_name()}-{dataset.tag()}"
    result_writer = ResultWriter(os.path.join("results", "hyperopt", experiment_name))
    logging.add_file_logger(result_writer.path("log.txt"))

    io_data = dataset.load_io_data()
    metric = RegressionMetricRRSE()
    evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=21)
    ev = RegressionModelEvaluation(io_data, evaluator_params=evaluator_params)

    def objective(search_space_element: Dict[str, Any]):
        log.info(f"Evaluating {search_space_element}")
        model = model_config.create_model(search_space_element)
        loss = ev.perform_simple_evaluation(model).get_eval_stats().compute_metric_value(metric)
        log.info(f"Loss[{metric.name}]={loss}")
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    trials_file = result_writer.path("trials.pickle")
    logging.getLogger("sensai").setLevel(logging.WARN)
    log.info(f"Starting hyperparameter optimisation for {model_config.model_name()} and {dataset}")
    hyperopt.fmin(objective, model_config.search_space(), algo=hyperopt.tpe.suggest, timeout=minutes * 60,
                  show_progressbar=False,
                  trials_save_file=trials_file, points_to_evaluate=model_config.initial_space())
    logging.getLogger("sensai").setLevel(logging.INFO)
    trials: hyperopt.Trials = load_pickle(trials_file)
    log.info(f"Best trial: {trials.best_trial}")


if __name__ == '__main__':
    logging.run_main(lambda: run_hyperopt(Dataset(), model_config=CatBoostHyperoptConfig(), minutes=5))
