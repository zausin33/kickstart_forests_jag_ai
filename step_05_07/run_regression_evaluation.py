import warnings

from sensai.evaluation import RegressionModelEvaluation, RegressionEvaluatorParams
from sensai.util import logging

from step_05_07.src.data import Dataset
from step_05_07.src.model_factory import ModelFactory

warnings.filterwarnings('ignore', category=UserWarning)

def main():
    # define & load dataset
    dataset = Dataset()
    io_data = dataset.load_io_data()

    # define models to be evaluated
    models = [
        ModelFactory.create_lgbm_regressor()
    ]

    # declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
    evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=0.2, fractional_split_random_seed=42)

    # use a high-level utility class for evaluating the models based on these parameters
    ev = RegressionModelEvaluation(io_data, evaluator_params=evaluator_params)
    ev.compare_models(models, fit_models=True)

if __name__ == '__main__':
    logging.run_main(main)