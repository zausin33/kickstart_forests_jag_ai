# import os

# from dagster import Definitions
# from lai_regression.src.constants import (
#     DATA_BASE_DIR,
#     MLFLOW_EXPERIMENT,
#     MLFLOW_PASSWORD,
#     MLFLOW_TRACKING_URL,
#     MLFLOW_USERNAME,
#     MODEL_BASE_DIR,
# )

# from lai_regression.src.resources.mlflow_session import MlflowSession

# definitions = Definitions(
#     resources={
#         "mlflow_session": MlflowSession(
#             tracking_url=MLFLOW_TRACKING_URL,
#             username=MLFLOW_USERNAME,
#             password=MLFLOW_PASSWORD,
#             experiment=MLFLOW_EXPERIMENT,
#         ),
#     },
# )