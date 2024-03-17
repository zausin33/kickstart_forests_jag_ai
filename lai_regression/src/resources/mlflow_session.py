"""MLflow session resource."""

import os
from datetime import datetime
import mlflow
from dagster import ConfigurableResource


class MlflowSession(ConfigurableResource):
    """MLflow session resource.

    Notes
    -----
    Since MLflow uses global state (Python globals / environment variables)
    to store its tracking configuration (endpoint URL, credentials), this
    resource can effectively only be parameterized with a single configuration
    per Dagster process."""

    tracking_url: str
    username: str | None
    password: str | None
    experiment: str

    def setup_for_execution(self) -> None:
        mlflow.set_tracking_uri(self.tracking_url)
        # MLflow expects credentials in environment variables
        if self.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
        if self.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password
        mlflow.set_experiment(self.experiment)

    def get_run(self) -> mlflow.ActiveRun:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"JAG-MLflow-Run-{current_time}"
        active_run = mlflow.active_run()
        if active_run is None:
            current_runs = mlflow.search_runs(
                filter_string=f"attributes.`run_name`='{run_name}'",
                output_format="list",
            )

            if current_runs:
                run_id = current_runs[0].info.run_id
                return mlflow.start_run(run_id=run_id, run_name=run_name)
            else:
                tags = {"current_runs": False}
                return mlflow.start_run(run_name=run_name, tags=tags)

        return active_run
