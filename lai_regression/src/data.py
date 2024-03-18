import logging

import pandas as pd
from sensai import InputOutputData
from sensai.util.string import ToStringMixin, TagBuilder

log = logging.getLogger(__name__)

COL_LEAF_AREA_INDEX = 'lai'
COL_ID = 'id'
COL_WETNESS = 'wetness'
COL_TREE_SPECIES = 'treeSpecies'
COLS_SENTINEL = ["Sentinel_2A_492.4", "Sentinel_2A_559.8", "Sentinel_2A_664.6", "Sentinel_2A_704.1",
                 "Sentinel_2A_740.5", "Sentinel_2A_782.8", "Sentinel_2A_832.8", "Sentinel_2A_864.7",
                 "Sentinel_2A_1613.7", "Sentinel_2A_2202.4"]
COLS_WAVELENGTH = [f"w{wavelength}" for wavelength in range(400, 2501)]

COLS_CATEGORICAL = [COL_TREE_SPECIES]
COLS_NUMERICAL = [COL_WETNESS] + COLS_SENTINEL + COLS_WAVELENGTH


class Dataset(ToStringMixin):
    def __init__(self,
                 dataset_filename: str = "RtmSimulation_kickstart.csv",
                 repository: str = "ai-kickstart",
                 branch: str = "main",
                 random_seed: int = 42
                 ):
        """
        The dataset for the leaf area index (LAI) prediction problem.

        :param dataset_filename: the name of the dataset file
        :param repository: the name of the lakeFS repository
        :param branch: the name of the branch in the lakeFS repository
        :param random_seed: the random seed to use when sampling data points
        """
        self.data_filename = dataset_filename
        self.repository = repository
        self.branch = branch
        self.random_seed = random_seed

    def tag(self):
        return TagBuilder(
            self.data_filename,
            self.repository,
            self.branch,
            glue="-"
        ) \
            .with_conditional(self.random_seed != 42, f"seed{self.random_seed}") \
            .build()

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        path = f"lakefs://{self.repository}/{self.branch}/{self.data_filename}"
        log.info(f"Loading {self} from {path}")
        df = pd.read_csv(path, index_col=0)
        return df

    def load_io_data(self) -> InputOutputData:
        """
        :return: the I/O data
        """
        return InputOutputData.from_data_frame(self.load_data_frame(), COL_LEAF_AREA_INDEX)
