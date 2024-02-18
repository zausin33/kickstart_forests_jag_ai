import logging
from typing import Tuple, Optional

import pandas as pd
from sensai import InputOutputData
from sensai.util.string import ToStringMixin

import config

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
    def __init__(self, num_samples: Optional[int] = None, random_seed: int = 42,
                 data_filename: str = 'RtmSimulation_kickstart.csv'):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param random_seed: the random seed to use when sampling data points
        :param data_filename: the filename of file containing the dataset
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.data_filename = data_filename

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        log.info(f"Loading {self} from {self.data_filename}")
        df = pd.read_csv(config.csv_data_path(self.data_filename), index_col=0)
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        return df

    def load_io_data(self) -> InputOutputData:
        """
        :return: the I/O data
        """
        return InputOutputData.from_data_frame(self.load_data_frame(), COL_LEAF_AREA_INDEX)