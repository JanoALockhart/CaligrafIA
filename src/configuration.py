from pathlib import Path
from typing import List
from dataclasses import dataclass

import yaml

@dataclass
class DatasetInfo:
    name: str
    path: Path
    train_percentage: float
    validation_percentage: float
    test_percentage: float

@dataclass
class Configuration:
    preprocessFolder: Path
    datasets: List[DatasetInfo]

class ConfigurationLoader:
    @staticmethod
    def load_configurations(path: Path = "../config.yaml") -> Configuration:
        with open(path, "r") as file:
            config_dict = yaml.safe_load(file)

        datasets = []
        datasets_list = config_dict["datasets"]

        for info in datasets_list:
            datasets.append(
                DatasetInfo(
                    name = info["name"],
                    path=info["path"],
                    train_percentage=info["splits"]["train"],
                    validation_percentage=info["splits"]["validation"],
                    test_percentage=info["splits"]["test"]
                )
            )

        appConfig = Configuration(
            preprocessFolder=config_dict["preprocessFolder"],
            datasets=datasets
        )

        return appConfig