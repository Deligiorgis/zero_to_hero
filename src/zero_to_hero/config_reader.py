"""
Reading Configuration files
"""
from pathlib import Path
from typing import Dict, Union

import yaml


def read_config(path: Union[str, Path]) -> Dict:
    """
    Reading configuration file based on the given path
    :param path: str or Path, path to the config file
    :return: Dict, dictionary generated by the config file which has a YAML type
    """
    print(f"Reading YAML config file from: {str(path)}")
    with Path(path).open(mode="r") as stream:
        config: Dict = yaml.safe_load(stream=stream)
    return config
