import os
import numpy as np
import pandas as pd
from pathlib import Path


# get the parent path of the given path
def get_parent_dir(path: os.path, level: int = 1) -> os.path:
    return Path(path).resolve().parents[level]


def get_likelihood():
    pass


def get_eer():
    pass