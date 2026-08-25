from config import INPUT_DIR
from pathlib import Path
import pandas as pd


def read_and_prepare_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
