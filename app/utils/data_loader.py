# data_loader.py
from typing import Union, IO
import pandas as pd


def load_csv(file: Union[str, IO], encoding_fallback: str = "latin-1") -> pd.DataFrame:
    """
    Load a CSV from a file path or file-like object.

    Args:
        file: path to csv or file-like object (e.g. uploaded file).
        encoding_fallback: encoding to try if default read fails.

    Returns:
        pandas.DataFrame

    Raises:
        ValueError if file cannot be read as csv.
    """
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file, encoding=encoding_fallback)
        except Exception as ex:
            raise ValueError(f"Cannot read CSV with fallback encoding: {ex}") from ex
    except Exception as ex:
        raise ValueError(f"Cannot read CSV: {ex}") from ex

    return df
