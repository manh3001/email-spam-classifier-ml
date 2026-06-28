"""Load and normalize the SMS spam dataset."""

from pathlib import Path

import pandas as pd

from .config import DATA_PATH

_LABEL_MAP = {"ham": 0, "spam": 1}


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read spam.csv, keep label/message columns, map labels to 0/1."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path, encoding="latin-1")
    if not {"v1", "v2"}.issubset(raw.columns):
        raise ValueError(
            f"Expected columns 'v1' and 'v2' in {path}, got {list(raw.columns)}"
        )

    df = raw[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower().map(_LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Found label values outside {'ham','spam'}")
    df["label"] = df["label"].astype(int)
    df["message"] = df["message"].astype(str)
    return df.reset_index(drop=True)
