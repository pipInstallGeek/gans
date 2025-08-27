import os
import pathlib

"""
Path helpers to normalise output locations across environments.

When running on Kaggle, the `/kaggle/working` directory is persisted between
session commits and holds the notebook outputs.  When running locally or in a
non-Kaggle environment the current working directory is used instead.
"""

def on_kaggle() -> bool:
    """Return True if executing inside Kaggle."""
    return os.path.exists("/kaggle/working") or os.path.exists("/kaggle")

def outputs_dir() -> pathlib.Path:
    """
    Determine the base directory for experiment outputs.

    On Kaggle this function points to `/kaggle/working/outputs`.  Locally it
    returns `./outputs`.  The directory is created if it does not already exist.
    """
    base = pathlib.Path("/kaggle/working") if on_kaggle() else pathlib.Path(".")
    p = base / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p