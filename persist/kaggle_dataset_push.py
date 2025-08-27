#!/usr/bin/env python
"""
Programmatically push training artifacts to a private Kaggle dataset.

When KAGGLE_USERNAME and KAGGLE_KEY environment variables are present this
script writes a kaggle.json file and uses the Kaggle API to either create a
new dataset or version an existing one.  If the credentials are missing the
script silently returns without raising an error, allowing it to be invoked in
environments where dataset upload is optional.

This helper is intended for automated runs where interactive prompts are
undesirable (e.g. overnight training jobs on Kaggle).  It packages all files
under the configured outputs directory into a timestamped folder and then
uploads that as a new dataset version.
"""

import os
import json
import time
import pathlib
import shutil
import tarfile

from kaggle.api.kaggle_api_extended import KaggleApi

from utils.paths import outputs_dir

def setup_kaggle_creds() -> bool:
    """Write ~/.kaggle/kaggle.json from env vars and return True if creds exist."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not (username and key):
        return False
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    creds_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(creds_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(creds_path, 0o600)
    return True

def main():
    # Do nothing if creds are missing
    if not setup_kaggle_creds():
        print("Kaggle creds not found; skipping dataset push.")
        return 0

    # Initialise API
    api = KaggleApi()
    api.authenticate()

    # Determine export directory and timestamp
    stamp = time.strftime("%Y%m%d_%H%M%S")
    export_dir = pathlib.Path("persist") / f"export_{stamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts from outputs_dir into export directory
    src = outputs_dir()
    # Copy directories if they exist
    for sub in ["models", "samples", "metrics", "plots"]:
        p = src / sub
        if p.exists():
            shutil.copytree(p, export_dir / sub, dirs_exist_ok=True)

    # Tar the directory to avoid file count limits
    tar_path = export_dir.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(export_dir, arcname=export_dir.name)

    # Compose slug from repository and username
    repo_name = os.getenv("REPO_NAME", "gans")
    repo_slug = os.getenv("REPO_SLUG", "gans")
    username = os.getenv("KAGGLE_USERNAME")
    slug = f"{repo_slug}-artifacts"
    ref = f"{username}/{slug}"

    # Build metadata file
    meta_path = export_dir / "dataset-metadata.json"
    meta = {
        "title": f"{repo_name} Artifacts",
        "id": ref,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    meta_path.write_text(json.dumps(meta))

    # Upload dataset
    try:
        # Check if dataset exists
        api.dataset_view(ref)
        # If exists, version it
        api.dataset_create_version(
            folder=str(export_dir),
            version_notes=f"Auto-save {stamp}",
            convert_to_csv=False,
            dir_mode="zip"
        )
        print("Dataset version created successfully.")
    except Exception:
        # Otherwise create new dataset
        api.dataset_create_new(
            folder=str(export_dir),
            convert_to_csv=False,
            dir_mode="zip"
        )
        print("Dataset created successfully.")

    print("Dataset push complete.")

if __name__ == "__main__":
    main()