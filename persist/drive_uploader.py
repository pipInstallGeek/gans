from __future__ import annotations

"""
Google Drive uploader utility for Kaggle or local runs.

Features:
- Works with a Service Account JSON (recommended for Kaggle, no prompts).
- Or with OAuth user credentials via token.json (pre-created locally) + client_secrets.json.
- Resumable uploads for large files (e.g., 1.9GB+).
- Overwrite behavior (update if a file with the same name exists in the folder).

Environment and file expectations (any one of these auth setups):
1) Service Account:
   - Set env var GDRIVE_CREDENTIALS=/path/to/service_account.json
   - Share the target Drive folder with the service account email.
2) OAuth (pre-created token):
   - Place client_secrets.json and token.json alongside your code (or set paths).
   - token.json must have offline access (contains a refresh token).

Usage example (Kaggle Notebook):
    from persist.drive_uploader import upload_to_drive
    zip_path = "/kaggle/working/outputs.zip"
    folder_id = "<your_drive_folder_id>"
    upload_to_drive(zip_path, folder_id, overwrite=True)

"""

import os
from pathlib import Path
from typing import Optional, Tuple

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


def _build_drive_service_with_service_account(sa_json: Path):
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_service_account_file(str(sa_json), scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _build_drive_service_with_oauth(client_secrets: Path, token_json: Path):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not token_json.exists():
        raise FileNotFoundError(
            f"token.json not found at {token_json}. Pre-generate it locally and ship to Kaggle."
        )
    creds = Credentials.from_authorized_user_file(str(token_json), scopes=SCOPES)
    if not creds.valid and creds.refresh_token:
        from google.auth.transport.requests import Request

        creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_existing_file(service, name: str, folder_id: str) -> Optional[str]:
    # Search for a file with exact name in the target folder.
    escaped_name = name.replace("'", "\\'")
    q = (
        f"name = '{escaped_name}' and "
        f"'{folder_id}' in parents and trashed = false"
    )
    res = service.files().list(q=q, fields="files(id, name)", pageSize=1).execute()
    items = res.get("files", [])
    return items[0]["id"] if items else None


def _upload_new(service, file_path: Path, folder_id: str, mime_type: Optional[str] = None) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {"name": file_path.name, "parents": [folder_id]}
    media = MediaFileUpload(
        str(file_path), mimetype=mime_type, resumable=True, chunksize=10 * 1024 * 1024
    )
    req = service.files().create(body=body, media_body=media, fields="id")
    return _resumable_upload(req)


def _update_existing(service, file_id: str, file_path: Path, mime_type: Optional[str] = None) -> str:
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(
        str(file_path), mimetype=mime_type, resumable=True, chunksize=10 * 1024 * 1024
    )
    req = service.files().update(fileId=file_id, media_body=media)
    return _resumable_upload(req)


def _resumable_upload(request) -> str:
    response = None
    while response is None:
        status, response = request.next_chunk()
        # Optional: you could print status.progress() here in a notebook.
    return response["id"]


def _detect_auth_paths(
    service_account_json: Optional[os.PathLike] = None,
    client_secrets_json: Optional[os.PathLike] = None,
    token_json: Optional[os.PathLike] = None,
) -> Tuple[str, Optional[Path], Optional[Path]]:
    """
    Returns a tuple (mode, path_a, path_b)
    - mode == 'sa' for service account; path_a is sa json.
    - mode == 'oauth' for oauth; path_a is client_secrets, path_b is token.json.
    """
    # Prefer explicit args, then env var, then local files.
    if service_account_json:
        return "sa", Path(service_account_json), None

    env_sa = os.getenv("GDRIVE_CREDENTIALS")
    if env_sa:
        return "sa", Path(env_sa), None

    cs = Path(client_secrets_json) if client_secrets_json else Path("client_secrets.json")
    tk = Path(token_json) if token_json else Path("token.json")
    if cs.exists() and tk.exists():
        return "oauth", cs, tk

    # Fallback to service account if a local SA file is present
    for cand in (Path("service_account.json"), Path("gdrive_sa.json")):
        if cand.exists():
            return "sa", cand, None

    raise FileNotFoundError(
        "No credentials found. Provide service account JSON via GDRIVE_CREDENTIALS or file, "
        "or provide client_secrets.json + token.json."
    )


def upload_to_drive(
    file_path: os.PathLike,
    folder_id: str,
    *,
    overwrite: bool = True,
    mime_type: Optional[str] = None,
    service_account_json: Optional[os.PathLike] = None,
    client_secrets_json: Optional[os.PathLike] = None,
    token_json: Optional[os.PathLike] = None,
) -> str:
    """
    Upload a file to Google Drive folder.

    - file_path: local file path to upload.
    - folder_id: Google Drive folder ID (the long ID from folder URL).
    - overwrite: if True, updates existing file with same name in folder.
    - mime_type: optional explicit MIME type; auto-detected if None.
    - Auth precedence: service account (if provided or env), else OAuth (client+token).

    Returns: the uploaded file ID.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    mode, a, b = _detect_auth_paths(
        service_account_json=service_account_json,
        client_secrets_json=client_secrets_json,
        token_json=token_json,
    )

    # Lazy import to avoid import cost unless used
    if mode == "sa":
        service = _build_drive_service_with_service_account(a)
    else:
        service = _build_drive_service_with_oauth(a, b)

    existing_id = _find_existing_file(service, p.name, folder_id) if overwrite else None
    if existing_id:
        return _update_existing(service, existing_id, p, mime_type)
    return _upload_new(service, p, folder_id, mime_type)


__all__ = ["upload_to_drive"]

