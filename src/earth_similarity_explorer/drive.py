"""Download Earth Engine exports from Google Drive."""

from pathlib import Path
from typing import Any

import ee
from google.auth.transport.requests import AuthorizedSession
from requests import Response

# The Drive REST API (v3): one URL to list files, one per file to download or delete
DRIVE_FILES_URL: str = "https://www.googleapis.com/drive/v3/files"


def download_drive_exports(
    folder_id: str, file_prefix: str, destination_dir: Path
) -> list[Path]:
    """Download the files in a Drive folder whose names start with a prefix.

    Each file is deleted from Drive (permanently, not to the trash) once it is
    downloaded, so a later export with the same prefix never picks up an old file.
    Earth Engine splits large exports into several files with the same prefix, so a
    list is returned.

    Raises:
        FileNotFoundError: no file in the folder matches the prefix.
        requests.HTTPError: a Drive request fails.
    """
    session: AuthorizedSession = AuthorizedSession(
        credentials=ee.data.get_persistent_credentials()
    )

    # Drive's "contains" matches name prefixes, word by word; startswith makes it exact
    listing: Response = session.get(
        url=DRIVE_FILES_URL,
        params={
            "q": f"'{folder_id}' in parents and name contains '{file_prefix}' "
            "and trashed = false",
            "fields": "files(id, name)",
            "pageSize": 1000,
        },
    )
    listing.raise_for_status()
    files: list[dict[str, Any]] = [
        file for file in listing.json()["files"] if file["name"].startswith(file_prefix)
    ]
    if not files:
        raise FileNotFoundError(f"no file {file_prefix}* in Drive folder {folder_id}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file in files:
        path: Path = destination_dir / file["name"]
        # Streamed in 1 MB chunks, so a large file never sits in memory as a whole
        with session.get(
            url=f"{DRIVE_FILES_URL}/{file['id']}", params={"alt": "media"}, stream=True
        ) as download:
            download.raise_for_status()
            with path.open(mode="wb") as local_file:
                for chunk in download.iter_content(chunk_size=2**20):
                    local_file.write(chunk)
        # Only after a complete download, so a failure never loses the export
        session.delete(url=f"{DRIVE_FILES_URL}/{file['id']}").raise_for_status()
        paths.append(path)
    return paths
