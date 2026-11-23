"""Google Earth Engine and Google Drive overhead, including downloading and exporting."""

import time
from pathlib import Path
from typing import Any

import ee
from google.auth.transport.requests import AuthorizedSession
from requests import Response

from earth_similarity_explorer.settings import DRIVE_FILES_URL, Settings


# EARTH ENGINE
def initialize_earth_engine() -> None:
    """Connect to Earth Engine with the Cloud project."""
    ee.Initialize(project=Settings().ee_project)


def wait_for_task(task: Any, label: str) -> dict[str, Any]:
    """
    Wait for a started Earth Engine task, print its progress, return its metadata.
    A line is printed when the state, the current stage or the whole percentage
    changes, and at least every 30 seconds while the task runs.

    Raises:
        RuntimeError: the task failed or was cancelled.
    """
    # The task ID finds the task in the Earth Engine Code Editor's task list.
    print(f"Earth Engine task {label}: {task.id} started", flush=True)

    started: float = time.monotonic()
    last_state: str = ""
    last_print: float = started
    while True:
        # The full operation record has progress, stages and compute use
        operation: dict[str, Any] = ee.data.getOperation(operation_name=task.name)
        metadata: dict[str, Any] = operation["metadata"]
        # Go through each stage as it completes
        stage: str = next(
            (
                stage["displayName"]
                for stage in metadata.get("stages", [])
                if int(stage.get("completeWorkUnits", 0)) < int(stage["totalWorkUnits"])
            ),
            "",
        )
        state: str = f"{metadata['state']} {metadata.get('progress', 0):.0%} {stage}"
        # Print on every change or after 30 s without a line to show a long task is alive
        if state != last_state or time.monotonic() - last_print >= 30:
            minutes, seconds = divmod(int(time.monotonic() - started), 60)
            eecu_hours: float = metadata.get("batchEecuUsageSeconds", 0) / 3600
            print(
                f"  {minutes:3d}:{seconds:02d}  {state.strip()}, "
                f"{eecu_hours:.3f} EECU-hours",
                flush=True,
            )
            last_state = state
            last_print = time.monotonic()
        if operation.get("done"):
            break
        time.sleep(10)

    # Failed and cancelled operations are also done, and carry Earth Engine's message
    if "error" in operation:
        raise RuntimeError(
            f"Earth Engine task {label}: {operation['error'].get('message')}"
        )
    return metadata


def export_to_drive(
    image: ee.Image,
    region: ee.Geometry,
    crs: str,
    crs_transform: list[float],
    drive_folder: str,
    file_prefix: str,
) -> str:
    """Export an image as GeoTIFF to a Google Drive folder and return the folder's ID.

    Blocks until the Earth Engine task has finished. The crs and crs_transform fix
    the output pixel grid. Pixels outside the region are masked.

    Raises:
        RuntimeError: the Earth Engine task failed or was cancelled.
    """
    task: Any = ee.batch.Export.image.toDrive(
        image=image,
        description=file_prefix,
        folder=drive_folder,
        fileNamePrefix=file_prefix,
        region=region,
        crs=crs,
        crsTransform=crs_transform,
        fileFormat="GeoTIFF",
    )
    task.start()
    metadata: dict[str, Any] = wait_for_task(task=task, label=file_prefix)

    # The finished task links the Drive folder it wrote to as .../#folders/<id>
    return metadata["destinationUris"][0].rsplit("/", 1)[-1]


def download_from_drive(
    folder_id: str, file_prefix: str, destination_dir: Path
) -> list[Path]:
    """Download files in a Drive folder whose names start with a prefix.

    Each file is deleted from Drive (permanently) once it is downloaded.
    Earth Engine splits large exports into several files with the same prefix, so a
    list is returned.

    Raises:
        FileNotFoundError: no file in the folder matches the prefix.
        requests.HTTPError: a Drive request fails.
    """
    session: AuthorizedSession = AuthorizedSession(
        credentials=ee.data.get_persistent_credentials()
    )

    # Drive's "contains" matches name prefixes word by word. startswith makes it exact
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
        # Delete only after a complete download, so a failure never loses the export
        session.delete(url=f"{DRIVE_FILES_URL}/{file['id']}").raise_for_status()
        print(f"Downloaded {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        paths.append(path)
    return paths


def export_and_download(
    image: ee.Image,
    region: ee.Geometry,
    crs: str,
    crs_transform: list[float],
    drive_folder: str,
    file_prefix: str,
    destination_dir: Path,
) -> list[Path]:
    """Export an image as GeoTIFF via a Google Drive folder and download the files."""
    folder_id: str = export_to_drive(
        image=image,
        region=region,
        crs=crs,
        crs_transform=crs_transform,
        drive_folder=drive_folder,
        file_prefix=file_prefix,
    )
    return download_from_drive(
        folder_id=folder_id, file_prefix=file_prefix, destination_dir=destination_dir
    )
