#!/usr/bin/env python3
"""Download the ROBUST-MIPS archive from Synapse without extra dependencies.

Authentication is read from SYNAPSE_AUTH_TOKEN or, when run interactively,
entered securely with --prompt-token. The token is never written to disk.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ENTITY_ID = "syn68915165"
FILE_HANDLE_ID = "161490988"
ENTITY_VERSION = 1
FILENAME = "RobustMIPS.zip"
REPO_API = "https://repo-prod.prod.sagebase.org"
FILE_API = "https://repo-prod.prod.sagebase.org/file/v1"


def request_json(url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Synapse request failed ({error.code}): {detail}") from error


def signed_download(token: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "requestedFiles": [
            {
                "fileHandleId": FILE_HANDLE_ID,
                "associateObjectId": ENTITY_ID,
                "associateObjectType": "FileEntity",
            }
        ],
        "includeFileHandles": True,
        "includePreSignedURLs": True,
        "includePreviewPreSignedURLs": False,
    }
    result = request_json(f"{FILE_API}/fileHandle/batch", token, payload)
    item = result["requestedFiles"][0]
    if "failureCode" in item:
        raise RuntimeError(f"Synapse did not authorize the file: {item['failureCode']}")
    url = item.get("preSignedURL")
    handle = item.get("fileHandle", {})
    if not url:
        raise RuntimeError(f"Synapse response did not contain a download URL: {item.keys()}")
    return url, handle


def download_with_resume(url: str, destination: Path, expected_size: int | None) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            print("Server did not accept resume request; restarting download.", file=sys.stderr)
            offset = 0
            partial.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        downloaded = offset
        started = time.monotonic()
        with partial.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.monotonic() - started, 0.001)
                transferred = downloaded - offset
                size_text = f"/{expected_size / 2**30:.2f} GiB" if expected_size else ""
                print(
                    f"\r{downloaded / 2**30:.2f}{size_text} GiB "
                    f"({transferred / 2**20 / elapsed:.1f} MiB/s)",
                    end="",
                    flush=True,
                )
    print()
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch: downloaded {partial.stat().st_size} bytes; expected {expected_size}"
        )
    partial.replace(destination)


def checksum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/archives"))
    parser.add_argument("--prompt-token", action="store_true", help="securely prompt for a Synapse PAT")
    args = parser.parse_args()

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token and args.prompt_token and sys.stdin.isatty():
        token = getpass.getpass("Synapse personal access token: ")
    if not token:
        print(
            "Missing SYNAPSE_AUTH_TOKEN. Create a Synapse personal access token with View and "
            "Download permissions, export it in the shell, and rerun this command. The token must "
            "not be committed or pasted into chat.",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / FILENAME
    url, handle = signed_download(token)
    expected_size = int(handle["contentSize"]) if handle.get("contentSize") is not None else None
    expected_md5 = handle.get("contentMd5")
    print(f"Downloading {ENTITY_ID} v{ENTITY_VERSION} to {destination}")
    if expected_size:
        print(f"Archive size: {expected_size / 2**30:.2f} GiB")
    download_with_resume(url, destination, expected_size)
    actual_md5 = checksum(destination)
    if expected_md5 and actual_md5.lower() != expected_md5.lower():
        raise RuntimeError(f"MD5 mismatch: got {actual_md5}; expected {expected_md5}")

    metadata = {
        "project_id": "syn64023381",
        "entity_id": ENTITY_ID,
        "entity_version": ENTITY_VERSION,
        "file_handle_id": FILE_HANDLE_ID,
        "filename": FILENAME,
        "content_size": destination.stat().st_size,
        "content_md5": actual_md5,
        "synapse_file_handle": handle,
        "downloaded_at_unix": int(time.time()),
    }
    metadata_path = args.output_dir / "RobustMIPS.synapse.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Verified archive and wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
