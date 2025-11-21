import os
import re
import glob
import shutil
import tarfile
import zipfile
import requests
import gdown
from urllib.parse import urlencode
from typing import Optional, Tuple


# Archive extractors

def is_archive(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"))

def _safe_join(base: str, *paths: str) -> str:
    final_path = os.path.abspath(os.path.join(base, *paths))
    base_abs = os.path.abspath(base)
    if not (final_path == base_abs or final_path.startswith(base_abs + os.sep)):
        raise Exception("Attempted Path Traversal in Archive")
    return final_path

def _path_parts(p: str):
    return os.path.normpath(p).split(os.sep)

def _should_extract(
    path: str,
    include_patterns: Optional[Tuple[str, ...]],
    exclude_subdirs: Optional[Tuple[str, ...]]
) -> bool:
    """
    True if 'path' should be extracted.
    - include_patterns=None -> allow all; else require any substring match.
    - exclude_subdirs: skip if any path component equals one of these.
    """
    if include_patterns is not None and not any(pat in path for pat in include_patterns):
        return False
    if exclude_subdirs:
        parts = set(_path_parts(path))
        if any(ex in parts for ex in exclude_subdirs):
            return False
    return True

def _safe_extract_zip(zf: zipfile.ZipFile, out_dir: str) -> None:
    for member in zf.infolist():
        dest = _safe_join(out_dir, member.filename)
        if member.is_dir():
            os.makedirs(dest, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(1 << 15)
                    if not chunk:
                        break
                    dst.write(chunk)


def _safe_extract_tar(tf: tarfile.TarFile,
                        out_dir: str,
                        include_patterns: Optional[Tuple[str, ...]] = ("RealBlur-J", "RealBlur_J"),
                        exclude_subdirs: Optional[Tuple[str, ...]] = ("Anaglyph", "gif", "kernel")
                    ) -> None:
    
    for m in tf.getmembers():
        # filter by name; directories may have no fileobj
        if not _should_extract(m.name, include_patterns, exclude_subdirs):
            continue

        dest = _safe_join(out_dir, m.name)
        if m.isdir():
            os.makedirs(dest, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            f = tf.extractfile(m)
            if f:
                with f, open(dest, "wb") as dst:
                    while True:
                        chunk = f.read(1 << 15)
                        if not chunk:
                            break
                        dst.write(chunk)



def safe_extract_archive(
    archive_path: str,
    extract_dir: str,
    delete_after: bool = True,
    include_patterns: Optional[Tuple[str, ...]] = ("RealBlur-J", "RealBlur_J"),
    exclude_subdirs: Optional[Tuple[str, ...]] = ("Anaglyph", "gif", "kernel")
) -> None:
    """
    Safely extract an archive into extract_dir with include/exclude filters.
    - include_patterns=None -> extract everything.
    - exclude_subdirs filters out unwanted subfolders anywhere in the path.
    """
    os.makedirs(extract_dir, exist_ok=True)
    lower = archive_path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract_zip(zf, extract_dir)
    else:
        mode = "r"
        if lower.endswith((".tar.gz", ".tgz")):
            mode = "r:gz"
        elif lower.endswith((".tar.bz2", ".tbz2")):
            mode = "r:bz2"
        elif lower.endswith((".tar.xz", ".txz")):
            mode = "r:xz"
        with tarfile.open(archive_path, mode) as tf:
            _safe_extract_tar(tf, extract_dir, include_patterns, exclude_subdirs)

    if delete_after:
        try:
            os.remove(archive_path)
            print(f"Deleted archive: {archive_path}")
        except Exception as e:
            print(f" Could not delete {archive_path}: {e}")




# Yandex Disk Downloader
def download_from_yadisk(short_url: str, filename: str, target_dir: str) -> str:
    """
    Download a publicly shared Yandex Disk file to target_dir/filename.
    Returns the full path to the downloaded file.
    """
    os.makedirs(target_dir, exist_ok=True)
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    final_url = base_url + urlencode(dict(public_key=short_url))

    r = requests.get(final_url, timeout=60)
    r.raise_for_status()
    href = r.json().get("href")
    if not href:
        raise RuntimeError("Yandex Disk response missing 'href' download link.")

    out_path = os.path.join(target_dir, filename)
    with requests.get(href, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(1 << 15):
                if chunk:
                    f.write(chunk)
    return out_path

def download_and_extract_from_yadisk(short_url: str, filename: str, data_dir: str = "data") -> str:
    """
    Download from Yandex Disk and extract into `data_dir/` if it's an archive.
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = download_from_yadisk(short_url, filename, data_dir)
    if is_archive(local_path):
        print(f"{filename} is being extracted ...")
        safe_extract_archive(local_path, data_dir)

    print(f"{filename} downloaded and extracted ")
    return local_path

# Google Drive Downloader

def download_and_extract_from_gdrive_file(share_url: str,
                                        filename: str,
                                        data_dir: str = "data",
                                        subfolder: str = "RealBlur",
                                        keep_only_realblur_j: bool = True,
                                        include_patterns_if_j: Optional[Tuple[str, ...]] = ("RealBlur-J", "RealBlur_J"),
                                        exclude_subdirs: Optional[Tuple[str, ...]] = ("Anaglyph", "gif", "kernel"),
                                        ) -> str:
    """
    Save the archive directly under data_dir/subfolder and extract there.
    - If keep_only_realblur_j=True (default), only RealBlur-J is extracted.
    - If keep_only_realblur_j=False, everything is extracted, but Anaglyph/gif/kernel
      are still excluded by default (set exclude_subdirs=None to keep them).
    """
    target_dir = os.path.join(data_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    out_path = os.path.join(target_dir, filename)
    print(f"Downloading {filename} from Google Drive...")
    gdown.download(url=share_url, output=out_path, quiet=False, fuzzy=True)

    if not os.path.exists(out_path):
        raise RuntimeError(f"Failed to download {filename} from {share_url}")

    if is_archive(out_path):
        print(f"{filename} is being extracted ...")
        # Decide include policy
        include_patterns = include_patterns_if_j if keep_only_realblur_j else None
        safe_extract_archive(
            out_path, target_dir, delete_after=True,
            include_patterns=include_patterns,
            exclude_subdirs=exclude_subdirs
        )

    print(f"{filename} downloaded and extracted ")
    return target_dir