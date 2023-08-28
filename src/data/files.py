from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


_cache_root: Path = Path(".cache")


def set_cache_root(root: Union[str, Path], create=False):
    global _cache_root
    _cache_root = Path(root)
    if create and not _cache_root.exists():
        _cache_root.mkdir(parents=True, exist_ok=True)


def get_cache_root() -> Path:
    global _cache_root
    return _cache_root


def get_file_path(name: Union[str, Path], root=None, ext=".h5") -> Path:
    """From a base name, get the Path object that specifies the 'ext' file in the '.cache' directory."""
    if root is None:
        root = get_cache_root()

    root_dir = Path(root)
    if not ext.startswith("."):
        ext = "." + ext
    path = Path(str(name).replace(ext, "") + ext)

    if root_dir not in path.parents:
        path = root_dir / path
    if not path.parent.exists():
        path.parent.mkdir()

    return path


def delete_files(paths: Union[str, Path, list[Union[str, Path]]]):
    """Delete the files in the list of paths."""
    if isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, Path):
        paths = [paths]

    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        if path.exists():
            path.unlink(missing_ok=True)


class cd(object):
    """Context manager for changing the current working directory"""

    def __init__(self, new_path, with_logs=True):
        self.new_path = os.path.expanduser(new_path)
        self.with_logs = with_logs

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.new_path)
        if self.with_logs:
            logger.info("in " + self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        if self.with_logs:
            logger.info("out " + self.new_path)
            logger.info("in " + self.savedPath)
