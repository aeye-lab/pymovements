"""
This module holds path specific funtions.
"""
from __future__ import annotations

import re
from pathlib import Path


def get_filepaths(
        rootpath: str | Path,
        extension: str | None = None,
        regex: re.Pattern | None = None,
) -> list[Path]:
    """
    Get filepaths from rootpath depending on extension or regular expression.
    Passing extension and regex is mutually exclusive.

    Parameters
    ----------
    rootpath: str | Path
        Root path to be traversed.
    extension: str, optional
        File extension to be filtered for.
    regex: re.Pattern, optional
        Regular expression filenames will be filtered for.

    Returns
    -------
    list[Path]

    Raises
    ------
    ValueError
        If both extension and regex is being passed.

    """
    if extension is not None and regex is not None:
        raise ValueError("extension and regex are mutually exclusive")

    rootpath = Path(rootpath)
    if not rootpath.is_dir():
        return []

    filepaths = []
    for childpath in rootpath.iterdir():
        if childpath.is_dir():
            filepaths.extend(get_filepaths(childpath, extension))
        else:
            # if extension specified and not matching, continue to next
            if extension and childpath.suffix != extension:
                continue
            # if regex specified and not matching, continue to next
            if regex and not regex.match(childpath.name):
                continue
            filepaths.append(childpath)
    return filepaths
