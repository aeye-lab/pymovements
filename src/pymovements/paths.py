import re
from pathlib import Path
from typing import List


def get_filepaths(
        rootpath: str,
        extension: str = None,
        regex: re.Pattern = None,
) -> List[Path]:
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
