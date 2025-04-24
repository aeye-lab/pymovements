from pymovements._utils._deprecated import DeprecatedClassMeta
from pymovements.gaze.gaze import Gaze


class GazeDataFrame(metaclass=DeprecatedClassMeta):
    _DeprecatedClassMeta__alias = Gaze
    _DeprecatedClassMeta__version_deprecated = 'v0.22.0'
    _DeprecatedClassMeta__version_removed = 'v0.27.0'
