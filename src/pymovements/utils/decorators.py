"""
This module holds utils for developers and is not part of the user API.
"""
from typing import Any


def auto_str(cls):
    """
    Automatically generate __str__() to include all arguments. Can be used as a decorator.
    """
    def shorten(value: Any):
        if isinstance(value, float):
            value = f'{value:.2f}'
        return value

    def __str__(self):
        attributes = ', '.join(f'{key}={shorten(value)}' for key, value in vars(self).items())
        return f'{type(self).__name__}({attributes})'

    cls.__str__ = __str__
    return cls
