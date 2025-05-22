from typing import Any


def asdict_factory(
    *,
    exclude_private: bool = True,
    exclude_none: bool = True,
):
    """Return asdict_factory for being used in dataclasses.asdict().

    Parameters
    ----------
    exclude_private: bool
        Exclude attributes that start with ``_``.
    exclude_none: bool
        Exclude attributes that are either ``None`` or that are objects that evaluate to
        ``False`` (e.g., ``[]``, ``{}``, ``EyeTracker()``). Attributes of type ``bool``,
        ``int``, and ``float`` are not excluded.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of dataset definition.
    """
    def _is_included(key: str, value: Any):
        # Exclude private fields from dictionary.
        if exclude_private and key.startswith('_'):
            return False
        # Exclude fields that evaluate to False (False, None, [], {}).
        if exclude_none and not isinstance(value, (bool, int, float)) and not value:
            return False
        # Otherwise include item.
        return True

    def _dict_factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
        return dict(item for item in data if _is_included(*item))

    return _dict_factory
