from __future__ import annotations

from typing import Union, Any

# sphinx automatically substitutes _Resources with the Union using the | operator.
_Resources = Union[dict[str, list[dict[str, str]]], dict[str, tuple[dict[str, str], ...]]]


class _HasResourcesIndexer:
    """Helper class for :py:meth:`~pymovements.dataset.DatasetDefinition.has_resources` property.

    Provides dynamic inference on the presence of any
    :py:meth:`~pymovements.dataset.DatasetDefinition.resources`.
    """

    def __init__(self) -> None:
        self._resources: _Resources = {}

    def set_resources(self, resources: _Resources) -> None:
        """Set dataset definition resources for lookup."""
        self._resources = resources

    def __getitem__(self, key: str) -> bool:
        """Lookup if resources of specific content are set."""
        try:
            return len(self._resources[key]) > 0
        except KeyError:  # if key not in self._resources
            return False
        except TypeError:  # if self._resources[key] doesn't implement __len__
            return False

    def __bool__(self) -> bool:
        """Lookup if resources of any content are set."""
        if not self._resources:
            return False

        # Get list of resource_lists and return False in case no values().
        try:
            list_of_resource_lists = self._resources.values()
        except AttributeError:  # if values() not implemented by self._resources
            return False

        # Check if any resources are actually set in dictionary.
        for resource_list in list_of_resource_lists:
            try:
                if len(resource_list) > 0:
                    return True
            except TypeError:  # if resources_list doesn't implement __len__
                return False
        return False

    def __eq__(self, other: Any) -> bool:
        """Return self == other.

        Automatically casts to bool if compared to a boolean.
        """
        if isinstance(other, bool):  # Needed to check equality against booleans.
            return self.__bool__() == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        """Return string with boolean value wheter any resources are set."""
        return str(self.__bool__())
