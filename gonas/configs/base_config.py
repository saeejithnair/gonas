"""Base Configs"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

# model instances
from typing_extensions import Literal

warnings.filterwarnings("ignore", module="torchvision")


# Pretty printing class
class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):  # pylint: disable=too-few-public-methods
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


@dataclass
class FlexibleInstantiateConfig(PrintableConfig):  # pylint: disable=too-few-public-methods
    """Config class for instantiating the class specified in the _target attribute.
    Unlike InstantiateConfig, this class doesn't require the target class to take in
    the config as the first argument. Instead, the fields of the config are passed in
    as keyword arguments to the target class.
    """

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object by passing in the config fields as
        keyword arguments.
        """
        fields_dict = {field.name: getattr(self, field.name) 
                       for field in fields(self) if field.name != "_target"}

        return self._target(**fields_dict, **kwargs)
