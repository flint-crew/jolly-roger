from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict


# Using same style a primary flint repo
class BaseOptions(BaseModel):
    """A base class that Options style flint classes can
    inherit from. This is derived from ``pydantic.BaseModel``,
    and can be used for validation of supplied values.

    Class derived from ``BaseOptions`` are immutable by
    default, and have the docstrings of attributes
    extracted.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        use_attribute_docstrings=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def with_options(self: Self, /, **kwargs: dict[str, Any]) -> Self:
        new_args = self.__dict__.copy()
        new_args.update(**kwargs)

        return self.__class__(**new_args)

    def _asdict(self: Self) -> dict[str, Any]:
        return self.model_dump()
