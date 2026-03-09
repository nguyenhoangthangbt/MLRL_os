"""Feature definition registry for reuse across experiments."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureDefinition(BaseModel):
    """A reusable feature transformation definition."""

    name: str
    source_column: str
    transform_type: str  # "lag" | "rolling_mean" | "rolling_std" | "trend" | "ratio" | "derived"
    parameters: dict[str, str | int | float] = Field(default_factory=dict)
    description: str = ""


class FeatureStore:
    """Registry of reusable feature definitions."""

    def __init__(self) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> str:
        """Register a feature definition.

        Args:
            definition: Feature definition to register.

        Returns:
            The feature name.
        """
        self._definitions[definition.name] = definition
        return definition.name

    def get(self, name: str) -> FeatureDefinition:
        """Get a feature definition by name.

        Raises:
            KeyError: If feature not found.
        """
        if name not in self._definitions:
            msg = f"Feature definition '{name}' not found"
            raise KeyError(msg)
        return self._definitions[name]

    def list_definitions(self) -> list[FeatureDefinition]:
        """List all registered feature definitions."""
        return list(self._definitions.values())

    def has(self, name: str) -> bool:
        """Check if a feature definition exists."""
        return name in self._definitions

    @property
    def count(self) -> int:
        """Number of registered feature definitions."""
        return len(self._definitions)
