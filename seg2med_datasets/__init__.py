"""Dataset preparation package.

Keep package initialization lightweight. Heavy dataset discovery and image IO
belong in the dataset adapters and preparation runner.
"""

from .schema import SCHEMA

__all__ = ["SCHEMA"]
