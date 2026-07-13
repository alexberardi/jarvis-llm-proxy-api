"""Service version, surfaced on /health.

Read from installed package metadata rather than duplicated as a literal, so it
can't drift from pyproject.toml.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("jarvis-llm-proxy-api")
except PackageNotFoundError:  # running from a raw checkout, not pip-installed
    __version__ = "unknown"
