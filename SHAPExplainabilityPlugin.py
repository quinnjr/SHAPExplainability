"""
SHAPExplainabilityPlugin — PluMA-convention entry point.

PluMA's Python plugin loader expects a file `<Name>Plugin.py` containing a
class `<Name>Plugin` with input(filename) / run() / output(filename)
methods. This module provides that wrapper around the core
SHAPExplainability class.

Parameter file format: tab-separated key-value pairs (PyIO convention).
See parameters.shap.txt for keys.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
"""

from __future__ import annotations

from SHAPExplainability import SHAPExplainability


class SHAPExplainabilityPlugin(SHAPExplainability):
    """PluMA-convention wrapper for SHAPExplainability.

    Inherits all behavior from SHAPExplainability. Its only purpose is
    to satisfy PluMA's file-name/class-name convention so the plugin
    can be loaded by PluMA's Python plugin loader without modification.
    """

    pass
