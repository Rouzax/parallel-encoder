"""Bundled configuration data.

This package exists so that ``presets.yaml`` ships inside the installed
distribution. ``encode.py`` resolves the default preset file relative to its
own location (``<install root>/config/presets.yaml``), which only works if
setuptools treats ``config`` as a package and copies the YAML alongside it.
"""
