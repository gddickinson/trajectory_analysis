"""
Resources Package
================

Contains application resources:
- Icons: Application and UI icons
- Example Data: Sample datasets for testing
- Themes: UI themes and stylesheets
- Documentation: Embedded help files
"""

import os
from pathlib import Path

# Get the resources directory path
RESOURCES_DIR = Path(__file__).parent

# Define resource subdirectories
ICONS_DIR = RESOURCES_DIR / "icons"
EXAMPLE_DATA_DIR = RESOURCES_DIR / "example_data"
THEMES_DIR = RESOURCES_DIR / "themes"
DOCS_DIR = RESOURCES_DIR / "docs"

def get_resource_path(resource_type: str, filename: str) -> Path:
    """Get the full path to a resource file.

    Args:
        resource_type: Type of resource ("icons", "example_data", "themes", "docs")
        filename: Name of the resource file

    Returns:
        Path to the resource file
    """
    resource_dirs = {
        "icons": ICONS_DIR,
        "example_data": EXAMPLE_DATA_DIR,
        "themes": THEMES_DIR,
        "docs": DOCS_DIR
    }

    if resource_type not in resource_dirs:
        raise ValueError(f"Unknown resource type: {resource_type}")

    resource_dir = resource_dirs[resource_type]
    resource_path = resource_dir / filename

    return resource_path

def get_icon_path(icon_name: str) -> Path:
    """Get path to an icon file."""
    return get_resource_path("icons", icon_name)

def get_example_data_path(data_name: str) -> Path:
    """Get path to an example data file."""
    return get_resource_path("example_data", data_name)

def get_theme_path(theme_name: str) -> Path:
    """Get path to a theme file."""
    return get_resource_path("themes", theme_name)

def list_resources(resource_type: str) -> list:
    """List all available resources of a given type.

    Args:
        resource_type: Type of resource to list

    Returns:
        List of available resource files
    """
    resource_dirs = {
        "icons": ICONS_DIR,
        "example_data": EXAMPLE_DATA_DIR,
        "themes": THEMES_DIR,
        "docs": DOCS_DIR
    }

    if resource_type not in resource_dirs:
        return []

    resource_dir = resource_dirs[resource_type]

    if not resource_dir.exists():
        return []

    return [f.name for f in resource_dir.iterdir() if f.is_file()]

def ensure_resource_dirs():
    """Ensure all resource directories exist."""
    for resource_dir in [ICONS_DIR, EXAMPLE_DATA_DIR, THEMES_DIR, DOCS_DIR]:
        resource_dir.mkdir(parents=True, exist_ok=True)

# Create resource directories on import
ensure_resource_dirs()

__all__ = [
    "RESOURCES_DIR",
    "ICONS_DIR",
    "EXAMPLE_DATA_DIR",
    "THEMES_DIR",
    "DOCS_DIR",
    "get_resource_path",
    "get_icon_path",
    "get_example_data_path",
    "get_theme_path",
    "list_resources",
    "ensure_resource_dirs",
]

# Resources package metadata
__resources_version__ = "1.0.0"

def get_resources_info():
    """Get information about available resources."""
    return {
        "version": __resources_version__,
        "directories": {
            "icons": str(ICONS_DIR),
            "example_data": str(EXAMPLE_DATA_DIR),
            "themes": str(THEMES_DIR),
            "docs": str(DOCS_DIR)
        },
        "available_resources": {
            "icons": list_resources("icons"),
            "example_data": list_resources("example_data"),
            "themes": list_resources("themes"),
            "docs": list_resources("docs")
        }
    }