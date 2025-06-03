# Add this utility function to find the project root and training data
# Save this as particle_tracker/utils/path_utils.py

#!/usr/bin/env python3
"""
Path Utilities
==============

Utilities for finding project paths and resources.
"""

import os
from pathlib import Path
from typing import Optional

def find_project_root() -> Path:
    """Find the project root directory containing particle_tracker module."""
    
    # Start from current file and work up
    current_path = Path(__file__).resolve()
    
    # Look for particle_tracker directory going up the tree
    for parent in current_path.parents:
        particle_tracker_dir = parent / "particle_tracker"
        if particle_tracker_dir.exists() and particle_tracker_dir.is_dir():
            # Check if it's really the particle_tracker module
            if (particle_tracker_dir / "__init__.py").exists():
                return parent
    
    # Fallback: assume we're already in the project
    return Path.cwd()

def get_default_training_data_path() -> Optional[str]:
    """Get the default path to SVM training data."""
    
    try:
        project_root = find_project_root()
        training_data_path = project_root / "particle_tracker" / "resources" / "training_data" / "tdTomato_37Degree_CytoD_training_feats.csv"
        
        if training_data_path.exists():
            return str(training_data_path)
        else:
            # Log that the file doesn't exist but don't error
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Default training data not found at: {training_data_path}")
            return None
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error finding default training data path: {e}")
        return None

def get_resources_directory() -> Path:
    """Get the resources directory path."""
    project_root = find_project_root()
    return project_root / "particle_tracker" / "resources"

def get_example_data_directory() -> Path:
    """Get the example data directory path."""
    return get_resources_directory() / "example_data"

def get_training_data_directory() -> Path:
    """Get the training data directory path."""
    return get_resources_directory() / "training_data"
