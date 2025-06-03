#!/usr/bin/env python3
"""
Debug Test Script
================

Test script to debug import and initialization issues step by step.
"""

import sys
import logging
from pathlib import Path

def test_imports():
    """Test all imports step by step."""
    print("Testing imports...")

    # Test PyQt6
    try:
        from PyQt6.QtWidgets import QApplication
        print("✓ PyQt6 imported successfully")
    except ImportError as e:
        print(f"✗ PyQt6 import failed: {e}")
        return False

    # Test pyqtgraph
    try:
        import pyqtgraph as pg
        print("✓ pyqtgraph imported successfully")
    except ImportError as e:
        print(f"✗ pyqtgraph import failed: {e}")
        return False

    # Test scientific libraries
    try:
        import numpy as np
        import pandas as pd
        print("✓ numpy and pandas imported successfully")
    except ImportError as e:
        print(f"✗ numpy/pandas import failed: {e}")
        return False

    # Test optional libraries
    try:
        import scipy
        print("✓ scipy imported successfully")
    except ImportError as e:
        print(f"⚠ scipy not available: {e}")

    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"⚠ scikit-learn not available: {e}")

    try:
        import skimage
        print("✓ scikit-image imported successfully")
    except ImportError as e:
        print(f"⚠ scikit-image not available: {e}")

    return True

def test_core_modules():
    """Test core module imports."""
    print("\nTesting core module imports...")

    try:
        from particle_tracker.utils.logging_config import setup_logging
        print("✓ logging_config imported successfully")
    except ImportError as e:
        print(f"✗ logging_config import failed: {e}")
        return False

    try:
        from particle_tracker.utils.config_manager import ConfigManager
        print("✓ config_manager imported successfully")
    except ImportError as e:
        print(f"✗ config_manager import failed: {e}")
        return False

    try:
        from particle_tracker.core.data_manager import DataManager
        print("✓ data_manager imported successfully")
    except ImportError as e:
        print(f"✗ data_manager import failed: {e}")
        return False

    try:
        from particle_tracker.core.analysis_engine import AnalysisEngine
        print("✓ analysis_engine imported successfully")
    except ImportError as e:
        print(f"✗ analysis_engine import failed: {e}")
        return False

    try:
        from particle_tracker.core.project_manager import ProjectManager
        print("✓ project_manager imported successfully")
    except ImportError as e:
        print(f"✗ project_manager import failed: {e}")
        return False

    return True

def test_gui_modules():
    """Test GUI module imports."""
    print("\nTesting GUI module imports...")

    try:
        from particle_tracker.gui.visualization_widget import VisualizationWidget
        print("✓ visualization_widget imported successfully")
    except ImportError as e:
        print(f"✗ visualization_widget import failed: {e}")
        return False

    try:
        from particle_tracker.gui.parameter_panels import ParameterPanelManager
        print("✓ parameter_panels imported successfully")
    except ImportError as e:
        print(f"✗ parameter_panels import failed: {e}")
        return False

    try:
        from particle_tracker.gui.data_browser import DataBrowserWidget
        print("✓ data_browser imported successfully")
    except ImportError as e:
        print(f"✗ data_browser import failed: {e}")
        return False

    try:
        from particle_tracker.gui.analysis_control import AnalysisControlWidget
        print("✓ analysis_control imported successfully")
    except ImportError as e:
        print(f"✗ analysis_control import failed: {e}")
        return False

    try:
        from particle_tracker.gui.logging_widget import LoggingWidget
        print("✓ logging_widget imported successfully")
    except ImportError as e:
        print(f"✗ logging_widget import failed: {e}")
        return False

    try:
        from particle_tracker.gui.main_window import MainWindow
        print("✓ main_window imported successfully")
    except ImportError as e:
        print(f"✗ main_window import failed: {e}")
        return False

    return True

def test_app_creation():
    """Test creating the main application."""
    print("\nTesting application creation...")

    try:
        from particle_tracker.app import ParticleTrackingApp
        print("✓ ParticleTrackingApp imported successfully")
    except ImportError as e:
        print(f"✗ ParticleTrackingApp import failed: {e}")
        return False

    try:
        # Create QApplication first
        from PyQt6.QtWidgets import QApplication
        app = QApplication([])
        print("✓ QApplication created successfully")
    except Exception as e:
        print(f"✗ QApplication creation failed: {e}")
        return False

    try:
        # Try to create the main app
        particle_app = ParticleTrackingApp(['test'], debug=True)
        print("✓ ParticleTrackingApp created successfully")
        return True
    except Exception as e:
        print(f"✗ ParticleTrackingApp creation failed: {e}")
        return False

def test_colormap_fix():
    """Test if the colormap fix worked."""
    print("\nTesting colormap functionality...")

    try:
        import pyqtgraph as pg

        # Test different colormaps
        colormap_names = ['viridis', 'plasma', 'inferno', 'magma', 'gray', 'hot']

        working_colormaps = []
        failed_colormaps = []

        for name in colormap_names:
            try:
                colormap = pg.colormap.get(name)
                working_colormaps.append(name)
                print(f"✓ {name} colormap works")
            except Exception as e:
                failed_colormaps.append(name)
                print(f"✗ {name} colormap failed: {e}")

        print(f"\nWorking colormaps: {working_colormaps}")
        print(f"Failed colormaps: {failed_colormaps}")

        if failed_colormaps:
            print("⚠ Some colormaps failed, but visualization widget should handle this gracefully")

        return True

    except Exception as e:
        print(f"✗ Colormap testing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Particle Tracker Debug Test")
    print("=" * 40)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test core modules
    if not test_core_modules():
        all_passed = False

    # Test GUI modules
    if not test_gui_modules():
        all_passed = False

    # Test colormap fix
    if not test_colormap_fix():
        all_passed = False

    # Test app creation
    if not test_app_creation():
        all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! The application should work now.")
        print("\nYou can now try running: python main.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install scipy scikit-learn scikit-image")
        print("- Check pyqtgraph version: pip install --upgrade pyqtgraph")
        print("- Ensure all __init__.py files exist in the package directories")

if __name__ == "__main__":
    main()
