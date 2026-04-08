#!/usr/bin/env python3
"""Debug import issues"""

def test_step_by_step():
    print("Testing step by step...")

    # Test basic imports first
    try:
        import numpy as np
        print("✓ numpy")
    except Exception as e:
        print(f"✗ numpy: {e}")
        return

    try:
        import pandas as pd
        print("✓ pandas")
    except Exception as e:
        print(f"✗ pandas: {e}")
        return

    # Test core modules
    try:
        from particle_tracker.core.data_manager import DataManager
        print("✓ DataManager")
    except Exception as e:
        print(f"✗ DataManager: {e}")

    # Test each analysis module individually
    try:
        from particle_tracker.analysis.detection import ParticleDetector
        print("✓ detection")
    except Exception as e:
        print(f"✗ detection: {e}")

    try:
        from particle_tracker.analysis.linking import ParticleLinker
        print("✓ linking")
    except Exception as e:
        print(f"✗ linking: {e}")

    try:
        from particle_tracker.analysis.features import FeatureCalculator
        print("✓ features")
    except Exception as e:
        print(f"✗ features: {e}")

    try:
        from particle_tracker.analysis.classification import TrajectoryClassifier
        print("✓ classification")
    except Exception as e:
        print(f"✗ classification: {e}")

if __name__ == "__main__":
    test_step_by_step()
