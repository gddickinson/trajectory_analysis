#!/usr/bin/env python3
def test_imports():
    try:
        from particle_tracker.app import ParticleTrackingApp
        print("✓ App imports successfully!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        print("Ready to run: python main.py")
    else:
        print("Fix imports first")
