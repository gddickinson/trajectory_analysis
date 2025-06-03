#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    try:
        from particle_tracker.app import ParticleTrackingApp
        app = ParticleTrackingApp(sys.argv, debug="--debug" in sys.argv)
        return app.exec()
    except ImportError as e:
        print(f"Error: {e}")
        print("Install PyQt6: pip install PyQt6")
        return 1

if __name__ == "__main__":
    sys.exit(main())
