#!/usr/bin/env python3
"""
Real-Time Monitor Launcher
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from realtime_monitor import main
    
    if __name__ == "__main__":
        sys.exit(main())
except ImportError as e:
    print(f"Error importing monitor: {e}")
    print("Make sure realtime_monitor.py is in the current directory")
    sys.exit(1)
