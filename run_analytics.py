#!/usr/bin/env python3
"""
Position Building Analytics Launcher
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from position_building_analytics import main
    
    if __name__ == "__main__":
        sys.exit(main())
except ImportError as e:
    print(f"Error importing analytics: {e}")
    print("Make sure position_building_analytics.py is in the current directory")
    sys.exit(1)
