#!/usr/bin/env python3
"""
Fractional Position Building Bot Launcher
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fractional_position_system import EnhancedFractionalTradingBot
    
    if __name__ == "__main__":
        bot = EnhancedFractionalTradingBot()
        bot.run()
except ImportError as e:
    print(f"Error importing fractional system: {e}")
    print("Make sure fractional_position_system.py is in the current directory")
    sys.exit(1)
