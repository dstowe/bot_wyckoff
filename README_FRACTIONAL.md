
# Enhanced Fractional Position Building System

## Quick Start (Windows)

### 1. Run the Trading Bot
Double-click: run_fractional_bot.bat
Or command line: python run_fractional_bot.py

### 2. View Analytics
Double-click: run_analytics.bat
Or command line: python run_analytics.py --all

### 3. Monitor in Real-Time
Double-click: run_monitor.bat
Or command line: python run_monitor.py

## Configuration

Configuration examples are available in the `config_examples/` directory:
- `conservative_config.json` - Low risk, smaller positions
- `balanced_config.json` - Balanced approach (recommended)
- `aggressive_config.json` - Higher risk, larger positions
- `small_account_config.json` - Optimized for accounts under $1000

## Features

### Position Building
- Fractional share support (0.00001 precision)
- Wyckoff phase-based allocation:
  - ST: 25% initial position (testing phase)
  - SOS: 50% addition (breakout confirmation)
  - LPS: 25% completion (support test)
  - BU: Opportunistic additions on pullbacks

### Scaling Out
- Automatic profit-taking at multiple levels:
  - 10% gain: Sell 25%
  - 20% gain: Sell another 25%
  - 35% gain: Sell another 25%
  - Distribution signals: Exit remaining 25%

### Account Size Optimization
- Dynamic position sizing based on account value
- Preserves minimum cash balance that scales with growth
- Multiple account support (Cash, Margin, IRA)

## Directory Structure
```
C:\bot_wyckoff/
├── data/                    # Database and session files
├── logs/                    # Trading logs
├── reports/                 # Analytics reports
├── config_examples/         # Configuration templates
├── fractional_position_system.py
├── position_building_analytics.py
├── realtime_monitor.py
├── migration_script.py
├── run_fractional_bot.py   # Python launcher
├── run_fractional_bot.bat  # Windows batch file
├── run_analytics.py        # Python launcher
├── run_analytics.bat       # Windows batch file
├── run_monitor.py          # Python launcher
└── run_monitor.bat         # Windows batch file
```

## Windows Notes

- Use .bat files for easy double-click execution
- All files use UTF-8 encoding for proper character support
- Console output automatically adapts to Windows terminal capabilities

## Database Tables

### Enhanced Tables
- `positions_enhanced`: Tracks position building progress
- `partial_sales`: Records scaling out actions
- `position_events`: Logs all position building events

## Setup Date
2025-08-07 21:27:41
