# -*- coding: utf-8 -*-
"""
Enhanced Exit Strategy Configuration 🎯
Customize exit strategy behavior for your trading style
"""

# VIX-based volatility thresholds
VOLATILITY_THRESHOLDS = {
    'LOW': 20.0,      # VIX below 20 = low volatility
    'MEDIUM': 25.0,   # VIX 20-25 = medium volatility  
    'HIGH': 35.0,     # VIX 25-35 = high volatility
    'CRISIS': 50.0    # VIX above 35 = crisis volatility
}

# Base profit targets (before VIX adjustments)
BASE_PROFIT_TARGETS = [
    {'gain_pct': 0.06, 'sell_pct': 0.15},  # 6% gain → sell 15%
    {'gain_pct': 0.12, 'sell_pct': 0.20},  # 12% gain → sell 20%  
    {'gain_pct': 0.20, 'sell_pct': 0.25},  # 20% gain → sell 25%
    {'gain_pct': 0.30, 'sell_pct': 0.40}   # 30% gain → sell 40%
]

# CORE FEATURE: VIX-based adjustments
VIX_ADJUSTMENTS = {
    'LOW_VIX': {
        'threshold': 20.0,
        'gain_multiplier': 1.33,  # Extend targets: 6%→8%, 12%→16%, 20%→27%, 30%→40%
        'sell_multiplier': 0.9,   # Sell less aggressively
        'description': 'Low volatility - let profits run longer'
    },
    'MEDIUM_VIX': {
        'threshold': 25.0,
        'gain_multiplier': 1.0,   # Keep base targets
        'sell_multiplier': 1.0,
        'description': 'Medium volatility - use base strategy'
    },
    'HIGH_VIX': {
        'threshold': 35.0,
        'gain_multiplier': 0.67,  # Tighten targets: 6%→4%, 12%→8%, 20%→13%, 30%→20%
        'sell_multiplier': 1.2,   # Sell more aggressively
        'description': 'High volatility - take profits early'
    },
    'CRISIS_VIX': {
        'threshold': 50.0,
        'gain_multiplier': 0.5,   # Very tight targets: 6%→3%, 12%→6%, 20%→10%, 30%→15%
        'sell_multiplier': 1.5,   # Sell very aggressively
        'description': 'Crisis volatility - preserve capital'
    }
}

# Wyckoff phase-specific adjustments
WYCKOFF_ADJUSTMENTS = {
    'ST': {'gain_multiplier': 0.8, 'reasoning': 'Spring entry - take profits early'},
    'SOS': {'gain_multiplier': 1.3, 'reasoning': 'SOS entry - let profits run'},
    'LPS': {'gain_multiplier': 1.1, 'reasoning': 'LPS entry - moderate profit taking'},
    'BU': {'gain_multiplier': 0.9, 'reasoning': 'Backup entry - cautious profit taking'},
    'Creek': {'gain_multiplier': 0.7, 'reasoning': 'Creek entry - take quick profits'}
}

# Time-based adjustments
TIME_ADJUSTMENTS = [
    {'max_days': 3, 'multiplier': 0.9, 'reasoning': 'Short term - quick profits'},
    {'max_days': 7, 'multiplier': 1.0, 'reasoning': 'Normal timeframe'},
    {'max_days': 21, 'multiplier': 1.1, 'reasoning': 'Medium term - let profits run'},
    {'max_days': 60, 'multiplier': 1.2, 'reasoning': 'Long term - extend targets'},
    {'max_days': 999, 'multiplier': 1.15, 'reasoning': 'Very long term - start scaling'}
]

# Emergency exit conditions
EMERGENCY_CONDITIONS = {
    'STOP_LOSS_PERCENT': -0.15,      # Exit if down 15%
    'VIX_EMERGENCY_LEVEL': 50.0,     # Emergency exits if VIX > 50
    'MAX_DRAWDOWN_PERCENT': -0.20,   # Emergency if position down 20%
}
