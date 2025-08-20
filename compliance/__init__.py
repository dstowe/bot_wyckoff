# compliance/__init__.py
"""
Compliance Package
=================
Trading compliance functionality including day trade checking and regulatory compliance

This package provides:
- Day trade compliance checking using real account data
- Regulatory compliance validation
- Trade restriction enforcement
"""

from .day_trade_checker import RealAccountDayTradeChecker

__all__ = [
    'RealAccountDayTradeChecker'
]