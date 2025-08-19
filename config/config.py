# config/config.py - COMPLETE VERSION WITH ALL METHODS
"""
Personal Trading Configuration - SINGLE SOURCE OF TRUTH
This is the ONLY configuration class - completely self-contained
NO inheritance or dependencies on any other config files
ALL configuration parameters are defined here
"""

import os
from datetime import datetime
from typing import List, Dict, Tuple

class PersonalTradingConfig:
    """
    STANDALONE COMPLETE TRADING CONFIGURATION
    This class contains ALL configuration parameters with no dependencies
    This is the ONLY configuration source for the entire trading system
    """

    # =================================================================
    # CORE TRADING PARAMETERS - ALL SELF-CONTAINED
    # =================================================================

    # Database
    DATABASE_PATH = "data/trading_data.db"
    
    # Account configurations
    ACCOUNT_CONFIGURATIONS = {
        'CASH': {
            'enabled': True,
            'day_trading_enabled': True,
            'options_enabled': True,
            'max_position_size': 0.25,  # 25% of account
            'min_trade_amount': 6.00,
            'max_trade_amount': 10000,
            'pdt_protection': False
        },
        'MARGIN': {
            'enabled': True,
            'day_trading_enabled': True,
            'options_enabled': True,
            'max_position_size': 0.25,  # 20% of account
            'min_trade_amount': 6.00,
            'max_trade_amount': 15000,
            'pdt_protection': True,
            'min_account_value_for_pdt': 25000
        },
        'IRA': {
            'enabled': False,  # Disabled by default
            'day_trading_enabled': False,
            'options_enabled': False,
            'max_position_size': 0.15,  # 15% of account
            'min_trade_amount': 100,
            'max_trade_amount': 5000,
            'pdt_protection': False
        },
        'ROTH': {
            'enabled': False,  # Disabled by default
            'day_trading_enabled': False,
            'options_enabled': False,
            'max_position_size': 0.15,  # 15% of account
            'min_trade_amount': 100,
            'max_trade_amount': 5000,
            'pdt_protection': False
        }
    }
    
    # Position management
    MAX_POSITIONS_TOTAL = 10
    MAX_POSITION_VALUE_PERCENT = 0.25  # 25% of account
    MIN_POSITION_VALUE = 5.00
    
    # Gap trading parameters
    GAP_MIN_SIZE = 0.01  # 1% minimum gap
    GAP_VOLUME_MULTIPLIER = 1.5  # 150% of average volume
    GAP_ENVIRONMENT_THRESHOLD = 0.15  # 15% of stocks must have gaps
    
    # Risk management
    STOP_LOSS_PERCENT = 0.05  # 5% stop loss
    TAKE_PROFIT_PERCENT = 0.10  # 10% take profit
    
    # Trading hours
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0
    
    # Watchlist symbols (example)
    WATCHLIST_SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'META', 'NVDA', 'NFLX', 'SPY', 'QQQ'
    ]
    
        
    # OPTIMIZATION 2: Market Regime Analysis Configuration
    REGIME_ANALYSIS = {
        'enabled': True,
        'update_frequency_hours': 1,
        'trend_ma_fast': 50,
        'trend_ma_slow': 200,
        'vix_thresholds': {
            'low': 20.0,
            'medium': 25.0,
            'high': 40.0,
            'crisis': 50.0
        },
        'position_adjustments': {
            'bear_market_reduction': 0.5,
            'high_vol_scaling_base': 30.0,
            'sector_weight_boost': 1.5
        }
    }
    
            
    
    # ENHANCEMENT: Wyckoff Reaccumulation Configuration
    REACCUMULATION_SETTINGS = {
        'enabled': True,
        'max_additions_per_position': 3,
        'max_additions_per_day': 3,
        'min_days_between_additions': 2,
        'max_position_size_multiplier': 2.0,
        'min_addition_percentage': 0.1,
        'max_addition_percentage': 0.5,
        'detection_parameters': {
            'reaccumulation_lookback': 30,
            'support_test_tolerance': 0.02,
            'volume_decline_threshold': 0.7,
            'ranging_threshold': 0.05
        }
    }
    
    def __init__(self):
        """Initialize configuration"""
        pass
    
    def get_account_config(self, account_type: str) -> Dict:
        """Get configuration for a specific account type"""
        account_key = account_type.upper().replace(' ACCOUNT', '')
        return self.ACCOUNT_CONFIGURATIONS.get(account_key, {})
    
    def is_account_enabled(self, account_type: str) -> bool:
        """Check if an account type is enabled for trading"""
        config = self.get_account_config(account_type)
        return config.get('enabled', False)
    
    def get_position_limit(self, account_type: str) -> float:
        """Get maximum position size for account type"""
        config = self.get_account_config(account_type)
        return config.get('max_position_size', self.MAX_POSITION_VALUE_PERCENT)
    
    def get_trade_limits(self, account_type: str) -> Tuple[float, float]:
        """Get min and max trade amounts for account type"""
        config = self.get_account_config(account_type)
        min_amount = config.get('min_trade_amount', self.MIN_POSITION_VALUE)
        max_amount = config.get('max_trade_amount', float('inf'))
        return min_amount, max_amount
    
    def can_day_trade(self, account_type: str, account_value: float = 0) -> bool:
        """Check if account can perform day trading"""
        config = self.get_account_config(account_type)
        
        if not config.get('day_trading_enabled', False):
            return False
        
        # Cash accounts can always day trade (using settled funds)
        if account_type.upper() in ['CASH']:
            return True
        
        # For margin accounts, check PDT requirements
        if config.get('pdt_protection', False):
            min_value = config.get('min_account_value_for_pdt', 25000)
            return account_value >= min_value
        
        return True
    
    def __str__(self) -> str:
        return f"PersonalTradingConfig(accounts: {len(self.ACCOUNT_CONFIGURATIONS)})"
    
    def __repr__(self) -> str:
        return self.__str__()