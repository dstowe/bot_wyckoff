# -*- coding: utf-8 -*-
"""
Enhanced Cash Management System for Wyckoff Trading Bot
Implements dynamic cash allocation based on account size and market regime
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging

class EnhancedCashManager:
    """
    Advanced cash management system that dynamically adjusts cash allocation
    based on market regime, account size, and Wyckoff phases
    """
    
    def __init__(self):
        # Market regime thresholds
        self.VIX_HIGH_THRESHOLD = 25.0
        self.VIX_LOW_THRESHOLD = 20.0
        self.TREND_LOOKBACK_DAYS = 20
        
        # Cash allocation ranges by market regime
        self.CASH_ALLOCATION = {
            'BULL_MARKET_LOW_VIX': 0.15,      # 15% cash in strong bull, low volatility
            'BULL_MARKET_HIGH_VIX': 0.25,     # 25% cash in bull market, high volatility  
            'RANGING_MARKET': 0.30,           # 30% cash in ranging market
            'BEAR_MARKET': 0.40,              # 40% cash in bear market
            'DISTRIBUTION_PHASE': 0.50,       # 50% cash in distribution phase
            'MARKET_CRASH': 0.80              # 80% cash in crash conditions
        }
        
        # Dynamic buffer calculation parameters
        self.BUFFER_BASE_PERCENTAGE = 0.002   # 0.2% of account value as base buffer
        self.BUFFER_MIN_AMOUNT = 25.0         # Minimum $25 buffer
        self.BUFFER_MAX_AMOUNT = 500.0        # Maximum $500 buffer
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for cash management operations"""
        logger = logging.getLogger('CashManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_market_regime(self, vix_symbol: str = "^VIX", 
                         spy_symbol: str = "SPY") -> str:
        """
        Determine current market regime based on VIX and trend analysis
        
        Returns:
            str: Market regime classification
        """
        try:
            # Get VIX data
            vix_data = yf.download(vix_symbol, period="5d", interval="1d", progress=False)
            if vix_data.empty:
                self.logger.warning("Could not fetch VIX data, defaulting to RANGING_MARKET")
                return "RANGING_MARKET"
            
            current_vix = vix_data['Close'].iloc[-1]
            
            # Check for market crash conditions
            if current_vix > 40:
                self.logger.info(f"MARKET CRASH detected - VIX: {current_vix:.2f}")
                return "MARKET_CRASH"
            
            # Get SPY trend data
            spy_data = yf.download(spy_symbol, period="30d", interval="1d", progress=False)
            if spy_data.empty:
                self.logger.warning("Could not fetch SPY data for trend analysis")
                return "RANGING_MARKET"
            
            # Calculate trend indicators
            spy_close = spy_data['Close']
            sma_20 = spy_close.rolling(window=self.TREND_LOOKBACK_DAYS).mean().iloc[-1]
            current_price = spy_close.iloc[-1]
            price_vs_sma = (current_price / sma_20 - 1) * 100
            
            # Calculate recent trend strength
            recent_returns = spy_close.pct_change(5).iloc[-1] * 100  # 5-day return
            
            # Determine market regime
            if current_vix < self.VIX_LOW_THRESHOLD:
                if price_vs_sma > 2 and recent_returns > 1:
                    regime = "BULL_MARKET_LOW_VIX"
                elif price_vs_sma > 0:
                    regime = "BULL_MARKET_LOW_VIX"
                else:
                    regime = "RANGING_MARKET"
            elif current_vix > self.VIX_HIGH_THRESHOLD:
                if price_vs_sma > 1:
                    regime = "BULL_MARKET_HIGH_VIX"
                elif price_vs_sma < -2:
                    regime = "BEAR_MARKET"
                else:
                    regime = "RANGING_MARKET"
            else:
                if price_vs_sma > 1:
                    regime = "BULL_MARKET_LOW_VIX"
                elif price_vs_sma < -2:
                    regime = "BEAR_MARKET"
                else:
                    regime = "RANGING_MARKET"
            
            self.logger.info(f"Market Regime: {regime} | VIX: {current_vix:.2f} | SPY vs SMA20: {price_vs_sma:+.2f}%")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error determining market regime: {e}")
            return "RANGING_MARKET"
    
    def detect_wyckoff_distribution_phase(self, symbol: str = "SPY") -> bool:
        """
        Detect if market is in Wyckoff distribution phase
        Look for signs of UTAD (Upthrust After Distribution) patterns
        
        Returns:
            bool: True if in distribution phase
        """
        try:
            # Get extended price data for distribution analysis
            data = yf.download(symbol, period="60d", interval="1d", progress=False)
            if len(data) < 30:
                return False
            
            close_prices = data['Close']
            volume = data['Volume']
            high_prices = data['High']
            
            # Look for distribution characteristics
            recent_highs = high_prices.rolling(window=20).max()
            current_high = high_prices.iloc[-1]
            highest_recent = recent_highs.iloc[-5:].max()
            
            # Check for multiple tests of highs with declining volume
            price_near_highs = (current_high / highest_recent) > 0.98
            
            # Volume analysis - declining volume on retests
            recent_volume = volume.iloc[-10:].mean()
            earlier_volume = volume.iloc[-30:-10].mean()
            volume_declining = recent_volume < (earlier_volume * 0.85)
            
            # Price action - lack of follow-through after highs
            recent_returns = close_prices.pct_change(5).iloc[-1]
            weak_momentum = recent_returns < 0.02  # Less than 2% in 5 days
            
            distribution_detected = price_near_highs and volume_declining and weak_momentum
            
            if distribution_detected:
                self.logger.info("Distribution phase detected - increasing cash allocation")
            
            return distribution_detected
            
        except Exception as e:
            self.logger.error(f"Error in distribution detection: {e}")
            return False
    
    def calculate_dynamic_buffer(self, account_value: float) -> float:
        """
        Calculate dynamic cash buffer based on account size
        
        Args:
            account_value: Total account value
            
        Returns:
            float: Recommended cash buffer amount
        """
        # Calculate percentage-based buffer
        percentage_buffer = account_value * self.BUFFER_BASE_PERCENTAGE
        
        # Apply min/max constraints
        dynamic_buffer = max(self.BUFFER_MIN_AMOUNT, 
                           min(percentage_buffer, self.BUFFER_MAX_AMOUNT))
        
        self.logger.info(f"Dynamic buffer calculated: ${dynamic_buffer:.2f} "
                        f"(Account: ${account_value:,.2f})")
        
        return dynamic_buffer
    
    def get_recommended_cash_allocation(self, account_value: float) -> Dict[str, float]:
        """
        Get comprehensive cash management recommendations
        
        Args:
            account_value: Total account value
            
        Returns:
            Dict with cash allocation recommendations
        """
        # Determine market regime
        market_regime = self.get_market_regime()
        
        # Check for distribution phase
        in_distribution = self.detect_wyckoff_distribution_phase()
        
        # Adjust regime if in distribution
        if in_distribution and market_regime not in ["MARKET_CRASH", "BEAR_MARKET"]:
            market_regime = "DISTRIBUTION_PHASE"
        
        # Get base cash allocation
        cash_percentage = self.CASH_ALLOCATION.get(market_regime, 0.30)
        
        # Calculate cash amounts
        recommended_cash = account_value * cash_percentage
        dynamic_buffer = self.calculate_dynamic_buffer(account_value)
        available_for_trading = account_value - recommended_cash - dynamic_buffer
        max_position_value = available_for_trading * 0.25  # 25% max position size
        
        recommendations = {
            'market_regime': market_regime,
            'in_distribution_phase': in_distribution,
            'cash_percentage': cash_percentage,
            'recommended_cash_amount': recommended_cash,
            'dynamic_buffer': dynamic_buffer,
            'available_for_trading': available_for_trading,
            'max_position_value': max_position_value,
            'max_position_percentage': (max_position_value / account_value) * 100
        }
        
        # Log recommendations
        self.logger.info(f"Cash Management Recommendations:")
        self.logger.info(f"   Market Regime: {market_regime}")
        self.logger.info(f"   Cash Allocation: {cash_percentage:.1%} (${recommended_cash:,.2f})")
        self.logger.info(f"   Dynamic Buffer: ${dynamic_buffer:.2f}")
        self.logger.info(f"   Available for Trading: ${available_for_trading:,.2f}")
        self.logger.info(f"   Max Position Size: ${max_position_value:,.2f} ({recommendations['max_position_percentage']:.1f}%)")
        
        return recommendations
    
    def should_reduce_position_sizes(self) -> Tuple[bool, float]:
        """
        Determine if position sizes should be reduced based on market conditions
        
        Returns:
            Tuple[bool, float]: (should_reduce, reduction_factor)
        """
        market_regime = self.get_market_regime()
        
        reduction_factors = {
            'BULL_MARKET_LOW_VIX': 1.0,      # No reduction
            'BULL_MARKET_HIGH_VIX': 0.8,     # 20% reduction
            'RANGING_MARKET': 0.7,           # 30% reduction
            'BEAR_MARKET': 0.5,              # 50% reduction
            'DISTRIBUTION_PHASE': 0.4,       # 60% reduction
            'MARKET_CRASH': 0.2              # 80% reduction
        }
        
        reduction_factor = reduction_factors.get(market_regime, 0.7)
        should_reduce = reduction_factor < 1.0
        
        if should_reduce:
            self.logger.info(f"Reducing position sizes by {(1-reduction_factor)*100:.0f}% "
                           f"due to {market_regime}")
        
        return should_reduce, reduction_factor
    
    def validate_trade_against_cash_rules(self, trade_value: float, 
                                        current_cash: float, 
                                        account_value: float) -> Tuple[bool, str]:
        """
        Validate if a trade respects cash management rules
        
        Args:
            trade_value: Dollar value of proposed trade
            current_cash: Current cash balance
            account_value: Total account value
            
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        recommendations = self.get_recommended_cash_allocation(account_value)
        
        # Check if trade would violate cash requirements
        cash_after_trade = current_cash - trade_value
        min_required_cash = recommendations['recommended_cash_amount'] + recommendations['dynamic_buffer']
        
        if cash_after_trade < min_required_cash:
            return False, f"Trade would leave insufficient cash: ${cash_after_trade:.2f} < ${min_required_cash:.2f} required"
        
        # Check position size limits
        if trade_value > recommendations['max_position_value']:
            return False, f"Trade exceeds max position size: ${trade_value:.2f} > ${recommendations['max_position_value']:.2f}"
        
        return True, "Trade approved"

def test_cash_manager():
    """Test the enhanced cash management system"""
    print("Testing Enhanced Cash Management System")
    print("=" * 50)
    
    cash_manager = EnhancedCashManager()
    
    # Test with different account sizes
    test_accounts = [10000, 50000, 100000, 500000]
    
    for account_value in test_accounts:
        print(f"\nTesting Account Value: ${account_value:,}")
        print("-" * 40)
        
        recommendations = cash_manager.get_recommended_cash_allocation(account_value)
        should_reduce, reduction_factor = cash_manager.should_reduce_position_sizes()
        
        print(f"Market Regime: {recommendations['market_regime']}")
        print(f"Cash Allocation: {recommendations['cash_percentage']:.1%}")
        print(f"Dynamic Buffer: ${recommendations['dynamic_buffer']:.2f}")
        print(f"Max Position: ${recommendations['max_position_value']:,.2f}")
        if should_reduce:
            print(f"Position Reduction: {(1-reduction_factor)*100:.0f}%")

if __name__ == "__main__":
    test_cash_manager()
