#!/usr/bin/env python3
"""
Enhanced Wyckoff Analyzer
=========================
Extracted from fractional_position_system.py

Provides advanced Wyckoff warning signal detection for exit management:
- UTAD (Upthrust After Distribution) detection
- SOW (Sign of Weakness) detection  
- Volume divergence analysis
- Context-based support break detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class WyckoffWarningSignal:
    """Advanced Wyckoff warning signals for exits"""
    symbol: str
    signal_type: str  # 'UTAD', 'SOW', 'VOL_DIVERGENCE', 'CONTEXT_STOP'
    strength: float
    price: float
    key_level: float  # Support/resistance level
    volume_data: Dict
    context: str
    urgency: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


class EnhancedWyckoffAnalyzer:
    """Enhanced Wyckoff analyzer with advanced warning signals - FIXED VERSION"""
    
    def __init__(self, logger):
        """
        Initialize the Enhanced Wyckoff Analyzer
        
        Args:
            logger: Logger instance for debug/error messages
        """
        self.logger = logger
    
    def analyze_advanced_warnings(self, symbol: str, data: pd.DataFrame, 
                                current_price: float, entry_data: Dict) -> List[WyckoffWarningSignal]:
        """
        Analyze for advanced Wyckoff warning signals - FIXED symbol handling
        
        Args:
            symbol: Stock symbol to analyze
            data: OHLCV pandas DataFrame
            current_price: Current stock price
            entry_data: Dictionary containing entry information including symbol and entry_phase
        
        Returns:
            List of WyckoffWarningSignal objects detected
        """
        warnings = []
        
        try:
            # FIXED: Extract actual symbol from entry_data if needed
            actual_symbol = entry_data.get('symbol', symbol)
            if '_' in actual_symbol:
                actual_symbol = actual_symbol.split('_')[0]
            
            self.logger.debug(f"Analyzing Wyckoff warnings for {actual_symbol}")
            
            # Only proceed if we have enough data
            if len(data) < 10:
                self.logger.debug(f"Insufficient data for {actual_symbol} analysis")
                return warnings
            
            # 1. UTAD - Upthrust After Distribution - NOW IMPLEMENTED
            utad_signal = self._detect_utad(actual_symbol, data, current_price)
            if utad_signal:
                warnings.append(utad_signal)
            
            # 2. SOW - Sign of Weakness  
            sow_signal = self._detect_sow(actual_symbol, data, current_price)
            if sow_signal:
                warnings.append(sow_signal)
            
            # 3. Volume Divergence
            vol_div_signal = self._detect_volume_divergence(actual_symbol, data, current_price)
            if vol_div_signal:
                warnings.append(vol_div_signal)
            
            # 4. Context-based support breaks
            context_signal = self._detect_context_breaks(actual_symbol, data, current_price, entry_data)
            if context_signal:
                warnings.append(context_signal)
                
        except Exception as e:
            self.logger.error(f"Error analyzing warnings for {symbol}: {e}")
        
        return warnings
    
    def _detect_utad(self, symbol: str, data: pd.DataFrame, current_price: float) -> Optional[WyckoffWarningSignal]:
        """
        FIXED: Detect Upthrust After Distribution (UTAD) pattern
        
        UTAD characteristics:
        - New high on lower volume followed by weakness
        - Indicates potential distribution phase
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            current_price: Current stock price
            
        Returns:
            WyckoffWarningSignal if UTAD detected, None otherwise
        """
        if len(data) < 20:
            return None
        
        try:
            # Look for UTAD pattern: new high on lower volume followed by weakness
            recent_data = data.tail(15)
            high_20 = data['High'].tail(20).max()
            is_near_high = current_price >= high_20 * 0.995
            
            if not is_near_high:
                return None
            
            # Check for volume characteristics of UTAD
            recent_volume = recent_data['Volume'].tail(5).mean()
            earlier_volume = data['Volume'].tail(20).head(10).mean()
            
            # UTAD shows lower volume on new highs
            if recent_volume < earlier_volume * 0.8:
                # Check for subsequent weakness
                last_5_close = recent_data['Close'].tail(5)
                first_close = last_5_close.iloc[0]
                last_close = last_5_close.iloc[-1]
                
                # Price should be declining after the high
                if last_close < first_close * 0.98:
                    strength = min(0.9, (earlier_volume / recent_volume - 1) * 0.5)
                    
                    return WyckoffWarningSignal(
                        symbol=symbol,
                        signal_type='UTAD',
                        strength=strength,
                        price=current_price,
                        key_level=high_20,
                        volume_data={'recent_vol': recent_volume, 'earlier_vol': earlier_volume},
                        context=f"UTAD: New high on weak volume, subsequent decline",
                        urgency='HIGH'
                    )
        except Exception as e:
            self.logger.debug(f"UTAD detection error for {symbol}: {e}")
        
        return None
        
    def _detect_sow(self, symbol: str, data: pd.DataFrame, current_price: float) -> Optional[WyckoffWarningSignal]:
        """
        Detect Sign of Weakness (SOW)
        
        SOW characteristics:
        - Heavy selling pressure on down days
        - Volume expansion on weakness
        - Indicates potential distribution
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            current_price: Current stock price
            
        Returns:
            WyckoffWarningSignal if SOW detected, None otherwise
        """
        if len(data) < 15:
            return None
        
        try:
            recent_data = data.tail(10)
            up_days = recent_data[recent_data['Close'] > recent_data['Close'].shift(1)]
            down_days = recent_data[recent_data['Close'] < recent_data['Close'].shift(1)]
            
            if len(up_days) < 3 or len(down_days) < 2:
                return None
            
            avg_up_volume = up_days['Volume'].mean()
            avg_down_volume = down_days['Volume'].mean()
            
            if avg_down_volume > avg_up_volume * 1.3:
                volume_ratio = avg_down_volume / avg_up_volume
                strength = min(0.8, (volume_ratio - 1.0) * 0.3)
                
                return WyckoffWarningSignal(
                    symbol=symbol,
                    signal_type='SOW',
                    strength=strength,
                    price=current_price,
                    key_level=data['Low'].tail(10).min(),
                    volume_data={'up_vol': avg_up_volume, 'down_vol': avg_down_volume},
                    context=f"Heavy selling on weakness, vol ratio: {volume_ratio:.2f}",
                    urgency='MEDIUM'
                )
        except Exception as e:
            self.logger.debug(f"SOW detection error for {symbol}: {e}")
        
        return None
    
    def _detect_volume_divergence(self, symbol: str, data: pd.DataFrame, current_price: float) -> Optional[WyckoffWarningSignal]:
        """
        Detect volume divergence on new highs
        
        Volume divergence characteristics:
        - New price highs with declining volume
        - Indicates weakening buying pressure
        - Warning sign for potential reversal
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            current_price: Current stock price
            
        Returns:
            WyckoffWarningSignal if volume divergence detected, None otherwise
        """
        if len(data) < 20:
            return None
        
        try:
            recent_high = data['High'].tail(20).max()
            is_near_high = current_price >= recent_high * 0.99
            
            if not is_near_high:
                return None
            
            high_days = data[data['High'] >= data['High'].rolling(10).max()]
            
            if len(high_days) < 3:
                return None
            
            recent_high_vol = high_days['Volume'].tail(2).mean()
            earlier_high_vol = high_days['Volume'].head(-2).mean() if len(high_days) > 2 else recent_high_vol
            
            if recent_high_vol < earlier_high_vol * 0.7:
                divergence_strength = 1.0 - (recent_high_vol / earlier_high_vol)
                
                return WyckoffWarningSignal(
                    symbol=symbol,
                    signal_type='VOL_DIVERGENCE',
                    strength=divergence_strength,
                    price=current_price,
                    key_level=recent_high,
                    volume_data={'recent_vol': recent_high_vol, 'earlier_vol': earlier_high_vol},
                    context=f"Volume declining on new highs: {divergence_strength:.2f}",
                    urgency='MEDIUM'
                )
        except Exception as e:
            self.logger.debug(f"Volume divergence error for {symbol}: {e}")
        
        return None
    
    def _detect_context_breaks(self, symbol: str, data: pd.DataFrame, current_price: float, 
                             entry_data: Dict) -> Optional[WyckoffWarningSignal]:
        """
        Detect breaks of key Wyckoff context levels
        
        Context breaks vary by entry phase:
        - ST/Creek: Support level breaks
        - SOS/BU: Break below entry level
        - LPS: Support test failures
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            current_price: Current stock price
            entry_data: Entry information including phase and avg_cost
            
        Returns:
            WyckoffWarningSignal if context break detected, None otherwise
        """
        try:
            entry_phase = entry_data.get('entry_phase', '')
            entry_price = entry_data.get('avg_cost', current_price)
            
            if entry_phase in ['ST', 'Creek']:
                support_level = data['Low'].tail(20).min()
                critical_level = support_level * 1.02
            elif entry_phase in ['SOS', 'BU']:
                support_level = entry_price * 0.95
                critical_level = support_level * 1.01
            elif entry_phase == 'LPS':
                support_level = data['Low'].tail(30).min()
                critical_level = support_level * 1.015
            else:
                return None
            
            if current_price < critical_level:
                break_severity = (critical_level - current_price) / critical_level
                
                return WyckoffWarningSignal(
                    symbol=symbol,
                    signal_type='CONTEXT_STOP',
                    strength=min(1.0, break_severity * 5),
                    price=current_price,
                    key_level=support_level,
                    volume_data={'break_severity': break_severity},
                    context=f"{entry_phase} support broken at {support_level:.2f}",
                    urgency='HIGH' if break_severity > 0.02 else 'MEDIUM'
                )
        except Exception as e:
            self.logger.debug(f"Context breaks detection error for {symbol}: {e}")
        
        return None