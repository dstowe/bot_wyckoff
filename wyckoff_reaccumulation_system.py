#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wyckoff Reaccumulation Position Addition System
Enhances the fractional position system to add to existing positions during reaccumulation phases
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class ReaccumulationSignal:
    """Signal to add to an existing position during reaccumulation"""
    symbol: str
    current_position_size: float
    reaccumulation_range: Dict[str, float]  # {'support': price, 'resistance': price}
    pullback_opportunity: bool
    volume_confirmation: bool
    absorption_strength: float  # 0.0 to 1.0
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    recommended_addition_pct: float  # Percentage of original position to add
    stop_loss_level: float
    reasoning: str

@dataclass
class ReaccumulationAnalysis:
    """Complete reaccumulation analysis for a position"""
    symbol: str
    is_in_reaccumulation: bool
    reaccumulation_duration_days: int
    range_support: float
    range_resistance: float
    current_price: float
    volume_trend: str  # 'INCREASING', 'DECREASING', 'STABLE'
    absorption_indicators: List[str]
    strength_indicators: List[str]
    next_breakout_target: float
    confidence_score: float

class WyckoffReaccumulationDetector:
    """Detects and analyzes Wyckoff reaccumulation phases"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Reaccumulation detection parameters
        self.MIN_REACCUMULATION_DAYS = 5  # Minimum days to consider reaccumulation
        self.MAX_REACCUMULATION_DAYS = 45  # Maximum reasonable reaccumulation period
        self.RANGE_TIGHTNESS_THRESHOLD = 0.08  # 8% range is considered tight
        self.VOLUME_DECLINE_THRESHOLD = 0.8  # Volume 20% below average
        self.ABSORPTION_VOLUME_THRESHOLD = 1.2  # Volume 20% above average on dips
        
        # Position addition parameters
        self.MAX_ADDITION_PCT = 0.5  # Maximum 50% addition to original position
        self.PULLBACK_ENTRY_THRESHOLD = 0.25  # Enter on 25% pullback within range
        self.STOP_LOSS_BUFFER = 0.02  # 2% below support for stop loss
    
    def analyze_position_for_reaccumulation(self, symbol: str, position_data: Dict, 
                                          lookback_days: int = 60) -> ReaccumulationAnalysis:
        """Analyze if an existing position is in reaccumulation phase"""
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{lookback_days + 20}d")
            
            if len(data) < 30:
                return self._create_no_reaccumulation_analysis(symbol)
            
            current_price = data['Close'].iloc[-1]
            
            # Step 1: Identify potential reaccumulation range
            range_analysis = self._identify_reaccumulation_range(data)
            
            if not range_analysis['is_valid_range']:
                return self._create_no_reaccumulation_analysis(symbol)
            
            # Step 2: Analyze volume characteristics
            volume_analysis = self._analyze_reaccumulation_volume(data, range_analysis)
            
            # Step 3: Check for absorption and strength signs
            absorption_indicators = self._detect_absorption_signs(data, range_analysis)
            strength_indicators = self._detect_strength_signs(data, range_analysis)
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_reaccumulation_confidence(
                range_analysis, volume_analysis, absorption_indicators, strength_indicators
            )
            
            # Step 5: Determine if in valid reaccumulation
            is_in_reaccumulation = (
                range_analysis['is_valid_range'] and
                confidence_score >= 0.6 and
                len(absorption_indicators) >= 2
            )
            
            return ReaccumulationAnalysis(
                symbol=symbol,
                is_in_reaccumulation=is_in_reaccumulation,
                reaccumulation_duration_days=range_analysis['duration_days'],
                range_support=range_analysis['support'],
                range_resistance=range_analysis['resistance'],
                current_price=current_price,
                volume_trend=volume_analysis['trend'],
                absorption_indicators=absorption_indicators,
                strength_indicators=strength_indicators,
                next_breakout_target=range_analysis['resistance'] * 1.05,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing reaccumulation for {symbol}: {e}")
            return self._create_no_reaccumulation_analysis(symbol)
    
    def _identify_reaccumulation_range(self, data: pd.DataFrame) -> Dict:
        """Identify the reaccumulation range boundaries"""
        try:
            # Look for consolidation in recent 20-45 day period
            recent_data = data.tail(45)
            
            # Find highest high and lowest low in recent period
            highest_high = recent_data['High'].max()
            lowest_low = recent_data['Low'].min()
            
            # Calculate range tightness
            range_size = (highest_high - lowest_low) / lowest_low
            
            # Check if range is reasonable for reaccumulation
            if range_size > self.RANGE_TIGHTNESS_THRESHOLD:
                return {'is_valid_range': False}
            
            # Find support and resistance levels
            support_tests = recent_data[recent_data['Low'] <= lowest_low * 1.02]
            resistance_tests = recent_data[recent_data['High'] >= highest_high * 0.98]
            
            # Need multiple tests of both levels
            if len(support_tests) < 2 or len(resistance_tests) < 2:
                return {'is_valid_range': False}
            
            # Calculate duration
            first_range_test = min(support_tests.index[0], resistance_tests.index[0])
            duration_days = (recent_data.index[-1] - first_range_test).days
            
            if duration_days < self.MIN_REACCUMULATION_DAYS:
                return {'is_valid_range': False}
            
            return {
                'is_valid_range': True,
                'support': lowest_low,
                'resistance': highest_high,
                'range_size_pct': range_size,
                'duration_days': duration_days,
                'support_tests': len(support_tests),
                'resistance_tests': len(resistance_tests)
            }
            
        except Exception as e:
            self.logger.debug(f"Error identifying reaccumulation range: {e}")
            return {'is_valid_range': False}
    
    def _analyze_reaccumulation_volume(self, data: pd.DataFrame, range_analysis: Dict) -> Dict:
        """Analyze volume characteristics during reaccumulation"""
        try:
            recent_data = data.tail(30)
            
            # Compare recent volume to longer-term average
            longer_term_avg = data['Volume'].tail(60).mean()
            recent_avg = recent_data['Volume'].mean()
            
            # Determine volume trend
            if recent_avg < longer_term_avg * self.VOLUME_DECLINE_THRESHOLD:
                volume_trend = 'DECREASING'
            elif recent_avg > longer_term_avg * 1.2:
                volume_trend = 'INCREASING'
            else:
                volume_trend = 'STABLE'
            
            # Analyze volume on dips vs rallies
            support_level = range_analysis.get('support', recent_data['Low'].min())
            resistance_level = range_analysis.get('resistance', recent_data['High'].max())
            
            # Volume on moves toward support (potential buying)
            dip_days = recent_data[recent_data['Low'] <= support_level * 1.02]
            rally_days = recent_data[recent_data['High'] >= resistance_level * 0.98]
            
            dip_volume = dip_days['Volume'].mean() if len(dip_days) > 0 else recent_avg
            rally_volume = rally_days['Volume'].mean() if len(rally_days) > 0 else recent_avg
            
            return {
                'trend': volume_trend,
                'recent_vs_longterm': recent_avg / longer_term_avg,
                'dip_volume': dip_volume,
                'rally_volume': rally_volume,
                'volume_on_dips_vs_rallies': dip_volume / rally_volume if rally_volume > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.debug(f"Error analyzing reaccumulation volume: {e}")
            return {'trend': 'STABLE', 'recent_vs_longterm': 1.0}
    
    def _detect_absorption_signs(self, data: pd.DataFrame, range_analysis: Dict) -> List[str]:
        """Detect signs of absorption (institutional accumulation)"""
        absorption_signs = []
        
        try:
            recent_data = data.tail(20)
            support_level = range_analysis.get('support', recent_data['Low'].min())
            
            # 1. Higher lows within the range
            lows = recent_data['Low']
            if len(lows) >= 3:
                recent_lows = lows.tail(3)
                if recent_lows.iloc[-1] > recent_lows.iloc[0]:
                    absorption_signs.append("Higher lows within range")
            
            # 2. Volume increase on dips to support
            support_touches = recent_data[recent_data['Low'] <= support_level * 1.015]
            if len(support_touches) >= 2:
                avg_volume = recent_data['Volume'].mean()
                support_volume = support_touches['Volume'].mean()
                if support_volume > avg_volume * self.ABSORPTION_VOLUME_THRESHOLD:
                    absorption_signs.append("High volume on support tests")
            
            # 3. Quick recoveries from support
            for i in range(1, min(len(recent_data), 10)):
                if (recent_data['Low'].iloc[-i] <= support_level * 1.01 and
                    recent_data['Close'].iloc[-i] > recent_data['Low'].iloc[-i] * 1.01):
                    absorption_signs.append("Quick recovery from support")
                    break
            
            # 4. Narrow range days (low volatility)
            narrow_days = 0
            for i in range(len(recent_data)):
                day_range = (recent_data['High'].iloc[i] - recent_data['Low'].iloc[i]) / recent_data['Low'].iloc[i]
                if day_range < 0.02:  # Less than 2% daily range
                    narrow_days += 1
            
            if narrow_days >= 3:
                absorption_signs.append("Multiple narrow range days")
            
        except Exception as e:
            self.logger.debug(f"Error detecting absorption signs: {e}")
        
        return absorption_signs
    
    def _detect_strength_signs(self, data: pd.DataFrame, range_analysis: Dict) -> List[str]:
        """Detect signs of strength building for breakout"""
        strength_signs = []
        
        try:
            recent_data = data.tail(15)
            resistance_level = range_analysis.get('resistance', recent_data['High'].max())
            
            # 1. Tests of resistance with progressively higher lows
            resistance_tests = recent_data[recent_data['High'] >= resistance_level * 0.98]
            if len(resistance_tests) >= 2:
                test_lows = resistance_tests['Low']
                if len(test_lows) >= 2 and test_lows.iloc[-1] > test_lows.iloc[0]:
                    strength_signs.append("Higher lows on resistance tests")
            
            # 2. Increasing volume on upward moves
            up_days = recent_data[recent_data['Close'] > recent_data['Close'].shift(1)]
            down_days = recent_data[recent_data['Close'] < recent_data['Close'].shift(1)]
            
            if len(up_days) >= 3 and len(down_days) >= 3:
                up_volume = up_days['Volume'].mean()
                down_volume = down_days['Volume'].mean()
                if up_volume > down_volume * 1.1:
                    strength_signs.append("Higher volume on up days")
            
            # 3. Close near highs of recent days
            recent_closes_near_highs = 0
            for i in range(min(5, len(recent_data))):
                close = recent_data['Close'].iloc[-(i+1)]
                high = recent_data['High'].iloc[-(i+1)]
                if close >= high * 0.98:
                    recent_closes_near_highs += 1
            
            if recent_closes_near_highs >= 3:
                strength_signs.append("Closes near daily highs")
            
        except Exception as e:
            self.logger.debug(f"Error detecting strength signs: {e}")
        
        return strength_signs
    
    def _calculate_reaccumulation_confidence(self, range_analysis: Dict, volume_analysis: Dict,
                                           absorption_indicators: List[str], 
                                           strength_indicators: List[str]) -> float:
        """Calculate confidence score for reaccumulation phase"""
        try:
            confidence = 0.0
            
            # Range validity (30% weight)
            if range_analysis.get('is_valid_range', False):
                confidence += 0.3
                
                # Bonus for multiple tests
                support_tests = range_analysis.get('support_tests', 0)
                resistance_tests = range_analysis.get('resistance_tests', 0)
                if support_tests >= 3 and resistance_tests >= 3:
                    confidence += 0.1
            
            # Volume characteristics (25% weight)
            volume_ratio = volume_analysis.get('recent_vs_longterm', 1.0)
            if 0.7 <= volume_ratio <= 0.9:  # Ideal declining volume
                confidence += 0.25
            elif 0.9 < volume_ratio <= 1.1:  # Stable volume
                confidence += 0.15
            
            # Absorption indicators (25% weight)
            absorption_score = min(len(absorption_indicators) / 4, 1.0) * 0.25
            confidence += absorption_score
            
            # Strength indicators (20% weight)
            strength_score = min(len(strength_indicators) / 3, 1.0) * 0.20
            confidence += strength_score
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Error calculating confidence: {e}")
            return 0.0
    
    def _create_no_reaccumulation_analysis(self, symbol: str) -> ReaccumulationAnalysis:
        """Create analysis indicating no reaccumulation detected"""
        return ReaccumulationAnalysis(
            symbol=symbol,
            is_in_reaccumulation=False,
            reaccumulation_duration_days=0,
            range_support=0.0,
            range_resistance=0.0,
            current_price=0.0,
            volume_trend='UNKNOWN',
            absorption_indicators=[],
            strength_indicators=[],
            next_breakout_target=0.0,
            confidence_score=0.0
        )

class ReaccumulationPositionManager:
    """Manages position additions during reaccumulation phases"""
    
    def __init__(self, detector: WyckoffReaccumulationDetector, logger=None):
        self.detector = detector
        self.logger = logger or logging.getLogger(__name__)
        
        # Position addition limits
        self.MAX_TOTAL_ADDITIONS = 2  # Maximum 2 additions per position
        self.MIN_DAYS_BETWEEN_ADDITIONS = 3  # Wait 3 days between additions
        self.MAX_POSITION_INCREASE = 1.0  # Maximum 100% increase from original
    
    def evaluate_position_addition_opportunity(self, symbol: str, position_data: Dict, 
                                             addition_history: List[Dict]) -> Optional[ReaccumulationSignal]:
        """Evaluate if we should add to an existing position"""
        try:
            # Step 1: Analyze for reaccumulation
            analysis = self.detector.analyze_position_for_reaccumulation(symbol, position_data)
            
            if not analysis.is_in_reaccumulation:
                return None
            
            # Step 2: Check addition limits
            if len(addition_history) >= self.MAX_TOTAL_ADDITIONS:
                self.logger.debug(f"Max additions reached for {symbol}")
                return None
            
            # Step 3: Check timing since last addition
            if addition_history:
                last_addition = max(addition_history, key=lambda x: x['date'])
                days_since_last = (datetime.now() - last_addition['date']).days
                if days_since_last < self.MIN_DAYS_BETWEEN_ADDITIONS:
                    self.logger.debug(f"Too soon since last addition for {symbol}")
                    return None
            
            # Step 4: Check for pullback opportunity
            pullback_opportunity = self._is_pullback_opportunity(analysis)
            
            if not pullback_opportunity:
                return None
            
            # Step 5: Calculate recommended addition size
            addition_pct = self._calculate_addition_percentage(analysis, addition_history)
            
            # Step 6: Determine risk level and stop loss
            risk_level, stop_loss = self._assess_addition_risk(analysis)
            
            return ReaccumulationSignal(
                symbol=symbol,
                current_position_size=position_data.get('shares', 0),
                reaccumulation_range={
                    'support': analysis.range_support,
                    'resistance': analysis.range_resistance
                },
                pullback_opportunity=pullback_opportunity,
                volume_confirmation=len(analysis.absorption_indicators) >= 2,
                absorption_strength=analysis.confidence_score,
                risk_level=risk_level,
                recommended_addition_pct=addition_pct,
                stop_loss_level=stop_loss,
                reasoning=f"Reaccumulation addition: {len(analysis.absorption_indicators)} absorption signs, "
                         f"{len(analysis.strength_indicators)} strength signs, confidence: {analysis.confidence_score:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating addition opportunity for {symbol}: {e}")
            return None
    
    def _is_pullback_opportunity(self, analysis: ReaccumulationAnalysis) -> bool:
        """Check if current price represents a good pullback entry"""
        try:
            range_size = analysis.range_resistance - analysis.range_support
            pullback_from_high = (analysis.range_resistance - analysis.current_price) / range_size
            
            # Good entry is in lower 25-75% of the range
            return 0.25 <= pullback_from_high <= 0.75 and analysis.current_price > analysis.range_support * 1.01
            
        except Exception:
            return False
    
    def _calculate_addition_percentage(self, analysis: ReaccumulationAnalysis, 
                                     addition_history: List[Dict]) -> float:
        """Calculate what percentage of original position to add"""
        try:
            # Base addition size based on confidence
            if analysis.confidence_score >= 0.8:
                base_addition = 0.4  # 40% of original position
            elif analysis.confidence_score >= 0.7:
                base_addition = 0.3  # 30% of original position
            else:
                base_addition = 0.2  # 20% of original position
            
            # Reduce for subsequent additions
            reduction_factor = 1.0 - (len(addition_history) * 0.2)
            
            final_addition = base_addition * reduction_factor
            
            return max(0.1, min(final_addition, 0.5))  # Between 10% and 50%
            
        except Exception:
            return 0.2  # Conservative default
    
    def _assess_addition_risk(self, analysis: ReaccumulationAnalysis) -> Tuple[str, float]:
        """Assess risk level and calculate stop loss for addition"""
        try:
            # Risk based on confidence and absorption strength
            if analysis.confidence_score >= 0.8 and len(analysis.absorption_indicators) >= 3:
                risk_level = 'LOW'
                stop_buffer = 0.015  # 1.5% below support
            elif analysis.confidence_score >= 0.7 and len(analysis.absorption_indicators) >= 2:
                risk_level = 'MEDIUM'
                stop_buffer = 0.02   # 2% below support
            else:
                risk_level = 'HIGH'
                stop_buffer = 0.03   # 3% below support
            
            stop_loss = analysis.range_support * (1 - stop_buffer)
            
            return risk_level, stop_loss
            
        except Exception:
            return 'HIGH', analysis.range_support * 0.97

# Integration function for the main fractional system
def integrate_reaccumulation_system(fractional_system):
    """Integrate reaccumulation logic into the existing fractional position system"""
    
    def enhanced_trading_cycle_with_reaccumulation(self) -> Tuple[int, int, int, int, int, int]:
        """Enhanced trading cycle that includes reaccumulation position additions"""
        
        # Call original method to get base results
        trades_executed, wyckoff_sells, profit_scales, emergency_exits, day_trades_blocked = self.run_enhanced_trading_cycle()
        
        reaccumulation_additions = 0
        
        try:
            # NEW: Step for reaccumulation analysis
            if not self.emergency_mode and len(self.get_current_positions()) > 0:
                self.logger.info("🔄 Analyzing existing positions for reaccumulation opportunities...")
                
                detector = WyckoffReaccumulationDetector(self.logger)
                position_manager = ReaccumulationPositionManager(detector, self.logger)
                
                current_positions = self.get_current_positions()
                
                for position_key, position in current_positions.items():
                    # Get addition history for this position (would need to be tracked in database)
                    addition_history = []  # TODO: Implement addition tracking in database
                    
                    # Evaluate for reaccumulation addition
                    addition_signal = position_manager.evaluate_position_addition_opportunity(
                        position['symbol'], position, addition_history
                    )
                    
                    if addition_signal:
                        self.logger.info(f"🎯 Reaccumulation opportunity: {addition_signal.symbol}")
                        self.logger.info(f"   Addition size: {addition_signal.recommended_addition_pct:.1%} of original position")
                        self.logger.info(f"   Risk level: {addition_signal.risk_level}")
                        self.logger.info(f"   Reasoning: {addition_signal.reasoning}")
                        
                        # Execute position addition (would need to implement this)
                        if self._execute_reaccumulation_addition(addition_signal, position):
                            reaccumulation_additions += 1
            
        except Exception as e:
            self.logger.error(f"Error in reaccumulation analysis: {e}")
        
        return trades_executed, wyckoff_sells, profit_scales, emergency_exits, day_trades_blocked, reaccumulation_additions
    
    # Replace the original method
    fractional_system.run_enhanced_trading_cycle = enhanced_trading_cycle_with_reaccumulation.__get__(fractional_system, type(fractional_system))
    
    return fractional_system

# Example integration into main trading system
if __name__ == "__main__":
    print("Wyckoff Reaccumulation Position Addition System")
    print("Integrate this module into your fractional_position_system.py")
    print("Use integrate_reaccumulation_system(fractional_system) to enable reaccumulation additions")