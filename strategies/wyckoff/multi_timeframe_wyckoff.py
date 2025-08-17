#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Wyckoff Analyzer - Strategic Improvement 5 📈
Enhanced signal quality through multi-timeframe confirmation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class MultiTimeframeSignal:
    """Enhanced signal with multi-timeframe analysis"""
    symbol: str
    primary_phase: str          # From daily timeframe
    entry_timing_phase: str     # From 4-hour timeframe
    precision_phase: str        # From 1-hour timeframe
    
    # Individual timeframe strengths
    daily_strength: float
    four_hour_strength: float
    one_hour_strength: float
    
    # Confirmation metrics
    timeframe_alignment: float  # 0.0 to 1.0 - how well timeframes align
    confirmation_score: float   # Overall multi-timeframe confirmation
    
    # Traditional fields
    price: float
    volume_confirmation: bool
    sector: str
    
    # Enhanced scoring
    base_strength: float        # Original single-timeframe strength
    enhanced_strength: float    # Multi-timeframe enhanced strength
    signal_quality: str         # 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe"""
    timeframe: str             # 'd1', '4h', '1h'
    primary_phase: str         # Best Wyckoff phase
    phase_strength: float      # Strength of the phase
    all_phases: Dict[str, float]  # All phase scores
    volume_confirmation: bool
    trend_direction: str       # 'BULLISH', 'BEARISH', 'NEUTRAL'
    support_resistance: Dict   # Key levels
    data_quality: float        # 0.0 to 1.0 - data sufficiency


class EnhancedMultiTimeframeWyckoffAnalyzer:
    """Enhanced Wyckoff analyzer with multi-timeframe confirmation"""
    
    def __init__(self, logger=None, db_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # NEW: Accept database manager to avoid duplicate downloads
        self.db_manager = db_manager

        # Timeframe configurations
        self.timeframes = {
            'daily': {'period': '1y', 'interval': '1d', 'min_bars': 50, 'weight': 0.5},
            '4hour': {'period': '3mo', 'interval': '4h', 'min_bars': 30, 'weight': 0.3},
            '1hour': {'period': '1mo', 'interval': '1h', 'min_bars': 20, 'weight': 0.2}
        }
        
        # Wyckoff phases
        self.accumulation_phases = ['PS', 'SC', 'AR', 'ST', 'Creek', 'SOS', 'LPS', 'BU']
        self.distribution_phases = ['PSY', 'BC', 'AD', 'UTAD', 'LPSY', 'SOW']
        
        # Signal quality thresholds
        self.quality_thresholds = {
            'EXCELLENT': 0.85,
            'GOOD': 0.70,
            'FAIR': 0.55,
            'POOR': 0.0
        }
        
        self.logger.info("🎯 Enhanced Multi-Timeframe Wyckoff Analyzer initialized")
    
    def analyze_symbol_multi_timeframe(self, symbol: str) -> Optional[MultiTimeframeSignal]:
        """Perform comprehensive multi-timeframe Wyckoff analysis"""
        try:
            # Fetch data for all timeframes
            timeframe_data = self._fetch_multi_timeframe_data(symbol)
            if not timeframe_data:
                return None
            
            # Analyze each timeframe
            timeframe_analyses = {}
            for tf_name, data in timeframe_data.items():
                analysis = self._analyze_single_timeframe(symbol, tf_name, data)
                if analysis:
                    timeframe_analyses[tf_name] = analysis
            
            if len(timeframe_analyses) < 2:
                return None
            
            # Create enhanced signal
            enhanced_signal = self._create_enhanced_signal(symbol, timeframe_analyses)
            
            if enhanced_signal and enhanced_signal.confirmation_score >= 0.5:
                return enhanced_signal
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Multi-timeframe analysis error for {symbol}: {e}")
            return None
    
    def _fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for all required timeframes with database optimization"""
        timeframe_data = {}
        
        try:
            # OPTIMIZATION: Use stored daily data first
            if self.db_manager is not None:
                try:
                    stored_daily = self.db_manager.get_data(symbol, period_days=365)
                    if stored_daily is not None and len(stored_daily) >= self.timeframes['daily']['min_bars']:
                        # Handle database column naming (lowercase vs uppercase)
                        if 'open' in stored_daily.columns:
                            # Convert to yfinance format
                            column_mapping = {
                                'open': 'Open', 'high': 'High', 'low': 'Low',
                                'close': 'Close', 'volume': 'Volume'
                            }
                            stored_daily = stored_daily.rename(columns=column_mapping)
                        
                        # Verify required columns exist
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in stored_daily.columns for col in required_cols):
                            timeframe_data['daily'] = stored_daily
                            self.logger.debug(f"✅ Using stored daily data for {symbol}")
                except Exception as e:
                    self.logger.debug(f"⚠️ Error accessing stored data for {symbol}: {e}")
            
            # Fetch missing timeframes from yfinance
            ticker = yf.Ticker(symbol)
            
            for tf_name, config in self.timeframes.items():
                # Skip daily if we already got it from database
                if tf_name == 'daily' and tf_name in timeframe_data:
                    continue
                
                try:
                    data = ticker.history(period=config['period'], interval=config['interval'])
                    
                    if len(data) >= config['min_bars'] and not data.empty:
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in data.columns for col in required_cols):
                            timeframe_data[tf_name] = data
                            self.logger.debug(f"📥 Downloaded {tf_name} data for {symbol}")
                except Exception:
                    continue
            
            return timeframe_data
            
        except Exception as e:
            self.logger.debug(f"❌ Error in multi-timeframe data fetch for {symbol}: {e}")
            return {}
    
    def _analyze_single_timeframe(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TimeframeAnalysis]:
        """Analyze Wyckoff patterns for a single timeframe"""
        try:
            if len(data) < 10:
                return None
            
            # Calculate Wyckoff phase scores
            phase_scores = self._calculate_wyckoff_phases(data, timeframe)
            
            # Find primary phase
            primary_phase = max(phase_scores, key=phase_scores.get)
            phase_strength = phase_scores[primary_phase]
            
            # Volume confirmation
            volume_confirmation = self._check_volume_confirmation(data, primary_phase)
            
            # Trend direction
            trend_direction = self._determine_trend_direction(data)
            
            # Support/resistance levels
            support_resistance = self._calculate_key_levels(data)
            
            # Data quality
            data_quality = min(1.0, len(data) / self.timeframes[timeframe]['min_bars'])
            
            return TimeframeAnalysis(
                timeframe=timeframe,
                primary_phase=primary_phase,
                phase_strength=phase_strength,
                all_phases=phase_scores,
                volume_confirmation=volume_confirmation,
                trend_direction=trend_direction,
                support_resistance=support_resistance,
                data_quality=data_quality
            )
            
        except Exception:
            return None
    
    def _calculate_wyckoff_phases(self, data: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """Calculate Wyckoff phase scores adapted for timeframe"""
        lookback_adjustments = {
            'daily': 1.0,
            '4hour': 0.6,
            '1hour': 0.3
        }
        
        adjustment = lookback_adjustments.get(timeframe, 1.0)
        
        return {
            'PS': self._calculate_ps_score(data, adjustment),
            'SC': self._calculate_sc_score(data, adjustment),
            'AR': self._calculate_ar_score(data, adjustment),
            'ST': self._calculate_st_score(data, adjustment),
            'Creek': self._calculate_creek_score(data, adjustment),
            'SOS': self._calculate_sos_score(data, adjustment),
            'LPS': self._calculate_lps_score(data, adjustment),
            'BU': self._calculate_bu_score(data, adjustment)
        }
    
    def _calculate_ps_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Preliminary Support calculation"""
        try:
            lookback = max(5, int(20 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 3:
                return 0.0
            
            volume_ma = data['Volume'].rolling(window=max(10, int(20 * adjustment))).mean()
            volume_ma_recent = volume_ma.tail(lookback)
            
            high_vol_decline = (
                (recent_data['Volume'] > volume_ma_recent * 1.3) &
                (recent_data['Close'] < recent_data['Close'].shift(1))
            ).sum() / len(recent_data)
            
            return min(1.0, high_vol_decline)
        except Exception:
            return 0.0
    
    def _calculate_sc_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Selling Climax calculation"""
        try:
            lookback = max(3, int(10 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 2:
                return 0.0
            
            volume_ma = data['Volume'].rolling(window=max(10, int(20 * adjustment))).mean().iloc[-1]
            max_volume = recent_data['Volume'].max()
            price_drop = (recent_data['Close'].min() / recent_data['Close'].max()) - 1
            
            volume_spike = max_volume > volume_ma * 2
            significant_drop = price_drop < -0.03
            
            return 0.8 if (volume_spike and significant_drop) else 0.2
        except Exception:
            return 0.0
    
    def _calculate_ar_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Automatic Reaction calculation"""
        try:
            lookback = max(5, int(15 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 3:
                return 0.0
            
            recent_low = recent_data['Low'].min()
            current_price = recent_data['Close'].iloc[-1]
            recovery = (current_price / recent_low) - 1
            
            return min(1.0, recovery * 10)
        except Exception:
            return 0.0
    
    def _calculate_st_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Secondary Test (Spring) calculation"""
        try:
            lookback = max(10, int(30 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 5:
                return 0.0
            
            earlier_low = recent_data['Low'].head(int(len(recent_data) * 0.7)).min()
            recent_low = recent_data['Low'].tail(5).min()
            
            volume_ma = data['Volume'].rolling(window=max(10, int(20 * adjustment))).mean()
            recent_volume = recent_data['Volume'].tail(5).mean()
            avg_volume = volume_ma.tail(lookback).mean()
            
            price_test = abs(recent_low - earlier_low) / earlier_low < 0.02
            volume_reduction = recent_volume < avg_volume * 0.8
            
            return 0.7 if (price_test and volume_reduction) else 0.1
        except Exception:
            return 0.0
    
    def _calculate_creek_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Creek (consolidation) calculation"""
        try:
            lookback = max(10, int(20 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 5:
                return 0.0
            
            price_range = (recent_data['High'].max() - recent_data['Low'].min()) / recent_data['Close'].mean()
            volume_decline = recent_data['Volume'].tail(5).mean() < recent_data['Volume'].head(10).mean()
            narrow_range = price_range < 0.05
            
            return 0.6 if (narrow_range and volume_decline) else 0.1
        except Exception:
            return 0.0
    
    def _calculate_sos_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Sign of Strength calculation"""
        try:
            lookback = max(5, int(15 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 3:
                return 0.0
            
            resistance_level = data['High'].tail(int(50 * adjustment)).max()
            current_price = recent_data['Close'].iloc[-1]
            breakout = current_price > resistance_level * 1.002
            
            volume_ma = data['Volume'].rolling(window=max(10, int(20 * adjustment))).mean()
            recent_volume = recent_data['Volume'].tail(3).mean()
            avg_volume = volume_ma.tail(lookback).mean()
            volume_confirmation = recent_volume > avg_volume * 1.2
            
            score = 0.0
            if breakout:
                score += 0.4
            if volume_confirmation:
                score += 0.3
            if breakout and volume_confirmation:
                score += 0.3
            
            return min(1.0, score)
        except Exception:
            return 0.0
    
    def _calculate_lps_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Last Point of Support calculation"""
        try:
            lookback = max(10, int(30 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 5:
                return 0.0
            
            support_level = recent_data['Low'].head(int(len(recent_data) * 0.7)).min()
            recent_low = recent_data['Low'].tail(5).min()
            
            volume_ma = data['Volume'].rolling(window=max(10, int(20 * adjustment))).mean()
            recent_volume = recent_data['Volume'].tail(5).mean()
            avg_volume = volume_ma.tail(lookback).mean()
            
            support_hold = recent_low > support_level * 0.98
            low_volume = recent_volume < avg_volume
            
            return 0.6 if (support_hold and low_volume) else 0.1
        except Exception:
            return 0.0
    
    def _calculate_bu_score(self, data: pd.DataFrame, adjustment: float) -> float:
        """Backup to Edge calculation"""
        try:
            lookback = max(10, int(25 * adjustment))
            recent_data = data.tail(lookback)
            
            if len(recent_data) < 5:
                return 0.0
            
            recent_high = recent_data['High'].head(int(len(recent_data) * 0.7)).max()
            current_price = recent_data['Close'].iloc[-1]
            pullback = 1 - (current_price / recent_high)
            healthy_pullback = 0.02 < pullback < 0.08
            
            return min(1.0, pullback * 5) if healthy_pullback else 0.1
        except Exception:
            return 0.0
    
    def _check_volume_confirmation(self, data: pd.DataFrame, phase: str) -> bool:
        """Check volume confirmation for phase"""
        try:
            if len(data) < 10:
                return False
            
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].tail(20).mean()
            
            volume_requirements = {
                'SOS': recent_volume > avg_volume * 1.2,
                'SC': recent_volume > avg_volume * 2.0,
                'PS': recent_volume > avg_volume * 1.3,
                'ST': recent_volume < avg_volume * 0.8,
                'LPS': recent_volume < avg_volume * 0.9,
                'Creek': recent_volume < avg_volume * 0.8,
                'BU': recent_volume < avg_volume * 1.1,
                'AR': recent_volume > avg_volume * 1.1
            }
            
            return volume_requirements.get(phase, recent_volume > avg_volume * 0.8)
        except Exception:
            return False
    
    def _determine_trend_direction(self, data: pd.DataFrame) -> str:
        """Determine trend direction"""
        try:
            if len(data) < 20:
                return 'NEUTRAL'
            
            ma_fast = data['Close'].rolling(window=10).mean()
            ma_slow = data['Close'].rolling(window=20).mean()
            
            current_fast = ma_fast.iloc[-1]
            current_slow = ma_slow.iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            if current_price > current_fast > current_slow:
                return 'BULLISH'
            elif current_price < current_fast < current_slow:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'
    
    def _calculate_key_levels(self, data: pd.DataFrame) -> Dict:
        """Calculate support/resistance levels"""
        try:
            if len(data) < 10:
                return {'support': 0, 'resistance': 0, 'current': 0}
            
            lookback = min(50, len(data))
            recent_data = data.tail(lookback)
            
            return {
                'support': recent_data['Low'].min(),
                'resistance': recent_data['High'].max(),
                'current': data['Close'].iloc[-1]
            }
        except Exception:
            return {'support': 0, 'resistance': 0, 'current': 0}
    
    def _create_enhanced_signal(self, symbol: str, timeframe_analyses: Dict[str, TimeframeAnalysis]) -> Optional[MultiTimeframeSignal]:
        """Create enhanced signal from multi-timeframe analyses"""
        try:
            daily_analysis = timeframe_analyses.get('daily')
            four_hour_analysis = timeframe_analyses.get('4hour')
            one_hour_analysis = timeframe_analyses.get('1hour')
            
            if not daily_analysis:
                return None
            
            # Calculate alignment and confirmation
            alignment_score = self._calculate_timeframe_alignment(timeframe_analyses)
            confirmation_score = self._calculate_confirmation_score(timeframe_analyses, alignment_score)
            signal_quality = self._determine_signal_quality(confirmation_score)
            
            # Get current price and volume
            price_source = one_hour_analysis or four_hour_analysis or daily_analysis
            current_price = price_source.support_resistance.get('current', 0)
            
            volume_confirmation = (
                (one_hour_analysis and one_hour_analysis.volume_confirmation) or
                (four_hour_analysis and four_hour_analysis.volume_confirmation) or
                (daily_analysis and daily_analysis.volume_confirmation)
            )
            
            # Calculate enhanced strength
            base_strength = daily_analysis.phase_strength
            enhanced_strength = self._calculate_enhanced_strength(timeframe_analyses, confirmation_score)
            
            return MultiTimeframeSignal(
                symbol=symbol,
                primary_phase=daily_analysis.primary_phase,
                entry_timing_phase=four_hour_analysis.primary_phase if four_hour_analysis else daily_analysis.primary_phase,
                precision_phase=one_hour_analysis.primary_phase if one_hour_analysis else (four_hour_analysis.primary_phase if four_hour_analysis else daily_analysis.primary_phase),
                
                daily_strength=daily_analysis.phase_strength,
                four_hour_strength=four_hour_analysis.phase_strength if four_hour_analysis else 0.0,
                one_hour_strength=one_hour_analysis.phase_strength if one_hour_analysis else 0.0,
                
                timeframe_alignment=alignment_score,
                confirmation_score=confirmation_score,
                
                price=current_price,
                volume_confirmation=volume_confirmation,
                sector='Unknown',
                
                base_strength=base_strength,
                enhanced_strength=enhanced_strength,
                signal_quality=signal_quality
            )
        except Exception:
            return None
    
    def _calculate_timeframe_alignment(self, analyses: Dict[str, TimeframeAnalysis]) -> float:
        """Calculate timeframe alignment score"""
        try:
            phases = [analysis.primary_phase for analysis in analyses.values()]
            
            accumulation_count = sum(1 for phase in phases if phase in self.accumulation_phases)
            distribution_count = sum(1 for phase in phases if phase in self.distribution_phases)
            
            total_phases = len(phases)
            if total_phases == 0:
                return 0.0
            
            max_same_type = max(accumulation_count, distribution_count)
            alignment = max_same_type / total_phases
            
            # Bonus for strong phases
            strong_phases = ['SOS', 'LPS', 'BU']
            strong_alignment = sum(1 for phase in phases if phase in strong_phases)
            
            if strong_alignment >= 2:
                alignment += 0.2
            
            return min(1.0, alignment)
        except Exception:
            return 0.0
    
    def _calculate_confirmation_score(self, analyses: Dict[str, TimeframeAnalysis], alignment: float) -> float:
        """Calculate overall confirmation score"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            
            for tf_name, analysis in analyses.items():
                weight = self.timeframes[tf_name]['weight']
                contribution = analysis.phase_strength * analysis.data_quality
                
                weighted_score += contribution * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            base_score = weighted_score / total_weight
            alignment_bonus = alignment * 0.3
            volume_bonus = 0.1 if any(a.volume_confirmation for a in analyses.values()) else 0.0
            timeframe_bonus = 0.1 if len(analyses) >= 3 else 0.05 if len(analyses) >= 2 else 0.0
            
            final_score = base_score + alignment_bonus + volume_bonus + timeframe_bonus
            return min(1.0, final_score)
        except Exception:
            return 0.0
    
    def _calculate_enhanced_strength(self, analyses: Dict[str, TimeframeAnalysis], confirmation: float) -> float:
        """Calculate enhanced strength"""
        try:
            max_individual_strength = max(a.phase_strength for a in analyses.values())
            confirmation_multiplier = 0.5 + (confirmation * 0.5)
            enhanced = max_individual_strength * confirmation_multiplier
            return min(1.0, enhanced)
        except Exception:
            return 0.0
    
    def _determine_signal_quality(self, confirmation_score: float) -> str:
        """Determine signal quality"""
        for quality, threshold in self.quality_thresholds.items():
            if confirmation_score >= threshold:
                return quality
        return 'POOR'
    
    def scan_market_enhanced(self, symbols: List[str], max_workers: int = 5) -> List[MultiTimeframeSignal]:
            """Enhanced market scanning with detailed multi-timeframe logging"""
            try:
                self.logger.info(f"🔍 Enhanced multi-timeframe scan of {len(symbols)} symbols...")
                
                enhanced_signals = []
                quality_summary = {'EXCELLENT': 0, 'GOOD': 0, 'FAIR': 0, 'POOR': 0}
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_symbol = {
                        executor.submit(self.analyze_symbol_multi_timeframe, symbol): symbol 
                        for symbol in symbols
                    }
                    
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            result = future.result()
                            if result:
                                # Update quality summary
                                quality_summary[result.signal_quality] += 1
                                
                                # DETAILED LOGGING FOR EACH SYMBOL - This is what you want to see!
                                quality_icon = {
                                    'EXCELLENT': '🌟',
                                    'GOOD': '✅', 
                                    'FAIR': '⚠️',
                                    'POOR': '❌'
                                }.get(result.signal_quality, '❓')
                                
                                self.logger.info(f"{quality_icon} {result.symbol}: {result.signal_quality} quality "
                                            f"(Phases: {result.primary_phase}/"
                                            f"{result.entry_timing_phase}/"
                                            f"{result.precision_phase})")
                                self.logger.info(f"   📈 Strengths: Daily={result.daily_strength:.2f} | "
                                            f"4H={result.four_hour_strength:.2f} | "
                                            f"1H={result.one_hour_strength:.2f}")
                                self.logger.info(f"   🎪 Alignment: {result.timeframe_alignment:.2f} | "
                                            f"Confirmation: {result.confirmation_score:.2f}")
                                self.logger.info(f"   💪 Enhanced Strength: {result.enhanced_strength:.2f}")
                                
                                # Add to signals list regardless of quality for complete analysis
                                enhanced_signals.append(result)
                                
                        except Exception as e:
                            self.logger.debug(f"Error analyzing {symbol}: {e}")
                            continue
                
                # Sort by enhanced strength
                enhanced_signals.sort(key=lambda x: x.enhanced_strength, reverse=True)
                
                # Enhanced summary logging
                self.logger.info(f"📈 Multi-Timeframe Quality Summary:")
                self.logger.info(f"   🌟 EXCELLENT: {quality_summary['EXCELLENT']}")
                self.logger.info(f"   ✅ GOOD: {quality_summary['GOOD']}")
                self.logger.info(f"   ⚠️ FAIR: {quality_summary['FAIR']}")
                self.logger.info(f"   ❌ POOR: {quality_summary['POOR']}")
                
                # Show top signals by quality
                excellent_signals = [s for s in enhanced_signals if s.signal_quality == 'EXCELLENT']
                good_signals = [s for s in enhanced_signals if s.signal_quality == 'GOOD']
                
                if excellent_signals:
                    self.logger.info(f"🌟 EXCELLENT signals: {[s.symbol for s in excellent_signals[:10]]}")
                if good_signals:
                    self.logger.info(f"✅ GOOD signals: {[s.symbol for s in good_signals[:10]]}")
                
                self.logger.info(f"🎯 Total enhanced signals found: {len(enhanced_signals)}")
                
                return enhanced_signals
                
            except Exception as e:
                self.logger.error(f"Error in enhanced market scanning: {e}")
                return []


# Helper functions
def filter_signals_by_quality(signals: List[MultiTimeframeSignal], min_quality: str = 'GOOD') -> List[MultiTimeframeSignal]:
    """Filter signals by quality"""
    quality_order = ['POOR', 'FAIR', 'GOOD', 'EXCELLENT']
    min_index = quality_order.index(min_quality)
    
    return [
        signal for signal in signals 
        if quality_order.index(signal.signal_quality) >= min_index
    ]


def create_enhanced_signal_from_legacy(symbol: str, legacy_signal, enhanced_analyzer) -> Optional[MultiTimeframeSignal]:
    """Convert legacy signal to enhanced"""
    try:
        enhanced_signal = enhanced_analyzer.analyze_symbol_multi_timeframe(symbol)
        if enhanced_signal:
            enhanced_signal.sector = getattr(legacy_signal, 'sector', 'Unknown')
            return enhanced_signal
        return None
    except Exception:
        return None
