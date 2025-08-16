#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPLETE AUTO-INTEGRATION SCRIPT 📈
Strategic Improvement 5: Signal Quality Enhancement
Run this script to automatically integrate multi-timeframe Wyckoff analysis

Usage: python integrate_signal_quality_enhancement.py
"""

import os
import sys
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# UTF-8 encoding for Windows compatibility
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

class SignalQualityAutoIntegrator:
    """Fully automated integration of signal quality enhancement"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.success_count = 0
        self.total_operations = 0
        self.log_entries = []
        
        # Color codes for Windows terminal
        self.colors = {
            'GREEN': '\033[92m',
            'RED': '\033[91m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'MAGENTA': '\033[95m',
            'CYAN': '\033[96m',
            'RESET': '\033[0m'
        }
    
    def log(self, level: str, message: str, details: str = ""):
        """Enhanced logging with colors and emojis"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Choose emoji and color based on level
        emoji_map = {
            'SUCCESS': '✅',
            'ERROR': '❌', 
            'WARNING': '⚠️',
            'INFO': '📝',
            'STEP': '🔧',
            'RESULT': '🎯'
        }
        
        color_map = {
            'SUCCESS': self.colors['GREEN'],
            'ERROR': self.colors['RED'],
            'WARNING': self.colors['YELLOW'],
            'INFO': self.colors['BLUE'],
            'STEP': self.colors['CYAN'],
            'RESULT': self.colors['MAGENTA']
        }
        
        emoji = emoji_map.get(level, '📋')
        color = color_map.get(level, self.colors['RESET'])
        
        # Format message
        full_message = f"[{timestamp}] {emoji} {message}"
        if details:
            full_message += f": {details}"
        
        # Print with color
        print(f"{color}{full_message}{self.colors['RESET']}")
        
        # Store for summary
        self.log_entries.append(full_message)
        
        # Track success rate
        if level == 'SUCCESS':
            self.success_count += 1
    
    def create_backup(self, file_path: str) -> bool:
        """Create timestamped backup of file"""
        try:
            original = Path(file_path)
            if not original.exists():
                self.log('WARNING', f'File not found for backup', file_path)
                return False
            
            backup_path = original.with_suffix(f'.backup_{self.timestamp}')
            shutil.copy2(original, backup_path)
            
            self.log('SUCCESS', f'Backup created', f'{file_path} → {backup_path.name}')
            return True
            
        except Exception as e:
            self.log('ERROR', f'Backup failed for {file_path}', str(e))
            return False
    
    def create_analyzer_files(self) -> bool:
        """Create the multi-timeframe analyzer files"""
        try:
            self.log('STEP', 'Creating multi-timeframe analyzer files...')
            
            # Create directory structure
            analyzer_dir = Path('strategies/wyckoff')
            analyzer_dir.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = analyzer_dir / '__init__.py'
            if not init_file.exists():
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write('# Wyckoff strategies module\n')
            
            # Create the main multi-timeframe analyzer file
            analyzer_file = analyzer_dir / 'multi_timeframe_wyckoff.py'
            
            # Full implementation content
            analyzer_content = '''#!/usr/bin/env python3
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
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
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
        """Fetch data for all required timeframes"""
        timeframe_data = {}
        
        try:
            ticker = yf.Ticker(symbol)
            
            for tf_name, config in self.timeframes.items():
                try:
                    data = ticker.history(period=config['period'], interval=config['interval'])
                    
                    if len(data) >= config['min_bars'] and not data.empty:
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in data.columns for col in required_cols):
                            timeframe_data[tf_name] = data
                except Exception:
                    continue
            
            return timeframe_data
            
        except Exception:
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
        """Enhanced market scanning"""
        try:
            self.logger.info(f"🔍 Enhanced multi-timeframe scan of {len(symbols)} symbols...")
            
            enhanced_signals = []
            
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
                            enhanced_signals.append(result)
                    except Exception:
                        pass
            
            enhanced_signals.sort(key=lambda x: x.enhanced_strength, reverse=True)
            
            self.logger.info(f"🎯 Found {len(enhanced_signals)} enhanced signals")
            for signal in enhanced_signals[:5]:
                self.logger.info(f"   {signal.symbol}: {signal.signal_quality} - {signal.enhanced_strength:.2f}")
            
            return enhanced_signals
        except Exception:
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
'''
            
            # Write the analyzer file
            with open(analyzer_file, 'w', encoding='utf-8') as f:
                f.write(analyzer_content)
            
            self.log('SUCCESS', 'Multi-timeframe analyzer created', str(analyzer_file))
            
            # Create the interface file
            interface_file = analyzer_dir / 'multi_timeframe_analyzer.py'
            interface_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Timeframe Analyzer Interface - Strategic Improvement 5 📈
"""

from .multi_timeframe_wyckoff import (
    MultiTimeframeSignal,
    TimeframeAnalysis,
    EnhancedMultiTimeframeWyckoffAnalyzer,
    create_enhanced_signal_from_legacy,
    filter_signals_by_quality
)

__all__ = [
    'MultiTimeframeSignal',
    'TimeframeAnalysis', 
    'EnhancedMultiTimeframeWyckoffAnalyzer',
    'create_enhanced_signal_from_legacy',
    'filter_signals_by_quality'
]
'''
            
            with open(interface_file, 'w', encoding='utf-8') as f:
                f.write(interface_content)
            
            self.log('SUCCESS', 'Analyzer interface created', str(interface_file))
            return True
            
        except Exception as e:
            self.log('ERROR', 'Failed to create analyzer files', str(e))
            return False
    
    def update_wyckoff_strategy(self) -> bool:
        """Update the main Wyckoff strategy file"""
        try:
            self.log('STEP', 'Updating Wyckoff strategy file...')
            
            file_path = 'strategies/wyckoff/wyckoff.py'
            
            # Create backup
            if not self.create_backup(file_path):
                return False
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add import for enhanced analyzer
            import_addition = '''
# ENHANCEMENT: Multi-timeframe analysis import - Strategic Improvement 5 📈
try:
    from .multi_timeframe_analyzer import (
        EnhancedMultiTimeframeWyckoffAnalyzer,
        MultiTimeframeSignal,
        filter_signals_by_quality
    )
    MULTI_TIMEFRAME_AVAILABLE = True
    print("✅ Multi-timeframe signal quality enhancement available")
except ImportError as e:
    MULTI_TIMEFRAME_AVAILABLE = False
    print(f"⚠️ Multi-timeframe enhancement not available: {e}")
'''
            
            # Insert after imports
            if 'import traceback' in content:
                content = content.replace(
                    'import traceback',
                    'import traceback' + import_addition
                )
            
            # Add enhanced scanning to WyckoffPnFStrategy class
            if 'class WyckoffPnFStrategy:' in content:
                # Find the __init__ method and enhance it
                init_pattern = r'(def __init__\(self\):\s*\n)(.*?)(def get_sp500_symbols)'
                
                enhanced_init = '''def __init__(self):
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.sector_analyzer = SectorRotationAnalyzer()
        self.db_manager = DatabaseManager()
        self.symbols = self.get_sp500_symbols()
        
        # ENHANCEMENT: Initialize multi-timeframe analyzer - Strategic Improvement 5 📈
        if MULTI_TIMEFRAME_AVAILABLE:
            self.enhanced_analyzer = EnhancedMultiTimeframeWyckoffAnalyzer()
            self.use_enhanced_analysis = True
            print("🎯 Enhanced multi-timeframe Wyckoff analysis enabled")
        else:
            self.enhanced_analyzer = None
            self.use_enhanced_analysis = False
            print("📊 Using standard single-timeframe analysis")

    '''
                
                content = re.sub(init_pattern, enhanced_init + r'\\3', content, flags=re.DOTALL)
                
                # Add enhanced scan method
                enhanced_method = '''
    def scan_market_enhanced(self, max_workers: int = 10) -> List[WyckoffSignal]:
        """Enhanced market scanning with multi-timeframe analysis"""
        try:
            if self.use_enhanced_analysis and self.enhanced_analyzer:
                print("🔍 Using enhanced multi-timeframe scanning")
                
                enhanced_signals = self.enhanced_analyzer.scan_market_enhanced(
                    self.symbols, max_workers=max_workers
                )
                
                # Convert to legacy format
                legacy_signals = []
                for enhanced_signal in enhanced_signals:
                    if enhanced_signal.signal_quality in ['GOOD', 'EXCELLENT']:
                        legacy_signal = WyckoffSignal(
                            symbol=enhanced_signal.symbol,
                            phase=enhanced_signal.primary_phase,
                            strength=enhanced_signal.enhanced_strength,
                            price=enhanced_signal.price,
                            volume_confirmation=enhanced_signal.volume_confirmation,
                            sector=enhanced_signal.sector,
                            combined_score=enhanced_signal.confirmation_score
                        )
                        legacy_signals.append(legacy_signal)
                
                print(f"🎯 Enhanced analysis found {len(legacy_signals)} high-quality signals")
                return legacy_signals
            else:
                return self.scan_market(max_workers)
                
        except Exception as e:
            print(f"❌ Enhanced scanning failed, using fallback: {e}")
            return self.scan_market(max_workers)
'''
                
                # Insert before main() function
                if 'def main():' in content:
                    content = content.replace('def main():', enhanced_method + '\ndef main():')
                
                # Update run_strategy to use enhanced scanning
                content = content.replace(
                    'signals = self.scan_market()',
                    'signals = self.scan_market_enhanced()'
                )
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.log('SUCCESS', 'Wyckoff strategy updated', file_path)
            return True
            
        except Exception as e:
            self.log('ERROR', 'Failed to update Wyckoff strategy', str(e))
            return False
    
    def update_fractional_system(self) -> bool:
        """Update the fractional position system"""
        try:
            self.log('STEP', 'Updating fractional position system...')
            
            file_path = 'fractional_position_system.py'
            
            # Create backup
            if not self.create_backup(file_path):
                return False
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add import
            import_addition = '''
# ENHANCEMENT: Multi-timeframe signal quality import - Strategic Improvement 5 📈
try:
    from strategies.wyckoff.multi_timeframe_analyzer import (
        EnhancedMultiTimeframeWyckoffAnalyzer,
        filter_signals_by_quality,
        MultiTimeframeSignal
    )
    SIGNAL_QUALITY_ENHANCEMENT = True
except ImportError:
    SIGNAL_QUALITY_ENHANCEMENT = False
'''
            
            # Insert after wyckoff import
            if 'from strategies.wyckoff.wyckoff import' in content:
                content = content.replace(
                    'from strategies.wyckoff.wyckoff import WyckoffPnFStrategy, WyckoffSignal',
                    'from strategies.wyckoff.wyckoff import WyckoffPnFStrategy, WyckoffSignal' + import_addition
                )
            
            # Add to bot initialization
            init_enhancement = '''        
        # ENHANCEMENT: Signal Quality Enhancement - Strategic Improvement 5 📈
        if SIGNAL_QUALITY_ENHANCEMENT:
            self.signal_quality_analyzer = EnhancedMultiTimeframeWyckoffAnalyzer(self.logger)
            self.logger.info("🎯 Signal Quality Enhancement (Multi-timeframe) enabled")
        else:
            self.signal_quality_analyzer = None
            self.logger.info("📊 Using standard signal analysis")
'''
            
            # Insert after emergency_mode initialization
            if 'self.emergency_mode = False' in content:
                content = content.replace(
                    'self.emergency_mode = False',
                    'self.emergency_mode = False' + init_enhancement
                )
            
            # Add signal filtering enhancement
            filtering_enhancement = '''
                        # ENHANCEMENT: Apply signal quality filtering - Strategic Improvement 5 📈
                        if SIGNAL_QUALITY_ENHANCEMENT and self.signal_quality_analyzer:
                            try:
                                enhanced_signals = []
                                for signal in buy_signals:
                                    enhanced_result = self.signal_quality_analyzer.analyze_symbol_multi_timeframe(signal.symbol)
                                    
                                    if enhanced_result and enhanced_result.signal_quality in ['GOOD', 'EXCELLENT']:
                                        signal.strength = enhanced_result.enhanced_strength
                                        signal.combined_score = enhanced_result.confirmation_score
                                        enhanced_signals.append(signal)
                                        
                                        self.logger.info(f"🎯 {signal.symbol}: {enhanced_result.signal_quality} quality "
                                                       f"(Phases: {enhanced_result.primary_phase}/"
                                                       f"{enhanced_result.entry_timing_phase}/"
                                                       f"{enhanced_result.precision_phase})")
                                
                                if enhanced_signals:
                                    self.logger.info(f"📈 Quality Enhancement: {len(enhanced_signals)}/{len(buy_signals)} signals passed")
                                    buy_signals = enhanced_signals
                                else:
                                    self.logger.info(f"⚠️ Quality Enhancement: No signals met criteria")
                                    buy_signals = []
                                    
                            except Exception as e:
                                self.logger.warning(f"⚠️ Signal quality enhancement failed: {e}")
'''
            
            # Insert before position sizing
            if '# FIXED: Conservative approach to signal selection WITH DAY TRADE CHECKING' in content:
                content = content.replace(
                    '# FIXED: Conservative approach to signal selection WITH DAY TRADE CHECKING',
                    filtering_enhancement + '\n                        # FIXED: Conservative approach to signal selection WITH DAY TRADE CHECKING'
                )
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.log('SUCCESS', 'Fractional system updated', file_path)
            return True
            
        except Exception as e:
            self.log('ERROR', 'Failed to update fractional system', str(e))
            return False
    
    def create_integration_summary(self) -> bool:
        """Create integration summary report"""
        try:
            self.log('STEP', 'Creating integration summary...')
            
            summary_file = f'signal_quality_integration_report_{self.timestamp}.txt'
            
            summary_content = f"""
═══════════════════════════════════════════════════════════════════════════════
SIGNAL QUALITY ENHANCEMENT INTEGRATION REPORT 📈
Strategic Improvement 5: Multi-timeframe Signal Quality
═══════════════════════════════════════════════════════════════════════════════

Integration Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Success Rate: {self.success_count}/{self.total_operations} operations successful

FILES MODIFIED:
✅ strategies/wyckoff/wyckoff.py (enhanced with multi-timeframe analysis)
✅ fractional_position_system.py (added signal quality filtering)

FILES CREATED:
✅ strategies/wyckoff/multi_timeframe_wyckoff.py (complete implementation)
✅ strategies/wyckoff/multi_timeframe_analyzer.py (interface)

BACKUP FILES:
🔄 All originals backed up with timestamp: {self.timestamp}

═══════════════════════════════════════════════════════════════════════════════
FEATURES ACTIVATED:
═══════════════════════════════════════════════════════════════════════════════

📈 Multi-Timeframe Analysis:
   - Daily charts: Primary Wyckoff phase identification  
   - 4-hour charts: Entry timing within the phase
   - 1-hour charts: Precise entry points

🎯 Signal Quality Scoring:
   - EXCELLENT: >85% confirmation score (highest quality)
   - GOOD: >70% confirmation score (high quality)
   - FAIR: >55% confirmation score (moderate quality)  
   - POOR: <55% confirmation score (low quality)

🔍 Enhanced Filtering:
   - Only trades GOOD+ quality signals
   - Multi-timeframe phase alignment verification
   - Volume confirmation across all timeframes
   - Timeframe-specific Wyckoff calculations

═══════════════════════════════════════════════════════════════════════════════
NEXT STEPS:
═══════════════════════════════════════════════════════════════════════════════

1. 🧪 TEST THE INTEGRATION:
   Run: python -c "from strategies.wyckoff.multi_timeframe_analyzer import EnhancedMultiTimeframeWyckoffAnalyzer; print('✅ Import successful')"

2. 🚀 RUN YOUR BOT:
   The enhanced system will automatically activate when you run your trading bot.
   Look for these messages in the logs:
   - "🎯 Signal Quality Enhancement (Multi-timeframe) enabled"
   - "📈 Quality Enhancement: X/Y signals passed quality filter"

3. 📊 MONITOR PERFORMANCE:
   Enhanced signals will show:
   - Primary/Entry/Precision phases (Daily/4H/1H timeframes)
   - Quality ratings (EXCELLENT, GOOD, FAIR, POOR)
   - Enhanced strength scores (0.0 to 1.0)
   - Multi-timeframe confirmation scores

═══════════════════════════════════════════════════════════════════════════════
INTEGRATION LOG:
═══════════════════════════════════════════════════════════════════════════════

"""
            
            for log_entry in self.log_entries:
                summary_content += f"{log_entry}\n"
            
            summary_content += f"""
═══════════════════════════════════════════════════════════════════════════════
ROLLBACK INSTRUCTIONS (if needed):
═══════════════════════════════════════════════════════════════════════════════

To rollback all changes:

1. Restore original files:
   copy strategies\\wyckoff\\wyckoff.py.backup_{self.timestamp} strategies\\wyckoff\\wyckoff.py
   copy fractional_position_system.py.backup_{self.timestamp} fractional_position_system.py

2. Remove new files:
   del strategies\\wyckoff\\multi_timeframe_wyckoff.py
   del strategies\\wyckoff\\multi_timeframe_analyzer.py

═══════════════════════════════════════════════════════════════════════════════
"""
            
            # Write summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            self.log('SUCCESS', 'Integration report created', summary_file)
            return True
            
        except Exception as e:
            self.log('ERROR', 'Failed to create integration summary', str(e))
            return False
    
    def run_complete_integration(self) -> bool:
        """Run the complete integration process"""
        try:
            print(f"{self.colors['CYAN']}{'='*80}{self.colors['RESET']}")
            print(f"{self.colors['MAGENTA']}🚀 SIGNAL QUALITY ENHANCEMENT AUTO-INTEGRATION{self.colors['RESET']}")
            print(f"{self.colors['CYAN']}Strategic Improvement 5: Multi-timeframe Signal Quality 📈{self.colors['RESET']}")
            print(f"{self.colors['CYAN']}{'='*80}{self.colors['RESET']}")
            
            # Define all operations
            operations = [
                ('Creating multi-timeframe analyzer files', self.create_analyzer_files),
                ('Updating Wyckoff strategy', self.update_wyckoff_strategy),
                ('Updating fractional system', self.update_fractional_system),
                ('Creating integration summary', self.create_integration_summary)
            ]
            
            self.total_operations = len(operations)
            
            # Execute operations
            for desc, operation in operations:
                self.log('STEP', f'Starting: {desc}')
                success = operation()
                if not success:
                    self.log('ERROR', f'Failed: {desc}')
                    break
            
            # Final results
            success_rate = (self.success_count / self.total_operations) * 100
            
            print(f"\n{self.colors['CYAN']}{'='*80}{self.colors['RESET']}")
            
            if self.success_count == self.total_operations:
                self.log('RESULT', f'🎉 INTEGRATION COMPLETED SUCCESSFULLY!')
                self.log('RESULT', f'Success Rate: {success_rate:.0f}% ({self.success_count}/{self.total_operations})')
                print(f"\n{self.colors['GREEN']}✅ Multi-timeframe signal quality enhancement is now ACTIVE{self.colors['RESET']}")
                print(f"{self.colors['GREEN']}🎯 Your bot will now use enhanced signal analysis automatically{self.colors['RESET']}")
                return True
            else:
                self.log('RESULT', f'⚠️ INTEGRATION PARTIALLY COMPLETED')
                self.log('RESULT', f'Success Rate: {success_rate:.0f}% ({self.success_count}/{self.total_operations})')
                print(f"\n{self.colors['YELLOW']}⚠️ Some operations failed - check the integration report{self.colors['RESET']}")
                return False
            
        except Exception as e:
            self.log('ERROR', 'Critical integration failure', str(e))
            return False


def main():
    """Main integration entry point"""
    try:
        print("🚀 Starting Signal Quality Enhancement Auto-Integration...")
        
        # Create and run integrator
        integrator = SignalQualityAutoIntegrator()
        success = integrator.run_complete_integration()
        
        # Final message
        if success:
            print(f"\n{integrator.colors['GREEN']}🎯 READY TO TRADE WITH ENHANCED SIGNAL QUALITY!{integrator.colors['RESET']}")
            print(f"{integrator.colors['GREEN']}Run your trading bot normally - enhancements will activate automatically{integrator.colors['RESET']}")
        else:
            print(f"\n{integrator.colors['YELLOW']}📋 Check integration report for details and next steps{integrator.colors['RESET']}")
        
        return success
        
    except KeyboardInterrupt:
        print(f"\n{integrator.colors['YELLOW']}🛑 Integration cancelled by user{integrator.colors['RESET']}")
        return False
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        return False


if __name__ == "__main__":
    main()