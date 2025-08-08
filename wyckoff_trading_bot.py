#!/usr/bin/env python3
"""
COMPLETE: Enhanced Wyckoff Trading Bot with Phase-Context Stop Loss System
Replaces hard percentage stops with Wyckoff pattern-aware logic
"""

import sys
import logging
import traceback
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import time

# Import existing systems
from main import MainSystem
from strategies.wyckoff.wyckoff import WyckoffPnFStrategy, WyckoffSignal
from config.config import PersonalTradingConfig


class WyckoffPriceLevelAnalyzer:
    """Analyzes price action to identify key Wyckoff levels"""
    
    def __init__(self, wb_client):
        self.wb = wb_client
    
    def get_recent_price_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Get recent price data for analysis"""
        try:
            # Get daily bars for the last 90 days
            df = self.wb.get_bars(symbol, interval='d1', count=days)
            if df.empty:
                return pd.DataFrame()
            
            # Ensure we have the required columns
            if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                return pd.DataFrame()
            
            return df.sort_index()
        except Exception as e:
            print(f"Error getting price data for {symbol}: {e}")
            return pd.DataFrame()
    
    def find_major_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Find major support and resistance levels using pivot points"""
        if df.empty or len(df) < lookback * 2:
            return {'support_levels': [], 'resistance_levels': []}
        
        highs = df['high'].values
        lows = df['low'].values
        
        # Find pivot highs (resistance)
        resistance_levels = []
        for i in range(lookback, len(highs) - lookback):
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                resistance_levels.append({
                    'price': highs[i],
                    'date': df.index[i],
                    'strength': self._calculate_level_strength(df, highs[i], 'resistance')
                })
        
        # Find pivot lows (support)
        support_levels = []
        for i in range(lookback, len(lows) - lookback):
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                support_levels.append({
                    'price': lows[i],
                    'date': df.index[i],
                    'strength': self._calculate_level_strength(df, lows[i], 'support')
                })
        
        # Sort by strength and keep top levels
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'support_levels': support_levels[:5],  # Top 5 support levels
            'resistance_levels': resistance_levels[:5]  # Top 5 resistance levels
        }
    
    def _calculate_level_strength(self, df: pd.DataFrame, level: float, level_type: str) -> float:
        """Calculate the strength of a support/resistance level"""
        touches = 0
        volume_at_level = 0
        tolerance = level * 0.02  # 2% tolerance
        
        for idx, row in df.iterrows():
            if level_type == 'support':
                if abs(row['low'] - level) <= tolerance:
                    touches += 1
                    volume_at_level += row['volume']
            else:  # resistance
                if abs(row['high'] - level) <= tolerance:
                    touches += 1
                    volume_at_level += row['volume']
        
        # Strength based on number of touches and volume
        avg_volume = df['volume'].mean()
        volume_factor = (volume_at_level / len(df)) / avg_volume if avg_volume > 0 else 1
        
        return touches * volume_factor
    
    def find_recent_breakout_level(self, df: pd.DataFrame, current_price: float) -> Optional[float]:
        """Find the breakout level for SOS phase"""
        if df.empty or len(df) < 20:
            return None
        
        # Look for recent resistance that was broken
        recent_data = df.tail(30)  # Last 30 days
        levels = self.find_major_support_resistance(recent_data, lookback=5)
        
        for resistance in levels['resistance_levels']:
            # Check if current price is above this resistance (breakout)
            if current_price > resistance['price'] * 1.01:  # 1% above to confirm breakout
                # Check if breakout happened recently (within last 10 days)
                days_since = (datetime.now() - resistance['date']).days
                if days_since <= 10:
                    return resistance['price']
        
        return None
    
    def find_pullback_low(self, df: pd.DataFrame, entry_date: datetime) -> Optional[float]:
        """Find the pullback low that triggered BU signal"""
        if df.empty:
            return None
        
        # Look at data since entry date
        entry_date_str = entry_date.strftime('%Y-%m-%d')
        try:
            recent_data = df[df.index >= entry_date_str]
            if len(recent_data) < 2:
                recent_data = df.tail(10)  # Fallback to last 10 days
            
            # Find the lowest low in recent pullback
            pullback_low = recent_data['low'].min()
            return pullback_low
        except:
            return df['low'].tail(10).min()  # Fallback
    
    def find_support_being_tested(self, df: pd.DataFrame, current_price: float) -> Optional[float]:
        """Find the support level being tested in LPS phase"""
        if df.empty:
            return None
        
        levels = self.find_major_support_resistance(df)
        
        # Find support level closest to current price (within 5%)
        for support in levels['support_levels']:
            price_diff = abs(current_price - support['price']) / current_price
            if price_diff <= 0.05:  # Within 5% of support
                return support['price']
        
        return None


class EnhancedWyckoffStopManager:
    """Enhanced stop loss manager with Wyckoff phase-context logic"""
    
    def __init__(self, database, price_analyzer, logger):
        self.database = database
        self.price_analyzer = price_analyzer
        self.logger = logger
        
        # Phase-specific configuration
        self.phase_config = {
            'ST': {
                'type': 'support_break',
                'buffer_pct': 0.03,  # 3% below major support
                'description': 'Stop if breaks significantly below previous major low'
            },
            'SOS': {
                'type': 'breakout_failure', 
                'buffer_pct': 0.02,  # 2% below breakout level
                'description': 'Stop if breaks back below breakout level'
            },
            'LPS': {
                'type': 'support_break',
                'buffer_pct': 0.025,  # 2.5% below support being tested
                'description': 'Stop if breaks the specific support level being tested'
            },
            'Creek': {
                'type': 'time_or_wide',
                'time_limit_days': 45,
                'wide_stop_pct': 0.15,  # 15% for consolidation
                'description': 'Time-based exit or very wide stops for consolidation'
            },
            'BU': {
                'type': 'pullback_break',
                'buffer_pct': 0.02,  # 2% below pullback low
                'description': 'Stop below the pullback low that triggered signal'
            }
        }
    
    def create_enhanced_database_schema(self):
        """Add new fields to stop_strategies table for context stops"""
        with sqlite3.connect(self.database.db_path) as conn:
            # Add new columns for context-aware stops
            columns_to_add = [
                ('key_support_level', 'REAL'),
                ('key_resistance_level', 'REAL'), 
                ('breakout_level', 'REAL'),
                ('pullback_low', 'REAL'),
                ('time_entered', 'TIMESTAMP'),
                ('context_data', 'TEXT'),  # JSON for additional data
                ('stop_reason', 'TEXT')     # Human readable stop reason
            ]
            
            for column_name, column_type in columns_to_add:
                try:
                    conn.execute(f'ALTER TABLE stop_strategies ADD COLUMN {column_name} {column_type}')
                    self.logger.info(f"Added column {column_name} to stop_strategies table")
                except sqlite3.OperationalError as e:
                    if 'duplicate column name' not in str(e):
                        self.logger.warning(f"Could not add column {column_name}: {e}")
    
    def create_wyckoff_stop_strategy(self, symbol: str, phase: str, initial_price: float, 
                                   entry_signal_data: Dict = None) -> bool:
        """Create Wyckoff phase-context stop strategy"""
        try:
            self.logger.info(f"🎯 Creating Wyckoff context stop for {symbol} ({phase})")
            
            # Get price data for analysis
            df = self.price_analyzer.get_recent_price_data(symbol)
            if df.empty:
                self.logger.warning(f"⚠️ No price data for {symbol}, using fallback percentage stop")
                return self._create_fallback_stop(symbol, phase, initial_price)
            
            # Create phase-specific stop strategy
            if phase == 'ST':
                return self._create_st_stop(symbol, initial_price, df, entry_signal_data)
            elif phase == 'SOS':
                return self._create_sos_stop(symbol, initial_price, df, entry_signal_data)
            elif phase == 'LPS':
                return self._create_lps_stop(symbol, initial_price, df, entry_signal_data)
            elif phase == 'Creek':
                return self._create_creek_stop(symbol, initial_price, df, entry_signal_data)
            elif phase == 'BU':
                return self._create_bu_stop(symbol, initial_price, df, entry_signal_data)
            else:
                self.logger.warning(f"⚠️ Unknown phase {phase}, using fallback stop")
                return self._create_fallback_stop(symbol, phase, initial_price)
                
        except Exception as e:
            self.logger.error(f"❌ Error creating Wyckoff stop for {symbol}: {e}")
            return self._create_fallback_stop(symbol, phase, initial_price)
    
    def _create_st_stop(self, symbol: str, initial_price: float, df: pd.DataFrame, 
                       signal_data: Dict = None) -> bool:
        """Create ST (Secondary Test) stop - below major support being retested"""
        levels = self.price_analyzer.find_major_support_resistance(df)
        
        # Find the major support level being retested
        major_support = None
        for support in levels['support_levels']:
            # ST should be near a major support level
            price_diff = abs(initial_price - support['price']) / initial_price
            if price_diff <= 0.08:  # Within 8% of support level
                major_support = support['price']
                break
        
        if not major_support:
            # Fallback: use recent significant low
            major_support = df['low'].tail(30).min()
            self.logger.info(f"📍 ST {symbol}: Using recent low ${major_support:.2f} as support reference")
        else:
            self.logger.info(f"📍 ST {symbol}: Found major support at ${major_support:.2f}")
        
        # Stop 3% below major support (not initial price)
        buffer_pct = self.phase_config['ST']['buffer_pct']
        stop_price = major_support * (1 - buffer_pct)
        
        context_data = {
            'phase': 'ST',
            'logic': 'support_break',
            'major_support': major_support,
            'buffer_pct': buffer_pct,
            'explanation': f'Stop if breaks {buffer_pct*100:.1f}% below major support ${major_support:.2f}'
        }
        
        return self._save_context_stop(
            symbol=symbol,
            strategy_type='WYCKOFF_ST',
            initial_price=initial_price,
            stop_price=stop_price,
            key_support_level=major_support,
            context_data=json.dumps(context_data),
            stop_reason=f"ST: Break below ${major_support:.2f} support"
        )
    
    def _create_sos_stop(self, symbol: str, initial_price: float, df: pd.DataFrame,
                        signal_data: Dict = None) -> bool:
        """Create SOS (Sign of Strength) stop - below breakout level"""
        breakout_level = self.price_analyzer.find_recent_breakout_level(df, initial_price)
        
        if not breakout_level:
            # Fallback: find recent resistance
            levels = self.price_analyzer.find_major_support_resistance(df)
            if levels['resistance_levels']:
                breakout_level = levels['resistance_levels'][0]['price']
                self.logger.info(f"📍 SOS {symbol}: Using recent resistance ${breakout_level:.2f} as breakout reference")
            else:
                breakout_level = initial_price * 0.97  # 3% below entry as fallback
                self.logger.warning(f"⚠️ SOS {symbol}: No breakout level found, using ${breakout_level:.2f}")
        else:
            self.logger.info(f"📍 SOS {symbol}: Found breakout level at ${breakout_level:.2f}")
        
        # Stop 2% below breakout level
        buffer_pct = self.phase_config['SOS']['buffer_pct']
        stop_price = breakout_level * (1 - buffer_pct)
        
        context_data = {
            'phase': 'SOS',
            'logic': 'breakout_failure',
            'breakout_level': breakout_level,
            'buffer_pct': buffer_pct,
            'explanation': f'Stop if breaks {buffer_pct*100:.1f}% below breakout ${breakout_level:.2f}'
        }
        
        return self._save_context_stop(
            symbol=symbol,
            strategy_type='WYCKOFF_SOS',
            initial_price=initial_price,
            stop_price=stop_price,
            breakout_level=breakout_level,
            context_data=json.dumps(context_data),
            stop_reason=f"SOS: Break below ${breakout_level:.2f} breakout"
        )
    
    def _create_lps_stop(self, symbol: str, initial_price: float, df: pd.DataFrame,
                        signal_data: Dict = None) -> bool:
        """Create LPS (Last Point of Support) stop - below specific support being tested"""
        support_being_tested = self.price_analyzer.find_support_being_tested(df, initial_price)
        
        if not support_being_tested:
            # Fallback: recent significant support
            levels = self.price_analyzer.find_major_support_resistance(df)
            if levels['support_levels']:
                support_being_tested = levels['support_levels'][0]['price']
                self.logger.info(f"📍 LPS {symbol}: Using major support ${support_being_tested:.2f}")
            else:
                support_being_tested = df['low'].tail(20).min()
                self.logger.warning(f"⚠️ LPS {symbol}: Using recent low ${support_being_tested:.2f}")
        else:
            self.logger.info(f"📍 LPS {symbol}: Found support being tested at ${support_being_tested:.2f}")
        
        # Stop 2.5% below support being tested
        buffer_pct = self.phase_config['LPS']['buffer_pct']
        stop_price = support_being_tested * (1 - buffer_pct)
        
        context_data = {
            'phase': 'LPS',
            'logic': 'support_break',
            'support_level': support_being_tested,
            'buffer_pct': buffer_pct,
            'explanation': f'Stop if breaks {buffer_pct*100:.1f}% below support ${support_being_tested:.2f}'
        }
        
        return self._save_context_stop(
            symbol=symbol,
            strategy_type='WYCKOFF_LPS',
            initial_price=initial_price,
            stop_price=stop_price,
            key_support_level=support_being_tested,
            context_data=json.dumps(context_data),
            stop_reason=f"LPS: Break below ${support_being_tested:.2f} support"
        )
    
    def _create_creek_stop(self, symbol: str, initial_price: float, df: pd.DataFrame,
                          signal_data: Dict = None) -> bool:
        """Create Creek stop - time-based or very wide percentage"""
        # Creek phase gets both time-based and wide percentage stop
        time_limit = self.phase_config['Creek']['time_limit_days']
        wide_stop_pct = self.phase_config['Creek']['wide_stop_pct']
        
        # Wide stop price (15% below entry)
        wide_stop_price = initial_price * (1 - wide_stop_pct)
        
        # Time exit date
        time_exit_date = datetime.now() + timedelta(days=time_limit)
        
        context_data = {
            'phase': 'Creek',
            'logic': 'time_or_wide',
            'wide_stop_pct': wide_stop_pct,
            'time_limit_days': time_limit,
            'time_exit_date': time_exit_date.isoformat(),
            'explanation': f'Exit after {time_limit} days OR if drops {wide_stop_pct*100:.0f}% below ${initial_price:.2f}'
        }
        
        self.logger.info(f"📍 Creek {symbol}: Wide stop ${wide_stop_price:.2f} (-{wide_stop_pct*100:.0f}%) or time exit in {time_limit} days")
        
        return self._save_context_stop(
            symbol=symbol,
            strategy_type='WYCKOFF_CREEK',
            initial_price=initial_price,
            stop_price=wide_stop_price,
            time_entered=datetime.now(),
            context_data=json.dumps(context_data),
            stop_reason=f"Creek: Time exit or break below ${wide_stop_price:.2f}"
        )
    
    def _create_bu_stop(self, symbol: str, initial_price: float, df: pd.DataFrame,
                       signal_data: Dict = None) -> bool:
        """Create BU (Back-Up) stop - below pullback low that triggered signal"""
        entry_date = datetime.now() - timedelta(days=5)  # Estimate entry was recent
        pullback_low = self.price_analyzer.find_pullback_low(df, entry_date)
        
        if not pullback_low:
            pullback_low = df['low'].tail(10).min()
            self.logger.warning(f"⚠️ BU {symbol}: Using recent low ${pullback_low:.2f} as pullback reference")
        else:
            self.logger.info(f"📍 BU {symbol}: Found pullback low at ${pullback_low:.2f}")
        
        # Stop 2% below pullback low
        buffer_pct = self.phase_config['BU']['buffer_pct']
        stop_price = pullback_low * (1 - buffer_pct)
        
        context_data = {
            'phase': 'BU',
            'logic': 'pullback_break',
            'pullback_low': pullback_low,
            'buffer_pct': buffer_pct,
            'explanation': f'Stop if breaks {buffer_pct*100:.1f}% below pullback low ${pullback_low:.2f}'
        }
        
        return self._save_context_stop(
            symbol=symbol,
            strategy_type='WYCKOFF_BU',
            initial_price=initial_price,
            stop_price=stop_price,
            pullback_low=pullback_low,
            context_data=json.dumps(context_data),
            stop_reason=f"BU: Break below ${pullback_low:.2f} pullback low"
        )
    
    def _create_fallback_stop(self, symbol: str, phase: str, initial_price: float) -> bool:
        """Create fallback percentage-based stop when context analysis fails"""
        fallback_pct = 0.08  # 8% fallback
        stop_price = initial_price * (1 - fallback_pct)
        
        context_data = {
            'phase': phase,
            'logic': 'fallback_percentage',
            'fallback_pct': fallback_pct,
            'explanation': f'Fallback {fallback_pct*100:.0f}% stop due to insufficient data'
        }
        
        self.logger.warning(f"⚠️ {symbol}: Using fallback {fallback_pct*100:.0f}% stop at ${stop_price:.2f}")
        
        return self._save_context_stop(
            symbol=symbol,
            strategy_type='WYCKOFF_FALLBACK',
            initial_price=initial_price,
            stop_price=stop_price,
            context_data=json.dumps(context_data),
            stop_reason=f"Fallback: {fallback_pct*100:.0f}% below ${initial_price:.2f}"
        )
    
    def _save_context_stop(self, symbol: str, strategy_type: str, initial_price: float,
                          stop_price: float, key_support_level: float = None,
                          key_resistance_level: float = None, breakout_level: float = None,
                          pullback_low: float = None, time_entered: datetime = None,
                          context_data: str = None, stop_reason: str = None) -> bool:
        """Save context-aware stop strategy to database"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                conn.execute('''
                    INSERT INTO stop_strategies (
                        symbol, strategy_type, initial_price, stop_price, stop_percentage,
                        key_support_level, key_resistance_level, breakout_level, pullback_low,
                        time_entered, context_data, stop_reason, is_active, bot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, strategy_type, initial_price, stop_price, 0.0,  # stop_percentage set to 0 for context stops
                    key_support_level, key_resistance_level, breakout_level, pullback_low,
                    time_entered or datetime.now(), context_data, stop_reason, True, self.database.bot_id
                ))
            
            self.logger.info(f"✅ Context stop created: {stop_reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving context stop for {symbol}: {e}")
            return False
    
    def check_wyckoff_stop_conditions(self, symbol: str, current_price: float, 
                                    current_volume: float = None) -> Tuple[bool, str]:
        """Check if Wyckoff context stop conditions are met"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                # Get active context stops for this symbol
                cursor = conn.execute('''
                    SELECT strategy_type, stop_price, key_support_level, key_resistance_level,
                           breakout_level, pullback_low, time_entered, context_data, stop_reason
                    FROM stop_strategies 
                    WHERE symbol = ? AND is_active = TRUE AND bot_id = ?
                    AND strategy_type LIKE 'WYCKOFF_%'
                ''', (symbol, self.database.bot_id))
                
                stops = cursor.fetchall()
                
                for stop in stops:
                    (strategy_type, stop_price, key_support, key_resistance, 
                     breakout_level, pullback_low, time_entered, context_data, stop_reason) = stop
                    
                    # Parse context data
                    try:
                        context = json.loads(context_data) if context_data else {}
                    except:
                        context = {}
                    
                    # Check phase-specific conditions
                    should_stop, reason = self._evaluate_phase_stop_condition(
                        strategy_type, current_price, stop_price, context,
                        key_support, key_resistance, breakout_level, pullback_low, time_entered
                    )
                    
                    if should_stop:
                        return True, f"{reason} | {stop_reason}"
                
                return False, "No stop conditions met"
                
        except Exception as e:
            self.logger.error(f"❌ Error checking Wyckoff stops for {symbol}: {e}")
            return False, f"Error checking stops: {e}"
    
    def _evaluate_phase_stop_condition(self, strategy_type: str, current_price: float,
                                     stop_price: float, context: Dict, key_support: float,
                                     key_resistance: float, breakout_level: float,
                                     pullback_low: float, time_entered: str) -> Tuple[bool, str]:
        """Evaluate specific phase stop condition"""
        
        if strategy_type == 'WYCKOFF_ST':
            # ST: Stop if breaks below major support with buffer
            if current_price <= stop_price:
                return True, f"ST: Broke below major support ${key_support:.2f}"
            
        elif strategy_type == 'WYCKOFF_SOS':
            # SOS: Stop if breaks back below breakout level
            if current_price <= stop_price:
                return True, f"SOS: Failed breakout, below ${breakout_level:.2f}"
            
        elif strategy_type == 'WYCKOFF_LPS':
            # LPS: Stop if breaks the support being tested
            if current_price <= stop_price:
                return True, f"LPS: Support failed at ${key_support:.2f}"
            
        elif strategy_type == 'WYCKOFF_CREEK':
            # Creek: Check both time and price conditions
            if current_price <= stop_price:
                return True, f"Creek: Wide stop triggered at ${stop_price:.2f}"
            
            # Check time condition
            if time_entered:
                try:
                    entry_time = datetime.fromisoformat(time_entered.replace('Z', '+00:00'))
                    days_held = (datetime.now() - entry_time).days
                    time_limit = context.get('time_limit_days', 45)
                    
                    if days_held >= time_limit:
                        return True, f"Creek: Time limit reached ({days_held} days)"
                except:
                    pass
            
        elif strategy_type == 'WYCKOFF_BU':
            # BU: Stop if breaks below pullback low
            if current_price <= stop_price:
                return True, f"BU: Broke below pullback low ${pullback_low:.2f}"
        
        return False, "No trigger"


class EnhancedTradingDatabase:
    """Enhanced database manager with stop loss tracking"""
    
    def __init__(self, db_path="data/trading_bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.bot_id = "wyckoff_bot_v1"  # Unique identifier for this bot
        self.init_database()
    
    def init_database(self):
        """Initialize database tables with enhanced stop loss support"""
        with sqlite3.connect(self.db_path) as conn:
            # Trading signals table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    strength REAL NOT NULL,
                    price REAL NOT NULL,
                    volume_confirmation BOOLEAN NOT NULL,
                    sector TEXT NOT NULL,
                    combined_score REAL NOT NULL,
                    action_taken TEXT,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced trades table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    total_value REAL NOT NULL,
                    signal_phase TEXT,
                    signal_strength REAL,
                    account_type TEXT,
                    order_id TEXT,
                    status TEXT DEFAULT 'PENDING',
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    trade_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced positions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    total_shares REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    total_invested REAL NOT NULL,
                    first_purchase_date TEXT NOT NULL,
                    last_purchase_date TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced stop strategies table with context fields
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stop_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,  -- 'STOP_LOSS' or 'TRAILING_STOP' or 'WYCKOFF_*'
                    initial_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    stop_percentage REAL NOT NULL,
                    trailing_high REAL,  -- For trailing stops
                    key_support_level REAL,  -- For Wyckoff context stops
                    key_resistance_level REAL,
                    breakout_level REAL,
                    pullback_low REAL,
                    time_entered TIMESTAMP,
                    context_data TEXT,  -- JSON for additional data
                    stop_reason TEXT,   -- Human readable stop reason
                    is_active BOOLEAN DEFAULT TRUE,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Bot runs table for tracking execution
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    signals_found INTEGER NOT NULL,
                    trades_executed INTEGER NOT NULL,
                    stop_losses_executed INTEGER DEFAULT 0,
                    errors_encountered INTEGER NOT NULL,
                    total_portfolio_value REAL,
                    available_cash REAL,
                    status TEXT NOT NULL,
                    log_details TEXT,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_date_symbol ON trades(date, symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stop_strategies_symbol ON stop_strategies(symbol, is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON positions(bot_id)')
    
    def log_signal(self, signal: WyckoffSignal, action_taken: str = None):
        """Log a trading signal"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO signals (date, symbol, phase, strength, price, volume_confirmation, 
                                   sector, combined_score, action_taken, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                signal.symbol,
                signal.phase,
                signal.strength,
                signal.price,
                signal.volume_confirmation,
                signal.sector,
                signal.combined_score,
                action_taken,
                self.bot_id
            ))
    
    def log_trade(self, symbol: str, action: str, quantity: float, price: float, 
                  signal_phase: str, signal_strength: float, account_type: str, 
                  order_id: str = None):
        """Log a trade execution"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trades (date, symbol, action, quantity, price, total_value, 
                                  signal_phase, signal_strength, account_type, order_id, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                symbol,
                action,
                quantity,
                price,
                quantity * price,
                signal_phase,
                signal_strength,
                account_type,
                order_id,
                self.bot_id
            ))
    
    def update_position(self, symbol: str, shares: float, cost: float, account_type: str):
        """Update position tracking"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if position exists
            existing = conn.execute(
                'SELECT total_shares, avg_cost, total_invested, first_purchase_date FROM positions WHERE symbol = ? AND bot_id = ?',
                (symbol, self.bot_id)
            ).fetchone()
            
            if existing:
                # Update existing position
                old_shares, old_avg_cost, old_invested, first_date = existing
                new_shares = old_shares + shares
                
                if new_shares > 0:
                    new_invested = old_invested + (shares * cost)
                    new_avg_cost = new_invested / new_shares
                else:
                    # Position closed
                    new_invested = 0
                    new_avg_cost = 0
                
                conn.execute('''
                    UPDATE positions 
                    SET total_shares = ?, avg_cost = ?, total_invested = ?, 
                        last_purchase_date = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND bot_id = ?
                ''', (new_shares, new_avg_cost, new_invested, today, symbol, self.bot_id))
            else:
                # Create new position
                conn.execute('''
                    INSERT INTO positions (symbol, total_shares, avg_cost, total_invested, 
                                         first_purchase_date, last_purchase_date, account_type, bot_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, shares, cost, shares * cost, today, today, account_type, self.bot_id))
    
    def deactivate_stop_strategies(self, symbol: str):
        """Deactivate all stop strategies for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE stop_strategies 
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND bot_id = ?
            ''', (symbol, self.bot_id))
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol (only this bot's positions)"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                'SELECT * FROM positions WHERE symbol = ? AND bot_id = ?', (symbol, self.bot_id)
            ).fetchone()
            
            if result:
                columns = ['symbol', 'total_shares', 'avg_cost', 'total_invested', 
                          'first_purchase_date', 'last_purchase_date', 'account_type', 'bot_id', 'updated_at']
                return dict(zip(columns, result))
            return None
    
    def get_todays_trades(self, symbol: str = None) -> List[Dict]:
        """Get today's trades for day trade prevention"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                query = '''
                    SELECT * FROM trades 
                    WHERE date = ? AND symbol = ? AND bot_id = ?
                    ORDER BY trade_datetime
                '''
                results = conn.execute(query, (today, symbol, self.bot_id)).fetchall()
            else:
                query = '''
                    SELECT * FROM trades 
                    WHERE date = ? AND bot_id = ?
                    ORDER BY trade_datetime
                '''
                results = conn.execute(query, (today, self.bot_id)).fetchall()
            
            columns = ['id', 'date', 'symbol', 'action', 'quantity', 'price', 'total_value',
                      'signal_phase', 'signal_strength', 'account_type', 'order_id', 'status',
                      'bot_id', 'trade_datetime', 'created_at']
            
            return [dict(zip(columns, row)) for row in results]
    
    def would_create_day_trade(self, symbol: str, action: str) -> bool:
        """Check if this trade would create a day trade"""
        today_trades = self.get_todays_trades(symbol)
        
        if not today_trades:
            return False
        
        # Count buys and sells today
        buys_today = sum(1 for trade in today_trades if trade['action'] == 'BUY')
        sells_today = sum(1 for trade in today_trades if trade['action'] == 'SELL')
        
        # Day trade occurs when we buy and sell the same security on the same day
        if action == 'SELL' and buys_today > 0:
            return True
        elif action == 'BUY' and sells_today > 0:
            return True
        
        return False
    
    def log_bot_run(self, signals_found: int, trades_executed: int, stop_losses_executed: int,
                    errors: int, portfolio_value: float, available_cash: float, 
                    status: str, log_details: str):
        """Log bot run statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO bot_runs (run_date, signals_found, trades_executed, stop_losses_executed,
                                    errors_encountered, total_portfolio_value, available_cash, 
                                    status, log_details, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signals_found,
                trades_executed,
                stop_losses_executed,
                errors,
                portfolio_value,
                available_cash,
                status,
                log_details,
                self.bot_id
            ))


class EnhancedWyckoffTradingBot:
    """Enhanced trading bot with Wyckoff phase-context stops and session validation"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        self.price_analyzer = None
        self.wyckoff_stop_manager = None
        
        # Dynamic position sizing configuration
        self.base_trade_amount = 5.00
        self.max_trade_amount = 50.00
        self.base_min_balance = 50.00
        
        # Scaling tiers for position sizing
        self.position_scaling_tiers = [
            {'min_cash': 0, 'trade_amount': 5.50},
            {'min_cash': 300, 'trade_amount': 10.00},
            {'min_cash': 600, 'trade_amount': 15.00},
            {'min_cash': 1000, 'trade_amount': 25.00},
            {'min_cash': 2000, 'trade_amount': 35.00},
            {'min_cash': 5000, 'trade_amount': 50.00},
        ]
        
        # Scaling tiers for minimum account balance to preserve
        self.balance_scaling_tiers = [
            {'min_cash': 0, 'min_balance': 50.00},
            {'min_cash': 500, 'min_balance': 100.00},
            {'min_cash': 1000, 'min_balance': 150.00},
            {'min_cash': 2000, 'min_balance': 200.00},
            {'min_cash': 5000, 'min_balance': 300.00},
            {'min_cash': 10000, 'min_balance': 500.00},
        ]
        
        # Trading phase filters for buy signals
        self.buy_phases = ['ST', 'Creek', 'SOS', 'LPS', 'BU']  # Accumulation phases
        self.sell_phases = ['PS', 'SC']  # Distribution phases
        self.min_signal_strength = 0.6  # Minimum signal strength
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the bot"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = logs_dir / f"enhanced_trading_bot_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🤖 ENHANCED WYCKOFF TRADING BOT WITH CONTEXT STOPS")
        self.logger.info(f"📝 Log: {log_filename.name}")
        self.logger.info("🎯 Using Wyckoff Phase-Context Stop Loss System")
        self.logger.info("💰 Using Dynamic Position Sizing Based on Available Capital")
        self.logger.info("🛡️ Using Dynamic Balance Preservation That Scales With Growth")
    
    def initialize_systems(self) -> bool:
        """Initialize all required systems including Wyckoff context stops"""
        try:
            self.logger.info("🔧 Initializing trading systems with Wyckoff context stops...")
            
            # Initialize main system (handles auth, accounts)
            self.main_system = MainSystem()
            
            # Initialize Wyckoff strategy
            self.wyckoff_strategy = WyckoffPnFStrategy()
            
            # Initialize enhanced database
            self.database = EnhancedTradingDatabase()
            
            # Initialize price analyzer
            self.price_analyzer = WyckoffPriceLevelAnalyzer(self.main_system.wb)
            
            # Initialize enhanced stop manager
            self.wyckoff_stop_manager = EnhancedWyckoffStopManager(
                self.database, self.price_analyzer, self.logger
            )
            
            # Create enhanced database schema
            self.wyckoff_stop_manager.create_enhanced_database_schema()
            
            # Log dynamic position sizing configuration
            self.logger.info("💰 Dynamic Position Sizing Configuration:")
            for tier in self.position_scaling_tiers:
                if tier == self.position_scaling_tiers[-1]:
                    self.logger.info(f"   ${tier['min_cash']:,}+: ${tier['trade_amount']:.2f} per trade")
                else:
                    next_tier = self.position_scaling_tiers[self.position_scaling_tiers.index(tier) + 1]
                    self.logger.info(f"   ${tier['min_cash']:,}-${next_tier['min_cash']-1:,}: ${tier['trade_amount']:.2f} per trade")
            
            # Log Wyckoff context stop configuration
            self.logger.info("🎯 Wyckoff Phase-Context Stop Configuration:")
            for phase, config in self.wyckoff_stop_manager.phase_config.items():
                if phase == 'Creek':
                    self.logger.info(f"   {phase}: {config['time_limit_days']} days OR {config['wide_stop_pct']*100:.0f}% wide stop")
                else:
                    self.logger.info(f"   {phase}: {config['type']} with {config['buffer_pct']*100:.1f}% buffer")
            
            self.logger.info("✅ All systems initialized with Wyckoff context stops")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize systems: {e}")
            return False
    
    def validate_and_refresh_session(self) -> bool:
        """Validate session and refresh trade token if needed"""
        try:
            wb = self.main_system.wb
            
            # Test if we can make a basic API call
            try:
                account_info = wb.get_account()
                if not account_info:
                    self.logger.warning("⚠️ Basic session test failed")
                    return self.full_reauthentication()
            except Exception as e:
                self.logger.warning(f"⚠️ Session validation failed: {e}")
                return self.full_reauthentication()
            
            # Test if trade token is valid
            try:
                wb.get_current_orders()
                self.logger.debug("✅ Trade token validation passed")
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if 'expired' in error_msg or 'login' in error_msg:
                    self.logger.warning("⚠️ Trade token expired, attempting refresh...")
                    return self.refresh_trade_token()
                else:
                    self.logger.warning(f"⚠️ Trade token test failed: {e}")
                    return self.refresh_trade_token()
                    
        except Exception as e:
            self.logger.error(f"❌ Session validation error: {e}")
            return False
    
    def refresh_trade_token(self) -> bool:
        """Refresh the trade token"""
        try:
            self.logger.info("🔄 Refreshing trade token...")
            
            # Load credentials
            credentials = self.main_system.credential_manager.load_credentials()
            
            # Get new trade token
            if self.main_system.wb.get_trade_token(credentials['trading_pin']):
                self.logger.info("✅ Trade token refreshed successfully")
                
                # Save the updated session
                self.main_system.session_manager.save_session(self.main_system.wb)
                return True
            else:
                self.logger.error("❌ Failed to refresh trade token")
                return self.full_reauthentication()
                
        except Exception as e:
            self.logger.error(f"❌ Error refreshing trade token: {e}")
            return self.full_reauthentication()
    
    def full_reauthentication(self) -> bool:
        """Perform full reauthentication when session can't be refreshed"""
        try:
            self.logger.warning("🔄 Session expired, performing full reauthentication...")
            
            # Clear existing session
            self.main_system.session_manager.clear_session()
            
            # Perform fresh login
            if self.main_system.login_manager.login_automatically():
                self.logger.info("✅ Full reauthentication successful")
                
                # Save new session
                self.main_system.session_manager.save_session(self.main_system.wb)
                return True
            else:
                self.logger.error("❌ Full reauthentication failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Full reauthentication error: {e}")
            return False
    
    def get_dynamic_trade_amount(self, total_available_cash: float) -> float:
        """Calculate trade size based on available cash using scaling tiers"""
        trade_amount = self.base_trade_amount
        
        for tier in reversed(self.position_scaling_tiers):
            if total_available_cash >= tier['min_cash']:
                trade_amount = tier['trade_amount']
                break
        
        return min(trade_amount, self.max_trade_amount)
    
    def get_dynamic_min_balance(self, total_available_cash: float) -> float:
        """Calculate minimum balance to preserve based on available cash using scaling tiers"""
        min_balance = self.base_min_balance
        
        for tier in reversed(self.balance_scaling_tiers):
            if total_available_cash >= tier['min_cash']:
                min_balance = tier['min_balance']
                break
        
        return min_balance
    
    def authenticate_and_setup_accounts(self) -> bool:
        """Authenticate and discover accounts"""
        try:
            # Run main system workflow (auth + account discovery)
            if not self.main_system.run():
                self.logger.error("❌ Main system initialization failed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Authentication failed: {e}")
            return False
    
    def get_enabled_accounts(self):
        """Get all accounts enabled for trading"""
        enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
        
        if not enabled_accounts:
            self.logger.error("❌ No enabled accounts found")
            return []
        
        # Sort accounts: Cash first, then Margin, then others
        def account_priority(account):
            if account.account_type in ['Cash Account', 'CASH']:
                return 1
            elif account.account_type in ['Margin Account', 'MARGIN']:
                return 2
            else:
                return 3
        
        enabled_accounts.sort(key=account_priority)
        return enabled_accounts
    
    def check_webull_day_trades(self, symbol: str, action: str) -> bool:
        """Check Webull's actual trading history for day trade prevention"""
        try:
            wb = self.main_system.wb
            
            # Get today's orders from Webull
            orders = wb.get_history_orders(status='All', count=50)
            
            if not orders or 'data' not in orders:
                return False
            
            today = datetime.now().strftime('%Y-%m-%d')
            today_orders = []
            
            # Filter for today's orders for this symbol
            for order in orders['data']:
                if (order.get('ticker', {}).get('symbol') == symbol and
                    order.get('createTime', '').startswith(today) and
                    order.get('status') in ['Filled', 'Partially Filled']):
                    today_orders.append(order)
            
            if not today_orders:
                return False
            
            # Count buys and sells
            buys_today = sum(1 for order in today_orders if order.get('action') == 'BUY')
            sells_today = sum(1 for order in today_orders if order.get('action') == 'SELL')
            
            # Would this create a day trade?
            if action == 'SELL' and buys_today > 0:
                return True
            elif action == 'BUY' and sells_today > 0:
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check Webull day trades for {symbol}: {e}")
            # If we can't check, be conservative and assume it might be a day trade
            return self.database.would_create_day_trade(symbol, action)
    
    def scan_for_signals(self) -> List[WyckoffSignal]:
        """Scan market for Wyckoff signals"""
        try:
            self.logger.info("🔍 Scanning market for Wyckoff signals...")
            
            # Update database with fresh data
            self.wyckoff_strategy.update_database()
            
            # Scan for signals
            signals = self.wyckoff_strategy.scan_market()
            
            # Get sector ranking for enhanced scoring
            sector_ranking = self.wyckoff_strategy.sector_analyzer.get_sector_ranking()
            sector_strength = dict(sector_ranking)
            
            # Calculate combined scores
            for signal in signals:
                sec_str = sector_strength.get(signal.sector, 0.0)
                signal.combined_score = (
                    signal.strength * 0.6 + 
                    (sec_str / 100) * 0.3 +
                    (0.1 if signal.volume_confirmation else 0)
                )
            
            # Sort by combined score
            signals.sort(key=lambda x: x.combined_score, reverse=True)
            
            self.logger.info(f"📊 Found {len(signals)} potential signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning for signals: {e}")
            return []
    
    def check_stop_losses(self) -> int:
        """Check Wyckoff context stops instead of percentage stops"""
        stop_losses_executed = 0
        
        try:
            # Get all our positions
            with sqlite3.connect(self.database.db_path) as conn:
                positions = conn.execute('''
                    SELECT symbol, total_shares FROM positions 
                    WHERE total_shares > 0 AND bot_id = ?
                ''', (self.database.bot_id,)).fetchall()
            
            if not positions:
                return 0
            
            self.logger.info(f"🎯 Checking Wyckoff context stops for {len(positions)} positions...")
            
            wb = self.main_system.wb
            
            for symbol, shares in positions:
                try:
                    # Get current price
                    quote_data = wb.get_quote(symbol)
                    if not quote_data or 'close' not in quote_data:
                        self.logger.warning(f"⚠️ Could not get quote for {symbol}")
                        continue
                    
                    current_price = float(quote_data['close'])
                    current_volume = float(quote_data.get('volume', 0))
                    
                    # Check Wyckoff context stop conditions
                    should_stop, stop_reason = self.wyckoff_stop_manager.check_wyckoff_stop_conditions(
                        symbol, current_price, current_volume
                    )
                    
                    if should_stop:
                        # Check for day trade
                        if self.check_webull_day_trades(symbol, 'SELL'):
                            self.logger.warning(f"⚠️ Skipping {symbol} context stop - would create day trade")
                            continue
                        
                        self.logger.info(f"🎯 Executing Wyckoff context stop for {symbol}: {stop_reason}")
                        
                        if self.execute_stop_loss_sell(symbol, shares, current_price, stop_reason):
                            stop_losses_executed += 1
                            # Deactivate stop strategies
                            self.database.deactivate_stop_strategies(symbol)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error checking context stop for {symbol}: {e}")
                    continue
            
            return stop_losses_executed
            
        except Exception as e:
            self.logger.error(f"❌ Error in context stop check: {e}")
            return stop_losses_executed
    
    def execute_stop_loss_sell(self, symbol: str, shares: float, current_price: float, 
                            reason: str, account=None) -> bool:
        """Execute a stop loss sell order with session validation"""
        try:
            # Validate session before trading
            if not self.validate_and_refresh_session():
                self.logger.error(f"❌ Session validation failed for {symbol} stop loss")
                return False
            
            # If no account specified, find the account that holds this position
            if account is None:
                position = self.database.get_position(symbol)
                if not position:
                    self.logger.error(f"❌ Could not find position for {symbol}")
                    return False
                
                target_account_type = position['account_type']
                enabled_accounts = self.get_enabled_accounts()
                account = next((acc for acc in enabled_accounts if acc.account_type == target_account_type), None)
                
                if not account:
                    self.logger.error(f"❌ Could not find account type {target_account_type} for {symbol}")
                    return False
            
            # GET POSITION INFO BEFORE UPDATING (for P&L calculation)
            position_before_sale = self.database.get_position(symbol)
            
            # Switch to the trading account
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to {account.account_type} account for {symbol}")
                return False
            
            self.logger.info(f"💸 Context Stop: Selling {shares} shares of {symbol} at ~${current_price:.2f} from {account.account_type}")
            self.logger.info(f"   Reason: {reason}")
            
            # Place market sell order
            wb = self.main_system.wb
            order_result = wb.place_order(
                stock=symbol,
                price=0,  # Market price
                action='SELL',
                orderType='MKT',
                enforce='DAY',
                quant=shares,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                self.logger.info(f"✅ Context stop order placed: {symbol} - Order ID: {order_id}")
                
                # Log the trade
                self.database.log_trade(
                    symbol=symbol,
                    action='SELL',
                    quantity=shares,
                    price=current_price,
                    signal_phase='WYCKOFF_CONTEXT_STOP',
                    signal_strength=1.0,
                    account_type=account.account_type,
                    order_id=order_id
                )
                
                # Calculate P&L BEFORE updating position
                if position_before_sale and position_before_sale['total_invested'] > 0:
                    profit_loss = (current_price - position_before_sale['avg_cost']) * shares
                    profit_loss_pct = (profit_loss / position_before_sale['total_invested']) * 100
                    self.logger.info(f"📊 Context Stop P&L for {symbol}: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
                
                # Update position
                self.database.update_position(
                    symbol=symbol,
                    shares=-shares,  # Negative to reduce position
                    cost=current_price,
                    account_type=account.account_type
                )
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                if 'expired' in error_msg.lower() or 'login' in error_msg.lower():
                    self.logger.warning(f"⚠️ Session expired during {symbol} context stop, retrying...")
                    if self.full_reauthentication():
                        return self.execute_stop_loss_sell(symbol, shares, current_price, reason, account)
                
                self.logger.error(f"❌ Context stop order failed for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing context stop for {symbol}: {e}")
            return False
    
    def filter_buy_signals(self, signals: List[WyckoffSignal]) -> List[WyckoffSignal]:
        """Filter signals for buy opportunities"""
        buy_signals = []
        
        for signal in signals:
            # Check if signal is in accumulation phase
            if signal.phase in self.buy_phases:
                # Check signal strength
                if signal.strength >= self.min_signal_strength:
                    # Check volume confirmation
                    if signal.volume_confirmation:
                        # Check for day trade
                        if self.check_webull_day_trades(signal.symbol, 'BUY'):
                            self.logger.warning(f"⚠️ Skipping {signal.symbol} - would create day trade")
                            continue
                        
                        buy_signals.append(signal)
                        self.logger.info(f"✅ Buy signal: {signal.symbol} ({signal.phase}) - Strength: {signal.strength:.2f}")
        
        return buy_signals
    
    def filter_sell_signals(self, signals: List[WyckoffSignal], held_positions: List[str]) -> List[WyckoffSignal]:
        """Filter signals for sell opportunities on held positions"""
        sell_signals = []
        
        for signal in signals:
            # Only consider signals for stocks we hold
            if signal.symbol in held_positions:
                # Check if signal is in distribution phase
                if signal.phase in self.sell_phases:
                    if signal.strength >= self.min_signal_strength:
                        # Check for day trade
                        if self.check_webull_day_trades(signal.symbol, 'SELL'):
                            self.logger.warning(f"⚠️ Skipping {signal.symbol} sell - would create day trade")
                            continue
                        
                        sell_signals.append(signal)
                        self.logger.info(f"🔴 Sell signal: {signal.symbol} ({signal.phase}) - Strength: {signal.strength:.2f}")
        
        return sell_signals
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions from database across all accounts"""
        positions = {}
        
        with sqlite3.connect(self.database.db_path) as conn:
            results = conn.execute('''
                SELECT symbol, total_shares, avg_cost, total_invested, account_type 
                FROM positions 
                WHERE total_shares > 0 AND bot_id = ?
            ''', (self.database.bot_id,)).fetchall()
            
            for symbol, shares, avg_cost, invested, account_type in results:
                positions[symbol] = {
                    'shares': shares,
                    'avg_cost': avg_cost,
                    'total_invested': invested,
                    'account_type': account_type
                }
        
        return positions
    
    def execute_buy_order(self, signal: WyckoffSignal, account, dynamic_trade_amount: float = None) -> bool:
        """Execute buy order with Wyckoff context stops instead of percentage stops"""
        try:
            # Validate session before trading
            if not self.validate_and_refresh_session():
                self.logger.error(f"❌ Session validation failed for {signal.symbol} buy order")
                return False
            
            # Use provided dynamic trade amount or calculate it
            trade_amount = dynamic_trade_amount or self.get_dynamic_trade_amount(account.settled_funds)
            
            # Switch to the trading account
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            # Get current quote to determine shares to buy
            wb = self.main_system.wb
            quote_data = wb.get_quote(signal.symbol)
            
            if not quote_data or 'close' not in quote_data:
                self.logger.error(f"❌ Could not get quote for {signal.symbol}")
                return False
            
            current_price = float(quote_data['close'])
            shares_to_buy = trade_amount / current_price
            shares_to_buy = float(round(shares_to_buy, 5))
            
            if shares_to_buy < 0.00001:
                self.logger.warning(f"⚠️ Share amount too small for {signal.symbol}: {shares_to_buy}")
                return False
            
            self.logger.info(f"💰 Buying {shares_to_buy} shares of {signal.symbol} at ~${current_price:.2f} (${trade_amount:.2f} position)")
            
            # Place market order
            order_result = wb.place_order(
                stock=signal.symbol,
                price=0,
                action='BUY',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_buy,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                self.logger.info(f"✅ Buy order placed: {signal.symbol} - Order ID: {order_id}")
                
                # Log the trade
                self.database.log_trade(
                    symbol=signal.symbol,
                    action='BUY',
                    quantity=shares_to_buy,
                    price=current_price,
                    signal_phase=signal.phase,
                    signal_strength=signal.strength,
                    account_type=account.account_type,
                    order_id=order_id
                )
                
                # Update position tracking
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=shares_to_buy,
                    cost=current_price,
                    account_type=account.account_type
                )
                
                # Create Wyckoff context stops instead of percentage stops
                signal_data = {
                    'strength': signal.strength,
                    'volume_confirmation': signal.volume_confirmation,
                    'sector': signal.sector,
                    'combined_score': signal.combined_score
                }
                
                success = self.wyckoff_stop_manager.create_wyckoff_stop_strategy(
                    symbol=signal.symbol,
                    phase=signal.phase,
                    initial_price=current_price,
                    entry_signal_data=signal_data
                )
                
                if success:
                    self.logger.info(f"🎯 Wyckoff context stop strategy created for {signal.symbol} ({signal.phase})")
                else:
                    self.logger.warning(f"⚠️ Context stop creation failed for {signal.symbol}, using fallback")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                if 'expired' in error_msg.lower() or 'login' in error_msg.lower():
                    self.logger.warning(f"⚠️ Session expired during {signal.symbol} buy, retrying...")
                    if self.full_reauthentication():
                        return self.execute_buy_order(signal, account, dynamic_trade_amount)
                
                self.logger.error(f"❌ Buy order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy order for {signal.symbol}: {e}")
            return False
    
    def execute_sell_order(self, signal: WyckoffSignal, position: Dict, account=None) -> bool:
        """Execute a sell order for entire position with session validation"""
        try:
            # Validate session before trading
            if not self.validate_and_refresh_session():
                self.logger.error(f"❌ Session validation failed for {signal.symbol} sell order")
                return False
            
            # If no account specified, find the account that holds this position
            if account is None:
                target_account_type = position['account_type']
                enabled_accounts = self.get_enabled_accounts()
                account = next((acc for acc in enabled_accounts if acc.account_type == target_account_type), None)
                
                if not account:
                    self.logger.error(f"❌ Could not find account type {target_account_type} for {signal.symbol}")
                    return False
            
            # Switch to the trading account
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to {account.account_type} account for {signal.symbol}")
                return False
            
            shares_to_sell = position['shares']
            
            self.logger.info(f"💸 Selling {shares_to_sell} shares of {signal.symbol} (Wyckoff Signal) from {account.account_type}")
            
            # Place market sell order
            wb = self.main_system.wb
            order_result = wb.place_order(
                stock=signal.symbol,
                price=0,  # Market price
                action='SELL',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_sell,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                self.logger.info(f"✅ Sell order placed: {signal.symbol} - Order ID: {order_id}")
                
                # Get current price for logging
                quote_data = wb.get_quote(signal.symbol)
                current_price = float(quote_data.get('close', signal.price)) if quote_data else signal.price
                
                # Log the trade
                self.database.log_trade(
                    symbol=signal.symbol,
                    action='SELL',
                    quantity=shares_to_sell,
                    price=current_price,
                    signal_phase=signal.phase,
                    signal_strength=signal.strength,
                    account_type=account.account_type,
                    order_id=order_id
                )
                
                # Calculate P&L BEFORE updating position
                if position.get('total_invested', 0) > 0:
                    profit_loss = (current_price - position['avg_cost']) * shares_to_sell
                    profit_loss_pct = (profit_loss / position['total_invested']) * 100
                    self.logger.info(f"📊 Wyckoff Sell P&L for {signal.symbol}: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
                
                # Update position (set to zero) AFTER P&L calculation
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=-shares_to_sell,  # Negative to reduce position
                    cost=current_price,
                    account_type=account.account_type
                )
                
                # Deactivate stop strategies
                self.database.deactivate_stop_strategies(signal.symbol)
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                if 'expired' in error_msg.lower() or 'login' in error_msg.lower():
                    self.logger.warning(f"⚠️ Session expired during {signal.symbol} sell, retrying...")
                    if self.full_reauthentication():
                        return self.execute_sell_order(signal, position, account)
                
                self.logger.error(f"❌ Sell order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing sell order for {signal.symbol}: {e}")
            return False
    
    def run_trading_cycle(self) -> Tuple[int, int]:
        """Run one complete trading cycle with multi-account support and session validation"""
        trades_executed = 0
        stop_losses_executed = 0
        errors = 0
        
        try:
            # Validate session at the start of trading cycle
            self.logger.info("🔐 Validating trading session...")
            if not self.validate_and_refresh_session():
                self.logger.error("❌ Session validation failed at start of trading cycle")
                return (0, 0), 1
            
            # Get all enabled accounts
            enabled_accounts = self.get_enabled_accounts()
            if not enabled_accounts:
                self.logger.error("❌ No enabled accounts found")
                return (0, 0), 1
            
            self.logger.info(f"💼 Found {len(enabled_accounts)} enabled accounts:")
            for i, account in enumerate(enabled_accounts, 1):
                self.logger.info(f"  {i}. {account.account_type}: ${account.settled_funds:.2f} available")
            
            # STEP 1: Check and execute Wyckoff context stops first
            self.logger.info("🎯 STEP 1: Checking Wyckoff context stops across all accounts...")
            stop_losses_executed = self.check_stop_losses()
            
            if stop_losses_executed > 0:
                self.logger.info(f"🎯 Executed {stop_losses_executed} Wyckoff context stops")
            
            # STEP 2: Check for Wyckoff signals
            self.logger.info("🔍 STEP 2: Scanning for Wyckoff signals...")
            
            # Scan for signals
            all_signals = self.scan_for_signals()
            if all_signals:
                # Log all signals
                for signal in all_signals:
                    self.database.log_signal(signal)
                
                # Get current positions from all accounts
                current_positions = self.get_current_positions()
                held_symbols = list(current_positions.keys())
                
                self.logger.info(f"📊 Current positions: {len(held_symbols)} stocks across all accounts")
                if held_symbols:
                    for symbol, pos in current_positions.items():
                        self.logger.info(f"  {symbol}: {pos['shares']:.5f} shares @ ${pos['avg_cost']:.2f} ({pos['account_type']})")
                
                # Check for sell signals first
                sell_signals = self.filter_sell_signals(all_signals, held_symbols)
                
                for signal in sell_signals[:3]:  # Limit to 3 sells per run
                    if signal.symbol in current_positions:
                        position = current_positions[signal.symbol]
                        
                        if self.execute_sell_order(signal, position):
                            trades_executed += 1
                            self.database.log_signal(signal, "WYCKOFF_SELL_EXECUTED")
                            # Remove from current positions
                            del current_positions[signal.symbol]
                        else:
                            errors += 1
                            self.database.log_signal(signal, "WYCKOFF_SELL_FAILED")
                
                # Check for buy signals (try multiple accounts)
                buy_signals = self.filter_buy_signals(all_signals)
                
                if buy_signals:
                    # Calculate total available cash across all accounts for dynamic sizing
                    total_available_cash = sum(acc.settled_funds for acc in enabled_accounts)
                    dynamic_trade_amount = self.get_dynamic_trade_amount(total_available_cash)
                    dynamic_min_balance = self.get_dynamic_min_balance(total_available_cash)
                    
                    self.logger.info(f"💰 Processing {len(buy_signals)} buy signals across multiple accounts...")
                    self.logger.info(f"📊 Dynamic position sizing: ${dynamic_trade_amount:.2f} per trade")
                    self.logger.info(f"🛡️ Dynamic balance preservation: ${dynamic_min_balance:.2f} minimum balance")
                    
                    executed_buys = 0
                    signal_index = 0
                    
                    # Try each account in priority order
                    for account_priority, account in enumerate(enabled_accounts, 1):
                        if signal_index >= len(buy_signals):
                            break
                        
                        # Calculate trades this account can afford
                        account_max_trades = int((account.settled_funds - dynamic_min_balance) / dynamic_trade_amount)
                        account_max_trades = min(account_max_trades, 3)  # Limit per account
                        
                        if account_max_trades <= 0:
                            self.logger.info(f"💸 {account.account_type}: Insufficient funds")
                            continue
                        
                        self.logger.info(f"💰 {account.account_type}: Can afford {account_max_trades} positions")
                        
                        account_buys = 0
                        
                        # Execute trades for this account
                        while signal_index < len(buy_signals) and account_buys < account_max_trades:
                            signal = buy_signals[signal_index]
                            signal_index += 1
                            
                            # Check if we already hold this stock
                            if signal.symbol in current_positions:
                                # Check if we want to add to the position
                                last_purchase = self.database.get_position(signal.symbol)
                                if last_purchase:
                                    # Only add if last purchase was more than 3 days ago
                                    last_date = datetime.strptime(last_purchase['last_purchase_date'], '%Y-%m-%d')
                                    if (datetime.now() - last_date).days >= 3:
                                        if self.execute_buy_order(signal, account, dynamic_trade_amount):
                                            trades_executed += 1
                                            executed_buys += 1
                                            account_buys += 1
                                            self.database.log_signal(signal, f"BUY_ADD_EXECUTED_{account.account_type}")
                                        else:
                                            errors += 1
                                            self.database.log_signal(signal, f"BUY_ADD_FAILED_{account.account_type}")
                                    else:
                                        self.database.log_signal(signal, "BUY_SKIP_RECENT")
                            else:
                                # New position
                                if self.execute_buy_order(signal, account, dynamic_trade_amount):
                                    trades_executed += 1
                                    executed_buys += 1
                                    account_buys += 1
                                    self.database.log_signal(signal, f"BUY_NEW_EXECUTED_{account.account_type}")
                                else:
                                    errors += 1
                                    self.database.log_signal(signal, f"BUY_NEW_FAILED_{account.account_type}")
                        
                        if account_buys > 0:
                            self.logger.info(f"✅ {account.account_type}: Executed {account_buys} buy orders")
                    
                    self.logger.info(f"📊 Total buy orders executed: {executed_buys}")
                
            else:
                self.logger.info("📭 No Wyckoff signals found")
            
            return (trades_executed, stop_losses_executed), errors
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
            return (trades_executed, stop_losses_executed), errors + 1
    
    def run(self) -> bool:
        """Main bot execution with Wyckoff context stops"""
        start_time = time.time()
        signals_found = 0
        trades_executed = 0
        stop_losses_executed = 0
        errors = 0
        log_details = ""
        
        try:
            self.logger.info("🚀 Starting Enhanced Wyckoff Trading Bot with Context Stops")
            
            # Initialize all systems
            if not self.initialize_systems():
                self.database.log_bot_run(0, 0, 0, 1, 0, 0, "INIT_FAILED", "System initialization failed")
                return False
            
            # Authenticate and setup accounts
            if not self.authenticate_and_setup_accounts():
                self.database.log_bot_run(0, 0, 0, 1, 0, 0, "AUTH_FAILED", "Authentication failed")
                return False
            
            # Run trading cycle
            (trades_executed, stop_losses_executed), cycle_errors = self.run_trading_cycle()
            errors += cycle_errors
            
            # Get portfolio summary from all enabled accounts
            enabled_accounts = self.get_enabled_accounts()
            total_portfolio_value = 0
            total_available_cash = 0
            
            for account in enabled_accounts:
                total_portfolio_value += account.net_liquidation
                total_available_cash += account.settled_funds
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create summary
            account_summary = f"{len(enabled_accounts)} accounts" if len(enabled_accounts) > 1 else enabled_accounts[0].account_type if enabled_accounts else "None"
            log_details = f"Execution time: {execution_time:.1f}s, Accounts: {account_summary}, Context stops: Wyckoff-aligned"
            
            status = "SUCCESS" if errors == 0 else "SUCCESS_WITH_ERRORS" if (trades_executed + stop_losses_executed) > 0 else "FAILED"
            
            # Log bot run
            self.database.log_bot_run(
                signals_found=len(self.wyckoff_strategy.symbols) if self.wyckoff_strategy else 0,
                trades_executed=trades_executed,
                stop_losses_executed=stop_losses_executed,
                errors=errors,
                portfolio_value=total_portfolio_value,
                available_cash=total_available_cash,
                status=status,
                log_details=log_details
            )
            
            # Final summary
            self.logger.info("📊 WYCKOFF CONTEXT STOP TRADING SESSION SUMMARY")
            self.logger.info(f"   Wyckoff Trades: {trades_executed}")
            self.logger.info(f"   Context Stops: {stop_losses_executed}")
            self.logger.info(f"   Total Actions: {trades_executed + stop_losses_executed}")
            self.logger.info(f"   Errors: {errors}")
            self.logger.info(f"   Total Portfolio Value: ${total_portfolio_value:.2f}")
            self.logger.info(f"   Total Available Cash: ${total_available_cash:.2f}")
            self.logger.info(f"   Position Size Used: ${self.get_dynamic_trade_amount(total_available_cash):.2f} per trade")
            self.logger.info(f"   Balance Preserved: ${self.get_dynamic_min_balance(total_available_cash):.2f} minimum balance")
            self.logger.info(f"   Execution Time: {execution_time:.1f}s")
            self.logger.info(f"   Stop Strategy: Wyckoff phase-context methodology")
            self.logger.info(f"   Session Management: Enhanced with validation")
            self.logger.info(f"   Accounts Used: {len(enabled_accounts)} ({', '.join(acc.account_type for acc in enabled_accounts)})")
            
            if (trades_executed + stop_losses_executed) > 0:
                self.logger.info("✅ Wyckoff context stop bot completed successfully with actions")
            elif errors == 0:
                self.logger.info("✅ Wyckoff context stop bot completed successfully (no actions needed)")
            else:
                self.logger.warning("⚠️ Wyckoff context stop bot completed with errors")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in Wyckoff context stop bot: {e}")
            self.logger.error(traceback.format_exc())
            
            # Log failed run
            self.database.log_bot_run(0, 0, 0, 1, 0, 0, "CRITICAL_ERROR", str(e))
            return False
        
        finally:
            # Cleanup
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point for the Wyckoff trading bot with context stops"""
    print("🤖 Enhanced Wyckoff Trading Bot with Phase-Context Stops Starting...")
    
    bot = EnhancedWyckoffTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Wyckoff context stop trading bot completed successfully!")
        sys.exit(0)
    else:
        print("❌ Wyckoff context stop trading bot failed! Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()