#!/usr/bin/env python3
"""
COMPLETE ENHANCED FRACTIONAL POSITION BUILDING SYSTEM - FULL VERSION
Integrates all advanced exit management, Wyckoff warnings, and portfolio protection
This is the complete replacement for fractional_position_system.py with ALL features
"""

import sys
import logging
import traceback
import sqlite3
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Import existing systems
from main import MainSystem
from strategies.wyckoff.wyckoff import WyckoffPnFStrategy, WyckoffSignal
from config.config import PersonalTradingConfig


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


@dataclass
class PositionRisk:
    """Position risk assessment"""
    symbol: str
    current_risk_pct: float
    time_held_days: int
    position_size_pct: float
    entry_phase: str
    volatility_percentile: float
    recommended_action: str


class EnhancedTradingDatabase:
    """
    DEFINITIVE: Enhanced database manager with comprehensive tracking
    This is the single source of truth for all database operations
    """
    
    def __init__(self, db_path="data/trading_bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.bot_id = "enhanced_wyckoff_bot_v2"
        self.init_database()
    
    def init_database(self):
        """Initialize database tables with enhanced capabilities"""
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
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
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
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    trade_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced positions table with FIXED multi-account support
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT NOT NULL,
                    account_type TEXT NOT NULL, 
                    total_shares REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    total_invested REAL NOT NULL,
                    first_purchase_date TEXT NOT NULL,
                    last_purchase_date TEXT NOT NULL,
                    entry_phase TEXT DEFAULT 'UNKNOWN',
                    entry_strength REAL DEFAULT 0.0,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, account_type, bot_id)
                )
            ''')
            
            # Enhanced positions table for detailed tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions_enhanced (
                    symbol TEXT NOT NULL,
                    account_type TEXT NOT NULL, 
                    total_shares REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    total_invested REAL NOT NULL,
                    first_purchase_date TEXT NOT NULL,
                    last_purchase_date TEXT NOT NULL,
                    entry_phase TEXT DEFAULT 'UNKNOWN',
                    entry_strength REAL DEFAULT 0.0,
                    position_size_pct REAL DEFAULT 0.1,
                    time_held_days INTEGER DEFAULT 0,
                    volatility_percentile REAL DEFAULT 0.5,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, account_type, bot_id)
                )
            ''')
            
            # Enhanced stop strategies table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stop_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    initial_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    stop_percentage REAL NOT NULL,
                    trailing_high REAL,
                    key_support_level REAL,
                    key_resistance_level REAL,
                    breakout_level REAL,
                    pullback_low REAL,
                    time_entered TIMESTAMP,
                    context_data TEXT,
                    stop_reason TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Partial sales tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS partial_sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sale_date TEXT NOT NULL,
                    shares_sold REAL NOT NULL,
                    sale_price REAL NOT NULL,
                    sale_reason TEXT NOT NULL,
                    remaining_shares REAL NOT NULL,
                    gain_pct REAL,
                    profit_amount REAL,
                    scaling_level TEXT,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Bot runs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    signals_found INTEGER NOT NULL,
                    trades_executed INTEGER NOT NULL,
                    wyckoff_sells INTEGER DEFAULT 0,
                    profit_scales INTEGER DEFAULT 0,
                    emergency_exits INTEGER DEFAULT 0,
                    errors_encountered INTEGER NOT NULL,
                    total_portfolio_value REAL,
                    available_cash REAL,
                    emergency_mode BOOLEAN DEFAULT FALSE,
                    market_condition TEXT,
                    portfolio_drawdown_pct REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    log_details TEXT,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_date_symbol ON trades(date, symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stop_strategies_symbol ON stop_strategies(symbol, is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON positions(bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_enhanced_bot_id ON positions_enhanced(bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_partial_sales_symbol ON partial_sales(symbol, sale_date)')
    
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
    
    def update_position(self, symbol: str, shares: float, cost: float, account_type: str, 
                       entry_phase: str = None, entry_strength: float = None):
        """
        DEFINITIVE: Update position tracking with enhanced data for a specific account
        Updates both positions and positions_enhanced tables
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # Update main positions table
            self._update_positions_table(conn, symbol, shares, cost, account_type, 
                                       entry_phase, entry_strength, today)
            
            # Update enhanced positions table
            self._update_positions_enhanced_table(conn, symbol, shares, cost, account_type, 
                                                entry_phase, entry_strength, today)
    
    def _update_positions_table(self, conn, symbol: str, shares: float, cost: float, 
                              account_type: str, entry_phase: str, entry_strength: float, today: str):
        """Update main positions table"""
        # Use symbol, account_type, AND bot_id to find the record
        existing = conn.execute(
            '''SELECT total_shares, avg_cost, total_invested, first_purchase_date, 
                    entry_phase, entry_strength FROM positions 
            WHERE symbol = ? AND account_type = ? AND bot_id = ?''',
            (symbol, account_type, self.bot_id)
        ).fetchone()
        
        if existing:
            old_shares, old_avg_cost, old_invested, first_date, old_phase, old_strength = existing
            new_shares = old_shares + shares
            
            if new_shares > 0:
                new_invested = old_invested + (shares * cost)
                new_avg_cost = new_invested / new_shares
                use_phase = entry_phase or old_phase or 'UNKNOWN'
                use_strength = entry_strength or old_strength or 0.0
            else:
                new_invested = 0
                new_avg_cost = 0
                use_phase = old_phase
                use_strength = old_strength
            
            conn.execute('''
                UPDATE positions 
                SET total_shares = ?, avg_cost = ?, total_invested = ?, 
                    last_purchase_date = ?, entry_phase = ?, entry_strength = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND account_type = ? AND bot_id = ?
            ''', (new_shares, new_avg_cost, new_invested, today, use_phase, use_strength, 
                  symbol, account_type, self.bot_id))
        else:
            # Insert new position with account_type
            conn.execute('''
                INSERT INTO positions (symbol, account_type, total_shares, avg_cost, total_invested, 
                                    first_purchase_date, last_purchase_date, 
                                    entry_phase, entry_strength, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, account_type, shares, cost, shares * cost, today, today,
                entry_phase or 'UNKNOWN', entry_strength or 0.0, self.bot_id))
    
    def _update_positions_enhanced_table(self, conn, symbol: str, shares: float, cost: float, 
                                       account_type: str, entry_phase: str, entry_strength: float, today: str):
        """Update enhanced positions table with additional metrics"""
        # Calculate additional metrics
        position_size_pct = 0.1  # Default, would be calculated based on portfolio value
        time_held_days = 0  # Would be calculated from first_purchase_date
        volatility_percentile = 0.5  # Default, would be calculated from market data
        
        # Use symbol, account_type, AND bot_id to find the record
        existing = conn.execute(
            '''SELECT total_shares, avg_cost, total_invested, first_purchase_date, 
                    entry_phase, entry_strength, position_size_pct, time_held_days, volatility_percentile 
            FROM positions_enhanced 
            WHERE symbol = ? AND account_type = ? AND bot_id = ?''',
            (symbol, account_type, self.bot_id)
        ).fetchone()
        
        if existing:
            old_shares, old_avg_cost, old_invested, first_date, old_phase, old_strength, old_size_pct, old_days, old_vol = existing
            new_shares = old_shares + shares
            
            # Calculate time held
            try:
                first_purchase_dt = datetime.strptime(first_date, '%Y-%m-%d')
                time_held_days = (datetime.now() - first_purchase_dt).days
            except:
                time_held_days = old_days
            
            if new_shares > 0:
                new_invested = old_invested + (shares * cost)
                new_avg_cost = new_invested / new_shares
                use_phase = entry_phase or old_phase or 'UNKNOWN'
                use_strength = entry_strength or old_strength or 0.0
            else:
                new_invested = 0
                new_avg_cost = 0
                use_phase = old_phase
                use_strength = old_strength
            
            conn.execute('''
                UPDATE positions_enhanced 
                SET total_shares = ?, avg_cost = ?, total_invested = ?, 
                    last_purchase_date = ?, entry_phase = ?, entry_strength = ?,
                    position_size_pct = ?, time_held_days = ?, volatility_percentile = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND account_type = ? AND bot_id = ?
            ''', (new_shares, new_avg_cost, new_invested, today, use_phase, use_strength,
                  position_size_pct, time_held_days, volatility_percentile,
                  symbol, account_type, self.bot_id))
        else:
            # Insert new enhanced position
            conn.execute('''
                INSERT INTO positions_enhanced (symbol, account_type, total_shares, avg_cost, total_invested, 
                                              first_purchase_date, last_purchase_date, 
                                              entry_phase, entry_strength, position_size_pct,
                                              time_held_days, volatility_percentile, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, account_type, shares, cost, shares * cost, today, today,
                  entry_phase or 'UNKNOWN', entry_strength or 0.0, position_size_pct,
                  time_held_days, volatility_percentile, self.bot_id))
    
    def get_position(self, symbol: str, account_type: str = None) -> Optional[Dict]:
        """Get current position for a symbol in a specific account or all accounts"""
        with sqlite3.connect(self.db_path) as conn:
            if account_type:
                # Get position for specific account from enhanced table
                result = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost, total_invested, 
                           first_purchase_date, last_purchase_date, entry_phase, 
                           entry_strength, position_size_pct, time_held_days, 
                           volatility_percentile, bot_id, updated_at
                    FROM positions_enhanced 
                    WHERE symbol = ? AND account_type = ? AND bot_id = ?
                ''', (symbol, account_type, self.bot_id)).fetchone()
                
                if result:
                    columns = ['symbol', 'account_type', 'total_shares', 'avg_cost', 'total_invested', 
                              'first_purchase_date', 'last_purchase_date', 'entry_phase', 
                              'entry_strength', 'position_size_pct', 'time_held_days',
                              'volatility_percentile', 'bot_id', 'updated_at']
                    return dict(zip(columns, result))
            else:
                # Get all positions for this symbol across accounts
                results = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost, total_invested, 
                           first_purchase_date, last_purchase_date, entry_phase, 
                           entry_strength, position_size_pct, time_held_days,
                           volatility_percentile, bot_id, updated_at
                    FROM positions_enhanced 
                    WHERE symbol = ? AND bot_id = ? AND total_shares > 0
                ''', (symbol, self.bot_id)).fetchall()
                
                if results:
                    columns = ['symbol', 'account_type', 'total_shares', 'avg_cost', 'total_invested', 
                              'first_purchase_date', 'last_purchase_date', 'entry_phase', 
                              'entry_strength', 'position_size_pct', 'time_held_days',
                              'volatility_percentile', 'bot_id', 'updated_at']
                    return [dict(zip(columns, row)) for row in results]
        
        return None
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions grouped by account"""
        positions = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost, total_invested,
                           entry_phase, entry_strength, first_purchase_date, last_purchase_date,
                           position_size_pct, time_held_days, volatility_percentile
                    FROM positions_enhanced 
                    WHERE total_shares > 0 AND bot_id = ?
                    ORDER BY account_type, symbol
                ''', (self.bot_id,)).fetchall()
                
                for row in results:
                    symbol, account_type, shares, avg_cost, invested, entry_phase, entry_strength, first_date, last_date, size_pct, days, vol_pct = row
                    
                    # Create account-specific key
                    position_key = f"{symbol}_{account_type}"
                    
                    positions[position_key] = {
                        'symbol': symbol,
                        'account_type': account_type,
                        'shares': shares,
                        'avg_cost': avg_cost,
                        'total_invested': invested,
                        'entry_phase': entry_phase or 'UNKNOWN',
                        'entry_strength': entry_strength or 0.0,
                        'first_purchase_date': first_date,
                        'last_purchase_date': last_date,
                        'position_size_pct': size_pct or 0.1,
                        'time_held_days': days or 0,
                        'volatility_percentile': vol_pct or 0.5
                    }
        except Exception as e:
            print(f"Error getting positions: {e}")
        
        return positions
    
    def log_partial_sale(self, symbol: str, shares_sold: float, sale_price: float, 
                        sale_reason: str, remaining_shares: float, gain_pct: float, 
                        profit_amount: float, scaling_level: str):
        """Log partial sale for tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO partial_sales (symbol, sale_date, shares_sold, sale_price, 
                                         sale_reason, remaining_shares, gain_pct, profit_amount, 
                                         scaling_level, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, datetime.now().strftime('%Y-%m-%d'), shares_sold, sale_price,
                sale_reason, remaining_shares, gain_pct, profit_amount, scaling_level, self.bot_id
            ))
    
    def already_scaled_at_level(self, symbol: str, gain_pct: float) -> bool:
        """Check if we already scaled at this gain level"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('''
                SELECT COUNT(*) FROM partial_sales 
                WHERE symbol = ? AND bot_id = ? AND gain_pct >= ? AND sale_date = ?
            ''', (symbol, self.bot_id, gain_pct - 0.01, datetime.now().strftime('%Y-%m-%d'))).fetchone()
            
            return result[0] > 0
    
    def deactivate_stop_strategies(self, symbol: str):
        """Deactivate all stop strategies for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE stop_strategies 
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND bot_id = ?
            ''', (symbol, self.bot_id))
    
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
    
    def log_bot_run(self, signals_found: int, trades_executed: int, wyckoff_sells: int,
                    profit_scales: int, emergency_exits: int, errors: int, 
                    portfolio_value: float, available_cash: float, emergency_mode: bool,
                    market_condition: str, portfolio_drawdown_pct: float,
                    status: str, log_details: str):
        """Log enhanced bot run statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO bot_runs (run_date, signals_found, trades_executed, wyckoff_sells,
                                    profit_scales, emergency_exits, errors_encountered, 
                                    total_portfolio_value, available_cash, emergency_mode,
                                    market_condition, portfolio_drawdown_pct, status, log_details, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signals_found, trades_executed, wyckoff_sells, profit_scales, emergency_exits,
                errors, portfolio_value, available_cash, emergency_mode, market_condition,
                portfolio_drawdown_pct, status, log_details, self.bot_id
            ))


class EnhancedWyckoffAnalyzer:
    """Enhanced Wyckoff analyzer with advanced warning signals"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def analyze_advanced_warnings(self, symbol: str, data: pd.DataFrame, 
                                current_price: float, entry_data: Dict) -> List[WyckoffWarningSignal]:
        """Analyze for advanced Wyckoff warning signals - FIXED symbol handling"""
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
            
            # 1. UTAD - Upthrust After Distribution
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
        
    def _detect_sow(self, symbol: str, data: pd.DataFrame, current_price: float) -> Optional[WyckoffWarningSignal]:
        """Detect Sign of Weakness"""
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
        """Detect volume divergence on new highs"""
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
            earlier_high_vol = high_days['Volume'].head(-2).mean()
            
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
        """Detect breaks of key Wyckoff context levels"""
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
                    urgency='CRITICAL' if break_severity > 0.03 else 'HIGH'
                )
        except Exception as e:
            self.logger.debug(f"Context break detection error for {symbol}: {e}")
        
        return None


class DynamicAccountManager:
    """Manages dynamic position sizing based on real account values"""
    
    def __init__(self, logger):
        self.logger = logger
        self.last_update = None
        self.cached_config = None
        
    def get_dynamic_config(self, account_manager) -> Dict:
        """Get dynamic configuration based on real account values"""
        try:
            enabled_accounts = account_manager.get_enabled_accounts()
            if not enabled_accounts:
                return self._get_fallback_config()
            
            total_value = sum(acc.net_liquidation for acc in enabled_accounts)
            total_cash = sum(acc.settled_funds for acc in enabled_accounts)
            
            self.logger.info(f"💰 Real Account Values - Total: ${total_value:.2f}, Cash: ${total_cash:.2f}")
            
            config = self._calculate_dynamic_parameters(total_value, total_cash, len(enabled_accounts))
            self.cached_config = config
            self.last_update = datetime.now()
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error getting dynamic config: {e}")
            return self._get_fallback_config()
    
    def _calculate_dynamic_parameters(self, total_value: float, total_cash: float, num_accounts: int) -> Dict:
        """Calculate trading parameters based on real account values"""
        
        if total_cash < 200:
            base_position_pct = 0.15
            max_positions = 3
            min_balance_pct = 0.25
        elif total_cash < 500:
            base_position_pct = 0.12
            max_positions = 4
            min_balance_pct = 0.20
        elif total_cash < 1000:
            base_position_pct = 0.10
            max_positions = 5
            min_balance_pct = 0.15
        else:
            base_position_pct = 0.08
            max_positions = 6
            min_balance_pct = 0.12
        
        base_position_size = total_cash * base_position_pct
        min_balance_preserve = total_cash * min_balance_pct
        
        if base_position_size < 5.0:
            base_position_size = min(5.0, total_cash * 0.5)
        
        wyckoff_phases = {
            'ST': {'initial_allocation': 0.60, 'allow_additions': False, 'max_total_allocation': 0.60},
            'SOS': {'initial_allocation': 0.70, 'allow_additions': True, 'max_total_allocation': 1.0},
            'LPS': {'initial_allocation': 0.50, 'allow_additions': True, 'max_total_allocation': 1.0},
            'BU': {'initial_allocation': 0.40, 'allow_additions': True, 'max_total_allocation': 1.0},
            'Creek': {'initial_allocation': 0.0, 'allow_additions': False, 'max_total_allocation': 0.0}
        }
        
        profit_targets = [
            {'gain_pct': 0.08, 'sell_pct': 0.20, 'description': '8% gain: Take 20% profit'},
            {'gain_pct': 0.15, 'sell_pct': 0.25, 'description': '15% gain: Take 25% more'},
            {'gain_pct': 0.25, 'sell_pct': 0.30, 'description': '25% gain: Take 30% more'},
            {'gain_pct': 0.40, 'sell_pct': 0.25, 'description': '40% gain: Take final 25%'}
        ]
        
        return {
            'total_value': total_value,
            'total_cash': total_cash,
            'base_position_size': base_position_size,
            'base_position_pct': base_position_pct,
            'min_balance_preserve': min_balance_preserve,
            'max_positions': max_positions,
            'num_accounts': num_accounts,
            'wyckoff_phases': wyckoff_phases,
            'profit_targets': profit_targets,
            'calculated_at': datetime.now().isoformat()
        }
    
    def _get_fallback_config(self) -> Dict:
        """Fallback configuration"""
        return {
            'total_value': 300.0,
            'total_cash': 300.0,
            'base_position_size': 15.0,
            'base_position_pct': 0.15,
            'min_balance_preserve': 75.0,
            'max_positions': 3,
            'num_accounts': 2,
            'wyckoff_phases': {
                'ST': {'initial_allocation': 0.60, 'allow_additions': False, 'max_total_allocation': 0.60},
                'SOS': {'initial_allocation': 0.70, 'allow_additions': True, 'max_total_allocation': 1.0},
                'LPS': {'initial_allocation': 0.50, 'allow_additions': True, 'max_total_allocation': 1.0},
                'BU': {'initial_allocation': 0.40, 'allow_additions': True, 'max_total_allocation': 1.0},
                'Creek': {'initial_allocation': 0.0, 'allow_additions': False, 'max_total_allocation': 0.0}
            },
            'profit_targets': [
                {'gain_pct': 0.08, 'sell_pct': 0.20, 'description': '8% gain: Take 20% profit'},
                {'gain_pct': 0.15, 'sell_pct': 0.25, 'description': '15% gain: Take 25% more'},
                {'gain_pct': 0.25, 'sell_pct': 0.30, 'description': '25% gain: Take 30% more'}
            ],
            'calculated_at': datetime.now().isoformat()
        }


class DynamicProfitTargetCalculator:
    """Calculate dynamic profit targets based on multiple factors"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def calculate_dynamic_targets(self, position: Dict, market_data: Dict, 
                                volatility: float) -> List[Dict]:
        """Calculate dynamic profit targets based on position context"""
        try:
            entry_phase = position.get('entry_phase', 'ST')
            position_size_pct = position.get('position_size_pct', 0.1)
            
            # Calculate time held
            first_purchase = datetime.strptime(position['first_purchase_date'], '%Y-%m-%d')
            time_held = (datetime.now() - first_purchase).days
            
            account_size = market_data.get('account_value', 1000)
            
            base_targets = [
                {'gain_pct': 0.08, 'sell_pct': 0.20},
                {'gain_pct': 0.15, 'sell_pct': 0.25},
                {'gain_pct': 0.25, 'sell_pct': 0.30},
                {'gain_pct': 0.40, 'sell_pct': 0.25}
            ]
            
            # Phase multipliers
            phase_multipliers = {
                'ST': 0.8, 'SOS': 1.2, 'LPS': 1.0, 'BU': 0.9, 'Creek': 0.7
            }
            multiplier = phase_multipliers.get(entry_phase, 1.0)
            
            # Position size adjustment
            if position_size_pct > 0.15:
                multiplier *= 0.9
            elif position_size_pct < 0.05:
                multiplier *= 1.1
            
            # Volatility adjustment
            if volatility > 0.8:
                volatility_adj = 1.3
            elif volatility > 0.5:
                volatility_adj = 1.1
            else:
                volatility_adj = 0.9
            
            # Time adjustment
            if time_held > 30:
                time_adj = 1.2
            elif time_held > 14:
                time_adj = 1.1
            else:
                time_adj = 1.0
            
            # Apply adjustments
            dynamic_targets = []
            for target in base_targets:
                adjusted_target = {
                    'gain_pct': target['gain_pct'] * multiplier * volatility_adj,
                    'sell_pct': target['sell_pct'] * time_adj,
                    'reasoning': f"Phase:{entry_phase}, Vol:{volatility:.2f}, Days:{time_held}"
                }
                dynamic_targets.append(adjusted_target)
            
            return dynamic_targets
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic targets: {e}")
            return base_targets


class PortfolioRiskManager:
    """Manages portfolio-level risk and emergency exits"""
    
    def __init__(self, database, logger):
        self.database = database
        self.logger = logger
        self.MAX_PORTFOLIO_DRAWDOWN = 0.15
        self.MAX_INDIVIDUAL_LOSS = 0.20
        self.VIX_CRASH_THRESHOLD = 40
        
    def assess_portfolio_risk(self, account_manager, current_positions: Dict) -> Dict:
        """Assess overall portfolio risk"""
        risk_assessment = {
            'portfolio_drawdown_pct': 0.0,
            'positions_at_risk': [],
            'emergency_exits_needed': [],
            'market_condition': 'NORMAL',
            'recommended_actions': []
        }
        
        try:
            enabled_accounts = account_manager.get_enabled_accounts()
            total_current_value = sum(acc.net_liquidation for acc in enabled_accounts)
            
            portfolio_drawdown = self._calculate_portfolio_drawdown(total_current_value)
            risk_assessment['portfolio_drawdown_pct'] = portfolio_drawdown
            
            for symbol, position in current_positions.items():
                position_risk = self._assess_individual_position_risk(symbol, position)
                if position_risk.current_risk_pct > 0.10:
                    risk_assessment['positions_at_risk'].append(position_risk)
                
                if position_risk.current_risk_pct > self.MAX_INDIVIDUAL_LOSS:
                    risk_assessment['emergency_exits_needed'].append(position_risk)
            
            market_condition = self._assess_market_conditions()
            risk_assessment['market_condition'] = market_condition
            
            recommendations = self._generate_risk_recommendations(risk_assessment)
            risk_assessment['recommended_actions'] = recommendations
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
        
        return risk_assessment
    
    def _calculate_portfolio_drawdown(self, current_value: float) -> float:
        """Calculate portfolio drawdown from high water mark"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                results = conn.execute('''
                    SELECT total_portfolio_value 
                    FROM bot_runs 
                    WHERE bot_id = ? AND total_portfolio_value > 0
                    ORDER BY created_at DESC 
                    LIMIT 30
                ''', (self.database.bot_id,)).fetchall()
                
                if not results:
                    return 0.0
                
                historical_values = [row[0] for row in results]
                high_water_mark = max(historical_values)
                
                if high_water_mark > 0:
                    drawdown = (high_water_mark - current_value) / high_water_mark
                    return max(0.0, drawdown)
                
        except Exception as e:
            self.logger.error(f"Error calculating portfolio drawdown: {e}")
        
        return 0.0
    
    def _assess_individual_position_risk(self, symbol: str, position: Dict) -> PositionRisk:
        """Assess risk for individual position - FIXED symbol handling"""
        try:
            # FIXED: Extract the actual symbol from position dict, not the position key
            actual_symbol = position.get('symbol', symbol.split('_')[0])
            
            # Clean symbol - remove any account type suffixes
            if '_' in actual_symbol:
                actual_symbol = actual_symbol.split('_')[0]
            
            self.logger.debug(f"Assessing risk for {actual_symbol} (from key: {symbol})")
            
            ticker = yf.Ticker(actual_symbol)
            hist_data = ticker.history(period="1d")
            
            if hist_data.empty:
                self.logger.warning(f"No price data available for {actual_symbol}")
                return PositionRisk(
                    symbol=actual_symbol,
                    current_risk_pct=0.0,
                    time_held_days=0,
                    position_size_pct=position.get('position_size_pct', 0.1),
                    entry_phase=position.get('entry_phase', 'UNKNOWN'),
                    volatility_percentile=0.0,
                    recommended_action='MONITOR'
                )
            
            current_price = hist_data['Close'].iloc[-1]
            avg_cost = position['avg_cost']
            current_risk_pct = (avg_cost - current_price) / avg_cost if avg_cost > 0 else 0
            
            first_purchase = datetime.strptime(position['first_purchase_date'], '%Y-%m-%d')
            time_held_days = (datetime.now() - first_purchase).days
            
            # Get longer history for volatility calculation
            hist_data_long = ticker.history(period="3mo")
            if len(hist_data_long) > 20:
                returns = hist_data_long['Close'].pct_change().dropna()
                volatility_percentile = np.percentile(np.abs(returns), 80) if len(returns) > 0 else 0.02
            else:
                volatility_percentile = 0.02
            
            if current_risk_pct > 0.20:
                recommended_action = "EMERGENCY_EXIT"
            elif current_risk_pct > 0.15:
                recommended_action = "CONSIDER_EXIT"
            elif current_risk_pct > 0.10:
                recommended_action = "MONITOR_CLOSELY"
            else:
                recommended_action = "HOLD"
            
            return PositionRisk(
                symbol=actual_symbol,
                current_risk_pct=current_risk_pct,
                time_held_days=time_held_days,
                position_size_pct=position.get('position_size_pct', 0.1),
                entry_phase=position.get('entry_phase', 'UNKNOWN'),
                volatility_percentile=volatility_percentile,
                recommended_action=recommended_action
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing position risk for {symbol}: {e}")
            # Extract symbol from position or key
            clean_symbol = position.get('symbol', symbol.split('_')[0])
            return PositionRisk(clean_symbol, 0.0, 0, 0.0, 'UNKNOWN', 0.0, 'ERROR')

        
    def _assess_market_conditions(self) -> str:
        """Assess overall market conditions"""
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            
            if len(vix_data) > 0:
                current_vix = vix_data['Close'].iloc[-1]
                
                if current_vix > self.VIX_CRASH_THRESHOLD:
                    return "MARKET_CRASH"
                elif current_vix > 30:
                    return "HIGH_VOLATILITY"
                elif current_vix > 20:
                    return "ELEVATED_RISK"
                else:
                    return "NORMAL"
            
        except Exception as e:
            self.logger.error(f"Error assessing market conditions: {e}")
        
        return "UNKNOWN"
    
    def _generate_risk_recommendations(self, risk_assessment: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_assessment['portfolio_drawdown_pct'] > self.MAX_PORTFOLIO_DRAWDOWN:
            recommendations.append("EMERGENCY: Portfolio drawdown exceeds 15% - consider liquidating all positions")
        elif risk_assessment['portfolio_drawdown_pct'] > 0.10:
            recommendations.append("WARNING: Portfolio down 10%+ - reduce position sizes and tighten stops")
        
        if risk_assessment['market_condition'] == 'MARKET_CRASH':
            recommendations.append("CRITICAL: VIX > 40 detected - emergency exit all positions")
        elif risk_assessment['market_condition'] == 'HIGH_VOLATILITY':
            recommendations.append("CAUTION: High volatility detected - reduce position sizes")
        
        if risk_assessment['emergency_exits_needed']:
            symbols = [pos.symbol for pos in risk_assessment['emergency_exits_needed']]
            recommendations.append(f"URGENT: Exit positions with >20% loss: {', '.join(symbols)}")
        
        if not recommendations:
            recommendations.append("Portfolio risk within acceptable parameters")
        
        return recommendations


class SmartFractionalPositionManager:
    """Enhanced position manager with Wyckoff selling and dynamic sizing"""
    
    def __init__(self, database, dynamic_account_manager, logger):
        self.database = database
        self.dynamic_manager = dynamic_account_manager
        self.logger = logger
        self.current_config = None
    
    def update_config(self, account_manager):
        """Update configuration based on current account values"""
        self.current_config = self.dynamic_manager.get_dynamic_config(account_manager)
        return self.current_config
    
    def get_position_size_for_signal(self, signal: WyckoffSignal) -> float:
        """Calculate position size for a specific Wyckoff signal"""
        if not self.current_config:
            return 10.0
        
        base_size = self.current_config['base_position_size']
        phase_config = self.current_config['wyckoff_phases'].get(signal.phase, {})
        initial_allocation = phase_config.get('initial_allocation', 0.5)
        
        position_size = base_size * initial_allocation
        position_size = max(position_size, 5.0)
        
        max_position = self.current_config['total_cash'] - self.current_config['min_balance_preserve']
        position_size = min(position_size, max_position)
        
        self.logger.debug(f"💰 {signal.symbol} ({signal.phase}): ${position_size:.2f} position")
        
        return position_size
    
    def check_dynamic_profit_scaling(self, symbol: str, position: Dict, dynamic_targets: List[Dict]) -> Optional[Dict]:
        """Check for profit scaling using dynamic targets"""
        try:
            # This would integrate with the main bot's quote system
            # For now, return None to indicate no scaling needed
            return None
        except Exception as e:
            self.logger.error(f"Error checking dynamic scaling for {symbol}: {e}")
            return None


class ComprehensiveExitManager:
    """Main exit management coordinator"""
    
    def __init__(self, database, logger):
        self.database = database
        self.logger = logger
        self.wyckoff_analyzer = EnhancedWyckoffAnalyzer(logger)
        self.profit_calculator = DynamicProfitTargetCalculator(logger)
        self.risk_manager = PortfolioRiskManager(database, logger)
        self.last_reconciliation = None
    
    def reconcile_positions(self, wb_client, account_manager) -> Dict:
        """Compare database positions with actual Webull holdings BY ACCOUNT"""
        reconciliation_report = {
            'discrepancies_found': [],
            'positions_synced': 0,
            'positions_corrected': 0,
            'ghost_positions_removed': 0
        }
        
        try:
            # Get real positions from each account
            enabled_accounts = account_manager.get_enabled_accounts()
            real_positions = {}
            
            for account in enabled_accounts:
                for position in account.positions:
                    # Create account-specific key
                    position_key = f"{position['symbol']}_{account.account_type}"
                    real_positions[position_key] = {
                        'symbol': position['symbol'],
                        'account_type': account.account_type,
                        'total_shares': position['quantity'],
                        'avg_cost': position['cost_price']
                    }
            
            # Get bot positions from database (by account)
            bot_positions = {}
            with sqlite3.connect(self.database.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost 
                    FROM positions 
                    WHERE bot_id = ?
                ''', (self.database.bot_id,)).fetchall()
                
                for symbol, account_type, shares, avg_cost in results:
                    position_key = f"{symbol}_{account_type}"
                    bot_positions[position_key] = {
                        'symbol': symbol,
                        'account_type': account_type,
                        'total_shares': shares,
                        'avg_cost': avg_cost
                    }
            
            # Compare and correct discrepancies
            all_position_keys = set(real_positions.keys()) | set(bot_positions.keys())
            
            for position_key in all_position_keys:
                real_pos = real_positions.get(position_key, {'total_shares': 0, 'avg_cost': 0})
                bot_pos = bot_positions.get(position_key, {'total_shares': 0, 'avg_cost': 0})
                
                real_shares = real_pos['total_shares']
                bot_shares = bot_pos['total_shares']
                
                if abs(real_shares - bot_shares) > 0.001:
                    symbol = position_key.split('_')[0]
                    account_type = '_'.join(position_key.split('_')[1:])
                    
                    discrepancy = {
                        'position_key': position_key,
                        'symbol': symbol,
                        'account_type': account_type,
                        'real_shares': real_shares,
                        'bot_shares': bot_shares,
                        'difference': real_shares - bot_shares
                    }
                    reconciliation_report['discrepancies_found'].append(discrepancy)
                    
                    # Auto-correct the position
                    if self._auto_correct_position_by_account(symbol, account_type, real_shares, real_pos['avg_cost']):
                        reconciliation_report['positions_corrected'] += 1
                
                reconciliation_report['positions_synced'] += 1
                
        except Exception as e:
            self.logger.error(f"Error during position reconciliation: {e}")
        
        return reconciliation_report
    
    def _auto_correct_position_by_account(self, symbol: str, account_type: str, real_shares: float, real_avg_cost: float) -> bool:
        """Auto-correct position discrepancies for a specific account."""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                if real_shares == 0:
                    # Remove ghost position from the specific account
                    conn.execute('''
                        DELETE FROM positions 
                        WHERE symbol = ? AND account_type = ? AND bot_id = ?
                    ''', (symbol, account_type, self.database.bot_id))
                    conn.execute('''
                        DELETE FROM positions_enhanced 
                        WHERE symbol = ? AND account_type = ? AND bot_id = ?
                    ''', (symbol, account_type, self.database.bot_id))
                    self.logger.info(f"Removed ghost position: {symbol} from {account_type}")
                else:
                    # Update/insert correct position in both tables
                    for table in ['positions', 'positions_enhanced']:
                        if table == 'positions_enhanced':
                            conn.execute('''
                                INSERT OR REPLACE INTO positions_enhanced 
                                (symbol, account_type, total_shares, avg_cost, total_invested,
                                 first_purchase_date, last_purchase_date, entry_phase, 
                                 entry_strength, position_size_pct, time_held_days,
                                 volatility_percentile, bot_id, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                symbol, account_type, real_shares, real_avg_cost, 
                                real_shares * real_avg_cost,
                                datetime.now().strftime('%Y-%m-%d'),
                                datetime.now().strftime('%Y-%m-%d'),
                                'RECONCILED', 0.0, 0.1, 0, 0.5, self.database.bot_id,
                                datetime.now().isoformat()
                            ))
                        else:
                            conn.execute('''
                                INSERT OR REPLACE INTO positions 
                                (symbol, account_type, total_shares, avg_cost, total_invested,
                                 first_purchase_date, last_purchase_date, entry_phase, 
                                 entry_strength, bot_id, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                symbol, account_type, real_shares, real_avg_cost, 
                                real_shares * real_avg_cost,
                                datetime.now().strftime('%Y-%m-%d'),
                                datetime.now().strftime('%Y-%m-%d'),
                                'RECONCILED', 0.0, self.database.bot_id,
                                datetime.now().isoformat()
                            ))
                    self.logger.info(f"Corrected position: {symbol} in {account_type} to {real_shares} shares @ ${real_avg_cost:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"Error auto-correcting position {symbol} in {account_type}: {e}")
            return False
    
    def run_comprehensive_analysis(self, wb_client, account_manager, current_positions: Dict) -> Dict:
        """Run complete exit analysis"""
        
        # Step 1: Reconcile positions
        self.logger.info("🔄 Reconciling positions...")
        reconciliation = self.reconcile_positions(wb_client, account_manager)
        
        # Step 2: Assess portfolio risk
        self.logger.info("⚠️ Assessing portfolio risk...")
        risk_assessment = self.risk_manager.assess_portfolio_risk(account_manager, current_positions)
        
        # Step 3: Analyze Wyckoff warnings
        self.logger.info("📊 Analyzing Wyckoff warnings...")
        wyckoff_warnings = {}
        
        for symbol, position in current_positions.items():
            try:
                ticker = yf.Ticker(position['symbol'])
                data = ticker.history(period="3mo")
                if len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                    warnings = self.wyckoff_analyzer.analyze_advanced_warnings(
                        position['symbol'], data, current_price, position
                    )
                    if warnings:
                        wyckoff_warnings[position['symbol']] = warnings
            except Exception as e:
                self.logger.error(f"Error analyzing {position['symbol']}: {e}")
        
        # Step 4: Calculate dynamic targets
        self.logger.info("💰 Calculating dynamic profit targets...")
        dynamic_targets = {}
        
        for symbol, position in current_positions.items():
            try:
                market_data = {'account_value': sum(acc.net_liquidation for acc in account_manager.get_enabled_accounts())}
                volatility = position.get('volatility_percentile', 0.5)
                targets = self.profit_calculator.calculate_dynamic_targets(position, market_data, volatility)
                dynamic_targets[position['symbol']] = targets
            except Exception as e:
                self.logger.error(f"Error calculating targets for {position['symbol']}: {e}")
        
        # Compile results
        analysis = {
            'reconciliation_report': reconciliation,
            'portfolio_risk_assessment': risk_assessment,
            'wyckoff_warnings': wyckoff_warnings,
            'dynamic_profit_targets': dynamic_targets,
            'immediate_actions_required': self._prioritize_actions(risk_assessment, wyckoff_warnings),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def _prioritize_actions(self, risk_assessment: Dict, wyckoff_warnings: Dict) -> List[Dict]:
        """Prioritize immediate actions"""
        actions = []
        
        # Emergency exits
        for position_risk in risk_assessment.get('emergency_exits_needed', []):
            actions.append({
                'action': 'EMERGENCY_EXIT',
                'symbol': position_risk.symbol,
                'reason': f"Position loss: {position_risk.current_risk_pct:.1%}",
                'urgency': 'CRITICAL',
                'priority': 1
            })
        
        # Critical Wyckoff warnings
        for symbol, warnings in wyckoff_warnings.items():
            for warning in warnings:
                if warning.urgency == 'CRITICAL':
                    actions.append({
                        'action': 'WYCKOFF_EXIT',
                        'symbol': symbol,
                        'reason': f"{warning.signal_type}: {warning.context}",
                        'urgency': 'CRITICAL',
                        'priority': 2
                    })
        
        # Market crash
        if risk_assessment.get('market_condition') == 'MARKET_CRASH':
            actions.append({
                'action': 'LIQUIDATE_ALL',
                'symbol': 'ALL',
                'reason': 'Market crash detected (VIX > 40)',
                'urgency': 'CRITICAL',
                'priority': 0
            })
        
        actions.sort(key=lambda x: x['priority'])
        return actions


class EnhancedFractionalTradingBot:
    """DEFINITIVE: Complete enhanced fractional trading bot with ALL features"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        self.dynamic_manager = None
        self.position_manager = None
        self.comprehensive_exit_manager = None
        
        # Enhanced features
        self.emergency_mode = False
        self.last_reconciliation = None
        
        # Trading parameters
        self.min_signal_strength = 0.5
        self.buy_phases = ['ST', 'SOS', 'LPS', 'BU']
        self.sell_phases = ['PS', 'SC']
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 ENHANCED FRACTIONAL TRADING BOT")
        self.logger.info("💰 Dynamic account sizing + Advanced Wyckoff exits")
        self.logger.info("🛡️ Portfolio protection + Position reconciliation")
    
    def initialize_systems(self) -> bool:
        """Initialize all enhanced systems"""
        try:
            self.logger.info("🔧 Initializing enhanced systems...")
            
            self.main_system = MainSystem()
            self.wyckoff_strategy = WyckoffPnFStrategy()
            self.database = EnhancedTradingDatabase()
            
            self.dynamic_manager = DynamicAccountManager(self.logger)
            self.position_manager = SmartFractionalPositionManager(
                self.database, self.dynamic_manager, self.logger
            )
            
            # Initialize comprehensive exit manager
            self.comprehensive_exit_manager = ComprehensiveExitManager(
                self.database, self.logger
            )
            
            self.logger.info("✅ Enhanced systems initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions from database"""
        return self.database.get_all_positions()
    
    def _ensure_valid_session(self) -> bool:
        """Ensure we have a valid session, refresh if needed"""
        try:
            # Test session with a simple API call
            if self.main_system.login_manager.check_login_status():
                return True
            
            self.logger.warning("⚠️ Session appears invalid, attempting refresh...")
            
            # Clear old session and force fresh login
            self.main_system.session_manager.clear_session()
            
            # Attempt fresh login
            if self.main_system.login_manager.login_automatically():
                self.logger.info("✅ Session refreshed successfully")
                self.main_system.session_manager.save_session(self.main_system.wb)
                return True
            else:
                self.logger.error("❌ Failed to refresh session")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error checking/refreshing session: {e}")
            return False    
        
    def execute_buy_order(self, signal: WyckoffSignal, account, position_size: float) -> bool:
        """Execute buy order with enhanced tracking and session management"""
        try:
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            # Check and refresh session before critical operations
            if not self._ensure_valid_session():
                self.logger.error(f"❌ Cannot establish valid session for {signal.symbol}")
                return False
            
            quote_data = self.main_system.wb.get_quote(signal.symbol)
            if not quote_data or 'close' not in quote_data:
                self.logger.error(f"❌ Could not get quote for {signal.symbol}")
                return False
            
            current_price = float(quote_data['close'])
            shares_to_buy = position_size / current_price
            shares_to_buy = round(shares_to_buy, 5)
            
            self.logger.info(f"💰 Buying {shares_to_buy:.5f} shares of {signal.symbol} at ${current_price:.2f}")
            self.logger.info(f"   Position: ${position_size:.2f} ({signal.phase} phase, strength: {signal.strength:.2f})")
            
            # Ensure session is still valid before placing order
            if not self._ensure_valid_session():
                self.logger.error(f"❌ Session expired before placing order for {signal.symbol}")
                return False
            
            order_result = self.main_system.wb.place_order(
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
                
                # Enhanced logging
                self.database.log_signal(signal, 'BUY_EXECUTED')
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
                
                # Enhanced position tracking WITH ACCOUNT TYPE
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=shares_to_buy,
                    cost=current_price,
                    account_type=account.account_type,
                    entry_phase=signal.phase,
                    entry_strength=signal.strength
                )
                
                self.logger.info(f"✅ Buy order executed: {signal.symbol} - Order ID: {order_id}")
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Buy order failed for {signal.symbol}: {error_msg}")
                
                # Check if it's a session issue
                if 'session' in error_msg.lower() or 'expired' in error_msg.lower():
                    self.logger.warning("⚠️ Session issue detected - will retry on next run")
                    # Clear session to force fresh login next time
                    self.main_system.session_manager.clear_session()
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy for {signal.symbol}: {e}")
            return False

    
    def execute_emergency_exit(self, action: Dict, current_positions: Dict) -> bool:
        """Execute emergency exits"""
        try:
            if action['action'] == 'LIQUIDATE_ALL':
                self.logger.critical("🚨 LIQUIDATING ALL POSITIONS - MARKET CRASH")
                success_count = 0
                for symbol, position in current_positions.items():
                    if self._emergency_liquidate_position(position['symbol'], position):
                        success_count += 1
                return success_count > 0
                
            elif action['symbol'] in [p['symbol'] for p in current_positions.values()]:
                # Find the position
                target_position = next((p for p in current_positions.values() 
                                      if p['symbol'] == action['symbol']), None)
                if target_position:
                    return self._emergency_liquidate_position(action['symbol'], target_position)
            
        except Exception as e:
            self.logger.error(f"❌ Error executing emergency action: {e}")
        
        return False
    
    def _emergency_liquidate_position(self, symbol: str, position: Dict) -> bool:
        """Emergency liquidation of single position"""
        try:
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts 
                           if acc.account_type == position['account_type']), None)
            
            if not account or not self.main_system.account_manager.switch_to_account(account):
                return False
            
            shares_to_sell = position['shares']
            
            self.logger.critical(f"🚨 EMERGENCY EXIT: {shares_to_sell:.5f} shares of {symbol}")
            
            order_result = self.main_system.wb.place_order(
                stock=symbol,
                price=0,
                action='SELL',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_sell,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'EMERGENCY')
                
                self.database.log_trade(
                    symbol=symbol,
                    action='EMERGENCY_SELL',
                    quantity=shares_to_sell,
                    price=0,
                    signal_phase='EMERGENCY',
                    signal_strength=1.0,
                    account_type=account.account_type,
                    order_id=order_id
                )
                
                self.database.update_position(
                    symbol=symbol,
                    shares=-shares_to_sell,
                    cost=0,
                    account_type=account.account_type
                )
                
                self.database.deactivate_stop_strategies(symbol)
                
                self.logger.critical(f"✅ Emergency exit executed: {symbol}")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in emergency liquidation for {symbol}: {e}")
        
        return False
    
    def execute_wyckoff_warning_exit(self, warning: WyckoffWarningSignal, position: Dict) -> bool:
        """Execute exit based on Wyckoff warning"""
        try:
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts 
                           if acc.account_type == position['account_type']), None)
            
            if not account or not self.main_system.account_manager.switch_to_account(account):
                return False
            
            shares_to_sell = position['shares']
            
            self.logger.warning(f"🔴 Wyckoff Warning Exit: {warning.symbol}")
            self.logger.warning(f"   Signal: {warning.signal_type} (Strength: {warning.strength:.2f})")
            self.logger.warning(f"   Context: {warning.context}")
            
            order_result = self.main_system.wb.place_order(
                stock=warning.symbol,
                price=0,
                action='SELL',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_sell,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'WARNING')
                
                self.database.log_trade(
                    symbol=warning.symbol,
                    action='WYCKOFF_WARNING_SELL',
                    quantity=shares_to_sell,
                    price=warning.price,
                    signal_phase=warning.signal_type,
                    signal_strength=warning.strength,
                    account_type=account.account_type,
                    order_id=order_id
                )
                
                self.database.update_position(
                    symbol=warning.symbol,
                    shares=-shares_to_sell,
                    cost=warning.price,
                    account_type=account.account_type
                )
                
                self.database.deactivate_stop_strategies(warning.symbol)
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Error executing Wyckoff warning exit: {e}")
        
        return False
    
    def check_enhanced_profit_scaling(self, symbol: str, position: Dict, dynamic_targets: List[Dict]) -> Optional[Dict]:
        """Check for profit scaling with dynamic targets"""
        try:
            quote_data = self.main_system.wb.get_quote(symbol)
            if not quote_data or 'close' not in quote_data:
                return None
            
            current_price = float(quote_data['close'])
            avg_cost = position['avg_cost']
            shares = position['shares']
            gain_pct = (current_price - avg_cost) / avg_cost
            
            for target in dynamic_targets:
                if gain_pct >= target['gain_pct']:
                    if not self.database.already_scaled_at_level(symbol, target['gain_pct']):
                        shares_to_sell = shares * target['sell_pct']
                        sale_value = shares_to_sell * current_price
                        
                        if sale_value >= 5.0:
                            return {
                                'symbol': symbol,
                                'shares_to_sell': shares_to_sell,
                                'current_price': current_price,
                                'gain_pct': gain_pct,
                                'profit_amount': (current_price - avg_cost) * shares_to_sell,
                                'reason': f"DYNAMIC_PROFIT_{target['gain_pct']*100:.0f}PCT",
                                'description': f"Dynamic target: {target['reasoning']}",
                                'remaining_shares': shares - shares_to_sell,
                                'account_type': position['account_type'],
                                'scaling_level': f"{target['gain_pct']*100:.0f}PCT"
                            }
                        break
        
        except Exception as e:
            self.logger.error(f"Error checking enhanced scaling for {symbol}: {e}")
        
        return None
    
    def execute_enhanced_profit_scaling(self, opportunity: Dict) -> bool:
        """Execute profit scaling with enhanced tracking and session management"""
        try:
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts 
                        if acc.account_type == opportunity['account_type']), None)
            
            if not account or not self.main_system.account_manager.switch_to_account(account):
                return False
            
            # Check session before executing
            if not self._ensure_valid_session():
                self.logger.error("❌ Cannot establish valid session for profit scaling")
                return False
            
            symbol = opportunity['symbol']
            shares_to_sell = opportunity['shares_to_sell']
            
            self.logger.info(f"💰 Enhanced Profit Scaling: {shares_to_sell:.5f} shares of {symbol}")
            self.logger.info(f"   {opportunity['description']} (${opportunity['profit_amount']:.2f} profit)")
            
            order_result = self.main_system.wb.place_order(
                stock=symbol,
                price=0,
                action='SELL',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_sell,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'SCALING')
                
                # Enhanced tracking
                self.database.log_trade(
                    symbol=symbol,
                    action='PROFIT_SCALING',
                    quantity=shares_to_sell,
                    price=opportunity['current_price'],
                    signal_phase='PROFIT_SCALING',
                    signal_strength=opportunity['gain_pct'],
                    account_type=opportunity['account_type'],
                    order_id=order_id
                )
                
                self.database.log_partial_sale(
                    symbol=symbol,
                    shares_sold=shares_to_sell,
                    sale_price=opportunity['current_price'],
                    sale_reason=opportunity['reason'],
                    remaining_shares=opportunity['remaining_shares'],
                    gain_pct=opportunity['gain_pct'],
                    profit_amount=opportunity['profit_amount'],
                    scaling_level=opportunity['scaling_level']
                )
                
                self.database.update_position(
                    symbol=symbol,
                    shares=-shares_to_sell,
                    cost=opportunity['current_price'],
                    account_type=opportunity['account_type']
                )
                
                self.logger.info(f"✅ Enhanced profit scaling executed: {symbol}")
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Profit scaling failed for {symbol}: {error_msg}")
                
                # Check if it's a session issue
                if 'session' in error_msg.lower() or 'expired' in error_msg.lower():
                    self.logger.warning("⚠️ Session issue detected during profit scaling")
                    self.main_system.session_manager.clear_session()
                
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error executing enhanced profit scaling: {e}")
            return False
    
    def run_periodic_reconciliation(self) -> bool:
        """Run position reconciliation periodically"""
        try:
            if (self.last_reconciliation is None or 
                (datetime.now() - self.last_reconciliation).total_seconds() > 3600):
                
                self.logger.info("🔄 Running periodic position reconciliation...")
                
                reconciliation = self.comprehensive_exit_manager.reconcile_positions(
                    self.main_system.wb, self.main_system.account_manager
                )
                
                if reconciliation['discrepancies_found']:
                    self.logger.warning(f"⚠️ Found {len(reconciliation['discrepancies_found'])} discrepancies")
                    for disc in reconciliation['discrepancies_found']:
                        self.logger.warning(f"   {disc['symbol']}: Real={disc['real_shares']:.3f}, Bot={disc['bot_shares']:.3f}")
                
                self.last_reconciliation = datetime.now()
                return True
        
        except Exception as e:
            self.logger.error(f"❌ Error in reconciliation: {e}")
        
        return False
    
    def run_enhanced_trading_cycle(self) -> Tuple[int, int, int, int]:
        """Enhanced trading cycle with comprehensive exit management"""
        trades_executed = 0
        wyckoff_sells = 0
        profit_scales = 0
        emergency_exits = 0
        
        try:
            # Step 1: Update configuration
            config = self.position_manager.update_config(self.main_system.account_manager)
            
            # Step 2: Get current positions
            current_positions = self.get_current_positions()
            
            # Step 3: Run comprehensive exit analysis
            self.logger.info("🔍 Running comprehensive exit analysis...")
            exit_analysis = self.comprehensive_exit_manager.run_comprehensive_analysis(
                self.main_system.wb, self.main_system.account_manager, current_positions
            )
            
            # Step 4: Handle critical immediate actions FIRST
            immediate_actions = exit_analysis['immediate_actions_required']
            
            for action in immediate_actions:
                if action['urgency'] == 'CRITICAL':
                    self.logger.warning(f"🚨 CRITICAL: {action['action']} for {action['symbol']} - {action['reason']}")
                    
                    if self.execute_emergency_exit(action, current_positions):
                        emergency_exits += 1
                        # Remove from current_positions to prevent further processing
                        positions_to_remove = [k for k, v in current_positions.items() 
                                             if v['symbol'] == action['symbol']]
                        for k in positions_to_remove:
                            del current_positions[k]
            
            # Step 5: Handle Wyckoff warning signals
            wyckoff_warnings = exit_analysis['wyckoff_warnings']
            
            for symbol, warnings in wyckoff_warnings.items():
                # Find positions for this symbol
                symbol_positions = [v for v in current_positions.values() if v['symbol'] == symbol]
                
                for position in symbol_positions:
                    for warning in warnings:
                        if warning.urgency in ['HIGH', 'CRITICAL']:
                            if self.execute_wyckoff_warning_exit(warning, position):
                                wyckoff_sells += 1
                                # Remove from current_positions
                                position_key = f"{symbol}_{position['account_type']}"
                                if position_key in current_positions:
                                    del current_positions[position_key]
                                break
            
            # Step 6: Enhanced profit scaling
            dynamic_targets = exit_analysis['dynamic_profit_targets']
            
            for symbol, position in current_positions.items():
                if position['symbol'] in dynamic_targets:
                    scaling_opportunity = self.check_enhanced_profit_scaling(
                        position['symbol'], position, dynamic_targets[position['symbol']]
                    )
                    
                    if scaling_opportunity:
                        if self.execute_enhanced_profit_scaling(scaling_opportunity):
                            profit_scales += 1
            
            # Step 7: Emergency mode check
            portfolio_risk = exit_analysis['portfolio_risk_assessment']
            
            if (portfolio_risk['portfolio_drawdown_pct'] > 0.12 or 
                portfolio_risk['market_condition'] in ['MARKET_CRASH', 'HIGH_VOLATILITY']):
                
                self.emergency_mode = True
                self.logger.warning("🚨 EMERGENCY MODE: Skipping new purchases")
                
                # Log enhanced run data
                self.database.log_bot_run(
                    signals_found=0,
                    trades_executed=trades_executed,
                    wyckoff_sells=wyckoff_sells,
                    profit_scales=profit_scales,
                    emergency_exits=emergency_exits,
                    errors=0,
                    portfolio_value=sum(acc.net_liquidation for acc in self.main_system.account_manager.get_enabled_accounts()),
                    available_cash=sum(acc.settled_funds for acc in self.main_system.account_manager.get_enabled_accounts()),
                    emergency_mode=True,
                    market_condition=portfolio_risk['market_condition'],
                    portfolio_drawdown_pct=portfolio_risk['portfolio_drawdown_pct'],
                    status="EMERGENCY_MODE",
                    log_details=f"Market: {portfolio_risk['market_condition']}, Drawdown: {portfolio_risk['portfolio_drawdown_pct']:.1%}"
                )
                
                return trades_executed, wyckoff_sells, profit_scales, emergency_exits
            else:
                self.emergency_mode = False
            
            # Step 8: Normal buy logic (only if NOT in emergency mode)
            if not self.emergency_mode:
                self.logger.info("🔍 Scanning for Wyckoff buy signals...")
                signals = self.wyckoff_strategy.scan_market()
                
                if signals:
                    buy_signals = [s for s in signals if (
                        s.phase in self.buy_phases and 
                        s.strength >= self.min_signal_strength and
                        s.volume_confirmation
                    )]
                    
                    if buy_signals and len(current_positions) < config['max_positions']:
                        enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
                        
                        for signal in buy_signals[:config['max_positions'] - len(current_positions)]:
                            best_account = max(enabled_accounts, key=lambda x: x.settled_funds)
                            position_size = self.position_manager.get_position_size_for_signal(signal)
                            
                            if best_account.settled_funds >= position_size + config['min_balance_preserve']:
                                if self.execute_buy_order(signal, best_account, position_size):
                                    trades_executed += 1
                                    best_account.settled_funds -= position_size
            
            return trades_executed, wyckoff_sells, profit_scales, emergency_exits
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced trading cycle: {e}")
            return trades_executed, wyckoff_sells, profit_scales, emergency_exits
    
    def run(self) -> bool:
        """Main execution with enhanced capabilities"""
        try:
            self.logger.info("🚀 Starting Enhanced Fractional Trading Bot")
            
            if not self.initialize_systems():
                return False
            
            if not self.main_system.run():
                return False
            
            # Initial reconciliation
            self.run_periodic_reconciliation()
            
            # Run enhanced trading cycle
            trades, wyckoff_sells, profit_scales, emergency_exits = self.run_enhanced_trading_cycle()
            
            # Enhanced summary
            total_actions = trades + wyckoff_sells + profit_scales + emergency_exits
            
            self.logger.info("📊 ENHANCED FRACTIONAL TRADING SESSION SUMMARY")
            self.logger.info(f"   Buy Orders: {trades}")
            self.logger.info(f"   Wyckoff Sells: {wyckoff_sells}")
            self.logger.info(f"   Profit Scaling: {profit_scales}")
            self.logger.info(f"   Emergency Exits: {emergency_exits}")
            self.logger.info(f"   Total Actions: {total_actions}")
            self.logger.info(f"   Emergency Mode: {'YES' if self.emergency_mode else 'NO'}")
            
            # Enhanced bot run logging
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            self.database.log_bot_run(
                signals_found=trades,
                trades_executed=total_actions,
                wyckoff_sells=wyckoff_sells,
                profit_scales=profit_scales,
                emergency_exits=emergency_exits,
                errors=0,
                portfolio_value=sum(acc.net_liquidation for acc in enabled_accounts),
                available_cash=sum(acc.settled_funds for acc in enabled_accounts),
                emergency_mode=self.emergency_mode,
                market_condition='UNKNOWN',  # Would get from risk assessment
                portfolio_drawdown_pct=0.0,  # Would calculate
                status="COMPLETED_ENHANCED",
                log_details=f"Actions: Buy={trades}, Wyckoff={wyckoff_sells}, Profit={profit_scales}, Emergency={emergency_exits}"
            )
            
            if total_actions > 0:
                self.logger.info("✅ Enhanced fractional bot completed with actions")
            else:
                self.logger.info("✅ Enhanced fractional bot completed (no actions needed)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
            return False
        
        finally:
            if self.main_system:
                self.main_system.cleanup()


def run_manual_analysis():
    """Standalone analysis without trading"""
    
    logger = logging.getLogger(__name__)
    
    try:
        main_system = MainSystem()
        if not main_system.run():
            print("❌ Failed to initialize main system")
            return
        
        database = EnhancedTradingDatabase()
        exit_manager = ComprehensiveExitManager(database, logger)
        
        current_positions = {}
        enabled_accounts = main_system.account_manager.get_enabled_accounts()
        
        for account in enabled_accounts:
            for position in account.positions:
                position_key = f"{position['symbol']}_{account.account_type}"
                current_positions[position_key] = {
                    'symbol': position['symbol'],
                    'account_type': account.account_type,
                    'shares': position['quantity'],
                    'avg_cost': position['cost_price'],
                    'entry_phase': 'UNKNOWN',
                    'first_purchase_date': datetime.now().strftime('%Y-%m-%d'),
                    'position_size_pct': 0.1,
                    'time_held_days': 0,
                    'volatility_percentile': 0.5
                }
        
        if not current_positions:
            print("📊 No positions found to analyze")
            return
        
        print(f"📊 Analyzing {len(current_positions)} positions...")
        
        exit_analysis = exit_manager.run_comprehensive_analysis(
            main_system.wb, main_system.account_manager, current_positions
        )
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EXIT ANALYSIS REPORT")
        print("="*80)
        
        # Portfolio risk
        risk = exit_analysis['portfolio_risk_assessment']
        print(f"\n📊 PORTFOLIO RISK:")
        print(f"   Drawdown: {risk['portfolio_drawdown_pct']:.1%}")
        print(f"   Market Condition: {risk['market_condition']}")
        print(f"   Positions at Risk: {len(risk['positions_at_risk'])}")
        print(f"   Emergency Exits Needed: {len(risk['emergency_exits_needed'])}")
        
        # Wyckoff warnings
        warnings = exit_analysis['wyckoff_warnings']
        if warnings:
            print(f"\n⚠️ WYCKOFF WARNINGS:")
            for symbol, warning_list in warnings.items():
                for warning in warning_list:
                    print(f"   {symbol}: {warning.signal_type} ({warning.urgency}) - {warning.context}")
        
        # Immediate actions
        actions = exit_analysis['immediate_actions_required']
        if actions:
            print(f"\n🚨 IMMEDIATE ACTIONS REQUIRED:")
            for action in actions:
                print(f"   {action['urgency']}: {action['action']} {action['symbol']} - {action['reason']}")
        
        # Reconciliation
        recon = exit_analysis['reconciliation_report']
        if recon['discrepancies_found']:
            print(f"\n🔄 POSITION DISCREPANCIES:")
            for disc in recon['discrepancies_found']:
                print(f"   {disc['symbol']} ({disc['account_type']}): Real={disc['real_shares']:.3f}, Bot={disc['bot_shares']:.3f}")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error in manual analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'main_system' in locals():
            main_system.cleanup()


def main():
    """Main entry point"""
    print("🚀 Enhanced Fractional Trading Bot Starting...")
    
    bot = EnhancedFractionalTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Enhanced fractional trading bot completed!")
        sys.exit(0)
    else:
        print("❌ Enhanced fractional trading bot failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Uncomment to run manual analysis instead of trading
    # run_manual_analysis()
    main()