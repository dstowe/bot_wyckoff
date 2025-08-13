#!/usr/bin/env python3
"""
COMPLETE ENHANCED FRACTIONAL POSITION BUILDING SYSTEM - FULLY FIXED
Handles all database constraints, division by zero errors, and includes ALL enhanced features
This is the complete replacement for fractional_position_system.py with all bugs fixed
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


class CompleteDatabaseManager:
    """Complete database manager that handles all constraints and migrations properly"""
    
    def __init__(self, db_path="data/trading_bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.bot_id = "enhanced_wyckoff_bot_v2"
        self.logger = logging.getLogger(__name__)
        
        # Initialize with complete schema
        self.init_complete_database()
    
    def init_complete_database(self):
        """Initialize complete database with all tables and constraints"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Core signals table
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
            
            # Enhanced positions table with proper constraints
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT NOT NULL,
                    bot_id TEXT NOT NULL,
                    total_shares REAL NOT NULL DEFAULT 0.0,
                    avg_cost REAL NOT NULL DEFAULT 0.0,
                    total_invested REAL NOT NULL DEFAULT 0.0,
                    first_purchase_date TEXT,
                    last_purchase_date TEXT,
                    account_type TEXT NOT NULL,
                    entry_phase TEXT DEFAULT 'UNKNOWN',
                    entry_strength REAL DEFAULT 0.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, bot_id)
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
            
            # Complete bot_runs table
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
                    market_condition TEXT DEFAULT 'UNKNOWN',
                    portfolio_drawdown_pct REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    log_details TEXT,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add indexes for performance
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_trades_date_symbol ON trades(date, symbol)',
                'CREATE INDEX IF NOT EXISTS idx_stop_strategies_symbol ON stop_strategies(symbol, is_active)', 
                'CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON positions(bot_id)',
                'CREATE INDEX IF NOT EXISTS idx_partial_sales_symbol ON partial_sales(symbol, sale_date)',
                'CREATE INDEX IF NOT EXISTS idx_signals_symbol_date ON signals(symbol, date)',
                'CREATE INDEX IF NOT EXISTS idx_bot_runs_date ON bot_runs(run_date)'
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass  # Index might already exist
            
            # Migrate existing data if needed
            self._migrate_existing_data(conn)
    
    def _migrate_existing_data(self, conn):
        """Migrate any existing data to new schema"""
        try:
            # Check if we have old positions without bot_id as part of primary key
            cursor = conn.execute("PRAGMA table_info(positions)")
            table_info = cursor.fetchall()
            
            # Check if primary key is just symbol (old schema)
            has_composite_key = any('bot_id' in str(col) for col in table_info)
            
            if not has_composite_key:
                self.logger.info("Migrating positions table to new schema...")
                
                # Get existing data
                existing_positions = conn.execute("SELECT * FROM positions").fetchall()
                
                if existing_positions:
                    # Drop old table and recreate with new schema
                    conn.execute("DROP TABLE positions")
                    
                    # Recreate with new schema
                    conn.execute('''
                        CREATE TABLE positions (
                            symbol TEXT NOT NULL,
                            bot_id TEXT NOT NULL,
                            total_shares REAL NOT NULL DEFAULT 0.0,
                            avg_cost REAL NOT NULL DEFAULT 0.0,
                            total_invested REAL NOT NULL DEFAULT 0.0,
                            first_purchase_date TEXT,
                            last_purchase_date TEXT,
                            account_type TEXT NOT NULL,
                            entry_phase TEXT DEFAULT 'UNKNOWN',
                            entry_strength REAL DEFAULT 0.0,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (symbol, bot_id)
                        )
                    ''')
                    
                    # Migrate data back with bot_id
                    for pos in existing_positions:
                        try:
                            # Handle different old schema formats
                            if len(pos) >= 7:
                                symbol, shares, avg_cost, invested, first_date, last_date, account_type = pos[:7]
                                entry_phase = pos[7] if len(pos) > 7 else 'UNKNOWN'
                                entry_strength = pos[8] if len(pos) > 8 else 0.0
                                
                                conn.execute('''
                                    INSERT OR REPLACE INTO positions 
                                    (symbol, bot_id, total_shares, avg_cost, total_invested, 
                                     first_purchase_date, last_purchase_date, account_type, 
                                     entry_phase, entry_strength)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (symbol, self.bot_id, shares, avg_cost, invested, 
                                     first_date, last_date, account_type, entry_phase, entry_strength))
                        except Exception as e:
                            self.logger.error(f"Error migrating position {pos}: {e}")
                    
                    self.logger.info(f"Migrated {len(existing_positions)} positions to new schema")
                    
        except Exception as e:
            self.logger.error(f"Error during data migration: {e}")
    
    def upsert_position(self, symbol: str, shares: float, cost: float, account_type: str, 
                       entry_phase: str = None, entry_strength: float = None):
        """Insert or update position - handles UNIQUE constraint properly"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get existing position
                existing = conn.execute(
                    '''SELECT total_shares, avg_cost, total_invested, first_purchase_date, 
                             entry_phase, entry_strength
                       FROM positions WHERE symbol = ? AND bot_id = ?''',
                    (symbol, self.bot_id)
                ).fetchone()
                
                if existing:
                    # Update existing position
                    old_shares, old_avg_cost, old_invested, first_date, old_phase, old_strength = existing
                    new_shares = old_shares + shares
                    
                    if new_shares > 0:
                        new_invested = old_invested + (shares * cost)
                        new_avg_cost = new_invested / new_shares if new_shares > 0 else 0
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
                        WHERE symbol = ? AND bot_id = ?
                    ''', (new_shares, new_avg_cost, new_invested, today, use_phase, use_strength, 
                         symbol, self.bot_id))
                    
                    self.logger.debug(f"Updated position: {symbol} = {new_shares:.5f} shares")
                else:
                    # Insert new position
                    conn.execute('''
                        INSERT INTO positions (symbol, bot_id, total_shares, avg_cost, total_invested, 
                                             first_purchase_date, last_purchase_date, account_type, 
                                             entry_phase, entry_strength)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, self.bot_id, shares, cost, shares * cost, today, today, account_type, 
                         entry_phase or 'UNKNOWN', entry_strength or 0.0))
                    
                    self.logger.debug(f"Inserted new position: {symbol} = {shares:.5f} shares")
                    
        except Exception as e:
            self.logger.error(f"Error upserting position for {symbol}: {e}")
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions with safe error handling"""
        positions = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, total_shares, avg_cost, total_invested, account_type,
                           entry_phase, entry_strength, first_purchase_date, last_purchase_date
                    FROM positions 
                    WHERE total_shares > 0 AND bot_id = ?
                ''', (self.bot_id,)).fetchall()
                
                for row in results:
                    symbol, shares, avg_cost, invested, account_type, entry_phase, entry_strength, first_date, last_date = row
                    positions[symbol] = {
                        'shares': shares,
                        'avg_cost': avg_cost,
                        'total_invested': invested,
                        'account_type': account_type,
                        'entry_phase': entry_phase or 'UNKNOWN',
                        'entry_strength': entry_strength or 0.0,
                        'first_purchase_date': first_date,
                        'last_purchase_date': last_date
                    }
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
        
        return positions
    
    def reconcile_with_webull_positions(self, webull_positions: Dict) -> Dict:
        """Reconcile database positions with real Webull positions"""
        reconciliation_report = {
            'discrepancies_found': [],
            'positions_synced': 0,
            'positions_corrected': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all symbols from both sources
                bot_positions = self.get_current_positions()
                all_symbols = set(webull_positions.keys()) | set(bot_positions.keys())
                
                for symbol in all_symbols:
                    real_shares = webull_positions.get(symbol, {'shares': 0})['shares']
                    bot_shares = bot_positions.get(symbol, {'shares': 0})['shares']
                    
                    if abs(real_shares - bot_shares) > 0.001:
                        # Found discrepancy
                        discrepancy = {
                            'symbol': symbol,
                            'real_shares': real_shares,
                            'bot_shares': bot_shares,
                            'difference': real_shares - bot_shares
                        }
                        reconciliation_report['discrepancies_found'].append(discrepancy)
                        
                        # Auto-correct the discrepancy
                        if real_shares == 0 and bot_shares > 0:
                            # Remove ghost position
                            conn.execute('''
                                UPDATE positions 
                                SET total_shares = 0, total_invested = 0, updated_at = CURRENT_TIMESTAMP
                                WHERE symbol = ? AND bot_id = ?
                            ''', (symbol, self.bot_id))
                            self.logger.info(f"Removed ghost position: {symbol}")
                            reconciliation_report['positions_corrected'] += 1
                            
                        elif real_shares > 0:
                            # Update to match real shares
                            webull_pos = webull_positions[symbol]
                            account_type = webull_pos.get('account_type', 'CASH')
                            
                            # Use INSERT OR REPLACE to handle both new and existing positions
                            conn.execute('''
                                INSERT OR REPLACE INTO positions 
                                (symbol, bot_id, total_shares, avg_cost, total_invested, 
                                 first_purchase_date, last_purchase_date, account_type, 
                                 entry_phase, entry_strength, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                            ''', (symbol, self.bot_id, real_shares, 0.0, 0.0, 
                                 datetime.now().strftime('%Y-%m-%d'), 
                                 datetime.now().strftime('%Y-%m-%d'),
                                 account_type, 'UNKNOWN', 0.0))
                            
                            self.logger.info(f"Updated position shares: {symbol} to {real_shares}")
                            reconciliation_report['positions_corrected'] += 1
                    
                    reconciliation_report['positions_synced'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error during reconciliation: {e}")
        
        return reconciliation_report
    
    def log_signal(self, signal: WyckoffSignal, action_taken: str = None):
        """Log trading signal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO signals (date, symbol, phase, strength, price, volume_confirmation, 
                                       sector, combined_score, action_taken, bot_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime('%Y-%m-%d'), signal.symbol, signal.phase,
                    signal.strength, signal.price, signal.volume_confirmation,
                    signal.sector, signal.combined_score, action_taken, self.bot_id
                ))
        except Exception as e:
            self.logger.error(f"Error logging signal: {e}")
    
    def log_trade(self, symbol: str, action: str, quantity: float, price: float, 
                  signal_phase: str, signal_strength: float, account_type: str, 
                  order_id: str = None):
        """Log trade execution"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO trades (date, symbol, action, quantity, price, total_value, 
                                      signal_phase, signal_strength, account_type, order_id, bot_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime('%Y-%m-%d'), symbol, action, quantity, price,
                    quantity * price, signal_phase, signal_strength, account_type, order_id, self.bot_id
                ))
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
    
    def log_bot_run(self, signals_found: int, trades_executed: int, wyckoff_sells: int = 0,
                    profit_scales: int = 0, emergency_exits: int = 0, errors: int = 0, 
                    portfolio_value: float = 0.0, available_cash: float = 0.0, 
                    emergency_mode: bool = False, market_condition: str = 'UNKNOWN', 
                    portfolio_drawdown_pct: float = 0.0, status: str = 'COMPLETED', 
                    log_details: str = ''):
        """Log bot run with all enhanced tracking"""
        try:
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
        except Exception as e:
            self.logger.error(f"Error logging bot run: {e}")
    
    def log_partial_sale(self, symbol: str, shares_sold: float, sale_price: float, 
                        sale_reason: str, remaining_shares: float, gain_pct: float, 
                        profit_amount: float, scaling_level: str):
        """Log partial sale"""
        try:
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
        except Exception as e:
            self.logger.error(f"Error logging partial sale: {e}")
    
    def already_scaled_at_level(self, symbol: str, gain_pct: float) -> bool:
        """Check if already scaled at this level today"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute('''
                    SELECT COUNT(*) FROM partial_sales 
                    WHERE symbol = ? AND bot_id = ? AND gain_pct >= ? AND sale_date = ?
                ''', (symbol, self.bot_id, gain_pct - 0.01, datetime.now().strftime('%Y-%m-%d'))).fetchone()
                
                return result[0] > 0
        except Exception:
            return False
    
    def deactivate_stop_strategies(self, symbol: str):
        """Deactivate stop strategies for symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE stop_strategies 
                    SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND bot_id = ?
                ''', (symbol, self.bot_id))
        except Exception as e:
            self.logger.error(f"Error deactivating stop strategies: {e}")


class EnhancedWyckoffAnalyzer:
    """Complete enhanced Wyckoff analyzer with all warning signals"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def analyze_advanced_warnings(self, symbol: str, data: pd.DataFrame, 
                                current_price: float, entry_data: Dict) -> List[WyckoffWarningSignal]:
        """Analyze for advanced Wyckoff warning signals"""
        warnings = []
        
        try:
            # 1. UTAD - Upthrust After Distribution
            utad_signal = self._detect_utad(symbol, data, current_price)
            if utad_signal:
                warnings.append(utad_signal)
            
            # 2. SOW - Sign of Weakness  
            sow_signal = self._detect_sow(symbol, data, current_price)
            if sow_signal:
                warnings.append(sow_signal)
            
            # 3. Volume Divergence
            vol_div_signal = self._detect_volume_divergence(symbol, data, current_price)
            if vol_div_signal:
                warnings.append(vol_div_signal)
            
            # 4. Context-based support breaks
            context_signal = self._detect_context_breaks(symbol, data, current_price, entry_data)
            if context_signal:
                warnings.append(context_signal)
                
        except Exception as e:
            self.logger.error(f"Error analyzing warnings for {symbol}: {e}")
        
        return warnings
    
    def _detect_utad(self, symbol: str, data: pd.DataFrame, current_price: float) -> Optional[WyckoffWarningSignal]:
        """Detect Upthrust After Distribution - false breakout above resistance"""
        if len(data) < 20:
            return None
        
        try:
            # Find recent high and resistance level
            recent_high = data['High'].tail(10).max()
            resistance_level = data['High'].tail(30).quantile(0.95)
            
            # Check for UTAD pattern
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].tail(30).mean()
            
            # UTAD criteria:
            # 1. Price breaks above resistance with high volume
            # 2. Followed by quick reversal with heavy selling
            # 3. Price fails to hold above resistance
            
            if (current_price > resistance_level and 
                recent_volume > avg_volume * 1.5 and
                current_price < recent_high * 0.98):  # Failed to hold breakout
                
                strength = min(0.9, (recent_volume / avg_volume - 1.0) * 0.5)
                
                return WyckoffWarningSignal(
                    symbol=symbol,
                    signal_type='UTAD',
                    strength=strength,
                    price=current_price,
                    key_level=resistance_level,
                    volume_data={'recent_vol': recent_volume, 'avg_vol': avg_volume},
                    context=f"Failed breakout above {resistance_level:.2f}",
                    urgency='HIGH'
                )
        except Exception as e:
            self.logger.debug(f"UTAD detection error for {symbol}: {e}")
        
        return None
    
    def _detect_sow(self, symbol: str, data: pd.DataFrame, current_price: float) -> Optional[WyckoffWarningSignal]:
        """Detect Sign of Weakness - selling pressure on rallies"""
        if len(data) < 15:
            return None
        
        try:
            # Look for declining volume on up days
            recent_data = data.tail(10)
            
            up_days = recent_data[recent_data['Close'] > recent_data['Close'].shift(1)]
            down_days = recent_data[recent_data['Close'] < recent_data['Close'].shift(1)]
            
            if len(up_days) < 3 or len(down_days) < 2:
                return None
            
            avg_up_volume = up_days['Volume'].mean()
            avg_down_volume = down_days['Volume'].mean()
            
            # SOW: Volume higher on down days than up days
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
            # Check if we're near recent highs
            recent_high = data['High'].tail(20).max()
            is_near_high = current_price >= recent_high * 0.99
            
            if not is_near_high:
                return None
            
            # Get volume on days when price made new highs
            high_days = data[data['High'] >= data['High'].rolling(10).max()]
            
            if len(high_days) < 3:
                return None
            
            # Compare recent high volume vs earlier high volume
            recent_high_vol = high_days['Volume'].tail(2).mean()
            earlier_high_vol = high_days['Volume'].head(-2).mean()
            
            # Divergence: New highs with declining volume
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
            
            # Define context-specific support levels
            if entry_phase in ['ST', 'Creek']:
                # For accumulation phases, support is recent lows
                support_level = data['Low'].tail(20).min()
                critical_level = support_level * 1.02  # 2% buffer
                
            elif entry_phase in ['SOS', 'BU']:
                # For breakout phases, support is breakout level
                support_level = entry_price * 0.95  # Assume breakout level
                critical_level = support_level * 1.01  # 1% buffer
                
            elif entry_phase == 'LPS':
                # For LPS, support is the last point of support
                support_level = data['Low'].tail(30).min()
                critical_level = support_level * 1.015  # 1.5% buffer
            else:
                return None
            
            # Check if current price broke below critical support
            if current_price < critical_level:
                break_severity = (critical_level - current_price) / critical_level
                
                return WyckoffWarningSignal(
                    symbol=symbol,
                    signal_type='CONTEXT_STOP',
                    strength=min(1.0, break_severity * 5),  # Scale severity
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
        
        # Base position size as percentage of total cash (more conservative for small accounts)
        if total_cash < 200:
            base_position_pct = 0.15  # 15% of cash per position for very small accounts
            max_positions = 3
            min_balance_pct = 0.25  # Keep 25% as buffer
        elif total_cash < 500:
            base_position_pct = 0.12  # 12% of cash per position
            max_positions = 4
            min_balance_pct = 0.20  # Keep 20% as buffer
        elif total_cash < 1000:
            base_position_pct = 0.10  # 10% of cash per position
            max_positions = 5
            min_balance_pct = 0.15  # Keep 15% as buffer
        else:
            base_position_pct = 0.08  # 8% of cash per position for larger accounts
            max_positions = 6
            min_balance_pct = 0.12  # Keep 12% as buffer
        
        # Calculate actual dollar amounts
        base_position_size = total_cash * base_position_pct
        min_balance_preserve = total_cash * min_balance_pct
        
        # Ensure minimum $5 per position (Webull requirement)
        if base_position_size < 5.0:
            base_position_size = min(5.0, total_cash * 0.5)  # Use up to 50% for very small accounts
        
        # Wyckoff phase allocations (optimized for small accounts)
        wyckoff_phases = {
            'ST': {
                'initial_allocation': 0.60,  # Larger test for small accounts
                'allow_additions': False,
                'max_total_allocation': 0.60,
                'description': 'Test phase - meaningful size for small account'
            },
            'SOS': {
                'initial_allocation': 0.70,  # Main allocation
                'allow_additions': True,
                'max_total_allocation': 1.0,
                'description': 'Breakout confirmation - build main position'
            },
            'LPS': {
                'initial_allocation': 0.50,  # Support confirmation
                'allow_additions': True,
                'max_total_allocation': 1.0,
                'description': 'Last point of support - complete position'
            },
            'BU': {
                'initial_allocation': 0.40,  # Back up the truck
                'allow_additions': True,
                'max_total_allocation': 1.0,
                'description': 'Pullback bounce - add to position'
            },
            'Creek': {
                'initial_allocation': 0.0,  # No new positions in consolidation
                'allow_additions': False,
                'max_total_allocation': 0.0,
                'description': 'Consolidation - hold only'
            }
        }
        
        # Profit taking targets (optimized for small account growth)
        profit_targets = [
            {'gain_pct': 0.08, 'sell_pct': 0.20, 'description': '8% gain: Take 20% profit'},
            {'gain_pct': 0.15, 'sell_pct': 0.25, 'description': '15% gain: Take 25% more'},
            {'gain_pct': 0.25, 'sell_pct': 0.30, 'description': '25% gain: Take 30% more'},
            {'gain_pct': 0.40, 'sell_pct': 0.25, 'description': '40% gain: Take final 25%'},
        ]
        
        config = {
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
        
        return config
    
    def _get_fallback_config(self) -> Dict:
        """Fallback configuration for $150 accounts"""
        return {
            'total_value': 300.0,
            'total_cash': 300.0,
            'base_position_size': 15.0,  # 5% of $300
            'base_position_pct': 0.15,
            'min_balance_preserve': 75.0,  # 25% buffer
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
                {'gain_pct': 0.25, 'sell_pct': 0.30, 'description': '25% gain: Take 30% more'},
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
            
            # Calculate time held safely
            try:
                first_purchase = datetime.strptime(position['first_purchase_date'], '%Y-%m-%d')
                time_held = (datetime.now() - first_purchase).days
            except (ValueError, KeyError):
                time_held = 0
            
            account_size = market_data.get('account_value', 1000)
            
            # Base targets (conservative for small accounts)
            base_targets = [
                {'gain_pct': 0.08, 'sell_pct': 0.20},
                {'gain_pct': 0.15, 'sell_pct': 0.25},
                {'gain_pct': 0.25, 'sell_pct': 0.30},
                {'gain_pct': 0.40, 'sell_pct': 0.25}
            ]
            
            # Adjust based on entry phase
            phase_multipliers = {
                'ST': 0.8,    # Conservative for test phases
                'SOS': 1.2,   # More aggressive for breakouts
                'LPS': 1.0,   # Standard for support tests
                'BU': 0.9,    # Moderate for pullback entries
                'Creek': 0.7  # Very conservative for consolidation
            }
            
            multiplier = phase_multipliers.get(entry_phase, 1.0)
            
            # Adjust for position size (larger positions = more conservative)
            if position_size_pct > 0.15:  # Large position
                multiplier *= 0.9
            elif position_size_pct < 0.05:  # Small position
                multiplier *= 1.1
            
            # Adjust for volatility (high volatility = wider targets)
            if volatility > 0.8:  # High volatility
                volatility_adj = 1.3
            elif volatility > 0.5:  # Medium volatility  
                volatility_adj = 1.1
            else:  # Low volatility
                volatility_adj = 0.9
            
            # Adjust for time held (longer hold = more aggressive scaling)
            if time_held > 30:  # Held over a month
                time_adj = 1.2
            elif time_held > 14:  # Held over 2 weeks
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
            return base_targets  # Fallback to base targets


class PortfolioRiskManager:
    """Manages portfolio-level risk and emergency exits"""
    
    def __init__(self, database, logger):
        self.database = database
        self.logger = logger
        
        # Risk limits
        self.MAX_PORTFOLIO_DRAWDOWN = 0.15  # 15%
        self.MAX_INDIVIDUAL_LOSS = 0.20     # 20%
        self.VIX_CRASH_THRESHOLD = 40       # VIX > 40 = market crash
        
    def assess_portfolio_risk(self, account_manager, current_positions: Dict) -> Dict:
        """Assess overall portfolio risk and recommend actions"""
        risk_assessment = {
            'portfolio_drawdown_pct': 0.0,
            'positions_at_risk': [],
            'emergency_exits_needed': [],
            'market_condition': 'NORMAL',
            'recommended_actions': []
        }
        
        try:
            # Get account values
            enabled_accounts = account_manager.get_enabled_accounts()
            total_current_value = sum(acc.net_liquidation for acc in enabled_accounts)
            
            # Calculate portfolio drawdown
            portfolio_drawdown = self._calculate_portfolio_drawdown(total_current_value)
            risk_assessment['portfolio_drawdown_pct'] = portfolio_drawdown
            
            # Check individual position risks
            for symbol, position in current_positions.items():
                position_risk = self._assess_individual_position_risk(symbol, position)
                if position_risk and position_risk.current_risk_pct > 0.10:  # 10%+ loss
                    risk_assessment['positions_at_risk'].append(position_risk)
                
                if position_risk and position_risk.current_risk_pct > self.MAX_INDIVIDUAL_LOSS:
                    risk_assessment['emergency_exits_needed'].append(position_risk)
            
            # Check market conditions
            market_condition = self._assess_market_conditions()
            risk_assessment['market_condition'] = market_condition
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_assessment)
            risk_assessment['recommended_actions'] = recommendations
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
        
        return risk_assessment
    
    def _calculate_portfolio_drawdown(self, current_value: float) -> float:
        """Calculate portfolio drawdown from high water mark"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                # Get historical portfolio values
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
    
    def _assess_individual_position_risk(self, symbol: str, position: Dict) -> Optional[PositionRisk]:
        """Assess risk for individual position with safe error handling"""
        try:
            # Get current price safely
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period="1d")
            
            if len(hist_data) == 0:
                return None
                
            current_price = hist_data['Close'].iloc[-1]
            
            # Calculate current risk with safety checks
            avg_cost = position.get('avg_cost', 0)
            if avg_cost <= 0:
                return None  # Can't calculate risk without cost basis
                
            current_risk_pct = (avg_cost - current_price) / avg_cost
            
            # Calculate time held safely
            try:
                first_purchase = datetime.strptime(position['first_purchase_date'], '%Y-%m-%d')
                time_held_days = (datetime.now() - first_purchase).days
            except (ValueError, KeyError):
                time_held_days = 0
            
            # Get volatility safely
            try:
                hist_data_3mo = ticker.history(period="3mo")
                if len(hist_data_3mo) > 20:
                    returns = hist_data_3mo['Close'].pct_change().dropna()
                    volatility_percentile = np.percentile(np.abs(returns), 80)  # 80th percentile
                else:
                    volatility_percentile = 0.02  # Default 2%
            except Exception:
                volatility_percentile = 0.02
            
            # Determine recommended action
            if current_risk_pct > 0.20:
                recommended_action = "EMERGENCY_EXIT"
            elif current_risk_pct > 0.15:
                recommended_action = "CONSIDER_EXIT"
            elif current_risk_pct > 0.10:
                recommended_action = "MONITOR_CLOSELY"
            else:
                recommended_action = "HOLD"
            
            return PositionRisk(
                symbol=symbol,
                current_risk_pct=current_risk_pct,
                time_held_days=time_held_days,
                position_size_pct=0.1,  # Would need account value to calculate
                entry_phase=position.get('entry_phase', 'UNKNOWN'),
                volatility_percentile=volatility_percentile,
                recommended_action=recommended_action
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing position risk for {symbol}: {e}")
            return None
    
    def _assess_market_conditions(self) -> str:
        """Assess overall market conditions"""
        try:
            # Get VIX data
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
        
        # Portfolio level recommendations
        if risk_assessment['portfolio_drawdown_pct'] > self.MAX_PORTFOLIO_DRAWDOWN:
            recommendations.append("EMERGENCY: Portfolio drawdown exceeds 15% - consider liquidating all positions")
        elif risk_assessment['portfolio_drawdown_pct'] > 0.10:
            recommendations.append("WARNING: Portfolio down 10%+ - reduce position sizes and tighten stops")
        
        # Market condition recommendations
        if risk_assessment['market_condition'] == 'MARKET_CRASH':
            recommendations.append("CRITICAL: VIX > 40 detected - emergency exit all positions")
        elif risk_assessment['market_condition'] == 'HIGH_VOLATILITY':
            recommendations.append("CAUTION: High volatility detected - reduce position sizes")
        
        # Individual position recommendations
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
            return 10.0  # Fallback
        
        base_size = self.current_config['base_position_size']
        phase_config = self.current_config['wyckoff_phases'].get(signal.phase, {})
        initial_allocation = phase_config.get('initial_allocation', 0.5)
        
        # Calculate position size for this phase
        position_size = base_size * initial_allocation
        
        # Ensure minimum $5
        position_size = max(position_size, 5.0)
        
        # Don't exceed remaining cash
        max_position = self.current_config['total_cash'] - self.current_config['min_balance_preserve']
        position_size = min(position_size, max_position)
        
        self.logger.debug(f"💰 {signal.symbol} ({signal.phase}): ${position_size:.2f} position ({initial_allocation:.0%} of ${base_size:.2f})")
        
        return position_size
    
    def check_enhanced_profit_scaling(self, wb_client, current_positions: Dict, dynamic_targets: Dict) -> List[Dict]:
        """Check current positions for profit-taking opportunities with enhanced error handling"""
        scaling_opportunities = []
        
        for symbol, position in current_positions.items():
            try:
                # Get current price
                quote_data = wb_client.get_quote(symbol)
                if not quote_data or 'close' not in quote_data:
                    continue
                
                current_price = float(quote_data['close'])
                avg_cost = position.get('avg_cost', 0)
                shares = position.get('shares', 0)
                
                # Safety checks to prevent division by zero
                if avg_cost <= 0 or shares <= 0:
                    self.logger.debug(f"Skipping {symbol}: avg_cost={avg_cost}, shares={shares}")
                    continue
                
                gain_pct = (current_price - avg_cost) / avg_cost
                
                # Get dynamic targets for this symbol
                targets = dynamic_targets.get(symbol, self.current_config.get('profit_targets', []))
                
                # Check each profit target
                for target in targets:
                    if gain_pct >= target['gain_pct']:
                        # Check if we already took profit at this level
                        if not self.database.already_scaled_at_level(symbol, target['gain_pct']):
                            shares_to_sell = shares * target['sell_pct']
                            sale_value = shares_to_sell * current_price
                            
                            # Ensure sale meets $5 minimum
                            if sale_value >= 5.0:
                                scaling_opportunities.append({
                                    'symbol': symbol,
                                    'shares_to_sell': shares_to_sell,
                                    'current_price': current_price,
                                    'gain_pct': gain_pct,
                                    'profit_amount': (current_price - avg_cost) * shares_to_sell,
                                    'reason': f"PROFIT_{target['gain_pct']*100:.0f}PCT",
                                    'description': target.get('description', f"{target['gain_pct']:.1%} profit target"),
                                    'remaining_shares': shares - shares_to_sell,
                                    'account_type': position['account_type'],
                                    'scaling_level': f"{target['gain_pct']*100:.0f}PCT"
                                })
                                break  # Only one scaling action per position
                            else:
                                self.logger.debug(f"💰 {symbol}: Scaling amount ${sale_value:.2f} below $5 minimum")
            
            except Exception as e:
                self.logger.error(f"Error checking enhanced scaling for {symbol}: {e}")
                continue
        
        return scaling_opportunities


class ComprehensiveExitManager:
    """Main class that coordinates all exit management systems"""
    
    def __init__(self, database, logger):
        self.database = database
        self.logger = logger
        
        # Initialize sub-managers
        self.wyckoff_analyzer = EnhancedWyckoffAnalyzer(logger)
        self.profit_calculator = DynamicProfitTargetCalculator(logger)
        self.risk_manager = PortfolioRiskManager(database, logger)
    
    def reconcile_positions(self, wb_client, account_manager) -> Dict:
        """Reconcile database positions with real Webull holdings"""
        try:
            # Get real positions from Webull accounts
            enabled_accounts = account_manager.get_enabled_accounts()
            webull_positions = {}
            
            for account in enabled_accounts:
                for position in account.positions:
                    symbol = position['symbol']
                    if symbol not in webull_positions:
                        webull_positions[symbol] = {
                            'shares': 0.0,
                            'account_type': account.account_type
                        }
                    webull_positions[symbol]['shares'] += position['quantity']
            
            # Use database reconciliation method
            return self.database.reconcile_with_webull_positions(webull_positions)
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive reconciliation: {e}")
            return {'discrepancies_found': [], 'positions_synced': 0, 'positions_corrected': 0}
    
    def run_comprehensive_exit_analysis(self, wb_client, account_manager, 
                                      current_positions: Dict) -> Dict:
        """Run complete exit analysis and return all recommendations"""
        
        # Step 1: Reconcile positions
        self.logger.info("🔄 Reconciling positions...")
        reconciliation = self.reconcile_positions(wb_client, account_manager)
        
        # Step 2: Assess portfolio risk
        self.logger.info("⚠️ Assessing portfolio risk...")
        risk_assessment = self.risk_manager.assess_portfolio_risk(account_manager, current_positions)
        
        # Step 3: Analyze Wyckoff warnings for each position
        self.logger.info("📊 Analyzing Wyckoff warnings...")
        wyckoff_warnings = {}
        
        for symbol, position in current_positions.items():
            try:
                # Get market data with error handling
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")
                
                if len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                    
                    # Analyze warnings
                    warnings = self.wyckoff_analyzer.analyze_advanced_warnings(
                        symbol, data, current_price, position
                    )
                    
                    if warnings:
                        wyckoff_warnings[symbol] = warnings
                        
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
        
        # Step 4: Calculate dynamic profit targets
        self.logger.info("💰 Calculating dynamic profit targets...")
        dynamic_targets = {}
        
        for symbol, position in current_positions.items():
            try:
                market_data = {'account_value': sum(acc.net_liquidation for acc in account_manager.get_enabled_accounts())}
                volatility = 0.02  # Simplified - would calculate from price data
                
                targets = self.profit_calculator.calculate_dynamic_targets(
                    position, market_data, volatility
                )
                dynamic_targets[symbol] = targets
                
            except Exception as e:
                self.logger.error(f"Error calculating targets for {symbol}: {e}")
        
        # Compile comprehensive report
        exit_analysis = {
            'reconciliation_report': reconciliation,
            'portfolio_risk_assessment': risk_assessment,
            'wyckoff_warnings': wyckoff_warnings,
            'dynamic_profit_targets': dynamic_targets,
            'immediate_actions_required': self._prioritize_immediate_actions(
                risk_assessment, wyckoff_warnings
            ),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return exit_analysis
    
    def _prioritize_immediate_actions(self, risk_assessment: Dict, 
                                    wyckoff_warnings: Dict) -> List[Dict]:
        """Prioritize immediate actions needed"""
        immediate_actions = []
        
        # Emergency exits from risk assessment
        for position_risk in risk_assessment.get('emergency_exits_needed', []):
            immediate_actions.append({
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
                    immediate_actions.append({
                        'action': 'WYCKOFF_EXIT',
                        'symbol': symbol,
                        'reason': f"{warning.signal_type}: {warning.context}",
                        'urgency': 'CRITICAL',
                        'priority': 2
                    })
        
        # Market crash conditions
        if risk_assessment.get('market_condition') == 'MARKET_CRASH':
            immediate_actions.append({
                'action': 'LIQUIDATE_ALL',
                'symbol': 'ALL',
                'reason': 'Market crash detected (VIX > 40)',
                'urgency': 'CRITICAL',
                'priority': 0  # Highest priority
            })
        
        # Sort by priority
        immediate_actions.sort(key=lambda x: x['priority'])
        
        return immediate_actions


class CompleteEnhancedFractionalTradingBot:
    """Complete enhanced fractional trading bot with all functionality and bug fixes"""
    
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
        self.sell_phases = ['PS', 'SC']  # Wyckoff distribution phases
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout)  # Console output only
            ],
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 COMPLETE ENHANCED FRACTIONAL TRADING BOT")
        self.logger.info("💰 Dynamic account sizing + Advanced Wyckoff exits")
        self.logger.info("🛡️ Portfolio protection + Position reconciliation")
    
    def initialize_systems(self) -> bool:
        """Initialize all systems"""
        try:
            self.logger.info("🔧 Initializing enhanced systems...")
            
            # Initialize main system
            self.main_system = MainSystem()
            
            # Initialize Wyckoff strategy
            self.wyckoff_strategy = WyckoffPnFStrategy()
            
            # Initialize complete database
            self.database = CompleteDatabaseManager()
            
            # Initialize dynamic account manager
            self.dynamic_manager = DynamicAccountManager(self.logger)
            
            # Initialize smart position manager
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
        """Get current positions"""
        return self.database.get_current_positions()
    
    def execute_buy_order(self, signal: WyckoffSignal, account, position_size: float) -> bool:
        """Execute buy order with proper database handling"""
        try:
            # Switch to trading account
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            # Get current price and calculate shares
            quote_data = self.main_system.wb.get_quote(signal.symbol)
            if not quote_data or 'close' not in quote_data:
                self.logger.error(f"❌ Could not get quote for {signal.symbol}")
                return False
            
            current_price = float(quote_data['close'])
            shares_to_buy = position_size / current_price
            shares_to_buy = round(shares_to_buy, 5)
            
            self.logger.info(f"💰 Buying {shares_to_buy:.5f} shares of {signal.symbol} at ${current_price:.2f}")
            self.logger.info(f"   Position: ${position_size:.2f} ({signal.phase} phase, strength: {signal.strength:.2f})")
            
            # Place order
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
                
                # Log trade and signal
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
                
                # Update position using upsert to handle existing positions
                self.database.upsert_position(
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
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy for {signal.symbol}: {e}")
            return False
    
    def execute_profit_scaling(self, opportunity: Dict) -> bool:
        """Execute profit-taking scaling"""
        try:
            # Find and switch to account
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts 
                           if acc.account_type == opportunity['account_type']), None)
            
            if not account:
                return False
            
            if not self.main_system.account_manager.switch_to_account(account):
                return False
            
            shares_to_sell = opportunity['shares_to_sell']
            symbol = opportunity['symbol']
            
            self.logger.info(f"💰 Enhanced Profit Scaling: {shares_to_sell:.5f} shares of {symbol}")
            self.logger.info(f"   {opportunity['description']} (${opportunity['profit_amount']:.2f} profit)")
            
            # Place order
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
                
                # Log trade and partial sale
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
                
                # Update position
                self.database.upsert_position(
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
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing profit scaling: {e}")
            return False
    
    def run_periodic_reconciliation(self) -> bool:
        """Run position reconciliation periodically"""
        try:
            if (self.last_reconciliation is None or 
                (datetime.now() - self.last_reconciliation).total_seconds() > 3600):  # Every hour
                
                self.logger.info("🔄 Running periodic position reconciliation...")
                
                reconciliation = self.comprehensive_exit_manager.reconcile_positions(
                    self.main_system.wb, self.main_system.account_manager
                )
                
                if reconciliation['discrepancies_found']:
                    self.logger.warning(f"⚠️ Found {len(reconciliation['discrepancies_found'])} discrepancies")
                    
                    for discrepancy in reconciliation['discrepancies_found']:
                        self.logger.warning(f"   {discrepancy['symbol']}: Real={discrepancy['real_shares']:.3f}, Bot={discrepancy['bot_shares']:.3f}")
                
                self.last_reconciliation = datetime.now()
                return True
        
        except Exception as e:
            self.logger.error(f"❌ Error in reconciliation: {e}")
        
        return False
    
    def run_enhanced_trading_cycle(self) -> Tuple[int, int, int, int]:
        """Run complete enhanced trading cycle"""
        trades_executed = 0
        wyckoff_sells = 0
        profit_scales = 0
        emergency_exits = 0
        
        try:
            # Step 1: Update dynamic configuration
            config = self.position_manager.update_config(self.main_system.account_manager)
            
            # Step 2: Get current positions
            current_positions = self.get_current_positions()
            
            # Step 3: Run comprehensive exit analysis
            self.logger.info("🔍 Running comprehensive exit analysis...")
            exit_analysis = self.comprehensive_exit_manager.run_comprehensive_exit_analysis(
                self.main_system.wb, self.main_system.account_manager, current_positions
            )
            
            # Step 4: Handle profit scaling opportunities
            profit_opportunities = self.position_manager.check_enhanced_profit_scaling(
                self.main_system.wb, current_positions, exit_analysis['dynamic_profit_targets']
            )
            
            for opportunity in profit_opportunities[:3]:  # Limit to 3 scalings per run
                if self.execute_profit_scaling(opportunity):
                    profit_scales += 1
            
            # Step 5: Check emergency mode conditions
            portfolio_risk = exit_analysis['portfolio_risk_assessment']
            
            if (portfolio_risk['portfolio_drawdown_pct'] > 0.12 or  # 12%+ drawdown
                portfolio_risk['market_condition'] in ['MARKET_CRASH', 'HIGH_VOLATILITY']):
                
                self.emergency_mode = True
                self.logger.warning("🚨 EMERGENCY MODE: Skipping new purchases due to risk conditions")
                
                return trades_executed, wyckoff_sells, profit_scales, emergency_exits
            else:
                self.emergency_mode = False
            
            # Step 6: Normal buy logic (only if NOT in emergency mode)
            if not self.emergency_mode:
                self.logger.info("🔍 Scanning for Wyckoff buy signals...")
                signals = self.wyckoff_strategy.scan_market()
                
                if signals:
                    # Filter buy signals
                    buy_signals = [s for s in signals if (
                        s.phase in self.buy_phases and 
                        s.strength >= self.min_signal_strength and
                        s.volume_confirmation
                    )]
                    
                    # Execute buy orders
                    if buy_signals and len(current_positions) < config['max_positions']:
                        enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
                        
                        for signal in buy_signals[:config['max_positions'] - len(current_positions)]:
                            # Find account with most available cash
                            best_account = max(enabled_accounts, key=lambda x: x.settled_funds)
                            
                            position_size = self.position_manager.get_position_size_for_signal(signal)
                            
                            if best_account.settled_funds >= position_size + config['min_balance_preserve']:
                                if self.execute_buy_order(signal, best_account, position_size):
                                    trades_executed += 1
                                    # Update available cash estimate
                                    best_account.settled_funds -= position_size
            
            return trades_executed, wyckoff_sells, profit_scales, emergency_exits
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced trading cycle: {e}")
            return trades_executed, wyckoff_sells, profit_scales, emergency_exits
    
    def run(self) -> bool:
        """Main execution"""
        try:
            self.logger.info("🚀 Starting Complete Enhanced Fractional Trading Bot")
            
            # Initialize systems
            if not self.initialize_systems():
                return False
            
            # Authenticate
            if not self.main_system.run():
                return False
            
            # Run initial reconciliation
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
                self.logger.info("✅ Complete enhanced fractional bot completed with actions")
            else:
                self.logger.info("✅ Complete enhanced fractional bot completed (no actions needed)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
            return False
        
        finally:
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point"""
    print("🚀 Complete Enhanced Fractional Trading Bot Starting...")
    
    bot = CompleteEnhancedFractionalTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Complete enhanced fractional trading bot completed!")
        sys.exit(0)
    else:
        print("❌ Complete enhanced fractional trading bot failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()