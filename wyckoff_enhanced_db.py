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


