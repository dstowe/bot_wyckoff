# database/enhanced_trading_db.py
"""
Enhanced Trading Database Manager
================================
Extracted from fractional_position_system.py - Comprehensive tracking system
This is the single source of truth for all database operations
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import required dataclasses that this database works with
@dataclass
class WyckoffSignal:
    """Represents a Wyckoff accumulation signal"""
    symbol: str
    phase: str
    strength: float
    price: float
    volume_confirmation: bool
    sector: str
    combined_score: float = 0.0

@dataclass
class DayTradeCheckResult:
    """Day trade compliance check result"""
    symbol: str
    action: str
    would_be_day_trade: bool
    db_trades_today: List[Dict]
    actual_trades_today: List[Dict]
    manual_trades_detected: bool
    recommendation: str  # 'ALLOW', 'BLOCK', 'EMERGENCY_OVERRIDE'
    details: str


class EnhancedTradingDatabase:
    """
    Enhanced database manager with comprehensive tracking
    
    This class provides all database operations for the trading system including:
    - Signal logging
    - Trade execution tracking  
    - Position management
    - Day trade compliance tracking
    - Bot run statistics
    - Portfolio analytics
    """
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        """
        Initialize the enhanced trading database
        
        Args:
            db_path: Path to the SQLite database file
        """
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
                    day_trade_check TEXT,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    trade_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced positions table with multi-account support
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
            
            # Day trade tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS day_trade_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    db_day_trade BOOLEAN NOT NULL,
                    actual_day_trade BOOLEAN NOT NULL,
                    manual_trades_detected BOOLEAN NOT NULL,
                    recommendation TEXT NOT NULL,
                    details TEXT,
                    emergency_override BOOLEAN DEFAULT FALSE,
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
                    day_trades_blocked INTEGER DEFAULT 0,
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
            
            # Add indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_date_symbol ON trades(date, symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stop_strategies_symbol ON stop_strategies(symbol, is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON positions(bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_enhanced_bot_id ON positions_enhanced(bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_partial_sales_symbol ON partial_sales(symbol, sale_date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_day_trade_checks_date ON day_trade_checks(check_date, symbol)')
    
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
                  order_id: str = None, day_trade_check: str = None):
        """Log a trade execution with day trade check info"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trades (date, symbol, action, quantity, price, total_value, 
                                  signal_phase, signal_strength, account_type, order_id, 
                                  day_trade_check, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                day_trade_check,
                self.bot_id
            ))
    
    def update_position(self, symbol: str, shares: float, cost: float, 
                       account_type: str, entry_phase: str = None, entry_strength: float = None):
        """Update position in both regular and enhanced tables"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            self._update_positions_table_fixed(conn, symbol, shares, cost, account_type, entry_phase, entry_strength, today)
            self._update_positions_enhanced_table_fixed(conn, symbol, shares, cost, account_type, entry_phase, entry_strength, today)
    
    def _update_positions_table_fixed(self, conn, symbol: str, shares: float, cost: float, 
                                     account_type: str, entry_phase: str, entry_strength: float, today: str):
        """Update the main positions table with proper account type handling"""
        # Get existing position
        existing = conn.execute(
            '''SELECT total_shares, avg_cost, total_invested, first_purchase_date, entry_phase, entry_strength 
               FROM positions 
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
                # Position closed
                new_invested = 0
                new_avg_cost = 0
                use_phase = old_phase
                use_strength = old_strength
            
            # Update existing position
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
    
    def _update_positions_enhanced_table_fixed(self, conn, symbol: str, shares: float, cost: float, 
                                             account_type: str, entry_phase: str, entry_strength: float, today: str):
        """Enhanced positions table update with proper synchronization"""
        
        # Get current enhanced position if it exists
        existing_enhanced = conn.execute(
            '''SELECT total_shares, avg_cost, total_invested, first_purchase_date, 
                    entry_phase, entry_strength, position_size_pct, time_held_days, volatility_percentile 
            FROM positions_enhanced 
            WHERE symbol = ? AND account_type = ? AND bot_id = ?''',
            (symbol, account_type, self.bot_id)
        ).fetchone()
        
        if existing_enhanced:
            old_shares, old_avg_cost, old_invested, first_date, old_phase, old_strength, old_size_pct, old_days, old_vol = existing_enhanced
            new_shares = old_shares + shares
            
            # Calculate time held
            try:
                first_purchase_dt = datetime.strptime(first_date, '%Y-%m-%d')
                time_held_days = (datetime.now() - first_purchase_dt).days
            except:
                time_held_days = old_days or 0
            
            if new_shares > 0:
                new_invested = old_invested + (shares * cost)
                new_avg_cost = new_invested / new_shares
                use_phase = entry_phase or old_phase or 'UNKNOWN'
                use_strength = entry_strength or old_strength or 0.0
            else:
                # Position closed
                new_invested = 0
                new_avg_cost = 0
                use_phase = old_phase
                use_strength = old_strength
            
            # Update enhanced position
            conn.execute('''
                UPDATE positions_enhanced 
                SET total_shares = ?, avg_cost = ?, total_invested = ?, 
                    last_purchase_date = ?, entry_phase = ?, entry_strength = ?,
                    position_size_pct = ?, time_held_days = ?, volatility_percentile = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND account_type = ? AND bot_id = ?
            ''', (new_shares, new_avg_cost, new_invested, today, use_phase, use_strength,
                  old_size_pct or 0.1, time_held_days, old_vol or 0.5,
                  symbol, account_type, self.bot_id))
            
        else:
            # Insert new enhanced position - only if we're adding shares
            if shares > 0:
                conn.execute('''
                    INSERT INTO positions_enhanced (symbol, account_type, total_shares, avg_cost, total_invested, 
                                                  first_purchase_date, last_purchase_date, 
                                                  entry_phase, entry_strength, position_size_pct,
                                                  time_held_days, volatility_percentile, bot_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, account_type, shares, cost, shares * cost, today, today,
                      entry_phase or 'UNKNOWN', entry_strength or 0.0, 0.1,
                      0, 0.5, self.bot_id))
    
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
                    WHERE symbol = ? AND bot_id = ?
                ''', (symbol, self.bot_id)).fetchall()
                
                if results:
                    columns = ['symbol', 'account_type', 'total_shares', 'avg_cost', 'total_invested', 
                              'first_purchase_date', 'last_purchase_date', 'entry_phase', 
                              'entry_strength', 'position_size_pct', 'time_held_days',
                              'volatility_percentile', 'bot_id', 'updated_at']
                    return [dict(zip(columns, row)) for row in results]
        
        return None
    
    def get_todays_trades(self, symbol: str = None) -> List[Dict]:
        """Get today's trades from database for specific symbol or all"""
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
                      'day_trade_check', 'bot_id', 'trade_datetime', 'created_at']
            
            return [dict(zip(columns, row)) for row in results]
    
    def would_create_day_trade(self, symbol: str, action: str) -> bool:
        """Check if this trade would create a day trade (DATABASE ONLY)"""
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
    
    def log_bot_run(self, signals_found: int, trades_executed: int, wyckoff_sells: int = 0,
                    profit_scales: int = 0, emergency_exits: int = 0, day_trades_blocked: int = 0,
                    errors: int = 0, portfolio_value: float = 0, available_cash: float = 0, 
                    emergency_mode: bool = False, market_condition: str = "NORMAL", 
                    portfolio_drawdown_pct: float = 0.0, status: str = "COMPLETED", log_details: str = ""):
        """Log enhanced bot run statistics with day trade blocking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO bot_runs (run_date, signals_found, trades_executed, wyckoff_sells,
                                    profit_scales, emergency_exits, day_trades_blocked, errors_encountered, 
                                    total_portfolio_value, available_cash, emergency_mode,
                                    market_condition, portfolio_drawdown_pct, status, log_details, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                signals_found, trades_executed, wyckoff_sells, profit_scales, emergency_exits,
                day_trades_blocked, errors, portfolio_value, available_cash, emergency_mode,
                market_condition, portfolio_drawdown_pct, status, log_details, self.bot_id
            ))
    
    def log_day_trade_check(self, check_result: DayTradeCheckResult, emergency_override: bool = False):
        """Log day trade compliance check"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO day_trade_checks (check_date, symbol, action, db_day_trade, 
                                            actual_day_trade, manual_trades_detected, 
                                            recommendation, details, emergency_override, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                check_result.symbol,
                check_result.action,
                check_result.would_be_day_trade,
                len(check_result.actual_trades_today) > 0,
                check_result.manual_trades_detected,
                check_result.recommendation,
                check_result.details,
                emergency_override,
                self.bot_id
            ))
    
    def log_partial_sale(self, symbol: str, shares_sold: float, sale_price: float, 
                        sale_reason: str, remaining_shares: float, gain_pct: float, 
                        profit_amount: float, scaling_level: str):
        """Log partial sale for profit scaling tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO partial_sales (symbol, sale_date, shares_sold, sale_price, 
                                         sale_reason, remaining_shares, gain_pct, 
                                         profit_amount, scaling_level, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().strftime('%Y-%m-%d'),
                shares_sold,
                sale_price,
                sale_reason,
                remaining_shares,
                gain_pct,
                profit_amount,
                scaling_level,
                self.bot_id
            ))
    
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
                    symbol, account_type, shares, avg_cost, invested, phase, strength, first_date, last_date, size_pct, days_held, volatility = row
                    
                    if account_type not in positions:
                        positions[account_type] = {}
                    
                    positions[account_type][symbol] = {
                        'symbol': symbol,
                        'account_type': account_type,
                        'total_shares': shares,
                        'avg_cost': avg_cost,
                        'total_invested': invested,
                        'entry_phase': phase,
                        'entry_strength': strength,
                        'first_purchase_date': first_date,
                        'last_purchase_date': last_date,
                        'position_size_pct': size_pct,
                        'time_held_days': days_held,
                        'volatility_percentile': volatility
                    }
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return positions
    
    def get_current_portfolio(self) -> Dict:
        """Get current portfolio summary"""
        portfolio = {
            'total_positions': 0,
            'total_invested': 0.0,
            'positions_by_account': {},
            'symbols': []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT account_type, symbol, total_shares, avg_cost, total_invested
                    FROM positions_enhanced 
                    WHERE total_shares > 0 AND bot_id = ?
                    ORDER BY account_type, symbol
                ''', (self.bot_id,)).fetchall()
                
                for row in results:
                    account_type, symbol, shares, avg_cost, invested = row
                    
                    if account_type not in portfolio['positions_by_account']:
                        portfolio['positions_by_account'][account_type] = {}
                    
                    portfolio['positions_by_account'][account_type][symbol] = {
                        'shares': shares,
                        'avg_cost': avg_cost,
                        'invested': invested
                    }
                    
                    portfolio['total_positions'] += 1
                    portfolio['total_invested'] += invested
                    
                    if symbol not in portfolio['symbols']:
                        portfolio['symbols'].append(symbol)
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return portfolio

    def get_positions_summary(self) -> Dict:
        """Get summary of all positions"""
        summary = {
            'total_positions': 0,
            'total_value': 0.0,
            'by_account': {},
            'by_symbol': {}
        }
        
        try:
            positions = self.get_all_positions()
            
            for account_type, account_positions in positions.items():
                summary['by_account'][account_type] = {
                    'count': len(account_positions),
                    'total_invested': sum(pos['total_invested'] for pos in account_positions.values())
                }
                
                for symbol, position in account_positions.items():
                    summary['total_positions'] += 1
                    summary['total_value'] += position['total_invested']
                    
                    if symbol not in summary['by_symbol']:
                        summary['by_symbol'][symbol] = {
                            'total_shares': 0,
                            'total_invested': 0,
                            'accounts': []
                        }
                    
                    summary['by_symbol'][symbol]['total_shares'] += position['total_shares']
                    summary['by_symbol'][symbol]['total_invested'] += position['total_invested']
                    summary['by_symbol'][symbol]['accounts'].append(account_type)
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return summary

    def get_account_positions(self, account_type: str) -> Dict:
        """Get all positions for a specific account"""
        positions = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, total_shares, avg_cost, total_invested, 
                           entry_phase, entry_strength, first_purchase_date, 
                           last_purchase_date, position_size_pct, time_held_days, 
                           volatility_percentile
                    FROM positions_enhanced 
                    WHERE account_type = ? AND total_shares > 0 AND bot_id = ?
                    ORDER BY symbol
                ''', (account_type, self.bot_id)).fetchall()
                
                for row in results:
                    symbol, shares, avg_cost, invested, phase, strength, first_date, last_date, size_pct, days_held, volatility = row
                    
                    positions[symbol] = {
                        'symbol': symbol,
                        'account_type': account_type,
                        'total_shares': shares,
                        'avg_cost': avg_cost,
                        'total_invested': invested,
                        'entry_phase': phase,
                        'entry_strength': strength,
                        'first_purchase_date': first_date,
                        'last_purchase_date': last_date,
                        'position_size_pct': size_pct,
                        'time_held_days': days_held,
                        'volatility_percentile': volatility
                    }
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return positions

    def get_position_count(self) -> int:
        """Get total number of positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute('''
                    SELECT COUNT(*) FROM positions_enhanced 
                    WHERE total_shares > 0 AND bot_id = ?
                ''', (self.bot_id,)).fetchone()
                
                return result[0] if result else 0
        
        except Exception as e:
            return 0

    def has_position(self, symbol: str, account_type: str = None) -> bool:
        """Check if we have a position in a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if account_type:
                    result = conn.execute('''
                        SELECT total_shares FROM positions_enhanced 
                        WHERE symbol = ? AND account_type = ? AND bot_id = ? AND total_shares > 0
                    ''', (symbol, account_type, self.bot_id)).fetchone()
                else:
                    result = conn.execute('''
                        SELECT total_shares FROM positions_enhanced 
                        WHERE symbol = ? AND bot_id = ? AND total_shares > 0
                    ''', (symbol, self.bot_id)).fetchone()
                
                return result is not None
        
        except Exception as e:
            return False

    def clear_position(self, symbol: str, account_type: str):
        """Clear/close a position (set shares to 0)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE positions_enhanced 
                    SET total_shares = 0, total_invested = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND account_type = ? AND bot_id = ?
                ''', (symbol, account_type, self.bot_id))
                
                conn.execute('''
                    UPDATE positions 
                    SET total_shares = 0, total_invested = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ? AND account_type = ? AND bot_id = ?
                ''', (symbol, account_type, self.bot_id))
                
        except Exception as e:
            # Log error but don't crash
            pass

    def get_symbols_list(self) -> List[str]:
        """Get list of all symbols we have positions in"""
        symbols = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = conn.execute('''
                    SELECT DISTINCT symbol FROM positions_enhanced 
                    WHERE total_shares > 0 AND bot_id = ?
                    ORDER BY symbol
                ''', (self.bot_id,)).fetchall()
                
                symbols = [row[0] for row in results]
        
        except Exception as e:
            # Log error but don't crash
            pass
        
        return symbols