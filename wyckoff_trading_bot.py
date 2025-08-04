#!/usr/bin/env python3
"""
Enhanced Wyckoff Automated Trading Bot with Dynamic Position Sizing
Features:
- Phase-based stop losses aligned with Wyckoff methodology
- Dynamic position sizing that scales with available capital
- Multi-account support (Cash + Margin)
- Day trade prevention
- Only manages positions created by this system
"""

import sys
import logging
import traceback
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
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
            
            # NEW: Stop loss strategies table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stop_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,  -- 'STOP_LOSS' or 'TRAILING_STOP'
                    initial_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    stop_percentage REAL NOT NULL,
                    trailing_high REAL,  -- For trailing stops
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
    
    def create_stop_strategy(self, symbol: str, initial_price: float, stop_loss_pct: float = 0.05, 
                           trailing_stop_pct: float = 0.08):
        """Create stop loss and trailing stop strategies for a position"""
        with sqlite3.connect(self.db_path) as conn:
            # Create fixed stop loss
            stop_loss_price = initial_price * (1 - stop_loss_pct)
            conn.execute('''
                INSERT INTO stop_strategies (symbol, strategy_type, initial_price, stop_price, 
                                           stop_percentage, bot_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, 'STOP_LOSS', initial_price, stop_loss_price, stop_loss_pct, self.bot_id))
            
            # Create trailing stop
            trailing_stop_price = initial_price * (1 - trailing_stop_pct)
            conn.execute('''
                INSERT INTO stop_strategies (symbol, strategy_type, initial_price, stop_price, 
                                           stop_percentage, trailing_high, bot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, 'TRAILING_STOP', initial_price, trailing_stop_price, 
                  trailing_stop_pct, initial_price, self.bot_id))
    
    def update_trailing_stops(self, symbol: str, current_price: float):
        """Update trailing stop prices based on current price"""
        with sqlite3.connect(self.db_path) as conn:
            # Get active trailing stop
            result = conn.execute('''
                SELECT id, trailing_high, stop_percentage FROM stop_strategies 
                WHERE symbol = ? AND strategy_type = 'TRAILING_STOP' AND is_active = TRUE AND bot_id = ?
            ''', (symbol, self.bot_id)).fetchone()
            
            if result:
                strategy_id, trailing_high, stop_percentage = result
                
                # Update if price reached new high
                if current_price > trailing_high:
                    new_stop_price = current_price * (1 - stop_percentage)
                    conn.execute('''
                        UPDATE stop_strategies 
                        SET trailing_high = ?, stop_price = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (current_price, new_stop_price, strategy_id))
    
    def get_active_stop_strategies(self, symbol: str = None) -> List[Dict]:
        """Get active stop strategies"""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                query = '''
                    SELECT * FROM stop_strategies 
                    WHERE symbol = ? AND is_active = TRUE AND bot_id = ?
                '''
                results = conn.execute(query, (symbol, self.bot_id)).fetchall()
            else:
                query = '''
                    SELECT * FROM stop_strategies 
                    WHERE is_active = TRUE AND bot_id = ?
                '''
                results = conn.execute(query, (self.bot_id,)).fetchall()
            
            columns = ['id', 'symbol', 'strategy_type', 'initial_price', 'stop_price', 
                      'stop_percentage', 'trailing_high', 'is_active', 'bot_id', 'created_at', 'updated_at']
            
            return [dict(zip(columns, row)) for row in results]
    
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
    """Enhanced trading bot with stop loss management and day trade prevention"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        
        # Dynamic position sizing configuration
        self.base_trade_amount = 5.00  # Starting trade size
        self.max_trade_amount = 50.00  # Maximum trade size
        self.min_account_balance = 50.00  # Minimum balance to keep
        
        # Scaling tiers for position sizing
        self.position_scaling_tiers = [
            {'min_cash': 0, 'trade_amount': 5.00},       # $5 for $0-199
            {'min_cash': 200, 'trade_amount': 10.00},    # $10 for $200-499
            {'min_cash': 500, 'trade_amount': 15.00},    # $15 for $500-999
            {'min_cash': 1000, 'trade_amount': 25.00},   # $25 for $1000-1999
            {'min_cash': 2000, 'trade_amount': 35.00},   # $35 for $2000-4999
            {'min_cash': 5000, 'trade_amount': 50.00},   # $50 for $5000+
        ]
        
        # Phase-based stop loss configuration (Wyckoff method)
        self.phase_stops = {
            'ST': 0.03,      # 3% - tight in accumulation
            'Creek': 0.02,   # 2% - very tight in consolidation  
            'SOS': 0.06,     # 6% - wider after breakout
            'LPS': 0.04,     # 4% - moderate at support test
            'BU': 0.05       # 5% - standard during pullback
        }
        
        # Default stop loss for unknown phases
        self.default_stop_loss = 0.05  # 5% default
        self.trailing_stop_percentage = 0.08  # 8% trailing stop
        
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
        self.logger.info("🤖 ENHANCED WYCKOFF TRADING BOT STARTED")
        self.logger.info(f"📝 Log: {log_filename.name}")
        self.logger.info("🛡️ Using Phase-Based Wyckoff Stop Loss Strategy")
        self.logger.info("💰 Using Dynamic Position Sizing Based on Available Capital")
    
    def initialize_systems(self) -> bool:
        """Initialize all required systems"""
        try:
            self.logger.info("🔧 Initializing trading systems...")
            
            # Initialize main system (handles auth, accounts)
            self.main_system = MainSystem()
            
            # Initialize Wyckoff strategy
            self.wyckoff_strategy = WyckoffPnFStrategy()
            
            # Initialize enhanced database
            self.database = EnhancedTradingDatabase()
            
            # Log dynamic position sizing configuration
            self.logger.info("💰 Dynamic Position Sizing Configuration:")
            for tier in self.position_scaling_tiers:
                if tier == self.position_scaling_tiers[-1]:
                    self.logger.info(f"   ${tier['min_cash']:,}+: ${tier['trade_amount']:.2f} per trade")
                else:
                    next_tier = self.position_scaling_tiers[self.position_scaling_tiers.index(tier) + 1]
                    self.logger.info(f"   ${tier['min_cash']:,}-${next_tier['min_cash']-1:,}: ${tier['trade_amount']:.2f} per trade")
            self.logger.info(f"   Maximum trade size: ${self.max_trade_amount:.2f}")
            
            # Log phase-based stop configuration
            self.logger.info("🎯 Phase-Based Stop Loss Configuration:")
            for phase, percentage in self.phase_stops.items():
                self.logger.info(f"   {phase}: {percentage*100:.1f}%")
            self.logger.info(f"   Default: {self.default_stop_loss*100:.1f}%")
            self.logger.info(f"   Trailing: {self.trailing_stop_percentage*100:.1f}%")
            
            self.logger.info("✅ All systems initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize systems: {e}")
            return False
    
    def get_dynamic_trade_amount(self, total_available_cash: float) -> float:
        """Calculate trade size based on available cash using scaling tiers"""
        # Find the appropriate tier
        trade_amount = self.base_trade_amount
        
        for tier in reversed(self.position_scaling_tiers):  # Start from highest tier
            if total_available_cash >= tier['min_cash']:
                trade_amount = tier['trade_amount']
                break
        
        # Cap at maximum
        trade_amount = min(trade_amount, self.max_trade_amount)
        
        return trade_amount
    
    def get_phase_stop_percentage(self, phase: str) -> float:
        """Get stop loss percentage based on Wyckoff phase"""
        return self.phase_stops.get(phase, self.default_stop_loss)
    
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
    
    def get_trading_account(self):
        """Get the best account for trading"""
        enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
        
        if not enabled_accounts:
            self.logger.error("❌ No enabled accounts found")
            return None
        
        # Prefer cash account for day trading, then margin
        for account in enabled_accounts:
            if account.account_type in ['Cash Account', 'CASH']:
                if account.settled_funds >= self.min_account_balance:
                    return account
        
        # Fallback to any account with sufficient funds
        for account in enabled_accounts:
            if account.settled_funds >= self.min_account_balance:
                return account
        
        self.logger.error("❌ No accounts with sufficient funds")
        return None
    
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
        """Check and execute stop losses for all positions across all accounts"""
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
            
            self.logger.info(f"🛡️ Checking stop losses for {len(positions)} positions across all accounts...")
            
            wb = self.main_system.wb
            
            for symbol, shares in positions:
                try:
                    # Get current price
                    quote_data = wb.get_quote(symbol)
                    if not quote_data or 'close' not in quote_data:
                        self.logger.warning(f"⚠️ Could not get quote for {symbol}")
                        continue
                    
                    current_price = float(quote_data['close'])
                    
                    # Update trailing stops
                    self.database.update_trailing_stops(symbol, current_price)
                    
                    # Get active stop strategies
                    stop_strategies = self.database.get_active_stop_strategies(symbol)
                    
                    should_sell = False
                    sell_reason = ""
                    
                    for strategy in stop_strategies:
                        if strategy['strategy_type'] == 'STOP_LOSS':
                            if current_price <= strategy['stop_price']:
                                should_sell = True
                                sell_reason = f"Stop Loss (${strategy['stop_price']:.2f})"
                                break
                        elif strategy['strategy_type'] == 'TRAILING_STOP':
                            if current_price <= strategy['stop_price']:
                                should_sell = True
                                sell_reason = f"Trailing Stop (${strategy['stop_price']:.2f})"
                                break
                    
                    if should_sell:
                        # Check for day trade
                        if self.check_webull_day_trades(symbol, 'SELL'):
                            self.logger.warning(f"⚠️ Skipping {symbol} stop loss - would create day trade")
                            continue
                        
                        self.logger.info(f"🛑 Executing stop loss for {symbol}: {sell_reason}")
                        
                        if self.execute_stop_loss_sell(symbol, shares, current_price, sell_reason):
                            stop_losses_executed += 1
                            # Deactivate stop strategies
                            self.database.deactivate_stop_strategies(symbol)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error checking stop loss for {symbol}: {e}")
                    continue
            
            return stop_losses_executed
            
        except Exception as e:
            self.logger.error(f"❌ Error in stop loss check: {e}")
            return stop_losses_executed
    
    def execute_stop_loss_sell(self, symbol: str, shares: float, current_price: float, 
                              reason: str, account=None) -> bool:
        """Execute a stop loss sell order"""
        try:
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
            
            # Switch to the trading account
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to {account.account_type} account for {symbol}")
                return False
            
            self.logger.info(f"💸 Stop Loss: Selling {shares} shares of {symbol} at ~${current_price:.2f} from {account.account_type}")
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
                self.logger.info(f"✅ Stop loss order placed: {symbol} - Order ID: {order_id}")
                
                # Log the trade
                self.database.log_trade(
                    symbol=symbol,
                    action='SELL',
                    quantity=shares,
                    price=current_price,
                    signal_phase='STOP_LOSS',
                    signal_strength=1.0,
                    account_type=account.account_type,
                    order_id=order_id
                )
                
                # Update position (set to zero)
                self.database.update_position(
                    symbol=symbol,
                    shares=-shares,  # Negative to reduce position
                    cost=current_price,
                    account_type=account.account_type
                )
                
                # Calculate P&L
                position = self.database.get_position(symbol)
                if position:
                    profit_loss = (current_price - position['avg_cost']) * shares
                    profit_loss_pct = (profit_loss / position['total_invested']) * 100
                    self.logger.info(f"📊 Stop Loss P&L for {symbol}: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Stop loss order failed for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing stop loss for {symbol}: {e}")
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
        """Get current positions from database across all accounts (only this bot's positions)"""
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
        """Execute a buy order with stop loss creation"""
        try:
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
            # Round down to avoid insufficient funds
            shares_to_buy = float(round(shares_to_buy, 5))  # Round to 5 decimal places
            
            if shares_to_buy < 0.00001:  # Minimum share amount
                self.logger.warning(f"⚠️ Share amount too small for {signal.symbol}: {shares_to_buy}")
                return False
            
            self.logger.info(f"💰 Buying {shares_to_buy} shares of {signal.symbol} at ~${current_price:.2f} (${trade_amount:.2f} position)")
            
            # Place market order
            order_result = wb.place_order(
                stock=signal.symbol,
                price=0,  # Market price
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
                
                # Create stop loss strategies with phase-specific percentages
                phase_stop_pct = self.get_phase_stop_percentage(signal.phase)
                self.database.create_stop_strategy(
                    symbol=signal.symbol,
                    initial_price=current_price,
                    stop_loss_pct=phase_stop_pct,
                    trailing_stop_pct=self.trailing_stop_percentage
                )
                
                stop_loss_price = current_price * (1 - phase_stop_pct)
                trailing_stop_price = current_price * (1 - self.trailing_stop_percentage)
                
                self.logger.info(f"🛡️ Wyckoff stop strategies created for {signal.symbol} ({signal.phase}):")
                self.logger.info(f"   Phase-Based Stop Loss: ${stop_loss_price:.2f} (-{phase_stop_pct*100:.1f}%)")
                self.logger.info(f"   Trailing Stop: ${trailing_stop_price:.2f} (-{self.trailing_stop_percentage*100:.1f}%)")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Buy order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy order for {signal.symbol}: {e}")
            return False
    
    def execute_sell_order(self, signal: WyckoffSignal, position: Dict, account=None) -> bool:
        """Execute a sell order for entire position"""
        try:
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
                
                # Update position (set to zero)
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=-shares_to_sell,  # Negative to reduce position
                    cost=current_price,
                    account_type=account.account_type
                )
                
                # Deactivate stop strategies
                self.database.deactivate_stop_strategies(signal.symbol)
                
                # Calculate P&L
                profit_loss = (current_price - position['avg_cost']) * shares_to_sell
                profit_loss_pct = (profit_loss / position['total_invested']) * 100
                
                self.logger.info(f"📊 Wyckoff Sell P&L for {signal.symbol}: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Sell order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing sell order for {signal.symbol}: {e}")
            return False
    
    def run_trading_cycle(self) -> Tuple[int, int]:
        """Run one complete trading cycle with multi-account support"""
        trades_executed = 0
        stop_losses_executed = 0
        errors = 0
        
        try:
            # Get all enabled accounts
            enabled_accounts = self.get_enabled_accounts()
            if not enabled_accounts:
                self.logger.error("❌ No enabled accounts found")
                return (0, 0), 1
            
            self.logger.info(f"💼 Found {len(enabled_accounts)} enabled accounts:")
            for i, account in enumerate(enabled_accounts, 1):
                self.logger.info(f"  {i}. {account.account_type}: ${account.settled_funds:.2f} available")
            
            # STEP 1: Check and execute stop losses first (across all accounts)
            self.logger.info("🛡️ STEP 1: Checking stop losses across all accounts...")
            stop_losses_executed = self.check_stop_losses()
            
            if stop_losses_executed > 0:
                self.logger.info(f"🛑 Executed {stop_losses_executed} stop losses")
            
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
                
                # Check for sell signals first (across all accounts)
                sell_signals = self.filter_sell_signals(all_signals, held_symbols)
                
                for signal in sell_signals[:3]:  # Limit to 3 sells per run
                    if signal.symbol in current_positions:
                        position = current_positions[signal.symbol]
                        
                        if self.execute_sell_order(signal, position):  # Account auto-detected
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
                    
                    self.logger.info(f"💰 Processing {len(buy_signals)} buy signals across multiple accounts...")
                    self.logger.info(f"📊 Dynamic position sizing: ${dynamic_trade_amount:.2f} per trade (based on ${total_available_cash:.2f} total cash)")
                    
                    executed_buys = 0
                    signal_index = 0
                    
                    # Try each account in priority order
                    for account_priority, account in enumerate(enabled_accounts, 1):
                        if signal_index >= len(buy_signals):
                            break
                        
                        # Calculate trades this account can afford with dynamic sizing
                        account_max_trades = int((account.settled_funds - self.min_account_balance) / dynamic_trade_amount)
                        account_max_trades = min(account_max_trades, 3)  # Limit per account
                        
                        if account_max_trades <= 0:
                            self.logger.info(f"💸 {account.account_type}: Insufficient funds (${account.settled_funds:.2f} < ${dynamic_trade_amount:.2f} + ${self.min_account_balance:.2f})")
                            continue
                        
                        self.logger.info(f"💰 {account.account_type}: Can afford {account_max_trades} positions at ${dynamic_trade_amount:.2f} each")
                        
                        account_buys = 0
                        
                        # Execute trades for this account
                        while signal_index < len(buy_signals) and account_buys < account_max_trades:
                            signal = buy_signals[signal_index]
                            signal_index += 1
                            
                            # Check if we already hold this stock (in any account)
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
                            self.logger.info(f"✅ {account.account_type}: Executed {account_buys} buy orders at ${dynamic_trade_amount:.2f} each")
                    
                    self.logger.info(f"📊 Total buy orders executed: {executed_buys} (${executed_buys * dynamic_trade_amount:.2f} invested)")
                    
                    # Log any remaining signals that couldn't be executed
                    remaining_signals = len(buy_signals) - signal_index
                    if remaining_signals > 0:
                        self.logger.info(f"⚠️ {remaining_signals} buy signals couldn't be executed (insufficient funds across all accounts)")
                
            else:
                self.logger.info("📭 No Wyckoff signals found")
            
            return (trades_executed, stop_losses_executed), errors
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
            return (trades_executed, stop_losses_executed), errors + 1
    
    def run(self) -> bool:
        """Main bot execution"""
        start_time = time.time()
        signals_found = 0
        trades_executed = 0
        stop_losses_executed = 0
        errors = 0
        log_details = ""
        
        try:
            self.logger.info("🚀 Starting Enhanced Wyckoff Trading Bot with Dynamic Position Sizing")
            
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
            log_details = f"Execution time: {execution_time:.1f}s, Accounts: {account_summary}"
            
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
            self.logger.info("📊 ENHANCED TRADING SESSION SUMMARY")
            self.logger.info(f"   Wyckoff Trades: {trades_executed}")
            self.logger.info(f"   Stop Losses: {stop_losses_executed}")
            self.logger.info(f"   Total Actions: {trades_executed + stop_losses_executed}")
            self.logger.info(f"   Errors: {errors}")
            self.logger.info(f"   Total Portfolio Value: ${total_portfolio_value:.2f}")
            self.logger.info(f"   Total Available Cash: ${total_available_cash:.2f}")
            self.logger.info(f"   Position Size Used: ${self.get_dynamic_trade_amount(total_available_cash):.2f} per trade")
            self.logger.info(f"   Execution Time: {execution_time:.1f}s")
            self.logger.info(f"   Stop Strategy: Phase-based Wyckoff method")
            self.logger.info(f"   Accounts Used: {len(enabled_accounts)} ({', '.join(acc.account_type for acc in enabled_accounts)})")
            
            if (trades_executed + stop_losses_executed) > 0:
                self.logger.info("✅ Enhanced bot completed successfully with actions")
            elif errors == 0:
                self.logger.info("✅ Enhanced bot completed successfully (no actions needed)")
            else:
                self.logger.warning("⚠️ Enhanced bot completed with errors")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in enhanced bot execution: {e}")
            self.logger.error(traceback.format_exc())
            
            # Log failed run
            self.database.log_bot_run(0, 0, 0, 1, 0, 0, "CRITICAL_ERROR", str(e))
            return False
        
        finally:
            # Cleanup
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point for the enhanced trading bot with dynamic position sizing"""
    print("🤖 Enhanced Wyckoff Trading Bot with Dynamic Position Sizing Starting...")
    
    bot = EnhancedWyckoffTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Enhanced trading bot completed successfully!")
        sys.exit(0)
    else:
        print("❌ Enhanced trading bot failed! Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()