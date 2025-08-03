#!/usr/bin/env python3
"""
Wyckoff Automated Trading Bot
Integrates with existing authentication, account management, and Wyckoff strategy
Designed to run with Windows Task Scheduler
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


class TradingDatabase:
    """Database manager for tracking trades and signals"""
    
    def __init__(self, db_path="data/trading_bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trades table
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Positions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    total_shares REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    total_invested REAL NOT NULL,
                    first_purchase_date TEXT NOT NULL,
                    last_purchase_date TEXT NOT NULL,
                    account_type TEXT NOT NULL,
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
                    errors_encountered INTEGER NOT NULL,
                    total_portfolio_value REAL,
                    available_cash REAL,
                    status TEXT NOT NULL,
                    log_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def log_signal(self, signal: WyckoffSignal, action_taken: str = None):
        """Log a trading signal"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO signals (date, symbol, phase, strength, price, volume_confirmation, 
                                   sector, combined_score, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                signal.symbol,
                signal.phase,
                signal.strength,
                signal.price,
                signal.volume_confirmation,
                signal.sector,
                signal.combined_score,
                action_taken
            ))
    
    def log_trade(self, symbol: str, action: str, quantity: float, price: float, 
                  signal_phase: str, signal_strength: float, account_type: str, 
                  order_id: str = None):
        """Log a trade execution"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trades (date, symbol, action, quantity, price, total_value, 
                                  signal_phase, signal_strength, account_type, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                order_id
            ))
    
    def update_position(self, symbol: str, shares: float, cost: float, account_type: str):
        """Update position tracking"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if position exists
            existing = conn.execute(
                'SELECT total_shares, avg_cost, total_invested, first_purchase_date FROM positions WHERE symbol = ?',
                (symbol,)
            ).fetchone()
            
            if existing:
                # Update existing position
                old_shares, old_avg_cost, old_invested, first_date = existing
                new_shares = old_shares + shares
                new_invested = old_invested + (shares * cost)
                new_avg_cost = new_invested / new_shares if new_shares > 0 else 0
                
                conn.execute('''
                    UPDATE positions 
                    SET total_shares = ?, avg_cost = ?, total_invested = ?, 
                        last_purchase_date = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                ''', (new_shares, new_avg_cost, new_invested, today, symbol))
            else:
                # Create new position
                conn.execute('''
                    INSERT INTO positions (symbol, total_shares, avg_cost, total_invested, 
                                         first_purchase_date, last_purchase_date, account_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, shares, cost, shares * cost, today, today, account_type))
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                'SELECT * FROM positions WHERE symbol = ?', (symbol,)
            ).fetchone()
            
            if result:
                columns = ['symbol', 'total_shares', 'avg_cost', 'total_invested', 
                          'first_purchase_date', 'last_purchase_date', 'account_type', 'updated_at']
                return dict(zip(columns, result))
            return None
    
    def log_bot_run(self, signals_found: int, trades_executed: int, errors: int, 
                    portfolio_value: float, available_cash: float, status: str, log_details: str):
        """Log bot run statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO bot_runs (run_date, signals_found, trades_executed, errors_encountered,
                                    total_portfolio_value, available_cash, status, log_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signals_found,
                trades_executed,
                errors,
                portfolio_value,
                available_cash,
                status,
                log_details
            ))


class WyckoffTradingBot:
    """Main trading bot that integrates all systems"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        self.trade_amount = 5.00  # $5 per trade
        self.min_account_balance = 50.00  # Minimum balance to keep
        
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
        log_filename = logs_dir / f"trading_bot_{timestamp}.log"
        
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
        self.logger.info("🤖 WYCKOFF TRADING BOT STARTED")
        self.logger.info(f"📝 Log: {log_filename.name}")
    
    def initialize_systems(self) -> bool:
        """Initialize all required systems"""
        try:
            self.logger.info("🔧 Initializing trading systems...")
            
            # Initialize main system (handles auth, accounts)
            self.main_system = MainSystem()
            
            # Initialize Wyckoff strategy
            self.wyckoff_strategy = WyckoffPnFStrategy()
            
            # Initialize database
            self.database = TradingDatabase()
            
            self.logger.info("✅ All systems initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize systems: {e}")
            return False
    
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
                        sell_signals.append(signal)
                        self.logger.info(f"🔴 Sell signal: {signal.symbol} ({signal.phase}) - Strength: {signal.strength:.2f}")
        
        return sell_signals
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions from database"""
        positions = {}
        
        with sqlite3.connect(self.database.db_path) as conn:
            results = conn.execute(
                'SELECT symbol, total_shares, avg_cost, total_invested FROM positions WHERE total_shares > 0'
            ).fetchall()
            
            for symbol, shares, avg_cost, invested in results:
                positions[symbol] = {
                    'shares': shares,
                    'avg_cost': avg_cost,
                    'total_invested': invested
                }
        
        return positions
    
    def execute_buy_order(self, signal: WyckoffSignal, account) -> bool:
        """Execute a buy order"""
        try:
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
            shares_to_buy = self.trade_amount / current_price
            # Round down to avoid insufficient funds
            shares_to_buy = float(round(shares_to_buy,5))  # Round to 2 decimal places
            
            if shares_to_buy < 0.00001:  # Minimum share amount
                self.logger.warning(f"⚠️ Share amount too small for {signal.symbol}: {shares_to_buy}")
                return False
            
            self.logger.info(f"💰 Buying {shares_to_buy} shares of {signal.symbol} at ~${current_price:.2f}")
            
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
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Buy order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy order for {signal.symbol}: {e}")
            return False
    
    def execute_sell_order(self, signal: WyckoffSignal, position: Dict, account) -> bool:
        """Execute a sell order for entire position"""
        try:
            # Switch to the trading account
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            shares_to_sell = position['shares']
            
            self.logger.info(f"💸 Selling {shares_to_sell} shares of {signal.symbol}")
            
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
                
                # Calculate P&L
                profit_loss = (current_price - position['avg_cost']) * shares_to_sell
                profit_loss_pct = (profit_loss / position['total_invested']) * 100
                
                self.logger.info(f"📊 P&L for {signal.symbol}: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Sell order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing sell order for {signal.symbol}: {e}")
            return False
    
    def run_trading_cycle(self) -> Tuple[int, int]:
        """Run one complete trading cycle"""
        trades_executed = 0
        errors = 0
        
        try:
            # Get trading account
            account = self.get_trading_account()
            if not account:
                self.logger.error("❌ No suitable trading account found")
                return 0, 1
            
            self.logger.info(f"💼 Using {account.account_type} account with ${account.settled_funds:.2f} available")
            
            # Check if we have enough funds for at least one trade
            if account.settled_funds < (self.trade_amount + self.min_account_balance):
                self.logger.warning(f"⚠️ Insufficient funds for trading. Available: ${account.settled_funds:.2f}")
                return 0, 0
            
            # Scan for signals
            all_signals = self.scan_for_signals()
            if not all_signals:
                self.logger.info("📭 No signals found")
                return 0, 0
            
            # Log all signals
            for signal in all_signals:
                self.database.log_signal(signal)
            
            # Get current positions
            current_positions = self.get_current_positions()
            held_symbols = list(current_positions.keys())
            
            self.logger.info(f"📊 Current positions: {len(held_symbols)} stocks")
            if held_symbols:
                for symbol, pos in current_positions.items():
                    self.logger.info(f"  {symbol}: {pos['shares']:.2f} shares @ ${pos['avg_cost']:.2f}")
            
            # Check for sell signals first
            sell_signals = self.filter_sell_signals(all_signals, held_symbols)
            
            for signal in sell_signals[:3]:  # Limit to 3 sells per run
                if signal.symbol in current_positions:
                    position = current_positions[signal.symbol]
                    
                    if self.execute_sell_order(signal, position, account):
                        trades_executed += 1
                        self.database.log_signal(signal, "SELL_EXECUTED")
                        # Remove from current positions
                        del current_positions[signal.symbol]
                    else:
                        errors += 1
                        self.database.log_signal(signal, "SELL_FAILED")
            
            # Check for buy signals
            buy_signals = self.filter_buy_signals(all_signals)
            
            # Calculate how many trades we can afford
            max_trades = int((account.settled_funds - self.min_account_balance) / self.trade_amount)
            max_trades = min(max_trades, 3)  # Limit to 3 buys per run
            
            self.logger.info(f"💰 Can afford {max_trades} new positions")
            
            executed_buys = 0
            for signal in buy_signals:
                if executed_buys >= max_trades:
                    break
                
                # Check if we already hold this stock
                if signal.symbol in current_positions:
                    # Check if we want to add to the position (optional logic)
                    last_purchase = self.database.get_position(signal.symbol)
                    if last_purchase:
                        # Only add if last purchase was more than 3 days ago
                        last_date = datetime.strptime(last_purchase['last_purchase_date'], '%Y-%m-%d')
                        if (datetime.now() - last_date).days >= 3:
                            if self.execute_buy_order(signal, account):
                                trades_executed += 1
                                executed_buys += 1
                                self.database.log_signal(signal, "BUY_ADD_EXECUTED")
                            else:
                                errors += 1
                                self.database.log_signal(signal, "BUY_ADD_FAILED")
                        else:
                            self.database.log_signal(signal, "BUY_SKIP_RECENT")
                else:
                    # New position
                    if self.execute_buy_order(signal, account):
                        trades_executed += 1
                        executed_buys += 1
                        self.database.log_signal(signal, "BUY_NEW_EXECUTED")
                    else:
                        errors += 1
                        self.database.log_signal(signal, "BUY_NEW_FAILED")
            
            return trades_executed, errors
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
            return trades_executed, errors + 1
    
    def run(self) -> bool:
        """Main bot execution"""
        start_time = time.time()
        signals_found = 0
        trades_executed = 0
        errors = 0
        log_details = ""
        
        try:
            self.logger.info("🚀 Starting Wyckoff Trading Bot")
            
            # Initialize all systems
            if not self.initialize_systems():
                self.database.log_bot_run(0, 0, 1, 0, 0, "INIT_FAILED", "System initialization failed")
                return False
            
            # Authenticate and setup accounts
            if not self.authenticate_and_setup_accounts():
                self.database.log_bot_run(0, 0, 1, 0, 0, "AUTH_FAILED", "Authentication failed")
                return False
            
            # Run trading cycle
            trades_executed, cycle_errors = self.run_trading_cycle()
            errors += cycle_errors
            
            # Get portfolio summary
            account = self.get_trading_account()
            portfolio_value = account.net_liquidation if account else 0
            available_cash = account.settled_funds if account else 0
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create summary
            log_details = f"Execution time: {execution_time:.1f}s, Account: {account.account_type if account else 'None'}"
            
            status = "SUCCESS" if errors == 0 else "SUCCESS_WITH_ERRORS" if trades_executed > 0 else "FAILED"
            
            # Log bot run
            self.database.log_bot_run(
                signals_found=len(self.wyckoff_strategy.symbols) if self.wyckoff_strategy else 0,
                trades_executed=trades_executed,
                errors=errors,
                portfolio_value=portfolio_value,
                available_cash=available_cash,
                status=status,
                log_details=log_details
            )
            
            # Final summary
            self.logger.info("📊 TRADING SESSION SUMMARY")
            self.logger.info(f"   Trades Executed: {trades_executed}")
            self.logger.info(f"   Errors: {errors}")
            self.logger.info(f"   Portfolio Value: ${portfolio_value:.2f}")
            self.logger.info(f"   Available Cash: ${available_cash:.2f}")
            self.logger.info(f"   Execution Time: {execution_time:.1f}s")
            
            if trades_executed > 0:
                self.logger.info("✅ Bot completed successfully with trades")
            elif errors == 0:
                self.logger.info("✅ Bot completed successfully (no trades needed)")
            else:
                self.logger.warning("⚠️ Bot completed with errors")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in bot execution: {e}")
            self.logger.error(traceback.format_exc())
            
            # Log failed run
            self.database.log_bot_run(0, 0, 1, 0, 0, "CRITICAL_ERROR", str(e))
            return False
        
        finally:
            # Cleanup
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point for the trading bot"""
    print("🤖 Wyckoff Trading Bot Starting...")
    
    bot = WyckoffTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Trading bot completed successfully!")
        sys.exit(0)
    else:
        print("❌ Trading bot failed! Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()