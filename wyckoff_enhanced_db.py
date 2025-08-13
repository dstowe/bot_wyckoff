#!/usr/bin/env python3
"""
SIMPLIFIED: Wyckoff Trading Bot with Enhanced Stop Loss System
Now imports the comprehensive database from fractional_position_system.py
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

# Import the comprehensive database from fractional_position_system
from fractional_position_system import EnhancedTradingDatabase


class WyckoffStopLossManager:
    """Manages Wyckoff pattern-aware stop losses"""
    
    def __init__(self, database, logger):
        self.database = database
        self.logger = logger
        
        # Stop loss configuration
        self.phase_stop_configs = {
            'ST': {
                'initial_stop_pct': 0.08,  # 8% initial stop
                'support_break_threshold': 0.02,  # 2% below key support
                'time_stop_days': 30,  # Time-based stop
                'volume_divergence_threshold': 0.5  # Volume decline threshold
            },
            'SOS': {
                'initial_stop_pct': 0.06,  # 6% initial stop
                'support_break_threshold': 0.015,  # 1.5% below breakout level
                'time_stop_days': 45,
                'volume_divergence_threshold': 0.4
            },
            'LPS': {
                'initial_stop_pct': 0.05,  # 5% initial stop
                'support_break_threshold': 0.01,  # 1% below support test
                'time_stop_days': 60,
                'volume_divergence_threshold': 0.3
            },
            'BU': {
                'initial_stop_pct': 0.07,  # 7% initial stop
                'support_break_threshold': 0.02,  # 2% below pullback low
                'time_stop_days': 40,
                'volume_divergence_threshold': 0.4
            }
        }
    
    def create_wyckoff_stop_strategy(self, symbol: str, entry_data: Dict, market_data: pd.DataFrame) -> bool:
        """Create Wyckoff pattern-aware stop loss strategy"""
        try:
            entry_phase = entry_data.get('entry_phase', 'ST')
            entry_price = entry_data.get('avg_cost', 0)
            account_type = entry_data.get('account_type', 'Cash Account')
            
            if entry_price <= 0:
                self.logger.error(f"Invalid entry price for {symbol}: {entry_price}")
                return False
            
            config = self.phase_stop_configs.get(entry_phase, self.phase_stop_configs['ST'])
            
            # Calculate key levels from market data
            key_levels = self._calculate_key_levels(market_data, entry_phase, entry_price)
            
            # Determine stop price based on Wyckoff context
            stop_price = self._calculate_wyckoff_stop_price(
                entry_price, entry_phase, key_levels, config
            )
            
            stop_percentage = (entry_price - stop_price) / entry_price
            
            # Create comprehensive context data
            context_data = {
                'entry_phase': entry_phase,
                'entry_price': entry_price,
                'account_type': account_type,
                'key_support': key_levels.get('support', 0),
                'key_resistance': key_levels.get('resistance', 0),
                'breakout_level': key_levels.get('breakout', 0),
                'config_used': config,
                'calculation_method': 'wyckoff_context',
                'created_date': datetime.now().isoformat()
            }
            
            # Store stop strategy in database
            with sqlite3.connect(self.database.db_path) as conn:
                conn.execute('''
                    INSERT INTO stop_strategies (
                        symbol, strategy_type, initial_price, stop_price, stop_percentage,
                        key_support_level, key_resistance_level, breakout_level,
                        time_entered, context_data, stop_reason, bot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    f'WYCKOFF_{entry_phase}',
                    entry_price,
                    stop_price,
                    stop_percentage,
                    key_levels.get('support', 0),
                    key_levels.get('resistance', 0),
                    key_levels.get('breakout', 0),
                    datetime.now().isoformat(),
                    json.dumps(context_data),
                    f'Wyckoff {entry_phase} context stop',
                    self.database.bot_id
                ))
            
            self.logger.info(f"✅ Created Wyckoff stop for {symbol}")
            self.logger.info(f"   Phase: {entry_phase}")
            self.logger.info(f"   Entry: ${entry_price:.2f}")
            self.logger.info(f"   Stop: ${stop_price:.2f} ({stop_percentage:.1%})")
            self.logger.info(f"   Key Support: ${key_levels.get('support', 0):.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating Wyckoff stop for {symbol}: {e}")
            return False
    
    def _calculate_key_levels(self, data: pd.DataFrame, phase: str, entry_price: float) -> Dict:
        """Calculate key Wyckoff levels from market data"""
        if len(data) < 20:
            return {'support': entry_price * 0.95, 'resistance': entry_price * 1.05, 'breakout': entry_price}
        
        try:
            # Calculate support and resistance levels based on phase
            if phase == 'ST':
                # For Secondary Test, look for the initial support
                support = data['Low'].tail(30).min()
                resistance = data['High'].tail(20).quantile(0.8)
                breakout = resistance
                
            elif phase == 'SOS':
                # For Sign of Strength, the breakout level is critical
                resistance = data['High'].tail(30).max()
                support = data['Low'].tail(20).quantile(0.2)
                breakout = resistance * 0.98  # Just below recent high
                
            elif phase == 'LPS':
                # For Last Point of Support, support level is key
                support = data['Low'].tail(40).min()
                resistance = data['High'].tail(30).quantile(0.9)
                breakout = support * 1.02  # Just above support
                
            elif phase == 'BU':
                # For Back-up, look for pullback levels
                support = data['Low'].tail(15).min()
                resistance = data['High'].tail(30).max()
                breakout = data['High'].tail(10).max()
                
            else:
                # Default calculation
                support = data['Low'].tail(20).min()
                resistance = data['High'].tail(20).max()
                breakout = entry_price
            
            return {
                'support': support,
                'resistance': resistance,
                'breakout': breakout
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating key levels: {e}")
            return {'support': entry_price * 0.95, 'resistance': entry_price * 1.05, 'breakout': entry_price}
    
    def _calculate_wyckoff_stop_price(self, entry_price: float, phase: str, 
                                    key_levels: Dict, config: Dict) -> float:
        """Calculate Wyckoff-aware stop price"""
        
        # Get configuration for this phase
        initial_stop_pct = config['initial_stop_pct']
        support_break_threshold = config['support_break_threshold']
        
        # Calculate different stop levels
        percentage_stop = entry_price * (1 - initial_stop_pct)
        
        support_level = key_levels.get('support', entry_price * 0.95)
        support_stop = support_level * (1 - support_break_threshold)
        
        # Use the higher (less aggressive) of the two stops
        # This prevents stops that are too tight below key support
        wyckoff_stop = max(percentage_stop, support_stop)
        
        # Ensure stop is below entry price
        wyckoff_stop = min(wyckoff_stop, entry_price * 0.98)
        
        return wyckoff_stop
    
    def check_wyckoff_stop_conditions(self, symbol: str, current_price: float, 
                                    current_data: pd.DataFrame) -> Optional[Dict]:
        """Check if any Wyckoff stop conditions are met"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                result = conn.execute('''
                    SELECT * FROM stop_strategies 
                    WHERE symbol = ? AND is_active = TRUE AND bot_id = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (symbol, self.database.bot_id)).fetchone()
                
                if not result:
                    return None
                
                # Parse the result
                columns = ['id', 'symbol', 'strategy_type', 'initial_price', 'stop_price', 
                          'stop_percentage', 'trailing_high', 'key_support_level', 
                          'key_resistance_level', 'breakout_level', 'pullback_low',
                          'time_entered', 'context_data', 'stop_reason', 'is_active', 
                          'bot_id', 'created_at', 'updated_at']
                
                stop_data = dict(zip(columns, result))
                
                # Check basic price stop
                if current_price <= stop_data['stop_price']:
                    return {
                        'triggered': True,
                        'reason': 'WYCKOFF_STOP_PRICE',
                        'stop_price': stop_data['stop_price'],
                        'current_price': current_price,
                        'message': f"Price ${current_price:.2f} hit Wyckoff stop ${stop_data['stop_price']:.2f}"
                    }
                
                # Parse context data for additional checks
                try:
                    context = json.loads(stop_data['context_data'])
                    entry_phase = context.get('entry_phase', 'ST')
                    config = self.phase_stop_configs.get(entry_phase, self.phase_stop_configs['ST'])
                    
                    # Check time-based stop
                    time_entered = datetime.fromisoformat(stop_data['time_entered'])
                    days_held = (datetime.now() - time_entered).days
                    
                    if days_held > config['time_stop_days']:
                        return {
                            'triggered': True,
                            'reason': 'WYCKOFF_TIME_STOP',
                            'days_held': days_held,
                            'max_days': config['time_stop_days'],
                            'message': f"Position held {days_held} days, exceeds {config['time_stop_days']} day limit"
                        }
                    
                    # Check support break for specific phases
                    key_support = stop_data['key_support_level']
                    if key_support > 0:
                        support_break_threshold = config['support_break_threshold']
                        critical_support = key_support * (1 - support_break_threshold)
                        
                        if current_price < critical_support:
                            return {
                                'triggered': True,
                                'reason': 'WYCKOFF_SUPPORT_BREAK',
                                'support_level': key_support,
                                'break_level': critical_support,
                                'current_price': current_price,
                                'message': f"Price ${current_price:.2f} broke critical support ${critical_support:.2f}"
                            }
                    
                    # Check volume divergence (if we have recent volume data)
                    if len(current_data) >= 10:
                        recent_volume = current_data['Volume'].tail(5).mean()
                        historical_volume = current_data['Volume'].tail(20).mean()
                        
                        if recent_volume < historical_volume * config['volume_divergence_threshold']:
                            return {
                                'triggered': True,
                                'reason': 'WYCKOFF_VOLUME_DIVERGENCE',
                                'recent_volume': recent_volume,
                                'historical_volume': historical_volume,
                                'threshold': config['volume_divergence_threshold'],
                                'message': f"Volume divergence detected: recent {recent_volume:.0f} vs historical {historical_volume:.0f}"
                            }
                    
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse context data for {symbol}")
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error checking Wyckoff stop conditions for {symbol}: {e}")
            return None


class SimplifiedWyckoffBot:
    """Simplified Wyckoff bot that uses the comprehensive database"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None  # Will use the comprehensive one from fractional_position_system
        self.config = PersonalTradingConfig()
        self.stop_manager = None
        
        # Trading parameters
        self.min_signal_strength = 0.5
        self.max_position_value = 500.0  # Maximum $ value per position
        self.max_positions = 5
        self.buy_phases = ['ST', 'SOS', 'LPS', 'BU']
        self.sell_phases = ['PS', 'SC']
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 SIMPLIFIED WYCKOFF BOT")
        self.logger.info("📊 Using comprehensive database from fractional_position_system")
    
    def initialize_systems(self) -> bool:
        """Initialize all systems"""
        try:
            self.logger.info("🔧 Initializing systems...")
            
            self.main_system = MainSystem()
            self.wyckoff_strategy = WyckoffPnFStrategy()
            # Use the comprehensive database
            self.database = EnhancedTradingDatabase()
            self.stop_manager = WyckoffStopLossManager(self.database, self.logger)
            
            self.logger.info("✅ Systems initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions using the comprehensive database"""
        return self.database.get_all_positions()
    
    def calculate_position_size(self, signal: WyckoffSignal, available_cash: float) -> float:
        """Calculate position size based on signal strength and available cash"""
        
        # Base position size
        base_size = min(self.max_position_value, available_cash * 0.2)
        
        # Adjust based on Wyckoff phase
        phase_multipliers = {
            'ST': 0.6,   # Smaller position for test
            'SOS': 1.0,  # Full position for breakout
            'LPS': 0.8,  # Good position for support test
            'BU': 0.7    # Moderate position for pullback
        }
        
        multiplier = phase_multipliers.get(signal.phase, 0.5)
        
        # Adjust based on signal strength
        strength_multiplier = min(signal.strength / 0.5, 1.5)  # Cap at 1.5x
        
        position_size = base_size * multiplier * strength_multiplier
        
        # Ensure minimum trade size
        position_size = max(position_size, 10.0)
        
        # Ensure we don't exceed available cash minus buffer
        position_size = min(position_size, available_cash - 50.0)
        
        return position_size
    
    def execute_buy_order(self, signal: WyckoffSignal, account, position_size: float) -> bool:
        """Execute buy order with Wyckoff stop loss creation"""
        try:
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            quote_data = self.main_system.wb.get_quote(signal.symbol)
            if not quote_data or 'close' not in quote_data:
                self.logger.error(f"❌ Could not get quote for {signal.symbol}")
                return False
            
            current_price = float(quote_data['close'])
            shares_to_buy = position_size / current_price
            shares_to_buy = round(shares_to_buy, 5)
            
            self.logger.info(f"💰 Buying {shares_to_buy:.5f} shares of {signal.symbol} at ${current_price:.2f}")
            self.logger.info(f"   Account: {account.account_type}")
            self.logger.info(f"   Position: ${position_size:.2f} ({signal.phase} phase, strength: {signal.strength:.2f})")
            
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
                
                # Log using comprehensive database
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
                
                # Update position with account type
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=shares_to_buy,
                    cost=current_price,
                    account_type=account.account_type,
                    entry_phase=signal.phase,
                    entry_strength=signal.strength
                )
                
                # Create Wyckoff stop loss strategy
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(signal.symbol)
                    hist_data = ticker.history(period="3mo")
                    
                    if not hist_data.empty:
                        entry_data = {
                            'entry_phase': signal.phase,
                            'avg_cost': current_price,
                            'account_type': account.account_type
                        }
                        
                        self.stop_manager.create_wyckoff_stop_strategy(
                            signal.symbol, entry_data, hist_data
                        )
                    
                except Exception as e:
                    self.logger.warning(f"Could not create stop loss for {signal.symbol}: {e}")
                
                self.logger.info(f"✅ Buy order executed: {signal.symbol} - Order ID: {order_id}")
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Buy order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy for {signal.symbol}: {e}")
            return False
    
    def check_stop_losses(self) -> int:
        """Check all positions for stop loss conditions"""
        stop_losses_executed = 0
        current_positions = self.get_current_positions()
        
        if not current_positions:
            return 0
        
        self.logger.info("🔍 Checking Wyckoff stop loss conditions...")
        
        for position_key, position in current_positions.items():
            symbol = position['symbol']
            account_type = position['account_type']
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                current_data = ticker.history(period="1mo")
                
                if current_data.empty:
                    continue
                
                current_price = float(current_data['Close'].iloc[-1])
                
                # Check Wyckoff stop conditions
                stop_condition = self.stop_manager.check_wyckoff_stop_conditions(
                    symbol, current_price, current_data
                )
                
                if stop_condition and stop_condition.get('triggered'):
                    self.logger.warning(f"🔴 Stop loss triggered for {symbol} ({account_type})")
                    self.logger.warning(f"   Reason: {stop_condition['reason']}")
                    self.logger.warning(f"   Message: {stop_condition['message']}")
                    
                    # Execute stop loss
                    if self.execute_stop_loss(symbol, position, stop_condition):
                        stop_losses_executed += 1
                
            except Exception as e:
                self.logger.error(f"Error checking stop for {symbol}: {e}")
        
        return stop_losses_executed
    
    def execute_stop_loss(self, symbol: str, position: Dict, stop_condition: Dict) -> bool:
        """Execute stop loss order"""
        try:
            account_type = position['account_type']
            shares_to_sell = position['shares']
            
            # Find the account
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts if acc.account_type == account_type), None)
            
            if not account:
                self.logger.error(f"❌ Could not find account {account_type} for {symbol}")
                return False
            
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account {account_type}")
                return False
            
            self.logger.info(f"🔴 Executing stop loss: {shares_to_sell:.5f} shares of {symbol}")
            self.logger.info(f"   Reason: {stop_condition['reason']}")
            
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
                order_id = order_result.get('orderId', 'STOP_LOSS')
                
                # Log the trade
                self.database.log_trade(
                    symbol=symbol,
                    action='STOP_LOSS_SELL',
                    quantity=shares_to_sell,
                    price=stop_condition.get('current_price', 0),
                    signal_phase='STOP_LOSS',
                    signal_strength=1.0,
                    account_type=account_type,
                    order_id=order_id
                )
                
                # Update position
                self.database.update_position(
                    symbol=symbol,
                    shares=-shares_to_sell,
                    cost=stop_condition.get('current_price', 0),
                    account_type=account_type
                )
                
                # Deactivate stop strategies
                self.database.deactivate_stop_strategies(symbol)
                
                self.logger.info(f"✅ Stop loss executed: {symbol}")
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Stop loss failed for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing stop loss for {symbol}: {e}")
            return False
    
    def run_trading_cycle(self) -> Tuple[int, int]:
        """Run simplified trading cycle"""
        trades_executed = 0
        stop_losses_executed = 0
        
        try:
            # Check stop losses first
            stop_losses_executed = self.check_stop_losses()
            
            # Get current positions and available cash
            current_positions = self.get_current_positions()
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            total_available_cash = sum(acc.settled_funds for acc in enabled_accounts)
            
            self.logger.info(f"💰 Available cash: ${total_available_cash:.2f}")
            self.logger.info(f"📊 Current positions: {len(current_positions)}")
            
            # Only buy if we have room for more positions
            if len(current_positions) < self.max_positions and total_available_cash > 100:
                
                self.logger.info("🔍 Scanning for Wyckoff signals...")
                signals = self.wyckoff_strategy.scan_market()
                
                if signals:
                    # Filter for buy signals
                    buy_signals = [s for s in signals if (
                        s.phase in self.buy_phases and 
                        s.strength >= self.min_signal_strength and
                        s.volume_confirmation
                    )]
                    
                    if buy_signals:
                        self.logger.info(f"📊 Found {len(buy_signals)} potential buy signals")
                        
                        # Sort by combined score
                        buy_signals.sort(key=lambda x: x.combined_score, reverse=True)
                        
                        # Execute best signals
                        for signal in buy_signals[:self.max_positions - len(current_positions)]:
                            # Find account with most cash
                            best_account = max(enabled_accounts, key=lambda x: x.settled_funds)
                            
                            if best_account.settled_funds > 100:
                                position_size = self.calculate_position_size(signal, best_account.settled_funds)
                                
                                if position_size >= 10.0:
                                    if self.execute_buy_order(signal, best_account, position_size):
                                        trades_executed += 1
                                        best_account.settled_funds -= position_size
                                    
                                    # Small delay between orders
                                    time.sleep(2)
                    else:
                        self.logger.info("📊 No qualifying buy signals found")
                else:
                    self.logger.info("📊 No signals returned from market scan")
            else:
                self.logger.info("📊 Skipping buy signals (position limits or insufficient cash)")
            
            return trades_executed, stop_losses_executed
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
            return trades_executed, stop_losses_executed
    
    def run(self) -> bool:
        """Main execution"""
        try:
            self.logger.info("🚀 Starting Simplified Wyckoff Trading Bot")
            
            if not self.initialize_systems():
                return False
            
            if not self.main_system.run():
                return False
            
            # Run trading cycle
            trades, stop_losses = self.run_trading_cycle()
            
            # Log results
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            self.database.log_bot_run(
                signals_found=trades,
                trades_executed=trades,
                stop_losses_executed=stop_losses,
                errors=0,
                portfolio_value=sum(acc.net_liquidation for acc in enabled_accounts),
                available_cash=sum(acc.settled_funds for acc in enabled_accounts),
                status="COMPLETED",
                log_details=f"Trades: {trades}, Stop Losses: {stop_losses}"
            )
            
            # Summary
            self.logger.info("📊 SIMPLIFIED WYCKOFF BOT SESSION SUMMARY")
            self.logger.info(f"   Buy Orders: {trades}")
            self.logger.info(f"   Stop Losses: {stop_losses}")
            self.logger.info(f"   Total Actions: {trades + stop_losses}")
            
            if trades + stop_losses > 0:
                self.logger.info("✅ Bot completed with actions")
            else:
                self.logger.info("✅ Bot completed (no actions needed)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
            return False
        
        finally:
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point"""
    print("🚀 Simplified Wyckoff Trading Bot Starting...")
    
    bot = SimplifiedWyckoffBot()
    success = bot.run()
    
    if success:
        print("✅ Simplified Wyckoff bot completed!")
        sys.exit(0)
    else:
        print("❌ Simplified Wyckoff bot failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()