#!/usr/bin/env python3
"""
ENHANCED: Fractional Position Building System with Wyckoff-Based Selling
- Automatic account value detection from Webull
- Dynamic position sizing based on real account balances
- Wyckoff PS/SC distribution signals for selling
- Simplified and optimized for small account growth
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


class DynamicAccountManager:
    """Manages dynamic position sizing based on real account values"""
    
    def __init__(self, logger):
        self.logger = logger
        self.last_update = None
        self.cached_config = None
        
    def get_dynamic_config(self, account_manager) -> Dict:
        """Get dynamic configuration based on real account values"""
        try:
            # Get enabled accounts and their real values
            enabled_accounts = account_manager.get_enabled_accounts()
            if not enabled_accounts:
                return self._get_fallback_config()
            
            # Calculate total portfolio value and available cash
            total_value = sum(acc.net_liquidation for acc in enabled_accounts)
            total_cash = sum(acc.settled_funds for acc in enabled_accounts)
            
            self.logger.info(f"💰 Real Account Values - Total: ${total_value:.2f}, Cash: ${total_cash:.2f}")
            
            # Create dynamic configuration based on real values
            config = self._calculate_dynamic_parameters(total_value, total_cash, len(enabled_accounts))
            
            # Cache the configuration
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
        
        # Log the calculated parameters
        self.logger.info(f"📊 Dynamic Config for ${total_cash:.2f} across {num_accounts} accounts:")
        self.logger.info(f"   Base Position Size: ${base_position_size:.2f} ({base_position_pct:.1%} of cash)")
        self.logger.info(f"   Max Positions: {max_positions}")
        self.logger.info(f"   Min Balance Preserve: ${min_balance_preserve:.2f}")
        self.logger.info(f"   Wyckoff ST Initial: {wyckoff_phases['ST']['initial_allocation']:.0%}")
        self.logger.info(f"   Wyckoff SOS Initial: {wyckoff_phases['SOS']['initial_allocation']:.0%}")
        
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
    
    def should_add_to_position(self, symbol: str, phase: str, current_allocation_pct: float) -> Tuple[bool, str, float]:
        """Check if we should add to existing position"""
        if not self.current_config:
            return False, "NO_CONFIG", 0.0
        
        phase_config = self.current_config['wyckoff_phases'].get(phase, {})
        
        if not phase_config.get('allow_additions', False):
            return False, f"PHASE_{phase}_NO_ADDITIONS", 0.0
        
        max_allocation = phase_config.get('max_total_allocation', 1.0)
        
        if current_allocation_pct >= max_allocation:
            return False, "MAX_ALLOCATION_REACHED", 0.0
        
        # Calculate how much more we can add
        additional_allocation = min(0.3, max_allocation - current_allocation_pct)  # Max 30% addition
        
        if additional_allocation < 0.1:  # Less than 10% addition not worth it
            return False, "ADDITION_TOO_SMALL", 0.0
        
        return True, f"ADD_{additional_allocation:.0%}_TO_POSITION", additional_allocation
    
    def check_wyckoff_sell_signals(self, signals: List[WyckoffSignal], current_positions: Dict) -> List[Tuple[WyckoffSignal, Dict]]:
        """Check for Wyckoff-based sell signals (PS/SC distribution phases)"""
        sell_signals = []
        
        # Distribution phases that signal selling
        distribution_phases = ['PS', 'SC']
        
        for signal in signals:
            if signal.symbol in current_positions and signal.phase in distribution_phases:
                if signal.strength >= 0.5 and signal.volume_confirmation:
                    position = current_positions[signal.symbol]
                    sell_signals.append((signal, position))
                    self.logger.info(f"🔴 Wyckoff Sell Signal: {signal.symbol} ({signal.phase}) - Strength: {signal.strength:.2f}")
        
        return sell_signals
    
    def check_profit_scaling_opportunities(self, wb_client, current_positions: Dict) -> List[Dict]:
        """Check current positions for profit-taking opportunities"""
        if not self.current_config:
            return []
        
        scaling_opportunities = []
        
        for symbol, position in current_positions.items():
            try:
                # Get current price
                quote_data = wb_client.get_quote(symbol)
                if not quote_data or 'close' not in quote_data:
                    continue
                
                current_price = float(quote_data['close'])
                avg_cost = position['avg_cost']
                shares = position['shares']
                gain_pct = (current_price - avg_cost) / avg_cost
                
                # Check each profit target
                for target in self.current_config['profit_targets']:
                    if gain_pct >= target['gain_pct']:
                        # Check if we already took profit at this level
                        if not self._already_scaled_at_level(symbol, target['gain_pct']):
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
                                    'description': target['description'],
                                    'remaining_shares': shares - shares_to_sell,
                                    'account_type': position['account_type']
                                })
                                break  # Only one scaling action per position
                            else:
                                self.logger.debug(f"💰 {symbol}: Scaling amount ${sale_value:.2f} below $5 minimum")
            
            except Exception as e:
                self.logger.error(f"Error checking scaling for {symbol}: {e}")
                continue
        
        return scaling_opportunities
    
    def _already_scaled_at_level(self, symbol: str, gain_pct: float) -> bool:
        """Check if we already scaled at this gain level"""
        # Implementation would check database for previous partial sales
        # For now, simplified version
        return False
    
    def create_enhanced_database_schema(self):
        """Create enhanced database schema"""
        with sqlite3.connect(self.database.db_path) as conn:
            # Enhanced positions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions_enhanced (
                    symbol TEXT PRIMARY KEY,
                    target_position_size REAL NOT NULL,
                    current_allocation_pct REAL NOT NULL DEFAULT 0.0,
                    total_shares REAL NOT NULL DEFAULT 0.0,
                    avg_cost REAL NOT NULL DEFAULT 0.0,
                    total_invested REAL NOT NULL DEFAULT 0.0,
                    entry_phases TEXT,
                    addition_count INTEGER DEFAULT 1,
                    max_additions INTEGER DEFAULT 3,
                    first_entry_date TEXT,
                    last_addition_date TEXT,
                    account_type TEXT NOT NULL,
                    wyckoff_score REAL,
                    position_status TEXT DEFAULT 'BUILDING',
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
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
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.logger.info("✅ Enhanced fractional database schema ready")


class SimplifiedFractionalTradingBot:
    """Simplified fractional trading bot with Wyckoff selling and dynamic account sizing"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        self.dynamic_manager = None
        self.position_manager = None
        
        # Trading parameters
        self.min_signal_strength = 0.5
        self.buy_phases = ['ST', 'SOS', 'LPS', 'BU']
        self.sell_phases = ['PS', 'SC']  # Wyckoff distribution phases
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging to console only - no smart_fractional_bot_ log files"""
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
        self.logger.info("🚀 SMART FRACTIONAL TRADING BOT")
        self.logger.info("💰 Dynamic account sizing based on real Webull values")
        self.logger.info("🎯 Wyckoff-based buying AND selling")
        self.logger.info("📈 Optimized for small account growth")
    
    def initialize_systems(self) -> bool:
        """Initialize all systems"""
        try:
            self.logger.info("🔧 Initializing smart fractional systems...")
            
            # Initialize main system
            self.main_system = MainSystem()
            
            # Initialize Wyckoff strategy
            self.wyckoff_strategy = WyckoffPnFStrategy()
            
            # Initialize database
            from wyckoff_enhanced_db import EnhancedTradingDatabase
            self.database = EnhancedTradingDatabase()
            
            # Initialize dynamic account manager
            self.dynamic_manager = DynamicAccountManager(self.logger)
            
            # Initialize smart position manager
            self.position_manager = SmartFractionalPositionManager(
                self.database, self.dynamic_manager, self.logger
            )
            self.position_manager.create_enhanced_database_schema()
            
            self.logger.info("✅ Smart fractional systems initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
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
    
    def execute_buy_order(self, signal: WyckoffSignal, account, position_size: float) -> bool:
        """Execute buy order"""
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
            self.logger.info(f"   Position: ${position_size:.2f} ({signal.phase} phase)")
            
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
                
                # Log trade
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
                
                # Update position
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=shares_to_buy,
                    cost=current_price,
                    account_type=account.account_type
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
    
    def execute_wyckoff_sell_order(self, signal: WyckoffSignal, position: Dict) -> bool:
        """Execute Wyckoff-based sell order"""
        try:
            # Find account that holds this position
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts 
                           if acc.account_type == position['account_type']), None)
            
            if not account:
                self.logger.error(f"❌ Could not find account for {signal.symbol}")
                return False
            
            # Switch to trading account
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            shares_to_sell = position['shares']
            
            self.logger.info(f"🔴 Wyckoff Sell: {shares_to_sell:.5f} shares of {signal.symbol}")
            self.logger.info(f"   Reason: {signal.phase} distribution signal (Strength: {signal.strength:.2f})")
            
            # Place sell order
            order_result = self.main_system.wb.place_order(
                stock=signal.symbol,
                price=0,
                action='SELL',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_sell,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                
                # Get current price for logging
                quote_data = self.main_system.wb.get_quote(signal.symbol)
                current_price = float(quote_data.get('close', signal.price)) if quote_data else signal.price
                
                # Log trade
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
                
                # Calculate P&L
                profit_loss = (current_price - position['avg_cost']) * shares_to_sell
                profit_loss_pct = (profit_loss / position['total_invested']) * 100
                self.logger.info(f"📊 Wyckoff Sell P&L: ${profit_loss:.2f} ({profit_loss_pct:.1f}%)")
                
                # Update position
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=-shares_to_sell,
                    cost=current_price,
                    account_type=account.account_type
                )
                
                self.logger.info(f"✅ Wyckoff sell executed: {signal.symbol} - Order ID: {order_id}")
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Sell order failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing Wyckoff sell for {signal.symbol}: {e}")
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
            
            self.logger.info(f"💰 Profit Scaling: {shares_to_sell:.5f} shares of {symbol}")
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
                # Log as partial sale and update position
                # Implementation details here...
                self.logger.info(f"✅ Profit scaling executed: {symbol}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing profit scaling: {e}")
            return False
    
    def run_trading_cycle(self) -> Tuple[int, int, int]:
        """Run complete trading cycle"""
        trades_executed = 0
        wyckoff_sells = 0
        profit_scales = 0
        
        try:
            # Update dynamic configuration based on real account values
            config = self.position_manager.update_config(self.main_system.account_manager)
            
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Get Wyckoff signals
            self.logger.info("🔍 Scanning for Wyckoff signals...")
            signals = self.wyckoff_strategy.scan_market()
            
            if signals:
                # Check for Wyckoff sell signals first
                wyckoff_sell_signals = self.position_manager.check_wyckoff_sell_signals(signals, current_positions)
                
                for signal, position in wyckoff_sell_signals:
                    if self.execute_wyckoff_sell_order(signal, position):
                        wyckoff_sells += 1
                        # Remove from current positions
                        if signal.symbol in current_positions:
                            del current_positions[signal.symbol]
                
                # Check for profit scaling opportunities
                profit_opportunities = self.position_manager.check_profit_scaling_opportunities(
                    self.main_system.wb, current_positions
                )
                
                for opportunity in profit_opportunities[:3]:  # Limit to 3 scalings per run
                    if self.execute_profit_scaling(opportunity):
                        profit_scales += 1
                
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
            
            return trades_executed, wyckoff_sells, profit_scales
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
            return trades_executed, wyckoff_sells, profit_scales
    
    def run(self) -> bool:
        """Main execution"""
        try:
            self.logger.info("🚀 Starting Smart Fractional Trading Bot")
            
            # Initialize systems
            if not self.initialize_systems():
                return False
            
            # Authenticate
            if not self.main_system.run():
                return False
            
            # Run trading cycle
            trades, sells, scales = self.run_trading_cycle()
            
            # Summary
            total_actions = trades + sells + scales
            
            self.logger.info("📊 SMART FRACTIONAL TRADING SESSION SUMMARY")
            self.logger.info(f"   Buy Orders: {trades}")
            self.logger.info(f"   Wyckoff Sells: {sells}")
            self.logger.info(f"   Profit Scaling: {scales}")
            self.logger.info(f"   Total Actions: {total_actions}")
            
            if total_actions > 0:
                self.logger.info("✅ Smart fractional bot completed with actions")
            else:
                self.logger.info("✅ Smart fractional bot completed (no actions needed)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
            return False
        
        finally:
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point"""
    print("🚀 Smart Fractional Trading Bot Starting...")
    
    bot = SimplifiedFractionalTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Smart fractional trading bot completed!")
        sys.exit(0)
    else:
        print("❌ Smart fractional trading bot failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()