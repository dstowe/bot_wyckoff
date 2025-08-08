#!/usr/bin/env python3
"""
ENHANCED: Wyckoff Trading Bot with Fractional Share Position Building & Scaling
Sophisticated entry/exit timing, position building, and scaling strategies
"""

import sys
import logging
import traceback
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import time

# Import existing systems
from main import MainSystem
from strategies.wyckoff.wyckoff import WyckoffPnFStrategy, WyckoffSignal
from config.config import PersonalTradingConfig


class FractionalPositionManager:
    """Manages fractional share position building across Wyckoff phases"""
    
    def __init__(self, database, logger):
        self.database = database
        self.logger = logger
        
        # Position building configuration by Wyckoff phase
        self.phase_allocation_config = {
            'ST': {
                'initial_allocation': 0.25,  # 25% of target position
                'description': 'Testing phase - start small',
                'allow_additions': False,
                'risk_level': 'HIGH'
            },
            'Creek': {
                'initial_allocation': 0.0,   # No new positions in consolidation
                'description': 'Consolidation - hold only',
                'allow_additions': False,
                'risk_level': 'NEUTRAL'
            },
            'SOS': {
                'initial_allocation': 0.50,  # 50% of target (or add 50% to existing)
                'description': 'Breakout confirmation - main position',
                'allow_additions': True,
                'max_total_allocation': 0.75,
                'risk_level': 'MEDIUM'
            },
            'LPS': {
                'initial_allocation': 0.25,  # Final 25% (or add remaining)
                'description': 'Support test confirmation - complete position',
                'allow_additions': True,
                'max_total_allocation': 1.0,
                'risk_level': 'LOW'
            },
            'BU': {
                'initial_allocation': 0.25,  # Add to existing on pullback bounce
                'description': 'Pullback bounce - opportunistic add',
                'allow_additions': True,
                'max_total_allocation': 1.0,
                'risk_level': 'MEDIUM'
            }
        }
        
        # Account size-based position sizing
        self.account_sizing_config = [
            {'min_account': 0, 'max_account': 500, 'entry_range': (5, 15), 'max_additions': 3},
            {'min_account': 500, 'max_account': 1000, 'entry_range': (10, 25), 'max_additions': 4},
            {'min_account': 1000, 'max_account': 2000, 'entry_range': (15, 35), 'max_additions': 5},
            {'min_account': 2000, 'max_account': float('inf'), 'entry_range': (25, 50), 'max_additions': 6}
        ]
        
        # Scaling out configuration
        self.scaling_out_config = {
            'profit_targets': [
                {'gain_pct': 0.10, 'sell_pct': 0.25, 'description': '10% gain: Take 25% profit'},
                {'gain_pct': 0.20, 'sell_pct': 0.25, 'description': '20% gain: Take another 25%'},
                {'gain_pct': 0.35, 'sell_pct': 0.25, 'description': '35% gain: Take another 25%'},
            ],
            'distribution_signals': {
                'phases': ['PS', 'SC'],
                'sell_pct': 1.0,  # Sell remaining position
                'description': 'Distribution signal: Exit remaining position'
            }
        }
    
    def create_enhanced_database_schema(self):
        """Create enhanced database schema for fractional position building"""
        with sqlite3.connect(self.database.db_path) as conn:
            # Enhanced positions table with position building tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions_enhanced (
                    symbol TEXT PRIMARY KEY,
                    target_position_size REAL NOT NULL,
                    current_allocation_pct REAL NOT NULL DEFAULT 0.0,
                    total_shares REAL NOT NULL DEFAULT 0.0,
                    avg_cost REAL NOT NULL DEFAULT 0.0,
                    total_invested REAL NOT NULL DEFAULT 0.0,
                    entry_phases TEXT,  -- JSON array of phases entered
                    addition_count INTEGER DEFAULT 0,
                    max_additions INTEGER DEFAULT 3,
                    first_entry_date TEXT,
                    last_addition_date TEXT,
                    account_type TEXT NOT NULL,
                    wyckoff_score REAL,
                    position_status TEXT DEFAULT 'BUILDING',  -- BUILDING, COMPLETE, SCALING_OUT
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Partial sales tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS partial_sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sale_date TEXT NOT NULL,
                    shares_sold REAL NOT NULL,
                    sale_price REAL NOT NULL,
                    sale_reason TEXT NOT NULL,  -- 'PROFIT_10PCT', 'PROFIT_20PCT', 'WYCKOFF_PS', etc.
                    remaining_shares REAL NOT NULL,
                    gain_pct REAL,
                    profit_amount REAL,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Position building events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS position_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,  -- 'INITIAL_ENTRY', 'ADDITION', 'PARTIAL_SALE', 'COMPLETE_EXIT'
                    event_date TEXT NOT NULL,
                    wyckoff_phase TEXT,
                    shares_traded REAL NOT NULL,
                    price REAL NOT NULL,
                    allocation_before REAL,
                    allocation_after REAL,
                    reasoning TEXT,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_enhanced_symbol ON positions_enhanced(symbol, bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_partial_sales_symbol ON partial_sales(symbol, bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_position_events_symbol ON position_events(symbol, bot_id)')
            
            self.logger.info("✅ Enhanced fractional position building database schema created")
    
    def get_account_sizing_config(self, account_value: float) -> Dict:
        """Get position sizing configuration based on account size"""
        for config in self.account_sizing_config:
            if config['min_account'] <= account_value < config['max_account']:
                return config
        
        # Default to largest account config
        return self.account_sizing_config[-1]
    
    def calculate_target_position_size(self, account_value: float, wyckoff_score: float) -> float:
        """Calculate target position size based on account value and Wyckoff score"""
        sizing_config = self.get_account_sizing_config(account_value)
        min_size, max_size = sizing_config['entry_range']
        
        # Adjust size based on Wyckoff score
        if wyckoff_score > 0.7:
            target_size = max_size  # Full allocation for high-conviction signals
        elif wyckoff_score > 0.5:
            target_size = min_size + (max_size - min_size) * 0.75  # 75% allocation
        elif wyckoff_score > 0.4:
            target_size = min_size + (max_size - min_size) * 0.50  # 50% allocation
        else:
            target_size = 0  # No position for low scores
        
        return target_size
    
    def should_add_to_position(self, symbol: str, phase: str, wyckoff_score: float) -> Tuple[bool, str, float]:
        """Determine if we should add to an existing position"""
        try:
            position = self.get_enhanced_position(symbol)
            if not position:
                return True, "NEW_POSITION", 0.0  # New position
            
            # Check if last addition was too recent
            if position['last_addition_date']:
                last_addition = datetime.strptime(position['last_addition_date'], '%Y-%m-%d')
                days_since_last = (datetime.now() - last_addition).days
                if days_since_last < 3:
                    return False, "TOO_RECENT", 0.0
            
            # Check if we've reached max additions
            if position['addition_count'] >= position['max_additions']:
                return False, "MAX_ADDITIONS_REACHED", 0.0
            
            # Check phase-specific rules
            phase_config = self.phase_allocation_config.get(phase, {})
            if not phase_config.get('allow_additions', False):
                return False, f"PHASE_{phase}_NO_ADDITIONS", 0.0
            
            current_allocation = position['current_allocation_pct']
            max_total = phase_config.get('max_total_allocation', 1.0)
            
            if current_allocation >= max_total:
                return False, "ALLOCATION_COMPLETE", 0.0
            
            # Calculate addition percentage
            if phase == 'SOS' and current_allocation < 0.75:
                addition_pct = min(0.50, max_total - current_allocation)
                return True, f"SOS_ADDITION_{addition_pct:.0%}", addition_pct
            elif phase == 'LPS' and current_allocation < 1.0:
                addition_pct = min(0.25, 1.0 - current_allocation)
                return True, f"LPS_COMPLETION_{addition_pct:.0%}", addition_pct
            elif phase == 'BU' and wyckoff_score > 0.6:
                addition_pct = min(0.25, max_total - current_allocation)
                return True, f"BU_BOUNCE_{addition_pct:.0%}", addition_pct
            
            return False, "NO_ADDITION_CRITERIA", 0.0
            
        except Exception as e:
            self.logger.error(f"Error checking position addition for {symbol}: {e}")
            return False, f"ERROR: {e}", 0.0
    
    def get_enhanced_position(self, symbol: str) -> Optional[Dict]:
        """Get enhanced position data for a symbol"""
        with sqlite3.connect(self.database.db_path) as conn:
            result = conn.execute('''
                SELECT * FROM positions_enhanced 
                WHERE symbol = ? AND bot_id = ?
            ''', (symbol, self.database.bot_id)).fetchone()
            
            if result:
                columns = [
                    'symbol', 'target_position_size', 'current_allocation_pct', 'total_shares',
                    'avg_cost', 'total_invested', 'entry_phases', 'addition_count', 'max_additions',
                    'first_entry_date', 'last_addition_date', 'account_type', 'wyckoff_score',
                    'position_status', 'bot_id', 'created_at', 'updated_at'
                ]
                position = dict(zip(columns, result))
                
                # Parse entry_phases JSON
                if position['entry_phases']:
                    try:
                        position['entry_phases'] = json.loads(position['entry_phases'])
                    except:
                        position['entry_phases'] = []
                else:
                    position['entry_phases'] = []
                
                return position
            return None
    
    def create_or_update_position(self, symbol: str, shares_traded: float, price: float, 
                                phase: str, account_value: float, wyckoff_score: float, 
                                account_type: str, is_addition: bool = False) -> bool:
        """Create new position or update existing position"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.database.db_path) as conn:
                existing_position = self.get_enhanced_position(symbol)
                
                if existing_position and is_addition:
                    # Update existing position
                    new_shares = existing_position['total_shares'] + shares_traded
                    new_invested = existing_position['total_invested'] + (shares_traded * price)
                    new_avg_cost = new_invested / new_shares if new_shares > 0 else 0
                    
                    # Update allocation percentage
                    new_allocation_pct = new_invested / existing_position['target_position_size']
                    new_allocation_pct = min(new_allocation_pct, 1.0)  # Cap at 100%
                    
                    # Update entry phases
                    entry_phases = existing_position['entry_phases'].copy()
                    if phase not in entry_phases:
                        entry_phases.append(phase)
                    
                    # Determine position status
                    if new_allocation_pct >= 0.95:  # 95% or more = complete
                        position_status = 'COMPLETE'
                    else:
                        position_status = 'BUILDING'
                    
                    conn.execute('''
                        UPDATE positions_enhanced 
                        SET total_shares = ?, avg_cost = ?, total_invested = ?,
                            current_allocation_pct = ?, entry_phases = ?, addition_count = addition_count + 1,
                            last_addition_date = ?, position_status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ? AND bot_id = ?
                    ''', (
                        new_shares, new_avg_cost, new_invested, new_allocation_pct,
                        json.dumps(entry_phases), today, position_status, symbol, self.database.bot_id
                    ))
                    
                    # Log position event
                    conn.execute('''
                        INSERT INTO position_events (
                            symbol, event_type, event_date, wyckoff_phase, shares_traded, price,
                            allocation_before, allocation_after, reasoning, bot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, 'ADDITION', today, phase, shares_traded, price,
                        existing_position['current_allocation_pct'], new_allocation_pct,
                        f"Added {shares_traded:.5f} shares in {phase} phase", self.database.bot_id
                    ))
                    
                    self.logger.info(f"📈 Position Updated: {symbol} now {new_allocation_pct:.1%} of target")
                    
                else:
                    # Create new position
                    target_size = self.calculate_target_position_size(account_value, wyckoff_score)
                    
                    if target_size == 0:
                        self.logger.warning(f"⚠️ Target position size is 0 for {symbol}")
                        return False
                    
                    sizing_config = self.get_account_sizing_config(account_value)
                    max_additions = sizing_config['max_additions']
                    
                    initial_invested = shares_traded * price
                    initial_allocation_pct = initial_invested / target_size
                    initial_allocation_pct = min(initial_allocation_pct, 1.0)
                    
                    conn.execute('''
                        INSERT INTO positions_enhanced (
                            symbol, target_position_size, current_allocation_pct, total_shares,
                            avg_cost, total_invested, entry_phases, addition_count, max_additions,
                            first_entry_date, last_addition_date, account_type, wyckoff_score,
                            position_status, bot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, target_size, initial_allocation_pct, shares_traded, price,
                        initial_invested, json.dumps([phase]), 1, max_additions, today, today,
                        account_type, wyckoff_score, 'BUILDING', self.database.bot_id
                    ))
                    
                    # Log position event
                    conn.execute('''
                        INSERT INTO position_events (
                            symbol, event_type, event_date, wyckoff_phase, shares_traded, price,
                            allocation_before, allocation_after, reasoning, bot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, 'INITIAL_ENTRY', today, phase, shares_traded, price,
                        0.0, initial_allocation_pct,
                        f"Initial {phase} entry: {initial_allocation_pct:.1%} of target position", 
                        self.database.bot_id
                    ))
                    
                    self.logger.info(f"🎯 New Position: {symbol} target ${target_size:.2f}, started with {initial_allocation_pct:.1%}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating/updating position for {symbol}: {e}")
            return False
    
    def check_scaling_out_opportunities(self, wb_client) -> List[Dict]:
        """Check all positions for scaling out opportunities"""
        scaling_opportunities = []
        
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                positions = conn.execute('''
                    SELECT symbol, total_shares, avg_cost, position_status
                    FROM positions_enhanced
                    WHERE total_shares > 0 AND bot_id = ?
                    ORDER BY symbol
                ''', (self.database.bot_id,)).fetchall()
            
            if not positions:
                return scaling_opportunities
            
            self.logger.info(f"📊 Checking {len(positions)} positions for scaling opportunities...")
            
            for symbol, shares, avg_cost, status in positions:
                try:
                    # Get current price
                    quote_data = wb_client.get_quote(symbol)
                    if not quote_data or 'close' not in quote_data:
                        continue
                    
                    current_price = float(quote_data['close'])
                    gain_pct = (current_price - avg_cost) / avg_cost
                    
                    # Check profit-based scaling
                    for target in self.scaling_out_config['profit_targets']:
                        if gain_pct >= target['gain_pct']:
                            # Check if we've already scaled at this level
                            if not self._already_scaled_at_level(symbol, target['gain_pct']):
                                shares_to_sell = shares * target['sell_pct']
                                
                                scaling_opportunities.append({
                                    'symbol': symbol,
                                    'shares_to_sell': shares_to_sell,
                                    'current_price': current_price,
                                    'gain_pct': gain_pct,
                                    'reason': f"PROFIT_{target['gain_pct']*100:.0f}PCT",
                                    'description': target['description'],
                                    'remaining_shares': shares - shares_to_sell
                                })
                                break  # Only one scaling action per position per run
                
                except Exception as e:
                    self.logger.error(f"Error checking scaling for {symbol}: {e}")
                    continue
            
            return scaling_opportunities
            
        except Exception as e:
            self.logger.error(f"Error checking scaling opportunities: {e}")
            return scaling_opportunities
    
    def _already_scaled_at_level(self, symbol: str, gain_pct: float) -> bool:
        """Check if we've already scaled out at this gain level"""
        with sqlite3.connect(self.database.db_path) as conn:
            result = conn.execute('''
                SELECT COUNT(*) FROM partial_sales
                WHERE symbol = ? AND sale_reason = ? AND bot_id = ?
            ''', (symbol, f"PROFIT_{gain_pct*100:.0f}PCT", self.database.bot_id)).fetchone()
            
            return result[0] > 0
    
    def execute_partial_sale(self, symbol: str, shares_to_sell: float, current_price: float,
                           reason: str, description: str, wb_client, account_manager) -> bool:
        """Execute a partial sale (scaling out)"""
        try:
            # Get position to find the right account
            position = self.get_enhanced_position(symbol)
            if not position:
                self.logger.error(f"❌ No position found for {symbol}")
                return False
            
            # Find the account that holds this position
            enabled_accounts = account_manager.get_enabled_accounts()
            target_account = next((acc for acc in enabled_accounts 
                                 if acc.account_type == position['account_type']), None)
            
            if not target_account:
                self.logger.error(f"❌ Could not find account for {symbol}")
                return False
            
            # Switch to the trading account
            if not account_manager.switch_to_account(target_account):
                self.logger.error(f"❌ Failed to switch to account for {symbol}")
                return False
            
            self.logger.info(f"💰 Scaling Out: Selling {shares_to_sell:.5f} shares of {symbol} - {description}")
            
            # Place market sell order
            order_result = wb_client.place_order(
                stock=symbol,
                price=0,  # Market price
                action='SELL',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_sell,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                remaining_shares = position['total_shares'] - shares_to_sell
                
                # Calculate profit
                profit_per_share = current_price - position['avg_cost']
                profit_amount = profit_per_share * shares_to_sell
                gain_pct = profit_per_share / position['avg_cost']
                
                # Log partial sale
                today = datetime.now().strftime('%Y-%m-%d')
                with sqlite3.connect(self.database.db_path) as conn:
                    conn.execute('''
                        INSERT INTO partial_sales (
                            symbol, sale_date, shares_sold, sale_price, sale_reason,
                            remaining_shares, gain_pct, profit_amount, bot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, today, shares_to_sell, current_price, reason,
                        remaining_shares, gain_pct, profit_amount, self.database.bot_id
                    ))
                    
                    # Update position
                    new_invested = remaining_shares * position['avg_cost']
                    new_allocation_pct = new_invested / position['target_position_size'] if position['target_position_size'] > 0 else 0
                    
                    if remaining_shares > 0:
                        conn.execute('''
                            UPDATE positions_enhanced
                            SET total_shares = ?, total_invested = ?, current_allocation_pct = ?,
                                position_status = 'SCALING_OUT', updated_at = CURRENT_TIMESTAMP
                            WHERE symbol = ? AND bot_id = ?
                        ''', (remaining_shares, new_invested, new_allocation_pct, symbol, self.database.bot_id))
                    else:
                        # Position completely closed
                        conn.execute('''
                            DELETE FROM positions_enhanced
                            WHERE symbol = ? AND bot_id = ?
                        ''', (symbol, self.database.bot_id))
                    
                    # Log position event
                    conn.execute('''
                        INSERT INTO position_events (
                            symbol, event_type, event_date, wyckoff_phase, shares_traded, price,
                            allocation_before, allocation_after, reasoning, bot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, 'PARTIAL_SALE', today, 'SCALING_OUT', -shares_to_sell, current_price,
                        position['current_allocation_pct'], new_allocation_pct,
                        f"Scaled out {shares_to_sell:.5f} shares: {description}", self.database.bot_id
                    ))
                
                self.logger.info(f"✅ Partial sale executed: {symbol} - Order ID: {order_id}")
                self.logger.info(f"💰 Profit realized: ${profit_amount:.2f} ({gain_pct*100:.1f}% gain)")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Partial sale failed for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing partial sale for {symbol}: {e}")
            return False


class EnhancedFractionalTradingBot:
    """Enhanced trading bot with fractional share position building and scaling"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        self.position_manager = None
        
        # Enhanced trading configuration
        self.min_signal_strength = 0.4  # Lower threshold for position building
        self.buy_phases = ['ST', 'SOS', 'LPS', 'BU']  # Phases for position building
        self.sell_phases = ['PS', 'SC']  # Distribution phases for scaling out
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup enhanced logging"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = logs_dir / f"fractional_trading_bot_{timestamp}.log"
        
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
        self.logger.info("🏗️ ENHANCED FRACTIONAL SHARE POSITION BUILDING BOT")
        self.logger.info(f"📝 Log: {log_filename.name}")
        self.logger.info("📊 Features: Position Building | Fractional Shares | Scaling Out")
    
    def initialize_systems(self) -> bool:
        """Initialize all systems including fractional position manager"""
        try:
            self.logger.info("🔧 Initializing fractional position building systems...")
            
            # Initialize main system
            self.main_system = MainSystem()
            
            # Initialize Wyckoff strategy
            self.wyckoff_strategy = WyckoffPnFStrategy()
            
            # Initialize database (reuse existing database manager)
            from wyckoff_trading_bot import EnhancedTradingDatabase
            self.database = EnhancedTradingDatabase()
            
            # Initialize fractional position manager
            self.position_manager = FractionalPositionManager(self.database, self.logger)
            self.position_manager.create_enhanced_database_schema()
            
            self.logger.info("✅ Fractional position building systems initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize systems: {e}")
            return False
    
    def run_fractional_trading_cycle(self) -> Tuple[int, int, int]:
        """Run enhanced trading cycle with position building and scaling"""
        trades_executed = 0
        scaling_actions = 0
        errors = 0
        
        try:
            self.logger.info("🔄 Starting fractional position building cycle...")
            
            # Get enabled accounts
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            if not enabled_accounts:
                self.logger.error("❌ No enabled accounts found")
                return 0, 0, 1
            
            # Calculate total account value for position sizing
            total_account_value = sum(acc.net_liquidation for acc in enabled_accounts)
            
            self.logger.info(f"💼 Total account value: ${total_account_value:.2f}")
            
            # STEP 1: Check for scaling out opportunities
            self.logger.info("💰 STEP 1: Checking for profit-taking opportunities...")
            
            scaling_opportunities = self.position_manager.check_scaling_out_opportunities(self.main_system.wb)
            
            for opportunity in scaling_opportunities[:5]:  # Limit to 5 scaling actions per run
                if self.position_manager.execute_partial_sale(
                    symbol=opportunity['symbol'],
                    shares_to_sell=opportunity['shares_to_sell'],
                    current_price=opportunity['current_price'],
                    reason=opportunity['reason'],
                    description=opportunity['description'],
                    wb_client=self.main_system.wb,
                    account_manager=self.main_system.account_manager
                ):
                    scaling_actions += 1
                else:
                    errors += 1
            
            # STEP 2: Scan for Wyckoff signals
            self.logger.info("🔍 STEP 2: Scanning for position building opportunities...")
            
            all_signals = self.wyckoff_strategy.scan_market()
            
            if all_signals:
                # Filter for position building signals
                position_building_signals = []
                
                for signal in all_signals:
                    if (signal.phase in self.buy_phases and 
                        signal.strength >= self.min_signal_strength and
                        signal.volume_confirmation):
                        
                        # Check if we should build/add to position
                        should_add, reason, addition_pct = self.position_manager.should_add_to_position(
                            signal.symbol, signal.phase, signal.combined_score or signal.strength
                        )
                        
                        if should_add:
                            signal.addition_info = {
                                'reason': reason,
                                'addition_pct': addition_pct,
                                'is_new_position': reason == "NEW_POSITION"
                            }
                            position_building_signals.append(signal)
                            
                            self.logger.info(f"🎯 Position Building Signal: {signal.symbol} ({signal.phase}) - {reason}")
                
                # STEP 3: Execute position building trades
                if position_building_signals:
                    self.logger.info(f"🏗️ STEP 3: Executing {len(position_building_signals)} position building trades...")
                    
                    for signal in position_building_signals[:10]:  # Limit to 10 trades per run
                        if self.execute_fractional_position_trade(signal, enabled_accounts, total_account_value):
                            trades_executed += 1
                        else:
                            errors += 1
                else:
                    self.logger.info("📭 No position building opportunities found")
            else:
                self.logger.info("📭 No Wyckoff signals found")
            
            return trades_executed, scaling_actions, errors
            
        except Exception as e:
            self.logger.error(f"❌ Error in fractional trading cycle: {e}")
            return trades_executed, scaling_actions, errors + 1
    
    def execute_fractional_position_trade(self, signal: WyckoffSignal, enabled_accounts: List, 
                                        total_account_value: float) -> bool:
        """Execute fractional position building trade"""
        try:
            # Determine trade amount based on addition info
            is_new_position = signal.addition_info['is_new_position']
            addition_pct = signal.addition_info['addition_pct']
            
            if is_new_position:
                # Calculate initial position size
                target_size = self.position_manager.calculate_target_position_size(
                    total_account_value, signal.combined_score or signal.strength
                )
                
                if target_size == 0:
                    self.logger.warning(f"⚠️ No target size calculated for {signal.symbol}")
                    return False
                
                # Use phase allocation for initial entry
                phase_config = self.position_manager.phase_allocation_config.get(signal.phase, {})
                initial_allocation = phase_config.get('initial_allocation', 0.25)
                trade_amount = target_size * initial_allocation
                
                self.logger.info(f"🎯 NEW POSITION: {signal.symbol} - ${trade_amount:.2f} ({initial_allocation:.0%} of ${target_size:.2f} target)")
                
            else:
                # Adding to existing position
                existing_position = self.position_manager.get_enhanced_position(signal.symbol)
                if not existing_position:
                    self.logger.error(f"❌ Expected existing position for {signal.symbol} not found")
                    return False
                
                trade_amount = existing_position['target_position_size'] * addition_pct
                
                self.logger.info(f"📈 POSITION ADDITION: {signal.symbol} - ${trade_amount:.2f} ({addition_pct:.0%} addition)")
            
            # Find best account for trade
            best_account = self.find_best_account_for_trade(enabled_accounts, trade_amount)
            if not best_account:
                self.logger.warning(f"⚠️ No suitable account found for {signal.symbol} trade")
                return False
            
            # Switch to trading account
            if not self.main_system.account_manager.switch_to_account(best_account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            # Get current price and calculate shares
            quote_data = self.main_system.wb.get_quote(signal.symbol)
            if not quote_data or 'close' not in quote_data:
                self.logger.error(f"❌ Could not get quote for {signal.symbol}")
                return False
            
            current_price = float(quote_data['close'])
            shares_to_buy = trade_amount / current_price
            shares_to_buy = round(shares_to_buy, 5)  # Webull supports 5 decimal places
            
            if shares_to_buy < 0.00001:
                self.logger.warning(f"⚠️ Share amount too small for {signal.symbol}: {shares_to_buy}")
                return False
            
            # Execute the trade
            self.logger.info(f"💰 Executing: {shares_to_buy:.5f} shares of {signal.symbol} at ~${current_price:.2f}")
            
            order_result = self.main_system.wb.place_order(
                stock=signal.symbol,
                price=0,  # Market order
                action='BUY',
                orderType='MKT',
                enforce='DAY',
                quant=shares_to_buy,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                
                # Log trade in original system
                self.database.log_trade(
                    symbol=signal.symbol,
                    action='BUY',
                    quantity=shares_to_buy,
                    price=current_price,
                    signal_phase=signal.phase,
                    signal_strength=signal.strength,
                    account_type=best_account.account_type,
                    order_id=order_id
                )
                
                # Update enhanced position tracking
                self.position_manager.create_or_update_position(
                    symbol=signal.symbol,
                    shares_traded=shares_to_buy,
                    price=current_price,
                    phase=signal.phase,
                    account_value=total_account_value,
                    wyckoff_score=signal.combined_score or signal.strength,
                    account_type=best_account.account_type,
                    is_addition=not is_new_position
                )
                
                self.logger.info(f"✅ Fractional trade executed: {signal.symbol} - Order ID: {order_id}")
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Trade failed for {signal.symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing fractional trade for {signal.symbol}: {e}")
            return False
    
    def find_best_account_for_trade(self, enabled_accounts: List, trade_amount: float):
        """Find the best account for a trade based on available funds"""
        # Sort accounts by available cash (descending)
        sorted_accounts = sorted(enabled_accounts, key=lambda x: x.settled_funds, reverse=True)
        
        # Find account with sufficient funds
        for account in sorted_accounts:
            if account.settled_funds >= trade_amount + 50:  # Keep $50 buffer
                return account
        
        return None
    
    def run(self) -> bool:
        """Main execution for fractional trading bot"""
        try:
            self.logger.info("🚀 Starting Enhanced Fractional Share Trading Bot")
            
            # Initialize systems
            if not self.initialize_systems():
                return False
            
            # Authenticate
            if not self.main_system.run():
                return False
            
            # Run fractional trading cycle
            trades, scaling, errors = self.run_fractional_trading_cycle()
            
            # Log summary
            total_actions = trades + scaling
            
            self.logger.info("📊 FRACTIONAL TRADING SESSION SUMMARY")
            self.logger.info(f"   Position Building Trades: {trades}")
            self.logger.info(f"   Scaling Out Actions: {scaling}")
            self.logger.info(f"   Total Actions: {total_actions}")
            self.logger.info(f"   Errors: {errors}")
            
            if total_actions > 0:
                self.logger.info("✅ Fractional trading bot completed successfully with actions")
            elif errors == 0:
                self.logger.info("✅ Fractional trading bot completed successfully (no actions needed)")
            else:
                self.logger.warning("⚠️ Fractional trading bot completed with errors")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in fractional trading bot: {e}")
            return False
        
        finally:
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point for enhanced fractional trading bot"""
    print("🏗️ Enhanced Fractional Share Position Building Bot Starting...")
    
    bot = EnhancedFractionalTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Fractional trading bot completed successfully!")
        sys.exit(0)
    else:
        print("❌ Fractional trading bot failed! Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()