#!/usr/bin/env python3
"""
REFACTORED ENHANCED FRACTIONAL POSITION BUILDING SYSTEM
Removed duplicated position sizing logic and consolidated regime-aware functionality
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

# Import market regime analyzer - OPTIMIZATION 2
from market_regime_analyzer import EnhancedMarketRegimeAnalyzer, RegimeAwarePositionSizer, MarketRegimeData
from position_sizing_optimizer import DynamicPositionSizer

# OPTIMIZATION 4: Enhanced Exit Strategy System
try:
    from enhanced_exit_strategy import EnhancedExitStrategyManager
    ENHANCED_EXIT_STRATEGY_AVAILABLE = True
    print("✅ Enhanced Exit Strategy System available")
except ImportError:
    ENHANCED_EXIT_STRATEGY_AVAILABLE = False
    print("⚠️ Enhanced Exit Strategy not available - using base system")

from strategies.wyckoff.wyckoff import WyckoffPnFStrategy, WyckoffSignal

# ENHANCEMENT: Multi-timeframe signal quality import - Strategic Improvement 5 📈
try:
    from strategies.wyckoff.multi_timeframe_analyzer import (
        EnhancedMultiTimeframeWyckoffAnalyzer,
        filter_signals_by_quality,
        MultiTimeframeSignal
    )
    SIGNAL_QUALITY_ENHANCEMENT = True
except ImportError:
    SIGNAL_QUALITY_ENHANCEMENT = False

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


class RegimeAwareConfigManager:
    """CONSOLIDATED: Manages regime-aware configuration and position sizing"""
    
    def __init__(self, logger):
        self.logger = logger
        self.last_update = None
        self.cached_config = None
        self.regime_analyzer = None
        
        # Initialize regime analyzer
        try:
            self.regime_analyzer = EnhancedMarketRegimeAnalyzer(self.logger)
            self.logger.info("📊 Regime analyzer initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Regime analyzer failed to initialize: {e}")
    
    def get_dynamic_config(self, account_manager) -> Dict:
        """Get dynamic configuration based on real account values WITH REGIME ADAPTATION"""
        try:
            enabled_accounts = account_manager.get_enabled_accounts()
            if not enabled_accounts:
                return self._get_fallback_config()
            
            total_value = sum(acc.net_liquidation for acc in enabled_accounts)
            total_cash = sum(acc.settled_funds for acc in enabled_accounts)
            
            self.logger.info(f"💰 Real Account Values - Total: ${total_value:.2f}, Cash: ${total_cash:.2f}")
            
            # Get current market regime
            regime_data = None
            regime_multiplier = 0.7  # Conservative fallback
            
            if self.regime_analyzer:
                try:
                    regime_data = self.regime_analyzer.analyze_market_regime()
                    regime_multiplier = regime_data.position_size_multiplier
                    
                    self.logger.info(f"📊 Regime Analysis: {regime_data.trend_regime} trend, {regime_data.volatility_regime} volatility")
                    self.logger.info(f"📊 Regime Position Multiplier: {regime_multiplier:.2f}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Regime analysis failed, using conservative multiplier: {e}")
            
            # Calculate regime-aware parameters
            config = self._calculate_regime_aware_parameters(
                total_value, total_cash, enabled_accounts, regime_multiplier, regime_data
            )
            
            self.cached_config = config
            self.last_update = datetime.now()
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error getting dynamic config: {e}")
            return self._get_fallback_config()
    
    def _calculate_regime_aware_parameters(self, total_value: float, total_cash: float, 
                                         accounts: List, regime_multiplier: float, 
                                         regime_data: Optional[MarketRegimeData]) -> Dict:
        """CONSOLIDATED: Calculate regime-aware parameters (single implementation)"""
        
        # Find the account with the most cash for primary trading
        max_cash_available = max(acc.settled_funds for acc in accounts) if accounts else total_cash
        
        # Base conservative position sizing
        if total_cash < 100:
            base_position_pct = 0.10
            max_positions = 3
            min_balance_pct = 0.25
        elif total_cash < 300:
            base_position_pct = 0.12
            max_positions = 6
            min_balance_pct = 0.15
        elif total_cash < 500:
            base_position_pct = 0.15
            max_positions = 12
            min_balance_pct = 0.12
        else:
            base_position_pct = 0.25
            max_positions = 25
            min_balance_pct = 0.10
        
        # Apply regime adjustments
        regime_adjusted_position_pct = base_position_pct * regime_multiplier
        
        # Adjust max positions based on regime
        if regime_data:
            if regime_data.trend_regime == 'BEAR':
                max_positions = max(1, max_positions - 1)
            elif regime_data.volatility_regime in ['HIGH', 'CRISIS']:
                max_positions = max(1, max_positions - 1)
            
            if regime_data.regime_confidence < 0.5:
                min_balance_pct += 0.10
        
        # Calculate position size with regime adjustment
        base_position_size = min(
            total_cash * regime_adjusted_position_pct,
            max_cash_available * 0.4,
            15.0
        )
        
        min_balance_preserve = total_cash * min_balance_pct
        
        if base_position_size < 5.0:
            base_position_size = min(5.0, total_cash * 0.3)
        
        wyckoff_phases = self._get_regime_aware_wyckoff_phases(regime_data)
        profit_targets = self._get_regime_aware_profit_targets(regime_data)
        
        return {
            'total_value': total_value,
            'total_cash': total_cash,
            'max_cash_per_account': max_cash_available,
            'base_position_size': base_position_size,
            'base_position_pct': regime_adjusted_position_pct,
            'min_balance_preserve': min_balance_preserve,
            'max_positions': max_positions,
            'num_accounts': len(accounts),
            'wyckoff_phases': wyckoff_phases,
            'profit_targets': profit_targets,
            'calculated_at': datetime.now().isoformat(),
            'regime_data': regime_data,
            'regime_multiplier': regime_multiplier,
            'max_position_per_account': max_cash_available * 0.4,
            'min_cash_buffer_per_account': 15.0
        }
    
    def _get_regime_aware_wyckoff_phases(self, regime_data: Optional[MarketRegimeData]) -> Dict:
        """CONSOLIDATED: Adjust Wyckoff phase allocations based on regime"""
        
        base_phases = {
            'ST': {'initial_allocation': 0.40, 'allow_additions': False, 'max_total_allocation': 0.40},
            'SOS': {'initial_allocation': 0.50, 'allow_additions': True, 'max_total_allocation': 0.80},
            'LPS': {'initial_allocation': 0.35, 'allow_additions': True, 'max_total_allocation': 0.70},
            'BU': {'initial_allocation': 0.30, 'allow_additions': True, 'max_total_allocation': 0.60},
            'Creek': {'initial_allocation': 0.0, 'allow_additions': False, 'max_total_allocation': 0.0}
        }
        
        if not regime_data:
            return base_phases
        
        if regime_data.trend_regime == 'BULL':
            base_phases['SOS']['initial_allocation'] = 0.60
            base_phases['SOS']['max_total_allocation'] = 0.90
            base_phases['LPS']['initial_allocation'] = 0.45
        elif regime_data.trend_regime == 'BEAR':
            base_phases['ST']['initial_allocation'] = 0.30
            base_phases['SOS']['initial_allocation'] = 0.35
            base_phases['SOS']['max_total_allocation'] = 0.60
            base_phases['LPS']['max_total_allocation'] = 0.50
        
        if regime_data.volatility_regime in ['HIGH', 'CRISIS']:
            for phase in base_phases.values():
                phase['initial_allocation'] *= 0.8
                phase['max_total_allocation'] *= 0.8
        
        return base_phases
    
    def _get_regime_aware_profit_targets(self, regime_data: Optional[MarketRegimeData]) -> List[Dict]:
        """CONSOLIDATED: Adjust profit targets based on regime"""
        
        base_targets = [
            {'gain_pct': 0.06, 'sell_pct': 0.15, 'description': '6% gain: Take 15% profit'},
            {'gain_pct': 0.12, 'sell_pct': 0.20, 'description': '12% gain: Take 20% more'},
            {'gain_pct': 0.20, 'sell_pct': 0.25, 'description': '20% gain: Take 25% more'},
            {'gain_pct': 0.30, 'sell_pct': 0.40, 'description': '30% gain: Take final 40%'}
        ]
        
        if not regime_data:
            return base_targets
        
        if regime_data.volatility_regime in ['HIGH', 'CRISIS']:
            for target in base_targets:
                target['gain_pct'] *= 0.75
                target['sell_pct'] *= 1.1
        elif regime_data.volatility_regime == 'LOW':
            for target in base_targets:
                target['gain_pct'] *= 1.25
        
        return base_targets
    
    def _get_fallback_config(self) -> Dict:
        """Conservative fallback configuration"""
        return {
            'total_value': 100.0,
            'total_cash': 100.0,
            'base_position_size': 8.0,
            'base_position_pct': 0.08,
            'min_balance_preserve': 40.0,
            'max_positions': 2,
            'num_accounts': 2,
            'wyckoff_phases': {
                'ST': {'initial_allocation': 0.40, 'allow_additions': False, 'max_total_allocation': 0.40},
                'SOS': {'initial_allocation': 0.50, 'allow_additions': True, 'max_total_allocation': 0.80},
                'LPS': {'initial_allocation': 0.35, 'allow_additions': True, 'max_total_allocation': 0.70},
                'BU': {'initial_allocation': 0.30, 'allow_additions': True, 'max_total_allocation': 0.60},
                'Creek': {'initial_allocation': 0.0, 'allow_additions': False, 'max_total_allocation': 0.0}
            },
            'profit_targets': [
                {'gain_pct': 0.06, 'sell_pct': 0.15, 'description': '6% gain: Take 15% profit'},
                {'gain_pct': 0.12, 'sell_pct': 0.20, 'description': '12% gain: Take 20% more'},
                {'gain_pct': 0.20, 'sell_pct': 0.25, 'description': '20% gain: Take 25% more'}
            ],
            'calculated_at': datetime.now().isoformat(),
            'max_position_per_account': 40.0,
            'min_cash_buffer_per_account': 15.0
        }


class RealAccountDayTradeChecker:
    """Real account day trade checking using Webull API"""
    
    def __init__(self, logger):
        self.logger = logger
        self.trade_cache = {}
        self.last_cache_update = None
    
    def get_actual_todays_trades(self, wb_client, symbol: str = None) -> List[Dict]:
        """Get TODAY'S trades from actual Webull account"""
        today = datetime.now().strftime('%Y-%m-%d')
        cache_key = f"{today}_{symbol or 'ALL'}"
        
        # Use cache if updated within last 5 minutes
        if (self.last_cache_update and 
            cache_key in self.trade_cache and
            (datetime.now() - self.last_cache_update).total_seconds() < 300):
            return self.trade_cache[cache_key]
        
        try:
            self.logger.debug(f"🔍 Fetching real account trades for {symbol or 'ALL'}")
            
            orders = []
            
            try:
                current_orders = wb_client.get_current_orders()
                history_orders = wb_client.get_history_orders(status='All', count=50)
                
                all_orders = []
                if current_orders:
                    all_orders.extend(current_orders)
                if history_orders:
                    all_orders.extend(history_orders)
                
                # Filter for today's FILLED trades
                today_orders = []
                for order in all_orders:
                    try:
                        order_date = order.get('createTime', order.get('orderDate', order.get('time', '')))
                        
                        if order_date:
                            if isinstance(order_date, str):
                                if len(order_date) > 10:
                                    if order_date.isdigit():
                                        order_dt = datetime.fromtimestamp(int(order_date) / 1000)
                                    else:
                                        try:
                                            order_dt = datetime.fromisoformat(order_date.replace('Z', '+00:00'))
                                        except:
                                            order_dt = datetime.strptime(order_date[:10], '%Y-%m-%d')
                                else:
                                    order_dt = datetime.strptime(order_date, '%Y-%m-%d')
                            else:
                                order_dt = datetime.fromtimestamp(order_date / 1000 if order_date > 1000000000000 else order_date)
                            
                            order_date_str = order_dt.strftime('%Y-%m-%d')
                            
                            order_status = order.get('status', order.get('orderStatus', '')).upper()
                            if (order_date_str == today and 
                                order_status in ['FILLED', 'PARTIALLY_FILLED', 'EXECUTED']):
                                
                                parsed_order = {
                                    'symbol': order.get('ticker', order.get('symbol', order.get('stock', ''))),
                                    'action': order.get('action', order.get('side', '')).upper(),
                                    'quantity': float(order.get('filledQuantity', 
                                                     order.get('quantity', 
                                                     order.get('qty', 0)))),
                                    'price': float(order.get('avgFilledPrice', 
                                                   order.get('price', 
                                                   order.get('fillPrice', 0)))),
                                    'time': order_date_str,
                                    'order_id': order.get('orderId', order.get('id', '')),
                                    'status': order_status
                                }
                                
                                if parsed_order['action'] in ['BUY', 'B']:
                                    parsed_order['action'] = 'BUY'
                                elif parsed_order['action'] in ['SELL', 'S']:
                                    parsed_order['action'] = 'SELL'
                                
                                if not symbol or parsed_order['symbol'] == symbol:
                                    today_orders.append(parsed_order)
                                    
                    except Exception as e:
                        self.logger.debug(f"Error parsing order: {e}")
                        continue
                
                orders = today_orders
                
            except AttributeError as e:
                self.logger.warning(f"⚠️ Webull API method not available: {e}")
                orders = []
            except Exception as e:
                self.logger.warning(f"⚠️ Could not get Webull order data: {e}")
                orders = []
            
            # Cache the results
            self.trade_cache[cache_key] = orders
            self.last_cache_update = datetime.now()
            
            if orders:
                self.logger.debug(f"📊 Found {len(orders)} real trades today for {symbol or 'ALL'}")
                for order in orders:
                    self.logger.debug(f"   {order['action']} {order['quantity']:.5f} {order['symbol']} @ ${order['price']:.2f}")
            
            return orders
            
        except Exception as e:
            self.logger.error(f"❌ Error getting actual trades: {e}")
            return []
    
    def detect_manual_trades(self, wb_client, database, symbol: str, account_manager) -> bool:
        """Detect manual trades with proper account handling"""
        try:
            db_positions = database.get_position(symbol)
            if not db_positions:
                return False
            
            if isinstance(db_positions, list):
                db_total_shares = sum(pos['total_shares'] for pos in db_positions)
            else:
                db_total_shares = db_positions['total_shares']
            
            # Get REAL position from ALL accounts
            real_total_shares = 0.0
            enabled_accounts = account_manager.get_enabled_accounts()
            for account in enabled_accounts:
                for position in account.positions:
                    if position['symbol'] == symbol:
                        real_total_shares += position['quantity']
            
            if abs(real_total_shares - db_total_shares) > 0.00001:
                self.logger.warning(f"Position mismatch for {symbol}: Real={real_total_shares:.5f}, DB={db_total_shares:.5f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting manual trades for {symbol}: {e}")
            return False
    
    def comprehensive_day_trade_check(self, wb_client, database, symbol: str, action: str, 
                                    account_manager, emergency: bool = False) -> DayTradeCheckResult:
        """Comprehensive day trade check"""
        
        # Check 1: Database trades
        db_trades = database.get_todays_trades(symbol)
        db_day_trade = database.would_create_day_trade(symbol, action)
        
        # Check 2: Actual account trades
        actual_trades = self.get_actual_todays_trades(wb_client, symbol)
        
        # Analyze actual trades for day trade potential
        actual_day_trade = False
        if actual_trades:
            buys_today = sum(1 for trade in actual_trades if trade['action'] in ['BUY', 'buy'])
            sells_today = sum(1 for trade in actual_trades if trade['action'] in ['SELL', 'sell'])
            
            if action.upper() == 'SELL' and buys_today > 0:
                actual_day_trade = True
            elif action.upper() == 'BUY' and sells_today > 0:
                actual_day_trade = True
        
        # Check 3: Manual trades detection
        manual_trades_detected = self.detect_manual_trades(wb_client, database, symbol, account_manager)
        
        # Determine final recommendation
        would_be_day_trade = db_day_trade or actual_day_trade
        
        if emergency:
            recommendation = 'EMERGENCY_OVERRIDE'
            details = f"Emergency override: allowing potential day trade for {symbol}"
        elif would_be_day_trade or manual_trades_detected:
            recommendation = 'BLOCK'
            details = f"Day trade detected - DB: {db_day_trade}, Actual: {actual_day_trade}, Manual: {manual_trades_detected}"
        else:
            recommendation = 'ALLOW'
            details = f"No day trade violation detected for {symbol}"
        
        result = DayTradeCheckResult(
            symbol=symbol,
            action=action,
            would_be_day_trade=would_be_day_trade,
            db_trades_today=db_trades,
            actual_trades_today=actual_trades,
            manual_trades_detected=manual_trades_detected,
            recommendation=recommendation,
            details=details
        )
        
        # Log the analysis
        if would_be_day_trade or manual_trades_detected:
            self.logger.warning(f"🚨 DAY TRADE CHECK: {symbol} {action}")
            self.logger.warning(f"   DB trades today: {len(db_trades)}")
            self.logger.warning(f"   Actual trades today: {len(actual_trades)}")
            self.logger.warning(f"   Manual trades detected: {manual_trades_detected}")
            self.logger.warning(f"   Recommendation: {recommendation}")
        else:
            self.logger.debug(f"✅ Day trade check passed: {symbol} {action}")
        
        return result


class EnhancedTradingDatabase:
    """Enhanced database manager with comprehensive tracking"""
    
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
                    day_trade_check TEXT,
                    bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
                    trade_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced positions table
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
            
            # Add indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_date_symbol ON trades(date, symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_enhanced_bot_id ON positions_enhanced(bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_day_trade_checks_date ON day_trade_checks(check_date, symbol)')
    
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
    
    def update_position(self, symbol: str, shares: float, cost: float, account_type: str, 
                       entry_phase: str = None, entry_strength: float = None):
        """Update position tracking with enhanced data for a specific account"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
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
                      'day_trade_check', 'bot_id', 'trade_datetime', 'created_at']
            
            return [dict(zip(columns, row)) for row in results]
    
    def would_create_day_trade(self, symbol: str, action: str) -> bool:
        """Check if this trade would create a day trade (DATABASE ONLY)"""
        today_trades = self.get_todays_trades(symbol)
        
        if not today_trades:
            return False
        
        buys_today = sum(1 for trade in today_trades if trade['action'] == 'BUY')
        sells_today = sum(1 for trade in today_trades if trade['action'] == 'SELL')
        
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
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signals_found, trades_executed, wyckoff_sells, profit_scales, emergency_exits,
                day_trades_blocked, errors, portfolio_value, available_cash, emergency_mode, 
                market_condition, portfolio_drawdown_pct, status, log_details, self.bot_id
            ))


class SmartFractionalPositionManager:
    """REFACTORED: Enhanced position manager with consolidated logic"""
    
    def __init__(self, database, config_manager, logger):
        self.database = database
        self.config_manager = config_manager  # Now uses RegimeAwareConfigManager
        self.logger = logger
        self.current_config = None
        
        # Add dynamic position sizer
        position_config = {
            'base_account_allocation': 0.25,
            'max_position_allocation': 0.15,
            'vix_threshold': 25.0,
            'vix_reduction_factor': 0.5,
            'sector_boost_factor': 0.25
        }
        self.position_sizer = DynamicPositionSizer(position_config)
    
    def update_config(self, account_manager):
        """Update configuration based on current account values"""
        self.current_config = self.config_manager.get_dynamic_config(account_manager)
        return self.current_config
    
    def get_position_size_for_signal(self, signal: WyckoffSignal, target_account) -> float:
        """Calculate position size WITH REGIME ADAPTATION"""
        if not self.current_config:
            return 5.0  # Very conservative fallback
        
        # Get account-specific available cash
        account_cash = target_account.settled_funds
        
        if account_cash < 20.0:  # Need minimum $20 to trade
            self.logger.warning(f"⚠️ Insufficient cash in {target_account.account_type}: ${account_cash:.2f}")
            return 0.0
        
        # Use dynamic position sizing
        sizing_result = self.position_sizer.calculate_dynamic_position_size(
            account_value=target_account.net_liquidation,
            symbol=signal.symbol,
            wyckoff_signal=signal.phase,
            signal_strength=signal.strength
        )
        position_size = sizing_result["final_position_value"]

        # Log the dynamic sizing decision
        self.logger.info(f"🎯 Dynamic sizing for {signal.symbol}: ${position_size:.2f}")
        self.logger.info(f"   Signal: {signal.phase}, Strength: {signal.strength:.2f}")
        self.logger.info(f"   VIX: {sizing_result['current_vix']:.1f}, Regime: {sizing_result['market_regime']}")
        
        # Apply sector regime weighting if available
        regime_data = self.current_config.get('regime_data')
        if regime_data and regime_data.sector_weights:
            try:
                if hasattr(self.config_manager, 'regime_analyzer') and self.config_manager.regime_analyzer:
                    sector_weight = self.config_manager.regime_analyzer.get_sector_weight_for_symbol(
                        signal.symbol, regime_data.sector_weights
                    )
                    position_size *= sector_weight
                    if sector_weight != 1.0:
                        self.logger.info(f"📊 Sector weight for {signal.symbol}: {sector_weight:.2f}")
            except Exception as e:
                self.logger.debug(f"Error applying sector weight: {e}")
        
        # Apply multiple safety limits (take the minimum)
        safety_limits = [
            position_size,  # Base calculation
            account_cash * 0.25,  # Max 25% of account cash  
            12.0,  # Hard cap at $12
            account_cash - 15.0  # Leave $15 buffer minimum
        ]
        
        position_size = min([limit for limit in safety_limits if limit > 0])
        
        # Ensure minimum viable position
        position_size = max(position_size, 5.0)
        
        # Final safety check
        if position_size >= account_cash - 10.0:
            position_size = max(5.0, account_cash - 15.0)
        
        # Log regime information if available
        regime_info = ""
        if regime_data:
            regime_info = f" [Regime: {regime_data.trend_regime}/{regime_data.volatility_regime}, Multiplier: {regime_data.position_size_multiplier:.2f}]"
        
        self.logger.info(f"💰 REGIME-AWARE: {signal.symbol} ({signal.phase}): ${position_size:.2f}{regime_info}")
        self.logger.info(f"   Account: {target_account.account_type}, Cash: ${account_cash:.2f}")
        
        return position_size


class EnhancedFractionalTradingBot:
    """REFACTORED: Complete enhanced fractional trading bot with consolidated logic"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        self.config_manager = None  # Will be RegimeAwareConfigManager
        self.position_manager = None
        self.day_trade_checker = None
        self.setup_logging()
        
        # Enhanced features
        self.emergency_mode = False        
        self.last_reconciliation = None
        self.day_trades_blocked_today = 0
        
        # Enhanced exit strategy manager
        self.enhanced_exit_manager = None
        if ENHANCED_EXIT_STRATEGY_AVAILABLE:
            try:
                self.enhanced_exit_manager = EnhancedExitStrategyManager(self.logger)
                self.logger.info("✅ Enhanced Exit Strategy System initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced Exit Strategy initialization failed: {e}")
                self.enhanced_exit_manager = None
        
        # Signal quality enhancement
        if SIGNAL_QUALITY_ENHANCEMENT:
            self.signal_quality_analyzer = EnhancedMultiTimeframeWyckoffAnalyzer(self.logger)
            self.logger.info("🎯 Signal Quality Enhancement (Multi-timeframe) enabled")
        else:
            self.signal_quality_analyzer = None
            self.logger.info("📊 Using standard signal analysis")
        
        # Trading parameters
        self.min_signal_strength = 0.5
        self.buy_phases = ['ST', 'SOS', 'LPS', 'BU']
        self.sell_phases = ['PS', 'SC']
    
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
        self.logger.info("🚀 REFACTORED ENHANCED FRACTIONAL TRADING BOT")
        self.logger.info("💰 Consolidated position sizing + Advanced Wyckoff exits")
        self.logger.info("🛡️ Portfolio protection + Position reconciliation")
        self.logger.info("🚨 Real account day trading protection + Compliance tracking")
        self.logger.info("📊 Market regime adaptation - Optimization 2")
    
    def initialize_systems(self) -> bool:
        """Initialize all enhanced systems"""
        try:
            self.logger.info("🔧 Initializing enhanced systems...")
            
            self.main_system = MainSystem()
            self.wyckoff_strategy = WyckoffPnFStrategy()
            self.database = EnhancedTradingDatabase()
            
            # Initialize CONSOLIDATED regime-aware config manager
            self.config_manager = RegimeAwareConfigManager(self.logger)
            
            # Initialize position manager with consolidated config manager
            self.position_manager = SmartFractionalPositionManager(
                self.database, self.config_manager, self.logger
            )
            
            # Initialize day trade checker
            self.day_trade_checker = RealAccountDayTradeChecker(self.logger)
            
            self.logger.info("✅ Refactored enhanced systems initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def _check_day_trade_compliance(self, symbol: str, action: str, emergency: bool = False) -> DayTradeCheckResult:
        """Check comprehensive day trading compliance"""
        return self.day_trade_checker.comprehensive_day_trade_check(
            self.main_system.wb, self.database, symbol, action, 
            self.main_system.account_manager, emergency
        )
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions from database"""
        return self.database.get_all_positions()
    
    def _ensure_valid_session(self) -> bool:
        """Ensure we have a valid session WITHOUT resetting account context"""
        try:
            # Store current account context before validation
            current_account_id = self.main_system.wb._account_id
            current_zone = self.main_system.wb.zone_var
            
            self.logger.debug(f"🔍 Validating session (preserving account context: {current_account_id})")
            
            # Test session with a simple API call
            try:
                test_quote = self.main_system.wb.get_quote('SPY')
                if test_quote and 'close' in test_quote:
                    self.logger.debug(f"✅ Session validation passed")
                    
                    # Restore account context
                    self.main_system.wb._account_id = current_account_id
                    self.main_system.wb.zone_var = current_zone
                    
                    return True
                else:
                    self.logger.warning("⚠️ Session validation failed")
                    return False
                    
            except Exception as test_error:
                self.logger.warning(f"⚠️ Session test failed: {test_error}")
                
                # Try to refresh session
                self.logger.info("🔄 Attempting session refresh...")
                self.main_system.session_manager.clear_session()
                
                if self.main_system.login_manager.login_automatically():
                    self.logger.info("✅ Session refreshed successfully")
                    
                    # Restore the account context after refresh
                    self.main_system.wb._account_id = current_account_id
                    self.main_system.wb.zone_var = current_zone
                    
                    self.main_system.session_manager.save_session(self.main_system.wb)
                    return True
                else:
                    self.logger.error("❌ Failed to refresh session")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error in session validation: {e}")
            return False
    
    def execute_buy_order(self, signal: WyckoffSignal, account, position_size: float) -> bool:
        """Execute buy order with DAY TRADE PROTECTION and strict cash validation"""
        try:
            # Day trade compliance check
            day_trade_check = self._check_day_trade_compliance(signal.symbol, 'BUY')
            self.database.log_day_trade_check(day_trade_check)
            
            if day_trade_check.recommendation == 'BLOCK':
                self.logger.warning(f"🚨 DAY TRADE BLOCKED: {signal.symbol} BUY - {day_trade_check.details}")
                self.day_trades_blocked_today += 1
                return False
            
            # Account switching and cash validation
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            # Strict cash validation
            available_cash = account.settled_funds
            min_buffer = 15.0
            
            if available_cash < position_size + min_buffer:
                self.logger.warning(f"⚠️ INSUFFICIENT CASH for {signal.symbol}")
                self.logger.warning(f"   Required: ${position_size:.2f} + ${min_buffer:.2f} buffer = ${position_size + min_buffer:.2f}")
                self.logger.warning(f"   Available: ${available_cash:.2f}")
                return False
            
            # Session validation and quote retrieval
            if not self._ensure_valid_session():
                self.logger.error(f"❌ Cannot establish valid session for {signal.symbol}")
                return False
            
            quote_data = self.main_system.wb.get_quote(signal.symbol)
            if not quote_data or 'close' not in quote_data:
                self.logger.error(f"❌ Could not get quote for {signal.symbol}")
                return False
            
            current_price = float(quote_data['close'])
            max_affordable_shares = (available_cash - min_buffer) / current_price
            shares_to_buy = position_size / current_price
            
            # Ensure we don't exceed what we can afford
            if shares_to_buy > max_affordable_shares:
                shares_to_buy = max_affordable_shares
                position_size = shares_to_buy * current_price
                self.logger.info(f"🔧 Reduced order to affordable size: {shares_to_buy:.5f} shares")
            
            shares_to_buy = round(shares_to_buy, 5)
            actual_cost = shares_to_buy * current_price
            
            # Final validation
            if shares_to_buy < 0.00001:
                self.logger.warning(f"⚠️ Order too small: {shares_to_buy} shares")
                return False
            
            if actual_cost > available_cash - min_buffer:
                self.logger.error(f"❌ COST VALIDATION FAILED: ${actual_cost:.2f} > ${available_cash - min_buffer:.2f}")
                return False
            
            # Execute the order
            self.logger.info(f"💰 Buying {shares_to_buy:.5f} shares of {signal.symbol}")
            self.logger.info(f"   Price: ${current_price:.2f}, Cost: ${actual_cost:.2f}")
            self.logger.info(f"   Cash before: ${available_cash:.2f}, Will remain: ${available_cash - actual_cost:.2f}")
            self.logger.info(f"   Day Trade Check: {day_trade_check.recommendation}")
            
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
                    order_id=order_id,
                    day_trade_check=day_trade_check.recommendation
                )
                
                # Enhanced position tracking
                self.database.update_position(
                    symbol=signal.symbol,
                    shares=shares_to_buy,
                    cost=current_price,
                    account_type=account.account_type,
                    entry_phase=signal.phase,
                    entry_strength=signal.strength
                )
                
                # Update account cash tracking
                account.settled_funds -= actual_cost
                
                self.logger.info(f"✅ Buy order executed: {signal.symbol} - Order ID: {order_id}")
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Buy order failed for {signal.symbol}: {error_msg}")
                
                # Check if it's a session issue
                if 'session' in error_msg.lower() or 'expired' in error_msg.lower():
                    self.logger.warning("⚠️ Session issue detected - will retry on next run")
                    self.main_system.session_manager.clear_session()
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy for {signal.symbol}: {e}")
            return False
    
    def run_enhanced_trading_cycle(self) -> Tuple[int, int, int, int, int, int]:
        """Trading cycle with consolidated regime-aware logic"""
        trades_executed = 0
        wyckoff_sells = 0
        profit_scales = 0
        emergency_exits = 0
        day_trades_blocked = 0
        enhanced_exits = 0
        
        try:
            # Reset daily counter
            self.day_trades_blocked_today = 0

            # Update configuration with regime awareness
            config = self.position_manager.update_config(self.main_system.account_manager)
            
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Normal buy logic (only if NOT in emergency mode)
            if not self.emergency_mode:
                self.logger.info("🔍 Scanning for Wyckoff buy signals...")
                
                # Use enhanced scanning if available
                if hasattr(self.wyckoff_strategy, 'use_enhanced_analysis') and self.wyckoff_strategy.use_enhanced_analysis:
                    signals = self.wyckoff_strategy.scan_market_enhanced()
                    self.logger.info("🎯 Using enhanced multi-timeframe signal scanning")
                else:
                    signals = self.wyckoff_strategy.scan_market()
                    self.logger.info("📊 Using standard single-timeframe scanning")
                
                if signals:
                    buy_signals = [s for s in signals if (
                        s.phase in self.buy_phases and 
                        s.strength >= self.min_signal_strength and
                        s.volume_confirmation
                    )]
                    
                    if buy_signals and len(current_positions) < config['max_positions']:
                        enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
                        
                        # Apply signal quality filtering if available
                        if SIGNAL_QUALITY_ENHANCEMENT and self.signal_quality_analyzer:
                            try:
                                enhanced_signals = []
                                for signal in buy_signals:
                                    enhanced_result = self.signal_quality_analyzer.analyze_symbol_multi_timeframe(signal.symbol)
                                    
                                    if enhanced_result and enhanced_result.signal_quality in ['GOOD', 'EXCELLENT']:
                                        signal.strength = enhanced_result.enhanced_strength
                                        signal.combined_score = enhanced_result.confirmation_score
                                        enhanced_signals.append(signal)
                                        
                                        self.logger.info(f"🎯 {signal.symbol}: {enhanced_result.signal_quality} quality "
                                                    f"(Phases: {enhanced_result.primary_phase}/"
                                                    f"{enhanced_result.entry_timing_phase}/"
                                                    f"{enhanced_result.precision_phase})")
                                
                                if enhanced_signals:
                                    self.logger.info(f"📈 Quality Enhancement: {len(enhanced_signals)}/{len(buy_signals)} signals passed")
                                    buy_signals = enhanced_signals
                                else:
                                    self.logger.info(f"⚠️ Quality Enhancement: No signals met criteria")
                                    buy_signals = []
                                    
                            except Exception as e:
                                self.logger.warning(f"⚠️ Signal quality enhancement failed: {e}")

                        # Process buy signals with day trade checking
                        for signal in buy_signals[:max(1, config['max_positions'] - len(current_positions))]:
                            
                            # Check day trade compliance BEFORE calculating position size
                            day_trade_check = self._check_day_trade_compliance(signal.symbol, 'BUY')
                            
                            if day_trade_check.recommendation == 'BLOCK':
                                self.logger.warning(f"🚨 BUY SIGNAL BLOCKED BY DAY TRADE RULES: {signal.symbol}")
                                self.day_trades_blocked_today += 1
                                continue
                            
                            best_account = max(enabled_accounts, key=lambda x: x.settled_funds)
                            
                            # Get position size using consolidated manager
                            position_size = self.position_manager.get_position_size_for_signal(signal, best_account)
                            
                            # Only proceed if we have a viable position size and sufficient cash
                            if (position_size > 0 and 
                                best_account.settled_funds >= position_size + config.get('min_cash_buffer_per_account', 15.0)):
                                
                                if self.execute_buy_order(signal, best_account, position_size):
                                    trades_executed += 1
                                    best_account.settled_funds -= position_size
                                    
                                    # Add small delay between orders
                                    time.sleep(2)
                            else:
                                self.logger.info(f"⚠️ Skipping {signal.symbol}: insufficient cash or invalid position size")
            
            day_trades_blocked = self.day_trades_blocked_today
            
            # Enhanced exit strategy execution if available
            if self.enhanced_exit_manager and not self.emergency_mode:
                try:
                    self.logger.info("🎯 Running Enhanced Exit Strategy Analysis...")
                    
                    current_positions = self.get_current_positions()
                    
                    if current_positions:
                        for position_key, position_data in current_positions.items():
                            try:
                                should_exit, reason, percentage = self.enhanced_exit_manager.should_exit_now(position_data)
                                
                                if should_exit and percentage > 0:
                                    symbol = position_data['symbol']
                                    shares_to_sell = position_data['shares'] * percentage
                                    
                                    # Day trade compliance check
                                    day_trade_check = self._check_day_trade_compliance(symbol, 'SELL')
                                    
                                    if day_trade_check.recommendation != 'BLOCK':
                                        self.logger.info(f"🎯 Enhanced exit signal: {symbol} - {reason}")
                                        self.logger.info(f"   Selling {percentage:.0%} ({shares_to_sell:.5f} shares)")
                                        enhanced_exits += 1
                                    else:
                                        self.logger.warning(f"🚨 Enhanced exit blocked by day trade rules: {symbol}")
                                        
                            except Exception as e:
                                self.logger.error(f"Error processing enhanced exit for {position_key}: {e}")
                                continue
                    
                    self.logger.info(f"🎯 Enhanced exit signals processed: {enhanced_exits}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Enhanced exit strategy error: {e}")

            return trades_executed, wyckoff_sells, profit_scales, enhanced_exits, emergency_exits, day_trades_blocked
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced trading cycle: {e}")
            return trades_executed, wyckoff_sells, profit_scales, enhanced_exits, emergency_exits, self.day_trades_blocked_today
    
    def run(self) -> bool:
        """Main execution with consolidated logic"""
        try:
            self.logger.info("🚀 Starting Refactored Enhanced Fractional Trading Bot")
            
            if not self.initialize_systems():
                return False
            
            if not self.main_system.run():
                return False
            
            # Run enhanced trading cycle
            trades, wyckoff_sells, profit_scales, enhanced_exits, emergency_exits, day_trades_blocked = self.run_enhanced_trading_cycle()
            
            # Enhanced summary
            total_actions = trades + wyckoff_sells + profit_scales + enhanced_exits + emergency_exits
            
            self.logger.info("📊 REFACTORED ENHANCED TRADING SESSION SUMMARY")
            self.logger.info(f"   Buy Orders: {trades}")
            self.logger.info(f"   Wyckoff Sells: {wyckoff_sells}")
            self.logger.info(f"   Profit Scaling: {profit_scales}")
            self.logger.info(f"   Enhanced Exits: {enhanced_exits}")
            self.logger.info(f"   Emergency Exits: {emergency_exits}")
            self.logger.info(f"   Day Trades Blocked: {day_trades_blocked}")
            self.logger.info(f"   Total Actions: {total_actions}")
            self.logger.info(f"   Emergency Mode: {'YES' if self.emergency_mode else 'NO'}")
            
            # Enhanced bot run logging
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            
            regime_summary = ""
            if hasattr(self.config_manager, 'regime_analyzer') and self.config_manager.regime_analyzer:
                try:
                    regime_data = self.config_manager.cached_config.get('regime_data')
                    if regime_data:
                        regime_summary = f", Regime: {regime_data.trend_regime}/{regime_data.volatility_regime}"
                except:
                    pass
            
            log_details_with_regime = f"REFACTORED: Buy={trades}, Wyckoff={wyckoff_sells}, Profit={profit_scales}, Enhanced={enhanced_exits}, Emergency={emergency_exits}, DayTradesBlocked={day_trades_blocked}{regime_summary}"
            
            self.database.log_bot_run(
                signals_found=trades,
                trades_executed=total_actions,
                wyckoff_sells=wyckoff_sells,
                profit_scales=profit_scales + enhanced_exits,
                emergency_exits=emergency_exits,
                day_trades_blocked=day_trades_blocked,
                errors=0,
                portfolio_value=sum(acc.net_liquidation for acc in enabled_accounts),
                available_cash=sum(acc.settled_funds for acc in enabled_accounts),
                emergency_mode=self.emergency_mode,
                market_condition='UNKNOWN',
                portfolio_drawdown_pct=0.0,
                status="COMPLETED_REFACTORED_ENHANCED",
                log_details=log_details_with_regime
            )
            
            if total_actions > 0:
                self.logger.info("✅ Refactored enhanced fractional bot completed with actions")
            else:
                self.logger.info("✅ Refactored enhanced fractional bot completed (no actions needed)")
            
            if day_trades_blocked > 0:
                self.logger.warning(f"⚠️ Day trade protection blocked {day_trades_blocked} potential violations")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
            return False
        
        finally:
            if self.main_system:
                self.main_system.cleanup()


def main():
    """Main entry point"""
    print("🚀 Refactored Enhanced Fractional Trading Bot Starting...")
    
    bot = EnhancedFractionalTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Refactored enhanced fractional trading bot completed!")
        sys.exit(0)
    else:
        print("❌ Refactored enhanced fractional trading bot failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        print("🚀 Starting Refactored Enhanced Fractional Trading Bot...")
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")