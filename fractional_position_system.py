#!/usr/bin/env python3
"""
COMPLETE ENHANCED FRACTIONAL POSITION BUILDING SYSTEM - WITH REAL ACCOUNT DAY TRADE PROTECTION
Integrates all advanced exit management, Wyckoff warnings, portfolio protection, and REAL account day trade checking
This version checks ACTUAL Webull account trades, not just database records
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
import time as time_module


# Import existing systems
from main import MainSystem

# ENHANCEMENT: Wyckoff Reaccumulation Position Addition
import yfinance as yf


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

@dataclass
class ReaccumulationSignal:
    """Enhanced reaccumulation signal for position additions"""
    symbol: str
    phase_type: str
    strength: float
    current_price: float
    support_level: float
    resistance_level: float
    addition_percentage: float
    reasoning: str
    volume_analysis: Dict
    timeframe_confluence: int  # Number of timeframes confirming
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'

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

class RealAccountDayTradeChecker:
    """FIXED: Real account day trade checking using Webull API"""
    
    def __init__(self, logger):
        self.logger = logger
        self.trade_cache = {}  # Cache to avoid repeated API calls
        self.last_cache_update = None
    
    def get_actual_todays_trades(self, wb_client, symbol: str = None) -> List[Dict]:
        """FIXED: Get TODAY'S trades from actual Webull account with correct API methods"""
        today = datetime.now().strftime('%Y-%m-%d')
        cache_key = f"{today}_{symbol or 'ALL'}"
        
        # Use cache if updated within last 5 minutes
        if (self.last_cache_update and 
            cache_key in self.trade_cache and
            (datetime.now() - self.last_cache_update).total_seconds() < 300):
            return self.trade_cache[cache_key]
        
        try:
            self.logger.debug(f"🔍 Fetching real account trades for {symbol or 'ALL'}")
            
            # Get actual trade history from Webull using CORRECT method names
            orders = []
            
            try:
                # FIXED: Use correct method names from webull library
                current_orders = wb_client.get_current_orders()  # Gets pending/open orders
                history_orders = wb_client.get_history_orders(status='All', count=50)  # Gets historical orders
                
                # Combine all orders
                all_orders = []
                if current_orders:
                    all_orders.extend(current_orders)
                if history_orders:
                    all_orders.extend(history_orders)
                
                # Filter for today's FILLED trades
                today_orders = []
                for order in all_orders:
                    try:
                        # Handle different possible date formats from Webull API
                        order_date = order.get('createTime', order.get('orderDate', order.get('time', '')))
                        
                        if order_date:
                            # Parse the date/timestamp
                            if isinstance(order_date, str):
                                if len(order_date) > 10:  # Timestamp format
                                    if order_date.isdigit():
                                        # Unix timestamp in milliseconds
                                        order_dt = datetime.fromtimestamp(int(order_date) / 1000)
                                    else:
                                        # ISO format or other string format
                                        try:
                                            order_dt = datetime.fromisoformat(order_date.replace('Z', '+00:00'))
                                        except:
                                            # Try parsing as standard format
                                            order_dt = datetime.strptime(order_date[:10], '%Y-%m-%d')
                                else:
                                    # Date string format
                                    order_dt = datetime.strptime(order_date, '%Y-%m-%d')
                            else:
                                # Numeric timestamp
                                order_dt = datetime.fromtimestamp(order_date / 1000 if order_date > 1000000000000 else order_date)
                            
                            order_date_str = order_dt.strftime('%Y-%m-%d')
                            
                            # Only include today's FILLED orders
                            order_status = order.get('status', order.get('orderStatus', '')).upper()
                            if (order_date_str == today and 
                                order_status in ['FILLED', 'PARTIALLY_FILLED', 'EXECUTED']):
                                
                                # Parse order data with multiple possible field names
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
                                
                                # Normalize action names
                                if parsed_order['action'] in ['BUY', 'B']:
                                    parsed_order['action'] = 'BUY'
                                elif parsed_order['action'] in ['SELL', 'S']:
                                    parsed_order['action'] = 'SELL'
                                
                                # Filter by symbol if specified
                                if not symbol or parsed_order['symbol'] == symbol:
                                    today_orders.append(parsed_order)
                                    
                    except Exception as e:
                        self.logger.debug(f"Error parsing order: {e}")
                        continue
                
                orders = today_orders
                
            except AttributeError as e:
                self.logger.warning(f"⚠️ Webull API method not available: {e}")
                self.logger.warning("⚠️ This may be due to an older version of the webull library")
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
            else:
                self.logger.debug(f"📊 No real trades found today for {symbol or 'ALL'}")
            
            return orders
            
        except Exception as e:
            self.logger.error(f"❌ Error getting actual trades: {e}")
            return []
        
    def detect_manual_trades(self, wb_client, database, symbol: str, account_manager) -> bool:
        """FIXED: Detect manual trades with proper account handling"""
        try:
            # Get database position for this symbol (any account)
            db_positions = database.get_position(symbol)  # Gets all accounts
            if not db_positions:
                return False
            
            # Handle both single position and list of positions
            if isinstance(db_positions, list):
                db_total_shares = sum(pos['total_shares'] for pos in db_positions)
            else:
                db_total_shares = db_positions['total_shares']
            
            # Get REAL position from ALL accounts
            real_total_shares = 0.0
            
            # Check all enabled accounts for this symbol
            enabled_accounts = account_manager.get_enabled_accounts()
            for account in enabled_accounts:
                for position in account.positions:
                    if position['symbol'] == symbol:
                        real_total_shares += position['quantity']
            
            # Compare with database total (with proper tolerance)
            if abs(real_total_shares - db_total_shares) > 0.00001:
                self.logger.warning(f"Position mismatch for {symbol}: Real={real_total_shares:.5f}, DB={db_total_shares:.5f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting manual trades for {symbol}: {e}")
            return False
    
    def comprehensive_day_trade_check(self, wb_client, database, symbol: str, action: str, 
                                    account_manager, emergency: bool = False) -> DayTradeCheckResult:
        """FIXED: Comprehensive day trade check with correct API usage"""
        
        # Check 1: Database trades
        db_trades = database.get_todays_trades(symbol)
        db_day_trade = database.would_create_day_trade(symbol, action)
        
        # Check 2: Actual account trades using FIXED method
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
        
        # Check 3: Manual trades detection - NOW WITH account_manager parameter
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

class WyckoffReaccumulationDetector:
    """🎯 ENHANCED: Wyckoff Reaccumulation Detection for Position Additions"""
    
    def __init__(self, logger):
        self.logger = logger
        self.REACCUMULATION_LOOKBACK = 45  # Extended for better pattern recognition
        self.SUPPORT_TEST_TOLERANCE = 0.015  # 1.5% tolerance for support tests
        self.RANGING_THRESHOLD = 0.08  # 8% maximum range for reaccumulation
        self.VOLUME_DECLINE_THRESHOLD = 0.75  # Volume should be 75% of earlier average
        self.MIN_CONSOLIDATION_DAYS = 10  # Minimum days in consolidation
        
        # Enhanced pattern recognition parameters
        self.ABSORPTION_STRENGTH_THRESHOLD = 0.6
        self.SPRING_DETECTION_ENABLED = True
        self.HIGHER_LOW_PROGRESSION_WEIGHT = 0.3
    
    def analyze_for_reaccumulation(self, symbol: str, position: Dict) -> Optional[ReaccumulationSignal]:
        """🎯 ENHANCED: Comprehensive reaccumulation analysis for position additions"""
        try:
            self.logger.debug(f"🔍 Analyzing {symbol} for reaccumulation addition opportunity...")
            
            # Get extended historical data for better pattern recognition
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo", interval="1d")
            
            if len(data) < self.REACCUMULATION_LOOKBACK:
                self.logger.debug(f"📊 Insufficient data for {symbol}: {len(data)} days")
                return None
            
            current_price = data['Close'].iloc[-1]
            recent_data = data.tail(self.REACCUMULATION_LOOKBACK)
            
            # 🎯 STEP 1: Identify potential reaccumulation range
            range_analysis = self._analyze_reaccumulation_range(recent_data, current_price)
            if not range_analysis['is_valid_range']:
                self.logger.debug(f"📈 {symbol}: No valid reaccumulation range detected")
                return None
            
            # 🎯 STEP 2: Volume analysis for absorption evidence
            volume_analysis = self._analyze_volume_absorption(data, recent_data)
            if not volume_analysis['shows_absorption']:
                self.logger.debug(f"📊 {symbol}: No volume absorption detected")
                return None
            
            # 🎯 STEP 3: Price action analysis
            price_action = self._analyze_price_action_strength(recent_data, range_analysis)
            
            # 🎯 STEP 4: Support strength evaluation
            support_strength = self._evaluate_support_strength(recent_data, range_analysis['support_level'])
            
            # 🎯 STEP 5: Calculate overall reaccumulation strength
            strength_components = {
                'range_quality': range_analysis['range_quality'],
                'volume_absorption': volume_analysis['absorption_strength'],
                'price_action': price_action['strength'],
                'support_strength': support_strength,
                'higher_lows': price_action['higher_lows_count'] * 0.1
            }
            
            # Weighted strength calculation
            overall_strength = (
                strength_components['range_quality'] * 0.25 +
                strength_components['volume_absorption'] * 0.30 +
                strength_components['price_action'] * 0.25 +
                strength_components['support_strength'] * 0.20
            )
            
            # Apply higher lows bonus
            overall_strength += min(0.2, strength_components['higher_lows'])
            overall_strength = min(1.0, overall_strength)
            
            self.logger.debug(f"📊 {symbol} reaccumulation strength: {overall_strength:.3f}")
            
            # 🎯 STEP 6: Determine if signal meets threshold
            if overall_strength >= self.ABSORPTION_STRENGTH_THRESHOLD:
                # Calculate position addition percentage based on strength
                addition_pct = self._calculate_addition_percentage(overall_strength, price_action)
                
                # Determine risk level
                risk_level = self._assess_risk_level(price_action, volume_analysis, range_analysis)
                
                # Enhanced reasoning
                reasoning_parts = []
                if range_analysis['is_tight_range']:
                    reasoning_parts.append(f"tight range ({range_analysis['range_pct']:.1%})")
                if volume_analysis['significant_decline']:
                    reasoning_parts.append(f"volume decline ({volume_analysis['decline_pct']:.1%})")
                if price_action['higher_lows_count'] >= 2:
                    reasoning_parts.append(f"{price_action['higher_lows_count']} higher lows")
                if price_action['spring_detected']:
                    reasoning_parts.append("spring action")
                
                reasoning = f"Reaccumulation: {', '.join(reasoning_parts)}"
                
                # Enhanced timeframe confluence (simplified for now)
                timeframe_confluence = 1  # Would be enhanced with multi-timeframe analysis
                
                signal = ReaccumulationSignal(
                    symbol=symbol,
                    phase_type='REACCUMULATION',
                    strength=overall_strength,
                    current_price=current_price,
                    support_level=range_analysis['support_level'],
                    resistance_level=range_analysis['resistance_level'],
                    addition_percentage=addition_pct,
                    reasoning=reasoning,
                    volume_analysis=volume_analysis,
                    timeframe_confluence=timeframe_confluence,
                    risk_level=risk_level
                )
                
                self.logger.info(f"🎯 Reaccumulation signal: {symbol} (strength: {overall_strength:.2f}, add: {addition_pct:.1%})")
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {symbol} for reaccumulation: {e}")
            return None
    
    def _analyze_reaccumulation_range(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze if price is in a valid reaccumulation range"""
        high = data['High'].max()
        low = data['Low'].min()
        range_size = high - low
        range_pct = range_size / current_price
        
        # Check if it's a reasonable range for reaccumulation
        is_valid_range = range_pct <= self.RANGING_THRESHOLD
        is_tight_range = range_pct <= 0.05  # Very tight ranges are stronger signals
        
        # Check if current price is in reasonable position within range
        position_in_range = (current_price - low) / range_size if range_size > 0 else 0.5
        
        # Range quality scoring
        range_quality = 0.0
        if is_valid_range:
            range_quality += 0.4
        if is_tight_range:
            range_quality += 0.3
        if 0.2 <= position_in_range <= 0.8:  # Not at extremes
            range_quality += 0.3
        
        return {
            'is_valid_range': is_valid_range,
            'is_tight_range': is_tight_range,
            'support_level': low,
            'resistance_level': high,
            'range_pct': range_pct,
            'position_in_range': position_in_range,
            'range_quality': range_quality
        }
    
    def _analyze_volume_absorption(self, full_data: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Analyze volume for absorption characteristics"""
        recent_volume = recent_data['Volume'].mean()
        
        # Compare with earlier period (before the reaccumulation)
        earlier_period = full_data.tail(90).head(45)  # 45 days before recent 45
        if len(earlier_period) < 20:
            earlier_volume = full_data['Volume'].mean()
        else:
            earlier_volume = earlier_period['Volume'].mean()
        
        # Calculate volume decline
        if earlier_volume > 0:
            volume_ratio = recent_volume / earlier_volume
            decline_pct = 1 - volume_ratio
        else:
            volume_ratio = 1.0
            decline_pct = 0.0
        
        # Volume characteristics
        shows_absorption = volume_ratio < self.VOLUME_DECLINE_THRESHOLD
        significant_decline = decline_pct > 0.3  # 30%+ decline
        
        # Absorption strength scoring
        absorption_strength = 0.0
        if shows_absorption:
            absorption_strength += 0.5
        if significant_decline:
            absorption_strength += 0.3
        
        # Bonus for very low volume
        if volume_ratio < 0.5:  # Volume less than half of earlier period
            absorption_strength += 0.2
        
        return {
            'shows_absorption': shows_absorption,
            'significant_decline': significant_decline,
            'recent_volume': recent_volume,
            'earlier_volume': earlier_volume,
            'volume_ratio': volume_ratio,
            'decline_pct': decline_pct,
            'absorption_strength': min(1.0, absorption_strength)
        }
    
    def _analyze_price_action_strength(self, data: pd.DataFrame, range_analysis: Dict) -> Dict:
        """Analyze price action for reaccumulation strength signs"""
        closes = data['Close']
        lows = data['Low']
        
        # Count higher lows within the range
        higher_lows_count = 0
        prev_low = lows.iloc[0]
        
        for i in range(1, len(lows)):
            current_low = lows.iloc[i]
            if current_low > prev_low * 1.002:  # At least 0.2% higher
                higher_lows_count += 1
                prev_low = current_low
        
        # Detect potential spring action (brief break below support followed by recovery)
        spring_detected = False
        support_level = range_analysis['support_level']
        
        for i in range(5, len(data)):  # Look in recent data
            if (lows.iloc[i] < support_level * 0.998 and  # Brief break below support
                closes.iloc[i] > support_level * 1.001):  # But close above support
                spring_detected = True
                break
        
        # Calculate price action strength
        strength = 0.0
        if higher_lows_count >= 2:
            strength += 0.4
        if higher_lows_count >= 3:
            strength += 0.2
        if spring_detected:
            strength += 0.3
        
        # Check for consistent closes in upper half of range
        range_size = range_analysis['resistance_level'] - range_analysis['support_level']
        mid_range = range_analysis['support_level'] + (range_size * 0.5)
        upper_half_closes = sum(1 for close in closes.tail(10) if close > mid_range)
        
        if upper_half_closes >= 7:  # 70% of recent closes in upper half
            strength += 0.2
        
        return {
            'strength': min(1.0, strength),
            'higher_lows_count': higher_lows_count,
            'spring_detected': spring_detected,
            'upper_half_closes': upper_half_closes
        }
    
    def _evaluate_support_strength(self, data: pd.DataFrame, support_level: float) -> float:
        """Evaluate the strength of the support level"""
        lows = data['Low']
        
        # Count how many times price tested support without breaking significantly
        support_tests = 0
        successful_holds = 0
        
        for low in lows:
            if low <= support_level * 1.01:  # Within 1% of support
                support_tests += 1
                if low >= support_level * 0.99:  # Held above 99% of support
                    successful_holds += 1
        
        if support_tests == 0:
            return 0.3  # No tests, uncertain strength
        
        hold_ratio = successful_holds / support_tests
        
        # Strength based on hold ratio and number of tests
        if hold_ratio >= 0.8 and support_tests >= 3:
            return 0.9  # Very strong support
        elif hold_ratio >= 0.7 and support_tests >= 2:
            return 0.7  # Strong support
        elif hold_ratio >= 0.6:
            return 0.5  # Moderate support
        else:
            return 0.2  # Weak support
    
    def _calculate_addition_percentage(self, strength: float, price_action: Dict) -> float:
        """Calculate what percentage to add to position based on signal strength"""
        # Base addition percentage from strength
        base_addition = 0.15 + (strength - 0.6) * 0.5  # 15% to 35% based on strength
        
        # Bonuses for strong signals
        if price_action['spring_detected']:
            base_addition += 0.1
        
        if price_action['higher_lows_count'] >= 3:
            base_addition += 0.05
        
        # Conservative cap
        return min(0.4, max(0.1, base_addition))
    
    def _assess_risk_level(self, price_action: Dict, volume_analysis: Dict, range_analysis: Dict) -> str:
        """Assess the risk level of the reaccumulation signal"""
        risk_score = 0
        
        # Low risk factors
        if price_action['spring_detected']:
            risk_score -= 1
        if volume_analysis['significant_decline']:
            risk_score -= 1
        if range_analysis['is_tight_range']:
            risk_score -= 1
        if price_action['higher_lows_count'] >= 3:
            risk_score -= 1
        
        # High risk factors
        if range_analysis['range_pct'] > 0.06:  # Wider ranges are riskier
            risk_score += 1
        if volume_analysis['volume_ratio'] > 0.8:  # Not much volume decline
            risk_score += 1
        
        if risk_score <= -2:
            return 'LOW'
        elif risk_score <= 0:
            return 'MEDIUM'
        else:
            return 'HIGH'


class EnhancedTradingDatabase:
    """
    DEFINITIVE: Enhanced database manager with comprehensive tracking - FIXED VERSION
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
                    day_trade_check TEXT,
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
            
            # Add indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_date_symbol ON trades(date, symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stop_strategies_symbol ON stop_strategies(symbol, is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_bot_id ON positions(bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_enhanced_bot_id ON positions_enhanced(bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_partial_sales_symbol ON partial_sales(symbol, sale_date)')
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
        """
        FIXED: Update position tracking with enhanced data for a specific account
        Updates both positions and positions_enhanced tables with proper sync
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            # Update main positions table
            self._update_positions_table(conn, symbol, shares, cost, account_type, 
                                       entry_phase, entry_strength, today)
            
            # FIXED: Update enhanced positions table with proper sync
            self._update_positions_enhanced_table_fixed(conn, symbol, shares, cost, account_type, 
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
    
    def _update_positions_enhanced_table_fixed(self, conn, symbol: str, shares: float, cost: float, 
                                             account_type: str, entry_phase: str, entry_strength: float, today: str):
        """FIXED: Enhanced positions table update with proper synchronization"""
        
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
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                signals_found, trades_executed, wyckoff_sells, profit_scales, emergency_exits,
                day_trades_blocked, errors, portfolio_value, available_cash, emergency_mode, 
                market_condition, portfolio_drawdown_pct, status, log_details, self.bot_id
            ))


class EnhancedWyckoffAnalyzer:
    """Enhanced Wyckoff analyzer with advanced warning signals - FIXED VERSION"""
    
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
            
            # 1. UTAD - Upthrust After Distribution - NOW IMPLEMENTED
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
    
    def _detect_utad(self, symbol: str, data: pd.DataFrame, current_price: float) -> Optional[WyckoffWarningSignal]:
        """FIXED: Detect Upthrust After Distribution (UTAD) pattern"""
        if len(data) < 20:
            return None
        
        try:
            # Look for UTAD pattern: new high on lower volume followed by weakness
            recent_data = data.tail(15)
            high_20 = data['High'].tail(20).max()
            is_near_high = current_price >= high_20 * 0.995
            
            if not is_near_high:
                return None
            
            # Check for volume characteristics of UTAD
            recent_volume = recent_data['Volume'].tail(5).mean()
            earlier_volume = data['Volume'].tail(20).head(10).mean()
            
            # UTAD shows lower volume on new highs
            if recent_volume < earlier_volume * 0.8:
                # Check for subsequent weakness
                last_5_close = recent_data['Close'].tail(5)
                first_close = last_5_close.iloc[0]
                last_close = last_5_close.iloc[-1]
                
                # Price should be declining after the high
                if last_close < first_close * 0.98:
                    strength = min(0.9, (earlier_volume / recent_volume - 1) * 0.5)
                    
                    return WyckoffWarningSignal(
                        symbol=symbol,
                        signal_type='UTAD',
                        strength=strength,
                        price=current_price,
                        key_level=high_20,
                        volume_data={'recent_vol': recent_volume, 'earlier_vol': earlier_volume},
                        context=f"UTAD: New high on weak volume, subsequent decline",
                        urgency='HIGH'
                    )
        except Exception as e:
            self.logger.debug(f"UTAD detection error for {symbol}: {e}")
        
        return None
        
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
            earlier_high_vol = high_days['Volume'].head(-2).mean() if len(high_days) > 2 else recent_high_vol
            
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
    """FIXED: Manages dynamic position sizing based on real account values"""
    
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
            
            # FIXED: Use conservative parameters
            config = self._calculate_conservative_parameters(total_value, total_cash, enabled_accounts)
            self.cached_config = config
            self.last_update = datetime.now()
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error getting dynamic config: {e}")
            return self._get_fallback_config()
    
    def _calculate_conservative_parameters(self, total_value: float, total_cash: float, accounts: List) -> Dict:
        """FIXED: Calculate much more conservative parameters"""
        
        # Find the account with the most cash for primary trading
        max_cash_available = max(acc.settled_funds for acc in accounts) if accounts else total_cash
        
        # FIXED: Much more conservative position sizing
        if total_cash < 100:
            base_position_pct = 0.08  # 8% of total cash
            max_positions = 2
            min_balance_pct = 0.40  # Keep 40% cash
        elif total_cash < 300:
            base_position_pct = 0.15  # 10% of total cash
            max_positions = 8
            min_balance_pct = 0.15  # Keep 30% cash
        elif total_cash < 500:
            base_position_pct = 0.20  # 12% of total cash
            max_positions = 12
            min_balance_pct = 0.10  # Keep 25% cash
        else:
            base_position_pct = 0.22  # 15% of total cash
            max_positions = 15
            min_balance_pct = 0.05  # Keep 20% cash
        
        # FIXED: Calculate based on per-account maximum to prevent overdrafts
        base_position_size = min(
            total_cash * base_position_pct,  # Percentage of total cash
            max_cash_available * 0.4,       # Max 40% of any single account
            20.0                            # Hard cap at $15 per position
        )
        
        min_balance_preserve = total_cash * min_balance_pct
        
        # Ensure minimum viable position size
        if base_position_size < 5.0:
            base_position_size = min(5.0, total_cash * 0.3)
        
        # FIXED: Much more conservative Wyckoff phases
        wyckoff_phases = {
            'ST': {'initial_allocation': 0.40, 'allow_additions': False, 'max_total_allocation': 0.40},
            'SOS': {'initial_allocation': 0.50, 'allow_additions': True, 'max_total_allocation': 0.80},
            'LPS': {'initial_allocation': 0.35, 'allow_additions': True, 'max_total_allocation': 0.70},
            'BU': {'initial_allocation': 0.30, 'allow_additions': True, 'max_total_allocation': 0.60},
            'Creek': {'initial_allocation': 0.0, 'allow_additions': False, 'max_total_allocation': 0.0}
        }
        
        # FIXED: More conservative profit targets
        profit_targets = [
            {'gain_pct': 0.02, 'sell_pct': 0.15, 'description': '6% gain: Take 15% profit'},
            {'gain_pct': 0.06, 'sell_pct': 0.20, 'description': '12% gain: Take 20% more'},
            {'gain_pct': 0.12, 'sell_pct': 0.25, 'description': '20% gain: Take 25% more'},
            {'gain_pct': 0.20, 'sell_pct': 0.40, 'description': '30% gain: Take final 40%'}
        ]
        
        return {
            'total_value': total_value,
            'total_cash': total_cash,
            'max_cash_per_account': max_cash_available,
            'base_position_size': base_position_size,
            'base_position_pct': base_position_pct,
            'min_balance_preserve': min_balance_preserve,
            'max_positions': max_positions,
            'num_accounts': len(accounts),
            'wyckoff_phases': wyckoff_phases,
            'profit_targets': profit_targets,
            'calculated_at': datetime.now().isoformat(),
            # FIXED: Add per-account limits
            'max_position_per_account': max_cash_available * 0.4,  # Max 40% of any account
            'min_cash_buffer_per_account': 15.0  # Keep $15 minimum in each account
        }
    
    def _get_fallback_config(self) -> Dict:
        """FIXED: More conservative fallback configuration"""
        return {
            'total_value': 100.0,
            'total_cash': 100.0,
            'base_position_size': 8.0,  # Much smaller
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
                {'gain_pct': 0.02, 'sell_pct': 0.15, 'description': '6% gain: Take 15% profit'},
                {'gain_pct': 0.06, 'sell_pct': 0.20, 'description': '12% gain: Take 20% more'},
                {'gain_pct': 0.20, 'sell_pct': 0.25, 'description': '20% gain: Take 25% more'}
            ],
            'calculated_at': datetime.now().isoformat(),
            'max_position_per_account': 40.0,
            'min_cash_buffer_per_account': 15.0
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
                {'gain_pct': 0.02, 'sell_pct': 0.15},
                {'gain_pct': 0.06, 'sell_pct': 0.20},
                {'gain_pct': 0.12, 'sell_pct': 0.25},
                {'gain_pct': 0.20, 'sell_pct': 0.40}
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
                    'sell_pct': target['sell_pct'] * time_adj * 0.8,  # FIXED: More conservative
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
    """FIXED: Enhanced position manager with Wyckoff selling and dynamic sizing"""
    
    def __init__(self, database, dynamic_account_manager, logger):
        self.database = database
        self.dynamic_manager = dynamic_account_manager
        self.logger = logger
        self.current_config = None
    
    def update_config(self, account_manager):
        """Update configuration based on current account values"""
        self.current_config = self.dynamic_manager.get_dynamic_config(account_manager)
        return self.current_config
    
    def get_position_size_for_signal(self, signal: WyckoffSignal, target_account) -> float:
        """FIXED: Calculate position size with strict account-specific limits"""
        if not self.current_config:
            return 5.0  # Very conservative fallback
        
        # Get account-specific available cash
        account_cash = target_account.settled_funds
        
        if account_cash < 20.0:  # Need minimum $20 to trade
            self.logger.warning(f"⚠️ Insufficient cash in {target_account.account_type}: ${account_cash:.2f}")
            return 0.0
        
        # Start with smaller base size
        base_size = self.current_config['base_position_size']
        phase_config = self.current_config['wyckoff_phases'].get(signal.phase, {})
        initial_allocation = phase_config.get('initial_allocation', 0.3)
        
        # Calculate position size
        position_size = base_size * initial_allocation
        
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
        
        self.logger.info(f"💰 CONSERVATIVE: {signal.symbol} ({signal.phase}): ${position_size:.2f}")
        self.logger.info(f"   Account: {target_account.account_type}, Cash: ${account_cash:.2f}")
        
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
        """FIXED: Compare database positions with actual Webull holdings BY ACCOUNT"""
        reconciliation_report = {
            'discrepancies_found': [],
            'positions_synced': 0,
            'positions_corrected': 0,
            'ghost_positions_removed': 0
        }
        
        try:
            self.logger.info("🔍 Starting position reconciliation...")
            
            # Get real positions from each account - FIXED METHOD
            enabled_accounts = account_manager.get_enabled_accounts()
            real_positions = {}
            
            for account in enabled_accounts:
                self.logger.debug(f"Checking {account.account_type} for positions...")
                
                # CRITICAL FIX: Use the account.positions that were already parsed correctly
                # Instead of trying to re-query them (which was failing)
                if account.positions:
                    for position in account.positions:
                        # Create account-specific key
                        position_key = f"{position['symbol']}_{account.account_type}"
                        real_positions[position_key] = {
                            'symbol': position['symbol'],
                            'account_type': account.account_type,
                            'total_shares': position['quantity'],
                            'avg_cost': position['cost_price']
                        }
                        self.logger.debug(f"   Found: {position['symbol']} = {position['quantity']:.5f} shares")
                else:
                    self.logger.debug(f"   No positions in {account.account_type}")
            
            self.logger.info(f"Real positions found: {len(real_positions)}")
            
            # Get bot positions from database (by account)
            bot_positions = {}
            with sqlite3.connect(self.database.db_path) as conn:
                results = conn.execute('''
                    SELECT symbol, account_type, total_shares, avg_cost 
                    FROM positions 
                    WHERE bot_id = ? AND total_shares > 0
                ''', (self.database.bot_id,)).fetchall()
                
                for symbol, account_type, shares, avg_cost in results:
                    position_key = f"{symbol}_{account_type}"
                    bot_positions[position_key] = {
                        'symbol': symbol,
                        'account_type': account_type,
                        'total_shares': shares,
                        'avg_cost': avg_cost
                    }
            
            self.logger.info(f"Database positions found: {len(bot_positions)}")
            
            # Compare and correct discrepancies - IMPROVED LOGIC
            all_position_keys = set(real_positions.keys()) | set(bot_positions.keys())
            
            for position_key in all_position_keys:
                real_pos = real_positions.get(position_key, {'total_shares': 0, 'avg_cost': 0})
                bot_pos = bot_positions.get(position_key, {'total_shares': 0, 'avg_cost': 0})
                
                real_shares = real_pos['total_shares']
                bot_shares = bot_pos['total_shares']
                
                # FIXED: Use more reasonable tolerance for fractional shares
                if abs(real_shares - bot_shares) > 0.00001:  # Changed from 0.001 to 0.00001
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
                    
                    self.logger.info(f"Discrepancy: {symbol} Real={real_shares:.5f}, DB={bot_shares:.5f}")
                    

                else:
                    self.logger.debug(f"Position matches: {position_key}")
                    
                reconciliation_report['positions_synced'] += 1
            
            if reconciliation_report['discrepancies_found']:
                self.logger.warning(f"Found {len(reconciliation_report['discrepancies_found'])} discrepancies")
            else:
                self.logger.info("✅ All positions match - no reconciliation needed")
                    
        except Exception as e:
            self.logger.error(f"Error during position reconciliation: {e}")
            import traceback
            traceback.print_exc()
        
        return reconciliation_report
    
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
    """DEFINITIVE: Complete enhanced fractional trading bot with REAL ACCOUNT DAY TRADE PROTECTION"""
    
    def __init__(self):
        self.logger = None
        self.main_system = None
        self.wyckoff_strategy = None
        self.database = None
        self.config = PersonalTradingConfig()
        self.dynamic_manager = None
        self.position_manager = None
        self.comprehensive_exit_manager = None
        self.day_trade_checker = None  # NEW: Real account day trade checker
        
        # Enhanced features
        self.emergency_mode = False
        self.last_reconciliation = None
        self.day_trades_blocked_today = 0
        
        # ENHANCEMENT: Reaccumulation components
        self.reaccumulation_detector = None
        self.positions_added_today = 0
        self.max_position_additions_per_day = 3  # NEW: Track blocked day trades
        
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
        self.logger.info("🚀 ENHANCED FRACTIONAL TRADING BOT - WITH REAL ACCOUNT DAY TRADE PROTECTION")
        self.logger.info("💰 Conservative position sizing + Advanced Wyckoff exits")
        self.logger.info("🛡️ Portfolio protection + Position reconciliation")
        self.logger.info("🚨 REAL account day trading protection + Compliance tracking")
    
    def initialize_systems(self) -> bool:
        """Initialize all enhanced systems including day trade protection"""
        try:
            self.logger.info("🔧 Initializing enhanced systems with day trade protection...")
            
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
            
            # NEW: Initialize real account day trade checker
            self.day_trade_checker = RealAccountDayTradeChecker(self.logger)
            
            self.logger.info("✅ Enhanced systems with day trade protection initialized")
            
            # ENHANCEMENT: Initialize reaccumulation detector
            self.reaccumulation_detector = WyckoffReaccumulationDetector(self.logger)
            self.logger.info("✅ Reaccumulation detector initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    def _check_day_trade_compliance(self, symbol: str, action: str, emergency: bool = False) -> DayTradeCheckResult:
        """NEW: Check comprehensive day trading compliance"""
        return self.day_trade_checker.comprehensive_day_trade_check(
            self.main_system.wb, self.database, symbol, action, 
            self.main_system.account_manager, emergency  # ADD account_manager parameter
        )
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions from database"""
        return self.database.get_all_positions()
    
    def _ensure_valid_session(self) -> bool:
        """FIXED: Ensure we have a valid session WITHOUT resetting account context"""
        try:
            # STORE current account context before validation
            current_account_id = self.main_system.wb._account_id
            current_zone = self.main_system.wb.zone_var
            
            self.logger.debug(f"🔍 Validating session (preserving account context: {current_account_id})")
            
            # Test session with a simple API call that doesn't change account context
            try:
                # Use get_quote instead of account-specific calls to test session
                test_quote = self.main_system.wb.get_quote('SPY')
                if test_quote and 'close' in test_quote:
                    self.logger.debug(f"✅ Session validation passed (quote successful)")
                    
                    # RESTORE account context (in case it got changed)
                    self.main_system.wb._account_id = current_account_id
                    self.main_system.wb.zone_var = current_zone
                    
                    return True
                else:
                    self.logger.warning("⚠️ Session validation failed (quote failed)")
                    return False
                    
            except Exception as test_error:
                self.logger.warning(f"⚠️ Session test failed: {test_error}")
                
                # Try to refresh session if the test failed
                self.logger.info("🔄 Attempting session refresh...")
                
                # Clear old session and force fresh login (but preserve account context)
                self.main_system.session_manager.clear_session()
                
                if self.main_system.login_manager.login_automatically():
                    self.logger.info("✅ Session refreshed successfully")
                    
                    # CRITICAL: Restore the account context after refresh
                    self.main_system.wb._account_id = current_account_id
                    self.main_system.wb.zone_var = current_zone
                    
                    # Save the refreshed session
                    self.main_system.session_manager.save_session(self.main_system.wb)
                    return True
                else:
                    self.logger.error("❌ Failed to refresh session")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error in session validation: {e}")
            return False
  
        
    def execute_buy_order(self, signal: WyckoffSignal, account, position_size: float) -> bool:
        """ENHANCED: Execute buy order with DAY TRADE PROTECTION and strict cash validation"""
        try:
            # STEP 1: DAY TRADE COMPLIANCE CHECK
            day_trade_check = self._check_day_trade_compliance(signal.symbol, 'BUY')
            
            # Log the check
            self.database.log_day_trade_check(day_trade_check)
            
            if day_trade_check.recommendation == 'BLOCK':
                self.logger.warning(f"🚨 DAY TRADE BLOCKED: {signal.symbol} BUY - {day_trade_check.details}")
                self.day_trades_blocked_today += 1
                return False
            elif day_trade_check.would_be_day_trade:
                self.logger.warning(f"⚠️ Day trade detected but proceeding: {day_trade_check.details}")
            
            # STEP 2: Account switching and cash validation
            if not self.main_system.account_manager.switch_to_account(account):
                self.logger.error(f"❌ Failed to switch to account for {signal.symbol}")
                return False
            
            # STRICT cash validation BEFORE attempting order
            available_cash = account.settled_funds
            min_buffer = 15.0  # Always keep $15 buffer
            
            if available_cash < position_size + min_buffer:
                self.logger.warning(f"⚠️ INSUFFICIENT CASH for {signal.symbol}")
                self.logger.warning(f"   Required: ${position_size:.2f} + ${min_buffer:.2f} buffer = ${position_size + min_buffer:.2f}")
                self.logger.warning(f"   Available: ${available_cash:.2f}")
                return False
            
            # STEP 3: Session validation and quote retrieval
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
                position_size = shares_to_buy * current_price  # Recalculate actual cost
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
            
            # STEP 4: Execute the order
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
                
                # Enhanced logging with day trade info
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
                
                # FIXED: Enhanced position tracking WITH ACCOUNT TYPE
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
                    self.logger.warning("⚠️ Session issue detected. . Logging in again.")
                    # Clear session to force fresh login next time
                    self.main_system.session_manager.clear_session()

                    self.main_system.login_manager.login_automatically()
                    # Save the refreshed session
                    self.main_system.session_manager.save_session(self.main_system.wb)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing buy for {signal.symbol}: {e}")
            return False

    
    def execute_emergency_exit(self, action: Dict, current_positions: Dict) -> bool:
            """Execute emergency exits WITH day trade compliance"""
            try:
                if action['action'] == 'LIQUIDATE_ALL':
                    self.logger.critical("🚨 LIQUIDATING ALL POSITIONS - MARKET CRASH")
                    success_count = 0
                    for symbol, position in current_positions.items():
                        if self._emergency_liquidate_position(position['symbol'], position, emergency=True):
                            success_count += 1
                    return success_count > 0
                    
                elif action['symbol'] in [p['symbol'] for p in current_positions.values()]:
                    # Find the position
                    target_position = next((p for p in current_positions.values() 
                                        if p['symbol'] == action['symbol']), None)
                    if target_position:
                        return self._emergency_liquidate_position(action['symbol'], target_position, emergency=True)
                
            except Exception as e:
                self.logger.error(f"❌ Error executing emergency action: {e}")
            
            return False
    
    def _emergency_liquidate_position(self, symbol: str, position: Dict, emergency: bool = False) -> bool:
        """Emergency liquidation with day trade protection and emergency override"""
        try:
            # STEP 1: Day trade compliance check with emergency override
            day_trade_check = self._check_day_trade_compliance(symbol, 'SELL', emergency=emergency)
            
            # Log the check
            self.database.log_day_trade_check(day_trade_check, emergency_override=emergency)
            
            if day_trade_check.recommendation == 'BLOCK' and not emergency:
                self.logger.warning(f"🚨 EMERGENCY EXIT BLOCKED BY DAY TRADE RULES: {symbol}")
                self.logger.warning(f"   {day_trade_check.details}")
                self.day_trades_blocked_today += 1
                return False
            elif day_trade_check.would_be_day_trade:
                self.logger.critical(f"🚨 EMERGENCY OVERRIDE: Day trade rules overridden for {symbol}")
            
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts 
                           if acc.account_type == position['account_type']), None)
            
            if not account or not self.main_system.account_manager.switch_to_account(account):
                return False
            
            total_shares_to_sell = position['shares']
            
            self.logger.critical(f"🚨 EMERGENCY EXIT: {total_shares_to_sell:.5f} shares of {symbol}")
            self.logger.critical(f"   Day Trade Status: {day_trade_check.recommendation}")
            
            # FIXED: Handle fractional share restrictions for emergency exits
            orders_executed = 0
            total_shares_sold = 0.0
            
            if total_shares_to_sell >= 1.0:
                # Split into whole shares and fractional remainder
                whole_shares = int(total_shares_to_sell)
                fractional_remainder = total_shares_to_sell - whole_shares
                
                self.logger.critical(f"   Emergency split: {whole_shares} whole shares + {fractional_remainder:.5f} fractional")
                
                # Execute whole share orders first
                for i in range(whole_shares):
                    order_result = self.main_system.wb.place_order(
                        stock=symbol,
                        price=0,
                        action='SELL',
                        orderType='MKT',
                        enforce='DAY',
                        quant=1.0,
                        outsideRegularTradingHour=False
                    )
                    
                    if order_result.get('success', False):
                        orders_executed += 1
                        total_shares_sold += 1.0
                        time_module.sleep(0.5)  # Brief delay
                    else:
                        self.logger.critical(f"   ❌ Emergency whole share order {i+1} failed")
                        break
                
                # Execute fractional remainder
                if fractional_remainder > 0.001:
                    time_module.sleep(0.5)
                    fractional_result = self.main_system.wb.place_order(
                        stock=symbol,
                        price=0,
                        action='SELL',
                        orderType='MKT',
                        enforce='DAY',
                        quant=fractional_remainder,
                        outsideRegularTradingHour=False
                    )
                    
                    if fractional_result.get('success', False):
                        orders_executed += 1
                        total_shares_sold += fractional_remainder
            else:
                # Single fractional order
                order_result = self.main_system.wb.place_order(
                    stock=symbol,
                    price=0,
                    action='SELL',
                    orderType='MKT',
                    enforce='DAY',
                    quant=total_shares_to_sell,
                    outsideRegularTradingHour=False
                )
                
                if order_result.get('success', False):
                    orders_executed = 1
                    total_shares_sold = total_shares_to_sell
            
            if orders_executed > 0:
                self.database.log_trade(
                    symbol=symbol,
                    action='EMERGENCY_SELL',
                    quantity=total_shares_sold,
                    price=0,
                    signal_phase='EMERGENCY',
                    signal_strength=1.0,
                    account_type=account.account_type,
                    order_id=f"EMERGENCY_{orders_executed}_ORDERS",
                    day_trade_check=day_trade_check.recommendation
                )
                
                self.database.update_position(
                    symbol=symbol,
                    shares=-total_shares_sold,
                    cost=0,
                    account_type=account.account_type
                )
                
                self.database.deactivate_stop_strategies(symbol)
                
                self.logger.critical(f"✅ Emergency exit executed: {symbol} ({total_shares_sold:.5f} shares)")
                return True
            else:
                self.logger.critical(f"❌ Emergency exit failed: {symbol}")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in emergency liquidation for {symbol}: {e}")
        
        return False
    
    def execute_wyckoff_warning_exit(self, warning: WyckoffWarningSignal, position: Dict) -> bool:
        """Execute Wyckoff warning exit with day trade protection"""
        try:
            # STEP 1: Day trade compliance check
            day_trade_check = self._check_day_trade_compliance(warning.symbol, 'SELL')
            
            # Log the check
            self.database.log_day_trade_check(day_trade_check)
            
            # Allow critical Wyckoff warnings to override day trade rules
            emergency_override = warning.urgency == 'CRITICAL'
            
            if day_trade_check.recommendation == 'BLOCK' and not emergency_override:
                self.logger.warning(f"🚨 WYCKOFF EXIT BLOCKED BY DAY TRADE RULES: {warning.symbol}")
                self.logger.warning(f"   {day_trade_check.details}")
                self.day_trades_blocked_today += 1
                return False
            elif day_trade_check.would_be_day_trade and emergency_override:
                self.logger.warning(f"🚨 CRITICAL WYCKOFF OVERRIDE: Day trade rules overridden for {warning.symbol}")
            
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            account = next((acc for acc in enabled_accounts 
                           if acc.account_type == position['account_type']), None)
            
            if not account or not self.main_system.account_manager.switch_to_account(account):
                return False
            
            total_shares_to_sell = position['shares']
            
            self.logger.warning(f"🔴 Wyckoff Warning Exit: {warning.symbol}")
            self.logger.warning(f"   Signal: {warning.signal_type} (Strength: {warning.strength:.2f})")
            self.logger.warning(f"   Context: {warning.context}")
            self.logger.warning(f"   Selling: {total_shares_to_sell:.5f} shares")
            self.logger.warning(f"   Day Trade Status: {day_trade_check.recommendation}")
            
            # FIXED: Handle fractional share restrictions for Wyckoff exits
            orders_executed = 0
            total_shares_sold = 0.0
            
            if total_shares_to_sell >= 1.0:
                # Split into whole shares and fractional remainder
                whole_shares = int(total_shares_to_sell)
                fractional_remainder = total_shares_to_sell - whole_shares
                
                self.logger.warning(f"   Wyckoff split: {whole_shares} whole shares + {fractional_remainder:.5f} fractional")
                
                # Execute whole share orders
                for i in range(whole_shares):
                    order_result = self.main_system.wb.place_order(
                        stock=warning.symbol,
                        price=0,
                        action='SELL',
                        orderType='MKT',
                        enforce='DAY',
                        quant=1.0,
                        outsideRegularTradingHour=False
                    )
                    
                    if order_result.get('success', False):
                        orders_executed += 1
                        total_shares_sold += 1.0
                        time_module.sleep(0.5)  # Brief delay
                    else:
                        self.logger.warning(f"   ❌ Wyckoff whole share order {i+1} failed")
                        break
                
                # Execute fractional remainder
                if fractional_remainder > 0.001:
                    time_module.sleep(0.5)
                    fractional_result = self.main_system.wb.place_order(
                        stock=warning.symbol,
                        price=0,
                        action='SELL',
                        orderType='MKT',
                        enforce='DAY',
                        quant=fractional_remainder,
                        outsideRegularTradingHour=False
                    )
                    
                    if fractional_result.get('success', False):
                        orders_executed += 1
                        total_shares_sold += fractional_remainder
            else:
                # Single fractional order
                order_result = self.main_system.wb.place_order(
                    stock=warning.symbol,
                    price=0,
                    action='SELL',
                    orderType='MKT',
                    enforce='DAY',
                    quant=total_shares_to_sell,
                    outsideRegularTradingHour=False
                )
                
                if order_result.get('success', False):
                    orders_executed = 1
                    total_shares_sold = total_shares_to_sell
            
            if orders_executed > 0:
                self.database.log_trade(
                    symbol=warning.symbol,
                    action='WYCKOFF_WARNING_SELL',
                    quantity=total_shares_sold,
                    price=warning.price,
                    signal_phase=warning.signal_type,
                    signal_strength=warning.strength,
                    account_type=account.account_type,
                    order_id=f"WYCKOFF_{orders_executed}_ORDERS",
                    day_trade_check=day_trade_check.recommendation
                )
                
                self.database.update_position(
                    symbol=warning.symbol,
                    shares=-total_shares_sold,
                    cost=warning.price,
                    account_type=account.account_type
                )
                
                self.database.deactivate_stop_strategies(warning.symbol)
                
                self.logger.warning(f"✅ Wyckoff warning exit executed: {warning.symbol} ({total_shares_sold:.5f} shares)")
                return True
            else:
                self.logger.error(f"❌ Wyckoff warning exit failed: {warning.symbol}")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error executing Wyckoff warning exit: {e}")
        
        return False
    
    def check_enhanced_profit_scaling(self, symbol: str, position: Dict, dynamic_targets: List[Dict]) -> Optional[Dict]:
        """FIXED: Check for profit scaling with day trade protection"""
        try:
            # STEP 1: Day trade compliance check for potential sale
            day_trade_check = self._check_day_trade_compliance(symbol, 'SELL')
            
            if day_trade_check.recommendation == 'BLOCK':
                self.logger.debug(f"🚨 Profit scaling blocked by day trade rules: {symbol}")
                return None
            
            quote_data = self.main_system.wb.get_quote(symbol)
            if not quote_data or 'close' not in quote_data:
                return None
            
            current_price = float(quote_data['close'])
            avg_cost = position['avg_cost']
            total_shares = position['shares']
            gain_pct = (current_price - avg_cost) / avg_cost
            
            self.logger.debug(f"Checking scaling for {symbol}: {total_shares:.5f} shares, {gain_pct:.1%} gain")
            
            for target in dynamic_targets:
                if gain_pct >= target['gain_pct']:
                    if not self.database.already_scaled_at_level(symbol, target['gain_pct']):
                        # CONSERVATIVE: Scale smaller percentages and ensure we don't oversell
                        conservative_sell_pct = target['sell_pct'] * 0.8  # Reduce by 20%
                        shares_to_sell = total_shares * conservative_sell_pct
                        
                        # Round to avoid precision issues
                        shares_to_sell = round(shares_to_sell, 5)
                        
                        # Ensure we don't try to sell more than 90% of position
                        max_sellable = total_shares * 0.9
                        shares_to_sell = min(shares_to_sell, max_sellable)
                        
                        # Validate minimum sale value and share amount
                        sale_value = shares_to_sell * current_price
                        
                        if sale_value >= 2.0 and shares_to_sell >= 0.00001:  # Minimum $2 sale
                            remaining_shares = total_shares - shares_to_sell
                            
                            self.logger.info(f"💰 Profit scaling opportunity: {symbol}")
                            self.logger.info(f"   Sell {shares_to_sell:.5f} of {total_shares:.5f} shares ({conservative_sell_pct:.1%})")
                            self.logger.info(f"   Gain: {gain_pct:.1%}, Value: ${sale_value:.2f}")
                            self.logger.info(f"   Day Trade Status: {day_trade_check.recommendation}")
                            
                            return {
                                'symbol': symbol,
                                'shares_to_sell': shares_to_sell,
                                'current_price': current_price,
                                'gain_pct': gain_pct,
                                'profit_amount': (current_price - avg_cost) * shares_to_sell,
                                'reason': f"CONSERVATIVE_SCALE_{target['gain_pct']*100:.0f}PCT",
                                'description': f"Conservative {conservative_sell_pct:.1%} scale at {gain_pct:.1%} gain",
                                'remaining_shares': remaining_shares,
                                'account_type': position['account_type'],
                                'scaling_level': f"{target['gain_pct']*100:.0f}PCT",
                                'day_trade_check': day_trade_check
                            }
                        else:
                            self.logger.debug(f"Sale too small for {symbol}: ${sale_value:.2f}, {shares_to_sell:.5f} shares")
                        break
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking profit scaling for {symbol}: {e}")
            return None
    
    def execute_enhanced_profit_scaling(self, opportunity: Dict) -> bool:
        """Execute profit scaling with day trade protection"""
        try:
            # Day trade check was already done in check_enhanced_profit_scaling
            day_trade_check = opportunity.get('day_trade_check')
            
            # Log the day trade check for scaling
            if day_trade_check:
                self.database.log_day_trade_check(day_trade_check)
            
            if day_trade_check and day_trade_check.recommendation == 'BLOCK':
                self.logger.warning(f"🚨 PROFIT SCALING BLOCKED BY DAY TRADE RULES: {opportunity['symbol']}")
                self.day_trades_blocked_today += 1
                return False
            
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
            total_shares_to_sell = opportunity['shares_to_sell']
            
            self.logger.info(f"💰 Enhanced Profit Scaling: {total_shares_to_sell:.5f} shares of {symbol}")
            self.logger.info(f"   {opportunity['description']} (${opportunity['profit_amount']:.2f} profit)")
            if day_trade_check:
                self.logger.info(f"   Day Trade Status: {day_trade_check.recommendation}")
            
            # FIXED: Handle fractional share restrictions (orders must be < 1.0 shares for fractional)
            orders_executed = 0
            total_shares_sold = 0.0
            
            if total_shares_to_sell >= 1.0:
                # Split into whole shares and fractional remainder
                whole_shares = int(total_shares_to_sell)
                fractional_remainder = total_shares_to_sell - whole_shares
                
                self.logger.info(f"   Splitting order: {whole_shares} whole shares + {fractional_remainder:.5f} fractional")
                
                # Execute whole share orders (each as separate orders to avoid issues)
                for i in range(whole_shares):
                    order_result = self.main_system.wb.place_order(
                        stock=symbol,
                        price=0,
                        action='SELL',
                        orderType='MKT',
                        enforce='DAY',
                        quant=1.0,  # Exactly 1 whole share
                        outsideRegularTradingHour=False
                    )
                    
                    if order_result.get('success', False):
                        orders_executed += 1
                        total_shares_sold += 1.0
                        self.logger.info(f"   ✅ Whole share order {i+1}/{whole_shares} executed")
                        time_module.sleep(1)  # Brief delay between orders
                    else:
                        error_msg = order_result.get('msg', 'Unknown error')
                        self.logger.warning(f"   ❌ Whole share order {i+1} failed: {error_msg}")
                        break
                
                # Execute fractional remainder if any
                if fractional_remainder > 0.001 and orders_executed > 0:  # Only if we have meaningful fractional amount
                    time_module.sleep(1)  # Brief delay
                    fractional_result = self.main_system.wb.place_order(
                        stock=symbol,
                        price=0,
                        action='SELL',
                        orderType='MKT',
                        enforce='DAY',
                        quant=fractional_remainder,
                        outsideRegularTradingHour=False
                    )
                    
                    if fractional_result.get('success', False):
                        orders_executed += 1
                        total_shares_sold += fractional_remainder
                        self.logger.info(f"   ✅ Fractional order ({fractional_remainder:.5f}) executed")
                    else:
                        error_msg = fractional_result.get('msg', 'Unknown error')
                        self.logger.warning(f"   ⚠️ Fractional order failed: {error_msg}")
            
            else:
                # Total is less than 1 share, can place as single fractional order
                order_result = self.main_system.wb.place_order(
                    stock=symbol,
                    price=0,
                    action='SELL',
                    orderType='MKT',
                    enforce='DAY',
                    quant=total_shares_to_sell,
                    outsideRegularTradingHour=False
                )
                
                if order_result.get('success', False):
                    orders_executed = 1
                    total_shares_sold = total_shares_to_sell
                    self.logger.info(f"   ✅ Single fractional order executed")
                else:
                    error_msg = order_result.get('msg', 'Unknown error')
                    self.logger.error(f"❌ Fractional profit scaling failed for {symbol}: {error_msg}")
                    
                    # Check if it's a session issue
                    if 'session' in error_msg.lower() or 'expired' in error_msg.lower():
                        self.logger.warning("⚠️ Session issue detected during profit scaling. Logging in again.")
                        self.main_system.session_manager.clear_session()
                        self.main_system.login_manager.login_automatically()
                        # Save the refreshed session
                        self.main_system.session_manager.save_session(self.main_system.wb)
                    return False
            
            # If we executed any orders successfully
            if orders_executed > 0 and total_shares_sold > 0:
                # Log successful trades - use the actual amount sold
                actual_profit = (opportunity['current_price'] - opportunity.get('avg_cost', opportunity['current_price'] * 0.9)) * total_shares_sold
                
                self.database.log_trade(
                    symbol=symbol,
                    action='PROFIT_SCALING',
                    quantity=total_shares_sold,
                    price=opportunity['current_price'],
                    signal_phase='PROFIT_SCALING',
                    signal_strength=opportunity['gain_pct'],
                    account_type=opportunity['account_type'],
                    order_id=f"SCALING_{orders_executed}_ORDERS",
                    day_trade_check=day_trade_check.recommendation if day_trade_check else 'ALLOWED'
                )
                
                self.database.log_partial_sale(
                    symbol=symbol,
                    shares_sold=total_shares_sold,
                    sale_price=opportunity['current_price'],
                    sale_reason=opportunity['reason'],
                    remaining_shares=opportunity['remaining_shares'] + (total_shares_to_sell - total_shares_sold),
                    gain_pct=opportunity['gain_pct'],
                    profit_amount=actual_profit,
                    scaling_level=opportunity['scaling_level']
                )
                
                self.database.update_position(
                    symbol=symbol,
                    shares=-total_shares_sold,
                    cost=opportunity['current_price'],
                    account_type=opportunity['account_type']
                )
                
                self.logger.info(f"✅ Enhanced profit scaling executed: {symbol}")
                self.logger.info(f"   Sold {total_shares_sold:.5f} of {total_shares_to_sell:.5f} shares in {orders_executed} order(s)")
                return True
            else:
                self.logger.error(f"❌ No orders executed successfully for {symbol} profit scaling")
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
    
    
    
    def scan_for_position_additions(self, current_positions: Dict) -> List[Dict]:
        """🎯 ENHANCED: Scan existing positions for reaccumulation addition opportunities"""
        addition_opportunities = []
        
        # Initialize reaccumulation detector if not already done
        if self.reaccumulation_detector is None:
            self.reaccumulation_detector = WyckoffReaccumulationDetector(self.logger)
            self.logger.info("🎯 Initialized enhanced reaccumulation detector")
        
        try:
            self.logger.info(f"🔍 Scanning {len(current_positions)} positions for reaccumulation additions...")
            
            for position_key, position in current_positions.items():
                symbol = position['symbol']
                
                # Check if position is eligible for additions
                if not self._is_position_eligible_for_addition(position):
                    self.logger.debug(f"❌ {symbol} not eligible for addition")
                    continue
                
                # Look for reaccumulation signal
                signal = self.reaccumulation_detector.analyze_for_reaccumulation(symbol, position)
                
                if signal and signal.strength >= 0.6:
                    addition_shares = self._calculate_addition_shares(signal, position)
                    
                    if addition_shares > 0:
                        addition_opportunities.append({
                            'signal': signal,
                            'position': position,
                            'addition_shares': addition_shares,
                            'estimated_cost': addition_shares * signal.current_price,
                            'risk_level': signal.risk_level
                        })
                        
                        self.logger.info(f"🎯 Reaccumulation opportunity: {symbol}")
                        self.logger.info(f"   Strength: {signal.strength:.2f}, Risk: {signal.risk_level}")
                        self.logger.info(f"   Add {signal.addition_percentage:.1%} ({addition_shares:.5f} shares)")
            
            # Sort by strength and risk level (prefer LOW risk, high strength)
            addition_opportunities.sort(key=lambda x: (
                -x['signal'].strength if x['risk_level'] == 'LOW' else -x['signal'].strength * 0.7
            ))
            
            # Limit to max additions per day
            limited_opportunities = addition_opportunities[:self.max_position_additions_per_day]
            
            if limited_opportunities:
                self.logger.info(f"🎯 Selected {len(limited_opportunities)} addition opportunities")
            else:
                self.logger.info("📊 No reaccumulation opportunities found")
            
            return limited_opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning for position additions: {e}")
            return []

    def _is_position_eligible_for_addition(self, position: Dict) -> bool:
        """🎯 ENHANCED: Check if position is eligible for reaccumulation additions"""
        try:
            symbol = position['symbol']
            current_shares = position.get('shares', 0)
            
            # Must have existing shares
            if current_shares <= 0:
                return False
            
            # Check current performance
            current_price = self._get_current_price(symbol)
            avg_cost = position.get('avg_cost', 0)
            
            if current_price and avg_cost:
                gain_pct = (current_price - avg_cost) / avg_cost
                
                # Don't add to positions down more than 20%
                if gain_pct < -0.20:
                    self.logger.debug(f"❌ {symbol} down {gain_pct:.1%}, too risky for addition")
                    return False
                
                # Don't add to positions up more than 50% (might be overextended)
                if gain_pct > 0.50:
                    self.logger.debug(f"❌ {symbol} up {gain_pct:.1%}, potentially overextended")
                    return False
            
            # Check time since last purchase (prefer positions held at least 5 days)
            last_purchase_date = position.get('last_purchase_date', '')
            if last_purchase_date:
                try:
                    last_purchase = datetime.strptime(last_purchase_date, '%Y-%m-%d')
                    days_since = (datetime.now() - last_purchase).days
                    if days_since < 5:
                        self.logger.debug(f"❌ {symbol} last purchase only {days_since} days ago")
                        return False
                except:
                    pass
            
            # Check entry phase - prefer certain Wyckoff phases for additions
            entry_phase = position.get('entry_phase', 'UNKNOWN')
            preferred_phases = ['SOS', 'LPS', 'BU']  # Sign of Strength, Last Point of Support, Backup
            
            if entry_phase in preferred_phases:
                self.logger.debug(f"✅ {symbol} has preferred entry phase: {entry_phase}")
                return True
            elif entry_phase in ['ST']:  # Stopping action - be more cautious
                # Only allow if position is profitable
                if current_price and avg_cost and (current_price >= avg_cost * 1.05):
                    return True
                else:
                    self.logger.debug(f"❌ {symbol} ST phase but not sufficiently profitable")
                    return False
            else:
                # Unknown or other phases - allow but be conservative
                return True
                
        except Exception as e:
            self.logger.debug(f"❌ Error checking eligibility for {position.get('symbol', 'unknown')}: {e}")
            return False

    def _calculate_addition_shares(self, signal: ReaccumulationSignal, position: Dict) -> float:
        """🎯 ENHANCED: Calculate number of shares to add with risk management"""
        try:
            current_shares = position.get('shares', 0)
            avg_cost = position.get('avg_cost', signal.current_price)
            current_position_value = current_shares * avg_cost
            
            # Base addition value from signal
            base_addition_value = current_position_value * signal.addition_percentage
            
            # Risk-based adjustments
            risk_multipliers = {
                'LOW': 1.0,
                'MEDIUM': 0.8,
                'HIGH': 0.6
            }
            
            risk_adjusted_value = base_addition_value * risk_multipliers.get(signal.risk_level, 0.8)
            
            # Apply conservative limits
            max_addition_limits = [
                risk_adjusted_value,
                current_position_value * 0.5,  # Never add more than 50% of current position
                800.0,  # Hard cap at $800 addition
                signal.current_price * 100  # Max 100 shares
            ]
            
            final_addition_value = min(max_addition_limits)
            
            # Calculate shares
            addition_shares = final_addition_value / signal.current_price
            
            # Round to reasonable precision
            addition_shares = round(addition_shares, 5)
            
            # Minimum viable addition
            min_addition_value = 20.0  # At least $20 addition
            if final_addition_value < min_addition_value:
                self.logger.debug(f"❌ Addition too small: ${final_addition_value:.2f}")
                return 0.0
            
            self.logger.debug(f"💰 {signal.symbol} addition calculation:")
            self.logger.debug(f"   Current position: {current_shares:.5f} shares (${current_position_value:.2f})")
            self.logger.debug(f"   Addition: {addition_shares:.5f} shares (${final_addition_value:.2f})")
            self.logger.debug(f"   Risk level: {signal.risk_level}")
            
            return addition_shares
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating addition shares: {e}")
            return 0.0

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            quote_data = self.main_system.wb.get_quote(symbol)
            if quote_data and 'close' in quote_data:
                return float(quote_data['close'])
        except Exception:
            pass
        return 0.0
    
    def execute_position_addition(self, opportunity: Dict) -> bool:
        """🎯 ENHANCED: Execute addition to existing position with full compliance"""
        try:
            signal = opportunity['signal']
            position = opportunity['position']
            addition_shares = opportunity['addition_shares']
            symbol = signal.symbol
            
            self.logger.info(f"🔄 Executing position addition for {symbol}...")
            
            # STEP 1: Comprehensive day trade compliance check
            day_trade_check = self._check_day_trade_compliance(symbol, 'BUY')
            self.database.log_day_trade_check(day_trade_check)
            
            if day_trade_check.recommendation == 'BLOCK':
                self.logger.warning(f"🚨 POSITION ADDITION BLOCKED: {symbol}")
                self.logger.warning(f"   Reason: {day_trade_check.details}")
                self.day_trades_blocked_today += 1
                return False
            elif day_trade_check.would_be_day_trade:
                self.logger.warning(f"⚠️ Day trade detected for addition: {symbol}")
            
            # STEP 2: Account management
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            target_account = next((acc for acc in enabled_accounts 
                                 if acc.account_type == position['account_type']), None)
            
            if not target_account:
                self.logger.error(f"❌ Target account not found: {position['account_type']}")
                return False
            
            # Switch to target account
            if not self.main_system.account_manager.switch_to_account(target_account):
                self.logger.error(f"❌ Failed to switch to account: {target_account.account_type}")
                return False
            
            # STEP 3: Enhanced cash validation
            required_cash = addition_shares * signal.current_price
            available_cash = target_account.settled_funds
            min_buffer = 20.0  # Increased buffer for position additions
            
            if available_cash < required_cash + min_buffer:
                self.logger.warning(f"⚠️ Insufficient cash for {symbol} addition:")
                self.logger.warning(f"   Required: ${required_cash:.2f} + ${min_buffer:.2f} buffer")
                self.logger.warning(f"   Available: ${available_cash:.2f}")
                return False
            
            # STEP 4: Session validation
            if not self._ensure_valid_session():
                self.logger.error(f"❌ Cannot establish valid session for {symbol} addition")
                return False
            
            # STEP 5: Final price validation
            current_quote = self.main_system.wb.get_quote(symbol)
            if not current_quote or 'close' not in current_quote:
                self.logger.error(f"❌ Cannot get current quote for {symbol}")
                return False
            
            current_market_price = float(current_quote['close'])
            
            # Check if price hasn't moved too much from signal price
            price_change = abs(current_market_price - signal.current_price) / signal.current_price
            if price_change > 0.03:  # More than 3% change
                self.logger.warning(f"⚠️ Price moved {price_change:.1%} since signal generation")
                # Re-calculate shares based on current price
                addition_shares = required_cash / current_market_price
                addition_shares = round(addition_shares, 5)
            
            # STEP 6: Enhanced logging before execution
            self.logger.info(f"🎯 ADDING TO POSITION: {symbol}")
            self.logger.info(f"   Current holding: {position['shares']:.5f} shares @ ${position['avg_cost']:.2f}")
            self.logger.info(f"   Adding: {addition_shares:.5f} shares @ ${current_market_price:.2f}")
            self.logger.info(f"   Addition cost: ${addition_shares * current_market_price:.2f}")
            self.logger.info(f"   Signal strength: {signal.strength:.2f}")
            self.logger.info(f"   Risk level: {signal.risk_level}")
            self.logger.info(f"   Reasoning: {signal.reasoning}")
            self.logger.info(f"   Day trade check: {day_trade_check.recommendation}")
            
            # STEP 7: Execute the order
            order_result = self.main_system.wb.place_order(
                stock=symbol,
                price=0,  # Market order
                action='BUY',
                orderType='MKT',
                enforce='DAY',
                quant=addition_shares,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                actual_cost = addition_shares * current_market_price
                
                # STEP 8: Enhanced logging and database updates
                self.database.log_trade(
                    symbol=symbol,
                    action='REACCUMULATION_ADD',
                    quantity=addition_shares,
                    price=current_market_price,
                    signal_phase=signal.phase_type,
                    signal_strength=signal.strength,
                    account_type=position['account_type'],
                    order_id=order_id,
                    day_trade_check=day_trade_check.recommendation
                )
                
                # Update position tracking
                self.database.update_position(
                    symbol=symbol,
                    shares=addition_shares,
                    cost=current_market_price,
                    account_type=position['account_type'],
                    entry_phase=signal.phase_type,
                    entry_strength=signal.strength
                )
                
                # Update account cash tracking
                target_account.settled_funds -= actual_cost
                self.positions_added_today += 1
                
                self.logger.info(f"✅ POSITION ADDITION EXECUTED: {symbol}")
                self.logger.info(f"   Order ID: {order_id}")
                self.logger.info(f"   Added {addition_shares:.5f} shares for ${actual_cost:.2f}")
                self.logger.info(f"   Remaining cash: ${target_account.settled_funds:.2f}")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Position addition order failed: {symbol}")
                self.logger.error(f"   Error: {error_msg}")
                
                # Check for session issues
                if 'session' in error_msg.lower() or 'expired' in error_msg.lower():
                    self.logger.warning("⚠️ Session issue detected during position addition")
                    self.main_system.session_manager.clear_session()
                    
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing position addition: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_enhanced_trading_cycle(self) -> Tuple[int, int, int, int, int, int]:
            """ENHANCED: Trading cycle with day trade protection"""
            trades_executed = 0
            wyckoff_sells = 0
            profit_scales = 0
            emergency_exits = 0
            
            positions_added = 0  # 🔧 CRITICAL FIX: Initialize positions_added
            
            try:
        
                # Step 1: Update configuration with conservative sizing
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
                
                # Step 6: Enhanced profit scaling with day trade protection
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
                    
                    # Log enhanced run data with day trade info
                    self.database.log_bot_run(
                        signals_found=0,
                        trades_executed=trades_executed,
                        wyckoff_sells=wyckoff_sells,
                        profit_scales=profit_scales,
                        emergency_exits=emergency_exits,
                        day_trades_blocked=self.day_trades_blocked_today,
                        errors=0,
                        portfolio_value=sum(acc.net_liquidation for acc in self.main_system.account_manager.get_enabled_accounts()),
                        available_cash=sum(acc.settled_funds for acc in self.main_system.account_manager.get_enabled_accounts()),
                        emergency_mode=True,
                        market_condition=portfolio_risk['market_condition'],
                        portfolio_drawdown_pct=portfolio_risk['portfolio_drawdown_pct'],
                        status="EMERGENCY_MODE",
                        log_details=f"Market: {portfolio_risk['market_condition']}, Drawdown: {portfolio_risk['portfolio_drawdown_pct']:.1%}, DayTrades Blocked: {self.day_trades_blocked_today}"
                    )
                    
                    return trades_executed, wyckoff_sells, profit_scales, emergency_exits, self.day_trades_blocked_today, positions_added
                else:
                    self.emergency_mode = False
                
                
            
                # ENHANCEMENT: Scan for position addition opportunities (BEFORE checking max_positions for NEW trades)
                if not self.emergency_mode:
                    self.logger.info("🔍 Scanning for Wyckoff reaccumulation position additions...")
                    
                    addition_opportunities = self.scan_for_position_additions(current_positions)
                    
                    if addition_opportunities:
                        self.logger.info(f"📈 Found {len(addition_opportunities)} position addition opportunities")
                        
                        for opportunity in addition_opportunities:
                            if self.positions_added_today >= self.max_position_additions_per_day:
                                self.logger.info("⚠️ Daily position addition limit reached")
                                break
                            
                            if self.execute_position_addition(opportunity):
                                positions_added += 1
                            
                            import time
                            time_module.sleep(1)
                    else:
                        self.logger.info("📊 No reaccumulation opportunities found")
                
                # Step 8: Normal buy logic (only if NOT in emergency mode) WITH DAY TRADE PROTECTION
                    if not self.emergency_mode:
                        self.logger.info("🔍 Scanning for Wyckoff buy signals...")
                        
                        # Use enhanced multi-timeframe scanning if available
                        if hasattr(self.wyckoff_strategy, 'use_enhanced_analysis') and self.wyckoff_strategy.use_enhanced_analysis:
                            signals = self.wyckoff_strategy.scan_market_enhanced()
                            self.logger.info("🎯 Using enhanced multi-timeframe signal scanning")
                        else:
                            signals = self.wyckoff_strategy.scan_market()
                            self.logger.info("📊 Using standard single-timeframe scanning")
                        
                        # Process signals if found
                        if signals:
                            buy_signals = [s for s in signals if (
                                s.phase in self.buy_phases and 
                                s.strength >= self.min_signal_strength and
                                s.volume_confirmation
                            )]
                            
                            self.logger.info(f"📊 Initial screening: {len(buy_signals)}/{len(signals)} signals passed basic criteria")
                            
                            # FIXED: Move enabled_accounts definition outside conditional block

                            
                            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()

                            
                            

                            
                            if buy_signals and len(current_positions) < config['max_positions']:
                                # Process buy signals WITH DAY TRADE CHECKING
                                for signal in buy_signals[:max(1, config['max_positions'] - len(current_positions))]:
                                    
                                    # STEP 1: Check day trade compliance BEFORE calculating position size
                                    day_trade_check = self._check_day_trade_compliance(signal.symbol, 'BUY')
                                    
                                    if day_trade_check.recommendation == 'BLOCK':
                                        self.logger.warning(f"🚨 BUY SIGNAL BLOCKED BY DAY TRADE RULES: {signal.symbol}")
                                        self.day_trades_blocked_today += 1
                                        continue  # Skip this signal
                                    
                                    best_account = max(enabled_accounts, key=lambda x: x.settled_funds)
                                    
                                    position_size = self.position_manager.get_position_size_for_signal(signal, best_account)

                                    
                                    # Only proceed if we have a viable position size and sufficient cash
                                    if (position_size > 0 and 
                                        best_account.settled_funds >= position_size + config.get('min_cash_buffer_per_account', 15.0)):
                                        
                                        self.logger.info(f"💰 Executing signal: {signal.symbol} (strength: {signal.strength:.2f})")

                                        # TEMPORARILY BLOCK BUY UNCOMMENT TO ENABLE TRADE EXECUTION
                                        if self.execute_buy_order(signal, best_account, position_size):
                                            trades_executed += 1
                                            best_account.settled_funds -= position_size
                                            
                                            # Add small delay between orders
                                            time_module.sleep(2)
                                    else:
                                        self.logger.info(f"⚠️ Skipping {signal.symbol}: insufficient cash or invalid position size")
                    else:
                        self.logger.info("🔍 No signals found to process")
                
                day_trades_blocked = self.day_trades_blocked_today
                return trades_executed, wyckoff_sells, profit_scales, emergency_exits, day_trades_blocked, positions_added
                
            except Exception as e:
                self.logger.error(f"❌ Error in enhanced trading cycle: {e}")
                return trades_executed, wyckoff_sells, profit_scales, emergency_exits, self.day_trades_blocked_today, positions_added
    
    def run(self) -> bool:
        """Main execution with enhanced day trade protection"""
        try:
            self.logger.info("🚀 Starting Enhanced Fractional Trading Bot with Day Trade Protection")
            
            if not self.initialize_systems():
                return False
            
            if not self.main_system.run():
                return False
            
            # Initial reconciliation
            self.run_periodic_reconciliation()
            
            # Run enhanced trading cycle with day trade protection
            trades, wyckoff_sells, profit_scales, emergency_exits, day_trades_blocked, positions_added = self.run_enhanced_trading_cycle()
            
            # Enhanced summary with day trade info
            total_actions = trades + wyckoff_sells + profit_scales + emergency_exits + positions_added
            
            self.logger.info("📊 ENHANCED FRACTIONAL TRADING SESSION SUMMARY")
            self.logger.info(f"   🆕 New Positions: {trades}")
            self.logger.info(f"   📈 Position Additions: {positions_added}")
            self.logger.info(f"   🔴 Wyckoff Sells: {wyckoff_sells}")
            self.logger.info(f"   💰 Profit Scaling: {profit_scales}")
            self.logger.info(f"   🚨 Emergency Exits: {emergency_exits}")
            self.logger.info(f"   🚫 Day Trades Blocked: {day_trades_blocked}")
            self.logger.info(f"   📊 Total Actions: {total_actions}")
            self.logger.info(f"   ⚠️ Emergency Mode: {'YES' if self.emergency_mode else 'NO'}")
            
            # Enhanced bot run logging with day trade tracking
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            self.database.log_bot_run(
                signals_found=trades,
                trades_executed=total_actions,
                wyckoff_sells=wyckoff_sells,
                profit_scales=profit_scales,
                emergency_exits=emergency_exits,
                day_trades_blocked=day_trades_blocked,
                errors=0,
                portfolio_value=sum(acc.net_liquidation for acc in enabled_accounts),
                available_cash=sum(acc.settled_funds for acc in enabled_accounts),
                emergency_mode=self.emergency_mode,
                market_condition='UNKNOWN',  # Would get from risk assessment
                portfolio_drawdown_pct=0.0,  # Would calculate
                status="COMPLETED_ENHANCED_DAY_TRADE_PROTECTION",
                log_details=f"Actions: Buy={trades}, Additions={positions_added}, Wyckoff={wyckoff_sells}, Profit={profit_scales}, Emergency={emergency_exits}, DayTradesBlocked={day_trades_blocked}"
            )
            
            if total_actions > 0:
                self.logger.info("✅ Enhanced fractional bot with day trade protection completed with actions")
            else:
                self.logger.info("✅ Enhanced fractional bot with day trade protection completed (no actions needed)")
            
            if day_trades_blocked > 0:
                self.logger.warning(f"⚠️ Day trade protection blocked {day_trades_blocked} potential violations")
            
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


def manual_database_sync_fix():
    """ADDED: Run this to manually fix any database sync issues"""
    import sqlite3
    from pathlib import Path
    
    db_path = Path("data/trading_bot.db")
    bot_id = "enhanced_wyckoff_bot_v2"
    
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    print("🔧 Fixing database synchronization issues...")
    
    with sqlite3.connect(db_path) as conn:
        # Get all positions from main table
        positions = conn.execute('''
            SELECT symbol, account_type, total_shares, avg_cost, total_invested,
                   first_purchase_date, last_purchase_date, entry_phase, entry_strength
            FROM positions 
            WHERE bot_id = ? AND total_shares > 0
        ''', (bot_id,)).fetchall()
        
        print(f"📊 Found {len(positions)} positions to sync")
        
        for pos in positions:
            symbol, account_type, shares, avg_cost, invested, first_date, last_date, phase, strength = pos
            
            # Calculate enhanced metrics
            try:
                first_dt = datetime.strptime(first_date, '%Y-%m-%d')
                time_held = (datetime.now() - first_dt).days
            except:
                time_held = 0
            
            # Update or insert into enhanced table
            conn.execute('''
                INSERT OR REPLACE INTO positions_enhanced (
                    symbol, account_type, total_shares, avg_cost, total_invested,
                    first_purchase_date, last_purchase_date, entry_phase, 
                    entry_strength, position_size_pct, time_held_days,
                    volatility_percentile, bot_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, account_type, shares, avg_cost, invested,
                first_date, last_date, phase or 'UNKNOWN', strength or 0.0,
                0.1, time_held, 0.5, bot_id, datetime.now().isoformat()
            ))
            
            print(f"✅ Synced: {symbol} ({account_type}) - {shares:.5f} shares")
    
    print("✅ Database sync complete!")


def show_day_trade_report():
    """NEW: Show day trading compliance report"""
    import sqlite3
    from pathlib import Path
    
    db_path = Path("data/trading_bot.db")
    bot_id = "enhanced_wyckoff_bot_v2"
    
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    print("\n" + "="*80)
    print("DAY TRADING COMPLIANCE REPORT")
    print("="*80)
    
    with sqlite3.connect(db_path) as conn:
        # Today's day trade checks
        today = datetime.now().strftime('%Y-%m-%d')
        
        checks = conn.execute('''
            SELECT symbol, action, db_day_trade, actual_day_trade, 
                   manual_trades_detected, recommendation, emergency_override, created_at
            FROM day_trade_checks 
            WHERE check_date = ? AND bot_id = ?
            ORDER BY created_at DESC
        ''', (today, bot_id)).fetchall()
        
        if checks:
            print(f"\n📊 TODAY'S DAY TRADE CHECKS ({len(checks)} total):")
            blocked_count = 0
            allowed_count = 0
            override_count = 0
            
            for check in checks:
                symbol, action, db_dt, actual_dt, manual, recommendation, emergency, timestamp = check
                
                status_icon = "🚨" if recommendation == 'BLOCK' else "✅" if recommendation == 'ALLOW' else "⚠️"
                emergency_text = " [EMERGENCY OVERRIDE]" if emergency else ""
                
                print(f"   {status_icon} {symbol} {action}: {recommendation}{emergency_text}")
                print(f"      DB: {bool(db_dt)}, Actual: {bool(actual_dt)}, Manual: {bool(manual)} - {timestamp}")
                
                if recommendation == 'BLOCK':
                    blocked_count += 1
                elif recommendation == 'ALLOW':
                    allowed_count += 1
                elif emergency:
                    override_count += 1
            
            print(f"\n📈 SUMMARY:")
            print(f"   ✅ Allowed: {allowed_count}")
            print(f"   🚨 Blocked: {blocked_count}")
            print(f"   ⚠️ Emergency Overrides: {override_count}")
        else:
            print(f"\n📊 No day trade checks found for today")
        
        # Bot run summary with day trade info
        recent_runs = conn.execute('''
            SELECT run_date, trades_executed, day_trades_blocked, status
            FROM bot_runs 
            WHERE bot_id = ?
            ORDER BY created_at DESC 
            LIMIT 10
        ''', (bot_id,)).fetchall()
        
        if recent_runs:
            print(f"\n📊 RECENT BOT RUNS:")
            for run in recent_runs:
                run_date, trades, blocked, status = run
                print(f"   {run_date}: {trades} trades, {blocked} day trades blocked - {status}")
    
    print("\n✅ Day trade report complete!")


def main():
    """Main entry point"""
    print("🚀 Enhanced Fractional Trading Bot with Real Account Day Trade Protection Starting...")
    
    bot = EnhancedFractionalTradingBot()
    success = bot.run()
    
    if success:
        print("✅ Trading bot with day trade protection successfully completedy!")
        sys.exit(0)
    else:
        print("❌ Trading bot with day trade protection failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Uncomment to run manual analysis instead of trading
    # run_manual_analysis()
    
    # Uncomment to run database sync fix
    # manual_database_sync_fix()
    
    # Uncomment to show day trading compliance report
    # show_day_trade_report()
    
    main()