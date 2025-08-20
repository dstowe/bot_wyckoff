# compliance/day_trade_checker.py
"""
Real Account Day Trade Checker
=============================
Extracted from fractional_position_system.py - Day trade compliance checking
Handles comprehensive day trade compliance using Webull API
"""

from datetime import datetime
from typing import Dict, List
from database import DayTradeCheckResult


class RealAccountDayTradeChecker:
    """FIXED: Real account day trade checking using Webull API"""
    
    def __init__(self, logger):
        """
        Initialize the day trade checker
        
        Args:
            logger: Logger instance for tracking compliance checks
        """
        self.logger = logger
        self.trade_cache = {}  # Cache to avoid repeated API calls
        self.last_cache_update = None
    
    def get_actual_todays_trades(self, wb_client, symbol: str = None) -> List[Dict]:
        """FIXED: Handle Webull date format MM/dd/yyyy"""
        try:
            today = datetime.now()
            today_str = today.strftime('%Y-%m-%d')  # 2025-08-19
            today_webull = today.strftime('%m/%d/%Y')  # 08/19/2025 format for Webull
            
            self.logger.debug(f"📊 Checking actual trades for {today_str} (Webull: {today_webull})")
            
            # Cache key for today's trades
            cache_key = f"trades_{today_str}"
            
            # Check cache first (avoid repeated API calls)
            if (cache_key in self.trade_cache and 
                self.last_cache_update and 
                (datetime.now() - self.last_cache_update).seconds < 300):  # 5 min cache
                
                self.logger.debug("📊 Using cached trade data")
                all_trades = self.trade_cache[cache_key]
            else:
                # Fetch fresh data from Webull
                try:
                    # FIXED: Get all trades for today - handle potential API variations
                    all_trades = []
                    
                    # Try different Webull API methods based on available interface
                    if hasattr(wb_client, 'get_account_trades'):
                        trades_response = wb_client.get_account_trades(start_date=today_webull, end_date=today_webull)
                        if trades_response and isinstance(trades_response, list):
                            all_trades = trades_response
                    elif hasattr(wb_client, 'get_orders'):
                        # Alternative method - get orders and filter for today
                        orders_response = wb_client.get_orders()
                        if orders_response and isinstance(orders_response, list):
                            for order in orders_response:
                                if (order.get('time', '').startswith(today_str) or 
                                    order.get('createTime', '').startswith(today_str) or
                                    order.get('orderTime', '').startswith(today_str)):
                                    all_trades.append(order)
                    
                    # Update cache
                    self.trade_cache[cache_key] = all_trades
                    self.last_cache_update = datetime.now()
                    
                    self.logger.debug(f"📊 Found {len(all_trades)} total trades for today")
                    
                except Exception as api_error:
                    self.logger.warning(f"⚠️ Webull API error: {api_error}")
                    all_trades = []
            
            # Filter by symbol if specified
            if symbol:
                symbol_trades = []
                for trade in all_trades:
                    # Handle different possible symbol field names
                    trade_symbol = (trade.get('symbol') or 
                                  trade.get('ticker') or 
                                  trade.get('instrument', {}).get('symbol', ''))
                    
                    if trade_symbol.upper() == symbol.upper():
                        symbol_trades.append(trade)
                
                self.logger.debug(f"📊 Found {len(symbol_trades)} trades for {symbol} today")
                return symbol_trades
            
            return all_trades
            
        except Exception as e:
            self.logger.error(f"❌ Error getting actual trades: {e}")
            return []
    
    def detect_manual_trades(self, wb_client, database, symbol: str, account_manager) -> bool:
        """Detect manual trades by comparing real vs database positions"""
        try:
            self.logger.debug(f"🔍 Checking for manual trades in {symbol}...")
            
            # Get database position
            db_positions = database.get_all_positions()
            db_total_shares = 0.0
            
            for account_type, positions in db_positions.items():
                if symbol in positions:
                    db_total_shares += positions[symbol].get('total_shares', 0.0)
            
            # Get actual position from enabled accounts
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