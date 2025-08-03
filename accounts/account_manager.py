# accounts/account_manager.py
import logging
from datetime import datetime
from typing import Dict, List, Optional
from .account_info import AccountInfo

class AccountManager:
    """Account Manager - Fully integrated with PersonalTradingConfig"""
    
    def __init__(self, wb, config, logger=None):
        self.wb = wb
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Account data
        self.accounts: Dict[str, AccountInfo] = {}
        self.current_account: Optional[AccountInfo] = None
        self.original_account_id = None
        self.original_zone = None
        
        # Account type mapping from Webull API
        self.account_type_mapping = {
            'CASH': 'Cash Account',
            'MRGN': 'Margin Account', 
            'IRA': 'IRA Account',
            'ROTH': 'Roth IRA Account'
        }
        
        self.broker_name_mapping = {
            'Individual': 'Individual Brokerage',
            'IRA': 'IRA Account',
            'Advisor': 'Advisor Account'
        }
    
    def discover_accounts(self) -> bool:
        """Discover all available accounts using proper Webull API"""
        try:
            self.logger.debug("🔍 Discovering Webull accounts automatically...")
            
            if not self.wb:
                self.logger.error("❌ Webull session not initialized")
                return False
            
            # Store original account settings
            self.original_account_id = getattr(self.wb, '_account_id', None)
            self.original_zone = getattr(self.wb, 'zone_var', None)
            
            # Get all account IDs using proper API
            headers = self.wb.build_req_headers()
            response = self.wb._session.get(self.wb._urls.account_id(), headers=headers, timeout=self.wb.timeout)
            result = response.json()
            
            self.logger.debug(f"Account discovery API response keys: {list(result.keys())}")
            
            if not result.get('success') or not result.get('data'):
                self.logger.error("❌ Failed to retrieve account list from Webull API")
                self.logger.error(f"   Response: {result}")
                return False
            
            accounts_data = result['data']
            self.logger.info(f"📊 Found {len(accounts_data)} account(s) from Webull API")
            
            # Process each account automatically
            active_accounts = 0
            for i, account_info in enumerate(accounts_data):
                self.logger.debug(f"Processing account {i+1}: {account_info.get('secAccountId', 'Unknown ID')}")
                
                account_id = account_info.get('secAccountId')
                status = account_info.get('status', 'Unknown')
                rzone = account_info.get('rzone', 'dc_core_r001')  # Provide default
                
                self.logger.debug(f"   Account ID: {account_id}")
                self.logger.debug(f"   Status: {status}")
                self.logger.debug(f"   RZone: {rzone}")
                
                if status != 'active' or not account_id:
                    self.logger.debug(f"⏭️ Skipping {status} account: {account_info.get('brokerName', 'Unknown')}")
                    continue
                
                # Determine account type
                account_type = self._determine_account_type(account_info)
                self.logger.debug(f"   Determined account type: {account_type}")
                
                # Create AccountInfo object
                account = AccountInfo(
                    account_id=str(account_id),
                    account_type=account_type,
                    status=status,
                    broker_name=account_info.get('brokerName', 'Unknown'),
                    broker_account_id=account_info.get('brokerAccountId', 'N/A'),
                    is_default=account_info.get('isDefault', False),
                    zone=str(rzone)  # Use the rzone from API response
                )
                
                self.accounts[str(account_id)] = account
                active_accounts += 1
                
                # Load detailed account info
                if self._load_account_details(account):
                    self.logger.debug(f"✅ {account_type} Account loaded: ${account.net_liquidation:.2f}")
                else:
                    self.logger.warning(f"⚠️ Could not load details for {account_type} account")
            
            if not self.accounts:
                self.logger.error("❌ No active accounts found")
                return False
            
            self.logger.info(f"✅ Successfully discovered {active_accounts} active accounts")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error discovering accounts: {str(e)}")
            self.logger.debug(f"Full exception details:", exc_info=True)
            return False
    
    def _determine_account_type(self, account_info: Dict) -> str:
        """Determine account type from Webull API response"""
        # Try account types first
        if 'accountTypes' in account_info and account_info['accountTypes']:
            account_types = account_info['accountTypes']
            if account_types:
                primary_type = account_types[0]
                mapped_type = self.account_type_mapping.get(primary_type, primary_type)
                self.logger.debug(f"   Account type from accountTypes: {primary_type} -> {mapped_type}")
                return mapped_type
        
        # Fallback to broker name
        broker_name = account_info.get('brokerName', 'Unknown')
        mapped_type = self.broker_name_mapping.get(broker_name, broker_name)
        self.logger.debug(f"   Account type from brokerName: {broker_name} -> {mapped_type}")
        return mapped_type
    
    def _load_account_details(self, account: AccountInfo) -> bool:
        """Load detailed information for a specific account"""
        try:
            self.logger.debug(f"Loading details for account {account.account_id} ({account.account_type})")
            
            # Switch to this account
            self.wb._account_id = account.account_id
            self.wb.zone_var = account.zone
            
            self.logger.debug(f"   Set wb._account_id to: {self.wb._account_id}")
            self.logger.debug(f"   Set wb.zone_var to: {self.wb.zone_var}")
            
            # Get account details
            account_data = self.wb.get_account()
            
            if not account_data:
                self.logger.warning(f"⚠️ No account data returned for {account.account_id}")
                return False
            
            self.logger.debug(f"   Account data keys: {list(account_data.keys()) if isinstance(account_data, dict) else type(account_data)}")
            
            # Extract net liquidation from TOP-LEVEL first
            if 'netLiquidation' in account_data:
                account.net_liquidation = float(account_data['netLiquidation'])
                self.logger.debug(f"💰 Found top-level netLiquidation: ${account.net_liquidation:.2f}")
            else:
                # Fallback to accountMembers
                for member in account_data.get('accountMembers', []):
                    key = member.get('key', '')
                    value = member.get('value', '')
                    
                    if key == 'netLiquidation':
                        account.net_liquidation = float(value)
                        self.logger.debug(f"💰 Found netLiquidation in accountMembers: ${account.net_liquidation:.2f}")
                        break
                    elif key == 'totalMarketValue':
                        account.net_liquidation = float(value)
                        self.logger.debug(f"💰 Using totalMarketValue as netLiquidation: ${account.net_liquidation:.2f}")
                        break
            
            # Extract available funds based on account type
            account.settled_funds = 0.0
            
            for member in account_data.get('accountMembers', []):
                key = member.get('key', '')
                value = member.get('value', '')
                
                # For cash accounts, use settledFunds
                if key == 'settledFunds' and account.account_type in ['Cash Account', 'CASH']:
                    account.settled_funds = float(value)
                    self.logger.debug(f"💵 Found settledFunds (cash): ${account.settled_funds:.2f}")
                    break
                
                # For margin accounts, use cashBalance
                elif key == 'cashBalance' and account.account_type in ['Margin Account', 'MRGN']:
                    account.settled_funds = float(value)
                    self.logger.debug(f"💵 Found cashBalance (margin): ${account.settled_funds:.2f}")
                    break
                
                # Fallback: use cashBalance for any account if settledFunds not found
                elif key == 'cashBalance' and account.settled_funds == 0:
                    account.settled_funds = float(value)
                    self.logger.debug(f"💵 Using cashBalance as fallback: ${account.settled_funds:.2f}")
            
            # If still no funds found, try alternative fields
            if account.settled_funds == 0:
                for member in account_data.get('accountMembers', []):
                    key = member.get('key', '')
                    value = member.get('value', '')
                    
                    if key in ['dayBuyingPower', 'availableFunds', 'buyingPower']:
                        account.settled_funds = float(value)
                        self.logger.debug(f"💵 Using {key} as funds: ${account.settled_funds:.2f}")
                        break
            
            # Extract positions
            account.positions = []
            for position in account_data.get('positions', []):
                pos_data = {
                    'symbol': position['ticker']['symbol'],
                    'quantity': float(position['position']),
                    'cost_price': float(position['costPrice']),
                    'current_price': float(position['lastPrice']),
                    'market_value': float(position['marketValue']),
                    'unrealized_pnl': float(position['unrealizedProfitLoss']),
                    'pnl_rate': float(position['unrealizedProfitLossRate']),
                    'last_open_time': position['lastOpenTime']
                }
                account.positions.append(pos_data)
            
            # Load day trading information
            self._load_day_trading_info(account, account_data)



            
            # Log final values
            self.logger.debug(f"📊 Final values for {account.account_type}:")
            self.logger.debug(f"   Net Liquidation: ${account.net_liquidation:.2f}")
            self.logger.debug(f"   Available Funds: ${account.settled_funds:.2f}")
            self.logger.debug(f"   Positions: {len(account.positions)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading account details for {account.account_id}: {str(e)}")
            self.logger.debug(f"Full exception details:", exc_info=True)
            return False

    def _load_day_trading_info(self, account: AccountInfo, account_data: Dict):
        """Load day trading information for PDT tracking"""
        try:
            # Extract PDT status from top level
            account.pdt_status = account_data.get('pdt', False)
            
            # Extract day trades remaining from accountMembers
            account.day_trades_remaining = None
            
            for member in account_data.get('accountMembers', []):
                key = member.get('key', '')
                value = member.get('value', '')
                
                if key == 'remainTradeTimes':
                    # Parse the remainTradeTimes string (e.g., "2,2,2,2,2")
                    try:
                        if isinstance(value, str) and value:
                            if value.lower() == 'unlimited':
                                # Cash accounts often show "Unlimited"
                                account.day_trades_remaining = 999
                                self.logger.debug(f"   Day trades remaining: Unlimited (set to 999)")
                            else:
                                # Split by comma and get the minimum (most restrictive day)
                                day_trades_list = [int(x.strip()) for x in value.split(',') if x.strip().isdigit()]
                                if day_trades_list:
                                    account.day_trades_remaining = min(day_trades_list)
                                    self.logger.debug(f"   Day trades remaining: {account.day_trades_remaining} (from {value})")
                                else:
                                    self.logger.warning(f"   Could not parse remainTradeTimes: {value}")
                        elif isinstance(value, (int, float)):
                            account.day_trades_remaining = int(value)
                    except Exception as e:
                        self.logger.warning(f"   Error parsing remainTradeTimes '{value}': {e}")
                        
            # Set defaults if not found
            if account.day_trades_remaining is None:
                if account.account_type in ['Cash Account', 'CASH']:
                    # Cash accounts don't have day trade limits
                    account.day_trades_remaining = 999  # Unlimited for cash
                elif account.pdt_status:
                    # PDT accounts (>=25K) have unlimited day trades
                    account.day_trades_remaining = 999
                else:
                    # Default for margin accounts without data
                    account.day_trades_remaining = 0
                    
            account.day_trades_used = 0  # Reset daily (would need to track this)
            account.last_day_trade_reset = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.debug(f"📊 Day Trading Info for {account.account_type}:")
            self.logger.debug(f"   PDT Status: {account.pdt_status}")
            self.logger.debug(f"   Day Trades Remaining: {account.day_trades_remaining}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load day trading info for {account.account_id}: {e}")
            # Set safe defaults
            account.day_trades_remaining = 0 if account.account_type not in ['Cash Account', 'CASH'] else 999
            account.pdt_status = False
            account.day_trades_used = 0
            account.last_day_trade_reset = datetime.now().strftime('%Y-%m-%d')
    
    def get_enabled_accounts(self) -> List[AccountInfo]:
        """Get list of accounts enabled for trading based on PersonalTradingConfig"""
        enabled_accounts = []
        
        for account in self.accounts.values():
            if account.is_enabled_for_trading(self.config):
                enabled_accounts.append(account)
                self.logger.debug(f"✅ Account enabled for trading: {account.account_type} (${account.settled_funds:.2f})")
            else:
                self.logger.debug(f"❌ Account disabled: {account.account_type}")
        
        return enabled_accounts
    
    def switch_to_account(self, account: AccountInfo) -> bool:
        """Switch trading context to specified account"""
        try:
            self.logger.debug(f"🔄 Switching to {account.account_type} account: {account.account_id}")
            
            # Switch Webull client to this account
            self.wb._account_id = account.account_id
            self.wb.zone_var = account.zone
            
            # Update current account
            self.current_account = account
            
            self.logger.debug(f"✅ Switched to {account.account_type} account")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error switching to account {account.account_id}: {str(e)}")
            return False
    
    def get_account_summary(self) -> Dict:
        """Get summary of all accounts using PersonalTradingConfig"""
        summary = {
            'total_accounts': len(self.accounts),
            'enabled_accounts': len(self.get_enabled_accounts()),
            'total_value': 0.0,
            'total_cash': 0.0,
            'total_positions': 0,
            'accounts': []
        }
        
        for account in self.accounts.values():
            account_config = account.get_account_config(self.config)
            
            account_summary = {
                'account_id': account.account_id,
                'account_type': account.account_type,
                'enabled': account.is_enabled_for_trading(self.config),
                'net_liquidation': account.net_liquidation,
                'settled_funds': account.settled_funds,
                'positions_count': len(account.positions),
                'day_trading_enabled': account_config.get('day_trading_enabled', False),
                'options_enabled': account_config.get('options_enabled', False),
                'day_trades_used': account.day_trades_used
            }
            
            summary['accounts'].append(account_summary)
            summary['total_value'] += account.net_liquidation
            summary['total_cash'] += account.settled_funds
            summary['total_positions'] += len(account.positions)
        
        return summary
    
    def refresh_account_details(self, account: AccountInfo) -> bool:
        """Refresh account details after a trade or other change"""
        return self._load_account_details(account)
    
    def get_account_by_id(self, account_id: str) -> Optional[AccountInfo]:
        """Get account by ID"""
        return self.accounts.get(account_id)
    
    def get_account_by_type(self, account_type: str) -> Optional[AccountInfo]:
        """Get first account of specified type"""
        for account in self.accounts.values():
            if account.account_type == account_type:
                return account
        return None
    
    def restore_original_account(self) -> bool:
        """Restore to the original account context"""
        try:
            if self.original_account_id and self.original_zone:
                self.wb._account_id = self.original_account_id
                self.wb.zone_var = self.original_zone
                self.current_account = self.get_account_by_id(self.original_account_id)
                self.logger.debug("✅ Restored to original account context")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Error restoring original account: {e}")
            return False