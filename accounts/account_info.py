# accounts/account_info.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AccountInfo:
    """Data class for account information"""
    account_id: str
    account_type: str  # 'CASH', 'MARGIN', 'IRA', etc.
    status: str        # 'active', 'unopen', etc.
    broker_name: str
    broker_account_id: str
    is_default: bool
    zone: str
    net_liquidation: float = 0.0
    settled_funds: float = 0.0
    positions: List[Dict] = None
    day_trades_used: int = 0
    last_day_trade_reset: str = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = []
    
    def get_allocation_info(self, config) -> Dict:
        """Get detailed account allocation information"""
        total_position_value = sum(pos.get('market_value', 0) for pos in self.positions)
        cash_percentage = (self.settled_funds / self.net_liquidation * 100) if self.net_liquidation > 0 else 0
        positions_percentage = (total_position_value / self.net_liquidation * 100) if self.net_liquidation > 0 else 0
        
        # Calculate maximum new position size
        max_new_position = self.net_liquidation * config.MAX_POSITION_VALUE_PERCENT
        max_affordable_with_cash = self.settled_funds
        
        return {
            'account_value': self.net_liquidation,
            'settled_funds': self.settled_funds,
            'total_position_value': total_position_value,
            'cash_percentage': cash_percentage,
            'positions_percentage': positions_percentage,
            'max_new_position': min(max_new_position, max_affordable_with_cash),
            'positions_count': len(self.positions),
            'max_positions': config.MAX_POSITIONS_TOTAL,
            'available_position_slots': max(0, config.MAX_POSITIONS_TOTAL - len(self.positions))
        }
    

    
    def is_enabled_for_trading(self, config) -> bool:
        """Check if this account is enabled for trading based on config"""
        config_key = self.account_type.upper().replace(' ACCOUNT', '')
        account_config = config.ACCOUNT_CONFIGURATIONS.get(config_key, {})
        return account_config.get('enabled', False)
    
    def get_account_config(self, config) -> Dict:
        """Get the configuration for this account type"""
        config_key = self.account_type.upper().replace(' ACCOUNT', '')
        return config.ACCOUNT_CONFIGURATIONS.get(config_key, {})
    
    def can_day_trade(self, config) -> bool:
            """Check if this account can perform day trading"""
            account_config = self.get_account_config(config)
            if not account_config.get('day_trading_enabled', False):
                return False
            
            # Cash accounts can always day trade (using settled funds)
            if self.account_type in ['Cash Account', 'CASH']:
                return True
            
            # For margin accounts, check day trades remaining first
            if hasattr(self, 'day_trades_remaining') and self.day_trades_remaining is not None:
                if self.day_trades_remaining >= 1:
                    return True
                else:
                    return False  # No day trades remaining
            
            # Fallback to PDT protection requirements (for accounts >= $25K)
            if account_config.get('pdt_protection', False):
                min_value = account_config.get('min_account_value_for_pdt', 25000)
                return self.net_liquidation >= min_value
            
            return False  # Default to false for margin accounts without day trade data
    
    def get_position_limits(self, config) -> Dict:
        """Get position limits for this account"""
        account_config = self.get_account_config(config)
        
        return {
            'max_position_size': account_config.get('max_position_size', config.MAX_POSITION_VALUE_PERCENT),
            'min_trade_amount': account_config.get('min_trade_amount', config.MIN_POSITION_VALUE),
            'max_trade_amount': account_config.get('max_trade_amount', float('inf')),
            'max_positions_total': config.MAX_POSITIONS_TOTAL
        }
    
    def __str__(self) -> str:
        return f"AccountInfo({self.account_type}, ID: {self.account_id}, Value: ${self.net_liquidation:.2f})"
    
    def __repr__(self) -> str:
        return self.__str__()