# -*- coding: utf-8 -*-
"""
Enhanced Cash Management - Usage Examples
How to use the new cash management features in your trading bot
"""

from enhanced_cash_manager import EnhancedCashManager

def example_position_sizing():
    """Example of how to use enhanced position sizing"""
    cash_manager = EnhancedCashManager()
    
    # Your account information
    account_value = 100000  # $100k account
    current_cash = 25000    # $25k cash
    stock_price = 50.0      # Stock costs $50/share
    
    # Get cash management recommendations
    recommendations = cash_manager.get_recommended_cash_allocation(account_value)
    
    print(f"Market Regime: {recommendations['market_regime']}")
    print(f"Recommended Cash: ${recommendations['recommended_cash_amount']:,.2f}")
    print(f"Available for Trading: ${recommendations['available_for_trading']:,.2f}")
    print(f"Max Position Size: ${recommendations['max_position_value']:,.2f}")
    
    # Calculate position size
    max_shares = int(recommendations['max_position_value'] / stock_price)
    
    # Apply regime-based reduction if needed
    should_reduce, reduction_factor = cash_manager.should_reduce_position_sizes()
    if should_reduce:
        max_shares = int(max_shares * reduction_factor)
        print(f"Position reduced by {(1-reduction_factor)*100:.0f}% due to market conditions")
    
    print(f"Maximum shares to buy: {max_shares}")
    
    return max_shares

def example_trade_validation():
    """Example of validating trades against cash rules"""
    cash_manager = EnhancedCashManager()
    
    # Trade parameters
    account_value = 50000
    current_cash = 15000
    trade_value = 8000  # Want to buy $8k worth of stock
    
    # Validate the trade
    is_valid, reason = cash_manager.validate_trade_against_cash_rules(
        trade_value, current_cash, account_value
    )
    
    if is_valid:
        print(f"✅ Trade approved: {reason}")
        # Proceed with trade
    else:
        print(f"❌ Trade rejected: {reason}")
        # Don't execute trade
    
    return is_valid

def example_integration_in_bot():
    """Example of how to integrate into your existing bot"""
    
    class TradingBot:
        def __init__(self):
            # Your existing initialization
            self.cash_manager = EnhancedCashManager()
        
        def calculate_position_size(self, account_value, stock_price):
            """Enhanced position sizing method"""
            # Get dynamic recommendations
            recommendations = self.cash_manager.get_recommended_cash_allocation(account_value)
            
            # Use dynamic max position value
            max_position_value = recommendations['max_position_value']
            
            # Apply regime-based reduction
            should_reduce, reduction_factor = self.cash_manager.should_reduce_position_sizes()
            if should_reduce:
                max_position_value *= reduction_factor
            
            return int(max_position_value / stock_price)
        
        def execute_trade(self, symbol, quantity, price):
            """Enhanced trade execution with cash validation"""
            trade_value = quantity * price
            current_cash = self.get_current_cash()  # Your method
            account_value = self.get_account_value()  # Your method
            
            # Validate trade
            is_valid, reason = self.cash_manager.validate_trade_against_cash_rules(
                trade_value, current_cash, account_value
            )
            
            if not is_valid:
                print(f"Trade rejected: {reason}")
                return False
            
            # Proceed with your existing trade logic
            return self.place_order(symbol, quantity, price)

if __name__ == "__main__":
    print("Enhanced Cash Management - Usage Examples")
    print("=" * 50)
    
    print("\n1. Position Sizing Example:")
    example_position_sizing()
    
    print("\n2. Trade Validation Example:")
    example_trade_validation()
    
    print("\n3. Check enhanced_cash_manager.py for full implementation details")
