#!/usr/bin/env python3
"""
Quick check to see what positions actually exist right now
"""

import json
from main import MainSystem

def quick_position_check():
    """Check what positions actually exist in each account"""
    
    print("🔍 QUICK POSITION REALITY CHECK")
    print("="*50)
    
    # Initialize system
    main_system = MainSystem()
    if not main_system.run():
        print("❌ Failed to initialize system")
        return
    
    try:
        enabled_accounts = main_system.account_manager.get_enabled_accounts()
        
        print(f"\n📊 ACCOUNT-BY-ACCOUNT POSITION CHECK:")
        
        total_positions_found = 0
        
        for account in enabled_accounts:
            print(f"\n🏦 {account.account_type} (ID: {account.account_id}):")
            print(f"   Balance: ${account.net_liquidation:.2f}")
            print(f"   Available: ${account.settled_funds:.2f}")
            print(f"   Reported positions: {len(account.positions)}")
            
            # Switch to this account explicitly
            print(f"   🔄 Switching to this account...")
            switch_result = main_system.account_manager.switch_to_account(account)
            print(f"   Switch result: {switch_result}")
            print(f"   Webull context: {main_system.wb._account_id}")
            
            # Get account data directly
            print(f"   📋 Getting account data...")
            account_data = main_system.wb.get_account()
            
            if account_data:
                print(f"   ✅ Account data retrieved")
                
                # Check positions in the API response
                if 'positions' in account_data and account_data['positions']:
                    positions = account_data['positions']
                    print(f"   📈 API shows {len(positions)} positions:")
                    
                    for pos in positions:
                        symbol = pos.get('ticker', {}).get('symbol', 'UNKNOWN')
                        quantity = float(pos.get('position', 0))
                        cost_price = float(pos.get('costPrice', 0))
                        current_price = float(pos.get('lastPrice', 0))
                        market_value = float(pos.get('marketValue', 0))
                        pnl = float(pos.get('unrealizedProfitLoss', 0))
                        
                        if quantity > 0:  # Only show actual positions
                            print(f"      📊 {symbol}: {quantity:.5f} shares")
                            print(f"         Cost: ${cost_price:.2f}, Current: ${current_price:.2f}")
                            print(f"         Value: ${market_value:.2f}, P&L: ${pnl:.2f}")
                            total_positions_found += 1
                        else:
                            print(f"      ⚪ {symbol}: 0 shares (closed position)")
                else:
                    print(f"   📊 No positions found in API response")
                
                # Also check the pre-parsed positions from account discovery
                print(f"   🔍 Account manager parsed positions:")
                if account.positions:
                    for pos in account.positions:
                        symbol = pos['symbol']
                        quantity = pos['quantity']
                        cost_price = pos['cost_price']
                        market_value = pos['market_value']
                        
                        if quantity > 0:
                            print(f"      📊 {symbol}: {quantity:.5f} shares @ ${cost_price:.2f} = ${market_value:.2f}")
                else:
                    print(f"      No parsed positions")
            else:
                print(f"   ❌ Could not get account data")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total positions found across all accounts: {total_positions_found}")
        
        if total_positions_found == 0:
            print(f"\n🤔 POSSIBLE EXPLANATIONS:")
            print(f"   1. You manually sold all positions")
            print(f"   2. Positions are in fractional amounts that don't show up")
            print(f"   3. Account switching during check is still broken")
            print(f"   4. Webull API is having issues")
            
            print(f"\n💡 NEXT STEPS:")
            print(f"   1. Check your Webull app/website manually")
            print(f"   2. If positions exist there but not here, account switching is still broken")
            print(f"   3. If positions don't exist there either, you sold them")
        else:
            print(f"\n✅ POSITIONS FOUND!")
            print(f"   The reconciliation issue is likely in the database comparison logic")
    
    except Exception as e:
        print(f"❌ Error during position check: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        main_system.cleanup()

if __name__ == "__main__":
    quick_position_check()