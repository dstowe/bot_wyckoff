#!/usr/bin/env python3
"""
Debug account switching mechanism to find why all orders go to cash account
"""

import json
from main import MainSystem
from datetime import datetime

def debug_account_switching():
    """Test account switching mechanism step by step"""
    
    print("🔍 DEBUGGING ACCOUNT SWITCHING MECHANISM")
    print("="*60)
    
    # Initialize system
    main_system = MainSystem()
    if not main_system.run():
        print("❌ Failed to initialize system")
        return
    
    try:
        enabled_accounts = main_system.account_manager.get_enabled_accounts()
        
        print(f"\n📊 DISCOVERED ACCOUNTS:")
        for i, account in enumerate(enabled_accounts):
            print(f"   {i+1}. {account.account_type}")
            print(f"      ID: {account.account_id}")
            print(f"      Zone: {account.zone}")
            print(f"      Balance: ${account.net_liquidation:.2f}")
            print(f"      Available: ${account.settled_funds:.2f}")
        
        print(f"\n🔧 INITIAL WEBULL STATE:")
        print(f"   wb._account_id: {main_system.wb._account_id}")
        print(f"   wb.zone_var: {main_system.wb.zone_var}")
        
        # Test switching to each account
        for i, account in enumerate(enabled_accounts):
            print(f"\n🔄 TESTING SWITCH TO {account.account_type}:")
            print(f"   Target Account ID: {account.account_id}")
            print(f"   Target Zone: {account.zone}")
            
            # Store current state
            before_account_id = main_system.wb._account_id
            before_zone = main_system.wb.zone_var
            
            print(f"   Before - wb._account_id: {before_account_id}")
            print(f"   Before - wb.zone_var: {before_zone}")
            
            # Attempt switch
            switch_result = main_system.account_manager.switch_to_account(account)
            print(f"   Switch Result: {switch_result}")
            
            # Check state after switch
            after_account_id = main_system.wb._account_id
            after_zone = main_system.wb.zone_var
            
            print(f"   After - wb._account_id: {after_account_id}")
            print(f"   After - wb.zone_var: {after_zone}")
            
            # Verify the switch actually worked
            if after_account_id == account.account_id and after_zone == account.zone:
                print(f"   ✅ Account switch successful")
            else:
                print(f"   ❌ Account switch FAILED!")
                print(f"      Expected ID: {account.account_id}, Got: {after_account_id}")
                print(f"      Expected Zone: {account.zone}, Got: {after_zone}")
            
            # Test API call to verify the active account
            print(f"   🧪 Testing API call with switched account...")
            try:
                # Use get_account() to verify which account is actually active
                account_data = main_system.wb.get_account()
                
                if account_data and 'accountId' in account_data:
                    active_account_id = str(account_data['accountId'])
                    print(f"   API Reports Active Account ID: {active_account_id}")
                    
                    if active_account_id == account.account_id:
                        print(f"   ✅ API confirms correct account is active")
                    else:
                        print(f"   ❌ API shows DIFFERENT account is active!")
                        print(f"      Expected: {account.account_id}")
                        print(f"      API Says: {active_account_id}")
                else:
                    print(f"   ⚠️ Could not determine active account from API")
                    print(f"   Account data keys: {list(account_data.keys()) if account_data else 'None'}")
                    
            except Exception as api_error:
                print(f"   ❌ API call failed: {api_error}")
            
            print(f"   " + "-"*50)
        
        # Test with a small quote request to see which account responds
        print(f"\n🧪 TESTING QUOTE REQUESTS PER ACCOUNT:")
        for account in enabled_accounts:
            print(f"\n   Testing {account.account_type} account:")
            
            # Switch to account
            main_system.account_manager.switch_to_account(account)
            
            try:
                # Get quote (this should work regardless of account)
                quote = main_system.wb.get_quote('AAPL')
                if quote and 'close' in quote:
                    print(f"   ✅ Quote successful: AAPL = ${quote['close']}")
                else:
                    print(f"   ❌ Quote failed or incomplete")
                    
                # More importantly, check if we can get account-specific data
                account_info = main_system.wb.get_account()
                if account_info:
                    # Look for account identifier in the response
                    reported_id = None
                    if 'accountId' in account_info:
                        reported_id = str(account_info['accountId'])
                    elif 'secAccountId' in account_info:
                        reported_id = str(account_info['secAccountId'])
                    
                    if reported_id:
                        print(f"   📋 API reports account ID: {reported_id}")
                        if reported_id == account.account_id:
                            print(f"   ✅ Correct account active")
                        else:
                            print(f"   ❌ WRONG account active! Expected: {account.account_id}")
                    else:
                        print(f"   ⚠️ Could not find account ID in API response")
                        print(f"   Available keys: {list(account_info.keys())}")
                        
            except Exception as e:
                print(f"   ❌ Error testing account: {e}")
        
        # Test order placement to see which account it would actually use
        print(f"\n🧪 DRY RUN ORDER TEST (NO ACTUAL ORDERS):")
        print("   This will test the order preparation without executing...")
        
        for account in enabled_accounts:
            print(f"\n   Testing order preparation for {account.account_type}:")
            
            # Switch to account
            switch_success = main_system.account_manager.switch_to_account(account)
            print(f"   Switch successful: {switch_success}")
            
            if switch_success:
                print(f"   Webull state - Account ID: {main_system.wb._account_id}")
                print(f"   Webull state - Zone: {main_system.wb.zone_var}")
                
                # Check what account the order would actually target
                # We can see this by looking at the URL that would be used
                try:
                    # The place_orders URL includes the account ID
                    expected_url = main_system.wb._urls.place_orders(main_system.wb._account_id)
                    print(f"   Order would target URL: {expected_url}")
                    print(f"   This targets account ID: {main_system.wb._account_id}")
                    
                    if main_system.wb._account_id == account.account_id:
                        print(f"   ✅ Order would target correct account")
                    else:
                        print(f"   ❌ Order would target WRONG account!")
                        print(f"      Expected: {account.account_id}")
                        print(f"      Would target: {main_system.wb._account_id}")
                        
                except Exception as e:
                    print(f"   ❌ Error checking order target: {e}")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        main_system.cleanup()

if __name__ == "__main__":
    debug_account_switching()