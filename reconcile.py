#!/usr/bin/env python3
"""
Reconcile database positions with actual Webull account positions
Fix the account assignments that were wrong due to the session bug
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from main import MainSystem

def reconcile_database_with_reality():
    """Fix database to match actual Webull account positions"""
    
    print("🔄 RECONCILING DATABASE WITH ACTUAL WEBULL ACCOUNTS")
    print("="*60)
    
    # Initialize system
    main_system = MainSystem()
    if not main_system.run():
        print("❌ Failed to initialize system")
        return
    
    try:
        enabled_accounts = main_system.account_manager.get_enabled_accounts()
        db_path = Path("data/trading_bot.db")
        bot_id = "enhanced_wyckoff_bot_v2"
        
        print(f"\n📊 STEP 1: GET ACTUAL WEBULL POSITIONS")
        
        # Get actual positions from each Webull account
        actual_positions = {}
        
        for account in enabled_accounts:
            print(f"\n   Checking {account.account_type} (ID: {account.account_id}):")
            
            # Switch to account and get positions
            main_system.account_manager.switch_to_account(account)
            account_data = main_system.wb.get_account()
            
            if account_data and 'positions' in account_data:
                positions = account_data['positions']
                print(f"      Found {len(positions)} positions")
                
                for pos in positions:
                    symbol = pos.get('ticker', {}).get('symbol', 'UNKNOWN')
                    quantity = float(pos.get('position', 0))
                    cost_price = float(pos.get('costPrice', 0))
                    market_value = float(pos.get('marketValue', 0))
                    
                    if quantity > 0:  # Only include actual positions
                        actual_positions[symbol] = {
                            'symbol': symbol,
                            'account_type': account.account_type,
                            'account_id': account.account_id,
                            'quantity': quantity,
                            'cost_price': cost_price,
                            'market_value': market_value
                        }
                        print(f"         {symbol}: {quantity:.5f} shares @ ${cost_price:.2f}")
            else:
                print(f"      No positions found")
        
        print(f"\n📋 STEP 2: GET DATABASE POSITIONS")
        
        # Get database positions
        db_positions = {}
        with sqlite3.connect(db_path) as conn:
            results = conn.execute('''
                SELECT symbol, account_type, total_shares, avg_cost, total_invested, 
                       first_purchase_date, last_purchase_date, entry_phase, entry_strength
                FROM positions 
                WHERE bot_id = ? AND total_shares > 0
                ORDER BY symbol
            ''', (bot_id,)).fetchall()
            
            print(f"   Found {len(results)} database positions:")
            for row in results:
                symbol, account_type, shares, avg_cost, invested, first_date, last_date, phase, strength = row
                db_positions[symbol] = {
                    'symbol': symbol,
                    'account_type': account_type,
                    'total_shares': shares,
                    'avg_cost': avg_cost,
                    'total_invested': invested,
                    'first_purchase_date': first_date,
                    'last_purchase_date': last_date,
                    'entry_phase': phase,
                    'entry_strength': strength
                }
                print(f"      {symbol}: {shares:.5f} shares @ ${avg_cost:.2f} in {account_type}")
        
        print(f"\n🔍 STEP 3: COMPARE AND IDENTIFY DISCREPANCIES")
        
        corrections_needed = []
        
        for symbol in set(list(actual_positions.keys()) + list(db_positions.keys())):
            actual = actual_positions.get(symbol)
            db = db_positions.get(symbol)
            
            if actual and db:
                # Position exists in both - check account type
                if actual['account_type'] != db['account_type']:
                    corrections_needed.append({
                        'symbol': symbol,
                        'issue': 'WRONG_ACCOUNT',
                        'db_account': db['account_type'],
                        'actual_account': actual['account_type'],
                        'actual_account_id': actual['account_id'],
                        'db_data': db,
                        'actual_data': actual
                    })
                    print(f"   ❌ {symbol}: DB says {db['account_type']}, actually in {actual['account_type']}")
                
                # Check quantities
                elif abs(actual['quantity'] - db['total_shares']) > 0.001:
                    corrections_needed.append({
                        'symbol': symbol,
                        'issue': 'QUANTITY_MISMATCH',
                        'db_quantity': db['total_shares'],
                        'actual_quantity': actual['quantity'],
                        'db_data': db,
                        'actual_data': actual
                    })
                    print(f"   ⚠️ {symbol}: Quantity mismatch - DB: {db['total_shares']:.5f}, Actual: {actual['quantity']:.5f}")
                else:
                    print(f"   ✅ {symbol}: Matches correctly")
            
            elif actual and not db:
                corrections_needed.append({
                    'symbol': symbol,
                    'issue': 'MISSING_FROM_DB',
                    'actual_data': actual
                })
                print(f"   ⚠️ {symbol}: In Webull but missing from database")
            
            elif db and not actual:
                corrections_needed.append({
                    'symbol': symbol,
                    'issue': 'GHOST_POSITION',
                    'db_data': db
                })
                print(f"   ⚠️ {symbol}: In database but not in Webull (ghost position)")
        
        if not corrections_needed:
            print(f"\n✅ No corrections needed - database matches Webull!")
            return
        
        print(f"\n🔧 STEP 4: APPLY CORRECTIONS")
        print(f"Found {len(corrections_needed)} issues to fix:")
        
        with sqlite3.connect(db_path) as conn:
            for correction in corrections_needed:
                symbol = correction['symbol']
                issue = correction['issue']
                
                print(f"\n   Fixing {symbol} ({issue}):")
                
                if issue == 'WRONG_ACCOUNT':
                    # Update account type in both tables
                    old_account = correction['db_account']
                    new_account = correction['actual_account']
                    new_account_id = correction['actual_account_id']
                    
                    print(f"      Changing account: {old_account} → {new_account}")
                    
                    # Update positions table
                    conn.execute('''
                        UPDATE positions 
                        SET account_type = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ? AND account_type = ? AND bot_id = ?
                    ''', (new_account, symbol, old_account, bot_id))
                    
                    # Update positions_enhanced table
                    conn.execute('''
                        UPDATE positions_enhanced 
                        SET account_type = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ? AND account_type = ? AND bot_id = ?
                    ''', (new_account, symbol, old_account, bot_id))
                    
                    # Update trades table to reflect correct account
                    conn.execute('''
                        UPDATE trades 
                        SET account_type = ?
                        WHERE symbol = ? AND account_type = ? AND bot_id = ?
                    ''', (new_account, symbol, old_account, bot_id))
                    
                    print(f"      ✅ Updated {symbol} to {new_account}")
                
                elif issue == 'QUANTITY_MISMATCH':
                    # Update quantities to match actual
                    db_data = correction['db_data']
                    actual_data = correction['actual_data']
                    
                    new_quantity = actual_data['quantity']
                    new_cost = actual_data['cost_price']
                    new_invested = new_quantity * new_cost
                    
                    print(f"      Updating quantity: {db_data['total_shares']:.5f} → {new_quantity:.5f}")
                    
                    # Update positions table
                    conn.execute('''
                        UPDATE positions 
                        SET total_shares = ?, avg_cost = ?, total_invested = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ? AND bot_id = ?
                    ''', (new_quantity, new_cost, new_invested, symbol, bot_id))
                    
                    # Update positions_enhanced table
                    conn.execute('''
                        UPDATE positions_enhanced 
                        SET total_shares = ?, avg_cost = ?, total_invested = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE symbol = ? AND bot_id = ?
                    ''', (new_quantity, new_cost, new_invested, symbol, bot_id))
                    
                    print(f"      ✅ Updated {symbol} quantities")
                
                elif issue == 'MISSING_FROM_DB':
                    # Add missing position to database
                    actual_data = correction['actual_data']
                    today = datetime.now().strftime('%Y-%m-%d')
                    
                    print(f"      Adding missing position to database")
                    
                    # Add to positions table
                    conn.execute('''
                        INSERT INTO positions (symbol, account_type, total_shares, avg_cost, total_invested,
                                             first_purchase_date, last_purchase_date, entry_phase, 
                                             entry_strength, bot_id, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        symbol, actual_data['account_type'], actual_data['quantity'], 
                        actual_data['cost_price'], actual_data['market_value'],
                        today, today, 'RECONCILED', 0.0, bot_id
                    ))
                    
                    # Add to positions_enhanced table
                    conn.execute('''
                        INSERT INTO positions_enhanced (symbol, account_type, total_shares, avg_cost, total_invested,
                                                      first_purchase_date, last_purchase_date, entry_phase, 
                                                      entry_strength, position_size_pct, time_held_days,
                                                      volatility_percentile, bot_id, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        symbol, actual_data['account_type'], actual_data['quantity'], 
                        actual_data['cost_price'], actual_data['market_value'],
                        today, today, 'RECONCILED', 0.0, 0.1, 0, 0.5, bot_id
                    ))
                    
                    print(f"      ✅ Added {symbol} to database")
                
                elif issue == 'GHOST_POSITION':
                    # Remove ghost position from database
                    print(f"      Removing ghost position from database")
                    
                    # Remove from both tables
                    conn.execute('''
                        DELETE FROM positions 
                        WHERE symbol = ? AND bot_id = ?
                    ''', (symbol, bot_id))
                    
                    conn.execute('''
                        DELETE FROM positions_enhanced 
                        WHERE symbol = ? AND bot_id = ?
                    ''', (symbol, bot_id))
                    
                    print(f"      ✅ Removed ghost position {symbol}")
        
        print(f"\n📊 STEP 5: VERIFICATION")
        
        # Verify corrections
        with sqlite3.connect(db_path) as conn:
            updated_positions = conn.execute('''
                SELECT symbol, account_type, total_shares, avg_cost
                FROM positions 
                WHERE bot_id = ? AND total_shares > 0
                ORDER BY account_type, symbol
            ''', (bot_id,)).fetchall()
            
            print(f"\n   Updated database positions:")
            for symbol, account_type, shares, avg_cost in updated_positions:
                print(f"      {symbol}: {shares:.5f} shares @ ${avg_cost:.2f} in {account_type}")
        
        print(f"\n✅ DATABASE RECONCILIATION COMPLETE!")
        print(f"Database now matches actual Webull account positions.")
        
    except Exception as e:
        print(f"❌ Error during reconciliation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        main_system.cleanup()

if __name__ == "__main__":
    reconcile_database_with_reality()