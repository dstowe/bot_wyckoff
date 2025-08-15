#!/usr/bin/env python3
"""
Check what the database currently thinks the positions are
Save this as: check_database_positions.py
"""

import sqlite3
from pathlib import Path
from datetime import datetime

def check_database_positions():
    """Check current database position records"""
    
    print("📋 CHECKING DATABASE POSITION RECORDS")
    print("="*50)
    
    db_path = Path("data/trading_bot.db")
    if not db_path.exists():
        print("❌ Database file not found!")
        return
    
    bot_id = "enhanced_wyckoff_bot_v2"
    
    try:
        with sqlite3.connect(db_path) as conn:
            print(f"\n📊 POSITIONS TABLE:")
            positions = conn.execute('''
                SELECT symbol, account_type, total_shares, avg_cost, total_invested, 
                       first_purchase_date, last_purchase_date, entry_phase, updated_at
                FROM positions 
                WHERE bot_id = ?
                ORDER BY account_type, symbol
            ''', (bot_id,)).fetchall()
            
            if positions:
                current_account = None
                for row in positions:
                    symbol, account_type, shares, avg_cost, invested, first_date, last_date, phase, updated = row
                    
                    if current_account != account_type:
                        current_account = account_type
                        print(f"\n   {account_type}:")
                    
                    status = "ACTIVE" if shares > 0 else "CLOSED"
                    print(f"      {symbol}: {shares:.5f} shares @ ${avg_cost:.2f} [{status}]")
                    print(f"         Invested: ${invested:.2f}, Phase: {phase}")
                    print(f"         First: {first_date}, Last: {last_date}")
                    print(f"         Updated: {updated}")
            else:
                print("   No positions found in database")
            
            print(f"\n📈 POSITIONS_ENHANCED TABLE:")
            enhanced_positions = conn.execute('''
                SELECT symbol, account_type, total_shares, avg_cost, total_invested, 
                       entry_phase, time_held_days, updated_at
                FROM positions_enhanced 
                WHERE bot_id = ?
                ORDER BY account_type, symbol
            ''', (bot_id,)).fetchall()
            
            if enhanced_positions:
                current_account = None
                for row in enhanced_positions:
                    symbol, account_type, shares, avg_cost, invested, phase, days, updated = row
                    
                    if current_account != account_type:
                        current_account = account_type
                        print(f"\n   {account_type}:")
                    
                    status = "ACTIVE" if shares > 0 else "CLOSED"
                    print(f"      {symbol}: {shares:.5f} shares @ ${avg_cost:.2f} [{status}]")
                    print(f"         Phase: {phase}, Days held: {days}")
                    print(f"         Updated: {updated}")
            else:
                print("   No enhanced positions found")
            
            print(f"\n📋 TODAY'S TRADES:")
            today = datetime.now().strftime('%Y-%m-%d')
            trades = conn.execute('''
                SELECT symbol, action, quantity, price, account_type, order_id, 
                       day_trade_check, trade_datetime
                FROM trades 
                WHERE bot_id = ? AND date = ?
                ORDER BY trade_datetime
            ''', (bot_id, today)).fetchall()
            
            if trades:
                print(f"   Trades for {today}:")
                for row in trades:
                    symbol, action, quantity, price, account_type, order_id, day_check, trade_time = row
                    print(f"      {symbol} {action}: {quantity:.5f} @ ${price:.2f}")
                    print(f"         Account: {account_type}, OrderID: {order_id}")
                    print(f"         Day Trade: {day_check}, Time: {trade_time}")
            else:
                print(f"   No trades found for {today}")
            
            print(f"\n🚨 DAY TRADE CHECKS:")
            day_checks = conn.execute('''
                SELECT symbol, action, db_day_trade, actual_day_trade, 
                       manual_trades_detected, recommendation, created_at
                FROM day_trade_checks 
                WHERE bot_id = ? AND check_date = ?
                ORDER BY created_at DESC
                LIMIT 10
            ''', (bot_id, today)).fetchall()
            
            if day_checks:
                print(f"   Recent day trade checks:")
                for row in day_checks:
                    symbol, action, db_dt, actual_dt, manual, recommendation, created = row
                    print(f"      {symbol} {action}: {recommendation}")
                    print(f"         DB Day Trade: {bool(db_dt)}, Actual: {bool(actual_dt)}")
                    print(f"         Manual Detected: {bool(manual)}, Time: {created}")
            else:
                print(f"   No day trade checks found")
    
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_positions()