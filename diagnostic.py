#!/usr/bin/env python3
"""
Specific Database Reader for Your Trading Bot Database
"""

import sqlite3
import pandas as pd
from datetime import datetime

def analyze_your_database():
    """Analyze your specific trading bot database"""
    db_path = "data/trading_bot.db"
    
    print("🔍 ANALYZING YOUR TRADING BOT DATABASE")
    print("=" * 60)
    
    with sqlite3.connect(db_path) as conn:
        # Check positions first
        print("\n📊 YOUR CURRENT POSITIONS:")
        print("-" * 30)
        
        positions_query = """
            SELECT symbol, total_shares, avg_cost, total_invested, account_type, 
                   first_purchase_date, last_purchase_date
            FROM positions 
            WHERE total_shares > 0 AND bot_id = 'wyckoff_bot_v1'
            ORDER BY symbol
        """
        
        positions = pd.read_sql_query(positions_query, conn)
        
        if not positions.empty:
            for _, pos in positions.iterrows():
                print(f"  📈 {pos['symbol']}: {pos['total_shares']:.5f} shares @ ${pos['avg_cost']:.2f}")
                print(f"     💰 Invested: ${pos['total_invested']:.2f} ({pos['account_type']})")
                print(f"     📅 Purchased: {pos['first_purchase_date']}")
                print()
        else:
            print("  ❌ No positions found")
        
        # Now check stop strategies
        print("🛡️ YOUR STOP LOSS STRATEGIES:")
        print("-" * 35)
        
        stop_query = """
            SELECT symbol, strategy_type, initial_price, stop_price, 
                   stop_percentage, trailing_high, is_active, created_at
            FROM stop_strategies 
            WHERE bot_id = 'wyckoff_bot_v1'
            ORDER BY symbol, strategy_type
        """
        
        stops = pd.read_sql_query(stop_query, conn)
        
        if not stops.empty:
            current_symbol = None
            for _, stop in stops.iterrows():
                if current_symbol != stop['symbol']:
                    current_symbol = stop['symbol']
                    print(f"\n  🏷️  {stop['symbol']} (Initial: ${stop['initial_price']:.2f}):")
                
                status = "✅ ACTIVE" if stop['is_active'] else "❌ INACTIVE"
                strategy_type = stop['strategy_type']
                stop_price = stop['stop_price']
                stop_percentage = stop['stop_percentage']
                
                if strategy_type == 'STOP_LOSS':
                    print(f"     🛑 Fixed Stop Loss:")
                    print(f"        Stop Price: ${stop_price:.2f}")
                    print(f"        Stop %: {stop_percentage*100:.1f}%")
                    print(f"        Status: {status}")
                    
                elif strategy_type == 'TRAILING_STOP':
                    trailing_high = stop['trailing_high']
                    print(f"     📉 Trailing Stop:")
                    print(f"        Stop Price: ${stop_price:.2f}")
                    print(f"        Stop %: {stop_percentage*100:.1f}%")
                    print(f"        Trailing High: ${trailing_high:.2f}")
                    print(f"        Status: {status}")
                
                print(f"        Created: {stop['created_at']}")
                print()
        else:
            print("  ❌ No stop strategies found")
        
        # Check recent trades
        print("📋 YOUR RECENT TRADES (Last 10):")
        print("-" * 35)
        
        trades_query = """
            SELECT date, symbol, action, quantity, price, total_value, 
                   signal_phase, account_type, trade_datetime
            FROM trades 
            WHERE bot_id = 'wyckoff_bot_v1'
            ORDER BY trade_datetime DESC
            LIMIT 10
        """
        
        trades = pd.read_sql_query(trades_query, conn)
        
        if not trades.empty:
            for _, trade in trades.iterrows():
                action_emoji = "💰" if trade['action'] == 'BUY' else "💸"
                print(f"  {action_emoji} {trade['date']}: {trade['action']} {trade['quantity']:.5f} {trade['symbol']}")
                print(f"     Price: ${trade['price']:.2f} | Total: ${trade['total_value']:.2f}")
                print(f"     Phase: {trade['signal_phase']} | Account: {trade['account_type']}")
                print()
        else:
            print("  ❌ No trades found")
        
        # Validate stop strategy calculations
        print("🔍 STOP STRATEGY VALIDATION:")
        print("-" * 30)
        
        validation_query = """
            SELECT 
                s.symbol,
                s.strategy_type,
                s.initial_price,
                s.stop_price,
                s.stop_percentage,
                s.trailing_high,
                p.avg_cost
            FROM stop_strategies s
            LEFT JOIN positions p ON s.symbol = p.symbol AND s.bot_id = p.bot_id
            WHERE s.is_active = TRUE AND s.bot_id = 'wyckoff_bot_v1'
            ORDER BY s.symbol, s.strategy_type
        """
        
        validations = pd.read_sql_query(validation_query, conn)
        
        if not validations.empty:
            issues_found = []
            
            for _, val in validations.iterrows():
                symbol = val['symbol']
                strategy_type = val['strategy_type']
                initial_price = val['initial_price']
                stop_price = val['stop_price']
                stop_percentage = val['stop_percentage']
                trailing_high = val['trailing_high']
                avg_cost = val['avg_cost']
                
                print(f"\n  📊 {symbol} - {strategy_type}:")
                print(f"     Initial Price: ${initial_price:.2f}")
                print(f"     Avg Cost: ${avg_cost:.2f}")
                print(f"     Stop Price: ${stop_price:.2f}")
                print(f"     Stop %: {stop_percentage*100:.1f}%")
                
                # Validate calculations
                expected_stop = initial_price * (1 - stop_percentage)
                
                if abs(stop_price - expected_stop) > 0.01:
                    issue = f"{symbol} {strategy_type}: Stop price ${stop_price:.2f} != expected ${expected_stop:.2f}"
                    issues_found.append(issue)
                    print(f"     ⚠️  ISSUE: Expected ${expected_stop:.2f}")
                else:
                    print(f"     ✅ Calculation correct")
                
                if strategy_type == 'TRAILING_STOP':
                    print(f"     Trailing High: ${trailing_high:.2f}")
                    
                    if trailing_high == initial_price:
                        print(f"     ✅ Trailing high correctly set to initial price")
                    else:
                        print(f"     ℹ️  Trailing high updated from initial ${initial_price:.2f}")
            
            if issues_found:
                print(f"\n⚠️  ISSUES FOUND:")
                for issue in issues_found:
                    print(f"  - {issue}")
            else:
                print(f"\n✅ All stop strategies are correctly calculated!")
        
        # Summary statistics
        print("\n📈 DATABASE SUMMARY:")
        print("-" * 20)
        
        total_positions = conn.execute("SELECT COUNT(*) FROM positions WHERE total_shares > 0 AND bot_id = 'wyckoff_bot_v1'").fetchone()[0]
        total_invested = conn.execute("SELECT SUM(total_invested) FROM positions WHERE total_shares > 0 AND bot_id = 'wyckoff_bot_v1'").fetchone()[0] or 0
        active_stops = conn.execute("SELECT COUNT(*) FROM stop_strategies WHERE is_active = TRUE AND bot_id = 'wyckoff_bot_v1'").fetchone()[0]
        total_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE bot_id = 'wyckoff_bot_v1'").fetchone()[0]
        
        print(f"  📊 Active Positions: {total_positions}")
        print(f"  💰 Total Invested: ${total_invested:.2f}")
        print(f"  🛡️  Active Stop Strategies: {active_stops}")
        print(f"  📋 Total Trades: {total_trades}")
        
        # Check for potential issues
        print(f"\n🔍 POTENTIAL ISSUES CHECK:")
        print("-" * 25)
        
        # Check for positions without stop strategies
        positions_without_stops = conn.execute("""
            SELECT p.symbol FROM positions p
            LEFT JOIN stop_strategies s ON p.symbol = s.symbol AND p.bot_id = s.bot_id AND s.is_active = TRUE
            WHERE p.total_shares > 0 AND p.bot_id = 'wyckoff_bot_v1' AND s.symbol IS NULL
        """).fetchall()
        
        if positions_without_stops:
            print(f"  ⚠️  Positions without stop strategies:")
            for (symbol,) in positions_without_stops:
                print(f"     - {symbol}")
        
        # Check for null values in stop strategies
        null_stops = conn.execute("""
            SELECT symbol, strategy_type FROM stop_strategies 
            WHERE (stop_price IS NULL OR stop_percentage IS NULL) 
            AND is_active = TRUE AND bot_id = 'wyckoff_bot_v1'
        """).fetchall()
        
        if null_stops:
            print(f"  ⚠️  Stop strategies with NULL values:")
            for symbol, strategy_type in null_stops:
                print(f"     - {symbol} {strategy_type}")
        
        if not positions_without_stops and not null_stops:
            print(f"  ✅ No issues detected!")
        
        print(f"\n" + "=" * 60)
        print(f"📅 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        analyze_your_database()
    except Exception as e:
        print(f"❌ Error analyzing database: {e}")
        import traceback
        traceback.print_exc()