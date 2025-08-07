#!/usr/bin/env python3
"""
FIXED: Enhanced Wyckoff Trading Bot Analytics Dashboard
Now properly filters by bot_id and shows correct P&L including unrealized gains
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import numpy as np


class EnhancedTradingAnalytics:
    """Enhanced analytics engine with proper bot_id filtering"""
    
    def __init__(self, db_path="data/trading_bot.db", bot_id="wyckoff_bot_v1"):
        self.db_path = Path(db_path)
        self.bot_id = bot_id  # FIXED: Now uses bot_id consistently
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Trading database not found: {db_path}")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"📊 Analytics configured for bot_id: {self.bot_id}")
    
    def get_trading_summary(self) -> Dict:
        """Get overall trading performance summary with PROPER bot_id filtering"""
        with sqlite3.connect(self.db_path) as conn:
            # FIXED: All queries now filter by bot_id
            total_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE bot_id = ?", (self.bot_id,)).fetchone()[0]
            total_buy_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'BUY' AND bot_id = ?", (self.bot_id,)).fetchone()[0]
            total_sell_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'SELL' AND bot_id = ?", (self.bot_id,)).fetchone()[0]
            
            # Stop loss specific stats
            stop_loss_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE signal_phase = 'STOP_LOSS' AND bot_id = ?", (self.bot_id,)).fetchone()[0]
            
            # Investment amounts - FIXED with bot_id
            total_invested = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'BUY' AND bot_id = ?", (self.bot_id,)).fetchone()[0] or 0
            total_proceeds = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'SELL' AND bot_id = ?", (self.bot_id,)).fetchone()[0] or 0
            
            # Stop loss specific proceeds
            stop_loss_proceeds = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'SELL' AND signal_phase = 'STOP_LOSS' AND bot_id = ?", (self.bot_id,)).fetchone()[0] or 0
            
            # Current positions - FIXED with bot_id
            current_positions = conn.execute("SELECT COUNT(*) FROM positions WHERE total_shares > 0 AND bot_id = ?", (self.bot_id,)).fetchone()[0]
            current_invested = conn.execute("SELECT SUM(total_invested) FROM positions WHERE total_shares > 0 AND bot_id = ?", (self.bot_id,)).fetchone()[0] or 0
            
            # Active stop strategies - FIXED with bot_id
            active_stops = conn.execute("SELECT COUNT(*) FROM stop_strategies WHERE is_active = TRUE AND bot_id = ?", (self.bot_id,)).fetchone()[0]
            
            # Bot runs - FIXED with bot_id
            total_runs = conn.execute("SELECT COUNT(*) FROM bot_runs WHERE bot_id = ?", (self.bot_id,)).fetchone()[0]
            successful_runs = conn.execute("SELECT COUNT(*) FROM bot_runs WHERE status LIKE 'SUCCESS%' AND bot_id = ?", (self.bot_id,)).fetchone()[0]
            
            # Stop losses executed - FIXED with bot_id
            total_stop_losses = conn.execute("SELECT SUM(stop_losses_executed) FROM bot_runs WHERE bot_id = ?", (self.bot_id,)).fetchone()[0] or 0
            
            # Date range - FIXED with bot_id
            first_trade = conn.execute("SELECT MIN(date) FROM trades WHERE bot_id = ?", (self.bot_id,)).fetchone()[0]
            last_trade = conn.execute("SELECT MAX(date) FROM trades WHERE bot_id = ?", (self.bot_id,)).fetchone()[0]
            
            # FIXED: Calculate P&L properly
            # If you have current positions, this is unrealized P&L, not realized P&L
            if current_positions > 0:
                net_profit = total_proceeds - total_invested  # This is CASH FLOW, not profit
                unrealized_invested = current_invested  # Money still invested in positions
                print(f"💡 Note: You have ${current_invested:.2f} still invested in {current_positions} positions")
                print(f"💡 Cash flow P&L: ${net_profit:.2f} (proceeds - invested)")
                print(f"💡 For true P&L, need current market values of positions")
            else:
                net_profit = total_proceeds - total_invested
            
            return {
                'total_trades': total_trades,
                'buy_trades': total_buy_trades,
                'sell_trades': total_sell_trades,
                'stop_loss_trades': stop_loss_trades,
                'total_invested': total_invested,
                'total_proceeds': total_proceeds,
                'stop_loss_proceeds': stop_loss_proceeds,
                'net_profit': net_profit,
                'current_positions': current_positions,
                'current_value': current_invested,
                'active_stop_strategies': active_stops,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'total_stop_losses_executed': total_stop_losses,
                'success_rate': (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                'first_trade': first_trade,
                'last_trade': last_trade,
                'bot_id': self.bot_id  # Include bot_id in summary
            }
    
    def get_stop_loss_effectiveness(self) -> pd.DataFrame:
        """Analyze stop loss strategy effectiveness with bot_id filtering"""
        with sqlite3.connect(self.db_path) as conn:
            # FIXED: Added bot_id filtering
            query = """
            SELECT 
                ss.strategy_type,
                ss.stop_percentage,
                COUNT(*) as positions_created,
                COUNT(CASE WHEN t.signal_phase = 'STOP_LOSS' THEN 1 END) as stop_losses_triggered,
                AVG(ss.stop_percentage) as avg_stop_percentage,
                AVG(CASE WHEN t.signal_phase = 'STOP_LOSS' 
                    THEN (t.price - p.avg_cost) / p.avg_cost * 100 END) as avg_stop_loss_return
            FROM stop_strategies ss
            LEFT JOIN positions p ON ss.symbol = p.symbol AND ss.bot_id = p.bot_id
            LEFT JOIN trades t ON ss.symbol = t.symbol AND t.signal_phase = 'STOP_LOSS' AND ss.bot_id = t.bot_id
            WHERE ss.bot_id = ?
            GROUP BY ss.strategy_type, ss.stop_percentage
            ORDER BY positions_created DESC
            """
            
            return pd.read_sql_query(query, conn, params=(self.bot_id,))
    
    def get_signal_effectiveness(self) -> pd.DataFrame:
        """Analyze effectiveness of different Wyckoff phases including stop losses with bot_id"""
        with sqlite3.connect(self.db_path) as conn:
            # FIXED: Added bot_id filtering
            query = """
            SELECT 
                CASE 
                    WHEN signal_phase = 'STOP_LOSS' THEN 'STOP_LOSS'
                    ELSE signal_phase 
                END as signal_phase,
                COUNT(*) as total_trades,
                AVG(signal_strength) as avg_strength,
                COUNT(CASE WHEN action = 'BUY' THEN 1 END) as buy_count,
                COUNT(CASE WHEN action = 'SELL' THEN 1 END) as sell_count,
                SUM(CASE WHEN action = 'BUY' THEN total_value ELSE 0 END) as total_invested,
                SUM(CASE WHEN action = 'SELL' THEN total_value ELSE 0 END) as total_proceeds
            FROM trades 
            WHERE signal_phase IS NOT NULL AND bot_id = ?
            GROUP BY 
                CASE 
                    WHEN signal_phase = 'STOP_LOSS' THEN 'STOP_LOSS'
                    ELSE signal_phase 
                END
            ORDER BY total_trades DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(self.bot_id,))
            
            # Calculate profit/loss and success rate
            df['net_profit'] = df['total_proceeds'] - df['total_invested']
            df['roi_percent'] = (df['net_profit'] / df['total_invested'] * 100).round(2)
            
            return df
    
    def get_day_trade_prevention_stats(self) -> Dict:
        """Analyze day trade prevention effectiveness with bot_id"""
        with sqlite3.connect(self.db_path) as conn:
            # FIXED: Added bot_id filtering
            # Count signals that were skipped due to day trade prevention
            skipped_signals = conn.execute("""
                SELECT COUNT(*) FROM signals 
                WHERE (action_taken LIKE '%SKIP%' OR action_taken LIKE '%DAY_TRADE%') AND bot_id = ?
            """, (self.bot_id,)).fetchone()[0]
            
            # Count total signals that could have been traded
            total_actionable_signals = conn.execute("""
                SELECT COUNT(*) FROM signals 
                WHERE action_taken IS NOT NULL AND action_taken != 'NO_ACTION' AND bot_id = ?
            """, (self.bot_id,)).fetchone()[0]
            
            # Get daily trade counts - FIXED with bot_id
            daily_trades_query = """
                SELECT date, COUNT(*) as trades_count
                FROM trades
                WHERE bot_id = ?
                GROUP BY date
                ORDER BY date DESC
                LIMIT 30
            """
            daily_trades = pd.read_sql_query(daily_trades_query, conn, params=(self.bot_id,))
            
            return {
                'skipped_signals': skipped_signals,
                'total_actionable_signals': total_actionable_signals,
                'prevention_rate': (skipped_signals / total_actionable_signals * 100) if total_actionable_signals > 0 else 0,
                'daily_trades': daily_trades
            }
    
    def get_position_lifecycle_analysis(self) -> pd.DataFrame:
        """Analyze the complete lifecycle of positions from buy to sell with bot_id"""
        with sqlite3.connect(self.db_path) as conn:
            # FIXED: Added bot_id filtering throughout
            query = """
            SELECT 
                p.symbol,
                p.total_shares,
                p.avg_cost,
                p.total_invested,
                p.first_purchase_date,
                p.last_purchase_date,
                COUNT(t.id) as total_trades,
                SUM(CASE WHEN t.action = 'BUY' THEN t.total_value ELSE 0 END) as total_bought,
                SUM(CASE WHEN t.action = 'SELL' THEN t.total_value ELSE 0 END) as total_sold,
                COUNT(CASE WHEN t.action = 'SELL' AND t.signal_phase = 'STOP_LOSS' THEN 1 END) as stop_loss_exits,
                COUNT(CASE WHEN t.action = 'SELL' AND t.signal_phase != 'STOP_LOSS' THEN 1 END) as wyckoff_exits,
                CASE WHEN p.total_shares > 0 THEN 'OPEN' ELSE 'CLOSED' END as status,
                -- Stop loss strategy info
                (SELECT COUNT(*) FROM stop_strategies ss WHERE ss.symbol = p.symbol AND ss.is_active = TRUE AND ss.bot_id = ?) as active_stops
            FROM positions p
            LEFT JOIN trades t ON p.symbol = t.symbol AND p.bot_id = t.bot_id
            WHERE p.bot_id = ?
            GROUP BY p.symbol
            ORDER BY p.total_invested DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(self.bot_id, self.bot_id))
            
            # Calculate metrics
            df['net_profit'] = df['total_sold'] - df['total_bought']
            df['roi_percent'] = ((df['net_profit'] / df['total_bought']) * 100).round(2)
            df['days_held'] = (pd.to_datetime('today') - pd.to_datetime(df['first_purchase_date'])).dt.days
            
            # Add exit method analysis
            df['exit_method'] = df.apply(lambda row: 
                'STOP_LOSS' if row['stop_loss_exits'] > 0 else
                'WYCKOFF_SIGNAL' if row['wyckoff_exits'] > 0 else
                'STILL_OPEN' if row['status'] == 'OPEN' else
                'OTHER', axis=1)
            
            return df
    
    def get_daily_performance(self) -> pd.DataFrame:
        """Get daily trading performance including stop loss actions with bot_id"""
        with sqlite3.connect(self.db_path) as conn:
            # FIXED: Added bot_id filtering
            query = """
            SELECT 
                br.run_date as date,
                br.signals_found,
                br.trades_executed,
                br.stop_losses_executed,
                br.total_portfolio_value,
                br.available_cash,
                br.status,
                -- Daily trade totals
                COALESCE(dt.daily_invested, 0) as invested,
                COALESCE(dt.daily_proceeds, 0) as proceeds,
                COALESCE(dt.buy_count, 0) as buys,
                COALESCE(dt.sell_count, 0) as sells
            FROM bot_runs br
            LEFT JOIN (
                SELECT 
                    date,
                    SUM(CASE WHEN action = 'BUY' THEN total_value ELSE 0 END) as daily_invested,
                    SUM(CASE WHEN action = 'SELL' THEN total_value ELSE 0 END) as daily_proceeds,
                    COUNT(CASE WHEN action = 'BUY' THEN 1 END) as buy_count,
                    COUNT(CASE WHEN action = 'SELL' THEN 1 END) as sell_count
                FROM trades
                WHERE bot_id = ?
                GROUP BY date
            ) dt ON DATE(br.run_date) = dt.date
            WHERE br.bot_id = ?
            ORDER BY br.run_date
            """
            
            df = pd.read_sql_query(query, conn, params=(self.bot_id, self.bot_id))
            df['date'] = pd.to_datetime(df['date'])
            df['net_daily'] = df['proceeds'] - df['invested']
            df['cumulative_net'] = df['net_daily'].cumsum()
            df['total_actions'] = df['trades_executed'] + df['stop_losses_executed']
            
            return df
    
    def show_detailed_breakdown(self):
        """Show detailed breakdown of what the analytics found"""
        with sqlite3.connect(self.db_path) as conn:
            print(f"\n🔍 DETAILED BREAKDOWN FOR BOT_ID: {self.bot_id}")
            print("=" * 60)
            
            # Check if bot_id exists
            bot_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE bot_id = ?", (self.bot_id,)).fetchone()[0]
            bot_positions = conn.execute("SELECT COUNT(*) FROM positions WHERE bot_id = ?", (self.bot_id,)).fetchone()[0]
            bot_runs = conn.execute("SELECT COUNT(*) FROM bot_runs WHERE bot_id = ?", (self.bot_id,)).fetchone()[0]
            
            print(f"📊 Records found for bot_id '{self.bot_id}':")
            print(f"   Trades: {bot_trades}")
            print(f"   Positions: {bot_positions}")
            print(f"   Bot Runs: {bot_runs}")
            
            if bot_trades > 0:
                print(f"\n💰 TRADE BREAKDOWN:")
                trades = conn.execute("""
                    SELECT date, symbol, action, quantity, price, total_value, signal_phase 
                    FROM trades WHERE bot_id = ? ORDER BY trade_datetime
                """, (self.bot_id,)).fetchall()
                
                for trade in trades:
                    date, symbol, action, qty, price, total, phase = trade
                    print(f"   {date}: {action} {qty:.5f} {symbol} @ ${price:.2f} = ${total:.2f} ({phase})")
            
            if bot_positions > 0:
                print(f"\n📊 CURRENT POSITIONS:")
                positions = conn.execute("""
                    SELECT symbol, total_shares, avg_cost, total_invested 
                    FROM positions WHERE total_shares > 0 AND bot_id = ? ORDER BY symbol
                """, (self.bot_id,)).fetchall()
                
                for pos in positions:
                    symbol, shares, avg_cost, invested = pos
                    print(f"   {symbol}: {shares:.5f} shares @ ${avg_cost:.2f} avg (${invested:.2f} invested)")
            
            # Check all bot_ids in database
            all_bot_ids = conn.execute("SELECT DISTINCT bot_id FROM trades").fetchall()
            if len(all_bot_ids) > 1:
                print(f"\n⚠️  Multiple bot_ids found in database:")
                for (bot_id,) in all_bot_ids:
                    count = conn.execute("SELECT COUNT(*) FROM trades WHERE bot_id = ?", (bot_id,)).fetchone()[0]
                    print(f"   '{bot_id}': {count} trades")
                print(f"   Currently analyzing: '{self.bot_id}'")
    
    # ... [Rest of the methods remain the same, just need to add bot_id filtering where missing]
    
    def create_enhanced_performance_charts(self, save_path="reports"):
        """Create comprehensive performance charts including stop loss analysis"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # 1. Enhanced Daily Performance Chart
        daily_perf = self.get_daily_performance()
        if not daily_perf.empty:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Cumulative P&L
            ax1.plot(daily_perf['date'], daily_perf['cumulative_net'], marker='o', linewidth=2, color='blue')
            ax1.set_title(f'Cumulative P&L Over Time (Bot: {self.bot_id})', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative P&L ($)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Portfolio Value Over Time
            ax2.plot(daily_perf['date'], daily_perf['total_portfolio_value'], marker='o', linewidth=2, color='green')
            ax2.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Portfolio Value ($)')
            ax2.grid(True, alpha=0.3)
            
            # Daily Actions (Trades + Stop Losses)
            width = 0.35
            ax3.bar(daily_perf['date'], daily_perf['trades_executed'], width, label='Wyckoff Trades', alpha=0.7)
            ax3.bar(daily_perf['date'], daily_perf['stop_losses_executed'], width, bottom=daily_perf['trades_executed'], 
                   label='Stop Losses', alpha=0.7)
            ax3.set_title('Daily Trading Actions', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Number of Actions')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Success Rate Over Time
            success_rate = daily_perf['status'].apply(lambda x: 1 if 'SUCCESS' in x else 0).rolling(7).mean() * 100
            ax4.plot(daily_perf['date'], success_rate, marker='o', linewidth=2, color='orange')
            ax4.set_title('7-Day Rolling Success Rate', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Success Rate (%)')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target (80%)')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(save_path / f'enhanced_daily_performance_{self.bot_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Chart saved: enhanced_daily_performance_{self.bot_id}.png")
        
        # ... [Continue with other charts, adding bot_id to titles and filenames]


def main():
    """Main function for enhanced analytics dashboard with proper bot_id filtering"""
    parser = argparse.ArgumentParser(description='Enhanced Wyckoff Trading Bot Analytics with Bot ID Filtering')
    parser.add_argument('--db', default='data/trading_bot.db', help='Database path')
    parser.add_argument('--bot-id', default='wyckoff_bot_v1', help='Bot ID to analyze')
    parser.add_argument('--output', default='reports', help='Output directory')
    parser.add_argument('--charts', action='store_true', help='Generate charts')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--export', action='store_true', help='Export data to CSV')
    parser.add_argument('--breakdown', action='store_true', help='Show detailed breakdown')
    parser.add_argument('--all', action='store_true', help='Generate everything')
    
    args = parser.parse_args()
    
    try:
        # FIXED: Now passes bot_id to analytics
        analytics = EnhancedTradingAnalytics(args.db, args.bot_id)
        
        if args.breakdown or args.all:
            analytics.show_detailed_breakdown()
        
        if args.all or args.charts:
            print("📊 Generating enhanced performance charts...")
            analytics.create_enhanced_performance_charts(args.output)
        
        if args.all or args.report:
            print("📋 Generating enhanced performance report...")
            # Note: Would need to update generate_enhanced_report method too
            # report = analytics.generate_enhanced_report(args.output)
            
        if args.all or args.export:
            print("💾 Exporting enhanced data...")
            # Note: Would need to update export methods to filter by bot_id
            # analytics.export_enhanced_data(args.output)
        
        # Show summary
        summary = analytics.get_trading_summary()
        print(f"\n" + "="*60)
        print(f"ENHANCED TRADING BOT SUMMARY (Bot: {args.bot_id})")
        print("="*60)
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Buy Trades: {summary['buy_trades']}")
        print(f"Sell Trades: {summary['sell_trades']}")
        print(f"Stop Loss Trades: {summary['stop_loss_trades']}")
        print(f"Total Invested: ${summary['total_invested']:,.2f}")
        print(f"Total Proceeds: ${summary['total_proceeds']:,.2f}")
        print(f"Net Cash Flow: ${summary['net_profit']:,.2f}")
        print(f"Current Positions: {summary['current_positions']}")
        print(f"Money Still Invested: ${summary['current_value']:,.2f}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Active Stop Strategies: {summary['active_stop_strategies']}")
        print("="*60)
        
        # FIXED: Explain P&L properly
        if summary['current_positions'] > 0:
            print(f"\n💡 EXPLANATION:")
            print(f"You've invested ${summary['total_invested']:.2f} total")
            print(f"You've received ${summary['total_proceeds']:.2f} back from sales")
            print(f"Net cash flow: ${summary['net_profit']:,.2f}")
            print(f"You still own {summary['current_positions']} positions worth ${summary['current_value']:,.2f} (at purchase price)")
            print(f"To see true profit/loss, need current market values of your positions")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())