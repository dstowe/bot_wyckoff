#!/usr/bin/env python3
"""
Enhanced Wyckoff Trading Bot Analytics Dashboard
Includes stop loss performance tracking and day trade prevention analysis
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
    """Enhanced analytics engine with stop loss tracking"""
    
    def __init__(self, db_path="data/trading_bot.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Trading database not found: {db_path}")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def get_trading_summary(self) -> Dict:
        """Get overall trading performance summary with stop loss metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            total_buy_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'BUY'").fetchone()[0]
            total_sell_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'SELL'").fetchone()[0]
            
            # Stop loss specific stats
            stop_loss_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE signal_phase = 'STOP_LOSS'").fetchone()[0]
            
            # Investment amounts
            total_invested = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'BUY'").fetchone()[0] or 0
            total_proceeds = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'SELL'").fetchone()[0] or 0
            
            # Stop loss specific proceeds
            stop_loss_proceeds = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'SELL' AND signal_phase = 'STOP_LOSS'").fetchone()[0] or 0
            
            # Current positions
            current_positions = conn.execute("SELECT COUNT(*) FROM positions WHERE total_shares > 0").fetchone()[0]
            current_value = conn.execute("SELECT SUM(total_invested) FROM positions WHERE total_shares > 0").fetchone()[0] or 0
            
            # Active stop strategies
            active_stops = conn.execute("SELECT COUNT(*) FROM stop_strategies WHERE is_active = TRUE").fetchone()[0]
            
            # Bot runs
            total_runs = conn.execute("SELECT COUNT(*) FROM bot_runs").fetchone()[0]
            successful_runs = conn.execute("SELECT COUNT(*) FROM bot_runs WHERE status LIKE 'SUCCESS%'").fetchone()[0]
            
            # Stop losses executed
            total_stop_losses = conn.execute("SELECT SUM(stop_losses_executed) FROM bot_runs").fetchone()[0] or 0
            
            # Date range
            first_trade = conn.execute("SELECT MIN(date) FROM trades").fetchone()[0]
            last_trade = conn.execute("SELECT MAX(date) FROM trades").fetchone()[0]
            
            return {
                'total_trades': total_trades,
                'buy_trades': total_buy_trades,
                'sell_trades': total_sell_trades,
                'stop_loss_trades': stop_loss_trades,
                'total_invested': total_invested,
                'total_proceeds': total_proceeds,
                'stop_loss_proceeds': stop_loss_proceeds,
                'net_profit': total_proceeds - total_invested,
                'current_positions': current_positions,
                'current_value': current_value,
                'active_stop_strategies': active_stops,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'total_stop_losses_executed': total_stop_losses,
                'success_rate': (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                'first_trade': first_trade,
                'last_trade': last_trade
            }
    
    def get_stop_loss_effectiveness(self) -> pd.DataFrame:
        """Analyze stop loss strategy effectiveness"""
        with sqlite3.connect(self.db_path) as conn:
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
            LEFT JOIN positions p ON ss.symbol = p.symbol
            LEFT JOIN trades t ON ss.symbol = t.symbol AND t.signal_phase = 'STOP_LOSS'
            GROUP BY ss.strategy_type, ss.stop_percentage
            ORDER BY positions_created DESC
            """
            
            return pd.read_sql_query(query, conn)
    
    def get_signal_effectiveness(self) -> pd.DataFrame:
        """Analyze effectiveness of different Wyckoff phases including stop losses"""
        with sqlite3.connect(self.db_path) as conn:
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
            WHERE signal_phase IS NOT NULL
            GROUP BY 
                CASE 
                    WHEN signal_phase = 'STOP_LOSS' THEN 'STOP_LOSS'
                    ELSE signal_phase 
                END
            ORDER BY total_trades DESC
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Calculate profit/loss and success rate
            df['net_profit'] = df['total_proceeds'] - df['total_invested']
            df['roi_percent'] = (df['net_profit'] / df['total_invested'] * 100).round(2)
            
            return df
    
    def get_day_trade_prevention_stats(self) -> Dict:
        """Analyze day trade prevention effectiveness"""
        with sqlite3.connect(self.db_path) as conn:
            # Count signals that were skipped due to day trade prevention
            skipped_signals = conn.execute("""
                SELECT COUNT(*) FROM signals 
                WHERE action_taken LIKE '%SKIP%' OR action_taken LIKE '%DAY_TRADE%'
            """).fetchone()[0]
            
            # Count total signals that could have been traded
            total_actionable_signals = conn.execute("""
                SELECT COUNT(*) FROM signals 
                WHERE action_taken IS NOT NULL AND action_taken != 'NO_ACTION'
            """).fetchone()[0]
            
            # Get daily trade counts
            daily_trades_query = """
                SELECT date, COUNT(*) as trades_count
                FROM trades
                GROUP BY date
                ORDER BY date DESC
                LIMIT 30
            """
            daily_trades = pd.read_sql_query(daily_trades_query, conn)
            
            return {
                'skipped_signals': skipped_signals,
                'total_actionable_signals': total_actionable_signals,
                'prevention_rate': (skipped_signals / total_actionable_signals * 100) if total_actionable_signals > 0 else 0,
                'daily_trades': daily_trades
            }
    
    def get_position_lifecycle_analysis(self) -> pd.DataFrame:
        """Analyze the complete lifecycle of positions from buy to sell"""
        with sqlite3.connect(self.db_path) as conn:
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
                (SELECT COUNT(*) FROM stop_strategies ss WHERE ss.symbol = p.symbol AND ss.is_active = TRUE) as active_stops
            FROM positions p
            LEFT JOIN trades t ON p.symbol = t.symbol
            GROUP BY p.symbol
            ORDER BY p.total_invested DESC
            """
            
            df = pd.read_sql_query(query, conn)
            
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
        """Get daily trading performance including stop loss actions"""
        with sqlite3.connect(self.db_path) as conn:
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
                GROUP BY date
            ) dt ON DATE(br.run_date) = dt.date
            ORDER BY br.run_date
            """
            
            df = pd.read_sql_query(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            df['net_daily'] = df['proceeds'] - df['invested']
            df['cumulative_net'] = df['net_daily'].cumsum()
            df['total_actions'] = df['trades_executed'] + df['stop_losses_executed']
            
            return df
    
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
            ax1.set_title('Cumulative Profit/Loss Over Time', fontsize=14, fontweight='bold')
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
            plt.savefig(save_path / 'enhanced_daily_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Stop Loss Effectiveness Chart
        stop_effectiveness = self.get_stop_loss_effectiveness()
        if not stop_effectiveness.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Stop Loss Trigger Rate
            trigger_rate = (stop_effectiveness['stop_losses_triggered'] / stop_effectiveness['positions_created'] * 100).fillna(0)
            ax1.bar(stop_effectiveness['strategy_type'], trigger_rate, alpha=0.7)
            ax1.set_title('Stop Loss Trigger Rate by Strategy', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Trigger Rate (%)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Average Stop Loss Return
            ax2.bar(stop_effectiveness['strategy_type'], stop_effectiveness['avg_stop_loss_return'].fillna(0), 
                   color=['red' if x < 0 else 'green' for x in stop_effectiveness['avg_stop_loss_return'].fillna(0)])
            ax2.set_title('Average Return When Stop Loss Triggered', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Return (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(save_path / 'stop_loss_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Enhanced Signal Effectiveness Chart
        signal_eff = self.get_signal_effectiveness()
        if not signal_eff.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Trading volume by phase (including stop losses)
            ax1.bar(signal_eff['signal_phase'], signal_eff['total_trades'])
            ax1.set_title('Trades by Signal Type', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Trades')
            ax1.tick_params(axis='x', rotation=45)
            
            # ROI by phase
            colors = ['red' if x < 0 else 'green' for x in signal_eff['roi_percent']]
            ax2.bar(signal_eff['signal_phase'], signal_eff['roi_percent'], color=colors)
            ax2.set_title('ROI by Signal Type', fontsize=14, fontweight='bold')
            ax2.set_ylabel('ROI (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(save_path / 'enhanced_signal_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Position Lifecycle Analysis Chart
        positions = self.get_position_lifecycle_analysis()
        if not positions.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Exit method distribution
            exit_counts = positions['exit_method'].value_counts()
            ax1.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Position Exit Methods', fontsize=14, fontweight='bold')
            
            # ROI by exit method
            roi_by_exit = positions.groupby('exit_method')['roi_percent'].mean()
            colors = ['red' if x < 0 else 'green' for x in roi_by_exit.values]
            ax2.bar(roi_by_exit.index, roi_by_exit.values, color=colors)
            ax2.set_title('Average ROI by Exit Method', fontsize=14, fontweight='bold')
            ax2.set_ylabel('ROI (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(save_path / 'position_lifecycle_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Enhanced charts saved to {save_path}/")
    
    def generate_enhanced_report(self, save_path="reports") -> str:
        """Generate comprehensive trading report with stop loss analysis"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        summary = self.get_trading_summary()
        signal_eff = self.get_signal_effectiveness()
        stop_eff = self.get_stop_loss_effectiveness()
        positions = self.get_position_lifecycle_analysis()
        day_trade_stats = self.get_day_trade_prevention_stats()
        
        report = f"""
# ENHANCED WYCKOFF TRADING BOT PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
- **Total Trades**: {summary['total_trades']} ({summary['buy_trades']} buys, {summary['sell_trades']} sells)
- **Stop Loss Trades**: {summary['stop_loss_trades']} ({summary['total_stop_losses_executed']} total executions)
- **Total Invested**: ${summary['total_invested']:,.2f}
- **Total Proceeds**: ${summary['total_proceeds']:,.2f}
- **Net Profit/Loss**: ${summary['net_profit']:,.2f}
- **Current Positions**: {summary['current_positions']} (${summary['current_value']:,.2f} invested)
- **Active Stop Strategies**: {summary['active_stop_strategies']}
- **Bot Success Rate**: {summary['success_rate']:.1f}% ({summary['successful_runs']}/{summary['total_runs']} runs)
- **Trading Period**: {summary['first_trade']} to {summary['last_trade']}

## STOP LOSS PERFORMANCE ANALYSIS
"""
        
        if not stop_eff.empty:
            for _, row in stop_eff.iterrows():
                trigger_rate = (row['stop_losses_triggered'] / row['positions_created'] * 100) if row['positions_created'] > 0 else 0
                report += f"- **{row['strategy_type']}**: {row['positions_created']} positions, {trigger_rate:.1f}% trigger rate"
                if pd.notna(row['avg_stop_loss_return']):
                    report += f", {row['avg_stop_loss_return']:.1f}% avg return\n"
                else:
                    report += "\n"
        
        report += "\n## SIGNAL EFFECTIVENESS ANALYSIS\n"
        if not signal_eff.empty:
            for _, row in signal_eff.head(8).iterrows():
                report += f"- **{row['signal_phase']}**: {row['total_trades']} trades, {row['roi_percent']:.1f}% ROI\n"
        
        report += "\n## DAY TRADE PREVENTION STATS\n"
        report += f"- **Signals Skipped**: {day_trade_stats['skipped_signals']} ({day_trade_stats['prevention_rate']:.1f}% of actionable signals)\n"
        report += f"- **Prevention Effectiveness**: Successfully avoided potential day trade violations\n"
        
        report += "\n## POSITION LIFECYCLE ANALYSIS\n"
        if not positions.empty:
            exit_summary = positions['exit_method'].value_counts()
            for exit_method, count in exit_summary.items():
                avg_roi = positions[positions['exit_method'] == exit_method]['roi_percent'].mean()
                report += f"- **{exit_method}**: {count} positions, {avg_roi:.1f}% avg ROI\n"
        
        report += "\n## TOP PERFORMING POSITIONS\n"
        if not positions.empty:
            top_positions = positions.nlargest(10, 'roi_percent')
            for _, row in top_positions.iterrows():
                status_emoji = "🟢" if row['status'] == 'OPEN' else "🔵"
                exit_emoji = "🛑" if row['exit_method'] == 'STOP_LOSS' else "📈" if row['exit_method'] == 'WYCKOFF_SIGNAL' else "⏳"
                report += f"- {status_emoji}{exit_emoji} **{row['symbol']}**: ${row['total_invested']:.2f} invested, {row['roi_percent']:.1f}% ROI ({row['exit_method']})\n"
        
        # Key insights
        report += "\n## KEY INSIGHTS\n"
        
        if not signal_eff.empty:
            best_signal = signal_eff.loc[signal_eff['roi_percent'].idxmax()]
            worst_signal = signal_eff.loc[signal_eff['roi_percent'].idxmin()]
            report += f"- Most profitable signal: **{best_signal['signal_phase']}** ({best_signal['roi_percent']:.1f}% ROI)\n"
            report += f"- Least profitable signal: **{worst_signal['signal_phase']}** ({worst_signal['roi_percent']:.1f}% ROI)\n"
        
        if summary['net_profit'] > 0:
            report += f"- Overall profitability: **PROFITABLE** (${summary['net_profit']:.2f} net gain)\n"
        else:
            report += f"- Overall profitability: **UNPROFITABLE** (${abs(summary['net_profit']):.2f} net loss)\n"
        
        # Stop loss insights
        stop_loss_effectiveness = (summary['stop_loss_trades'] / summary['total_trades'] * 100) if summary['total_trades'] > 0 else 0
        report += f"- Stop loss utilization: **{stop_loss_effectiveness:.1f}%** of all trades\n"
        
        if summary['stop_loss_proceeds'] > 0:
            stop_loss_recovery = (summary['stop_loss_proceeds'] / summary['total_invested'] * 100) if summary['total_invested'] > 0 else 0
            report += f"- Capital preserved by stop losses: **{stop_loss_recovery:.1f}%** of total invested\n"
        
        # Recommendations
        report += "\n## RECOMMENDATIONS\n"
        
        if not signal_eff.empty and worst_signal['roi_percent'] < -5:
            report += f"- Consider reducing exposure to **{worst_signal['signal_phase']}** signals (currently {worst_signal['roi_percent']:.1f}% ROI)\n"
        
        if summary['success_rate'] < 80:
            report += "- Bot success rate below 80% - review error logs and system stability\n"
        
        if summary['current_positions'] > 15:
            report += "- High number of open positions - consider position sizing limits\n"
        
        if not positions.empty:
            long_positions = positions[positions['days_held'] > 30]
            if len(long_positions) > 5:
                report += f"- Consider reviewing {len(long_positions)} positions held longer than 30 days\n"
        
        # Stop loss recommendations
        if stop_loss_effectiveness < 10:
            report += "- Stop loss utilization is low - consider tightening stop loss percentages\n"
        elif stop_loss_effectiveness > 50:
            report += "- High stop loss utilization - consider loosening stop loss percentages or improving entry timing\n"
        
        # Save report
        report_path = save_path / f"enhanced_trading_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📋 Enhanced report saved to {report_path}")
        return report
    
    def export_enhanced_data(self, save_path="reports"):
        """Export all enhanced data to CSV files"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Export all tables including new ones
            tables = ['trades', 'signals', 'positions', 'bot_runs', 'stop_strategies']
            
            for table in tables:
                try:
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    csv_path = save_path / f"{table}.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"💾 Exported {table} to {csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not export {table}: {e}")


def main():
    """Main function for enhanced analytics dashboard"""
    parser = argparse.ArgumentParser(description='Enhanced Wyckoff Trading Bot Analytics')
    parser.add_argument('--db', default='data/trading_bot.db', help='Database path')
    parser.add_argument('--output', default='reports', help='Output directory')
    parser.add_argument('--charts', action='store_true', help='Generate charts')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--export', action='store_true', help='Export data to CSV')
    parser.add_argument('--all', action='store_true', help='Generate everything')
    
    args = parser.parse_args()
    
    try:
        analytics = EnhancedTradingAnalytics(args.db)
        
        if args.all or args.charts:
            print("📊 Generating enhanced performance charts...")
            analytics.create_enhanced_performance_charts(args.output)
        
        if args.all or args.report:
            print("📋 Generating enhanced performance report...")
            report = analytics.generate_enhanced_report(args.output)
            
            # Also print summary to console
            summary = analytics.get_trading_summary()
            print("\n" + "="*60)
            print("ENHANCED TRADING BOT SUMMARY")
            print("="*60)
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Stop Loss Trades: {summary['stop_loss_trades']}")
            print(f"Net P&L: ${summary['net_profit']:,.2f}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Open Positions: {summary['current_positions']}")
            print(f"Active Stop Strategies: {summary['active_stop_strategies']}")
            print("="*60)
        
        if args.all or args.export:
            print("💾 Exporting enhanced data...")
            analytics.export_enhanced_data(args.output)
        
        if not any([args.charts, args.report, args.export, args.all]):
            # Default: show summary
            summary = analytics.get_trading_summary()
            print("📊 ENHANCED QUICK SUMMARY")
            print(f"Trades: {summary['total_trades']} | Stop Losses: {summary['stop_loss_trades']} | P&L: ${summary['net_profit']:,.2f} | Success: {summary['success_rate']:.1f}%")
            print("Use --help for more options")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())