#!/usr/bin/env python3
"""
Wyckoff Trading Bot Analytics Dashboard
Provides comprehensive analysis of trading performance and signal effectiveness
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


class TradingAnalytics:
    """Analytics engine for trading bot performance"""
    
    def __init__(self, db_path="data/trading_bot.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Trading database not found: {db_path}")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def get_trading_summary(self) -> Dict:
        """Get overall trading performance summary"""
        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            total_buy_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'BUY'").fetchone()[0]
            total_sell_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'SELL'").fetchone()[0]
            
            # Investment amounts
            total_invested = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'BUY'").fetchone()[0] or 0
            total_proceeds = conn.execute("SELECT SUM(total_value) FROM trades WHERE action = 'SELL'").fetchone()[0] or 0
            
            # Current positions
            current_positions = conn.execute("SELECT COUNT(*) FROM positions WHERE total_shares > 0").fetchone()[0]
            current_value = conn.execute("SELECT SUM(total_invested) FROM positions WHERE total_shares > 0").fetchone()[0] or 0
            
            # Bot runs
            total_runs = conn.execute("SELECT COUNT(*) FROM bot_runs").fetchone()[0]
            successful_runs = conn.execute("SELECT COUNT(*) FROM bot_runs WHERE status LIKE 'SUCCESS%'").fetchone()[0]
            
            # Date range
            first_trade = conn.execute("SELECT MIN(date) FROM trades").fetchone()[0]
            last_trade = conn.execute("SELECT MAX(date) FROM trades").fetchone()[0]
            
            return {
                'total_trades': total_trades,
                'buy_trades': total_buy_trades,
                'sell_trades': total_sell_trades,
                'total_invested': total_invested,
                'total_proceeds': total_proceeds,
                'net_profit': total_proceeds - total_invested,
                'current_positions': current_positions,
                'current_value': current_value,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                'first_trade': first_trade,
                'last_trade': last_trade
            }
    
    def get_signal_effectiveness(self) -> pd.DataFrame:
        """Analyze effectiveness of different Wyckoff phases"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT 
                signal_phase,
                COUNT(*) as total_trades,
                AVG(signal_strength) as avg_strength,
                COUNT(CASE WHEN action = 'BUY' THEN 1 END) as buy_count,
                COUNT(CASE WHEN action = 'SELL' THEN 1 END) as sell_count,
                SUM(CASE WHEN action = 'BUY' THEN total_value ELSE 0 END) as total_invested,
                SUM(CASE WHEN action = 'SELL' THEN total_value ELSE 0 END) as total_proceeds
            FROM trades 
            WHERE signal_phase IS NOT NULL
            GROUP BY signal_phase
            ORDER BY total_trades DESC
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Calculate profit/loss and success rate
            df['net_profit'] = df['total_proceeds'] - df['total_invested']
            df['roi_percent'] = (df['net_profit'] / df['total_invested'] * 100).round(2)
            
            return df
    
    def get_sector_performance(self) -> pd.DataFrame:
        """Analyze performance by sector"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT 
                s.sector,
                COUNT(t.id) as total_trades,
                AVG(s.combined_score) as avg_score,
                COUNT(CASE WHEN t.action = 'BUY' THEN 1 END) as buys,
                COUNT(CASE WHEN t.action = 'SELL' THEN 1 END) as sells
            FROM signals s
            LEFT JOIN trades t ON s.symbol = t.symbol AND s.date = t.date
            WHERE s.action_taken IS NOT NULL
            GROUP BY s.sector
            ORDER BY total_trades DESC
            """
            
            return pd.read_sql_query(query, conn)
    
    def get_daily_performance(self) -> pd.DataFrame:
        """Get daily trading performance"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT 
                date,
                COUNT(*) as trades,
                SUM(CASE WHEN action = 'BUY' THEN total_value ELSE 0 END) as invested,
                SUM(CASE WHEN action = 'SELL' THEN total_value ELSE 0 END) as proceeds,
                COUNT(CASE WHEN action = 'BUY' THEN 1 END) as buys,
                COUNT(CASE WHEN action = 'SELL' THEN 1 END) as sells
            FROM trades
            GROUP BY date
            ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            df['net_daily'] = df['proceeds'] - df['invested']
            df['cumulative_net'] = df['net_daily'].cumsum()
            
            return df
    
    def get_position_analysis(self) -> pd.DataFrame:
        """Analyze current and past positions"""
        with sqlite3.connect(self.db_path) as conn:
            # Get all symbols that have been traded
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
                CASE WHEN p.total_shares > 0 THEN 'OPEN' ELSE 'CLOSED' END as status
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
            
            return df
    
    def create_performance_charts(self, save_path="reports"):
        """Create comprehensive performance charts"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # 1. Daily Performance Chart
        daily_perf = self.get_daily_performance()
        if not daily_perf.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Cumulative P&L
            ax1.plot(daily_perf['date'], daily_perf['cumulative_net'], marker='o', linewidth=2)
            ax1.set_title('Cumulative Profit/Loss Over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative P&L ($)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Daily trading volume
            ax2.bar(daily_perf['date'], daily_perf['invested'], alpha=0.7, label='Invested')
            ax2.bar(daily_perf['date'], daily_perf['proceeds'], alpha=0.7, label='Proceeds')
            ax2.set_title('Daily Trading Volume', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Amount ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path / 'daily_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Signal Effectiveness Chart
        signal_eff = self.get_signal_effectiveness()
        if not signal_eff.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Trading volume by phase
            ax1.bar(signal_eff['signal_phase'], signal_eff['total_trades'])
            ax1.set_title('Trades by Wyckoff Phase', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Trades')
            ax1.tick_params(axis='x', rotation=45)
            
            # ROI by phase
            colors = ['green' if x > 0 else 'red' for x in signal_eff['roi_percent']]
            ax2.bar(signal_eff['signal_phase'], signal_eff['roi_percent'], color=colors)
            ax2.set_title('ROI by Wyckoff Phase', fontsize=14, fontweight='bold')
            ax2.set_ylabel('ROI (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(save_path / 'signal_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Sector Performance Chart
        sector_perf = self.get_sector_performance()
        if not sector_perf.empty:
            plt.figure(figsize=(12, 8))
            
            # Create bubble chart: x=buys, y=sells, size=avg_score
            plt.scatter(sector_perf['buys'], sector_perf['sells'], 
                       s=sector_perf['avg_score']*500, alpha=0.6)
            
            # Add labels
            for i, row in sector_perf.iterrows():
                plt.annotate(row['sector'], (row['buys'], row['sells']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            plt.xlabel('Buy Signals')
            plt.ylabel('Sell Signals')
            plt.title('Sector Performance (Bubble Size = Avg Score)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path / 'sector_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Position Analysis Chart
        positions = self.get_position_analysis()
        if not positions.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Top positions by investment
            top_positions = positions.head(10)
            colors = ['green' if x == 'OPEN' else 'blue' for x in top_positions['status']]
            ax1.barh(top_positions['symbol'], top_positions['total_invested'], color=colors)
            ax1.set_title('Top Positions by Investment', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Total Invested ($)')
            
            # ROI distribution
            closed_positions = positions[positions['status'] == 'CLOSED']
            if not closed_positions.empty:
                ax2.hist(closed_positions['roi_percent'], bins=20, alpha=0.7, edgecolor='black')
                ax2.set_title('ROI Distribution (Closed Positions)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('ROI (%)')
                ax2.set_ylabel('Frequency')
                ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(save_path / 'position_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Charts saved to {save_path}/")
    
    def generate_report(self, save_path="reports") -> str:
        """Generate comprehensive trading report"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        summary = self.get_trading_summary()
        signal_eff = self.get_signal_effectiveness()
        sector_perf = self.get_sector_performance()
        positions = self.get_position_analysis()
        
        report = f"""
# WYCKOFF TRADING BOT PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
- **Total Trades**: {summary['total_trades']} ({summary['buy_trades']} buys, {summary['sell_trades']} sells)
- **Total Invested**: ${summary['total_invested']:,.2f}
- **Total Proceeds**: ${summary['total_proceeds']:,.2f}
- **Net Profit/Loss**: ${summary['net_profit']:,.2f}
- **Current Positions**: {summary['current_positions']} (${summary['current_value']:,.2f} invested)
- **Bot Success Rate**: {summary['success_rate']:.1f}% ({summary['successful_runs']}/{summary['total_runs']} runs)
- **Trading Period**: {summary['first_trade']} to {summary['last_trade']}

## SIGNAL EFFECTIVENESS ANALYSIS
Best performing Wyckoff phases:
"""
        
        if not signal_eff.empty:
            for _, row in signal_eff.head(5).iterrows():
                report += f"- **{row['signal_phase']}**: {row['total_trades']} trades, {row['roi_percent']:.1f}% ROI\n"
        
        report += "\n## SECTOR PERFORMANCE\n"
        if not sector_perf.empty:
            for _, row in sector_perf.head(5).iterrows():
                report += f"- **{row['sector']}**: {row['total_trades']} trades (Score: {row['avg_score']:.2f})\n"
        
        report += "\n## TOP POSITIONS\n"
        if not positions.empty:
            for _, row in positions.head(10).iterrows():
                status_emoji = "🟢" if row['status'] == 'OPEN' else "🔵"
                report += f"- {status_emoji} **{row['symbol']}**: ${row['total_invested']:.2f} invested, {row['roi_percent']:.1f}% ROI\n"
        
        # Key insights
        report += "\n## KEY INSIGHTS\n"
        
        if not signal_eff.empty:
            best_phase = signal_eff.loc[signal_eff['roi_percent'].idxmax(), 'signal_phase']
            best_roi = signal_eff.loc[signal_eff['roi_percent'].idxmax(), 'roi_percent']
            report += f"- Most profitable phase: **{best_phase}** ({best_roi:.1f}% ROI)\n"
        
        if summary['net_profit'] > 0:
            report += f"- Overall profitability: **PROFITABLE** (${summary['net_profit']:.2f} net gain)\n"
        else:
            report += f"- Overall profitability: **UNPROFITABLE** (${abs(summary['net_profit']):.2f} net loss)\n"
        
        open_positions = positions[positions['status'] == 'OPEN']
        if not open_positions.empty:
            avg_days_held = open_positions['days_held'].mean()
            report += f"- Average holding period for open positions: **{avg_days_held:.0f} days**\n"
        
        # Recommendations
        report += "\n## RECOMMENDATIONS\n"
        
        if not signal_eff.empty:
            worst_phase = signal_eff.loc[signal_eff['roi_percent'].idxmin(), 'signal_phase']
            worst_roi = signal_eff.loc[signal_eff['roi_percent'].idxmin(), 'roi_percent']
            if worst_roi < -5:
                report += f"- Consider reducing exposure to **{worst_phase}** signals (currently {worst_roi:.1f}% ROI)\n"
        
        if summary['success_rate'] < 80:
            report += "- Bot success rate below 80% - review error logs and system stability\n"
        
        if summary['current_positions'] > 15:
            report += "- High number of open positions - consider position sizing limits\n"
        
        # Save report
        report_path = save_path / f"trading_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📋 Report saved to {report_path}")
        return report
    
    def export_data(self, save_path="reports"):
        """Export all data to CSV files"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Export all tables
            tables = ['trades', 'signals', 'positions', 'bot_runs']
            
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                csv_path = save_path / f"{table}.csv"
                df.to_csv(csv_path, index=False)
                print(f"💾 Exported {table} to {csv_path}")


def main():
    """Main function for analytics dashboard"""
    parser = argparse.ArgumentParser(description='Wyckoff Trading Bot Analytics')
    parser.add_argument('--db', default='data/trading_bot.db', help='Database path')
    parser.add_argument('--output', default='reports', help='Output directory')
    parser.add_argument('--charts', action='store_true', help='Generate charts')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--export', action='store_true', help='Export data to CSV')
    parser.add_argument('--all', action='store_true', help='Generate everything')
    
    args = parser.parse_args()
    
    try:
        analytics = TradingAnalytics(args.db)
        
        if args.all or args.charts:
            print("📊 Generating performance charts...")
            analytics.create_performance_charts(args.output)
        
        if args.all or args.report:
            print("📋 Generating performance report...")
            report = analytics.generate_report(args.output)
            
            # Also print summary to console
            summary = analytics.get_trading_summary()
            print("\n" + "="*50)
            print("TRADING BOT SUMMARY")
            print("="*50)
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Net P&L: ${summary['net_profit']:,.2f}")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Open Positions: {summary['current_positions']}")
            print("="*50)
        
        if args.all or args.export:
            print("💾 Exporting data...")
            analytics.export_data(args.output)
        
        if not any([args.charts, args.report, args.export, args.all]):
            # Default: show summary
            summary = analytics.get_trading_summary()
            print("📊 QUICK SUMMARY")
            print(f"Trades: {summary['total_trades']} | P&L: ${summary['net_profit']:,.2f} | Success: {summary['success_rate']:.1f}%")
            print("Use --help for more options")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())