#!/usr/bin/env python3
"""
Position Building Analytics Dashboard
Comprehensive analytics for fractional share position building and scaling strategies
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
import json


class PositionBuildingAnalytics:
    """Analytics engine for fractional position building strategies"""
    
    def __init__(self, db_path="data/trading_bot.db", bot_id="wyckoff_bot_v1"):
        self.db_path = Path(db_path)
        self.bot_id = bot_id
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Trading database not found: {db_path}")
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"📊 Position Building Analytics for bot_id: {self.bot_id}")
    
    def get_position_building_summary(self) -> Dict:
        """Get overall position building performance summary"""
        with sqlite3.connect(self.db_path) as conn:
            # Current positions with building status
            current_positions = conn.execute("""
                SELECT COUNT(*) as total_positions,
                       SUM(CASE WHEN position_status = 'BUILDING' THEN 1 ELSE 0 END) as building_positions,
                       SUM(CASE WHEN position_status = 'COMPLETE' THEN 1 ELSE 0 END) as complete_positions,
                       SUM(CASE WHEN position_status = 'SCALING_OUT' THEN 1 ELSE 0 END) as scaling_positions,
                       SUM(target_position_size) as total_target_value,
                       SUM(total_invested) as total_invested,
                       AVG(current_allocation_pct) as avg_allocation_pct
                FROM positions_enhanced WHERE bot_id = ?
            """, (self.bot_id,)).fetchone()
            
            # Position events summary
            events_summary = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM position_events WHERE bot_id = ?
                GROUP BY event_type
            """, (self.bot_id,)).fetchall()
            
            # Partial sales summary
            sales_summary = conn.execute("""
                SELECT COUNT(*) as total_sales,
                       SUM(profit_amount) as total_profit,
                       AVG(gain_pct) as avg_gain_pct,
                       SUM(shares_sold * sale_price) as total_proceeds
                FROM partial_sales WHERE bot_id = ?
            """, (self.bot_id,)).fetchone()
            
            # Phase entry analysis
            phase_analysis = conn.execute("""
                SELECT wyckoff_phase, COUNT(*) as entries
                FROM position_events 
                WHERE event_type = 'INITIAL_ENTRY' AND bot_id = ?
                GROUP BY wyckoff_phase
            """, (self.bot_id,)).fetchall()
            
            return {
                'current_positions': {
                    'total': current_positions[0] or 0,
                    'building': current_positions[1] or 0,
                    'complete': current_positions[2] or 0,
                    'scaling_out': current_positions[3] or 0,
                    'total_target_value': current_positions[4] or 0,
                    'total_invested': current_positions[5] or 0,
                    'avg_allocation_pct': current_positions[6] or 0
                },
                'events': dict(events_summary),
                'partial_sales': {
                    'total_sales': sales_summary[0] or 0,
                    'total_profit': sales_summary[1] or 0,
                    'avg_gain_pct': sales_summary[2] or 0,
                    'total_proceeds': sales_summary[3] or 0
                },
                'phase_entries': dict(phase_analysis)
            }
    
    def get_position_building_effectiveness(self) -> pd.DataFrame:
        """Analyze effectiveness of position building by phase"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT 
                pe.symbol,
                pe.wyckoff_phase,
                pe.shares_traded,
                pe.price as entry_price,
                pe.allocation_after,
                pe.event_date,
                pos.position_status,
                pos.avg_cost,
                pos.current_allocation_pct,
                pos.target_position_size
            FROM position_events pe
            LEFT JOIN positions_enhanced pos ON pe.symbol = pos.symbol AND pe.bot_id = pos.bot_id
            WHERE pe.event_type IN ('INITIAL_ENTRY', 'ADDITION') AND pe.bot_id = ?
            ORDER BY pe.symbol, pe.event_date
            """
            
            return pd.read_sql_query(query, conn, params=(self.bot_id,))
    
    def get_scaling_out_performance(self) -> pd.DataFrame:
        """Analyze scaling out performance"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT 
                symbol,
                sale_date,
                shares_sold,
                sale_price,
                sale_reason,
                gain_pct,
                profit_amount,
                remaining_shares
            FROM partial_sales 
            WHERE bot_id = ?
            ORDER BY sale_date, symbol
            """
            
            return pd.read_sql_query(query, conn, params=(self.bot_id,))
    
    def get_position_lifecycle_analysis(self) -> pd.DataFrame:
        """Analyze complete position lifecycles"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT 
                pos.symbol,
                pos.target_position_size,
                pos.current_allocation_pct,
                pos.total_shares,
                pos.avg_cost,
                pos.total_invested,
                pos.entry_phases,
                pos.addition_count,
                pos.position_status,
                pos.first_entry_date,
                pos.last_addition_date,
                pos.wyckoff_score,
                -- Calculate days to complete position
                CASE 
                    WHEN pos.current_allocation_pct >= 0.95 THEN 
                        julianday(pos.last_addition_date) - julianday(pos.first_entry_date)
                    ELSE NULL 
                END as days_to_complete,
                -- Get total profit from scaling out
                COALESCE(sales.total_profit, 0) as realized_profit,
                COALESCE(sales.total_sales, 0) as scaling_actions
            FROM positions_enhanced pos
            LEFT JOIN (
                SELECT symbol, 
                       SUM(profit_amount) as total_profit,
                       COUNT(*) as total_sales
                FROM partial_sales 
                WHERE bot_id = ?
                GROUP BY symbol
            ) sales ON pos.symbol = sales.symbol
            WHERE pos.bot_id = ?
            ORDER BY pos.first_entry_date DESC
            """
            
            return pd.read_sql_query(query, conn, params=(self.bot_id, self.bot_id))
    
    def get_phase_performance_analysis(self) -> Dict:
        """Analyze performance by Wyckoff phase"""
        with sqlite3.connect(self.db_path) as conn:
            # Initial entries by phase
            initial_entries = conn.execute("""
                SELECT wyckoff_phase, COUNT(*) as entries, AVG(price) as avg_entry_price
                FROM position_events 
                WHERE event_type = 'INITIAL_ENTRY' AND bot_id = ?
                GROUP BY wyckoff_phase
            """, (self.bot_id,)).fetchall()
            
            # Additions by phase
            additions = conn.execute("""
                SELECT wyckoff_phase, COUNT(*) as additions, AVG(price) as avg_add_price
                FROM position_events 
                WHERE event_type = 'ADDITION' AND bot_id = ?
                GROUP BY wyckoff_phase
            """, (self.bot_id,)).fetchall()
            
            # Success rate by initial entry phase
            success_analysis = conn.execute("""
                SELECT 
                    pe.wyckoff_phase,
                    COUNT(DISTINCT pe.symbol) as positions_started,
                    COUNT(DISTINCT CASE WHEN pos.current_allocation_pct >= 0.95 THEN pe.symbol END) as completed_positions,
                    COUNT(DISTINCT CASE WHEN sales.symbol IS NOT NULL THEN pe.symbol END) as profitable_exits
                FROM position_events pe
                LEFT JOIN positions_enhanced pos ON pe.symbol = pos.symbol AND pe.bot_id = pos.bot_id
                LEFT JOIN partial_sales sales ON pe.symbol = sales.symbol AND pe.bot_id = sales.bot_id
                WHERE pe.event_type = 'INITIAL_ENTRY' AND pe.bot_id = ?
                GROUP BY pe.wyckoff_phase
            """, (self.bot_id,)).fetchall()
            
            return {
                'initial_entries': {phase: {'count': count, 'avg_price': price} 
                                  for phase, count, price in initial_entries},
                'additions': {phase: {'count': count, 'avg_price': price} 
                            for phase, count, price in additions},
                'success_rates': {phase: {
                    'started': started,
                    'completed': completed,
                    'profitable': profitable,
                    'completion_rate': completed/started if started > 0 else 0,
                    'profit_rate': profitable/started if started > 0 else 0
                } for phase, started, completed, profitable in success_analysis}
            }
    
    def generate_position_building_report(self, output_dir: str = "reports") -> str:
        """Generate comprehensive position building report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"position_building_report_{timestamp}.txt"
        
        # Get all analytics data
        summary = self.get_position_building_summary()
        effectiveness_df = self.get_position_building_effectiveness()
        scaling_df = self.get_scaling_out_performance()
        lifecycle_df = self.get_position_lifecycle_analysis()
        phase_performance = self.get_phase_performance_analysis()
        
        # Generate report
        report = f"""
=== FRACTIONAL SHARE POSITION BUILDING PERFORMANCE REPORT ===

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Bot ID: {self.bot_id}

=== CURRENT PORTFOLIO STATUS ===
Total Positions: {summary['current_positions']['total']}
  - Building: {summary['current_positions']['building']}
  - Complete: {summary['current_positions']['complete']}
  - Scaling Out: {summary['current_positions']['scaling_out']}

Portfolio Allocation:
  - Target Value: ${summary['current_positions']['total_target_value']:,.2f}
  - Invested: ${summary['current_positions']['total_invested']:,.2f}
  - Average Allocation: {summary['current_positions']['avg_allocation_pct']:.1%}

=== POSITION BUILDING ACTIVITY ===
"""
        
        # Add events summary
        for event_type, count in summary['events'].items():
            report += f"{event_type.replace('_', ' ').title()}: {count}\n"
        
        report += f"""
=== SCALING OUT PERFORMANCE ===
Total Scaling Actions: {summary['partial_sales']['total_sales']}
Total Profit Realized: ${summary['partial_sales']['total_profit']:,.2f}
Average Gain per Sale: {summary['partial_sales']['avg_gain_pct']:.1%}
Total Proceeds: ${summary['partial_sales']['total_proceeds']:,.2f}

=== WYCKOFF PHASE ANALYSIS ===
"""
        
        # Add phase performance
        for phase, data in phase_performance['success_rates'].items():
            if data['started'] > 0:
                report += f"""
{phase} Phase:
  - Positions Started: {data['started']}
  - Completion Rate: {data['completion_rate']:.1%}
  - Profit Rate: {data['profit_rate']:.1%}
"""
        
        report += f"""
=== POSITION BUILDING EFFECTIVENESS ===
"""
        
        if not effectiveness_df.empty:
            # Analyze by phase
            phase_stats = effectiveness_df.groupby('wyckoff_phase').agg({
                'shares_traded': ['count', 'mean'],
                'entry_price': 'mean',
                'allocation_after': 'mean'
            }).round(2)
            
            for phase in phase_stats.index:
                entries = phase_stats.loc[phase, ('shares_traded', 'count')]
                avg_shares = phase_stats.loc[phase, ('shares_traded', 'mean')]
                avg_price = phase_stats.loc[phase, ('entry_price', 'mean')]
                avg_allocation = phase_stats.loc[phase, ('allocation_after', 'mean')]
                
                report += f"""
{phase} Phase Building:
  - Entries: {entries}
  - Avg Shares per Entry: {avg_shares:.5f}
  - Avg Entry Price: ${avg_price:.2f}
  - Avg Allocation After: {avg_allocation:.1%}
"""
        
        report += f"""
=== INDIVIDUAL POSITION ANALYSIS ===
"""
        
        if not lifecycle_df.empty:
            # Show top positions by performance
            for _, position in lifecycle_df.head(10).iterrows():
                entry_phases = json.loads(position['entry_phases']) if position['entry_phases'] else []
                phases_str = ', '.join(entry_phases)
                
                report += f"""
{position['symbol']}:
  - Target Size: ${position['target_position_size']:.2f}
  - Allocation: {position['current_allocation_pct']:.1%}
  - Invested: ${position['total_invested']:.2f}
  - Entry Phases: {phases_str}
  - Additions: {position['addition_count']}
  - Status: {position['position_status']}
  - Realized Profit: ${position['realized_profit']:.2f}
  - Scaling Actions: {position['scaling_actions']}
"""
                
                if position['days_to_complete']:
                    report += f"  - Days to Complete: {position['days_to_complete']:.0f}\n"
                report += "\n"
        
        report += f"""
=== SCALING OUT DETAILS ===
"""
        
        if not scaling_df.empty:
            # Group scaling actions by reason
            scaling_summary = scaling_df.groupby('sale_reason').agg({
                'profit_amount': ['sum', 'mean', 'count'],
                'gain_pct': 'mean'
            }).round(2)
            
            for reason in scaling_summary.index:
                total_profit = scaling_summary.loc[reason, ('profit_amount', 'sum')]
                avg_profit = scaling_summary.loc[reason, ('profit_amount', 'mean')]
                count = scaling_summary.loc[reason, ('profit_amount', 'count')]
                avg_gain = scaling_summary.loc[reason, ('gain_pct', 'mean')]
                
                report += f"""
{reason}:
  - Actions: {count}
  - Total Profit: ${total_profit:.2f}
  - Average Profit: ${avg_profit:.2f}
  - Average Gain: {avg_gain:.1%}
"""
        
        # Calculate key metrics
        if summary['current_positions']['total'] > 0:
            allocation_efficiency = summary['current_positions']['total_invested'] / summary['current_positions']['total_target_value']
        else:
            allocation_efficiency = 0
        
        total_actions = sum(summary['events'].values())
        
        report += f"""
=== KEY PERFORMANCE METRICS ===
Position Building Efficiency: {allocation_efficiency:.1%}
Total Position Building Actions: {total_actions}
Average Position Allocation: {summary['current_positions']['avg_allocation_pct']:.1%}
Scaling Out Success Rate: {summary['partial_sales']['avg_gain_pct']:.1%}
Portfolio Utilization: {summary['current_positions']['total_invested']:,.0f} / {summary['current_positions']['total_target_value']:,.0f}

=== RECOMMENDATIONS ===
"""
        
        # Add recommendations based on performance
        if summary['current_positions']['building'] > summary['current_positions']['complete']:
            report += "- Focus on completing existing positions before starting new ones\n"
        
        if summary['partial_sales']['avg_gain_pct'] > 0.15:
            report += "- Scaling out strategy is performing well\n"
        elif summary['partial_sales']['total_sales'] > 0:
            report += "- Consider adjusting scaling out thresholds\n"
        
        if allocation_efficiency < 0.7:
            report += "- Consider increasing position sizes or reducing targets\n"
        
        report += f"""
=== END OF REPORT ===
Report saved to: {report_file}
"""
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return str(report_file)
    
    def create_position_building_charts(self, output_dir: str = "reports"):
        """Create visualization charts for position building performance"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get data
        summary = self.get_position_building_summary()
        lifecycle_df = self.get_position_lifecycle_analysis()
        scaling_df = self.get_scaling_out_performance()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Position Status Distribution
        if summary['current_positions']['total'] > 0:
            status_data = [
                summary['current_positions']['building'],
                summary['current_positions']['complete'],
                summary['current_positions']['scaling_out']
            ]
            status_labels = ['Building', 'Complete', 'Scaling Out']
            
            ax1.pie(status_data, labels=status_labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Position Status Distribution', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Positions', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Position Status Distribution', fontsize=14, fontweight='bold')
        
        # 2. Allocation Progress
        if not lifecycle_df.empty:
            allocation_data = lifecycle_df['current_allocation_pct'].values
            ax2.hist(allocation_data, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(allocation_data.mean(), color='red', linestyle='--', 
                       label=f'Average: {allocation_data.mean():.1%}')
            ax2.set_xlabel('Allocation Percentage')
            ax2.set_ylabel('Number of Positions')
            ax2.set_title('Position Allocation Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Scaling Out Performance
        if not scaling_df.empty and len(scaling_df) > 0:
            scaling_by_reason = scaling_df.groupby('sale_reason')['profit_amount'].sum()
            scaling_by_reason.plot(kind='bar', ax=ax3, color='lightgreen')
            ax3.set_title('Profit by Scaling Reason', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Total Profit ($)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Scaling Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Profit by Scaling Reason', fontsize=14, fontweight='bold')
        
        # 4. Position Building Timeline
        if not lifecycle_df.empty:
            # Convert dates and plot position starts over time
            lifecycle_df['first_entry_date'] = pd.to_datetime(lifecycle_df['first_entry_date'])
            timeline_data = lifecycle_df.groupby(lifecycle_df['first_entry_date'].dt.date).size()
            
            timeline_data.plot(kind='line', marker='o', ax=ax4, color='purple')
            ax4.set_title('Position Building Timeline', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Positions Started')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = output_path / f"position_building_charts_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Charts saved: {chart_file}")
        return str(chart_file)


def main():
    """Main function for position building analytics"""
    parser = argparse.ArgumentParser(description='Position Building Analytics Dashboard')
    parser.add_argument('--db', default='data/trading_bot.db', help='Database path')
    parser.add_argument('--bot-id', default='wyckoff_bot_v1', help='Bot ID to analyze')
    parser.add_argument('--output', default='reports', help='Output directory')
    parser.add_argument('--charts', action='store_true', help='Generate charts')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--all', action='store_true', help='Generate everything')
    
    args = parser.parse_args()
    
    try:
        analytics = PositionBuildingAnalytics(args.db, args.bot_id)
        
        if args.all or args.report:
            print("📋 Generating position building report...")
            report_file = analytics.generate_position_building_report(args.output)
            print(f"✅ Report saved: {report_file}")
        
        if args.all or args.charts:
            print("📊 Generating position building charts...")
            chart_file = analytics.create_position_building_charts(args.output)
            print(f"✅ Charts saved: {chart_file}")
        
        # Show summary
        summary = analytics.get_position_building_summary()
        print(f"\n" + "="*60)
        print(f"POSITION BUILDING SUMMARY (Bot: {args.bot_id})")
        print("="*60)
        print(f"Current Positions: {summary['current_positions']['total']}")
        print(f"  - Building: {summary['current_positions']['building']}")
        print(f"  - Complete: {summary['current_positions']['complete']}")
        print(f"  - Scaling Out: {summary['current_positions']['scaling_out']}")
        print(f"Target Portfolio Value: ${summary['current_positions']['total_target_value']:,.2f}")
        print(f"Currently Invested: ${summary['current_positions']['total_invested']:,.2f}")
        print(f"Average Allocation: {summary['current_positions']['avg_allocation_pct']:.1%}")
        print(f"Scaling Actions: {summary['partial_sales']['total_sales']}")
        print(f"Realized Profit: ${summary['partial_sales']['total_profit']:,.2f}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())