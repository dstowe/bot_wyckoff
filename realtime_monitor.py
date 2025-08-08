#!/usr/bin/env python3
"""
Real-Time Position Building Monitor
Live dashboard for monitoring fractional position building and scaling activities
"""

import sqlite3
import json
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from dataclasses import dataclass
import threading


@dataclass
class PositionStatus:
    """Current status of a position"""
    symbol: str
    target_size: float
    current_allocation: float
    total_invested: float
    shares: float
    avg_cost: float
    entry_phases: List[str]
    status: str
    last_update: str
    unrealized_pnl: Optional[float] = None
    current_price: Optional[float] = None


@dataclass
class ScalingEvent:
    """Scaling out event"""
    symbol: str
    shares_sold: float
    sale_price: float
    profit_amount: float
    gain_pct: float
    reason: str
    date: str


class RealTimeMonitor:
    """Real-time monitoring dashboard for position building"""
    
    def __init__(self, db_path="data/trading_bot.db", bot_id="wyckoff_bot_v1", 
                 refresh_interval=30):
        self.db_path = Path(db_path)
        self.bot_id = bot_id
        self.refresh_interval = refresh_interval
        self.running = False
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Initialize display state
        self.last_positions = {}
        self.last_events = []
        self.start_time = datetime.now()
        
        # Optional: Initialize price fetching (requires webull client)
        self.price_fetcher = None
        try:
            from webull.webull import webull
            self.wb = webull()
            # Note: Would need authentication for live prices
        except:
            self.wb = None
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_current_positions(self) -> List[PositionStatus]:
        """Get current position status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT symbol, target_position_size, current_allocation_pct, total_invested,
                       total_shares, avg_cost, entry_phases, position_status, updated_at
                FROM positions_enhanced 
                WHERE bot_id = ? AND total_shares > 0
                ORDER BY total_invested DESC
            """, (self.bot_id,))
            
            positions = []
            for row in cursor.fetchall():
                symbol, target_size, allocation_pct, invested, shares, avg_cost, phases_json, status, updated = row
                
                try:
                    entry_phases = json.loads(phases_json) if phases_json else []
                except:
                    entry_phases = []
                
                # Try to get current price for P&L calculation
                current_price = self.get_current_price(symbol)
                unrealized_pnl = None
                
                if current_price:
                    unrealized_pnl = (current_price - avg_cost) * shares
                
                positions.append(PositionStatus(
                    symbol=symbol,
                    target_size=target_size,
                    current_allocation=allocation_pct,
                    total_invested=invested,
                    shares=shares,
                    avg_cost=avg_cost,
                    entry_phases=entry_phases,
                    status=status,
                    last_update=updated,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl
                ))
            
            return positions
    
    def get_recent_scaling_events(self, hours=24) -> List[ScalingEvent]:
        """Get recent scaling out events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT symbol, shares_sold, sale_price, profit_amount, gain_pct, sale_reason, sale_date
                FROM partial_sales 
                WHERE bot_id = ? AND sale_date >= ?
                ORDER BY created_at DESC
                LIMIT 10
            """, (self.bot_id, cutoff_str))
            
            events = []
            for row in cursor.fetchall():
                symbol, shares_sold, price, profit, gain_pct, reason, date = row
                events.append(ScalingEvent(
                    symbol=symbol,
                    shares_sold=shares_sold,
                    sale_price=price,
                    profit_amount=profit,
                    gain_pct=gain_pct,
                    reason=reason,
                    date=date
                ))
            
            return events
    
    def get_recent_position_events(self, hours=24) -> List[Dict]:
        """Get recent position building events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT symbol, event_type, event_date, wyckoff_phase, shares_traded, 
                       price, reasoning, created_at
                FROM position_events 
                WHERE bot_id = ? AND event_date >= ?
                ORDER BY created_at DESC
                LIMIT 15
            """, (self.bot_id, cutoff_str))
            
            events = []
            for row in cursor.fetchall():
                symbol, event_type, date, phase, shares, price, reasoning, created = row
                events.append({
                    'symbol': symbol,
                    'event_type': event_type,
                    'date': date,
                    'phase': phase,
                    'shares': shares,
                    'price': price,
                    'reasoning': reasoning,
                    'created_at': created
                })
            
            return events
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price (if webull client available)"""
        if not self.wb:
            return None
        
        try:
            quote = self.wb.get_quote(symbol)
            if quote and 'close' in quote:
                return float(quote['close'])
        except:
            pass
        
        return None
    
    def get_portfolio_summary(self) -> Dict:
        """Get overall portfolio summary"""
        with sqlite3.connect(self.db_path) as conn:
            # Position summary
            position_summary = conn.execute("""
                SELECT 
                    COUNT(*) as total_positions,
                    SUM(CASE WHEN position_status = 'BUILDING' THEN 1 ELSE 0 END) as building,
                    SUM(CASE WHEN position_status = 'COMPLETE' THEN 1 ELSE 0 END) as complete,
                    SUM(CASE WHEN position_status = 'SCALING_OUT' THEN 1 ELSE 0 END) as scaling,
                    SUM(target_position_size) as total_target,
                    SUM(total_invested) as total_invested,
                    AVG(current_allocation_pct) as avg_allocation
                FROM positions_enhanced WHERE bot_id = ?
            """, (self.bot_id,)).fetchone()
            
            # Recent activity (last 24 hours)
            recent_activity = conn.execute("""
                SELECT COUNT(*) FROM position_events 
                WHERE bot_id = ? AND event_date >= date('now', '-1 day')
            """, (self.bot_id,)).fetchone()[0]
            
            # Recent scaling (last 7 days)
            recent_scaling = conn.execute("""
                SELECT COUNT(*), COALESCE(SUM(profit_amount), 0)
                FROM partial_sales 
                WHERE bot_id = ? AND sale_date >= date('now', '-7 days')
            """, (self.bot_id,)).fetchall()[0]
            
            # Bot runs today
            today_runs = conn.execute("""
                SELECT COUNT(*) FROM bot_runs 
                WHERE bot_id = ? AND DATE(run_date) = DATE('now')
            """, (self.bot_id,)).fetchone()[0]
            
            return {
                'total_positions': position_summary[0] or 0,
                'building_positions': position_summary[1] or 0,
                'complete_positions': position_summary[2] or 0,
                'scaling_positions': position_summary[3] or 0,
                'total_target_value': position_summary[4] or 0,
                'total_invested': position_summary[5] or 0,
                'avg_allocation': position_summary[6] or 0,
                'recent_activity': recent_activity,
                'recent_scaling_count': recent_scaling[0],
                'recent_scaling_profit': recent_scaling[1],
                'bot_runs_today': today_runs
            }
    
    def format_positions_display(self, positions: List[PositionStatus]) -> str:
        """Format positions for display"""
        if not positions:
            return "No positions found"
        
        display = "┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n"
        display += "│ Symbol  │   Target    │ Allocation  │  Invested   │   Shares    │  Avg Cost   │   P&L       │\n"
        display += "├─────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n"
        
        for pos in positions[:10]:  # Show top 10 positions
            allocation_bar = self.create_progress_bar(pos.current_allocation, 1.0, 9)
            phases_str = ','.join(pos.entry_phases[:3])  # First 3 phases
            
            # P&L display
            if pos.unrealized_pnl is not None:
                pnl_str = f"${pos.unrealized_pnl:+7.2f}"
                if pos.unrealized_pnl > 0:
                    pnl_str = f"\033[92m{pnl_str}\033[0m"  # Green
                elif pos.unrealized_pnl < 0:
                    pnl_str = f"\033[91m{pnl_str}\033[0m"  # Red
            else:
                pnl_str = "    N/A    "
            
            # Status color
            status_color = ""
            if pos.status == 'BUILDING':
                status_color = "\033[93m"  # Yellow
            elif pos.status == 'COMPLETE':
                status_color = "\033[92m"  # Green
            elif pos.status == 'SCALING_OUT':
                status_color = "\033[94m"  # Blue
            
            display += f"│ {status_color}{pos.symbol:7s}\033[0m │ ${pos.target_size:10.2f} │ {allocation_bar} │ ${pos.total_invested:10.2f} │ {pos.shares:10.5f} │ ${pos.avg_cost:10.2f} │ {pnl_str} │\n"
        
        display += "└─────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘"
        
        return display
    
    def create_progress_bar(self, current: float, target: float, width: int = 10) -> str:
        """Create a text progress bar"""
        if target == 0:
            return " " * width
        
        percentage = min(current / target, 1.0)
        filled = int(percentage * width)
        bar = "█" * filled + "░" * (width - filled)
        
        # Add percentage
        pct_str = f"{percentage:.0%}"
        return f"{bar[:width-len(pct_str)]}{pct_str}"
    
    def format_recent_events(self, events: List[Dict]) -> str:
        """Format recent events for display"""
        if not events:
            return "No recent events"
        
        display = "Recent Position Building Events:\n"
        display += "─" * 60 + "\n"
        
        for event in events[:8]:  # Show last 8 events
            time_str = event['created_at'][:19] if event['created_at'] else event['date']
            event_emoji = {
                'INITIAL_ENTRY': '🎯',
                'ADDITION': '📈',
                'PARTIAL_SALE': '💰',
                'MIGRATION': '🔄'
            }.get(event['event_type'], '📊')
            
            phase_str = f"({event['phase']})" if event['phase'] else ""
            shares_str = f"{event['shares']:8.5f}" if event['shares'] else "     N/A"
            
            display += f"{event_emoji} {time_str[:16]} {event['symbol']:6s} {event['event_type']:12s} {phase_str:8s} {shares_str} @ ${event['price']:7.2f}\n"
        
        return display
    
    def format_scaling_events(self, events: List[ScalingEvent]) -> str:
        """Format scaling events for display"""
        if not events:
            return "No recent scaling events"
        
        display = "Recent Profit Taking (Scaling Out):\n"
        display += "─" * 60 + "\n"
        
        total_profit = 0
        for event in events:
            profit_str = f"${event.profit_amount:+7.2f}"
            if event.profit_amount > 0:
                profit_str = f"\033[92m{profit_str}\033[0m"  # Green
            
            gain_str = f"{event.gain_pct:+6.1%}"
            if event.gain_pct > 0:
                gain_str = f"\033[92m{gain_str}\033[0m"  # Green
            
            display += f"💰 {event.date} {event.symbol:6s} {event.shares_sold:8.5f} @ ${event.sale_price:7.2f} = {profit_str} ({gain_str})\n"
            total_profit += event.profit_amount
        
        if total_profit > 0:
            display += "─" * 60 + "\n"
            display += f"Total Profit Realized: \033[92m${total_profit:+.2f}\033[0m\n"
        
        return display
    
    def display_dashboard(self):
        """Display the complete dashboard"""
        self.clear_screen()
        
        # Header
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        print("═" * 120)
        print(f"🏗️  FRACTIONAL POSITION BUILDING MONITOR - {current_time}")
        print(f"Bot ID: {self.bot_id} | Uptime: {uptime_str} | Refresh: {self.refresh_interval}s")
        print("═" * 120)
        
        # Portfolio Summary
        summary = self.get_portfolio_summary()
        
        print(f"\n📊 PORTFOLIO OVERVIEW")
        print(f"┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐")
        print(f"│ Total Positions │ Building        │ Complete        │ Scaling Out     │")
        print(f"├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
        print(f"│ {summary['total_positions']:15d} │ {summary['building_positions']:15d} │ {summary['complete_positions']:15d} │ {summary['scaling_positions']:15d} │")
        print(f"└─────────────────┴─────────────────┴─────────────────┴─────────────────┘")
        
        allocation_efficiency = (summary['total_invested'] / summary['total_target_value'] * 100) if summary['total_target_value'] > 0 else 0
        
        print(f"\n💰 CAPITAL ALLOCATION")
        print(f"Target Portfolio Value: ${summary['total_target_value']:,.2f}")
        print(f"Currently Invested:     ${summary['total_invested']:,.2f}")
        print(f"Allocation Efficiency:  {allocation_efficiency:.1f}%")
        print(f"Average Position Fill:  {summary['avg_allocation']:.1%}")
        
        print(f"\n🚀 ACTIVITY METRICS")
        print(f"Recent Activity (24h): {summary['recent_activity']} events")
        print(f"Recent Scaling (7d):   {summary['recent_scaling_count']} sales, ${summary['recent_scaling_profit']:+.2f} profit")
        print(f"Bot Runs Today:        {summary['bot_runs_today']}")
        
        # Current Positions
        print(f"\n🎯 CURRENT POSITIONS")
        positions = self.get_current_positions()
        print(self.format_positions_display(positions))
        
        # Two columns for events
        print(f"\n📋 RECENT ACTIVITY")
        
        # Get recent events
        position_events = self.get_recent_position_events()
        scaling_events = self.get_recent_scaling_events()
        
        # Split into two columns
        events_display = self.format_recent_events(position_events)
        scaling_display = self.format_scaling_events(scaling_events)
        
        # Print side by side
        events_lines = events_display.split('\n')
        scaling_lines = scaling_display.split('\n')
        
        max_lines = max(len(events_lines), len(scaling_lines))
        
        for i in range(max_lines):
            left_line = events_lines[i] if i < len(events_lines) else ""
            right_line = scaling_lines[i] if i < len(scaling_lines) else ""
            
            # Remove ANSI codes for width calculation
            import re
            left_clean = re.sub(r'\033\[[0-9;]*m', '', left_line)
            right_clean = re.sub(r'\033\[[0-9;]*m', '', right_line)
            
            left_padded = left_line + " " * (62 - len(left_clean))
            print(f"{left_padded} │ {right_line}")
        
        # Status bar
        print("\n" + "═" * 120)
        print(f"💡 Commands: [Ctrl+C] Exit | [R] Refresh Now | [S] Save Report")
        print(f"🔄 Next refresh in {self.refresh_interval}s...")
    
    def run_monitor(self):
        """Run the monitoring loop"""
        self.running = True
        print("🚀 Starting Real-Time Position Building Monitor...")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                self.display_dashboard()
                
                # Sleep with interrupt handling
                for _ in range(self.refresh_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                
        except KeyboardInterrupt:
            self.running = False
            print("\n\n👋 Monitor stopped by user")
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
            self.running = False
    
    def save_snapshot_report(self) -> str:
        """Save a snapshot report of current status"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"position_monitor_snapshot_{timestamp}.txt"
        
        positions = self.get_current_positions()
        summary = self.get_portfolio_summary()
        recent_events = self.get_recent_position_events()
        scaling_events = self.get_recent_scaling_events()
        
        report = f"""
=== POSITION BUILDING MONITOR SNAPSHOT ===

Snapshot Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Bot ID: {self.bot_id}

=== PORTFOLIO SUMMARY ===
Total Positions: {summary['total_positions']}
  - Building: {summary['building_positions']}
  - Complete: {summary['complete_positions']}
  - Scaling Out: {summary['scaling_positions']}

Target Portfolio Value: ${summary['total_target_value']:,.2f}
Currently Invested: ${summary['total_invested']:,.2f}
Allocation Efficiency: {(summary['total_invested'] / summary['total_target_value'] * 100) if summary['total_target_value'] > 0 else 0:.1f}%

=== CURRENT POSITIONS ===
"""
        
        for pos in positions:
            pnl_str = f"${pos.unrealized_pnl:+.2f}" if pos.unrealized_pnl else "N/A"
            phases_str = ', '.join(pos.entry_phases)
            
            report += f"""
{pos.symbol}:
  - Target Size: ${pos.target_size:.2f}
  - Allocation: {pos.current_allocation:.1%}
  - Invested: ${pos.total_invested:.2f}
  - Shares: {pos.shares:.5f} @ ${pos.avg_cost:.2f}
  - Entry Phases: {phases_str}
  - Status: {pos.status}
  - Unrealized P&L: {pnl_str}
"""
        
        report += f"""
=== RECENT EVENTS ===
"""
        
        for event in recent_events[:10]:
            report += f"{event['created_at']}: {event['symbol']} {event['event_type']} - {event['reasoning']}\n"
        
        report += f"""
=== RECENT SCALING ===
"""
        
        for event in scaling_events:
            report += f"{event.date}: {event.symbol} sold {event.shares_sold:.5f} @ ${event.sale_price:.2f} = ${event.profit_amount:+.2f} ({event.gain_pct:+.1%})\n"
        
        with open(filename, 'w') as f:
            f.write(report)
        
        return filename


class InteractiveMonitor(RealTimeMonitor):
    """Interactive version of the monitor with keyboard commands"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = {
            'r': self.refresh_now,
            's': self.save_report,
            'q': self.quit_monitor,
            'h': self.show_help
        }
    
    def refresh_now(self):
        """Force immediate refresh"""
        pass  # Just return to trigger refresh
    
    def save_report(self):
        """Save current snapshot"""
        try:
            filename = self.save_snapshot_report()
            print(f"\n✅ Snapshot saved to: {filename}")
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\n❌ Error saving report: {e}")
            input("\nPress Enter to continue...")
    
    def quit_monitor(self):
        """Quit the monitor"""
        self.running = False
    
    def show_help(self):
        """Show help screen"""
        print("\n" + "="*60)
        print("POSITION BUILDING MONITOR - HELP")
        print("="*60)
        print("Commands:")
        print("  r - Refresh now")
        print("  s - Save snapshot report")
        print("  h - Show this help")
        print("  q - Quit monitor")
        print("  Ctrl+C - Force quit")
        print("="*60)
        input("\nPress Enter to continue...")


def main():
    """Main monitor execution"""
    parser = argparse.ArgumentParser(description='Real-Time Position Building Monitor')
    parser.add_argument('--db', default='data/trading_bot.db', help='Database path')
    parser.add_argument('--bot-id', default='wyckoff_bot_v1', help='Bot ID to monitor')
    parser.add_argument('--refresh', type=int, default=30, help='Refresh interval in seconds')
    parser.add_argument('--snapshot', action='store_true', help='Save snapshot and exit')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive mode')
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            monitor = InteractiveMonitor(args.db, args.bot_id, args.refresh)
        else:
            monitor = RealTimeMonitor(args.db, args.bot_id, args.refresh)
        
        if args.snapshot:
            # Just save a snapshot and exit
            filename = monitor.save_snapshot_report()
            print(f"📊 Snapshot saved to: {filename}")
            return 0
        
        # Run the monitor
        monitor.run_monitor()
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())