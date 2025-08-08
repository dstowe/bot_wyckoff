#!/usr/bin/env python3
"""
Migration Script: Convert Existing Positions to Enhanced Fractional System
Safely migrates your existing trading bot data to the new position building system
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class PositionMigrationManager:
    """Handles migration from basic position tracking to enhanced fractional system"""
    
    def __init__(self, db_path="data/trading_bot.db", bot_id="wyckoff_bot_v1"):
        self.db_path = Path(db_path)
        self.bot_id = bot_id
        self.logger = self.setup_logging()
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def setup_logging(self):
        """Setup migration logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def backup_database(self) -> str:
        """Create a backup of the database before migration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"trading_bot_backup_{timestamp}.db"
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"✅ Database backed up to: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"❌ Failed to backup database: {e}")
            raise
    
    def check_migration_needed(self) -> Dict:
        """Check what migration steps are needed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if enhanced tables exist
            tables_check = {}
            enhanced_tables = [
                'positions_enhanced',
                'partial_sales', 
                'position_events'
            ]
            
            for table in enhanced_tables:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table,))
                tables_check[table] = cursor.fetchone() is not None
            
            # Check existing positions
            cursor.execute("""
                SELECT COUNT(*) FROM positions 
                WHERE total_shares > 0 AND bot_id = ?
            """, (self.bot_id,))
            existing_positions = cursor.fetchone()[0]
            
            # Check existing trades
            cursor.execute("""
                SELECT COUNT(*) FROM trades WHERE bot_id = ?
            """, (self.bot_id,))
            existing_trades = cursor.fetchone()[0]
            
            # Check if migration already done
            if tables_check.get('positions_enhanced'):
                cursor.execute("""
                    SELECT COUNT(*) FROM positions_enhanced WHERE bot_id = ?
                """, (self.bot_id,))
                migrated_positions = cursor.fetchone()[0]
            else:
                migrated_positions = 0
            
            return {
                'enhanced_tables_exist': all(tables_check.values()),
                'tables_status': tables_check,
                'existing_positions': existing_positions,
                'existing_trades': existing_trades,
                'migrated_positions': migrated_positions,
                'migration_needed': existing_positions > migrated_positions
            }
    
    def create_enhanced_schema(self):
        """Create the enhanced database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Create positions_enhanced table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions_enhanced (
                    symbol TEXT PRIMARY KEY,
                    target_position_size REAL NOT NULL,
                    current_allocation_pct REAL NOT NULL DEFAULT 0.0,
                    total_shares REAL NOT NULL DEFAULT 0.0,
                    avg_cost REAL NOT NULL DEFAULT 0.0,
                    total_invested REAL NOT NULL DEFAULT 0.0,
                    entry_phases TEXT,  -- JSON array of phases entered
                    addition_count INTEGER DEFAULT 1,
                    max_additions INTEGER DEFAULT 3,
                    first_entry_date TEXT,
                    last_addition_date TEXT,
                    account_type TEXT NOT NULL,
                    wyckoff_score REAL DEFAULT 0.5,
                    position_status TEXT DEFAULT 'COMPLETE',
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create partial_sales table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS partial_sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sale_date TEXT NOT NULL,
                    shares_sold REAL NOT NULL,
                    sale_price REAL NOT NULL,
                    sale_reason TEXT NOT NULL,
                    remaining_shares REAL NOT NULL,
                    gain_pct REAL,
                    profit_amount REAL,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create position_events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS position_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    wyckoff_phase TEXT,
                    shares_traded REAL NOT NULL,
                    price REAL NOT NULL,
                    allocation_before REAL DEFAULT 0.0,
                    allocation_after REAL DEFAULT 1.0,
                    reasoning TEXT,
                    bot_id TEXT DEFAULT 'wyckoff_bot_v1',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_positions_enhanced_symbol ON positions_enhanced(symbol, bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_partial_sales_symbol ON partial_sales(symbol, bot_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_position_events_symbol ON position_events(symbol, bot_id)')
            
            self.logger.info("✅ Enhanced database schema created")
    
    def migrate_existing_positions(self) -> int:
        """Migrate existing positions to enhanced system"""
        migrated_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            # Get existing positions
            cursor = conn.execute("""
                SELECT symbol, total_shares, avg_cost, total_invested, 
                       first_purchase_date, last_purchase_date, account_type
                FROM positions 
                WHERE total_shares > 0 AND bot_id = ?
            """, (self.bot_id,))
            
            existing_positions = cursor.fetchall()
            
            for position in existing_positions:
                symbol, shares, avg_cost, invested, first_date, last_date, account_type = position
                
                # Check if already migrated
                existing_enhanced = conn.execute("""
                    SELECT COUNT(*) FROM positions_enhanced 
                    WHERE symbol = ? AND bot_id = ?
                """, (symbol, self.bot_id)).fetchone()[0]
                
                if existing_enhanced > 0:
                    self.logger.info(f"⏭️  {symbol} already migrated, skipping")
                    continue
                
                # Estimate target position size and configuration
                target_size, entry_phases, status = self.estimate_position_parameters(
                    symbol, shares, avg_cost, invested, first_date, last_date, conn
                )
                
                # Calculate allocation percentage
                allocation_pct = min(invested / target_size, 1.0) if target_size > 0 else 1.0
                
                # Determine max additions based on account value estimate
                max_additions = self.estimate_max_additions(invested)
                
                # Insert into enhanced positions
                conn.execute("""
                    INSERT INTO positions_enhanced (
                        symbol, target_position_size, current_allocation_pct, total_shares,
                        avg_cost, total_invested, entry_phases, addition_count, max_additions,
                        first_entry_date, last_addition_date, account_type, wyckoff_score,
                        position_status, bot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, target_size, allocation_pct, shares, avg_cost, invested,
                    json.dumps(entry_phases), len(entry_phases), max_additions,
                    first_date, last_date, account_type, 0.5, status, self.bot_id
                ))
                
                # Create position event for the migrated position
                conn.execute("""
                    INSERT INTO position_events (
                        symbol, event_type, event_date, wyckoff_phase, shares_traded, price,
                        allocation_before, allocation_after, reasoning, bot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, 'MIGRATION', first_date, 'UNKNOWN', shares, avg_cost,
                    0.0, allocation_pct, f"Migrated from legacy position system", self.bot_id
                ))
                
                migrated_count += 1
                self.logger.info(f"✅ Migrated {symbol}: ${invested:.2f} → {allocation_pct:.1%} of ${target_size:.2f} target")
        
        return migrated_count
    
    def estimate_position_parameters(self, symbol: str, shares: float, avg_cost: float,
                                   invested: float, first_date: str, last_date: str,
                                   conn) -> tuple:
        """Estimate target size, entry phases, and status for existing position"""
        
        # Get trade history for this symbol to estimate phases
        trades_cursor = conn.execute("""
            SELECT signal_phase, quantity, price, date FROM trades 
            WHERE symbol = ? AND bot_id = ? AND action = 'BUY'
            ORDER BY trade_datetime
        """, (symbol, self.bot_id))
        
        trades = trades_cursor.fetchall()
        
        # Estimate entry phases from trade history
        entry_phases = []
        if trades:
            for trade in trades:
                phase = trade[0]
                if phase and phase not in entry_phases:
                    entry_phases.append(phase)
        
        # If no phase info, estimate based on position characteristics
        if not entry_phases:
            if last_date != first_date:
                entry_phases = ['ST', 'SOS']  # Multiple entries suggest building
            else:
                entry_phases = ['SOS']  # Single entry
        
        # Estimate target position size
        # Assume current position is 75% of target if it was being built
        if len(entry_phases) > 1:
            target_size = invested / 0.75  # Assume 75% allocation
            status = 'BUILDING'
        else:
            target_size = invested  # Single entry = complete position
            status = 'COMPLETE'
        
        # Round target size to reasonable increment
        target_size = round(target_size, 2)
        
        return target_size, entry_phases, status
    
    def estimate_max_additions(self, position_value: float) -> int:
        """Estimate max additions based on position value"""
        if position_value < 50:
            return 3
        elif position_value < 150:
            return 4
        elif position_value < 300:
            return 5
        else:
            return 6
    
    def migrate_historical_trades(self) -> int:
        """Migrate sell trades to partial_sales where appropriate"""
        migrated_sales = 0
        
        with sqlite3.connect(self.db_path) as conn:
            # Find sell trades that might be partial sales
            cursor = conn.execute("""
                SELECT t.symbol, t.date, t.quantity, t.price, t.total_value, t.signal_phase
                FROM trades t
                WHERE t.action = 'SELL' AND t.bot_id = ?
                ORDER BY t.symbol, t.trade_datetime
            """, (self.bot_id,))
            
            sell_trades = cursor.fetchall()
            
            for trade in sell_trades:
                symbol, date, quantity, price, total_value, phase = trade
                
                # Check if this was likely a partial sale
                # (position still exists after the sale)
                remaining_position = conn.execute("""
                    SELECT total_shares FROM positions_enhanced 
                    WHERE symbol = ? AND bot_id = ?
                """, (symbol, self.bot_id)).fetchone()
                
                if remaining_position and remaining_position[0] > 0:
                    # This was likely a partial sale
                    remaining_shares = remaining_position[0]
                    
                    # Estimate sale reason based on phase or other factors
                    if phase == 'STOP_LOSS':
                        sale_reason = 'STOP_LOSS'
                    elif phase in ['PS', 'SC']:
                        sale_reason = 'WYCKOFF_DISTRIBUTION'
                    else:
                        sale_reason = 'PROFIT_TAKING'
                    
                    # Get position info for profit calculation
                    position_info = conn.execute("""
                        SELECT avg_cost FROM positions_enhanced 
                        WHERE symbol = ? AND bot_id = ?
                    """, (symbol, self.bot_id)).fetchone()
                    
                    if position_info:
                        avg_cost = position_info[0]
                        gain_pct = (price - avg_cost) / avg_cost if avg_cost > 0 else 0
                        profit_amount = (price - avg_cost) * quantity
                        
                        # Insert into partial_sales
                        conn.execute("""
                            INSERT INTO partial_sales (
                                symbol, sale_date, shares_sold, sale_price, sale_reason,
                                remaining_shares, gain_pct, profit_amount, bot_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol, date, quantity, price, sale_reason,
                            remaining_shares, gain_pct, profit_amount, self.bot_id
                        ))
                        
                        migrated_sales += 1
                        self.logger.info(f"✅ Migrated partial sale: {symbol} {quantity:.5f} shares at ${price:.2f}")
        
        return migrated_sales
    
    def validate_migration(self) -> Dict:
        """Validate the migration was successful"""
        with sqlite3.connect(self.db_path) as conn:
            # Count original positions
            original_positions = conn.execute("""
                SELECT COUNT(*) FROM positions 
                WHERE total_shares > 0 AND bot_id = ?
            """, (self.bot_id,)).fetchone()[0]
            
            # Count migrated positions
            migrated_positions = conn.execute("""
                SELECT COUNT(*) FROM positions_enhanced WHERE bot_id = ?
            """, (self.bot_id,)).fetchone()[0]
            
            # Check total invested amounts match
            original_invested = conn.execute("""
                SELECT SUM(total_invested) FROM positions 
                WHERE total_shares > 0 AND bot_id = ?
            """, (self.bot_id,)).fetchone()[0] or 0
            
            migrated_invested = conn.execute("""
                SELECT SUM(total_invested) FROM positions_enhanced WHERE bot_id = ?
            """, (self.bot_id,)).fetchone()[0] or 0
            
            # Check for any errors
            validation_errors = []
            
            if migrated_positions < original_positions:
                validation_errors.append(f"Missing positions: {original_positions - migrated_positions}")
            
            if abs(original_invested - migrated_invested) > 0.01:
                validation_errors.append(f"Investment mismatch: ${original_invested:.2f} vs ${migrated_invested:.2f}")
            
            return {
                'original_positions': original_positions,
                'migrated_positions': migrated_positions,
                'original_invested': original_invested,
                'migrated_invested': migrated_invested,
                'validation_errors': validation_errors,
                'migration_successful': len(validation_errors) == 0
            }
    
    def run_full_migration(self, create_backup=True) -> Dict:
        """Run the complete migration process"""
        self.logger.info("🚀 Starting Enhanced Position System Migration")
        
        # Step 1: Check what needs to be migrated
        status = self.check_migration_needed()
        self.logger.info(f"📊 Migration Status Check:")
        self.logger.info(f"   Enhanced tables exist: {status['enhanced_tables_exist']}")
        self.logger.info(f"   Existing positions: {status['existing_positions']}")
        self.logger.info(f"   Already migrated: {status['migrated_positions']}")
        self.logger.info(f"   Migration needed: {status['migration_needed']}")
        
        if not status['migration_needed']:
            self.logger.info("✅ No migration needed - system already up to date")
            return {'success': True, 'message': 'No migration needed'}
        
        # Step 2: Create backup
        backup_path = None
        if create_backup:
            backup_path = self.backup_database()
        
        try:
            # Step 3: Create enhanced schema
            self.create_enhanced_schema()
            
            # Step 4: Migrate positions
            migrated_positions = self.migrate_existing_positions()
            self.logger.info(f"✅ Migrated {migrated_positions} positions")
            
            # Step 5: Migrate historical trades
            migrated_sales = self.migrate_historical_trades()
            self.logger.info(f"✅ Migrated {migrated_sales} partial sales")
            
            # Step 6: Validate migration
            validation = self.validate_migration()
            
            if validation['migration_successful']:
                self.logger.info("🎉 Migration completed successfully!")
                self.logger.info(f"   Original positions: {validation['original_positions']}")
                self.logger.info(f"   Migrated positions: {validation['migrated_positions']}")
                self.logger.info(f"   Total invested: ${validation['migrated_invested']:.2f}")
                
                return {
                    'success': True,
                    'backup_path': backup_path,
                    'migrated_positions': migrated_positions,
                    'migrated_sales': migrated_sales,
                    'validation': validation
                }
            else:
                self.logger.error("❌ Migration validation failed!")
                for error in validation['validation_errors']:
                    self.logger.error(f"   - {error}")
                
                return {
                    'success': False,
                    'errors': validation['validation_errors'],
                    'backup_path': backup_path
                }
        
        except Exception as e:
            self.logger.error(f"❌ Migration failed: {e}")
            if backup_path:
                self.logger.info(f"💾 Database backup available at: {backup_path}")
            raise
    
    def generate_migration_report(self) -> str:
        """Generate a detailed migration report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"migration_report_{timestamp}.txt"
        
        with sqlite3.connect(self.db_path) as conn:
            # Get enhanced positions
            enhanced_positions = conn.execute("""
                SELECT symbol, target_position_size, current_allocation_pct, total_shares,
                       avg_cost, total_invested, entry_phases, position_status
                FROM positions_enhanced WHERE bot_id = ?
                ORDER BY total_invested DESC
            """, (self.bot_id,)).fetchall()
            
            # Get position events
            position_events = conn.execute("""
                SELECT symbol, event_type, event_date, shares_traded, reasoning
                FROM position_events WHERE bot_id = ? AND event_type = 'MIGRATION'
                ORDER BY event_date
            """, (self.bot_id,)).fetchall()
        
        report = f"""
=== ENHANCED POSITION SYSTEM MIGRATION REPORT ===

Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Bot ID: {self.bot_id}

=== MIGRATED POSITIONS ===
Total Positions: {len(enhanced_positions)}

"""
        
        for position in enhanced_positions:
            symbol, target_size, allocation_pct, shares, avg_cost, invested, phases_json, status = position
            phases = json.loads(phases_json) if phases_json else []
            phases_str = ', '.join(phases)
            
            report += f"""
{symbol}:
  - Current Investment: ${invested:.2f}
  - Target Position Size: ${target_size:.2f}
  - Allocation Progress: {allocation_pct:.1%}
  - Shares: {shares:.5f} @ ${avg_cost:.2f}
  - Entry Phases: {phases_str}
  - Status: {status}
"""
        
        report += f"""

=== MIGRATION EVENTS ===
Total Migration Events: {len(position_events)}

"""
        
        for event in position_events:
            symbol, event_type, date, shares, reasoning = event
            report += f"{date}: {symbol} - {reasoning}\n"
        
        report += f"""

=== NEXT STEPS ===
1. Verify all positions appear correctly in the enhanced system
2. Run the fractional trading bot to begin position building
3. Monitor with: python position_building_analytics.py --all
4. Configure scaling out thresholds as needed

=== CONFIGURATION RECOMMENDATIONS ===
Based on your migrated positions, consider:
- Account sizing appropriate for your portfolio value
- Wyckoff phase allocations that match your strategy
- Scaling out targets that align with your profit goals

Report saved to: {report_file}
"""
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report_file


def main():
    """Main migration script execution"""
    parser = argparse.ArgumentParser(description='Migrate to Enhanced Fractional Position System')
    parser.add_argument('--db', default='data/trading_bot.db', help='Database path')
    parser.add_argument('--bot-id', default='wyckoff_bot_v1', help='Bot ID to migrate')
    parser.add_argument('--no-backup', action='store_true', help='Skip database backup')
    parser.add_argument('--check-only', action='store_true', help='Only check migration status')
    parser.add_argument('--report', action='store_true', help='Generate migration report')
    
    args = parser.parse_args()
    
    try:
        migrator = PositionMigrationManager(args.db, args.bot_id)
        
        if args.check_only:
            # Just check status
            status = migrator.check_migration_needed()
            print(f"\n📊 MIGRATION STATUS CHECK")
            print(f"Enhanced tables exist: {status['enhanced_tables_exist']}")
            print(f"Existing positions: {status['existing_positions']}")
            print(f"Already migrated: {status['migrated_positions']}")
            print(f"Migration needed: {status['migration_needed']}")
            return 0
        
        # Run full migration
        result = migrator.run_full_migration(create_backup=not args.no_backup)
        
        if result['success']:
            print(f"\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
            print(f"Migrated {result.get('migrated_positions', 0)} positions")
            print(f"Migrated {result.get('migrated_sales', 0)} partial sales")
            
            if result.get('backup_path'):
                print(f"Database backup: {result['backup_path']}")
        else:
            print(f"\n❌ MIGRATION FAILED!")
            for error in result.get('errors', []):
                print(f"  - {error}")
            return 1
        
        # Generate report if requested
        if args.report:
            report_file = migrator.generate_migration_report()
            print(f"Migration report saved: {report_file}")
        
        print(f"\n🚀 Ready to run enhanced fractional trading bot!")
        print(f"Next steps:")
        print(f"  1. python fractional_position_system.py")
        print(f"  2. python position_building_analytics.py --all")
        
        return 0
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())