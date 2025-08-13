#!/usr/bin/env python3
"""
Fix positions_enhanced Table for Multi-Account Support
This table also needs the composite primary key fix
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import shutil

def fix_positions_enhanced_table(db_path="data/trading_bot.db"):
    """
    Fix positions_enhanced table to support multiple accounts per symbol
    """
    print("🔄 Fixing positions_enhanced table for multi-account support...")
    
    db_path = Path(db_path)
    if not db_path.exists():
        print("❌ Database file not found.")
        return
    
    # Backup existing database
    backup_path = db_path.with_suffix(f'.backup_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
    print(f"📋 Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if positions_enhanced table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions_enhanced'")
        if not cursor.fetchone():
            print("📊 positions_enhanced table not found. Creating with correct schema...")
            create_positions_enhanced_table(conn)
            return
        
        # Check current schema
        cursor.execute("PRAGMA table_info(positions_enhanced)")
        columns = cursor.fetchall()
        
        print("📊 Current positions_enhanced table schema:")
        for col in columns:
            print(f"   {col[1]} {col[2]} {'PRIMARY KEY' if col[5] else ''}")
        
        # Check if we already have the correct schema
        column_names = [col[1] for col in columns]
        if 'account_type' in column_names:
            # Check primary key
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='positions_enhanced'")
            table_sql = cursor.fetchone()[0]
            
            if 'PRIMARY KEY (symbol, account_type, bot_id)' in table_sql:
                print("✅ positions_enhanced table already has correct schema!")
                return
        
        # Get existing data
        cursor.execute("SELECT * FROM positions_enhanced")
        existing_data = cursor.fetchall()
        
        # Get column info for mapping
        cursor.execute("PRAGMA table_info(positions_enhanced)")
        old_columns = [col[1] for col in cursor.fetchall()]
        
        print(f"📊 Found {len(existing_data)} existing enhanced positions to migrate")
        
        # Drop old table and create new one
        cursor.execute("DROP TABLE positions_enhanced")
        
        # Create new table with correct schema
        create_positions_enhanced_table(conn)
        
        # Migrate existing data
        for row in existing_data:
            # Map old columns to new schema
            data_dict = dict(zip(old_columns, row))
            
            # Set defaults for missing columns
            account_type = data_dict.get('account_type', 'Cash Account')  # Default
            entry_phase = data_dict.get('entry_phase', 'UNKNOWN')
            entry_strength = data_dict.get('entry_strength', 0.0)
            bot_id = data_dict.get('bot_id', 'enhanced_wyckoff_bot_v2')
            
            # Insert with proper account_type handling
            cursor.execute('''
                INSERT INTO positions_enhanced (
                    symbol, account_type, total_shares, avg_cost, total_invested,
                    first_purchase_date, last_purchase_date, entry_phase, 
                    entry_strength, position_size_pct, time_held_days,
                    volatility_percentile, bot_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_dict['symbol'],
                account_type,
                data_dict.get('total_shares', 0),
                data_dict.get('avg_cost', 0),
                data_dict.get('total_invested', 0),
                data_dict.get('first_purchase_date', datetime.now().strftime('%Y-%m-%d')),
                data_dict.get('last_purchase_date', datetime.now().strftime('%Y-%m-%d')),
                entry_phase,
                entry_strength,
                data_dict.get('position_size_pct', 0.1),
                data_dict.get('time_held_days', 0),
                data_dict.get('volatility_percentile', 0.5),
                bot_id,
                datetime.now().isoformat()
            ))
        
        print(f"✅ Migrated {len(existing_data)} enhanced positions to new schema")

def create_positions_enhanced_table(conn):
    """Create positions_enhanced table with correct multi-account schema"""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS positions_enhanced (
            symbol TEXT NOT NULL,
            account_type TEXT NOT NULL, 
            total_shares REAL NOT NULL,
            avg_cost REAL NOT NULL,
            total_invested REAL NOT NULL,
            first_purchase_date TEXT NOT NULL,
            last_purchase_date TEXT NOT NULL,
            entry_phase TEXT DEFAULT 'UNKNOWN',
            entry_strength REAL DEFAULT 0.0,
            position_size_pct REAL DEFAULT 0.1,
            time_held_days INTEGER DEFAULT 0,
            volatility_percentile REAL DEFAULT 0.5,
            bot_id TEXT DEFAULT 'enhanced_wyckoff_bot_v2',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, account_type, bot_id)
        )
    ''')
    print("✅ Created positions_enhanced table with correct multi-account schema")

def verify_both_tables(db_path="data/trading_bot.db"):
    """Verify both positions tables are correct"""
    print("\n🔍 Verifying both positions tables...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check positions table
        cursor.execute("SELECT symbol, account_type, COUNT(*) FROM positions GROUP BY symbol, account_type")
        positions_results = cursor.fetchall()
        
        print(f"\n📊 Positions table (should show multi-account):")
        for symbol, account_type, count in positions_results:
            print(f"   {symbol} ({account_type}): {count} record(s)")
        
        # Check positions_enhanced table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions_enhanced'")
        if cursor.fetchone():
            cursor.execute("SELECT symbol, account_type, COUNT(*) FROM positions_enhanced GROUP BY symbol, account_type")
            enhanced_results = cursor.fetchall()
            
            print(f"\n📊 Positions_enhanced table:")
            for symbol, account_type, count in enhanced_results:
                print(f"   {symbol} ({account_type}): {count} record(s)")
        else:
            print(f"\n📊 Positions_enhanced table: Not found")

def test_multi_account_insert_enhanced(db_path="data/trading_bot.db"):
    """Test inserting same symbol in different accounts for positions_enhanced"""
    print("\n🧪 Testing multi-account insert for positions_enhanced...")
    
    with sqlite3.connect(db_path) as conn:
        try:
            # Insert MSFT in Cash account
            conn.execute('''
                INSERT OR REPLACE INTO positions_enhanced (
                    symbol, account_type, total_shares, avg_cost, total_invested,
                    first_purchase_date, last_purchase_date, entry_phase, 
                    entry_strength, position_size_pct, time_held_days,
                    volatility_percentile, bot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'MSFT', 'Cash Account', 10.0, 300.0, 3000.0,
                '2025-01-01', '2025-01-01', 'SOS', 0.75,
                0.15, 5, 0.6, 'enhanced_wyckoff_bot_v2'
            ))
            
            # Insert MSFT in Margin account
            conn.execute('''
                INSERT OR REPLACE INTO positions_enhanced (
                    symbol, account_type, total_shares, avg_cost, total_invested,
                    first_purchase_date, last_purchase_date, entry_phase, 
                    entry_strength, position_size_pct, time_held_days,
                    volatility_percentile, bot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'MSFT', 'Margin Account', 5.0, 310.0, 1550.0,
                '2025-01-02', '2025-01-02', 'LPS', 0.60,
                0.12, 3, 0.7, 'enhanced_wyckoff_bot_v2'
            ))
            
            print("✅ Successfully inserted MSFT in both accounts for positions_enhanced!")
            
            # Verify
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, account_type, total_shares, avg_cost FROM positions_enhanced WHERE symbol = 'MSFT'")
            results = cursor.fetchall()
            
            for symbol, account_type, shares, cost in results:
                print(f"   {symbol} ({account_type}): {shares} shares @ ${cost}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🗄️ POSITIONS_ENHANCED TABLE MULTI-ACCOUNT FIX")
    print("="*50)
    
    # Fix positions_enhanced table
    fix_positions_enhanced_table()
    
    # Verify both tables
    verify_both_tables()
    
    # Test functionality
    test_multi_account_insert_enhanced()
    
    print("\n✅ positions_enhanced table fix completed!")