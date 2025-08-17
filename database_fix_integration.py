#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Fix Integration Script
Automatically updates your trading bot with the fixed database manager
Windows compatible with UTF-8 encoding support
"""

import os
import shutil
import re
from datetime import datetime
from pathlib import Path

class DatabaseFixIntegrator:
    """Integrates the fixed database manager into your existing codebase"""
    
    def __init__(self):
        self.backup_folder = Path("backups") / f"database_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.target_files = [
            "strategies/wyckoff/wyckoff.py"
        ]
        
    def create_backup(self):
        """Create backup of original files"""
        print("📁 Creating backup of original files...")
        self.backup_folder.mkdir(parents=True, exist_ok=True)
        
        for file_path in self.target_files:
            if Path(file_path).exists():
                backup_path = self.backup_folder / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                print(f"   ✅ Backed up: {file_path}")
            else:
                print(f"   ⚠️ File not found: {file_path}")
        
        print(f"📁 Backup created in: {self.backup_folder}")
    
    def update_wyckoff_file(self):
        """Update the main wyckoff.py file with fixed database manager"""
        file_path = Path("strategies/wyckoff/wyckoff.py")
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        print(f"🔧 Updating {file_path}...")
        
        # Read the original file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define the new DatabaseManager class
        new_database_manager = '''class DatabaseManager:
    """Enhanced Database Manager with better yfinance integration and debugging"""
    
    def __init__(self, db_name: str = "data/stock_data.db"):
        # Ensure data directory exists
        self.db_path = Path(db_name)
        self.db_path.parent.mkdir(exist_ok=True)
        
        self.db_name = str(self.db_path)
        self.logger = logging.getLogger(__name__)
        
        # Create connection
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.create_table()
        
        # Add debug mode for troubleshooting
        self.debug = True

    def create_table(self):
        """Create the stock_data table with proper schema"""
        query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            symbol TEXT NOT NULL,
            date TIMESTAMP NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
        """
        self.conn.cursor().execute(query)
        self.conn.commit()
        
        # Add index for faster queries
        self.conn.cursor().execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_data(symbol, date DESC);
        """)
        self.conn.commit()

    def is_data_stale(self, symbol: str, hours_threshold: int = 6) -> bool:
        """
        Check if data is stale - now checks by hours instead of days
        During market hours, data older than 6 hours is considered stale
        """
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT MAX(date), MAX(last_updated) 
                FROM stock_data 
                WHERE symbol = ?
            """, (symbol,))
            
            result = cur.fetchone()
            
            if not result[0]:  # No data at all
                if self.debug:
                    print(f"🔍 No data found for {symbol} - considering stale")
                return True
            
            # Check both the data date and when it was last updated
            import pandas as pd
            last_data_date = pd.to_datetime(result[0])
            last_updated = pd.to_datetime(result[1]) if result[1] else last_data_date
            
            # Remove timezone for comparison
            if last_data_date.tz is not None:
                last_data_date = last_data_date.tz_localize(None)
            if last_updated.tz is not None:
                last_updated = last_updated.tz_localize(None)
                
            now = datetime.now()
            
            # Check if last update was more than threshold hours ago
            hours_since_update = (now - last_updated).total_seconds() / 3600
            
            # More aggressive staleness detection
            is_stale = hours_since_update > hours_threshold
            
            if self.debug and hours_since_update > 1:
                print(f"🔍 {symbol}: Hours since update: {hours_since_update:.1f}, Stale: {is_stale}")
            
            return is_stale
            
        except Exception as e:
            print(f"❌ Error checking staleness for {symbol}: {e}")
            return True  # Assume stale if we can't check

    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Enhanced save_data method with better error handling and debugging
        """
        if data.empty:
            print(f"⚠️ No data to save for {symbol}")
            return False

        try:
            if self.debug:
                print(f"💾 Saving data for {symbol} - {len(data)} rows")

            # Create a clean DataFrame for saving
            data_to_save = pd.DataFrame(index=data.index)
            
            # Handle different DataFrame structures from yfinance
            # Map yfinance columns to database columns
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            
            # Handle both regular and MultiIndex columns
            for yf_col, db_col in column_mapping.items():
                if isinstance(data.columns, pd.MultiIndex):
                    # Handle MultiIndex columns (when downloading multiple symbols)
                    if yf_col in data.columns.get_level_values(0):
                        # Find the column that matches
                        for col in data.columns:
                            if col[0] == yf_col:
                                data_to_save[db_col] = data[col]
                                break
                else:
                    # Handle regular columns (single symbol download)
                    if yf_col in data.columns:
                        data_to_save[db_col] = data[yf_col]
            
            # Add symbol and timestamp
            data_to_save['symbol'] = symbol
            data_to_save['last_updated'] = datetime.now()
            
            # Clean the data
            data_to_save = data_to_save.dropna()
            
            if data_to_save.empty:
                print(f"⚠️ No valid data after cleaning for {symbol}")
                return False

            # Ensure index is timezone-naive for SQLite
            if data_to_save.index.tz is not None:
                data_to_save.index = data_to_save.index.tz_localize(None)

            # Save to database with proper error handling
            with self.conn:
                # Get date range for deletion
                min_date = data_to_save.index.min()
                max_date = data_to_save.index.max()
                
                # Delete existing data in this date range
                cursor = self.conn.cursor()
                cursor.execute("""
                    DELETE FROM stock_data 
                    WHERE symbol = ? AND date >= ? AND date <= ?
                """, (symbol, min_date.isoformat(), max_date.isoformat()))
                
                # Insert new data
                data_to_save.to_sql('stock_data', self.conn, if_exists='append', index=True)
                
                if self.debug:
                    print(f"✅ Successfully saved {len(data_to_save)} rows for {symbol}")
                
            return True
            
        except Exception as e:
            print(f"❌ Error saving data for {symbol}: {e}")
            return False

    def get_data(self, symbol: str, period_days: int = 365) -> pd.DataFrame | None:
        """Enhanced get_data method with better debugging"""
        try:
            start_date = datetime.now() - timedelta(days=period_days)
            
            df = pd.read_sql_query("""
                SELECT date, open, high, low, close, volume 
                FROM stock_data 
                WHERE symbol = ? AND date >= ?
                ORDER BY date
            """, self.conn, params=(symbol, start_date), 
               index_col='date', parse_dates=['date'])
            
            if df.empty:
                if self.debug:
                    print(f"🔍 No database data found for {symbol}")
                return None
            
            # Ensure timezone-naive datetime index
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            if self.debug and len(df) > 0:
                print(f"📊 Retrieved {len(df)} rows for {symbol} (latest: {df.index.max().date()})")
            
            return df
            
        except Exception as e:
            print(f"❌ Error reading from DB for {symbol}: {e}")
            return None'''

        # Find and replace the old DatabaseManager class
        # Pattern to match the entire class definition
        class_pattern = r'class DatabaseManager:.*?(?=\nclass|\n\n\nclass|\Z)'
        
        if re.search(class_pattern, content, re.DOTALL):
            # Replace the old class with the new one
            updated_content = re.sub(class_pattern, new_database_manager, content, flags=re.DOTALL)
            print("✅ Replaced existing DatabaseManager class")
        else:
            print("⚠️ Could not find existing DatabaseManager class pattern")
            return False
        
        # Update the update_database method in WyckoffPnFStrategy
        updated_update_method = '''    def update_database(self):
        """Enhanced database update with better progress tracking"""
        print("🔍 Checking database for stale data...")
        
        # Check all symbols for staleness (now using 6-hour threshold)
        stale_symbols = []
        fresh_symbols = []
        
        for symbol in self.symbols:
            if self.db_manager.is_data_stale(symbol):
                stale_symbols.append(symbol)
            else:
                fresh_symbols.append(symbol)

        if not stale_symbols:
            print(f"✅ Database is up-to-date ({len(fresh_symbols)} symbols current)")
            return

        print(f"📥 Downloading fresh data for {len(stale_symbols)} symbols...")
        print(f"📊 {len(fresh_symbols)} symbols already current")
        
        # Download symbols individually with progress tracking
        success_count = 0
        for i, symbol in enumerate(stale_symbols, 1):
            try:
                print(f"📥 Downloading {symbol} ({i}/{len(stale_symbols)})...", end='')
                
                # Use Ticker object for cleaner single-symbol data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='1y', auto_adjust=True, repair=True)
                
                if not df.empty:
                    if self.db_manager.save_data(symbol, df):
                        print(f" ✅ Success ({len(df)} rows)")
                        success_count += 1
                    else:
                        print(f" ❌ Save failed")
                else:
                    print(f" ⚠️ No data returned")
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
        
        print(f"\\n📊 Database update complete: {success_count}/{len(stale_symbols)} successful")'''

        # Replace the update_database method
        method_pattern = r'    def update_database\(self\):.*?(?=\n    def|\n\nclass|\Z)'
        updated_content = re.sub(method_pattern, updated_update_method, updated_content, flags=re.DOTALL)
        
        # Add required imports at the top if not present
        import_additions = []
        
        if 'from pathlib import Path' not in updated_content:
            import_additions.append('from pathlib import Path')
        
        if import_additions:
            # Find the import section and add our imports
            import_pattern = r'(import warnings.*?\n)'
            if re.search(import_pattern, updated_content, re.DOTALL):
                additional_imports = '\n'.join(import_additions) + '\n'
                updated_content = re.sub(import_pattern, r'\1' + additional_imports, updated_content)
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Successfully updated {file_path}")
        return True
    
    def create_database_test_script(self):
        """Create a test script to verify the database fix works"""
        test_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Fix Test Script
Run this to verify the database updates are working correctly
"""

import sys
import os
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_database_updates():
    """Test the fixed database system"""
    print("🧪 Testing Database Update System")
    print("=" * 50)
    
    try:
        # Import the updated strategy
        from strategies.wyckoff.wyckoff import WyckoffPnFStrategy
        
        print("✅ Successfully imported WyckoffPnFStrategy")
        
        # Create strategy instance
        strategy = WyckoffPnFStrategy()
        print("✅ Successfully created strategy instance")
        
        # Test with a small subset of symbols for faster testing
        original_symbols = strategy.symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        strategy.symbols = test_symbols
        
        print(f"🔍 Testing with {len(test_symbols)} symbols: {test_symbols}")
        
        # Force database update
        print("\\n📥 Starting database update test...")
        strategy.update_database()
        
        # Test data retrieval
        print("\\n📊 Testing data retrieval...")
        for symbol in test_symbols:
            data = strategy.db_manager.get_data(symbol, period_days=30)
            if data is not None and len(data) > 0:
                print(f"   ✅ {symbol}: {len(data)} rows, latest: {data.index.max().date()}")
            else:
                print(f"   ❌ {symbol}: No data found")
        
        # Restore original symbols
        strategy.symbols = original_symbols
        
        print("\\n🎉 Database test completed successfully!")
        print("\\n💡 Tips:")
        print("   - Data is now updated more frequently (every 6 hours vs 3 days)")
        print("   - Better error handling and progress tracking")
        print("   - Debug information shows what's happening")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_database_status():
    """Check the current database status"""
    try:
        from strategies.wyckoff.wyckoff import WyckoffPnFStrategy
        
        strategy = WyckoffPnFStrategy()
        
        print("\\n📊 Database Status Check")
        print("-" * 30)
        
        # Check if database file exists
        db_path = Path("data/stock_data.db")
        if db_path.exists():
            print(f"✅ Database file exists: {db_path}")
            print(f"   Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            print(f"❌ Database file not found: {db_path}")
            return
        
        # Check a few symbols for staleness
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        print(f"\\n🔍 Staleness check for {test_symbols}:")
        
        for symbol in test_symbols:
            is_stale = strategy.db_manager.is_data_stale(symbol)
            status = "🔴 STALE" if is_stale else "🟢 FRESH"
            print(f"   {symbol}: {status}")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")

if __name__ == "__main__":
    print("🚀 Database Fix Verification")
    print("=" * 40)
    
    # Run the main test
    if test_database_updates():
        print("\\n" + "=" * 40)
        check_database_status()
    else:
        print("\\n❌ Tests failed - check the error messages above")
        sys.exit(1)
'''
        
        test_file_path = Path("test_database_fix.py")
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        print(f"📄 Created test script: {test_file_path}")
    
    def run_integration(self):
        """Run the complete integration process"""
        print("🚀 Database Fix Integration Starting...")
        print("=" * 50)
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Update the main file
            if not self.update_wyckoff_file():
                print("❌ Failed to update wyckoff.py")
                return False
            
            # Step 3: Create test script
            self.create_database_test_script()
            
            print("\n🎉 Integration Complete!")
            print("=" * 30)
            print("✅ Updated: strategies/wyckoff/wyckoff.py")
            print("✅ Created: test_database_fix.py")
            print(f"✅ Backup: {self.backup_folder}")
            
            print("\n📋 Next Steps:")
            print("1. Run: python test_database_fix.py")
            print("2. Check that data updates are working")
            print("3. Your database will now update every 6 hours instead of 3 days")
            print("4. Much better error handling and progress tracking")
            
            print("\n🔧 What was fixed:")
            print("• Database staleness check now uses hours (6h) instead of days (3d)")
            print("• Better column mapping from yfinance to database")
            print("• Enhanced error handling and debugging")
            print("• Progress tracking during updates")
            print("• Automatic cleanup of old data before inserting new data")
            print("• Support for both regular and MultiIndex DataFrames from yfinance")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    integrator = DatabaseFixIntegrator()
    success = integrator.run_integration()
    
    if success:
        print("\n🎯 Ready to test! Run: python test_database_fix.py")
    else:
        print("\n❌ Integration failed - check error messages above")