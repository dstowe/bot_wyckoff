#!/usr/bin/env python3
"""
Windows-Compatible Setup Script for Enhanced Fractional Position Building System
Fixed for Windows encoding issues and Unicode character support
"""

import os
import sys
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class WindowsSafeOutput:
    """Handle Windows console output safely without Unicode errors"""
    
    def __init__(self):
        # Detect if we can use Unicode
        self.can_unicode = self._test_unicode_support()
        
        # Define Unicode to ASCII fallbacks
        self.unicode_fallbacks = {
            '✅': '[OK]',
            '❌': '[ERR]',
            '🚀': '[START]',
            '📊': '[DATA]',
            '🎉': '[SUCCESS]',
            '⚠️': '[WARN]',
            '💡': '[TIP]',
            '🔧': '[SETUP]',
            '📝': '[LOG]',
            '💰': '[MONEY]',
            '🎯': '[TARGET]',
            '🏗️': '[BUILD]',
            '🔍': '[CHECK]',
            '📋': '[REPORT]',
            '👋': '[BYE]',
            '⛔': '[STOP]',
            '🔄': '[REFRESH]',
        }
    
    def _test_unicode_support(self):
        """Test if the current environment supports Unicode output"""
        try:
            # Try to encode a Unicode character
            test_char = '✅'
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
                test_char.encode(sys.stdout.encoding)
                return True
            else:
                # Try with default encoding
                test_char.encode('utf-8')
                return True
        except (UnicodeEncodeError, LookupError):
            return False
    
    def safe_print(self, text):
        """Print text safely, replacing Unicode chars if needed"""
        if self.can_unicode:
            try:
                print(text)
                return
            except UnicodeEncodeError:
                pass
        
        # Fallback: replace Unicode characters
        safe_text = text
        for unicode_char, ascii_replacement in self.unicode_fallbacks.items():
            safe_text = safe_text.replace(unicode_char, ascii_replacement)
        
        print(safe_text)
    
    def safe_write_file(self, filepath, content):
        """Write file safely with UTF-8 encoding"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except UnicodeEncodeError:
            # Fallback: replace Unicode characters and write
            safe_content = content
            for unicode_char, ascii_replacement in self.unicode_fallbacks.items():
                safe_content = safe_content.replace(unicode_char, ascii_replacement)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(safe_content)
            return True


class WindowsCompatibleSetup:
    """Windows-compatible setup manager for the fractional position building system"""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir or os.getcwd())
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        self.config_dir = self.base_dir / "config_examples"
        
        # Initialize safe output handler
        self.output = WindowsSafeOutput()
        
        # File mappings for the enhanced system
        self.required_files = {
            'fractional_position_system.py': 'Enhanced fractional position building bot',
            'position_building_analytics.py': 'Analytics dashboard for position tracking',
            'migration_script.py': 'Migration from existing system',
            'realtime_monitor.py': 'Real-time monitoring dashboard',
            'fractional_config_examples.py': 'Configuration examples and templates'
        }
        
        self.setup_log = []
        
    def log_step(self, message: str, success: bool = True):
        """Log a setup step with Windows-safe output"""
        status = "[OK]" if success else "[ERR]"
        if self.output.can_unicode:
            status = "✅" if success else "❌"
        
        log_entry = f"{status} {message}"
        self.output.safe_print(log_entry)
        self.setup_log.append(log_entry)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        self.log_step("Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log_step("Python 3.8+ required", False)
            return False
        
        # Check for required modules
        required_modules = [
            'sqlite3', 'pandas', 'numpy', 'matplotlib', 'seaborn'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.log_step(f"Missing modules: {', '.join(missing_modules)}", False)
            self.output.safe_print(f"[TIP] Install with: pip install {' '.join(missing_modules)}")
            return False
        
        # Check for existing webull module
        try:
            from webull.webull import webull
            self.log_step("Webull module found")
        except ImportError:
            self.log_step("Webull module not found - trading functions will be limited", False)
            self.output.safe_print("[TIP] Install webull module for full functionality")
        
        # Check for existing database
        db_path = self.data_dir / "trading_bot.db"
        if db_path.exists():
            self.log_step(f"Existing database found: {db_path}")
        else:
            self.log_step("No existing database - will create new one")
        
        return True
    
    def create_directory_structure(self) -> bool:
        """Create necessary directory structure"""
        self.log_step("Creating directory structure...")
        
        directories = [
            self.data_dir,
            self.logs_dir,
            self.reports_dir,
            self.config_dir,
            self.base_dir / "strategies" / "wyckoff",
            self.base_dir / "auth",
            self.base_dir / "accounts",
            self.base_dir / "config"
        ]
        
        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.log_step(f"Created directory: {directory.name}")
            
            return True
        except Exception as e:
            self.log_step(f"Error creating directories: {e}", False)
            return False
    
    def check_existing_system(self) -> Dict:
        """Check what parts of the system already exist"""
        existing_system = {
            'has_database': False,
            'has_positions': False,
            'has_trades': False,
            'has_enhanced_tables': False,
            'migration_needed': False
        }
        
        db_path = self.data_dir / "trading_bot.db"
        if db_path.exists():
            existing_system['has_database'] = True
            
            try:
                with sqlite3.connect(db_path) as conn:
                    # Check for existing positions
                    cursor = conn.execute("SELECT COUNT(*) FROM positions WHERE total_shares > 0")
                    position_count = cursor.fetchone()[0]
                    existing_system['has_positions'] = position_count > 0
                    
                    # Check for existing trades
                    cursor = conn.execute("SELECT COUNT(*) FROM trades")
                    trade_count = cursor.fetchone()[0]
                    existing_system['has_trades'] = trade_count > 0
                    
                    # Check for enhanced tables
                    cursor = conn.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name='positions_enhanced'
                    """)
                    existing_system['has_enhanced_tables'] = cursor.fetchone() is not None
                    
                    # Migration needed if we have positions but no enhanced tables
                    existing_system['migration_needed'] = (
                        existing_system['has_positions'] and 
                        not existing_system['has_enhanced_tables']
                    )
            
            except Exception as e:
                self.log_step(f"Error checking database: {e}", False)
        
        return existing_system
    
    def install_fractional_system_files(self) -> bool:
        """Install the fractional system files"""
        self.log_step("Installing fractional position building system files...")
        
        for filename, description in self.required_files.items():
            file_path = self.base_dir / filename
            if file_path.exists():
                self.log_step(f"Found: {filename} - {description}")
            else:
                self.log_step(f"Missing: {filename} - {description}", False)
                self.output.safe_print(f"[TIP] Please ensure {filename} is in the project directory")
        
        return True
    
    def setup_configuration_examples(self) -> bool:
        """Create configuration examples with Windows-safe file writing"""
        self.log_step("Setting up configuration examples...")
        
        try:
            # Try to import and generate configurations
            try:
                sys.path.append(str(self.base_dir))
                from fractional_config_examples import FractionalTradingConfigurations
                
                configs = FractionalTradingConfigurations()
                
                # Save example configurations with Windows-safe writing
                for config_name in ['conservative', 'balanced', 'aggressive', 'small_account']:
                    filename = self.config_dir / f"{config_name}_config.json"
                    
                    # Get the configuration
                    config = configs.configurations[config_name]
                    serializable_config = configs._make_serializable(config)
                    
                    # Write with UTF-8 encoding (Windows-safe)
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(serializable_config, f, indent=2, ensure_ascii=False)
                    
                    self.log_step(f"Created configuration: {config_name}")
                
                return True
                
            except ImportError:
                self.log_step("Configuration module not available - creating basic examples", False)
                
                # Create basic configuration examples
                basic_config = {
                    "name": "Basic Configuration",
                    "description": "Default configuration for small accounts",
                    "account_sizing": [
                        {"min_account": 0, "max_account": 500, "entry_range": [5, 15], "max_additions": 3},
                        {"min_account": 500, "max_account": 1000, "entry_range": [10, 25], "max_additions": 4}
                    ],
                    "signal_thresholds": {
                        "min_strength": 0.4,
                        "min_volume_confirmation": True
                    }
                }
                
                with open(self.config_dir / "basic_config.json", 'w', encoding='utf-8') as f:
                    json.dump(basic_config, f, indent=2, ensure_ascii=False)
                
                return True
                
        except Exception as e:
            self.log_step(f"Error setting up configurations: {e}", False)
            return False
    
    def run_migration_if_needed(self, existing_system: Dict) -> bool:
        """Run migration if needed"""
        if not existing_system['migration_needed']:
            self.log_step("No migration needed")
            return True
        
        self.log_step("Migration needed - starting migration process...")
        
        try:
            # Try to import and run migration
            sys.path.append(str(self.base_dir))
            from migration_script import PositionMigrationManager
            
            migrator = PositionMigrationManager(
                str(self.data_dir / "trading_bot.db"),
                "wyckoff_bot_v1"
            )
            
            # Run migration
            result = migrator.run_full_migration(create_backup=True)
            
            if result['success']:
                self.log_step(f"Migration completed - migrated {result.get('migrated_positions', 0)} positions")
                return True
            else:
                self.log_step("Migration failed", False)
                return False
                
        except ImportError:
            self.log_step("Migration module not available - manual migration required", False)
            return False
        except Exception as e:
            self.log_step(f"Migration error: {e}", False)
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create convenient startup scripts with Windows-safe encoding"""
        self.log_step("Creating startup scripts...")
        
        try:
            # Main trading bot script
            trading_script_content = '''#!/usr/bin/env python3
"""
Fractional Position Building Bot Launcher
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fractional_position_system import EnhancedFractionalTradingBot
    
    if __name__ == "__main__":
        bot = EnhancedFractionalTradingBot()
        bot.run()
except ImportError as e:
    print(f"Error importing fractional system: {e}")
    print("Make sure fractional_position_system.py is in the current directory")
    sys.exit(1)
'''
            
            trading_script = self.base_dir / "run_fractional_bot.py"
            self.output.safe_write_file(trading_script, trading_script_content)
            
            # Analytics script
            analytics_script_content = '''#!/usr/bin/env python3
"""
Position Building Analytics Launcher
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from position_building_analytics import main
    
    if __name__ == "__main__":
        sys.exit(main())
except ImportError as e:
    print(f"Error importing analytics: {e}")
    print("Make sure position_building_analytics.py is in the current directory")
    sys.exit(1)
'''
            
            analytics_script = self.base_dir / "run_analytics.py"
            self.output.safe_write_file(analytics_script, analytics_script_content)
            
            # Monitor script
            monitor_script_content = '''#!/usr/bin/env python3
"""
Real-Time Monitor Launcher
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from realtime_monitor import main
    
    if __name__ == "__main__":
        sys.exit(main())
except ImportError as e:
    print(f"Error importing monitor: {e}")
    print("Make sure realtime_monitor.py is in the current directory")
    sys.exit(1)
'''
            
            monitor_script = self.base_dir / "run_monitor.py"
            self.output.safe_write_file(monitor_script, monitor_script_content)
            
            # Windows batch files for easier execution
            bat_content_bot = f'''@echo off
cd /d "{self.base_dir}"
python run_fractional_bot.py
pause
'''
            
            bat_content_analytics = f'''@echo off
cd /d "{self.base_dir}"
python run_analytics.py --all
pause
'''
            
            bat_content_monitor = f'''@echo off
cd /d "{self.base_dir}"
python run_monitor.py
pause
'''
            
            with open(self.base_dir / "run_fractional_bot.bat", 'w') as f:
                f.write(bat_content_bot)
            
            with open(self.base_dir / "run_analytics.bat", 'w') as f:
                f.write(bat_content_analytics)
            
            with open(self.base_dir / "run_monitor.bat", 'w') as f:
                f.write(bat_content_monitor)
            
            self.log_step("Created startup scripts (.py and .bat files)")
            return True
            
        except Exception as e:
            self.log_step(f"Error creating startup scripts: {e}", False)
            return False
    
    def create_documentation(self) -> bool:
        """Create basic documentation with Windows-safe encoding"""
        self.log_step("Creating documentation...")
        
        try:
            readme_content = f"""
# Enhanced Fractional Position Building System

## Quick Start (Windows)

### 1. Run the Trading Bot
Double-click: run_fractional_bot.bat
Or command line: python run_fractional_bot.py

### 2. View Analytics
Double-click: run_analytics.bat
Or command line: python run_analytics.py --all

### 3. Monitor in Real-Time
Double-click: run_monitor.bat
Or command line: python run_monitor.py

## Configuration

Configuration examples are available in the `config_examples/` directory:
- `conservative_config.json` - Low risk, smaller positions
- `balanced_config.json` - Balanced approach (recommended)
- `aggressive_config.json` - Higher risk, larger positions
- `small_account_config.json` - Optimized for accounts under $1000

## Features

### Position Building
- Fractional share support (0.00001 precision)
- Wyckoff phase-based allocation:
  - ST: 25% initial position (testing phase)
  - SOS: 50% addition (breakout confirmation)
  - LPS: 25% completion (support test)
  - BU: Opportunistic additions on pullbacks

### Scaling Out
- Automatic profit-taking at multiple levels:
  - 10% gain: Sell 25%
  - 20% gain: Sell another 25%
  - 35% gain: Sell another 25%
  - Distribution signals: Exit remaining 25%

### Account Size Optimization
- Dynamic position sizing based on account value
- Preserves minimum cash balance that scales with growth
- Multiple account support (Cash, Margin, IRA)

## Directory Structure
```
{self.base_dir}/
├── data/                    # Database and session files
├── logs/                    # Trading logs
├── reports/                 # Analytics reports
├── config_examples/         # Configuration templates
├── fractional_position_system.py
├── position_building_analytics.py
├── realtime_monitor.py
├── migration_script.py
├── run_fractional_bot.py   # Python launcher
├── run_fractional_bot.bat  # Windows batch file
├── run_analytics.py        # Python launcher
├── run_analytics.bat       # Windows batch file
├── run_monitor.py          # Python launcher
└── run_monitor.bat         # Windows batch file
```

## Windows Notes

- Use .bat files for easy double-click execution
- All files use UTF-8 encoding for proper character support
- Console output automatically adapts to Windows terminal capabilities

## Database Tables

### Enhanced Tables
- `positions_enhanced`: Tracks position building progress
- `partial_sales`: Records scaling out actions
- `position_events`: Logs all position building events

## Setup Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            readme_path = self.base_dir / "README_FRACTIONAL.md"
            self.output.safe_write_file(readme_path, readme_content)
            
            self.log_step("Created documentation")
            return True
            
        except Exception as e:
            self.log_step(f"Error creating documentation: {e}", False)
            return False
    
    def verify_installation(self) -> bool:
        """Verify the installation is complete and working"""
        self.log_step("Verifying installation...")
        
        verification_passed = True
        
        # Check directories
        for directory in [self.data_dir, self.logs_dir, self.reports_dir]:
            if not directory.exists():
                self.log_step(f"Missing directory: {directory}", False)
                verification_passed = False
        
        # Check database can be accessed
        db_path = self.data_dir / "trading_bot.db"
        try:
            with sqlite3.connect(db_path) as conn:
                # Try to create enhanced tables if they don't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions_enhanced (
                        symbol TEXT PRIMARY KEY,
                        target_position_size REAL NOT NULL,
                        current_allocation_pct REAL NOT NULL DEFAULT 0.0,
                        total_shares REAL NOT NULL DEFAULT 0.0,
                        avg_cost REAL NOT NULL DEFAULT 0.0,
                        total_invested REAL NOT NULL DEFAULT 0.0,
                        entry_phases TEXT,
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
                """)
                self.log_step("Database accessible and enhanced schema available")
        except Exception as e:
            self.log_step(f"Database verification failed: {e}", False)
            verification_passed = False
        
        return verification_passed
    
    def generate_setup_report(self) -> str:
        """Generate a setup completion report with Windows-safe encoding"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.base_dir / f"setup_report_{timestamp}.txt"
        
        existing_system = self.check_existing_system()
        
        report_content = f"""
=== FRACTIONAL POSITION BUILDING SYSTEM SETUP REPORT ===

Setup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Base Directory: {self.base_dir}

=== SETUP LOG ===
"""
        
        for log_entry in self.setup_log:
            report_content += f"{log_entry}\n"
        
        report_content += f"""

=== SYSTEM STATUS ===
Database Exists: {existing_system['has_database']}
Existing Positions: {existing_system['has_positions']}
Existing Trades: {existing_system['has_trades']}
Enhanced Tables: {existing_system['has_enhanced_tables']}
Migration Needed: {existing_system['migration_needed']}

=== WINDOWS BATCH FILES CREATED ===
run_fractional_bot.bat - Double-click to run trading bot
run_analytics.bat - Double-click to generate analytics
run_monitor.bat - Double-click to start real-time monitor

=== NEXT STEPS ===
1. Verify your trading credentials are set up:
   python -c "from auth.credentials import setup_credentials_interactive; setup_credentials_interactive()"

2. Run the fractional trading bot:
   Double-click: run_fractional_bot.bat
   Or: python run_fractional_bot.py

3. Monitor position building:
   Double-click: run_monitor.bat
   Or: python run_monitor.py

4. Generate analytics:
   Double-click: run_analytics.bat
   Or: python run_analytics.py --all

=== CONFIGURATION ===
Configuration examples are available in: {self.config_dir}
- conservative_config.json
- balanced_config.json  
- aggressive_config.json
- small_account_config.json

=== WINDOWS NOTES ===
- All files use UTF-8 encoding for proper character support
- Console output adapts automatically to Windows terminal capabilities
- Use .bat files for easy double-click execution
- All startup scripts include proper path handling for Windows

=== SUPPORT ===
For issues or questions:
1. Check the logs in: {self.logs_dir}
2. Review the README_FRACTIONAL.md
3. Run verification: python setup_fractional_system.py --verify-only

Setup Report: {report_path}
"""
        
        self.output.safe_write_file(report_path, report_content)
        return str(report_path)
    
    def run_complete_setup(self, migrate: bool = True) -> bool:
        """Run the complete setup process with Windows-safe output"""
        header = "ENHANCED FRACTIONAL POSITION BUILDING SYSTEM SETUP"
        if self.output.can_unicode:
            self.output.safe_print("🚀 " + header)
        else:
            self.output.safe_print("[START] " + header)
        
        self.output.safe_print("=" * 60)
        self.output.safe_print("")
        
        # Step 1: Prerequisites
        if not self.check_prerequisites():
            self.log_step("Prerequisites check failed - aborting setup", False)
            return False
        
        # Step 2: Directory structure
        if not self.create_directory_structure():
            self.log_step("Directory creation failed - aborting setup", False)
            return False
        
        # Step 3: Check existing system
        existing_system = self.check_existing_system()
        
        # Step 4: Install files
        if not self.install_fractional_system_files():
            self.log_step("File installation check failed", False)
        
        # Step 5: Configuration examples
        if not self.setup_configuration_examples():
            self.log_step("Configuration setup failed", False)
        
        # Step 6: Migration if needed
        if migrate and existing_system['migration_needed']:
            if not self.run_migration_if_needed(existing_system):
                self.log_step("Migration failed", False)
        
        # Step 7: Startup scripts
        if not self.create_startup_scripts():
            self.log_step("Startup script creation failed", False)
        
        # Step 8: Documentation
        if not self.create_documentation():
            self.log_step("Documentation creation failed", False)
        
        # Step 9: Verification
        verification_passed = self.verify_installation()
        
        # Step 10: Generate report
        report_path = self.generate_setup_report()
        
        self.output.safe_print("")
        self.output.safe_print("=" * 60)
        
        if verification_passed:
            if self.output.can_unicode:
                self.log_step("🎉 SETUP COMPLETED SUCCESSFULLY!")
            else:
                self.log_step("[SUCCESS] SETUP COMPLETED SUCCESSFULLY!")
            
            self.output.safe_print("")
            self.output.safe_print("Next Steps:")
            self.output.safe_print("1. Double-click: run_fractional_bot.bat    # Run the enhanced trading bot")
            self.output.safe_print("2. Double-click: run_monitor.bat           # Monitor in real-time")
            self.output.safe_print("3. Double-click: run_analytics.bat         # Generate analytics")
            self.output.safe_print("")
            self.output.safe_print(f"[REPORT] Setup report saved: {report_path}")
        else:
            self.log_step("[WARN] SETUP COMPLETED WITH WARNINGS", False)
            self.output.safe_print("Please review the setup report and address any issues.")
        
        return verification_passed


def main():
    """Main setup script execution with Windows compatibility"""
    parser = argparse.ArgumentParser(description='Enhanced Fractional Position Building System Setup (Windows Compatible)')
    parser.add_argument('--base-dir', help='Base directory for installation (default: current directory)')
    parser.add_argument('--no-migrate', action='store_true', help='Skip automatic migration')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing installation')
    parser.add_argument('--config-only', action='store_true', help='Only setup configuration examples')
    
    args = parser.parse_args()
    
    try:
        setup_manager = WindowsCompatibleSetup(args.base_dir)
        
        if args.verify_only:
            # Just verify existing installation
            setup_manager.log_step("Running verification check...")
            existing_system = setup_manager.check_existing_system()
            verification_passed = setup_manager.verify_installation()
            
            print()
            print("=== VERIFICATION RESULTS ===")
            for key, value in existing_system.items():
                print(f"{key}: {value}")
            
            if verification_passed:
                print("[OK] Installation verification passed")
                return 0
            else:
                print("[ERR] Installation verification failed")
                return 1
        
        elif args.config_only:
            # Just setup configurations
            setup_manager.create_directory_structure()
            success = setup_manager.setup_configuration_examples()
            return 0 if success else 1
        
        else:
            # Full setup
            success = setup_manager.run_complete_setup(migrate=not args.no_migrate)
            return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n[STOP] Setup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n[ERR] Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())