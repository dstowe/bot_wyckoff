#!/usr/bin/env python3
"""
Enhanced Automated Multi-Account Trading System - FIXED MAIN.PY
Handles authentication, account discovery, and comprehensive logging
"""

import logging
import sys
import traceback
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

# Import our modules
from webull.webull import webull
from auth.credentials import CredentialManager
from auth.login_manager import LoginManager
from auth.session_manager import SessionManager
from accounts.account_manager import AccountManager
from config.config import PersonalTradingConfig

class MainSystem:
    """
    Enhanced Automated Multi-Account Trading System - COMPLETE IMPLEMENTATION
    Handles authentication, account discovery, and logging to /logs directory
    """
    
    def __init__(self):
        # Initialize all attributes first
        self.logger = None
        self.wb = None
        self.config = None
        self.credential_manager = None
        self.login_manager = None
        self.session_manager = None
        self.account_manager = None
        self.is_logged_in = False
        
        # Set up logging first
        self.setup_logging()
        
        # Initialize the system
        self._initialize_system()
    
    def setup_logging(self):
        """Set up comprehensive logging to /logs directory"""
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Create timestamped log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = logs_dir / f"trading_system_{timestamp}.log"
            
            # Configure root logger
            logging.basicConfig(
                level=logging.INFO,  # Changed from DEBUG to INFO for cleaner output
                format='%(asctime)s - %(levelname)s - %(message)s',  # Simplified format
                datefmt='%H:%M:%S',  # Shorter time format
                handlers=[
                    logging.FileHandler(log_filename, encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)  # Also log to console
                ],
                force=True  # Force reconfiguration
            )
            
            # Create logger for this module
            self.logger = logging.getLogger(__name__)
            self.logger.info("🚀 ENHANCED MULTI-ACCOUNT TRADING SYSTEM")
            self.logger.info(f"📝 Log: {log_filename.name}")
            print()  # Add spacing for readability
            
        except Exception as e:
            print(f"❌ CRITICAL: Failed to setup logging: {e}")
            print(traceback.format_exc())
            sys.exit(1)
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("🔧 Initializing system components...")
            
            # Initialize all components
            self.config = PersonalTradingConfig()
            
            # Initialize webull with data folder support
            self.wb = webull()
            # Webull now automatically uses data folder for DID storage
            
            self.credential_manager = CredentialManager(logger=self.logger)
            self.login_manager = LoginManager(self.wb, self.credential_manager, logger=self.logger)
            self.session_manager = SessionManager(logger=self.logger)
            self.account_manager = AccountManager(self.wb, self.config, logger=self.logger)
            
            self.logger.info("✅ All components initialized")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ CRITICAL: Failed to initialize system: {e}")
            else:
                print(f"❌ CRITICAL: Failed to initialize system: {e}")
            raise
        
    def authenticate(self) -> bool:
        """Handle authentication using the modular authentication system"""
        try:
            self.logger.info("🔐 Authenticating...")
            
            # Check if credentials exist
            if not self.credential_manager.credentials_exist():
                self.logger.error("❌ No encrypted credentials found!")
                self.logger.info("💡 Run: python -c \"from auth.credentials import setup_credentials_interactive; setup_credentials_interactive()\"")
                return False
            
            # Load credentials and set DID automatically
            credentials = self.credential_manager.load_credentials()
            stored_did = credentials.get('did')
            
            if stored_did:
                # Set DID using the webull instance's method (now uses data folder)
                self.wb._set_did(stored_did, data_folder='data')
                # Also update the webull instance's _did attribute
                self.wb._did = stored_did
                self.logger.info("✅ DID set from stored credentials")
            else:
                self.logger.warning("⚠️ No DID stored - may cause image verification errors")
                self.logger.info("💡 If authentication fails, run: python tests/check_did.py")
            
            # Try existing session first
            self.logger.debug("Attempting to load existing session...")
            session_loaded = self.session_manager.auto_manage_session(self.wb)
            
            if session_loaded:
                self.logger.debug("Session loaded, verifying with API...")
                if self.login_manager.check_login_status():
                    self.logger.info("✅ Using existing session")
                    self.is_logged_in = True
                    return True
                else:
                    # Session loaded but API verification failed
                    self.logger.debug("Session loaded but API verification failed")
                    self.logger.info("🔄 Session expired on server, logging in fresh...")
                    self.session_manager.clear_session()
            else:
                # No session or couldn't load
                self.logger.debug("No session loaded")
                self.logger.info("🔑 No valid session, logging in fresh...")
            
            # Perform fresh login
            if self.login_manager.login_automatically():
                self.logger.info("✅ Authentication successful")
                self.is_logged_in = True
                self.session_manager.save_session(self.wb)
                return True
            else:
                self.logger.error("❌ Authentication failed")
                if not stored_did:
                    self.logger.error("💡 Add browser DID: python tests/check_did.py")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Authentication error: {e}")
            return False

    
    def discover_accounts(self) -> bool:
        """Discover and load all available trading accounts"""
        try:
            self.logger.info("🔍 Discovering accounts...")
            
            if not self.is_logged_in:
                self.logger.error("❌ Not authenticated")
                return False
            
            if self.account_manager.discover_accounts():
                summary = self.account_manager.get_account_summary()
                
                print()  # Add spacing
                self.logger.info("📊 ACCOUNT SUMMARY")
                self.logger.info(f"   Total Accounts: {summary['total_accounts']}")
                self.logger.info(f"   Total Value: ${summary['total_value']:,.2f}")
                self.logger.info(f"   Available Cash: ${summary['total_cash']:,.2f}")
                print()
                
                # Show each account
                for i, account in enumerate(summary['accounts'], 1):
                    status = "✅ ENABLED" if account['enabled'] else "❌ DISABLED"
                    self.logger.info(f"Account {i}: {account['account_type']} - {status}")
                    self.logger.info(f"   Balance: ${account['net_liquidation']:,.2f}")
                    self.logger.info(f"   Available: ${account['settled_funds']:,.2f}")
                    self.logger.info(f"   Positions: {account['positions_count']}")
                    if account['enabled']:
                        self.logger.info(f"   Day Trading: {'✅' if account['day_trading_enabled'] else '❌'} | Options: {'✅' if account['options_enabled'] else '❌'}")
                    print()
                
                return True
            else:
                self.logger.error("❌ Account discovery failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Account discovery error: {e}")
            return False
    
    def log_system_status(self):
        """Log essential system status"""
        try:
            login_info = self.login_manager.get_login_info()
            self.logger.info("📊 SYSTEM STATUS")
            self.logger.info(f"   Authentication: {'✅' if login_info['is_logged_in'] else '❌'}")
            self.logger.info(f"   Accounts Loaded: {len(self.account_manager.accounts) if self.account_manager.accounts else 0}")
            enabled_count = len(self.account_manager.get_enabled_accounts()) if self.account_manager else 0
            self.logger.info(f"   Enabled for Trading: {enabled_count}")
            print()
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system status: {e}")
    
    def run(self) -> bool:
        """Run the complete system workflow"""
        try:
            # Step 1: Authentication
            if not self.authenticate():
                self.logger.error("❌ Authentication failed")
                return False
            
            # Step 2: Account Discovery
            if not self.discover_accounts():
                self.logger.error("❌ Account discovery failed")
                return False
            
            # Step 3: System Status
            self.log_system_status()
            
            # Success
            self.logger.info("🎉 SYSTEM READY FOR TRADING")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System workflow failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            # Restore original account context if needed
            if self.account_manager:
                self.account_manager.restore_original_account()
            
            # # Logout if needed
            # if self.login_manager and self.is_logged_in:
            #     self.login_manager.logout()
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Cleanup warning: {e}")
            else:
                print(f"⚠️ Cleanup warning: {e}")


def main():
    """Main entry point for enhanced automated system"""
    system = None
    success = False
    
    try:
        # Initialize and run the system
        system = MainSystem()
        success = system.run()
        
        # Exit with appropriate code
        if success:
            print("✅ System completed successfully!")
            sys.exit(0)
        else:
            print("❌ System failed! Check logs for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        if system and system.logger:
            system.logger.info("🛑 Interrupted by user")
        else:
            print("🛑 Interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        if system and system.logger:
            system.logger.error(f"❌ Unexpected error: {e}")
        else:
            print(f"❌ Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        # Always attempt cleanup
        if system:
            system.cleanup()


if __name__ == "__main__":
    main()