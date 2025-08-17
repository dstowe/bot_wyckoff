# auth/login_manager.py
import time
import logging
import requests
from typing import Tuple, Dict
from .credentials import CredentialManager

class LoginManager:
    """Handles login operations with retry logic and error handling"""
    
    def __init__(self, wb, credential_manager: CredentialManager = None, logger=None):
        self.wb = wb
        self.credential_manager = credential_manager or CredentialManager()
        self.logger = logger or logging.getLogger(__name__)
        self.is_logged_in = False
        
        # Login retry settings
        self.max_login_attempts = 3
        self.base_login_delay = 30  # seconds
        self.max_trade_token_attempts = 3
        self.base_trade_token_delay = 10  # seconds
    
    def login_automatically(self) -> bool:
        """Automated login using stored credentials with retry logic"""
        for attempt in range(1, self.max_login_attempts + 1):
            try:
                self.logger.info(f"Starting automated login attempt {attempt}/{self.max_login_attempts}...")
                
                # Load credentials
                credentials = self.credential_manager.load_credentials()
                
                # Validate credentials
                if not self.credential_manager.validate_credentials(credentials):
                    self.logger.error("❌ Invalid credentials found")
                    return False
                
                # Set DID if provided
                if credentials.get('did'):
                    self.wb._did = credentials['did']
                    self.wb._set_did(credentials['did'])
                    self.logger.info("DID set from stored credentials")
                
                # Login to Webull
                self.logger.info("Attempting Webull login...")
                login_result = self.wb.login(
                    username=credentials['username'],
                    password=credentials['password']
                )
                
                # Check login result
                if 'accessToken' in login_result:
                    self.logger.info("✅ Webull login successful")
                    self.is_logged_in = True
                    
                    # Try to get trade token with retries
                    if self._get_trade_token_with_retry(credentials['trading_pin']):
                        self.logger.info("✅ Complete login process successful")
                        return True
                    else:
                        self.logger.error("❌ Failed to get trade token after retries")
                        # Continue to retry loop for complete login failure
                        
                else:
                    # Analyze login failure
                    error_msg = login_result.get('msg', 'Unknown error')
                    error_code = login_result.get('code', 'unknown')
                    
                    self.logger.warning(f"❌ Login attempt {attempt} failed: {error_msg} (Code: {error_code})")
                    
                    # Check if this is a retryable error
                    if not self._is_retryable_login_error(login_result):
                        self.logger.error(f"❌ Non-retryable login error: {error_msg}")
                        return False
                
            except Exception as e:
                self.logger.warning(f"❌ Login attempt {attempt} exception: {e}")
                
                # Check if this is a retryable exception
                if not self._is_retryable_exception(e):
                    self.logger.error(f"❌ Non-retryable exception during login: {e}")
                    return False
            
            # If we get here, we need to retry
            if attempt < self.max_login_attempts:
                # Calculate delay with exponential backoff
                delay = self.base_login_delay * (2 ** (attempt - 1))  # 30, 60, 120 seconds
                delay = min(delay, 300)  # Cap at 5 minutes
                
                self.logger.info(f"⏳ Waiting {delay} seconds before retry {attempt + 1}...")
                time.sleep(delay)
            else:
                self.logger.error(f"❌ All {self.max_login_attempts} login attempts failed")
        
        return False
    
    def _get_trade_token_with_retry(self, trading_pin: str) -> bool:
        """Get trade token with retry logic"""
        for attempt in range(1, self.max_trade_token_attempts + 1):
            try:
                self.logger.info(f"Getting trade token (attempt {attempt}/{self.max_trade_token_attempts})...")
                
                if self.wb.get_trade_token(trading_pin):
                    self.logger.info("✅ Trade token obtained successfully")
                    return True
                else:
                    self.logger.warning(f"❌ Trade token attempt {attempt} failed")
                    
            except Exception as e:
                self.logger.warning(f"❌ Trade token attempt {attempt} exception: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_trade_token_attempts:
                delay = self.base_trade_token_delay * attempt  # 10, 20 seconds
                self.logger.info(f"⏳ Waiting {delay} seconds before trade token retry...")
                time.sleep(delay)
        
        self.logger.error(f"❌ Failed to get trade token after {self.max_trade_token_attempts} attempts")
        return False
    
    def _is_retryable_login_error(self, login_result: Dict) -> bool:
        """Determine if a login error is retryable"""
        error_code = login_result.get('code', '').lower()
        error_msg = login_result.get('msg', '').lower()
        
        # Non-retryable errors (permanent failures)
        non_retryable_codes = [
            'phone.illegal',
            'user.passwd.error',
            'account.freeze',
            'account.lock',
            'user.not.exist'
        ]
        
        non_retryable_messages = [
            'invalid username',
            'invalid password', 
            'account suspended',
            'account locked',
            'user not found'
        ]
        
        # Check for non-retryable conditions
        for code in non_retryable_codes:
            if code in error_code:
                return False
        
        for msg in non_retryable_messages:
            if msg in error_msg:
                return False
        
        # Default to retryable for unknown errors
        return True
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Determine if an exception is retryable"""
        # Network-related exceptions are retryable
        retryable_exceptions = [
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            ConnectionError,
            TimeoutError,
            OSError  # Can include network issues
        ]
        
        for exc_type in retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # Default to retryable for unknown exceptions
        return True
    
    def login_with_credentials(self, username: str, password: str, trading_pin: str, 
                             device_name: str = '', mfa: str = '', 
                             question_id: str = '', question_answer: str = '') -> bool:
        """Login with explicit credentials (for setup or testing)"""
        try:
            self.logger.info("Attempting login with provided credentials...")
            
            login_result = self.wb.login(
                username=username,
                password=password,
                device_name=device_name,
                mfa=mfa,
                question_id=question_id,
                question_answer=question_answer
            )
            
            if 'accessToken' in login_result:
                self.logger.info("✅ Login successful")
                self.is_logged_in = True
                
                # Get trade token
                if self.wb.get_trade_token(trading_pin):
                    self.logger.info("✅ Trade token obtained")
                    return True
                else:
                    self.logger.error("❌ Failed to get trade token")
                    return False
            else:
                self.logger.error(f"❌ Login failed: {login_result.get('msg', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Exception during login: {e}")
            return False
    
    def logout(self) -> bool:
        """Logout from Webull"""
        try:
            if self.is_logged_in:
                response_code = self.wb.logout()
                if response_code == 200:
                    self.logger.info("🔐 Logged out successfully")
                    self.is_logged_in = False
                    return True
                else:
                    self.logger.warning(f"⚠️ Logout returned code: {response_code}")
                    return False
            else:
                self.logger.info("Already logged out")
                return True
        except Exception as e:
            self.logger.warning(f"⚠️ Logout warning: {e}")
            return False
    
    def check_login_status(self) -> bool:
        """Check if currently logged in and properly initialize account context"""
        try:
            # STORE current account context
            current_account_id = self.wb._account_id
            current_zone = self.wb.zone_var
            
            # Test with a simple API call that doesn't change account context
            try:
                # Use get_quote instead of get_account_id to test login status
                test_quote = self.wb.get_quote('SPY')
                if test_quote and 'close' in test_quote:
                    self.is_logged_in = True
                    
                    # RESTORE account context
                    self.wb._account_id = current_account_id
                    self.wb.zone_var = current_zone
                    
                    self.logger.debug(f"Login status verified, account context preserved: {current_account_id}")
                    return True
                else:
                    self.is_logged_in = False
                    return False
            except Exception as e:
                self.logger.debug(f"Login status check failed: {e}")
                self.is_logged_in = False
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking login status: {e}")
            self.is_logged_in = False
            return False
    
    def refresh_login(self) -> bool:
        """Refresh the login session if possible - DISABLED DUE TO ERRORS"""
        try:
            self.logger.info("Login refresh requested but disabled due to errors")
            self.logger.info("Refresh functionality has been disabled - will require fresh login")
            self.is_logged_in = False
            return False
            
            # ORIGINAL CODE COMMENTED OUT:
            # self.logger.info("Attempting to refresh login session...")
            # result = self.wb.refresh_login()
            # 
            # if 'accessToken' in result and result['accessToken']:
            #     self.logger.info("✅ Login session refreshed successfully")
            #     self.is_logged_in = True
            #     return True
            # else:
            #     self.logger.warning("❌ Failed to refresh login session")
            #     self.is_logged_in = False
            #     return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in refresh login: {e}")
            self.is_logged_in = False
            return False
    
    def get_login_info(self) -> Dict:
        """Get information about current login status"""
        return {
            'is_logged_in': self.is_logged_in,
            'access_token_exists': bool(getattr(self.wb, '_access_token', None)),
            'trade_token_exists': bool(getattr(self.wb, '_trade_token', None)),
            'account_id': getattr(self.wb, '_account_id', None),
            'uuid': getattr(self.wb, '_uuid', None)
        }