# auth/session_manager.py
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

class SessionManager:
    """Manages trading session persistence and token management in data folder"""
    
    def __init__(self, data_folder="data", session_file="webull_session.json", logger=None):
        # Create data folder if it doesn't exist
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        
        # Set session file path in data folder
        self.session_file = self.data_folder / session_file
        self.logger = logger or logging.getLogger(__name__)
        self.session_data = {}
    
    def save_session(self, wb) -> bool:
        """Save current session data"""
        try:
            session_data = {
                'access_token': getattr(wb, '_access_token', ''),
                'refresh_token': getattr(wb, '_refresh_token', ''),
                'token_expire': getattr(wb, '_token_expire', ''),
                'uuid': getattr(wb, '_uuid', ''),
                'account_id': getattr(wb, '_account_id', ''),
                'trade_token': getattr(wb, '_trade_token', ''),
                'zone_var': getattr(wb, 'zone_var', ''),
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.session_data = session_data
            self.logger.debug(f"Session saved to {self.session_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self, wb) -> bool:
        """Load session data into webull instance with better validation"""
        try:
            if not self.session_file.exists():
                self.logger.debug("No session file found")
                return False
            
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            self.session_data = session_data
            self.logger.debug(f"Session file loaded from {self.session_file}")
            
            # Basic validation - check if session has required fields
            required_fields = ['access_token', 'refresh_token', 'uuid']
            missing_fields = [field for field in required_fields if not session_data.get(field)]
            
            if missing_fields:
                self.logger.debug(f"Session missing required fields: {missing_fields}")
                return False
            
            # Check token expiration first (before API calls)
            if self._is_token_expired(session_data):
                self.logger.debug("Session token has expired")
                return False
            
            # Check session age (48 hours max)
            if self._is_session_too_old(session_data):
                self.logger.debug("Session is too old")
                return False
            
            # Apply session data to webull instance
            wb._access_token = session_data.get('access_token', '')
            wb._refresh_token = session_data.get('refresh_token', '')
            wb._token_expire = session_data.get('token_expire', '')
            wb._uuid = session_data.get('uuid', '')
            wb._account_id = session_data.get('account_id', '')
            wb._trade_token = session_data.get('trade_token', '')
            wb.zone_var = session_data.get('zone_var', 'dc_core_r1')
            
            self.logger.debug("Session data applied to webull instance")
            
            # For now, trust the session if it passes basic validation
            # The login_manager will do the actual verification
            self.logger.debug("Session loaded and appears valid")
            return True
            
        except Exception as e:
            self.logger.debug(f"Error loading session: {e}")
            return False
    
    def _is_token_expired(self, session_data: Dict) -> bool:
        """Check if the token has expired based on token_expire field"""
        try:
            token_expire = session_data.get('token_expire', '')
            if not token_expire:
                return False  # No expiration info, assume valid
            
            # Parse token expiration
            expire_time = datetime.fromisoformat(token_expire.replace('+0000', '+00:00'))
            current_time = datetime.now(expire_time.tzinfo)
            
            # Add 5 minute buffer
            if expire_time <= current_time + timedelta(minutes=5):
                self.logger.debug("Token expires soon or has expired")
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking token expiration: {e}")
            return False  # If we can't check, assume it's valid
    
    def _is_session_too_old(self, session_data: Dict) -> bool:
        """Check if session is too old (48 hours)"""
        try:
            saved_at = session_data.get('saved_at', '')
            if not saved_at:
                return False  # No timestamp, assume valid
            
            saved_time = datetime.fromisoformat(saved_at)
            age = datetime.now() - saved_time
            
            # Sessions older than 48 hours are considered stale
            if age > timedelta(hours=48):
                self.logger.debug(f"Session age: {age.total_seconds()/3600:.1f} hours")
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking session age: {e}")
            return False  # If we can't check, assume it's valid
    
    def clear_session(self) -> bool:
        """Clear stored session data"""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
                self.logger.debug("Session file removed")
            
            self.session_data = {}
            return True
            
        except Exception as e:
            self.logger.debug(f"Error clearing session: {e}")
            return False
    
    def auto_manage_session(self, wb, force_refresh=False) -> bool:
        """Automatically manage session with simplified logic"""
        try:
            # Try to load existing session first
            if not force_refresh:
                if self.load_session(wb):
                    self.logger.debug("Session loaded successfully")
                    return True
                else:
                    self.logger.debug("Session validation failed")
            
            # No valid session found
            return False
            
        except Exception as e:
            self.logger.debug(f"Error in session management: {e}")
            self.session_data = {}
            return False
    
    def get_session_info(self) -> Dict:
        """Get information about current session"""
        if not self.session_data:
            return {'exists': False}
        
        info = {
            'exists': True,
            'access_token_exists': bool(self.session_data.get('access_token')),
            'refresh_token_exists': bool(self.session_data.get('refresh_token')),
            'trade_token_exists': bool(self.session_data.get('trade_token')),
            'account_id': self.session_data.get('account_id'),
            'saved_at': self.session_data.get('saved_at'),
            'token_expire': self.session_data.get('token_expire'),
            'storage_location': str(self.session_file)
        }
        
        # Calculate time until expiration
        token_expire = self.session_data.get('token_expire', '')
        if token_expire:
            try:
                expire_time = datetime.fromisoformat(token_expire.replace('+0000', '+00:00'))
                current_time = datetime.now(expire_time.tzinfo)
                time_until_expire = expire_time - current_time
                
                info['expires_in_minutes'] = int(time_until_expire.total_seconds() / 60)
                info['is_expired'] = time_until_expire.total_seconds() <= 0
                
            except ValueError:
                info['expires_in_minutes'] = None
                info['is_expired'] = None
        
        return info
    
    def backup_session(self, backup_suffix=None) -> bool:
        """Create a backup of current session"""
        try:
            if not self.session_file.exists():
                return False
            
            if backup_suffix is None:
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            backup_file = self.session_file.with_suffix(f'.backup_{backup_suffix}.json')
            
            with open(self.session_file, 'r') as source:
                with open(backup_file, 'w') as backup:
                    backup.write(source.read())
            
            self.logger.debug(f"Session backed up to {backup_file}")
            return True
            
        except Exception as e:
            self.logger.debug(f"Error backing up session: {e}")
            return False