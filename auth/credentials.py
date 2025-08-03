# auth/credentials.py
import os
import json
import logging
from datetime import datetime
from cryptography.fernet import Fernet
from pathlib import Path

class CredentialManager:
    """Handles encrypted credential storage and retrieval in data folder"""
    
    def __init__(self, data_folder="data", credentials_file="trading_credentials.enc", key_file="trading_key.key", logger=None):
        # Create data folder if it doesn't exist
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        
        # Set file paths in data folder
        self.credentials_file = self.data_folder / credentials_file
        self.key_file = self.data_folder / key_file
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_key(self):
        """Generate encryption key for credentials"""
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        return key
    
    def load_key(self):
        """Load encryption key"""
        try:
            with open(self.key_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Encryption key file {self.key_file} not found!")
            return None
    
    def encrypt_credentials(self, username: str, password: str, trading_pin: str, did: str = None) -> bool:
        """Encrypt and store trading credentials"""
        try:
            # Generate or load key
            if not self.key_file.exists():
                key = self.generate_key()
                self.logger.info("Generated new encryption key")
            else:
                key = self.load_key()
            
            if key is None:
                raise Exception("Could not load encryption key")
            
            f = Fernet(key)
            
            # Create credentials dictionary
            credentials = {
                'username': username,
                'password': password,
                'trading_pin': trading_pin,
                'did': did,
                'created_date': datetime.now().isoformat()
            }
            
            # Encrypt credentials
            credentials_json = json.dumps(credentials)
            encrypted_credentials = f.encrypt(credentials_json.encode())
            
            # Save encrypted credentials
            with open(self.credentials_file, 'wb') as file:
                file.write(encrypted_credentials)
            
            self.logger.info(f"Credentials encrypted and saved to {self.credentials_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt credentials: {e}")
            return False
    
    def load_credentials(self) -> dict:
        """Load and decrypt trading credentials"""
        try:
            # Check if files exist
            if not self.credentials_file.exists():
                raise FileNotFoundError(f"Credentials file {self.credentials_file} not found")
            
            if not self.key_file.exists():
                raise FileNotFoundError(f"Key file {self.key_file} not found")
            
            # Load key and decrypt
            key = self.load_key()
            if key is None:
                raise Exception("Could not load encryption key")
            
            f = Fernet(key)
            
            # Read and decrypt credentials
            with open(self.credentials_file, 'rb') as file:
                encrypted_credentials = file.read()
            
            decrypted_credentials = f.decrypt(encrypted_credentials)
            credentials = json.loads(decrypted_credentials.decode())
            
            self.logger.info("Credentials loaded successfully")
            return credentials
            
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
            raise
    
    def credentials_exist(self) -> bool:
        """Check if encrypted credentials exist"""
        return self.credentials_file.exists() and self.key_file.exists()
    
    def delete_credentials(self) -> bool:
        """Delete stored credentials (for security)"""
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
                self.logger.info("Credentials file deleted")
            
            if self.key_file.exists():
                self.key_file.unlink()
                self.logger.info("Key file deleted")
            
            return True
        except Exception as e:
            self.logger.error(f"Error deleting credentials: {e}")
            return False
    
    def update_credentials(self, username: str = None, password: str = None, 
                          trading_pin: str = None, did: str = None) -> bool:
        """Update existing credentials (load current, modify, save)"""
        try:
            # Load existing credentials
            current_creds = self.load_credentials()
            
            # Update only provided fields
            if username is not None:
                current_creds['username'] = username
            if password is not None:
                current_creds['password'] = password
            if trading_pin is not None:
                current_creds['trading_pin'] = trading_pin
            if did is not None:
                current_creds['did'] = did
            
            # Update timestamp
            current_creds['updated_date'] = datetime.now().isoformat()
            
            # Re-encrypt and save
            return self.encrypt_credentials(
                current_creds['username'],
                current_creds['password'],
                current_creds['trading_pin'],
                current_creds.get('did')
            )
            
        except Exception as e:
            self.logger.error(f"Error updating credentials: {e}")
            return False
    
    def validate_credentials(self, credentials: dict) -> bool:
        """Validate that credentials contain required fields"""
        required_fields = ['username', 'password', 'trading_pin']
        
        for field in required_fields:
            if field not in credentials or not credentials[field]:
                self.logger.error(f"Missing or empty required field: {field}")
                return False
        
        return True
    
    def get_credential_info(self) -> dict:
        """Get non-sensitive info about stored credentials"""
        try:
            if not self.credentials_exist():
                return {'exists': False}
            
            credentials = self.load_credentials()
            
            return {
                'exists': True,
                'username': credentials.get('username', 'Unknown'),
                'created_date': credentials.get('created_date', 'Unknown'),
                'updated_date': credentials.get('updated_date'),
                'has_did': bool(credentials.get('did')),
                'storage_location': str(self.credentials_file)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting credential info: {e}")
            return {'exists': False, 'error': str(e)}

def setup_credentials_interactive():
    """Interactive setup for credentials with improved DID handling"""
    print("🔐 ENHANCED MULTI-ACCOUNT TRADING CREDENTIALS SETUP")
    print("="*60)
    print("This is a ONE-TIME setup to securely store your trading credentials.")
    print("After this, the enhanced system will run automatically without prompts.")
    print("📁 Files will be stored in the 'data' folder for better organization.")
    print()
    
    username = input("Enter your Webull username/email: ")
    password = input("Enter your Webull password: ")
    trading_pin = input("Enter your 6-digit Webull trading PIN: ")
    
    # Improved DID explanation and capture
    print()
    print("🖼️  BROWSER DID SETUP (Prevents Image Verification Errors)")
    print("=" * 60)
    print("A Browser DID prevents image verification errors during login.")
    print("This is HIGHLY RECOMMENDED for automated trading.")
    print()
    print("To get your Browser DID:")
    print("1. Open https://www.webull.com/ in your browser")
    print("2. Log in normally (complete any image verification)")
    print("3. Press F12 (Developer Tools)")
    print("4. Go to 'Network' tab")
    print("5. Refresh the page (F5)")
    print("6. Click any request → 'Headers' → find 'did' field")
    print("7. Copy the 32-character hex string")
    print()
    
    use_browser_did = input("Do you want to add a Browser DID now? (highly recommended) (y/n): ").lower().strip()
    did = None
    
    if use_browser_did in ['y', 'yes']:
        did = input("Enter your Browser DID (32 characters): ").strip()
        if did:
            if len(did) == 32:
                print("✅ Browser DID looks good!")
            else:
                print(f"⚠️  Warning: Expected 32 characters, got {len(did)}")
                confirm = input("Continue anyway? (y/n): ").lower().strip()
                if confirm not in ['y', 'yes']:
                    did = None
    
    if not did:
        print("⚠️  No Browser DID provided.")
        print("   You may get image verification errors.")
        print("   If this happens, run: python tests/check_did.py")
    
    # Create credential manager and encrypt credentials
    cred_manager = CredentialManager()
    success = cred_manager.encrypt_credentials(username, password, trading_pin, did)
    
    if success:
        print()
        print("✅ Setup complete! Your credentials are now encrypted and stored.")
        print(f"📁 Location: {cred_manager.credentials_file}")
        if did:
            print("✅ Browser DID included - should prevent image verification errors")
        else:
            print("⚠️  No Browser DID - you may need to add one later if you get errors")
        
        print()
        print("🚀 You can now run the automated trading system:")
        print("   python main.py")
        print()
        if not did:
            print("💡 If you get image verification errors:")
            print("   python tests/check_did.py")
        
        return True
    else:
        print("❌ Failed to setup credentials. Please try again.")
        return False
    

if __name__ == "__main__":
        
    try:
        setup_credentials_interactive()
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")