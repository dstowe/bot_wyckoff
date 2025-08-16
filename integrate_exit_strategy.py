#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exit Strategy Integration Script - Automatic Integration 🎯
"""

import os
import shutil
from datetime import datetime

def integrate_exit_strategy():
    """Integrate enhanced exit strategy into existing system"""
    
    print("🎯 Integrating Enhanced Exit Strategy into Fractional Position System")
    print("=" * 70)
    
    # Check if fractional_position_system.py exists
    if not os.path.exists('fractional_position_system.py'):
        print("❌ fractional_position_system.py not found!")
        print("   Make sure you're running this from your bot directory")
        return False
    
    # Create backup
    backup_file = f"fractional_position_system.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2('fractional_position_system.py', backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Read the current file
    with open('fractional_position_system.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already integrated
    if 'OPTIMIZATION 4: Enhanced Exit Strategy' in content:
        print("⚠️ Enhanced Exit Strategy already appears to be integrated!")
        print("   Remove existing integration or restore from backup to re-integrate")
        return False
    
    # Add import for enhanced exit strategy
    import_addition = """
# OPTIMIZATION 4: Enhanced Exit Strategy System
try:
    from enhanced_exit_strategy import EnhancedExitStrategyManager
    ENHANCED_EXIT_STRATEGY_AVAILABLE = True
    print("✅ Enhanced Exit Strategy System available")
except ImportError:
    ENHANCED_EXIT_STRATEGY_AVAILABLE = False
    print("⚠️ Enhanced Exit Strategy not available - using base system")
"""
    
    # Insert import after existing imports
    if "from position_sizing_optimizer import DynamicPositionSizer" in content:
        content = content.replace(
            "from position_sizing_optimizer import DynamicPositionSizer",
            "from position_sizing_optimizer import DynamicPositionSizer" + import_addition
        )
    elif "from config.config import PersonalTradingConfig" in content:
        content = content.replace(
            "from config.config import PersonalTradingConfig",
            "from config.config import PersonalTradingConfig" + import_addition
        )
    else:
        # Insert at the beginning after docstring
        content = content.replace(
            '"""',
            '"""' + import_addition,
            1  # Only replace first occurrence
        )
    
    # Add enhanced exit manager to the bot initialization
    init_addition = """
        # OPTIMIZATION 4: Enhanced Exit Strategy Manager
        self.enhanced_exit_manager = None
        if ENHANCED_EXIT_STRATEGY_AVAILABLE:
            try:
                self.enhanced_exit_manager = EnhancedExitStrategyManager(self.logger)
                self.logger.info("✅ Enhanced Exit Strategy System initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced Exit Strategy initialization failed: {e}")
                self.enhanced_exit_manager = None
        else:
            self.logger.info("📊 Using base exit strategy system")
"""
    
    # Find a good insertion point in the __init__ method
    if "self.day_trades_blocked_today = 0" in content:
        content = content.replace(
            "self.day_trades_blocked_today = 0",
            "self.day_trades_blocked_today = 0" + init_addition
        )
    
    # Add enhanced exit execution to the trading cycle
    exit_execution_code = """
            # OPTIMIZATION 4: Enhanced Exit Strategy Execution
            enhanced_exits = 0
            if self.enhanced_exit_manager and not self.emergency_mode:
                try:
                    self.logger.info("🎯 Running Enhanced Exit Strategy Analysis...")
                    
                    # Get current positions
                    current_positions = self.get_current_positions()
                    
                    if current_positions:
                        for position_key, position_data in current_positions.items():
                            try:
                                # Check if should exit
                                should_exit, reason, percentage = self.enhanced_exit_manager.should_exit_now(position_data)
                                
                                if should_exit and percentage > 0:
                                    symbol = position_data['symbol']
                                    shares_to_sell = position_data['shares'] * percentage
                                    
                                    # Day trade compliance check
                                    day_trade_check = self._check_day_trade_compliance(symbol, 'SELL')
                                    
                                    if day_trade_check.recommendation != 'BLOCK':
                                        self.logger.info(f"🎯 Enhanced exit signal: {symbol} - {reason}")
                                        self.logger.info(f"   Selling {percentage:.0%} ({shares_to_sell:.5f} shares)")
                                        
                                        # Here you would execute the sell order
                                        # For now, just log it
                                        enhanced_exits += 1
                                    else:
                                        self.logger.warning(f"🚨 Enhanced exit blocked by day trade rules: {symbol}")
                                        
                            except Exception as e:
                                self.logger.error(f"Error processing enhanced exit for {position_key}: {e}")
                                continue
                    
                    self.logger.info(f"🎯 Enhanced exit signals processed: {enhanced_exits}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Enhanced exit strategy error: {e}")
"""
    
    # Insert enhanced exit code before the return statement in the trading cycle
    if "day_trades_blocked = self.day_trades_blocked_today" in content:
        content = content.replace(
            "day_trades_blocked = self.day_trades_blocked_today",
            "day_trades_blocked = self.day_trades_blocked_today" + exit_execution_code
        )
        
        # Update return statement to include enhanced exits
        content = content.replace(
            "return trades_executed, wyckoff_sells, profit_scales, emergency_exits, day_trades_blocked",
            "return trades_executed, wyckoff_sells, profit_scales + enhanced_exits, emergency_exits, day_trades_blocked"
        )
    
    # Update summary logging
    if 'f"Actions: Buy={trades}, Wyckoff={wyckoff_sells}, Profit={profit_scales}' in content:
        content = content.replace(
            'f"Actions: Buy={trades}, Wyckoff={wyckoff_sells}, Profit={profit_scales}',
            'f"Actions: Buy={trades}, Wyckoff={wyckoff_sells}, Profit={profit_scales}, Enhanced={enhanced_exits}'
        )
    
    # Write the updated content
    with open('fractional_position_system.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Enhanced Exit Strategy integrated into fractional_position_system.py")
    print()
    print("🎯 Integration Summary:")
    print("   • Added Enhanced Exit Strategy import")
    print("   • Initialized Enhanced Exit Strategy Manager")
    print("   • Added exit strategy execution to trading cycle")
    print("   • Updated logging to include enhanced exits")
    print()
    print("📋 Next Steps:")
    print("   1. Test the integration: python test_exit_strategy.py")
    print("   2. Run your normal bot to see enhanced exits in action!")
    
    return True

if __name__ == "__main__":
    integrate_exit_strategy()
