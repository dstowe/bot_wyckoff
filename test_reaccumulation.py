#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Reaccumulation Functionality
Run this script to test the reaccumulation detection without trading
"""

import logging
from fractional_position_system import WyckoffReaccumulationDetector

def test_reaccumulation():
    """Test reaccumulation detection on sample symbols"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create detector
    detector = WyckoffReaccumulationDetector(logger)
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("🔍 Testing Wyckoff Reaccumulation Detection")
    print("=" * 50)
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        
        # Mock position data
        mock_position = {
            'symbol': symbol,
            'shares': 10.0,
            'avg_cost': 150.0,
            'account_type': 'CASH'
        }
        
        try:
            signal = detector.analyze_for_reaccumulation(symbol, mock_position)
            
            if signal:
                print(f"✅ Reaccumulation signal detected!")
                print(f"   Phase: {signal.phase_type}")
                print(f"   Strength: {signal.strength:.2f}")
                print(f"   Addition %: {signal.addition_percentage:.1%}")
                print(f"   Current Price: ${signal.current_price:.2f}")
                print(f"   Support Level: ${signal.support_level:.2f}")
                print(f"   Reasoning: {signal.reasoning}")
            else:
                print(f"❌ No reaccumulation signal detected")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    print("\n✅ Reaccumulation test complete!")
    print("\n💡 If signals were found, the system is working correctly.")
    print("   Now you can run your main trading bot with position addition capabilities!")

if __name__ == "__main__":
    test_reaccumulation()
