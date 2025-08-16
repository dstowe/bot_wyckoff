#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Exit Strategy System 🎯
"""

import sys
import logging
from datetime import datetime

def test_enhanced_exit_strategy():
    """Test the enhanced exit strategy system"""
    
    print("🎯 Enhanced Exit Strategy Tester")
    print("=" * 50)
    
    try:
        # Import the enhanced exit strategy
        from enhanced_exit_strategy import EnhancedExitStrategyManager
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        logger = logging.getLogger(__name__)
        
        # Create manager
        exit_manager = EnhancedExitStrategyManager(logger)
        
        # Get current VIX
        current_vix = exit_manager.get_cached_vix()
        volatility_regime = exit_manager.volatility_calculator.get_volatility_regime(current_vix)
        
        print(f"📊 Current Market Conditions:")
        print(f"   VIX: {current_vix:.1f} ({volatility_regime} volatility)")
        print()
        
        # Show volatility adjustments
        adjustments = exit_manager.volatility_calculator.VOLATILITY_ADJUSTMENTS[volatility_regime]
        print(f"📈 Current Exit Strategy Adjustments:")
        print(f"   {adjustments['description']}")
        print(f"   Gain Multiplier: {adjustments['gain_multiplier']:.2f}")
        print(f"   Sell Multiplier: {adjustments['sell_multiplier']:.2f}")
        print()
        
        # Test with sample positions
        test_positions = [
            {
                'symbol': 'AAPL',
                'account_type': 'CASH',
                'shares': 10.5,
                'avg_cost': 180.0,
                'entry_phase': 'SOS',
                'position_size_pct': 0.12,
                'first_purchase_date': '2024-11-01'
            },
            {
                'symbol': 'MSFT',
                'account_type': 'MARGIN',
                'shares': 5.0,
                'avg_cost': 350.0,
                'entry_phase': 'LPS',
                'position_size_pct': 0.08,
                'first_purchase_date': '2024-10-15'
            }
        ]
        
        print("🔍 Testing Exit Strategy Analysis:")
        print("-" * 40)
        
        for position in test_positions:
            print(f"\n📈 {position['symbol']} Analysis:")
            
            # Create exit plan
            exit_plan = exit_manager.create_comprehensive_exit_plan(position)
            
            print(f"   Current Price: ${exit_plan.current_price:.2f}")
            print(f"   Entry Price: ${exit_plan.entry_price:.2f}")
            print(f"   Current Gain: {exit_plan.current_gain_pct:+.1%}")
            print(f"   Recommendation: {exit_plan.recommended_action}")
            print(f"   Urgency: {exit_plan.exit_urgency}")
            
            # Show exit targets with VIX adjustments
            print(f"   📊 Dynamic Exit Targets (VIX {current_vix:.1f}):")
            base_targets = [(0.06, 0.15), (0.12, 0.20), (0.20, 0.25), (0.30, 0.40)]
            
            for i, (base_gain, base_sell) in enumerate(base_targets):
                if i < len(exit_plan.exit_targets):
                    target = exit_plan.exit_targets[i]
                    status = "✅ HIT" if exit_plan.current_gain_pct >= target.gain_percentage else "⏳ Pending"
                    
                    # Show the adjustment
                    gain_change = (target.gain_percentage / base_gain - 1) * 100
                    sell_change = (target.sell_percentage / base_sell - 1) * 100
                    
                    print(f"     {i+1}. {target.gain_percentage:.1%} gain ({gain_change:+.0f}%) → Sell {target.sell_percentage:.0%} ({sell_change:+.0f}%) {status}")
            
            # Check if should exit now
            should_exit, reason, percentage = exit_manager.should_exit_now(position)
            if should_exit:
                print(f"   🚨 EXIT SIGNAL: {reason} (Sell {percentage:.0%})")
            else:
                print(f"   ✅ HOLD: {reason}")
        
        # Test different VIX scenarios
        print(f"\n🌪️ Testing Different Volatility Scenarios:")
        print("-" * 50)
        
        test_vix_levels = [15.0, 22.0, 28.0, 45.0]
        
        for vix in test_vix_levels:
            regime = exit_manager.volatility_calculator.get_volatility_regime(vix)
            targets = exit_manager.volatility_calculator.calculate_volatility_adjusted_targets(vix)
            
            print(f"\n   VIX {vix:.1f} ({regime}):")
            for i, target in enumerate(targets, 1):
                print(f"     Target {i}: {target.gain_percentage:.1%} → Sell {target.sell_percentage:.0%}")
        
        print("\n✅ Enhanced Exit Strategy Test Complete!")
        print()
        print("🚀 Key Features Demonstrated:")
        print("   • Dynamic profit targets based on VIX volatility")
        print("   • Position-specific adjustments (time, Wyckoff phase)")
        print("   • Automatic exit recommendations")
        print("   • Integration with day trade protection")
        print()
        print("📈 Expected Results:")
        print("   • Low VIX (<20): Extended targets, let profits run")
        print("   • High VIX (>25): Tighter targets, take profits early") 
        print("   • Crisis VIX (>35): Very tight targets, preserve capital")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure enhanced_exit_strategy.py is in the current directory")
        print("Run: python integrate_exit_strategy.py first")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_exit_strategy()
