#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Exit Strategy Refinement System - Optimization 4 🎯
Dynamic profit scaling based on volatility, position characteristics, and market conditions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import time

@dataclass
class ExitTarget:
    """Individual exit target with context"""
    gain_percentage: float
    sell_percentage: float
    target_type: str  # 'PROFIT_SCALING', 'VOLATILITY_BASED', 'TIME_BASED', 'WYCKOFF_WARNING'
    urgency: str     # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    reasoning: str
    vix_context: float
    days_held: int
    is_active: bool = True

@dataclass
class PositionExitPlan:
    """Complete exit plan for a position"""
    symbol: str
    account_type: str
    current_shares: float
    entry_price: float
    current_price: float
    current_gain_pct: float
    exit_targets: List[ExitTarget]
    wyckoff_warnings: List[str]
    recommended_action: str  # 'HOLD', 'PARTIAL_EXIT', 'FULL_EXIT', 'EMERGENCY_EXIT'
    exit_urgency: str
    plan_created_at: datetime

class VolatilityBasedExitCalculator:
    """Core calculator for volatility-based exit targets"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # VIX thresholds for exit strategy adjustment
        self.VIX_LOW = 20.0
        self.VIX_MEDIUM = 25.0
        self.VIX_HIGH = 35.0
        self.VIX_CRISIS = 50.0
        
        # Base profit targets (current system)
        self.BASE_TARGETS = [
            {'gain_pct': 0.06, 'sell_pct': 0.15},  # 6% → sell 15%
            {'gain_pct': 0.12, 'sell_pct': 0.20},  # 12% → sell 20%
            {'gain_pct': 0.20, 'sell_pct': 0.25},  # 20% → sell 25%
            {'gain_pct': 0.30, 'sell_pct': 0.40}   # 30% → sell 40%
        ]
        
        # Volatility adjustments - THIS IS THE CORE IMPROVEMENT
        self.VOLATILITY_ADJUSTMENTS = {
            'LOW': {
                'gain_multiplier': 1.33,  # Extend targets: 6%→8%, 12%→16%, 20%→27%, 30%→40%
                'sell_multiplier': 0.9,   # Sell less aggressively
                'description': 'Low volatility (VIX <20) - let profits run'
            },
            'MEDIUM': {
                'gain_multiplier': 1.0,   # Keep base targets
                'sell_multiplier': 1.0,
                'description': 'Medium volatility (VIX 20-25) - base strategy'
            },
            'HIGH': {
                'gain_multiplier': 0.67,  # Tighten targets: 6%→4%, 12%→8%, 20%→13%, 30%→20%
                'sell_multiplier': 1.2,   # Sell more aggressively
                'description': 'High volatility (VIX >25) - take profits early'
            },
            'CRISIS': {
                'gain_multiplier': 0.5,   # Very tight targets: 6%→3%, 12%→6%, 20%→10%, 30%→15%
                'sell_multiplier': 1.5,   # Sell very aggressively
                'description': 'Crisis volatility (VIX >35) - preserve capital'
            }
        }
    
    def get_current_vix(self) -> float:
        """Get current VIX level"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="2d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            self.logger.debug(f"Error fetching VIX: {e}")
        return 25.0  # Default fallback
    
    def get_volatility_regime(self, vix_level: float) -> str:
        """Determine volatility regime based on VIX"""
        if vix_level >= self.VIX_CRISIS:
            return 'CRISIS'
        elif vix_level >= self.VIX_HIGH:
            return 'HIGH'
        elif vix_level >= self.VIX_MEDIUM:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def calculate_volatility_adjusted_targets(self, vix_level: float) -> List[ExitTarget]:
        """Calculate volatility-adjusted profit targets - CORE FUNCTIONALITY"""
        volatility_regime = self.get_volatility_regime(vix_level)
        adjustments = self.VOLATILITY_ADJUSTMENTS[volatility_regime]
        
        adjusted_targets = []
        
        for i, base_target in enumerate(self.BASE_TARGETS):
            adjusted_gain = base_target['gain_pct'] * adjustments['gain_multiplier']
            adjusted_sell = min(1.0, base_target['sell_pct'] * adjustments['sell_multiplier'])
            
            target = ExitTarget(
                gain_percentage=adjusted_gain,
                sell_percentage=adjusted_sell,
                target_type='VOLATILITY_BASED',
                urgency='MEDIUM' if volatility_regime in ['LOW', 'MEDIUM'] else 'HIGH',
                reasoning=f"VIX {vix_level:.1f} - {adjustments['description']}",
                vix_context=vix_level,
                days_held=0  # Will be updated per position
            )
            adjusted_targets.append(target)
        
        return adjusted_targets

class PositionSpecificExitCalculator:
    """Calculate position-specific exit adjustments"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_time_based_adjustments(self, days_held: int, base_targets: List[ExitTarget]) -> List[ExitTarget]:
        """Adjust targets based on time held"""
        time_adjustments = []
        
        for target in base_targets:
            # Time-based multipliers
            if days_held <= 3:
                time_multiplier = 0.9
                urgency_boost = 'Short-term trade - take quick profits'
            elif days_held <= 7:
                time_multiplier = 1.0
                urgency_boost = 'Normal timeframe'
            elif days_held <= 21:
                time_multiplier = 1.1
                urgency_boost = 'Medium-term hold - let profits run'
            elif days_held <= 60:
                time_multiplier = 1.2
                urgency_boost = 'Long-term position - extend targets'
            else:
                time_multiplier = 1.15
                urgency_boost = 'Very long hold - start scaling out'
            
            adjusted_target = ExitTarget(
                gain_percentage=target.gain_percentage * time_multiplier,
                sell_percentage=target.sell_percentage,
                target_type='TIME_BASED',
                urgency=target.urgency,
                reasoning=f"{target.reasoning} + Time ({days_held}d): {urgency_boost}",
                vix_context=target.vix_context,
                days_held=days_held
            )
            time_adjustments.append(adjusted_target)
        
        return time_adjustments
    
    def calculate_wyckoff_phase_adjustments(self, entry_phase: str, base_targets: List[ExitTarget]) -> List[ExitTarget]:
        """Adjust targets based on Wyckoff entry phase"""
        
        # Phase-specific target multipliers
        phase_adjustments = {
            'ST': {
                'gain_multiplier': 0.8,   # Tighter targets for Spring entries
                'sell_multiplier': 1.1,
                'reasoning': 'Spring entry - take profits early'
            },
            'SOS': {
                'gain_multiplier': 1.3,   # Extend targets for Sign of Strength
                'sell_multiplier': 0.9,
                'reasoning': 'SOS entry - let profits run'
            },
            'LPS': {
                'gain_multiplier': 1.1,   # Slightly extend for Last Point of Support
                'sell_multiplier': 1.0,
                'reasoning': 'LPS entry - moderate profit taking'
            },
            'BU': {
                'gain_multiplier': 0.9,   # Slightly tighter for Backup
                'sell_multiplier': 1.05,
                'reasoning': 'Backup entry - cautious profit taking'
            },
            'Creek': {
                'gain_multiplier': 0.7,   # Much tighter for Creek
                'sell_multiplier': 1.2,
                'reasoning': 'Creek entry - take quick profits'
            }
        }
        
        adjustment = phase_adjustments.get(entry_phase, {
            'gain_multiplier': 1.0,
            'sell_multiplier': 1.0,
            'reasoning': 'Unknown phase - base targets'
        })
        
        wyckoff_adjustments = []
        
        for target in base_targets:
            adjusted_target = ExitTarget(
                gain_percentage=target.gain_percentage * adjustment['gain_multiplier'],
                sell_percentage=min(1.0, target.sell_percentage * adjustment['sell_multiplier']),
                target_type='WYCKOFF_PHASE',
                urgency=target.urgency,
                reasoning=f"{target.reasoning} + {adjustment['reasoning']}",
                vix_context=target.vix_context,
                days_held=target.days_held
            )
            wyckoff_adjustments.append(adjusted_target)
        
        return wyckoff_adjustments

class EnhancedExitStrategyManager:
    """Main exit strategy manager that combines all factors"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.volatility_calculator = VolatilityBasedExitCalculator(logger)
        self.position_calculator = PositionSpecificExitCalculator(logger)
        
        # Cache for VIX to avoid repeated API calls
        self._vix_cache = None
        self._vix_cache_time = None
        self._vix_cache_duration = 300  # 5 minutes
    
    def get_cached_vix(self) -> float:
        """Get VIX with caching"""
        now = datetime.now()
        
        if (self._vix_cache is None or 
            self._vix_cache_time is None or
            (now - self._vix_cache_time).total_seconds() > self._vix_cache_duration):
            
            self._vix_cache = self.volatility_calculator.get_current_vix()
            self._vix_cache_time = now
            self.logger.debug(f"Updated VIX cache: {self._vix_cache:.1f}")
        
        return self._vix_cache
    
    def create_comprehensive_exit_plan(self, position: Dict) -> PositionExitPlan:
        """Create a comprehensive exit plan for a position"""
        try:
            # Extract position information
            symbol = position.get('symbol', 'UNKNOWN')
            account_type = position.get('account_type', 'UNKNOWN')
            current_shares = position.get('shares', 0)
            entry_price = position.get('avg_cost', 0)
            entry_phase = position.get('entry_phase', 'UNKNOWN')
            position_size_pct = position.get('position_size_pct', 0.1)
            
            # Calculate days held
            first_purchase_str = position.get('first_purchase_date', datetime.now().strftime('%Y-%m-%d'))
            try:
                first_purchase = datetime.strptime(first_purchase_str, '%Y-%m-%d')
                days_held = (datetime.now() - first_purchase).days
            except:
                days_held = 0
            
            # Get current price
            current_price = self._get_current_price(symbol)
            if current_price == 0:
                current_price = entry_price
            
            # Calculate current gain
            current_gain_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Step 1: Get base volatility-adjusted targets
            current_vix = self.get_cached_vix()
            base_targets = self.volatility_calculator.calculate_volatility_adjusted_targets(current_vix)
            
            # Step 2: Apply time-based adjustments
            time_adjusted = self.position_calculator.calculate_time_based_adjustments(days_held, base_targets)
            
            # Step 3: Apply Wyckoff phase adjustments
            final_targets = self.position_calculator.calculate_wyckoff_phase_adjustments(entry_phase, time_adjusted)
            
            # Step 4: Determine overall recommendation
            recommendation, urgency = self._determine_overall_recommendation(
                current_gain_pct, final_targets, current_vix, days_held
            )
            
            exit_plan = PositionExitPlan(
                symbol=symbol,
                account_type=account_type,
                current_shares=current_shares,
                entry_price=entry_price,
                current_price=current_price,
                current_gain_pct=current_gain_pct,
                exit_targets=final_targets,
                wyckoff_warnings=[],
                recommended_action=recommendation,
                exit_urgency=urgency,
                plan_created_at=datetime.now()
            )
            
            return exit_plan
            
        except Exception as e:
            self.logger.error(f"Error creating exit plan for {position.get('symbol', 'UNKNOWN')}: {e}")
            return self._create_fallback_exit_plan(position)
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            self.logger.debug(f"Error getting price for {symbol}: {e}")
        return 0.0
    
    def _determine_overall_recommendation(self, current_gain_pct: float, targets: List[ExitTarget], 
                                        vix_level: float, days_held: int) -> Tuple[str, str]:
        """Determine overall recommendation and urgency"""
        
        # Check if any target has been hit
        targets_hit = [t for t in targets if current_gain_pct >= t.gain_percentage]
        
        if not targets_hit:
            # No targets hit yet
            if current_gain_pct < -0.15:  # Down more than 15%
                return 'EMERGENCY_EXIT', 'CRITICAL'
            elif current_gain_pct < -0.10:  # Down more than 10%
                return 'CONSIDER_EXIT', 'HIGH'
            elif current_gain_pct < -0.05:  # Down more than 5%
                return 'MONITOR_CLOSELY', 'MEDIUM'
            else:
                return 'HOLD', 'LOW'
        
        # Some targets hit - determine urgency
        if vix_level > 35:  # High volatility
            urgency = 'HIGH'
        elif vix_level > 25:  # Medium volatility
            urgency = 'MEDIUM'
        else:
            urgency = 'LOW'
        
        # Multiple targets hit or high gains
        if len(targets_hit) >= 2 or current_gain_pct > 0.25:
            return 'PARTIAL_EXIT', urgency
        else:
            return 'CONSIDER_PARTIAL_EXIT', urgency
    
    def _create_fallback_exit_plan(self, position: Dict) -> PositionExitPlan:
        """Create a simple fallback exit plan if main calculation fails"""
        symbol = position.get('symbol', 'UNKNOWN')
        
        # Simple base targets
        fallback_targets = [
            ExitTarget(0.06, 0.15, 'FALLBACK', 'LOW', 'Fallback plan - basic targets', 25.0, 0),
            ExitTarget(0.12, 0.20, 'FALLBACK', 'LOW', 'Fallback plan - basic targets', 25.0, 0),
            ExitTarget(0.20, 0.25, 'FALLBACK', 'LOW', 'Fallback plan - basic targets', 25.0, 0)
        ]
        
        return PositionExitPlan(
            symbol=symbol,
            account_type=position.get('account_type', 'UNKNOWN'),
            current_shares=position.get('shares', 0),
            entry_price=position.get('avg_cost', 0),
            current_price=position.get('avg_cost', 0),
            current_gain_pct=0.0,
            exit_targets=fallback_targets,
            wyckoff_warnings=[],
            recommended_action='HOLD',
            exit_urgency='LOW',
            plan_created_at=datetime.now()
        )
    
    def should_exit_now(self, position: Dict) -> Tuple[bool, str, float]:
        """
        Determine if a position should be exited now
        Returns: (should_exit, reason, percentage_to_sell)
        """
        exit_plan = self.create_comprehensive_exit_plan(position)
        
        # Check for targets that have been hit
        for target in exit_plan.exit_targets:
            if exit_plan.current_gain_pct >= target.gain_percentage:
                return True, target.reasoning, target.sell_percentage
        
        # Check for emergency exit conditions
        if exit_plan.current_gain_pct < -0.15:  # Down more than 15%
            return True, "Stop loss triggered - preserving capital", 1.0
        
        if exit_plan.exit_urgency == 'CRITICAL':
            return True, "Critical exit signal", 0.5
        
        return False, "Hold position", 0.0
    
    def generate_exit_report(self, positions: List[Dict]) -> str:
        """Generate a comprehensive exit strategy report"""
        
        current_vix = self.get_cached_vix()
        volatility_regime = self.volatility_calculator.get_volatility_regime(current_vix)
        
        report = f"""
=== ENHANCED EXIT STRATEGY REPORT 🎯 ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Market Conditions: VIX {current_vix:.1f} ({volatility_regime} volatility)

=== VOLATILITY-ADJUSTED TARGETS ===
{self.volatility_calculator.VOLATILITY_ADJUSTMENTS[volatility_regime]['description']}

=== POSITION ANALYSIS ===
"""
        
        immediate_exits = []
        partial_exits = []
        holds = []
        
        for position in positions:
            exit_plan = self.create_comprehensive_exit_plan(position)
            
            if exit_plan.recommended_action in ['FULL_EXIT', 'EMERGENCY_EXIT']:
                immediate_exits.append(exit_plan)
            elif exit_plan.recommended_action in ['PARTIAL_EXIT', 'CONSIDER_PARTIAL_EXIT']:
                partial_exits.append(exit_plan)
            else:
                holds.append(exit_plan)
        
        # Immediate exits
        if immediate_exits:
            report += f"\n🚨 IMMEDIATE EXITS RECOMMENDED ({len(immediate_exits)}):\n"
            for plan in immediate_exits:
                report += f"  {plan.symbol}: {plan.current_gain_pct:+.1%} gain, {plan.exit_urgency} urgency\n"
                report += f"    Reason: {plan.exit_targets[0].reasoning if plan.exit_targets else 'Emergency exit'}\n"
        
        # Partial exits
        if partial_exits:
            report += f"\n📊 PARTIAL EXITS RECOMMENDED ({len(partial_exits)}):\n"
            for plan in partial_exits:
                next_target = next((t for t in plan.exit_targets if plan.current_gain_pct >= t.gain_percentage), None)
                if next_target:
                    report += f"  {plan.symbol}: {plan.current_gain_pct:+.1%} → Sell {next_target.sell_percentage:.0%}\n"
                    report += f"    Target: {next_target.gain_percentage:.1%} gain ({next_target.target_type})\n"
        
        # Holds
        if holds:
            report += f"\n✅ HOLD POSITIONS ({len(holds)}):\n"
            for plan in holds[:10]:  # Show first 10
                next_target = next((t for t in plan.exit_targets if plan.current_gain_pct < t.gain_percentage), None)
                if next_target:
                    report += f"  {plan.symbol}: {plan.current_gain_pct:+.1%} → Next target: {next_target.gain_percentage:.1%}\n"
        
        return report


# Test function
def test_exit_strategy_system():
    """Test the enhanced exit strategy system"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    exit_manager = EnhancedExitStrategyManager()
    
    # Test positions
    test_positions = [
        {
            'symbol': 'AAPL',
            'account_type': 'CASH',
            'shares': 10.5,
            'avg_cost': 150.0,
            'entry_phase': 'SOS',
            'position_size_pct': 0.12,
            'first_purchase_date': '2024-11-01'
        },
        {
            'symbol': 'MSFT',
            'account_type': 'MARGIN', 
            'shares': 5.0,
            'avg_cost': 300.0,
            'entry_phase': 'LPS',
            'position_size_pct': 0.08,
            'first_purchase_date': '2024-10-15'
        }
    ]
    
    print("🎯 Testing Enhanced Exit Strategy System")
    print("=" * 50)
    
    # Test individual position analysis
    for position in test_positions:
        print(f"\nAnalyzing {position['symbol']}:")
        exit_plan = exit_manager.create_comprehensive_exit_plan(position)
        
        print(f"  Current gain: {exit_plan.current_gain_pct:+.1%}")
        print(f"  Recommendation: {exit_plan.recommended_action} ({exit_plan.exit_urgency})")
        print(f"  Exit targets:")
        
        for i, target in enumerate(exit_plan.exit_targets[:3], 1):
            status = "✅ HIT" if exit_plan.current_gain_pct >= target.gain_percentage else "⏳ Pending"
            print(f"    {i}. {target.gain_percentage:.1%} gain → Sell {target.sell_percentage:.0%} ({status})")
    
    # Test overall report
    print(f"\n{exit_manager.generate_exit_report(test_positions)}")
    
    return exit_manager

if __name__ == "__main__":
    test_exit_strategy_system()
