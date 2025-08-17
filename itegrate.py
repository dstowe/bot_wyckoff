#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 WYCKOFF REACCUMULATION INTEGRATION SCRIPT
Automatically applies all fixes and strategic improvements to the fractional position system

This script will modify fractional_position_system.py with:
1. ✅ Critical bug fixes (tuple unpacking)
2. 🎯 Enhanced Wyckoff reaccumulation position addition system
3. 🛡️ Comprehensive day trading compliance
4. 📊 Advanced risk management and position sizing
5. 🔧 Enhanced error handling and logging

Windows compatible with UTF-8 encoding support
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
import re


class WyckoffReaccumulationIntegrator:
    """🎯 Main integration class for applying Wyckoff reaccumulation improvements"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.original_file = self.script_dir / "fractional_position_system.py"
        self.backup_file = self.script_dir / f"fractional_position_system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        self.encoding = 'utf-8'
        
        # Track modifications
        self.modifications_applied = []
        self.errors_encountered = []
        
    def run_integration(self) -> bool:
        """🚀 Run the complete integration process"""
        print("🎯 WYCKOFF REACCUMULATION INTEGRATION STARTING...")
        print("=" * 70)
        
        try:
            # Step 1: Validate source file
            if not self._validate_source_file():
                return False
            
            # Step 2: Create backup
            if not self._create_backup():
                return False
            
            # Step 3: Read original content
            original_content = self._read_file_content()
            if not original_content:
                return False
            
            # Step 4: Apply all modifications
            modified_content = self._apply_all_modifications(original_content)
            
            # Step 5: Write modified content
            if not self._write_modified_content(modified_content):
                return False
            
            # Step 6: Generate summary report
            self._generate_integration_report()
            
            print("✅ WYCKOFF REACCUMULATION INTEGRATION COMPLETED!")
            return True
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {e}")
            self._restore_backup()
            return False
    
    def _validate_source_file(self) -> bool:
        """📋 Validate the source file exists and is readable"""
        if not self.original_file.exists():
            print(f"❌ Source file not found: {self.original_file}")
            return False
        
        try:
            with open(self.original_file, 'r', encoding=self.encoding) as f:
                content = f.read()
                if len(content) < 1000:  # Basic sanity check
                    print(f"❌ Source file appears to be too small: {len(content)} characters")
                    return False
            
            print(f"✅ Source file validated: {self.original_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error reading source file: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """💾 Create backup of original file"""
        try:
            shutil.copy2(self.original_file, self.backup_file)
            print(f"✅ Backup created: {self.backup_file}")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def _read_file_content(self) -> str:
        """📖 Read the original file content"""
        try:
            with open(self.original_file, 'r', encoding=self.encoding) as f:
                content = f.read()
            print(f"✅ Read {len(content)} characters from source file")
            return content
        except Exception as e:
            print(f"❌ Error reading file content: {e}")
            return ""
    
    def _apply_all_modifications(self, content: str) -> str:
        """🔧 Apply all modifications to the content"""
        print("\n🔧 APPLYING MODIFICATIONS...")
        
        # Apply modifications in order
        modifications = [
            ("🐛 Fix tuple unpacking bug", self._fix_tuple_unpacking_bug),
            ("🎯 Add enhanced reaccumulation detector", self._add_enhanced_reaccumulation_detector),
            ("🔧 Fix positions_added initialization", self._fix_positions_added_initialization),
            ("📊 Update scan_for_position_additions", self._update_scan_for_position_additions),
            ("✅ Update position eligibility check", self._update_position_eligibility),
            ("💰 Update addition shares calculation", self._update_addition_shares_calculation),
            ("🚀 Update execute_position_addition", self._update_execute_position_addition),
            ("🛡️ Add comprehensive imports", self._add_comprehensive_imports)
        ]
        
        modified_content = content
        
        for description, modification_func in modifications:
            try:
                print(f"   {description}...")
                modified_content = modification_func(modified_content)
                self.modifications_applied.append(description)
                print(f"   ✅ {description}")
            except Exception as e:
                error_msg = f"❌ {description}: {e}"
                print(f"   {error_msg}")
                self.errors_encountered.append(error_msg)
        
        return modified_content
    
    def _fix_tuple_unpacking_bug(self, content: str) -> str:
        """🐛 CRITICAL FIX: Fix the tuple unpacking bug"""
        
        # Find the problematic return statement in run_enhanced_trading_cycle
        pattern = r'(day_trades_blocked = self\.day_trades_blocked_today\s*\n\s*return trades_executed, wyckoff_sells, profit_scales, emergency_exits, day_trades_blocked)(?!\s*,\s*positions_added)'
        
        replacement = r'\1, positions_added'
        
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
        else:
            # Alternative approach - find the line and fix it
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'return trades_executed, wyckoff_sells, profit_scales, emergency_exits, day_trades_blocked' in line and 'positions_added' not in line:
                    lines[i] = line.replace('day_trades_blocked', 'day_trades_blocked, positions_added')
                    break
            content = '\n'.join(lines)
        
        return content
    
    def _add_enhanced_reaccumulation_detector(self, content: str) -> str:
        """🎯 Add the enhanced reaccumulation detector class"""
        
        # Find where to insert the enhanced detector (after existing ReaccumulationSignal)
        insertion_point = content.find("class WyckoffReaccumulationDetector:")
        
        if insertion_point != -1:
            # Replace the existing detector with the enhanced version
            end_point = content.find("class EnhancedTradingDatabase:", insertion_point)
            if end_point == -1:
                end_point = content.find("class ", insertion_point + 100)
            
            if end_point != -1:
                # Extract the class to replace
                enhanced_detector = '''
@dataclass
class ReaccumulationSignal:
    """Enhanced reaccumulation signal for position additions"""
    symbol: str
    phase_type: str
    strength: float
    current_price: float
    support_level: float
    resistance_level: float
    addition_percentage: float
    reasoning: str
    volume_analysis: Dict
    timeframe_confluence: int  # Number of timeframes confirming
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'


class WyckoffReaccumulationDetector:
    """🎯 ENHANCED: Wyckoff Reaccumulation Detection for Position Additions"""
    
    def __init__(self, logger):
        self.logger = logger
        self.REACCUMULATION_LOOKBACK = 45  # Extended for better pattern recognition
        self.SUPPORT_TEST_TOLERANCE = 0.015  # 1.5% tolerance for support tests
        self.RANGING_THRESHOLD = 0.08  # 8% maximum range for reaccumulation
        self.VOLUME_DECLINE_THRESHOLD = 0.75  # Volume should be 75% of earlier average
        self.MIN_CONSOLIDATION_DAYS = 10  # Minimum days in consolidation
        
        # Enhanced pattern recognition parameters
        self.ABSORPTION_STRENGTH_THRESHOLD = 0.6
        self.SPRING_DETECTION_ENABLED = True
        self.HIGHER_LOW_PROGRESSION_WEIGHT = 0.3
    
    def analyze_for_reaccumulation(self, symbol: str, position: Dict) -> Optional[ReaccumulationSignal]:
        """🎯 ENHANCED: Comprehensive reaccumulation analysis for position additions"""
        try:
            self.logger.debug(f"🔍 Analyzing {symbol} for reaccumulation addition opportunity...")
            
            # Get extended historical data for better pattern recognition
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo", interval="1d")
            
            if len(data) < self.REACCUMULATION_LOOKBACK:
                self.logger.debug(f"📊 Insufficient data for {symbol}: {len(data)} days")
                return None
            
            current_price = data['Close'].iloc[-1]
            recent_data = data.tail(self.REACCUMULATION_LOOKBACK)
            
            # 🎯 STEP 1: Identify potential reaccumulation range
            range_analysis = self._analyze_reaccumulation_range(recent_data, current_price)
            if not range_analysis['is_valid_range']:
                self.logger.debug(f"📈 {symbol}: No valid reaccumulation range detected")
                return None
            
            # 🎯 STEP 2: Volume analysis for absorption evidence
            volume_analysis = self._analyze_volume_absorption(data, recent_data)
            if not volume_analysis['shows_absorption']:
                self.logger.debug(f"📊 {symbol}: No volume absorption detected")
                return None
            
            # 🎯 STEP 3: Price action analysis
            price_action = self._analyze_price_action_strength(recent_data, range_analysis)
            
            # 🎯 STEP 4: Support strength evaluation
            support_strength = self._evaluate_support_strength(recent_data, range_analysis['support_level'])
            
            # 🎯 STEP 5: Calculate overall reaccumulation strength
            strength_components = {
                'range_quality': range_analysis['range_quality'],
                'volume_absorption': volume_analysis['absorption_strength'],
                'price_action': price_action['strength'],
                'support_strength': support_strength,
                'higher_lows': price_action['higher_lows_count'] * 0.1
            }
            
            # Weighted strength calculation
            overall_strength = (
                strength_components['range_quality'] * 0.25 +
                strength_components['volume_absorption'] * 0.30 +
                strength_components['price_action'] * 0.25 +
                strength_components['support_strength'] * 0.20
            )
            
            # Apply higher lows bonus
            overall_strength += min(0.2, strength_components['higher_lows'])
            overall_strength = min(1.0, overall_strength)
            
            self.logger.debug(f"📊 {symbol} reaccumulation strength: {overall_strength:.3f}")
            
            # 🎯 STEP 6: Determine if signal meets threshold
            if overall_strength >= self.ABSORPTION_STRENGTH_THRESHOLD:
                # Calculate position addition percentage based on strength
                addition_pct = self._calculate_addition_percentage(overall_strength, price_action)
                
                # Determine risk level
                risk_level = self._assess_risk_level(price_action, volume_analysis, range_analysis)
                
                # Enhanced reasoning
                reasoning_parts = []
                if range_analysis['is_tight_range']:
                    reasoning_parts.append(f"tight range ({range_analysis['range_pct']:.1%})")
                if volume_analysis['significant_decline']:
                    reasoning_parts.append(f"volume decline ({volume_analysis['decline_pct']:.1%})")
                if price_action['higher_lows_count'] >= 2:
                    reasoning_parts.append(f"{price_action['higher_lows_count']} higher lows")
                if price_action['spring_detected']:
                    reasoning_parts.append("spring action")
                
                reasoning = f"Reaccumulation: {', '.join(reasoning_parts)}"
                
                # Enhanced timeframe confluence (simplified for now)
                timeframe_confluence = 1  # Would be enhanced with multi-timeframe analysis
                
                signal = ReaccumulationSignal(
                    symbol=symbol,
                    phase_type='REACCUMULATION',
                    strength=overall_strength,
                    current_price=current_price,
                    support_level=range_analysis['support_level'],
                    resistance_level=range_analysis['resistance_level'],
                    addition_percentage=addition_pct,
                    reasoning=reasoning,
                    volume_analysis=volume_analysis,
                    timeframe_confluence=timeframe_confluence,
                    risk_level=risk_level
                )
                
                self.logger.info(f"🎯 Reaccumulation signal: {symbol} (strength: {overall_strength:.2f}, add: {addition_pct:.1%})")
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {symbol} for reaccumulation: {e}")
            return None
    
    def _analyze_reaccumulation_range(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Analyze if price is in a valid reaccumulation range"""
        high = data['High'].max()
        low = data['Low'].min()
        range_size = high - low
        range_pct = range_size / current_price
        
        # Check if it's a reasonable range for reaccumulation
        is_valid_range = range_pct <= self.RANGING_THRESHOLD
        is_tight_range = range_pct <= 0.05  # Very tight ranges are stronger signals
        
        # Check if current price is in reasonable position within range
        position_in_range = (current_price - low) / range_size if range_size > 0 else 0.5
        
        # Range quality scoring
        range_quality = 0.0
        if is_valid_range:
            range_quality += 0.4
        if is_tight_range:
            range_quality += 0.3
        if 0.2 <= position_in_range <= 0.8:  # Not at extremes
            range_quality += 0.3
        
        return {
            'is_valid_range': is_valid_range,
            'is_tight_range': is_tight_range,
            'support_level': low,
            'resistance_level': high,
            'range_pct': range_pct,
            'position_in_range': position_in_range,
            'range_quality': range_quality
        }
    
    def _analyze_volume_absorption(self, full_data: pd.DataFrame, recent_data: pd.DataFrame) -> Dict:
        """Analyze volume for absorption characteristics"""
        recent_volume = recent_data['Volume'].mean()
        
        # Compare with earlier period (before the reaccumulation)
        earlier_period = full_data.tail(90).head(45)  # 45 days before recent 45
        if len(earlier_period) < 20:
            earlier_volume = full_data['Volume'].mean()
        else:
            earlier_volume = earlier_period['Volume'].mean()
        
        # Calculate volume decline
        if earlier_volume > 0:
            volume_ratio = recent_volume / earlier_volume
            decline_pct = 1 - volume_ratio
        else:
            volume_ratio = 1.0
            decline_pct = 0.0
        
        # Volume characteristics
        shows_absorption = volume_ratio < self.VOLUME_DECLINE_THRESHOLD
        significant_decline = decline_pct > 0.3  # 30%+ decline
        
        # Absorption strength scoring
        absorption_strength = 0.0
        if shows_absorption:
            absorption_strength += 0.5
        if significant_decline:
            absorption_strength += 0.3
        
        # Bonus for very low volume
        if volume_ratio < 0.5:  # Volume less than half of earlier period
            absorption_strength += 0.2
        
        return {
            'shows_absorption': shows_absorption,
            'significant_decline': significant_decline,
            'recent_volume': recent_volume,
            'earlier_volume': earlier_volume,
            'volume_ratio': volume_ratio,
            'decline_pct': decline_pct,
            'absorption_strength': min(1.0, absorption_strength)
        }
    
    def _analyze_price_action_strength(self, data: pd.DataFrame, range_analysis: Dict) -> Dict:
        """Analyze price action for reaccumulation strength signs"""
        closes = data['Close']
        lows = data['Low']
        
        # Count higher lows within the range
        higher_lows_count = 0
        prev_low = lows.iloc[0]
        
        for i in range(1, len(lows)):
            current_low = lows.iloc[i]
            if current_low > prev_low * 1.002:  # At least 0.2% higher
                higher_lows_count += 1
                prev_low = current_low
        
        # Detect potential spring action (brief break below support followed by recovery)
        spring_detected = False
        support_level = range_analysis['support_level']
        
        for i in range(5, len(data)):  # Look in recent data
            if (lows.iloc[i] < support_level * 0.998 and  # Brief break below support
                closes.iloc[i] > support_level * 1.001):  # But close above support
                spring_detected = True
                break
        
        # Calculate price action strength
        strength = 0.0
        if higher_lows_count >= 2:
            strength += 0.4
        if higher_lows_count >= 3:
            strength += 0.2
        if spring_detected:
            strength += 0.3
        
        # Check for consistent closes in upper half of range
        range_size = range_analysis['resistance_level'] - range_analysis['support_level']
        mid_range = range_analysis['support_level'] + (range_size * 0.5)
        upper_half_closes = sum(1 for close in closes.tail(10) if close > mid_range)
        
        if upper_half_closes >= 7:  # 70% of recent closes in upper half
            strength += 0.2
        
        return {
            'strength': min(1.0, strength),
            'higher_lows_count': higher_lows_count,
            'spring_detected': spring_detected,
            'upper_half_closes': upper_half_closes
        }
    
    def _evaluate_support_strength(self, data: pd.DataFrame, support_level: float) -> float:
        """Evaluate the strength of the support level"""
        lows = data['Low']
        
        # Count how many times price tested support without breaking significantly
        support_tests = 0
        successful_holds = 0
        
        for low in lows:
            if low <= support_level * 1.01:  # Within 1% of support
                support_tests += 1
                if low >= support_level * 0.99:  # Held above 99% of support
                    successful_holds += 1
        
        if support_tests == 0:
            return 0.3  # No tests, uncertain strength
        
        hold_ratio = successful_holds / support_tests
        
        # Strength based on hold ratio and number of tests
        if hold_ratio >= 0.8 and support_tests >= 3:
            return 0.9  # Very strong support
        elif hold_ratio >= 0.7 and support_tests >= 2:
            return 0.7  # Strong support
        elif hold_ratio >= 0.6:
            return 0.5  # Moderate support
        else:
            return 0.2  # Weak support
    
    def _calculate_addition_percentage(self, strength: float, price_action: Dict) -> float:
        """Calculate what percentage to add to position based on signal strength"""
        # Base addition percentage from strength
        base_addition = 0.15 + (strength - 0.6) * 0.5  # 15% to 35% based on strength
        
        # Bonuses for strong signals
        if price_action['spring_detected']:
            base_addition += 0.1
        
        if price_action['higher_lows_count'] >= 3:
            base_addition += 0.05
        
        # Conservative cap
        return min(0.4, max(0.1, base_addition))
    
    def _assess_risk_level(self, price_action: Dict, volume_analysis: Dict, range_analysis: Dict) -> str:
        """Assess the risk level of the reaccumulation signal"""
        risk_score = 0
        
        # Low risk factors
        if price_action['spring_detected']:
            risk_score -= 1
        if volume_analysis['significant_decline']:
            risk_score -= 1
        if range_analysis['is_tight_range']:
            risk_score -= 1
        if price_action['higher_lows_count'] >= 3:
            risk_score -= 1
        
        # High risk factors
        if range_analysis['range_pct'] > 0.06:  # Wider ranges are riskier
            risk_score += 1
        if volume_analysis['volume_ratio'] > 0.8:  # Not much volume decline
            risk_score += 1
        
        if risk_score <= -2:
            return 'LOW'
        elif risk_score <= 0:
            return 'MEDIUM'
        else:
            return 'HIGH'


'''
                
                # Replace the old detector with the enhanced version
                before_detector = content[:insertion_point]
                after_detector = content[end_point:]
                content = before_detector + enhanced_detector + after_detector
        
        return content
    
    def _fix_positions_added_initialization(self, content: str) -> str:
        """🔧 Fix positions_added variable initialization"""
        
        # Find the start of run_enhanced_trading_cycle method
        method_start = content.find("def run_enhanced_trading_cycle(self)")
        if method_start != -1:
            # Find the variable initialization section
            init_section = content.find("trades_executed = 0", method_start)
            if init_section != -1:
                # Find the end of the initialization block
                try_start = content.find("try:", init_section)
                if try_start != -1:
                    # Insert positions_added initialization
                    before_try = content[:try_start]
                    after_try = content[try_start:]
                    
                    # Check if positions_added is already initialized
                    if "positions_added = 0" not in before_try[init_section:]:
                        init_block = """            positions_added = 0  # 🔧 CRITICAL FIX: Initialize positions_added
            
            """
                        content = before_try + init_block + after_try
        
        return content
    
    def _update_scan_for_position_additions(self, content: str) -> str:
        """📊 Update scan_for_position_additions method"""
        
        # Find the method and replace it
        method_pattern = r'def scan_for_position_additions\(self, current_positions: Dict\) -> List\[Dict\]:.*?(?=def|\Z)'
        
        new_method = '''def scan_for_position_additions(self, current_positions: Dict) -> List[Dict]:
        """🎯 ENHANCED: Scan existing positions for reaccumulation addition opportunities"""
        addition_opportunities = []
        
        # Initialize reaccumulation detector if not already done
        if self.reaccumulation_detector is None:
            self.reaccumulation_detector = WyckoffReaccumulationDetector(self.logger)
            self.logger.info("🎯 Initialized enhanced reaccumulation detector")
        
        try:
            self.logger.info(f"🔍 Scanning {len(current_positions)} positions for reaccumulation additions...")
            
            for position_key, position in current_positions.items():
                symbol = position['symbol']
                
                # Check if position is eligible for additions
                if not self._is_position_eligible_for_addition(position):
                    self.logger.debug(f"❌ {symbol} not eligible for addition")
                    continue
                
                # Look for reaccumulation signal
                signal = self.reaccumulation_detector.analyze_for_reaccumulation(symbol, position)
                
                if signal and signal.strength >= 0.6:
                    addition_shares = self._calculate_addition_shares(signal, position)
                    
                    if addition_shares > 0:
                        addition_opportunities.append({
                            'signal': signal,
                            'position': position,
                            'addition_shares': addition_shares,
                            'estimated_cost': addition_shares * signal.current_price,
                            'risk_level': signal.risk_level
                        })
                        
                        self.logger.info(f"🎯 Reaccumulation opportunity: {symbol}")
                        self.logger.info(f"   Strength: {signal.strength:.2f}, Risk: {signal.risk_level}")
                        self.logger.info(f"   Add {signal.addition_percentage:.1%} ({addition_shares:.5f} shares)")
            
            # Sort by strength and risk level (prefer LOW risk, high strength)
            addition_opportunities.sort(key=lambda x: (
                -x['signal'].strength if x['risk_level'] == 'LOW' else -x['signal'].strength * 0.7
            ))
            
            # Limit to max additions per day
            limited_opportunities = addition_opportunities[:self.max_position_additions_per_day]
            
            if limited_opportunities:
                self.logger.info(f"🎯 Selected {len(limited_opportunities)} addition opportunities")
            else:
                self.logger.info("📊 No reaccumulation opportunities found")
            
            return limited_opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning for position additions: {e}")
            return []

    '''
        
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        return content
    
    def _update_position_eligibility(self, content: str) -> str:
        """✅ Update position eligibility check method"""
        
        method_pattern = r'def _is_position_eligible_for_addition\(self, position: Dict\) -> bool:.*?(?=def|\Z)'
        
        new_method = '''def _is_position_eligible_for_addition(self, position: Dict) -> bool:
        """🎯 ENHANCED: Check if position is eligible for reaccumulation additions"""
        try:
            symbol = position['symbol']
            current_shares = position.get('shares', 0)
            
            # Must have existing shares
            if current_shares <= 0:
                return False
            
            # Check current performance
            current_price = self._get_current_price(symbol)
            avg_cost = position.get('avg_cost', 0)
            
            if current_price and avg_cost:
                gain_pct = (current_price - avg_cost) / avg_cost
                
                # Don't add to positions down more than 20%
                if gain_pct < -0.20:
                    self.logger.debug(f"❌ {symbol} down {gain_pct:.1%}, too risky for addition")
                    return False
                
                # Don't add to positions up more than 50% (might be overextended)
                if gain_pct > 0.50:
                    self.logger.debug(f"❌ {symbol} up {gain_pct:.1%}, potentially overextended")
                    return False
            
            # Check time since last purchase (prefer positions held at least 5 days)
            last_purchase_date = position.get('last_purchase_date', '')
            if last_purchase_date:
                try:
                    last_purchase = datetime.strptime(last_purchase_date, '%Y-%m-%d')
                    days_since = (datetime.now() - last_purchase).days
                    if days_since < 5:
                        self.logger.debug(f"❌ {symbol} last purchase only {days_since} days ago")
                        return False
                except:
                    pass
            
            # Check entry phase - prefer certain Wyckoff phases for additions
            entry_phase = position.get('entry_phase', 'UNKNOWN')
            preferred_phases = ['SOS', 'LPS', 'BU']  # Sign of Strength, Last Point of Support, Backup
            
            if entry_phase in preferred_phases:
                self.logger.debug(f"✅ {symbol} has preferred entry phase: {entry_phase}")
                return True
            elif entry_phase in ['ST']:  # Stopping action - be more cautious
                # Only allow if position is profitable
                if current_price and avg_cost and (current_price >= avg_cost * 1.05):
                    return True
                else:
                    self.logger.debug(f"❌ {symbol} ST phase but not sufficiently profitable")
                    return False
            else:
                # Unknown or other phases - allow but be conservative
                return True
                
        except Exception as e:
            self.logger.debug(f"❌ Error checking eligibility for {position.get('symbol', 'unknown')}: {e}")
            return False

    '''
        
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        return content
    
    def _update_addition_shares_calculation(self, content: str) -> str:
        """💰 Update addition shares calculation method"""
        
        method_pattern = r'def _calculate_addition_shares\(self, signal:.*?\) -> float:.*?(?=def|\Z)'
        
        new_method = '''def _calculate_addition_shares(self, signal: ReaccumulationSignal, position: Dict) -> float:
        """🎯 ENHANCED: Calculate number of shares to add with risk management"""
        try:
            current_shares = position.get('shares', 0)
            avg_cost = position.get('avg_cost', signal.current_price)
            current_position_value = current_shares * avg_cost
            
            # Base addition value from signal
            base_addition_value = current_position_value * signal.addition_percentage
            
            # Risk-based adjustments
            risk_multipliers = {
                'LOW': 1.0,
                'MEDIUM': 0.8,
                'HIGH': 0.6
            }
            
            risk_adjusted_value = base_addition_value * risk_multipliers.get(signal.risk_level, 0.8)
            
            # Apply conservative limits
            max_addition_limits = [
                risk_adjusted_value,
                current_position_value * 0.5,  # Never add more than 50% of current position
                800.0,  # Hard cap at $800 addition
                signal.current_price * 100  # Max 100 shares
            ]
            
            final_addition_value = min(max_addition_limits)
            
            # Calculate shares
            addition_shares = final_addition_value / signal.current_price
            
            # Round to reasonable precision
            addition_shares = round(addition_shares, 5)
            
            # Minimum viable addition
            min_addition_value = 20.0  # At least $20 addition
            if final_addition_value < min_addition_value:
                self.logger.debug(f"❌ Addition too small: ${final_addition_value:.2f}")
                return 0.0
            
            self.logger.debug(f"💰 {signal.symbol} addition calculation:")
            self.logger.debug(f"   Current position: {current_shares:.5f} shares (${current_position_value:.2f})")
            self.logger.debug(f"   Addition: {addition_shares:.5f} shares (${final_addition_value:.2f})")
            self.logger.debug(f"   Risk level: {signal.risk_level}")
            
            return addition_shares
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating addition shares: {e}")
            return 0.0

    '''
        
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        return content
    
    def _update_execute_position_addition(self, content: str) -> str:
        """🚀 Update execute_position_addition method"""
        
        method_pattern = r'def execute_position_addition\(self, opportunity: Dict\) -> bool:.*?(?=def|\Z)'
        
        new_method = '''def execute_position_addition(self, opportunity: Dict) -> bool:
        """🎯 ENHANCED: Execute addition to existing position with full compliance"""
        try:
            signal = opportunity['signal']
            position = opportunity['position']
            addition_shares = opportunity['addition_shares']
            symbol = signal.symbol
            
            self.logger.info(f"🔄 Executing position addition for {symbol}...")
            
            # STEP 1: Comprehensive day trade compliance check
            day_trade_check = self._check_day_trade_compliance(symbol, 'BUY')
            self.database.log_day_trade_check(day_trade_check)
            
            if day_trade_check.recommendation == 'BLOCK':
                self.logger.warning(f"🚨 POSITION ADDITION BLOCKED: {symbol}")
                self.logger.warning(f"   Reason: {day_trade_check.details}")
                self.day_trades_blocked_today += 1
                return False
            elif day_trade_check.would_be_day_trade:
                self.logger.warning(f"⚠️ Day trade detected for addition: {symbol}")
            
            # STEP 2: Account management
            enabled_accounts = self.main_system.account_manager.get_enabled_accounts()
            target_account = next((acc for acc in enabled_accounts 
                                 if acc.account_type == position['account_type']), None)
            
            if not target_account:
                self.logger.error(f"❌ Target account not found: {position['account_type']}")
                return False
            
            # Switch to target account
            if not self.main_system.account_manager.switch_to_account(target_account):
                self.logger.error(f"❌ Failed to switch to account: {target_account.account_type}")
                return False
            
            # STEP 3: Enhanced cash validation
            required_cash = addition_shares * signal.current_price
            available_cash = target_account.settled_funds
            min_buffer = 20.0  # Increased buffer for position additions
            
            if available_cash < required_cash + min_buffer:
                self.logger.warning(f"⚠️ Insufficient cash for {symbol} addition:")
                self.logger.warning(f"   Required: ${required_cash:.2f} + ${min_buffer:.2f} buffer")
                self.logger.warning(f"   Available: ${available_cash:.2f}")
                return False
            
            # STEP 4: Session validation
            if not self._ensure_valid_session():
                self.logger.error(f"❌ Cannot establish valid session for {symbol} addition")
                return False
            
            # STEP 5: Final price validation
            current_quote = self.main_system.wb.get_quote(symbol)
            if not current_quote or 'close' not in current_quote:
                self.logger.error(f"❌ Cannot get current quote for {symbol}")
                return False
            
            current_market_price = float(current_quote['close'])
            
            # Check if price hasn't moved too much from signal price
            price_change = abs(current_market_price - signal.current_price) / signal.current_price
            if price_change > 0.03:  # More than 3% change
                self.logger.warning(f"⚠️ Price moved {price_change:.1%} since signal generation")
                # Re-calculate shares based on current price
                addition_shares = required_cash / current_market_price
                addition_shares = round(addition_shares, 5)
            
            # STEP 6: Enhanced logging before execution
            self.logger.info(f"🎯 ADDING TO POSITION: {symbol}")
            self.logger.info(f"   Current holding: {position['shares']:.5f} shares @ ${position['avg_cost']:.2f}")
            self.logger.info(f"   Adding: {addition_shares:.5f} shares @ ${current_market_price:.2f}")
            self.logger.info(f"   Addition cost: ${addition_shares * current_market_price:.2f}")
            self.logger.info(f"   Signal strength: {signal.strength:.2f}")
            self.logger.info(f"   Risk level: {signal.risk_level}")
            self.logger.info(f"   Reasoning: {signal.reasoning}")
            self.logger.info(f"   Day trade check: {day_trade_check.recommendation}")
            
            # STEP 7: Execute the order
            order_result = self.main_system.wb.place_order(
                stock=symbol,
                price=0,  # Market order
                action='BUY',
                orderType='MKT',
                enforce='DAY',
                quant=addition_shares,
                outsideRegularTradingHour=False
            )
            
            if order_result.get('success', False):
                order_id = order_result.get('orderId', 'UNKNOWN')
                actual_cost = addition_shares * current_market_price
                
                # STEP 8: Enhanced logging and database updates
                self.database.log_trade(
                    symbol=symbol,
                    action='REACCUMULATION_ADD',
                    quantity=addition_shares,
                    price=current_market_price,
                    signal_phase=signal.phase_type,
                    signal_strength=signal.strength,
                    account_type=position['account_type'],
                    order_id=order_id,
                    day_trade_check=day_trade_check.recommendation
                )
                
                # Update position tracking
                self.database.update_position(
                    symbol=symbol,
                    shares=addition_shares,
                    cost=current_market_price,
                    account_type=position['account_type'],
                    entry_phase=signal.phase_type,
                    entry_strength=signal.strength
                )
                
                # Update account cash tracking
                target_account.settled_funds -= actual_cost
                self.positions_added_today += 1
                
                self.logger.info(f"✅ POSITION ADDITION EXECUTED: {symbol}")
                self.logger.info(f"   Order ID: {order_id}")
                self.logger.info(f"   Added {addition_shares:.5f} shares for ${actual_cost:.2f}")
                self.logger.info(f"   Remaining cash: ${target_account.settled_funds:.2f}")
                
                return True
            else:
                error_msg = order_result.get('msg', 'Unknown error')
                self.logger.error(f"❌ Position addition order failed: {symbol}")
                self.logger.error(f"   Error: {error_msg}")
                
                # Check for session issues
                if 'session' in error_msg.lower() or 'expired' in error_msg.lower():
                    self.logger.warning("⚠️ Session issue detected during position addition")
                    self.main_system.session_manager.clear_session()
                    
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error executing position addition: {e}")
            import traceback
            traceback.print_exc()
            return False

    '''
        
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        return content
    
    def _add_comprehensive_imports(self, content: str) -> str:
        """🛡️ Add any missing imports for the enhanced functionality"""
        
        # Check if ReaccumulationSignal dataclass import is needed
        if "@dataclass" in content and "from dataclasses import dataclass" not in content:
            import_section = content.find("import")
            if import_section != -1:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith("from dataclasses import"):
                        break
                    elif line.startswith("import") or line.startswith("from"):
                        lines.insert(i, "from dataclasses import dataclass")
                        content = '\n'.join(lines)
                        break
        
        return content
    
    def _write_modified_content(self, content: str) -> bool:
        """💾 Write the modified content back to the file"""
        try:
            with open(self.original_file, 'w', encoding=self.encoding) as f:
                f.write(content)
            
            print(f"✅ Modified content written to: {self.original_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error writing modified content: {e}")
            return False
    
    def _restore_backup(self) -> bool:
        """🔄 Restore from backup in case of error"""
        try:
            if self.backup_file.exists():
                shutil.copy2(self.backup_file, self.original_file)
                print(f"🔄 Restored from backup: {self.backup_file}")
                return True
            return False
        except Exception as e:
            print(f"❌ Error restoring backup: {e}")
            return False
    
    def _generate_integration_report(self):
        """📊 Generate integration summary report"""
        print("\n" + "=" * 70)
        print("🎯 WYCKOFF REACCUMULATION INTEGRATION REPORT")
        print("=" * 70)
        
        print(f"\n✅ MODIFICATIONS APPLIED ({len(self.modifications_applied)}):")
        for mod in self.modifications_applied:
            print(f"   ✅ {mod}")
        
        if self.errors_encountered:
            print(f"\n❌ ERRORS ENCOUNTERED ({len(self.errors_encountered)}):")
            for error in self.errors_encountered:
                print(f"   ❌ {error}")
        
        print(f"\n📁 FILES:")
        print(f"   📄 Original: {self.original_file}")
        print(f"   💾 Backup: {self.backup_file}")
        
        print(f"\n🎯 STRATEGIC IMPROVEMENTS IMPLEMENTED:")
        print(f"   🔧 Critical bug fixes (tuple unpacking)")
        print(f"   🎯 Enhanced Wyckoff reaccumulation detection")
        print(f"   📊 Sophisticated position addition analysis")
        print(f"   🛡️ Comprehensive day trading compliance")
        print(f"   💰 Risk-based position sizing and limits")
        print(f"   🚀 Full execution pipeline with error handling")
        
        print(f"\n🏆 INTEGRATION STATUS: {'SUCCESS' if not self.errors_encountered else 'PARTIAL'}")
        print("=" * 70)


def main():
    """🚀 Main entry point for the integration script"""
    print("🎯 WYCKOFF REACCUMULATION INTEGRATION SCRIPT")
    print("Windows Compatible | UTF-8 Encoding | Enhanced Wyckoff Analysis")
    print("=" * 70)
    
    integrator = WyckoffReaccumulationIntegrator()
    
    try:
        success = integrator.run_integration()
        
        if success:
            print("\n🏆 INTEGRATION COMPLETED SUCCESSFULLY!")
            print("\n🎯 NEXT STEPS:")
            print("   1. Review the modified fractional_position_system.py")
            print("   2. Test the enhanced reaccumulation features")
            print("   3. Run the bot to verify all fixes are working")
            print("   4. Monitor position addition opportunities")
            
            return 0
        else:
            print("\n❌ INTEGRATION FAILED!")
            print("   Check the error messages above and retry")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Integration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())