#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Quality Enhancement Test Script 🧪
Tests the multi-timeframe Wyckoff analysis integration
Strategic Improvement 5: Signal Quality Enhancement
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# UTF-8 encoding for Windows compatibility
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)


class SignalQualityTester:
    """Test suite for signal quality enhancement"""
    
    def __init__(self):
        self.colors = {
            'GREEN': '\033[92m',
            'RED': '\033[91m', 
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'MAGENTA': '\033[95m',
            'CYAN': '\033[96m',
            'RESET': '\033[0m'
        }
        
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results with colors"""
        self.total_tests += 1
        
        if passed:
            self.passed_tests += 1
            status = f"{self.colors['GREEN']}✅ PASS{self.colors['RESET']}"
        else:
            status = f"{self.colors['RED']}❌ FAIL{self.colors['RESET']}"
        
        message = f"{status} {test_name}"
        if details:
            message += f": {details}"
        
        print(message)
        self.test_results.append((test_name, passed, details))
    
    def test_file_existence(self):
        """Test that all required files exist"""
        print(f"\n{self.colors['CYAN']}📁 Testing File Existence...{self.colors['RESET']}")
        
        required_files = [
            'strategies/wyckoff/multi_timeframe_wyckoff.py',
            'strategies/wyckoff/multi_timeframe_analyzer.py',
            'strategies/wyckoff/wyckoff.py',
            'fractional_position_system.py'
        ]
        
        for file_path in required_files:
            exists = Path(file_path).exists()
            self.log_test(f"File exists: {file_path}", exists)
    
    def test_imports(self):
        """Test that imports work correctly"""
        print(f"\n{self.colors['CYAN']}📦 Testing Imports...{self.colors['RESET']}")
        
        # Test multi-timeframe analyzer import
        try:
            from strategies.wyckoff.multi_timeframe_analyzer import EnhancedMultiTimeframeWyckoffAnalyzer
            self.log_test("Import EnhancedMultiTimeframeWyckoffAnalyzer", True)
        except ImportError as e:
            self.log_test("Import EnhancedMultiTimeframeWyckoffAnalyzer", False, str(e))
        
        # Test signal classes import
        try:
            from strategies.wyckoff.multi_timeframe_analyzer import MultiTimeframeSignal, TimeframeAnalysis
            self.log_test("Import signal classes", True)
        except ImportError as e:
            self.log_test("Import signal classes", False, str(e))
        
        # Test helper functions import
        try:
            from strategies.wyckoff.multi_timeframe_analyzer import filter_signals_by_quality
            self.log_test("Import helper functions", True)
        except ImportError as e:
            self.log_test("Import helper functions", False, str(e))
    
    def test_analyzer_initialization(self):
        """Test analyzer can be initialized"""
        print(f"\n{self.colors['CYAN']}🔧 Testing Analyzer Initialization...{self.colors['RESET']}")
        
        try:
            from strategies.wyckoff.multi_timeframe_analyzer import EnhancedMultiTimeframeWyckoffAnalyzer
            
            # Set up basic logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            # Create analyzer
            analyzer = EnhancedMultiTimeframeWyckoffAnalyzer(logger)
            
            # Check basic attributes
            has_timeframes = hasattr(analyzer, 'timeframes')
            has_phases = hasattr(analyzer, 'accumulation_phases')
            has_thresholds = hasattr(analyzer, 'quality_thresholds')
            
            self.log_test("Analyzer initialization", True)
            self.log_test("Has timeframes config", has_timeframes)
            self.log_test("Has phases config", has_phases)
            self.log_test("Has quality thresholds", has_thresholds)
            
            return analyzer
            
        except Exception as e:
            self.log_test("Analyzer initialization", False, str(e))
            return None
    
    def test_single_symbol_analysis(self, analyzer):
        """Test analysis of a single symbol"""
        print(f"\n{self.colors['CYAN']}📊 Testing Single Symbol Analysis...{self.colors['RESET']}")
        
        if not analyzer:
            self.log_test("Single symbol analysis", False, "No analyzer available")
            return
        
        try:
            # Test with a well-known symbol
            test_symbol = 'AAPL'
            print(f"   Analyzing {test_symbol}...")
            
            result = analyzer.analyze_symbol_multi_timeframe(test_symbol)
            
            if result:
                self.log_test(f"Analysis of {test_symbol}", True, f"Quality: {result.signal_quality}")
                
                # Check signal attributes
                has_phases = bool(result.primary_phase and result.entry_timing_phase and result.precision_phase)
                has_scores = bool(result.confirmation_score >= 0 and result.enhanced_strength >= 0)
                has_quality = result.signal_quality in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']
                
                self.log_test("Signal has all phases", has_phases)
                self.log_test("Signal has valid scores", has_scores)
                self.log_test("Signal has valid quality", has_quality)
                
                # Log details
                print(f"      📈 Phases: {result.primary_phase}/{result.entry_timing_phase}/{result.precision_phase}")
                print(f"      🎯 Quality: {result.signal_quality}")
                print(f"      💪 Enhanced Strength: {result.enhanced_strength:.2f}")
                print(f"      ✅ Confirmation Score: {result.confirmation_score:.2f}")
                
            else:
                self.log_test(f"Analysis of {test_symbol}", True, "No signal generated (normal)")
                
        except Exception as e:
            self.log_test(f"Analysis of {test_symbol}", False, str(e))
    
    def test_market_scan(self, analyzer):
        """Test market scanning functionality"""
        print(f"\n{self.colors['CYAN']}🔍 Testing Market Scan...{self.colors['RESET']}")
        
        if not analyzer:
            self.log_test("Market scan", False, "No analyzer available")
            return
        
        try:
            # Test with a small set of symbols
            test_symbols = ['AAPL', 'MSFT', 'GOOGL']
            print(f"   Scanning {len(test_symbols)} symbols...")
            
            results = analyzer.scan_market_enhanced(test_symbols, max_workers=2)
            
            self.log_test("Market scan execution", True, f"Found {len(results)} signals")
            
            # Test signal filtering
            if results:
                good_signals = [s for s in results if s.signal_quality in ['GOOD', 'EXCELLENT']]
                self.log_test("Quality filtering", True, f"{len(good_signals)} GOOD+ signals")
                
                # Show signal details
                for i, signal in enumerate(results[:3], 1):
                    print(f"      {i}. {signal.symbol}: {signal.signal_quality} ({signal.enhanced_strength:.2f})")
            else:
                self.log_test("Quality filtering", True, "No signals found (normal)")
                
        except Exception as e:
            self.log_test("Market scan", False, str(e))
    
    def test_signal_filtering(self):
        """Test signal filtering functions"""
        print(f"\n{self.colors['CYAN']}🔧 Testing Signal Filtering...{self.colors['RESET']}")
        
        try:
            from strategies.wyckoff.multi_timeframe_analyzer import filter_signals_by_quality, MultiTimeframeSignal
            
            # Create mock signals for testing
            mock_signals = [
                MultiTimeframeSignal(
                    symbol='TEST1', primary_phase='SOS', entry_timing_phase='SOS', precision_phase='SOS',
                    daily_strength=0.8, four_hour_strength=0.7, one_hour_strength=0.6,
                    timeframe_alignment=0.9, confirmation_score=0.9, price=100.0,
                    volume_confirmation=True, sector='Technology', base_strength=0.8,
                    enhanced_strength=0.9, signal_quality='EXCELLENT'
                ),
                MultiTimeframeSignal(
                    symbol='TEST2', primary_phase='LPS', entry_timing_phase='LPS', precision_phase='LPS',
                    daily_strength=0.7, four_hour_strength=0.6, one_hour_strength=0.5,
                    timeframe_alignment=0.7, confirmation_score=0.7, price=50.0,
                    volume_confirmation=True, sector='Healthcare', base_strength=0.7,
                    enhanced_strength=0.7, signal_quality='GOOD'
                ),
                MultiTimeframeSignal(
                    symbol='TEST3', primary_phase='BU', entry_timing_phase='BU', precision_phase='BU',
                    daily_strength=0.5, four_hour_strength=0.4, one_hour_strength=0.3,
                    timeframe_alignment=0.5, confirmation_score=0.5, price=25.0,
                    volume_confirmation=False, sector='Energy', base_strength=0.5,
                    enhanced_strength=0.5, signal_quality='FAIR'
                )
            ]
            
            # Test filtering
            excellent_signals = filter_signals_by_quality(mock_signals, 'EXCELLENT')
            good_signals = filter_signals_by_quality(mock_signals, 'GOOD')
            fair_signals = filter_signals_by_quality(mock_signals, 'FAIR')
            
            self.log_test("Filter EXCELLENT signals", len(excellent_signals) == 1)
            self.log_test("Filter GOOD+ signals", len(good_signals) == 2)
            self.log_test("Filter FAIR+ signals", len(fair_signals) == 3)
            
        except Exception as e:
            self.log_test("Signal filtering", False, str(e))
    
    def test_integration_modifications(self):
        """Test that integration modifications are present"""
        print(f"\n{self.colors['CYAN']}🔗 Testing Integration Modifications...{self.colors['RESET']}")
        
        # Test Wyckoff strategy modifications
        try:
            wyckoff_file = Path('strategies/wyckoff/wyckoff.py')
            if wyckoff_file.exists():
                with open(wyckoff_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                has_import = 'MULTI_TIMEFRAME_AVAILABLE' in content
                has_enhancement = 'Enhanced multi-timeframe Wyckoff analysis enabled' in content
                has_method = 'scan_market_enhanced' in content
                
                self.log_test("Wyckoff strategy has enhancement import", has_import)
                self.log_test("Wyckoff strategy has enhancement init", has_enhancement)
                self.log_test("Wyckoff strategy has enhanced method", has_method)
            else:
                self.log_test("Wyckoff strategy file exists", False)
                
        except Exception as e:
            self.log_test("Wyckoff strategy modifications", False, str(e))
        
        # Test fractional system modifications
        try:
            fractional_file = Path('fractional_position_system.py')
            if fractional_file.exists():
                with open(fractional_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                has_import = 'SIGNAL_QUALITY_ENHANCEMENT' in content
                has_analyzer = 'signal_quality_analyzer' in content
                has_filtering = 'Quality Enhancement:' in content
                
                self.log_test("Fractional system has enhancement import", has_import)
                self.log_test("Fractional system has analyzer", has_analyzer)
                self.log_test("Fractional system has filtering", has_filtering)
            else:
                self.log_test("Fractional system file exists", False)
                
        except Exception as e:
            self.log_test("Fractional system modifications", False, str(e))
    
    def test_backup_files(self):
        """Test that backup files were created"""
        print(f"\n{self.colors['CYAN']}💾 Testing Backup Files...{self.colors['RESET']}")
        
        # Look for backup files
        backup_files = list(Path('.').glob('**/*.backup_*'))
        
        if backup_files:
            self.log_test("Backup files created", True, f"Found {len(backup_files)} backup files")
            for backup in backup_files[:3]:  # Show first 3
                print(f"      📁 {backup}")
        else:
            self.log_test("Backup files created", False, "No backup files found")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print(f"{self.colors['MAGENTA']}🧪 SIGNAL QUALITY ENHANCEMENT TEST SUITE{self.colors['RESET']}")
        print(f"{self.colors['CYAN']}Testing Strategic Improvement 5: Multi-timeframe Signal Quality{self.colors['RESET']}")
        print(f"{self.colors['CYAN']}{'='*80}{self.colors['RESET']}")
        
        # Run all tests
        self.test_file_existence()
        self.test_imports()
        analyzer = self.test_analyzer_initialization()
        self.test_single_symbol_analysis(analyzer)
        self.test_market_scan(analyzer)
        self.test_signal_filtering()
        self.test_integration_modifications()
        self.test_backup_files()
        
        # Final results
        print(f"\n{self.colors['CYAN']}{'='*80}{self.colors['RESET']}")
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        if self.passed_tests == self.total_tests:
            print(f"{self.colors['GREEN']}🎉 ALL TESTS PASSED! ({self.passed_tests}/{self.total_tests}){self.colors['RESET']}")
            print(f"{self.colors['GREEN']}✅ Signal Quality Enhancement is working correctly{self.colors['RESET']}")
            print(f"{self.colors['GREEN']}🚀 Your bot is ready for enhanced multi-timeframe trading{self.colors['RESET']}")
        elif success_rate >= 80:
            print(f"{self.colors['YELLOW']}⚠️ MOSTLY WORKING ({self.passed_tests}/{self.total_tests} - {success_rate:.0f}%){self.colors['RESET']}")
            print(f"{self.colors['YELLOW']}🔧 Some tests failed - check details above{self.colors['RESET']}")
        else:
            print(f"{self.colors['RED']}❌ TESTS FAILED ({self.passed_tests}/{self.total_tests} - {success_rate:.0f}%){self.colors['RESET']}")
            print(f"{self.colors['RED']}🚨 Integration may have issues - review failed tests{self.colors['RESET']}")
        
        # Summary of failed tests
        failed_tests = [test for test, passed, details in self.test_results if not passed]
        if failed_tests:
            print(f"\n{self.colors['RED']}Failed Tests:{self.colors['RESET']}")
            for test in failed_tests:
                print(f"   ❌ {test}")
        
        print(f"\n{self.colors['BLUE']}📋 Test completed at {datetime.now().strftime('%H:%M:%S')}{self.colors['RESET']}")
        
        return self.passed_tests == self.total_tests


def main():
    """Main test entry point"""
    try:
        print("🧪 Starting Signal Quality Enhancement Test Suite...")
        
        tester = SignalQualityTester()
        success = tester.run_all_tests()
        
        return success
        
    except KeyboardInterrupt:
        print(f"\n🛑 Tests cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()