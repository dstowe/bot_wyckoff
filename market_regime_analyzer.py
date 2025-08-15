#!/usr/bin/env python3
"""
Enhanced Market Regime Analyzer - Optimization 2
Comprehensive market regime detection including trend, volatility, and sector analysis
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
class MarketRegimeData:
    """Complete market regime analysis results"""
    trend_regime: str          # 'BULL', 'BEAR', 'RANGING'
    trend_strength: float      # 0.0 to 1.0
    volatility_regime: str     # 'LOW', 'MEDIUM', 'HIGH', 'CRISIS'
    volatility_level: float    # Actual VIX level
    sector_leaders: List[Tuple[str, float]]  # [(sector, relative_strength), ...]
    sector_weights: Dict[str, float]         # {sector: weight_multiplier, ...}
    position_size_multiplier: float         # Overall position sizing adjustment
    regime_confidence: float               # 0.0 to 1.0 confidence in regime detection
    last_updated: datetime
    raw_data: Dict                        # Raw market data for debugging


class EnhancedMarketRegimeAnalyzer:
    """
    Enhanced Market Regime Analyzer - Goes beyond simple VIX thresholds
    Analyzes trend regimes, volatility regimes, and sector rotation patterns
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Trend regime parameters
        self.TREND_MA_FAST = 50    # Fast moving average for trend
        self.TREND_MA_SLOW = 200   # Slow moving average for trend
        self.RANGING_THRESHOLD = 0.02  # 2% threshold for ranging market
        
        # Volatility regime thresholds (VIX levels)
        self.VIX_LOW = 20.0
        self.VIX_MEDIUM = 25.0  
        self.VIX_HIGH = 40.0
        self.VIX_CRISIS = 50.0
        
        # Sector rotation parameters
        self.SECTOR_LOOKBACK_DAYS = 20
        self.TOP_SECTOR_THRESHOLD = 0.75  # Top 75th percentile
        self.SECTOR_WEIGHT_BOOST = 1.5    # 1.5x allocation for strong sectors
        
        # Position sizing adjustments
        self.BEAR_MARKET_REDUCTION = 0.5   # 50% reduction in bear markets
        self.HIGH_VOL_SCALING_BASE = 30.0  # Scale by (30/VIX) when VIX > 20
        
        # Market indices for trend analysis
        self.MARKET_INDICES = {
            'SPY': '^GSPC',    # S&P 500
            'QQQ': '^IXIC',    # NASDAQ
            'IWM': '^RUT'      # Russell 2000
        }
        
        # Sector ETFs for rotation analysis
        self.SECTOR_ETFS = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Consumer_Discretionary': 'XLY',
            'Consumer_Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real_Estate': 'XLRE',
            'Materials': 'XLB',
            'Industrials': 'XLI',
            'Communication': 'XLC'
        }
        
        # Cache for market data (refresh every 30 minutes)
        self._cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=30)
    
    def analyze_market_regime(self) -> MarketRegimeData:
        """
        Comprehensive market regime analysis
        Returns complete regime data including trend, volatility, and sector analysis
        """
        try:
            self.logger.info("🔍 Analyzing comprehensive market regime...")
            
            # Step 1: Analyze trend regime
            trend_regime, trend_strength = self._analyze_trend_regime()
            
            # Step 2: Analyze volatility regime  
            volatility_regime, vix_level = self._analyze_volatility_regime()
            
            # Step 3: Analyze sector rotation
            sector_leaders, sector_weights = self._analyze_sector_rotation()
            
            # Step 4: Calculate overall position sizing adjustment
            position_multiplier = self._calculate_position_multiplier(
                trend_regime, volatility_regime, vix_level
            )
            
            # Step 5: Calculate regime confidence
            confidence = self._calculate_regime_confidence(
                trend_regime, trend_strength, volatility_regime, vix_level
            )
            
            regime_data = MarketRegimeData(
                trend_regime=trend_regime,
                trend_strength=trend_strength,
                volatility_regime=volatility_regime,
                volatility_level=vix_level,
                sector_leaders=sector_leaders,
                sector_weights=sector_weights,
                position_size_multiplier=position_multiplier,
                regime_confidence=confidence,
                last_updated=datetime.now(),
                raw_data=self._get_raw_debug_data()
            )
            
            self._log_regime_summary(regime_data)
            return regime_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in market regime analysis: {e}")
            return self._get_fallback_regime_data()
    
    def _analyze_trend_regime(self) -> Tuple[str, float]:
        """Analyze overall market trend regime using multiple indices"""
        try:
            trend_scores = []
            
            for index_name, symbol in self.MARKET_INDICES.items():
                try:
                    # Get market data
                    data = self._get_market_data(symbol, period='1y')
                    if data is None or len(data) < self.TREND_MA_SLOW:
                        continue
                    
                    # Calculate moving averages
                    fast_ma = data['Close'].rolling(self.TREND_MA_FAST).mean()
                    slow_ma = data['Close'].rolling(self.TREND_MA_SLOW).mean()
                    
                    current_price = data['Close'].iloc[-1]
                    current_fast_ma = fast_ma.iloc[-1] 
                    current_slow_ma = slow_ma.iloc[-1]
                    
                    # Calculate trend strength
                    if current_price > current_fast_ma > current_slow_ma:
                        # Strong bull trend
                        trend_strength = min(1.0, (current_price - current_slow_ma) / current_slow_ma / 0.2)
                        trend_scores.append(('BULL', trend_strength))
                    elif current_price < current_fast_ma < current_slow_ma:
                        # Strong bear trend  
                        trend_strength = min(1.0, (current_slow_ma - current_price) / current_slow_ma / 0.2)
                        trend_scores.append(('BEAR', trend_strength))
                    else:
                        # Check for ranging market
                        price_range = (data['High'].tail(50).max() - data['Low'].tail(50).min()) / current_price
                        if price_range < self.RANGING_THRESHOLD:
                            trend_scores.append(('RANGING', 0.5))
                        else:
                            # Mixed signals - weak trend
                            weak_strength = 0.3
                            if current_price > current_slow_ma:
                                trend_scores.append(('BULL', weak_strength))
                            else:
                                trend_scores.append(('BEAR', weak_strength))
                    
                    self.logger.debug(f"   {index_name}: {trend_scores[-1]}")
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing {index_name}: {e}")
                    continue
            
            if not trend_scores:
                return 'RANGING', 0.5
            
            # Aggregate trend scores
            bull_votes = [score for regime, score in trend_scores if regime == 'BULL']
            bear_votes = [score for regime, score in trend_scores if regime == 'BEAR']
            ranging_votes = [score for regime, score in trend_scores if regime == 'RANGING']
            
            bull_strength = sum(bull_votes) / len(trend_scores)
            bear_strength = sum(bear_votes) / len(trend_scores) 
            ranging_strength = sum(ranging_votes) / len(trend_scores)
            
            # Determine overall regime
            if bull_strength > bear_strength and bull_strength > ranging_strength:
                return 'BULL', bull_strength
            elif bear_strength > bull_strength and bear_strength > ranging_strength:
                return 'BEAR', bear_strength
            else:
                return 'RANGING', ranging_strength
                
        except Exception as e:
            self.logger.error(f"Error in trend regime analysis: {e}")
            return 'RANGING', 0.5
    
    def _analyze_volatility_regime(self) -> Tuple[str, float]:
        """Analyze volatility regime using VIX and market volatility"""
        try:
            # Get VIX data
            vix_data = self._get_market_data('^VIX', period='6mo')
            if vix_data is None or len(vix_data) < 20:
                return 'MEDIUM', 25.0
            
            current_vix = float(vix_data['Close'].iloc[-1])
            avg_vix_20d = float(vix_data['Close'].tail(20).mean())
            
            self.logger.debug(f"   Current VIX: {current_vix:.1f}, 20d avg: {avg_vix_20d:.1f}")
            
            # Determine volatility regime
            if current_vix >= self.VIX_CRISIS:
                return 'CRISIS', current_vix
            elif current_vix >= self.VIX_HIGH:
                return 'HIGH', current_vix
            elif current_vix >= self.VIX_MEDIUM:
                return 'MEDIUM', current_vix
            else:
                return 'LOW', current_vix
                
        except Exception as e:
            self.logger.error(f"Error in volatility regime analysis: {e}")
            return 'MEDIUM', 25.0
    
    def _analyze_sector_rotation(self) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        """Analyze sector rotation and determine sector weights"""
        try:
            sector_performance = {}
            
            # Get SPY performance for relative comparison
            spy_data = self._get_market_data('SPY', period='6mo')
            if spy_data is None:
                return [], {}
            
            spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-self.SECTOR_LOOKBACK_DAYS] - 1) * 100
            
            # Analyze each sector ETF
            for sector_name, etf_symbol in self.SECTOR_ETFS.items():
                try:
                    sector_data = self._get_market_data(etf_symbol, period='6mo')
                    if sector_data is None or len(sector_data) < self.SECTOR_LOOKBACK_DAYS:
                        continue
                    
                    # Calculate sector relative performance vs SPY
                    sector_return = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-self.SECTOR_LOOKBACK_DAYS] - 1) * 100
                    relative_performance = sector_return - spy_return
                    
                    sector_performance[sector_name] = relative_performance
                    self.logger.debug(f"   {sector_name}: {relative_performance:+.1f}% vs SPY")
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing sector {sector_name}: {e}")
                    continue
            
            if not sector_performance:
                return [], {}
            
            # Sort sectors by performance
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate sector weights
            sector_weights = {}
            total_sectors = len(sorted_sectors)
            top_quartile_count = max(1, total_sectors // 4)
            
            for i, (sector, performance) in enumerate(sorted_sectors):
                if i < top_quartile_count and performance > 0:
                    # Top quartile performing sectors get boost
                    sector_weights[sector] = self.SECTOR_WEIGHT_BOOST
                elif performance > 0:
                    # Positive performing sectors get slight boost
                    sector_weights[sector] = 1.1
                else:
                    # Underperforming sectors get reduction
                    sector_weights[sector] = 0.8
            
            self.logger.debug(f"   Top sectors: {sorted_sectors[:3]}")
            return sorted_sectors, sector_weights
            
        except Exception as e:
            self.logger.error(f"Error in sector rotation analysis: {e}")
            return [], {}
    
    def _calculate_position_multiplier(self, trend_regime: str, volatility_regime: str, vix_level: float) -> float:
        """Calculate overall position size multiplier based on regime analysis"""
        try:
            multiplier = 1.0
            
            # Trend regime adjustment
            if trend_regime == 'BEAR':
                multiplier *= self.BEAR_MARKET_REDUCTION  # 50% reduction in bear markets
                self.logger.debug(f"   Bear market reduction: {self.BEAR_MARKET_REDUCTION}")
            elif trend_regime == 'RANGING':
                multiplier *= 0.8  # 20% reduction in ranging markets
                self.logger.debug(f"   Ranging market reduction: 0.8")
            
            # Volatility regime adjustment  
            if volatility_regime in ['HIGH', 'CRISIS'] and vix_level > 20:
                vix_adjustment = min(self.HIGH_VOL_SCALING_BASE / vix_level, 1.0)
                multiplier *= vix_adjustment
                self.logger.debug(f"   VIX adjustment ({vix_level:.1f}): {vix_adjustment:.2f}")
            elif volatility_regime == 'MEDIUM' and vix_level > 20:
                vix_adjustment = min(self.HIGH_VOL_SCALING_BASE / vix_level, 1.0)
                multiplier *= vix_adjustment
                self.logger.debug(f"   VIX adjustment ({vix_level:.1f}): {vix_adjustment:.2f}")
            
            # Cap the multiplier to reasonable bounds
            multiplier = max(0.1, min(2.0, multiplier))
            
            return multiplier
            
        except Exception as e:
            self.logger.error(f"Error calculating position multiplier: {e}")
            return 0.5  # Conservative fallback
    
    def _calculate_regime_confidence(self, trend_regime: str, trend_strength: float, 
                                   volatility_regime: str, vix_level: float) -> float:
        """Calculate confidence in regime detection"""
        try:
            confidence_factors = []
            
            # Trend confidence
            if trend_regime in ['BULL', 'BEAR']:
                confidence_factors.append(min(1.0, trend_strength))
            else:
                confidence_factors.append(0.6)  # Ranging markets are harder to detect
            
            # Volatility confidence
            if volatility_regime in ['LOW', 'HIGH']:
                # Clear volatility regimes are more confident
                if vix_level < 15 or vix_level > 35:
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.6)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def _get_market_data(self, symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """Get market data with caching"""
        try:
            cache_key = f"{symbol}_{period}"
            
            # Check cache
            if (self._cache_timestamp and 
                datetime.now() - self._cache_timestamp < self._cache_duration and
                cache_key in self._cache):
                return self._cache[cache_key]
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            # Update cache
            if self._cache_timestamp is None or datetime.now() - self._cache_timestamp >= self._cache_duration:
                self._cache.clear()
                self._cache_timestamp = datetime.now()
            
            self._cache[cache_key] = data
            return data
            
        except Exception as e:
            self.logger.debug(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _get_raw_debug_data(self) -> Dict:
        """Get raw data for debugging purposes"""
        return {
            'cache_keys': list(self._cache.keys()),
            'cache_timestamp': self._cache_timestamp,
            'analysis_timestamp': datetime.now()
        }
    
    def _log_regime_summary(self, regime_data: MarketRegimeData):
        """Log comprehensive regime analysis summary"""
        self.logger.info("📊 MARKET REGIME ANALYSIS SUMMARY")
        self.logger.info(f"   Trend Regime: {regime_data.trend_regime} (strength: {regime_data.trend_strength:.2f})")
        self.logger.info(f"   Volatility Regime: {regime_data.volatility_regime} (VIX: {regime_data.volatility_level:.1f})")
        self.logger.info(f"   Position Multiplier: {regime_data.position_size_multiplier:.2f}")
        self.logger.info(f"   Regime Confidence: {regime_data.regime_confidence:.2f}")
        
        if regime_data.sector_leaders:
            top_3_sectors = regime_data.sector_leaders[:3]
            self.logger.info(f"   Top Sectors: {', '.join([f'{sector}({perf:+.1f}%)' for sector, perf in top_3_sectors])}")
        
        if regime_data.sector_weights:
            boosted_sectors = [sector for sector, weight in regime_data.sector_weights.items() if weight > 1.0]
            if boosted_sectors:
                self.logger.info(f"   Boosted Sectors: {', '.join(boosted_sectors)}")
    
    def _get_fallback_regime_data(self) -> MarketRegimeData:
        """Conservative fallback regime data when analysis fails"""
        return MarketRegimeData(
            trend_regime='RANGING',
            trend_strength=0.5,
            volatility_regime='MEDIUM', 
            volatility_level=25.0,
            sector_leaders=[],
            sector_weights={},
            position_size_multiplier=0.5,  # Conservative sizing
            regime_confidence=0.3,
            last_updated=datetime.now(),
            raw_data={'error': 'Fallback regime data used'}
        )
    
    def get_sector_weight_for_symbol(self, symbol: str, sector_weights: Dict[str, float]) -> float:
        """Get sector weight multiplier for a specific symbol"""
        # This is a simplified mapping - in practice you'd want a more comprehensive mapping
        symbol_to_sector = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Technology', 'NVDA': 'Technology', 'META': 'Technology', 'NFLX': 'Technology',
            
            # Healthcare  
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            
            # Financials
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            
            # Consumer Discretionary
            'HD': 'Consumer_Discretionary', 'MCD': 'Consumer_Discretionary', 'DIS': 'Consumer_Discretionary',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
        }
        
        sector = symbol_to_sector.get(symbol, 'Technology')  # Default to Technology
        return sector_weights.get(sector, 1.0)


class RegimeAwarePositionSizer:
    """
    Position sizer that adapts to market regime
    Integrates with existing position sizing logic
    """
    
    def __init__(self, base_position_sizer, regime_analyzer: EnhancedMarketRegimeAnalyzer, logger=None):
        self.base_sizer = base_position_sizer
        self.regime_analyzer = regime_analyzer
        self.logger = logger or logging.getLogger(__name__)
        self.current_regime = None
        self.regime_update_frequency = timedelta(hours=1)  # Update regime every hour
        self.last_regime_update = None
    
    def get_regime_adjusted_position_size(self, signal, target_account, base_position_size: float) -> float:
        """
        Calculate position size adjusted for current market regime
        """
        try:
            # Update regime if needed
            if (self.last_regime_update is None or 
                datetime.now() - self.last_regime_update > self.regime_update_frequency):
                self.current_regime = self.regime_analyzer.analyze_market_regime()
                self.last_regime_update = datetime.now()
            
            if not self.current_regime:
                return base_position_size
            
            # Apply regime multiplier
            regime_adjusted_size = base_position_size * self.current_regime.position_size_multiplier
            
            # Apply sector weighting
            sector_weight = self.regime_analyzer.get_sector_weight_for_symbol(
                signal.symbol, self.current_regime.sector_weights
            )
            final_size = regime_adjusted_size * sector_weight
            
            # Log the adjustments
            if regime_adjusted_size != base_position_size or sector_weight != 1.0:
                self.logger.info(f"📊 Regime adjustments for {signal.symbol}:")
                self.logger.info(f"   Base size: ${base_position_size:.2f}")
                self.logger.info(f"   Regime multiplier: {self.current_regime.position_size_multiplier:.2f}")
                self.logger.info(f"   Sector weight: {sector_weight:.2f}")
                self.logger.info(f"   Final size: ${final_size:.2f}")
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error in regime-adjusted position sizing: {e}")
            return base_position_size  # Fallback to base size
    
    def should_reduce_trading_activity(self) -> bool:
        """Check if trading activity should be reduced based on regime"""
        if not self.current_regime:
            return False
        
        # Reduce activity in high stress regimes
        if (self.current_regime.volatility_regime in ['HIGH', 'CRISIS'] or
            self.current_regime.trend_regime == 'BEAR' or
            self.current_regime.regime_confidence < 0.4):
            return True
        
        return False
    
    def get_regime_summary(self) -> str:
        """Get current regime summary for logging"""
        if not self.current_regime:
            return "No regime data available"
        
        return (f"Trend: {self.current_regime.trend_regime}, "
                f"Volatility: {self.current_regime.volatility_regime} ({self.current_regime.volatility_level:.1f}), "
                f"Position Multiplier: {self.current_regime.position_size_multiplier:.2f}")


# Example usage function
def test_market_regime_analyzer():
    """Test the market regime analyzer"""
    logging.basicConfig(level=logging.INFO)
    
    analyzer = EnhancedMarketRegimeAnalyzer()
    regime_data = analyzer.analyze_market_regime()
    
    print("\n" + "="*80)
    print("ENHANCED MARKET REGIME ANALYSIS TEST")
    print("="*80)
    print(f"Trend Regime: {regime_data.trend_regime} (strength: {regime_data.trend_strength:.2f})")
    print(f"Volatility Regime: {regime_data.volatility_regime} (VIX: {regime_data.volatility_level:.1f})")
    print(f"Position Size Multiplier: {regime_data.position_size_multiplier:.2f}")
    print(f"Regime Confidence: {regime_data.regime_confidence:.2f}")
    
    if regime_data.sector_leaders:
        print(f"\nTop 5 Sectors:")
        for i, (sector, performance) in enumerate(regime_data.sector_leaders[:5], 1):
            weight = regime_data.sector_weights.get(sector, 1.0)
            print(f"  {i}. {sector}: {performance:+.1f}% (weight: {weight:.1f}x)")
    
    print(f"\nLast Updated: {regime_data.last_updated}")
    return regime_data


if __name__ == "__main__":
    test_market_regime_analyzer()