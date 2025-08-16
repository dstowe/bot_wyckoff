# -*- coding: utf-8 -*-
"""
Enhanced Position Sizing Module for Wyckoff Trading Bot 💰
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import logging

class DynamicPositionSizer:
    def __init__(self, config: dict):
        self.base_account_allocation = config.get('base_account_allocation', 0.25)
        self.max_position_allocation = config.get('max_position_allocation', 0.15)
        self.vix_threshold = config.get('vix_threshold', 25.0)
        self.vix_reduction_factor = config.get('vix_reduction_factor', 0.5)
        self.sector_boost_factor = config.get('sector_boost_factor', 0.25)
        
        self.signal_strength_multipliers = {
            'ST': 0.40,   # Spring = 40% of normal
            'BU': 0.60,   # Backup = 60% of normal  
            'SOS': 0.60,  # Sign of Strength = 60% of normal
            'UTAD': 0.30, # Upthrust After Distribution = 30% of normal
            'SOW': 0.40,  # Sign of Weakness = 40% of normal
            'DEFAULT': 1.0
        }
        
        self.sector_etfs = {
            'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
            'Consumer Discretionary': 'XLY', 'Communication Services': 'XLC',
            'Industrials': 'XLI', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
            'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB'
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_current_vix(self):
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="2d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return 20.0
        except:
            return 20.0
    
    def calculate_sector_momentum(self, symbol, lookback_days=20):
        try:
            sector_returns = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 5)
            
            for sector_name, etf_symbol in self.sector_etfs.items():
                try:
                    etf = yf.Ticker(etf_symbol)
                    hist = etf.history(start=start_date, end=end_date)
                    if len(hist) >= lookback_days:
                        returns = (hist['Close'].iloc[-1] / hist['Close'].iloc[-lookback_days] - 1) * 100
                        sector_returns[sector_name] = returns
                except:
                    continue
            
            if not sector_returns:
                return {'symbol_sector': 'Unknown', 'momentum_percentile': 50.0, 'sector_multiplier': 1.0}
            
            symbol_sector = self._get_symbol_sector(symbol)
            returns_list = list(sector_returns.values())
            symbol_return = sector_returns.get(symbol_sector, np.median(returns_list))
            
            percentile = (np.sum(np.array(returns_list) <= symbol_return) / len(returns_list)) * 100
            
            if percentile >= 75:
                sector_multiplier = 1 + self.sector_boost_factor
            elif percentile <= 25:
                sector_multiplier = 1 - (self.sector_boost_factor * 0.5)
            else:
                sector_multiplier = 1.0
            
            return {
                'symbol_sector': symbol_sector,
                'momentum_percentile': percentile,
                'sector_multiplier': sector_multiplier
            }
        except:
            return {'symbol_sector': 'Unknown', 'momentum_percentile': 50.0, 'sector_multiplier': 1.0}
    
    def _get_symbol_sector(self, symbol):
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO']
        financial_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        
        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in healthcare_symbols:
            return 'Healthcare'
        elif symbol in financial_symbols:
            return 'Financials'
        else:
            return 'Consumer Discretionary'
    
    def calculate_dynamic_position_size(self, account_value, symbol, wyckoff_signal, signal_strength=1.0):
        try:
            base_position_value = account_value * self.base_account_allocation
            
            signal_multiplier = self.signal_strength_multipliers.get(wyckoff_signal.upper(), 1.0)
            
            current_vix = self.get_current_vix()
            if current_vix > self.vix_threshold:
                vix_multiplier = self.vix_reduction_factor
                market_regime = "HIGH_VOLATILITY"
            else:
                vix_multiplier = 1.0
                market_regime = "NORMAL_VOLATILITY"
            
            sector_data = self.calculate_sector_momentum(symbol)
            sector_multiplier = sector_data['sector_multiplier']
            
            total_multiplier = signal_multiplier * vix_multiplier * sector_multiplier * signal_strength
            
            adjusted_position_value = base_position_value * total_multiplier
            max_position_value = account_value * self.max_position_allocation
            final_position_value = min(adjusted_position_value, max_position_value)
            
            final_position_percentage = (final_position_value / account_value) * 100
            
            result = {
                'base_position_value': base_position_value,
                'final_position_value': final_position_value,
                'final_position_percentage': final_position_percentage,
                'signal_multiplier': signal_multiplier,
                'vix_multiplier': vix_multiplier,
                'sector_multiplier': sector_multiplier,
                'total_multiplier': total_multiplier,
                'current_vix': current_vix,
                'market_regime': market_regime,
                'sector_data': sector_data,
                'wyckoff_signal': wyckoff_signal,
                'was_capped_by_max': final_position_value == max_position_value
            }
            
            # Log the decision
            log_msg = (f"🎯 POSITION SIZING for {symbol}: "
                      f"{wyckoff_signal} signal (×{signal_multiplier:.2f}), "
                      f"VIX {current_vix:.1f} (×{vix_multiplier:.2f}), "
                      f"Sector ×{sector_multiplier:.2f} = "
                      f"{final_position_percentage:.1f}% allocation")
            self.logger.info(log_msg)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            fallback_value = account_value * 0.10
            return {
                'base_position_value': fallback_value,
                'final_position_value': fallback_value,
                'final_position_percentage': 10.0,
                'signal_multiplier': 0.5,
                'vix_multiplier': 0.5,
                'sector_multiplier': 1.0,
                'total_multiplier': 0.25,
                'current_vix': 25.0,
                'market_regime': "ERROR_FALLBACK",
                'sector_data': {'symbol_sector': 'Unknown', 'momentum_percentile': 50.0, 'sector_multiplier': 1.0},
                'wyckoff_signal': wyckoff_signal,
                'was_capped_by_max': False
            }
