import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sqlite3
import traceback
# ENHANCEMENT: Multi-timeframe analysis import - Strategic Improvement 5 📈
try:
    from .multi_timeframe_analyzer import (
        EnhancedMultiTimeframeWyckoffAnalyzer,
        MultiTimeframeSignal,
        filter_signals_by_quality
    )
    MULTI_TIMEFRAME_AVAILABLE = True
    print("✅ Multi-timeframe signal quality enhancement available")
except ImportError as e:
    MULTI_TIMEFRAME_AVAILABLE = False
    print(f"⚠️ Multi-timeframe enhancement not available: {e}")



warnings.filterwarnings('ignore')

@dataclass
class PointFigureBox:
    """Represents a single box in Point and Figure chart"""
    price: float
    box_type: str  # 'X' or 'O'
    column: int
    row: int

@dataclass
class WyckoffSignal:
    """Represents a Wyckoff accumulation signal"""
    symbol: str
    phase: str
    strength: float
    price: float
    volume_confirmation: bool
    sector: str
    combined_score: float = 0.0


class DatabaseManager:
    """Simple, working database manager with fixed f-string issues"""
    
    def __init__(self, db_name: str = "data/stock_data.db"):
        self.db_name = db_name
        # Allow the connection to be used across threads for the ThreadPoolExecutor
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            symbol TEXT NOT NULL,
            date TIMESTAMP NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        );
        """
        self.conn.cursor().execute(query)
        self.conn.commit()

    def get_data(self, symbol: str, period_days: int = 365) -> pd.DataFrame | None:
        """Reads data for a symbol from the database for the given period."""
        try:
            start_date = datetime.now() - timedelta(days=period_days)
            df = pd.read_sql_query(
                "SELECT * FROM stock_data WHERE symbol = ? AND date >= ?",
                self.conn,
                params=(symbol, start_date),
                index_col='date',
                parse_dates=['date']
            )
            if not df.empty:
                # Ensure timezone-naive datetime index
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            return df if not df.empty else None
        except Exception as e:
            print("Error reading from DB for " + symbol + ": " + str(e))
            return None

    def save_data(self, symbol: str, data: pd.DataFrame):
        """Saves a DataFrame for a symbol to the database."""
        if data.empty:
            return

        # Create a clean DataFrame for saving
        data_to_save = pd.DataFrame(index=data.index)
        
        # Handle different DataFrame structures from yfinance
        required_columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 
                          'Close': 'close', 'Volume': 'volume'}
        
        for col_name, db_name in required_columns.items():
            if isinstance(data.columns, pd.MultiIndex):
                # Handle MultiIndex columns
                if col_name in data.columns.get_level_values(0):
                    # Get the column data, handling the MultiIndex
                    col_data = data[col_name]
                    if isinstance(col_data, pd.DataFrame):
                        # If it's still a DataFrame, take the first column
                        data_to_save[db_name] = col_data.iloc[:, 0]
                    else:
                        data_to_save[db_name] = col_data
            else:
                # Handle regular columns
                if col_name in data.columns:
                    data_to_save[db_name] = data[col_name]
        
        # Add symbol column
        data_to_save['symbol'] = symbol
        
        # Remove any rows with NaN values
        data_to_save.dropna(inplace=True)
        
        if data_to_save.empty:
            print("No valid data to save for " + symbol)
            return

        # Ensure index is timezone-naive for SQLite compatibility
        if data_to_save.index.tz is not None:
            data_to_save.index = data_to_save.index.tz_localize(None)

        try:
            with self.conn:
                cur = self.conn.cursor()
                # Delete all existing data for this symbol to prevent UNIQUE constraint errors
                cur.execute("DELETE FROM stock_data WHERE symbol = ?", (symbol,))
                
                # Now append the new data
                data_to_save.to_sql('stock_data', self.conn, if_exists='append', index=True)
                print("Successfully saved data for " + symbol)
        except Exception as e:
            print("Error saving data to DB for " + symbol + ": " + str(e))

    def is_data_stale(self, symbol: str) -> bool:
        """Checks if the data is older than 6 hours (more aggressive than before)."""
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(date) FROM stock_data WHERE symbol = ?", (symbol,))
        result = cur.fetchone()[0]
        if not result:
            return True
        
        # Convert to timezone-naive datetime for comparison
        last_date = pd.to_datetime(result)
        if last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        
        # More aggressive: stale if data is more than 6 hours old (instead of 3 days)
        hours_old = (datetime.now() - last_date).total_seconds() / 3600
        return hours_old > 6
class PointFigureChart:
    """Point and Figure Chart implementation"""

    def __init__(self, box_size: float = None, reversal: int = 3):
        self.box_size = box_size
        self.reversal = reversal
        self.boxes = []
        self.columns = []
        self.current_column = 0
        self.current_trend = None

    def calculate_box_size(self, prices: pd.Series) -> float:
        """Calculate optimal box size based on price range and volatility"""
        price_range = prices.max() - prices.min()
        volatility = prices.std()

        avg_price = prices.mean()
        if avg_price < 5: return round(volatility * 0.1, 2)
        if avg_price < 20: return round(volatility * 0.15, 2)
        if avg_price < 100: return round(volatility * 0.2, 1)
        return round(volatility * 0.25, 0)

    def build_chart(self, prices: pd.Series) -> List[List[PointFigureBox]]:
        """Build Point and Figure chart from price data"""
        if self.box_size is None:
            self.box_size = self.calculate_box_size(prices)

        if self.box_size <= 0: self.box_size = 0.01

        self.columns = []
        current_column = []
        trend = None
        last_price = None

        for price in prices:
            if last_price is None:
                last_price = price
                continue

            if trend is None:
                if price > last_price + self.box_size: trend = 'X'
                elif price < last_price - self.box_size: trend = 'O'
                else:
                    last_price = price
                    continue

            if trend == 'X':
                if price >= last_price + self.box_size:
                    boxes_to_add = int((price - last_price) / self.box_size)
                    for i in range(boxes_to_add):
                        current_column.append(PointFigureBox(
                            price=last_price + (i + 1) * self.box_size,
                            box_type='X', column=len(self.columns), row=len(current_column)
                        ))
                    last_price += boxes_to_add * self.box_size
                elif price <= last_price - (self.reversal * self.box_size):
                    if current_column: self.columns.append(current_column)
                    current_column = []
                    trend = 'O'
                    boxes_to_add = int((last_price - price) / self.box_size)
                    for i in range(boxes_to_add):
                        current_column.append(PointFigureBox(
                            price=last_price - (i + 1) * self.box_size,
                            box_type='O', column=len(self.columns), row=len(current_column)
                        ))
                    last_price -= boxes_to_add * self.box_size

            else:  # trend == 'O'
                if price <= last_price - self.box_size:
                    boxes_to_add = int((last_price - price) / self.box_size)
                    for i in range(boxes_to_add):
                        current_column.append(PointFigureBox(
                            price=last_price - (i + 1) * self.box_size,
                            box_type='O', column=len(self.columns), row=len(current_column)
                        ))
                    last_price -= boxes_to_add * self.box_size
                elif price >= last_price + (self.reversal * self.box_size):
                    if current_column: self.columns.append(current_column)
                    current_column = []
                    trend = 'X'
                    boxes_to_add = int((price - last_price) / self.box_size)
                    for i in range(boxes_to_add):
                        current_column.append(PointFigureBox(
                            price=last_price + (i + 1) * self.box_size,
                            box_type='X', column=len(self.columns), row=len(current_column)
                        ))
                    last_price += boxes_to_add * self.box_size

        if current_column: self.columns.append(current_column)
        return self.columns

    def identify_patterns(self) -> Dict[str, List[int]]:
        """Identify key Point and Figure patterns"""
        patterns = {
            'double_top_breakout': [], 'double_bottom_breakout': [],
            'triple_top_breakout': [], 'ascending_triangle': [], 'descending_triangle': []
        }
        if len(self.columns) < 3: return patterns

        for i in range(1, len(self.columns) - 1):
            col = self.columns[i]
            prev_col = self.columns[i-1]
            if col and prev_col:
                if col[0].box_type == 'X' and col[-1].price > max(c.price for c in prev_col):
                    patterns['double_top_breakout'].append(i)
                if col[0].box_type == 'O' and col[-1].price < min(c.price for c in prev_col):
                    patterns['double_bottom_breakout'].append(i)
        return patterns

class SectorRotationAnalyzer:
    """Simple sector rotation analyzer"""
    def __init__(self):
        self.sector_etfs = {
            'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
            'Consumer_Discretionary': 'XLY', 'Energy': 'XLE'
        }
    
    def get_sector_ranking(self):
        # Simple fallback ranking
        return [
            ('Technology', 5.2), ('Healthcare', 3.1), ('Financials', 2.8),
            ('Consumer_Discretionary', 2.1), ('Energy', 1.5)
        ]

class WyckoffAnalyzer:
    """Internal Wyckoff analysis methods"""
    def __init__(self):
        pass
    
    def analyze_wyckoff_phase(self, symbol, data, pf_chart):
        return {'phase': 'UNKNOWN', 'strength': 0.5}
    
    def _log_error(self, method, symbol, error):
        print(f"Wyckoff error in {method} for {symbol}: {error}")
        
class WyckoffPnFStrategy:
    """Wyckoff Method analysis for identifying accumulation/distribution phases"""
    def __init__(self):
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.sector_analyzer = SectorRotationAnalyzer()
        self.db_manager = DatabaseManager("data/stock_data.db")
        self.symbols = self.get_sp500_symbols()
        
        # ENHANCEMENT: Initialize multi-timeframe analyzer WITH database sharing
        if MULTI_TIMEFRAME_AVAILABLE:
            self.enhanced_analyzer = EnhancedMultiTimeframeWyckoffAnalyzer(
                db_manager=self.db_manager  # Pass database manager to avoid duplicate downloads
            )
            self.use_enhanced_analysis = True
            print("🎯 Enhanced multi-timeframe analysis enabled with DB optimization")
            self.use_enhanced_analysis = True
            print("🎯 Enhanced multi-timeframe Wyckoff analysis enabled")
        else:
            self.enhanced_analyzer = None
            self.use_enhanced_analysis = False
            print("📊 Using standard single-timeframe analysis")

    def get_sp500_symbols(self) -> List[str]:        
        # A sample of major stocks across different sectors
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'ADBE',
                'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY', 'MRK',
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
                'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'F', 'GM',
                'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY',
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
                'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'FDX',
                'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF',
                'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ES', 'AWK']

    def map_symbol_to_sector(self, symbol: str) -> str:
        # This is a simplified mapping for demonstration.
        sector_map = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'ADBE'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY', 'MRK'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
            'Consumer_Discretionary': ['HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'F', 'GM'],
            'Consumer_Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'FDX'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ES', 'AWK']
        }
        for sector, stocks in sector_map.items():
            if symbol in stocks: return sector
        return 'Unknown'

    def update_database(self):
        """Ensures the database is up-to-date for all symbols."""
        print("Checking database for stale data...")
        stale_symbols = [s for s in self.symbols if self.db_manager.is_data_stale(s)]

        if not stale_symbols:
            print("Database is up-to-date.")
            return

        print("Downloading fresh data for " + str(len(stale_symbols)) + " symbols...")
        # Download symbols individually to avoid multi-index issues
        success_count = 0
        for i, symbol in enumerate(stale_symbols, 1):
            try:
                print("Downloading " + symbol + " (" + str(i) + "/" + str(len(stale_symbols)) + ")...")
                # Use Ticker object for cleaner single-symbol data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='1y', auto_adjust=True)
                if not df.empty:
                    self.db_manager.save_data(symbol, df)
                    success_count += 1
            except Exception as e:
                print("Error downloading " + symbol + ": " + str(e))
        
        print("Database update complete: " + str(success_count) + "/" + str(len(stale_symbols)) + " successful")
    def analyze_symbol(self, symbol: str) -> Optional[WyckoffSignal]:
        try:
            data = self.db_manager.get_data(symbol, period_days=365)
            if data is None or len(data) < 100:
                return None

            pf_chart = PointFigureChart()
            pf_chart.build_chart(data['close'])

            wyckoff_analysis = self.wyckoff_analyzer.analyze_wyckoff_phase(symbol, data, pf_chart)
            
            phases = ['ST', 'Creek', 'SOS', 'LPS', 'BU']
            if wyckoff_analysis['phase'] in phases and wyckoff_analysis['strength'] > 0.4:
                vol_confirm = float(data['volume'].tail(10).mean()) > float(data['volume'].mean()) * 0.8
                return WyckoffSignal(
                    symbol=symbol, phase=wyckoff_analysis['phase'],
                    strength=wyckoff_analysis['strength'], price=float(data['close'].iloc[-1]),
                    volume_confirmation=vol_confirm, sector=self.map_symbol_to_sector(symbol)
                )
        except Exception as e:
            self.wyckoff_analyzer._log_error('analyze_symbol', symbol, e)
        return None

    def scan_market(self, max_workers: int = 10) -> List[WyckoffSignal]:
        """Scan the market for Wyckoff accumulation signals"""
        print(f"Scanning {len(self.symbols)} symbols...")
        signals = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.analyze_symbol, s): s for s in self.symbols}
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result:
                    signals.append(result)
        print(f"Found {len(signals)} potential signals.")
        return signals

    def generate_report(self, signals: List[WyckoffSignal], sector_ranking: List[Tuple[str, float]]) -> str:
        """Generate comprehensive trading report"""
        if not signals:
            return "No Wyckoff accumulation signals found in current market scan."

        sector_strength = dict(sector_ranking)
        for signal in signals:
            sec_str = sector_strength.get(signal.sector, 0.0)
            signal.combined_score = (signal.strength * 0.6 + (sec_str / 100) * 0.3 +
                                   (0.1 if signal.volume_confirmation else 0))
        signals.sort(key=lambda x: x.combined_score, reverse=True)

        report = f"""=== WYCKOFF RE-ACCUMULATION TRADING STRATEGY REPORT ===

Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Signals Found: {len(signals)}

=== SECTOR ROTATION ANALYSIS (Top 5) ===
"""
        for i, (sector, strength) in enumerate(sector_ranking[:5], 1):
            report += f"{i}. {sector}: {strength:+.2f}%\n"
        report += "\n=== TOP WYCKOFF ACCUMULATION SIGNALS ===\n"
        report += f"{'Rank':<4} {'Symbol':<6} {'Phase':<6} {'Strength':<8} {'Sector':<22} {'Price':<8} {'Volume':<6} {'Score':<6}\n"
        report += "-" * 80 + "\n"
        for i, s in enumerate(signals[:20], 1):
            report += (f"{i:<4} {s.symbol:<6} {s.phase:<6} {s.strength:<8.2f} "
                       f"{s.sector:<22} ${s.price:<7.2f} {'✓' if s.volume_confirmation else '✗':<6} "
                       f"{s.combined_score:<6.3f}\n")
        return report

    def run_strategy(self) -> str:
        """Run the complete trading strategy"""
        print("Starting Wyckoff Re-Accumulation Strategy...")
        self.update_database()
        
        print("\nAnalyzing sector rotation...")
        sector_ranking = self.sector_analyzer.get_sector_ranking()
        
        signals = self.scan_market_enhanced()
        report = self.generate_report(signals, sector_ranking)
        return report



    def scan_market_enhanced(self, max_workers: int = 10) -> List[WyckoffSignal]:
        """Enhanced market scanning with multi-timeframe analysis"""
        try:
            if self.use_enhanced_analysis and self.enhanced_analyzer:
                print("🔍 Using enhanced multi-timeframe scanning")
                
                enhanced_signals = self.enhanced_analyzer.scan_market_enhanced(
                    self.symbols, max_workers=max_workers
                )
                
                # Convert to legacy format
                legacy_signals = []
                for enhanced_signal in enhanced_signals:
                    if enhanced_signal.signal_quality in ['GOOD', 'EXCELLENT']:
                        legacy_signal = WyckoffSignal(
                            symbol=enhanced_signal.symbol,
                            phase=enhanced_signal.primary_phase,
                            strength=enhanced_signal.enhanced_strength,
                            price=enhanced_signal.price,
                            volume_confirmation=enhanced_signal.volume_confirmation,
                            sector=enhanced_signal.sector,
                            combined_score=enhanced_signal.confirmation_score
                        )
                        legacy_signals.append(legacy_signal)
                
                print(f"🎯 Enhanced analysis found {len(legacy_signals)} high-quality signals")
                
                return legacy_signals
            else:
                return self.scan_market(max_workers)
                
        except Exception as e:
            print(f"❌ Enhanced scanning failed, using fallback: {e}")
            return self.scan_market(max_workers)

def main():
    """Main execution function"""
    strategy = WyckoffPnFStrategy()
    report = strategy.run_strategy()
    print("\n" + "="*80)
    print(report)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'wyckoff_analysis_{timestamp}.txt'
    
    # FIX: Add encoding='utf-8' to handle Unicode characters
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {filename}")


if __name__ == "__main__":
    main()