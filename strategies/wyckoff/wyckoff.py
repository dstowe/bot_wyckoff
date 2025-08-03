import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sqlite3
import traceback


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
    """Handles SQLite database operations for storing and retrieving stock data."""
    def __init__(self, db_name: str = "stock_data.db"):
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
            print(f"Error reading from DB for {symbol}: {e}")
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
            print(f"No valid data to save for {symbol}")
            return

        # Ensure index is timezone-naive for SQLite compatibility
        if data_to_save.index.tz is not None:
            data_to_save.index = data_to_save.index.tz_localize(None)

        try:
            with self.conn:
                min_date, max_date = data_to_save.index.min(), data_to_save.index.max()
                
                # Convert Timestamp objects to ISO 8601 string format for SQLite
                min_date_str = min_date.isoformat()
                max_date_str = max_date.isoformat()
                
                cur = self.conn.cursor()
                # Use the string versions of the dates in the query
                cur.execute(
                    "DELETE FROM stock_data WHERE symbol = ? AND date >= ? AND date <= ?",
                    (symbol, min_date_str, max_date_str)
                )
                
                # Now append the new data
                data_to_save.to_sql('stock_data', self.conn, if_exists='append', index=True)
        except Exception as e:
            print(f"Error saving data to DB for {symbol}: {e}")
            # Print more debug info
            print(f"DataFrame columns: {data.columns}")
            print(f"Data to save columns: {data_to_save.columns}")

    def is_data_stale(self, symbol: str) -> bool:
        """Checks if the data is older than the last trading day."""
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(date) FROM stock_data WHERE symbol = ?", (symbol,))
        result = cur.fetchone()[0]
        if not result:
            return True
        
        # Convert to timezone-naive datetime for comparison
        last_date = pd.to_datetime(result)
        if last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        
        # Stale if data is more than 3 days old (to account for weekends)
        return (datetime.now() - last_date) > timedelta(days=3)


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


class WyckoffAnalyzer:
    """Wyckoff Method analysis for identifying accumulation/distribution phases"""
    def __init__(self):
        self.phases = ['PS', 'SC', 'AR', 'ST', 'Creek', 'SOS', 'LPS', 'BU']

    def analyze_wyckoff_phase(self, symbol: str, data: pd.DataFrame, pf_chart: PointFigureChart) -> Dict:
        if len(data) < 50: return {'phase': 'Insufficient_Data', 'strength': 0}

        volume_ma = data['volume'].rolling(20).mean()
        recent_data = data.tail(20)
        phase_scores = {
            'PS': self._calculate_ps_score(symbol, recent_data, volume_ma.tail(20)),
            'SC': self._calculate_sc_score(symbol, recent_data, volume_ma.tail(20)),
            'AR': self._calculate_ar_score(symbol, recent_data),
            'ST': self._calculate_st_score(symbol, recent_data, volume_ma.tail(20)),
            'Creek': self._calculate_creek_score(symbol, recent_data),
            'SOS': self._calculate_sos_score(symbol, recent_data, volume_ma.tail(20), pf_chart),
            'LPS': self._calculate_lps_score(symbol, recent_data, volume_ma.tail(20)),
            'BU': self._calculate_bu_score(symbol, recent_data)
        }
        best_phase = max(phase_scores, key=phase_scores.get)
        return {'phase': best_phase, 'strength': phase_scores[best_phase], 'all_scores': phase_scores}

    def _log_error(self, func_name, symbol, e):
        print(f"\n--- ERROR in {func_name} for {symbol} ---\nError: {e}")
        traceback.print_exc()
        print("--- END ERROR ---")

    def _calculate_ps_score(self, symbol: str, data: pd.DataFrame, volume_ma: pd.Series) -> float:
        if len(data) < 5: return 0.0
        try:
            aligned_volume_ma = volume_ma.reindex(data.index, method='ffill')
            high_vol = data['volume'] > aligned_volume_ma * 1.3
            price_dec = data['close'] < data['close'].shift(1)
            return float((high_vol & price_dec).sum()) / 5.0
        except Exception as e:
            self._log_error('_calculate_ps_score', symbol, e); return 0.0

    def _calculate_sc_score(self, symbol: str, data: pd.DataFrame, volume_ma: pd.Series) -> float:
        if len(data) < 3: return 0.0
        try:
            aligned_volume_ma = volume_ma.reindex(data.index, method='ffill')
            vol_spike = float(data['volume'].max()) > float(aligned_volume_ma.max()) * 2
            price_drop = (float(data['close'].min()) / float(data['close'].max()) - 1) < -0.05
            return 0.8 if (vol_spike and price_drop) else 0.2
        except Exception as e:
            self._log_error('_calculate_sc_score', symbol, e); return 0.0

    def _calculate_ar_score(self, symbol: str, data: pd.DataFrame) -> float:
        if len(data) < 5: return 0.0
        try:
            recovery = (float(data['close'].iloc[-1]) / float(data['low'].min())) - 1
            return min(recovery * 5, 1.0)
        except Exception as e:
            self._log_error('_calculate_ar_score', symbol, e); return 0.0

    def _calculate_st_score(self, symbol: str, data: pd.DataFrame, volume_ma: pd.Series) -> float:
        if len(data) < 10: return 0.0
        try:
            recent_low = float(data['low'].tail(5).min())
            prev_low = float(data['low'].head(10).min())
            aligned_vol_ma = volume_ma.reindex(data.index, method='ffill')
            low_vol_test = float(data['volume'].tail(5).mean()) < float(aligned_vol_ma.mean())
            price_test = abs(recent_low - prev_low) / prev_low < 0.02
            return 0.7 if (price_test and low_vol_test) else 0.1
        except Exception as e:
            self._log_error('_calculate_st_score', symbol, e); return 0.0

    def _calculate_creek_score(self, symbol: str, data: pd.DataFrame) -> float:
        if len(data) < 10: return 0.0
        try:
            price_range = (float(data['high'].max()) - float(data['low'].min())) / float(data['close'].mean())
            vol_decline = float(data['volume'].tail(5).mean()) < float(data['volume'].head(10).mean())
            return 0.6 if (price_range < 0.05 and vol_decline) else 0.1
        except Exception as e:
            self._log_error('_calculate_creek_score', symbol, e); return 0.0

    def _calculate_sos_score(self, symbol: str, data: pd.DataFrame, volume_ma: pd.Series, pf_chart: PointFigureChart) -> float:
        if len(data) < 5: return 0.0
        try:
            price_breakout = float(data['close'].iloc[-1]) > float(data['high'].head(15).max())
            aligned_vol_ma = volume_ma.reindex(data.index, method='ffill')
            vol_confirm = float(data['volume'].tail(3).mean()) > float(aligned_vol_ma.mean()) * 1.2
            pf_breakout = len(pf_chart.identify_patterns()['double_top_breakout']) > 0
            score = sum([s for c, s in zip([price_breakout, vol_confirm, pf_breakout], [0.4, 0.3, 0.3]) if c])
            return score
        except Exception as e:
            self._log_error('_calculate_sos_score', symbol, e); return 0.0

    def _calculate_lps_score(self, symbol: str, data: pd.DataFrame, volume_ma: pd.Series) -> float:
        if len(data) < 10: return 0.0
        try:
            support_test = float(data['low'].tail(5).min()) > float(data['low'].head(10).min()) * 0.98
            aligned_vol_ma = volume_ma.reindex(data.index, method='ffill')
            low_vol = float(data['volume'].tail(5).mean()) < float(aligned_vol_ma.mean())
            return 0.6 if (support_test and low_vol) else 0.1
        except Exception as e:
            self._log_error('_calculate_lps_score', symbol, e); return 0.0

    def _calculate_bu_score(self, symbol: str, data: pd.DataFrame) -> float:
        if len(data) < 10: return 0.0
        try:
            pullback = 1 - (float(data['close'].iloc[-1]) / float(data['high'].head(10).max()))
            return min(pullback * 3, 1.0) if pullback > 0.02 else 0.1
        except Exception as e:
            self._log_error('_calculate_bu_score', symbol, e); return 0.0


class SectorRotationAnalyzer:
    """Analyze sector rotation patterns"""
    def __init__(self):
        self.sector_etfs = {'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
                            'Consumer_Discretionary': 'XLY', 'Consumer_Staples': 'XLP',
                            'Energy': 'XLE', 'Utilities': 'XLU', 'Real_Estate': 'XLRE',
                            'Materials': 'XLB', 'Industrials': 'XLI', 'Communication': 'XLC'}

    def get_sector_ranking(self, period: str = '6mo') -> List[Tuple[str, float]]:
        """Get sectors ranked by relative strength"""
        try:
            # Download SPY first using Ticker object
            spy_ticker = yf.Ticker('SPY')
            spy = spy_ticker.history(period=period)
            if spy.empty:
                print("Could not download SPY data")
                return []
            spy_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1)
            
            # Download ETFs individually using Ticker objects
            perf = {}
            for sector, etf in self.sector_etfs.items():
                try:
                    etf_ticker = yf.Ticker(etf)
                    etf_data = etf_ticker.history(period=period)
                    if not etf_data.empty and 'Close' in etf_data.columns:
                        etf_close = etf_data['Close'].dropna()
                        if len(etf_close) > 1:
                            etf_return = (etf_close.iloc[-1] / etf_close.iloc[0] - 1)
                            perf[sector] = (etf_return - spy_return) * 100
                except Exception as e:
                    print(f"Error downloading {etf}: {e}")
                    continue
            
            return sorted(perf.items(), key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"Could not download sector data: {e}")
            return []


class WyckoffPnFStrategy:
    """Main trading strategy combining Wyckoff, Point and Figure, and Sector Rotation"""
    def __init__(self):
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.sector_analyzer = SectorRotationAnalyzer()
        self.db_manager = DatabaseManager()
        self.symbols = self.get_sp500_symbols()

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

        print(f"Downloading fresh data for {len(stale_symbols)} symbols...")
        # Download symbols individually to avoid multi-index issues
        for i, symbol in enumerate(stale_symbols, 1):
            try:
                print(f"Downloading {symbol} ({i}/{len(stale_symbols)})...", end='\r')
                # Use Ticker object for cleaner single-symbol data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='1y', auto_adjust=True)
                if not df.empty:
                    self.db_manager.save_data(symbol, df)
            except Exception as e:
                print(f"\nError downloading {symbol}: {e}")
        
        print("\nDatabase update complete.")

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
        
        signals = self.scan_market()
        report = self.generate_report(signals, sector_ranking)
        return report


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