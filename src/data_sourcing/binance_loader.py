import os
import pandas as pd
from datetime import datetime
from typing import Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException


class BinanceLoader:
    """
    A data loader class for fetching historical cryptocurrency data from Binance.

    This class provides methods to fetch historical OHLCV data for any symbol
    available on Binance, with automatic pagination to handle API limits.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the BinanceLoader with API credentials.

        Args:
            api_key: Binance API key. If None, reads from BINANCE_API_KEY environment variable.
            api_secret: Binance API secret. If None, reads from BINANCE_API_SECRET environment variable.
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')

        # Initialize Binance client
        try:
            self.client = Client(self.api_key, self.api_secret)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Binance client: {e}")

    def fetch_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol and time range.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValueError: If symbol is invalid or dates are malformed
            ConnectionError: If API connection fails
        """
        try:
            # Convert date strings to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            # Validate date range
            if start_dt >= end_dt:
                raise ValueError("Start date must be before end date")

            # Fetch data with pagination
            all_klines = []
            current_start = start_dt

            while current_start < end_dt:
                try:
                    # Fetch klines for current batch
                    klines = self.client.get_historical_klines(
                        symbol=symbol.upper(),
                        interval=interval,
                        start_str=current_start.strftime('%Y-%m-%d'),
                        end_str=end_dt.strftime('%Y-%m-%d'),
                        limit=1000  # Maximum allowed by Binance
                    )

                    if not klines:
                        break

                    all_klines.extend(klines)

                    # Update current_start to the timestamp of the last kline + 1 interval
                    last_timestamp = klines[-1][0]
                    current_start = datetime.fromtimestamp(last_timestamp / 1000)

                    # If we got less than 1000 klines, we've reached the end
                    if len(klines) < 1000:
                        break

                except BinanceAPIException as e:
                    if "Invalid symbol" in str(e):
                        raise ValueError(f"Invalid symbol: {symbol}")
                    else:
                        raise ConnectionError(f"Binance API error: {e}")
                except BinanceRequestException as e:
                    raise ConnectionError(f"Binance request error: {e}")

            if not all_klines:
                raise ValueError(f"No data found for symbol {symbol} in the specified date range")

            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Keep only the required columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)

            # Remove any duplicate timestamps (can happen at boundaries)
            df = df[~df.index.duplicated(keep='first')]

            # Sort by timestamp
            df.sort_index(inplace=True)

            return df

        except ValueError as e:
            raise e
        except Exception as e:
            raise ConnectionError(f"Failed to fetch historical data: {e}")

    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """
        Save DataFrame to CSV file in the data/raw/ directory.

        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            interval: Kline interval

        Returns:
            Path to the saved CSV file
        """
        # Ensure data/raw directory exists
        raw_data_dir = os.path.join('data', 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)

        # Create filename
        filename = f"{symbol.upper()}-{interval}.csv"
        filepath = os.path.join(raw_data_dir, filename)

        # Save to CSV
        df.to_csv(filepath)

        return filepath