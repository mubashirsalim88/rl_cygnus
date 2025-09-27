import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Any, Optional
from hurst import compute_Hc


class FeatureEngine:
    """
    A comprehensive feature engineering pipeline for financial time series data.

    This class provides methods to transform raw OHLCV data into a rich set of features
    including technical indicators, derivative features, regime features, and normalized
    features suitable for quantitative analysis and machine learning.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEngine with raw OHLCV data.

        Args:
            df: DataFrame containing OHLCV data with columns: open, high, low, close, volume
        """
        # Create a copy to avoid modifying the original data
        self.df = df.copy()

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def add_log_returns(self) -> 'FeatureEngine':
        """
        Calculate logarithmic returns of the close price.

        Returns:
            Self for method chaining
        """
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        return self

    def add_multi_scale_features(self, configs: List[Dict[str, Any]]) -> 'FeatureEngine':
        """
        Add technical indicators based on configuration.

        Args:
            configs: List of dictionaries defining indicators to calculate.
                    Examples:
                    - {'indicator': 'RSI', 'period': 14}
                    - {'indicator': 'MACD', 'fast': 12, 'slow': 26, 'signal': 9}

        Returns:
            Self for method chaining
        """
        for config in configs:
            indicator = config['indicator'].upper()

            if indicator == 'RSI':
                period = config.get('period', 14)
                rsi = ta.rsi(self.df['close'], length=period)
                self.df[f'RSI_{period}'] = rsi

            elif indicator == 'MACD':
                fast = config.get('fast', 12)
                slow = config.get('slow', 26)
                signal = config.get('signal', 9)

                macd_data = ta.macd(self.df['close'], fast=fast, slow=slow, signal=signal)

                # MACD returns a DataFrame with columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                if macd_data is not None:
                    # Extract the MACD line, signal line, and histogram
                    macd_col = f'MACD_{fast}_{slow}_{signal}'
                    signal_col = f'MACDs_{fast}_{slow}_{signal}'
                    histogram_col = f'MACDh_{fast}_{slow}_{signal}'

                    self.df[macd_col] = macd_data.iloc[:, 0]  # MACD line
                    self.df[histogram_col] = macd_data.iloc[:, 1]  # MACD histogram
                    self.df[signal_col] = macd_data.iloc[:, 2]  # Signal line

            elif indicator == 'EMA':
                period = config.get('period', 20)
                ema = ta.ema(self.df['close'], length=period)
                if ema is not None:
                    self.df[f'EMA_{period}'] = ema

            else:
                print(f"Warning: Indicator '{indicator}' not implemented yet")

        return self

    def add_derivative_features(self, configs: List[Dict[str, Any]]) -> 'FeatureEngine':
        """
        Add derivative features (velocity and acceleration) for specified columns.

        Args:
            configs: List of dictionaries defining derivative calculations.
                    Example: {
                        'column': 'RSI_14',
                        'velocity_period': 5,
                        'acceleration_period': 5,
                        'smoothing_period': 3
                    }

        Returns:
            Self for method chaining
        """
        for config in configs:
            column = config['column']
            velocity_period = config.get('velocity_period', 5)
            acceleration_period = config.get('acceleration_period', 5)
            smoothing_period = config.get('smoothing_period', 3)

            if column not in self.df.columns:
                print(f"Warning: Column '{column}' not found, skipping derivative features")
                continue

            # Calculate velocity (Rate of Change)
            velocity = self.df[column].pct_change(periods=velocity_period) * 100
            velocity_smooth = velocity.rolling(window=smoothing_period).mean()
            velocity_col_name = f'{column}_velo_{velocity_period}_smooth_{smoothing_period}'
            self.df[velocity_col_name] = velocity_smooth

            # Calculate acceleration (second-order difference)
            acceleration = self.df[column].diff(periods=acceleration_period).diff(periods=1)
            acceleration_smooth = acceleration.rolling(window=smoothing_period).mean()
            acceleration_col_name = f'{column}_accel_{acceleration_period}_smooth_{smoothing_period}'
            self.df[acceleration_col_name] = acceleration_smooth

        return self

    def add_regime_features(self, adx_period: int = 14, atr_period: int = 14, hurst_period: int = 100) -> 'FeatureEngine':
        """
        Add regime-based features including ADX, normalized ATR, and Hurst exponent.

        Args:
            adx_period: Period for ADX calculation
            atr_period: Period for ATR calculation
            hurst_period: Rolling window for Hurst exponent calculation

        Returns:
            Self for method chaining
        """
        # Calculate ADX (Average Directional Index)
        adx = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=adx_period)
        if adx is not None:
            self.df[f'ADX_{adx_period}'] = adx.iloc[:, 0]  # ADX column

        # Calculate ATR (Average True Range) and normalize by close price
        atr = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=atr_period)
        if atr is not None:
            self.df[f'ATRp_{atr_period}'] = (atr / self.df['close']) * 100

        # Calculate rolling Hurst exponent
        hurst_values = []
        close_prices = self.df['close'].values

        for i in range(len(close_prices)):
            if i < hurst_period - 1:
                hurst_values.append(np.nan)
            else:
                # Get the window of data
                window_data = close_prices[i - hurst_period + 1:i + 1]

                try:
                    # Calculate Hurst exponent for this window
                    H, c, data = compute_Hc(window_data, kind='price', simplified=True)
                    hurst_values.append(H)
                except:
                    # If calculation fails, use NaN
                    hurst_values.append(np.nan)

        self.df[f'Hurst_{hurst_period}'] = hurst_values

        return self

    def normalize_features(self, columns: List[str], window: int = 100) -> 'FeatureEngine':
        """
        Apply rolling-window Z-score standardization to specified columns.

        Args:
            columns: List of column names to normalize
            window: Rolling window size for normalization

        Returns:
            Self for method chaining
        """
        for column in columns:
            if column not in self.df.columns:
                print(f"Warning: Column '{column}' not found, skipping normalization")
                continue

            # Calculate rolling mean and standard deviation
            rolling_mean = self.df[column].rolling(window=window).mean()
            rolling_std = self.df[column].rolling(window=window).std()

            # Calculate Z-score normalization
            normalized_col = (self.df[column] - rolling_mean) / rolling_std
            normalized_col_name = f'{column}_norm_{window}'
            self.df[normalized_col_name] = normalized_col

        return self

    def process_all(self, configs: Dict[str, Any]) -> pd.DataFrame:
        """
        Main processing method that applies all feature engineering steps.

        Args:
            configs: Configuration dictionary containing parameters for all methods.
                    Example: {
                        'multi_scale': [
                            {'indicator': 'RSI', 'period': 14},
                            {'indicator': 'MACD', 'fast': 12, 'slow': 26, 'signal': 9}
                        ],
                        'derivatives': [
                            {
                                'column': 'RSI_14',
                                'velocity_period': 5,
                                'acceleration_period': 5,
                                'smoothing_period': 3
                            }
                        ],
                        'regime': {
                            'adx_period': 14,
                            'atr_period': 14,
                            'hurst_period': 100
                        },
                        'normalize': {
                            'columns': ['RSI_14', 'MACD_12_26_9'],
                            'window': 100
                        }
                    }

        Returns:
            Processed DataFrame with all features
        """
        # Step 1: Add log returns
        self.add_log_returns()

        # Step 2: Add multi-scale features (technical indicators)
        if 'multi_scale' in configs:
            self.add_multi_scale_features(configs['multi_scale'])

        # Step 3: Add derivative features
        if 'derivatives' in configs:
            self.add_derivative_features(configs['derivatives'])

        # Step 4: Add regime features
        if 'regime' in configs:
            regime_config = configs['regime']
            self.add_regime_features(
                adx_period=regime_config.get('adx_period', 14),
                atr_period=regime_config.get('atr_period', 14),
                hurst_period=regime_config.get('hurst_period', 100)
            )

        # Step 5: Normalize features
        if 'normalize' in configs:
            normalize_config = configs['normalize']
            self.normalize_features(
                columns=normalize_config.get('columns', []),
                window=normalize_config.get('window', 100)
            )

        return self.df