#!/usr/bin/env python3
"""
Feature Engineering Pipeline Script

This script fetches historical cryptocurrency data from Binance and processes it
through a comprehensive feature engineering pipeline to create features suitable
for quantitative analysis and machine learning.

Usage:
    python scripts/run_feature_engineering.py --symbol BTCUSDT --interval 1h --start_date 2023-01-01 --end_date 2023-12-31
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the src directory to the Python path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_sourcing.binance_loader import BinanceLoader
    from components.feature_engineering import FeatureEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run 'python setup_project.py' from the project root to install dependencies.")
    sys.exit(1)


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the feature engineering pipeline for cryptocurrency data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_feature_engineering.py --symbol BTCUSDT --interval 1h --start_date 2023-01-01 --end_date 2023-12-31
  python scripts/run_feature_engineering.py --symbol ETHUSDT --interval 4h --start_date 2022-06-01 --end_date 2023-06-01
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading pair symbol (e.g., BTCUSDT, ETHUSDT)'
    )

    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        help='Kline interval (e.g., 1h, 4h, 1d)'
    )

    parser.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end_date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )

    return parser.parse_args()


def get_feature_config():
    """
    Define comprehensive feature engineering configuration.

    Returns:
        Dictionary containing configuration for all feature engineering steps
    """
    return {
        'multi_scale': [
            # RSI indicators with different periods
            {'indicator': 'RSI', 'period': 14},
            {'indicator': 'RSI', 'period': 28},

            # MACD indicators with different configurations
            {'indicator': 'MACD', 'fast': 12, 'slow': 26, 'signal': 9},
            {'indicator': 'MACD', 'fast': 24, 'slow': 52, 'signal': 18},

            # EMA indicators for trend identification
            {'indicator': 'EMA', 'period': 200}
        ],

        'regime': {
            'adx_period': 14,
            'atr_period': 14,
            'hurst_period': 100
        },

        'derivatives': [
            # Derivative features for RSI_14
            {
                'column': 'RSI_14',
                'velocity_period': 5,
                'acceleration_period': 5,
                'smoothing_period': 3
            },
            # Derivative features for MACD histogram
            {
                'column': 'MACDh_12_26_9',
                'velocity_period': 5,
                'acceleration_period': 5,
                'smoothing_period': 3
            }
        ],

        'normalize': {
            'columns': ['RSI_14', 'RSI_28', 'ADX_14'],
            'window': 100
        }
    }


def main():
    """Main execution function."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    args = parse_arguments()

    # Create unique file stem based on symbol, interval, and date range
    file_stem = f"{args.symbol}-{args.interval}_{args.start_date}_to_{args.end_date}"

    logger.info("="*60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("="*60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"File stem: {file_stem}")
    logger.info("-"*60)

    try:
        # Step 1: Initialize BinanceLoader
        logger.info("Step 1: Initializing Binance data loader...")
        loader = BinanceLoader()
        logger.info("✓ Binance loader initialized successfully")

        # Step 2: Fetch historical data
        logger.info("Step 2: Fetching historical data from Binance...")
        raw_df = loader.fetch_historical_data(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )
        logger.info(f"✓ Successfully fetched {len(raw_df)} data points")
        logger.info(f"✓ Data range: {raw_df.index[0]} to {raw_df.index[-1]}")

        # Step 3: Save raw data
        logger.info("Step 3: Saving raw data...")
        raw_filename = f"{file_stem}.csv"
        raw_file_path = loader.save_to_csv(raw_df, args.symbol, args.interval, filename=raw_filename)
        logger.info(f"✓ Raw data saved to: {raw_file_path}")

        # Step 4: Initialize FeatureEngine
        logger.info("Step 4: Initializing Feature Engineering pipeline...")
        feature_engine = FeatureEngine(raw_df)
        logger.info("✓ Feature engine initialized successfully")

        # Step 5: Process features
        logger.info("Step 5: Processing features...")
        feature_config = get_feature_config()

        # Log the configuration being used
        logger.info("Feature configuration:")
        logger.info(f"  - Multi-scale indicators: {len(feature_config['multi_scale'])} configs")
        logger.info(f"  - Regime features: ADX({feature_config['regime']['adx_period']}), "
                   f"ATR({feature_config['regime']['atr_period']}), "
                   f"Hurst({feature_config['regime']['hurst_period']})")
        logger.info(f"  - Derivative features: {len(feature_config['derivatives'])} configs")
        logger.info(f"  - Normalization: {len(feature_config['normalize']['columns'])} columns, "
                   f"window={feature_config['normalize']['window']}")

        # Process all features
        processed_df = feature_engine.process_all(feature_config)
        logger.info(f"✓ Feature processing completed")
        logger.info(f"✓ Final feature matrix shape: {processed_df.shape}")
        logger.info(f"✓ Total features created: {len(processed_df.columns)} columns")

        # Step 6: Save processed features
        logger.info("Step 6: Saving processed features...")

        # Ensure processed directory exists
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Create descriptive filename with dynamic naming
        features_filename = f"{file_stem}_features.csv"
        features_filepath = processed_dir / features_filename

        # Save the processed DataFrame
        processed_df.to_csv(features_filepath)
        logger.info(f"✓ Processed features saved to: {features_filepath}")

        # Step 7: Pipeline completion summary
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Summary:")
        logger.info(f"  - Raw data points: {len(raw_df)}")
        logger.info(f"  - Features created: {len(processed_df.columns)}")
        logger.info(f"  - Final data shape: {processed_df.shape}")
        logger.info(f"  - Raw data file: {raw_file_path}")
        logger.info(f"  - Features file: {features_filepath}")
        logger.info("="*60)

        # Display feature columns for reference
        logger.info("Created feature columns:")
        for i, col in enumerate(processed_df.columns, 1):
            logger.info(f"  {i:2d}. {col}")

        logger.info("\nPipeline execution completed successfully! 🎉")

    except Exception as e:
        logger.error("="*60)
        logger.error("PIPELINE FAILED!")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}")
        logger.error("Please check the error message above and try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()