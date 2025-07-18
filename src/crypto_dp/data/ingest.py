"""
Data ingestion module for crypto market data.

Supports multiple data sources:
- CCXT for exchange data (public + private REST & websockets)
- CoinGecko for market cap and price data
- DuckDB for efficient storage and querying
"""

from typing import Optional, List, Dict, Any
import datetime as dt
import logging

import ccxt
import duckdb
import polars as pl
from pycoingecko import CoinGeckoAPI


logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbol: str,
    start: int,
    end: int,
    timeframe: str = "1h",
    exchange: str = "binance"
) -> pl.DataFrame:
    """
    Fetch OHLCV data from a crypto exchange.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        start: Start timestamp in milliseconds
        end: End timestamp in milliseconds
        timeframe: Timeframe for candlesticks ('1m', '5m', '1h', '1d', etc.)
        exchange: Exchange name (default: 'binance')
    
    Returns:
        Polars DataFrame with OHLCV data
    """
    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange)
        ex = exchange_class({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # Fetch data
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=start, limit=None)
        
        # Convert to DataFrame
        df = pl.DataFrame(
            data,
            schema=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Filter by end time
        df = df.filter(pl.col("timestamp") < end)
        
        # Add metadata columns
        df = df.with_columns([
            pl.lit(symbol).alias("symbol"),
            pl.lit(timeframe).alias("timeframe"),
            pl.lit(exchange).alias("exchange"),
            pl.from_epoch(pl.col("timestamp") // 1000).alias("datetime")
        ])
        
        logger.info(f"Fetched {len(df)} rows for {symbol} from {exchange}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch {symbol} from {exchange}: {e}")
        raise


def fetch_coingecko_data(
    coin_ids: List[str],
    vs_currency: str = "usd",
    days: int = 365
) -> pl.DataFrame:
    """
    Fetch historical price data from CoinGecko.
    
    Args:
        coin_ids: List of CoinGecko coin IDs
        vs_currency: Currency to price against (default: 'usd')
        days: Number of days of historical data
    
    Returns:
        Polars DataFrame with price, market cap, and volume data
    """
    cg = CoinGeckoAPI()
    
    all_data = []
    for coin_id in coin_ids:
        try:
            data = cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # Extract time series data
            prices = data['prices']
            market_caps = data['market_caps']
            volumes = data['total_volumes']
            
            # Create DataFrame
            df = pl.DataFrame({
                "timestamp": [p[0] for p in prices],
                "price": [p[1] for p in prices],
                "market_cap": [m[1] for m in market_caps],
                "volume": [v[1] for v in volumes],
                "coin_id": coin_id,
                "vs_currency": vs_currency
            })
            
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp") // 1000).alias("datetime")
            )
            
            all_data.append(df)
            logger.info(f"Fetched {len(df)} rows for {coin_id} from CoinGecko")
            
        except Exception as e:
            logger.error(f"Failed to fetch {coin_id} from CoinGecko: {e}")
            continue
    
    if all_data:
        return pl.concat(all_data)
    else:
        return pl.DataFrame()


def load_to_duck(
    db_path: str,
    df: pl.DataFrame,
    table: str,
    mode: str = "replace"
) -> None:
    """
    Load DataFrame to DuckDB.
    
    Args:
        db_path: Path to DuckDB database file
        df: Polars DataFrame to load
        table: Table name
        mode: Load mode ('replace', 'append', 'upsert')
    """
    con = None
    try:
        con = duckdb.connect(db_path)
        
        # Register the DataFrame with DuckDB so it can be referenced in SQL
        con.register("temp_df", df)
        
        if mode == "replace":
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM temp_df")
        elif mode == "append":
            con.execute(f"INSERT INTO {table} SELECT * FROM temp_df")
        elif mode == "upsert":
            # Simple upsert based on timestamp and symbol
            con.execute(f"""
                INSERT INTO {table} 
                SELECT * FROM temp_df 
                WHERE NOT EXISTS (
                    SELECT 1 FROM {table} t2 
                    WHERE t2.timestamp = temp_df.timestamp 
                    AND t2.symbol = temp_df.symbol
                )
            """)
        
        logger.info(f"Loaded {len(df)} rows to {table} in {db_path}")
        
    except Exception as e:
        logger.error(f"Failed to load data to {table}: {e}")
        raise
    finally:
        if con is not None:
            con.close()


def get_top_crypto_symbols(limit: int = 50) -> List[str]:
    """
    Get top cryptocurrency symbols by market cap.
    
    Args:
        limit: Number of top coins to return
    
    Returns:
        List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT', ...])
    """
    try:
        cg = CoinGeckoAPI()
        coins = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=limit,
            page=1
        )
        
        # Convert to exchange symbols (assuming USDT pairs)
        symbols = []
        for coin in coins:
            symbol = coin['symbol'].upper()
            if symbol not in ['USDT', 'USDC', 'BUSD']:  # Skip stablecoins
                symbols.append(f"{symbol}/USDT")
        
        return symbols[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get top crypto symbols: {e}")
        return []


def create_sample_dataset(
    db_path: str = "crypto_data.db",
    symbols: Optional[List[str]] = None,
    days: int = 365
) -> None:
    """
    Create a sample crypto dataset for testing and development.
    
    Args:
        db_path: Path to DuckDB database file
        symbols: List of symbols to fetch (if None, uses top 10)
        days: Number of days of historical data
    """
    if symbols is None:
        symbols = get_top_crypto_symbols(10)
    
    # Calculate time range
    end_time = int(dt.datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    logger.info(f"Creating sample dataset with {len(symbols)} symbols")
    
    for symbol in symbols:
        try:
            # Fetch OHLCV data
            df = fetch_ohlcv(symbol, start_time, end_time, timeframe="1h")
            
            if not df.is_empty():
                load_to_duck(db_path, df, "ohlcv", mode="append")
                
        except Exception as e:
            logger.warning(f"Skipped {symbol}: {e}")
            continue
    
    logger.info(f"Sample dataset created at {db_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    create_sample_dataset()