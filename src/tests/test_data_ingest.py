"""
Tests for crypto data ingestion module.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import polars as pl
import jax.numpy as jnp

from crypto_dp.data.ingest import (
    fetch_ohlcv,
    fetch_coingecko_data,
    load_to_duck,
    get_top_crypto_symbols,
    create_sample_dataset
)


class TestFetchOHLCV:
    """Test OHLCV data fetching functionality."""
    
    @patch('crypto_dp.data.ingest.ccxt')
    def test_fetch_ohlcv_success(self, mock_ccxt):
        """Test successful OHLCV data fetching."""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000.0, 48000.0, 46000.0, 47500.0, 1.5],
            [1640998800000, 47500.0, 48500.0, 47000.0, 48000.0, 2.0],
        ]
        mock_ccxt.binance.return_value = mock_exchange
        
        # Test fetch
        start_time = 1640995200000
        end_time = 1641081600000
        
        df = fetch_ohlcv("BTC/USDT", start_time, end_time, "1h", "binance")
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "symbol" in df.columns
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert df["symbol"][0] == "BTC/USDT"
    
    @patch('crypto_dp.data.ingest.ccxt')
    def test_fetch_ohlcv_empty_data(self, mock_ccxt):
        """Test handling of empty OHLCV data."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_ccxt.binance.return_value = mock_exchange
        
        df = fetch_ohlcv("BTC/USDT", 0, 1000, "1h")
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
    
    @patch('crypto_dp.data.ingest.ccxt')
    def test_fetch_ohlcv_error_handling(self, mock_ccxt):
        """Test error handling in OHLCV fetching."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")
        mock_ccxt.binance.return_value = mock_exchange
        
        with pytest.raises(Exception, match="API Error"):
            fetch_ohlcv("BTC/USDT", 0, 1000, "1h")


class TestFetchCoinGeckoData:
    """Test CoinGecko data fetching functionality."""
    
    @patch('crypto_dp.data.ingest.CoinGeckoAPI')
    def test_fetch_coingecko_data_success(self, mock_cg_class):
        """Test successful CoinGecko data fetching."""
        mock_cg = Mock()
        mock_cg.get_coin_market_chart_by_id.return_value = {
            'prices': [[1640995200000, 47000.0], [1640998800000, 47500.0]],
            'market_caps': [[1640995200000, 900000000000], [1640998800000, 910000000000]],
            'total_volumes': [[1640995200000, 30000000000], [1640998800000, 32000000000]]
        }
        mock_cg_class.return_value = mock_cg
        
        df = fetch_coingecko_data(["bitcoin"], "usd", 7)
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "coin_id" in df.columns
        assert "price" in df.columns
        assert "market_cap" in df.columns
        assert df["coin_id"][0] == "bitcoin"
    
    @patch('crypto_dp.data.ingest.CoinGeckoAPI')
    def test_fetch_coingecko_data_error(self, mock_cg_class):
        """Test error handling in CoinGecko fetching."""
        mock_cg = Mock()
        mock_cg.get_coin_market_chart_by_id.side_effect = Exception("Rate limit")
        mock_cg_class.return_value = mock_cg
        
        df = fetch_coingecko_data(["bitcoin"], "usd", 7)
        
        # Should return empty DataFrame on error
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


class TestDuckDBOperations:
    """Test DuckDB loading and querying operations."""
    
    def test_load_to_duck_replace(self):
        """Test loading data to DuckDB with replace mode."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        # Remove the empty file so DuckDB can create a new database
        os.unlink(db_path)
        
        try:
            # Create test data
            df = pl.DataFrame({
                "timestamp": [1640995200000, 1640998800000],
                "symbol": ["BTC/USDT", "BTC/USDT"],
                "price": [47000.0, 47500.0],
                "volume": [1.5, 2.0]
            })
            
            # Load to DuckDB
            load_to_duck(db_path, df, "test_table", "replace")
            
            # Verify data was loaded
            import duckdb
            con = duckdb.connect(db_path)
            result = con.execute("SELECT COUNT(*) FROM test_table").fetchone()
            assert result[0] == 2
            con.close()
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestHelperFunctions:
    """Test utility and helper functions."""
    
    @patch('crypto_dp.data.ingest.CoinGeckoAPI')
    def test_get_top_crypto_symbols(self, mock_cg_class):
        """Test getting top crypto symbols."""
        mock_cg = Mock()
        mock_cg.get_coins_markets.return_value = [
            {"symbol": "btc", "market_cap": 900000000000},
            {"symbol": "eth", "market_cap": 400000000000},
            {"symbol": "bnb", "market_cap": 80000000000},
        ]
        mock_cg_class.return_value = mock_cg
        
        symbols = get_top_crypto_symbols(3)
        
        assert len(symbols) <= 3
        assert all("/" in symbol for symbol in symbols)  # Should be trading pairs
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
    
    @patch('crypto_dp.data.ingest.CoinGeckoAPI')
    def test_get_top_crypto_symbols_error(self, mock_cg_class):
        """Test error handling in getting top symbols."""
        mock_cg = Mock()
        mock_cg.get_coins_markets.side_effect = Exception("API Error")
        mock_cg_class.return_value = mock_cg
        
        symbols = get_top_crypto_symbols(5)
        
        assert symbols == []  # Should return empty list on error


class TestCreateSampleDataset:
    """Test sample dataset creation functionality."""
    
    @patch('crypto_dp.data.ingest.get_top_crypto_symbols')
    @patch('crypto_dp.data.ingest.fetch_ohlcv')
    def test_create_sample_dataset(self, mock_fetch, mock_symbols):
        """Test creating a sample dataset."""
        # Mock dependencies
        mock_symbols.return_value = ["BTC/USDT", "ETH/USDT"]
        mock_fetch.return_value = pl.DataFrame({
            "timestamp": [1640995200000],
            "symbol": ["BTC/USDT"],
            "open": [47000.0],
            "high": [48000.0],
            "low": [46000.0],
            "close": [47500.0],
            "volume": [1.5],
            "timeframe": ["1h"],
            "exchange": ["binance"],
            "datetime": ["2022-01-01 00:00:00"]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            # Create sample dataset
            create_sample_dataset(db_path, None, 1)  # 1 day of data
            
            # Should not raise exceptions
            assert os.path.exists(db_path)
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__])