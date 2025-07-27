#!/usr/bin/env python3
"""
QNTI Strategy Tester and Backtesting Engine
Advanced strategy testing with historical data simulation and performance analysis
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import threading
import time
import yfinance as yf
import MetaTrader5 as mt5
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('QNTI_StrategyTester')

class StrategyType(Enum):
    """Trading strategy types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    GRID = "grid"
    MARTINGALE = "martingale"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class PositionType(Enum):
    """Position types"""
    BUY = "buy"
    SELL = "sell"

class TradeStatus(Enum):
    """Trade status"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"

@dataclass
class BacktestTrade:
    """Individual backtest trade record"""
    trade_id: str
    symbol: str
    position_type: PositionType
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    lot_size: float = 0.01
    profit_loss: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    comment: str = ""

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    test_id: str
    ea_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    total_profit: float
    total_loss: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    trades: List[BacktestTrade]
    equity_curve: List[Tuple[datetime, float]]
    parameters: Dict[str, Any]
    execution_time: float
    created_at: datetime

class HistoricalDataManager:
    """Manages historical data retrieval and caching"""
    
    def __init__(self, data_dir: str = "qnti_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir = self.data_dir / "historical_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, 
                          end_date: datetime, source: str = "uploaded", 
                          data_store: Optional['HistoricalDataStore'] = None) -> pd.DataFrame:
        """Get historical data from uploaded files, MT5, or alternative sources"""
        try:
            # Try uploaded data first if available
            if source == "uploaded" and data_store:
                uploaded_data = data_store.get_uploaded_data(symbol, timeframe)
                if uploaded_data is not None and not uploaded_data.empty:
                    # Filter by date range
                    filtered_data = uploaded_data[
                        (uploaded_data.index >= start_date) & 
                        (uploaded_data.index <= end_date)
                    ]
                    if not filtered_data.empty:
                        logger.info(f"Using uploaded data for {symbol} {timeframe}: {len(filtered_data)} bars")
                        return filtered_data
                    else:
                        logger.warning(f"No uploaded data in date range for {symbol} {timeframe}")
            
            # Try MT5 second
            if source in ["mt5", "uploaded"] and mt5.initialize():  # type: ignore
                data = self._get_mt5_data(symbol, timeframe, start_date, end_date)
                if data is not None and not data.empty:
                    return data
                    
            # Fallback to Yahoo Finance
            logger.warning(f"MT5 data not available for {symbol}, using Yahoo Finance")
            return self._get_yahoo_data(symbol, timeframe, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    def _get_mt5_data(self, symbol: str, timeframe: str, start_date: datetime, 
                      end_date: datetime) -> Optional[pd.DataFrame]:
        """Get data from MT5"""
        try:
            # Map timeframe to MT5 constant
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            if timeframe not in timeframe_map:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                return None
                
            mt5_timeframe = timeframe_map[timeframe]
            
            # Get rates
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No MT5 data available for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting MT5 data: {e}")
            return None
    
    def _get_yahoo_data(self, symbol: str, timeframe: str, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Get data from Yahoo Finance as fallback"""
        try:
            # Map forex symbols to Yahoo format
            if len(symbol) == 6:  # Forex pair
                yahoo_symbol = f"{symbol[:3]}{symbol[3:]}=X"
            else:
                yahoo_symbol = symbol
                
            # Map timeframe to Yahoo interval
            interval_map = {
                "M1": "1m",
                "M5": "5m",
                "M15": "15m",
                "M30": "30m",
                "H1": "1h",
                "H4": "4h",
                "D1": "1d"
            }
            
            interval = interval_map.get(timeframe, "1h")
            
            # Download data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No Yahoo Finance data available for {symbol}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data: {e}")
            return pd.DataFrame()

class HistoricalDataStore:
    """Manages uploaded historical data files from MT5 export script"""
    
    def __init__(self, data_dir: str = "qnti_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.uploads_dir = self.data_dir / "uploaded_data"
        self.uploads_dir.mkdir(exist_ok=True)
        
        # Database for uploaded data metadata
        self.db_path = self.data_dir / "uploaded_data.db"
        self._init_database()
        
        # Cache for parsed data
        self._data_cache = {}
        self._cache_max_size = 10  # Maximum number of datasets to cache
        
        logger.info(f"HistoricalDataStore initialized with uploads dir: {self.uploads_dir}")
    
    def _init_database(self):
        """Initialize database for uploaded data metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                symbol TEXT,
                period INTEGER,
                timeframe TEXT,
                bars INTEGER,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                file_size INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT,
                metadata TEXT,
                is_valid BOOLEAN DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def upload_file(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Process uploaded historical data file"""
        try:
            # Validate file extension
            if not filename.endswith('.json'):
                return {"success": False, "error": "Only JSON files are supported"}
            
            # Save file to uploads directory
            file_path = self.uploads_dir / filename
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Parse and validate JSON content
            try:
                data = json.loads(file_content.decode('utf-8'))
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid JSON format: {str(e)}"}
            
            # Validate data structure
            validation_result = self._validate_data_structure(data)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}
            
            # Extract metadata
            metadata = self._extract_metadata(data)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO uploaded_data 
                (filename, symbol, period, timeframe, bars, start_date, end_date, 
                 file_size, file_path, metadata, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                metadata["symbol"],
                metadata["period"],
                metadata["timeframe"],
                metadata["bars"],
                metadata["start_date"],
                metadata["end_date"],
                len(file_content),
                str(file_path),
                json.dumps(metadata),
                True
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully uploaded {filename}: {metadata['symbol']} {metadata['timeframe']} with {metadata['bars']} bars")
            
            return {
                "success": True,
                "filename": filename,
                "symbol": metadata["symbol"],
                "timeframe": metadata["timeframe"],
                "bars": metadata["bars"],
                "start_date": metadata["start_date"],
                "end_date": metadata["end_date"]
            }
            
        except Exception as e:
            logger.error(f"Error uploading file {filename}: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_data_structure(self, data: Dict) -> Dict[str, Any]:
        """Validate the structure of uploaded JSON data"""
        required_fields = [
            "symbol", "period", "bars", "time", "open", "high", "low", "close", "volume"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return {"valid": False, "error": f"Missing required field: {field}"}
        
        # Check if arrays have consistent length
        bars = data["bars"]
        array_fields = ["time", "open", "high", "low", "close", "volume"]
        
        for field in array_fields:
            if len(data[field]) != bars:
                return {"valid": False, "error": f"Array length mismatch for {field}: expected {bars}, got {len(data[field])}"}
        
        # Check if we have enough data
        if bars < 100:
            return {"valid": False, "error": f"Not enough data: {bars} bars (minimum 100 required)"}
        
        return {"valid": True}
    
    def _extract_metadata(self, data: Dict) -> Dict[str, Any]:
        """Extract metadata from uploaded data"""
        # Convert time format (minutes since millennium) to datetime
        millennium = datetime(2000, 1, 1)
        times = [millennium + timedelta(minutes=int(t)) for t in data["time"]]
        
        # Map period to timeframe string
        period_map = {
            1: "M1", 5: "M5", 15: "M15", 30: "M30", 
            60: "H1", 240: "H4", 1440: "D1"
        }
        
        timeframe = period_map.get(data["period"], f"T{data['period']}")
        
        return {
            "symbol": data["symbol"],
            "period": data["period"],
            "timeframe": timeframe,
            "bars": data["bars"],
            "start_date": times[0].isoformat(),
            "end_date": times[-1].isoformat(),
            "digits": data.get("digits", 5),
            "spread": data.get("spread", 0),
            "commission": data.get("commission", 0),
            "point_value": data.get("pointValue", 1),
            "lot_size": data.get("lotSize", 100000),
            "terminal": data.get("terminal", "MetaTrader"),
            "server": data.get("server", "Unknown"),
            "company": data.get("company", "Unknown")
        }
    
    def get_uploaded_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get uploaded data for a specific symbol and timeframe"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._data_cache:
                logger.debug(f"Returning cached data for {symbol} {timeframe}")
                return self._data_cache[cache_key]
            
            # Query database for matching data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Map timeframe to period
            timeframe_map = {
                "M1": 1, "M5": 5, "M15": 15, "M30": 30,
                "H1": 60, "H4": 240, "D1": 1440
            }
            period = timeframe_map.get(timeframe)
            
            if period is None:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                return None
            
            cursor.execute('''
                SELECT file_path FROM uploaded_data 
                WHERE symbol = ? AND period = ? AND is_valid = 1
                ORDER BY upload_date DESC LIMIT 1
            ''', (symbol, period))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                logger.debug(f"No uploaded data found for {symbol} {timeframe}")
                return None
            
            file_path = result[0]
            
            # Load and parse JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = self._json_to_dataframe(data)
            
            # Cache the result
            self._manage_cache(cache_key, df)
            
            logger.info(f"Loaded uploaded data for {symbol} {timeframe}: {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Error loading uploaded data for {symbol} {timeframe}: {e}")
            return None
    
    def _json_to_dataframe(self, data: Dict) -> pd.DataFrame:
        """Convert JSON data to pandas DataFrame"""
        # Convert time format
        millennium = datetime(2000, 1, 1)
        times = [millennium + timedelta(minutes=int(t)) for t in data["time"]]
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': times,
            'Open': [float(x) for x in data["open"]],
            'High': [float(x) for x in data["high"]],
            'Low': [float(x) for x in data["low"]],
            'Close': [float(x) for x in data["close"]],
            'Volume': [int(x) for x in data["volume"]]
        })
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        return df
    
    def _manage_cache(self, key: str, data: pd.DataFrame):
        """Manage cache size and add new data"""
        if len(self._data_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._data_cache))
            del self._data_cache[oldest_key]
        
        self._data_cache[key] = data
    
    def get_available_data(self) -> List[Dict[str, Any]]:
        """Get list of all available uploaded data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT filename, symbol, timeframe, bars, start_date, end_date, 
                       upload_date, file_size, is_valid
                FROM uploaded_data
                ORDER BY upload_date DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            data_list = []
            for row in results:
                data_list.append({
                    "filename": row[0],
                    "symbol": row[1],
                    "timeframe": row[2],
                    "bars": row[3],
                    "start_date": row[4],
                    "end_date": row[5],
                    "upload_date": row[6],
                    "file_size": row[7],
                    "is_valid": bool(row[8])
                })
            
            return data_list
            
        except Exception as e:
            logger.error(f"Error getting available data: {e}")
            return []
    
    def delete_uploaded_data(self, filename: str) -> bool:
        """Delete uploaded data file and database entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get file path
            cursor.execute('SELECT file_path FROM uploaded_data WHERE filename = ?', (filename,))
            result = cursor.fetchone()
            
            if result:
                file_path = Path(result[0])
                
                # Delete file if it exists
                if file_path.exists():
                    file_path.unlink()
                
                # Delete database entry
                cursor.execute('DELETE FROM uploaded_data WHERE filename = ?', (filename,))
                conn.commit()
                
                # Clear cache
                cache_key = f"{filename.split('.')[0]}"  # Remove extension
                if cache_key in self._data_cache:
                    del self._data_cache[cache_key]
                
                logger.info(f"Deleted uploaded data: {filename}")
                conn.close()
                return True
            
            conn.close()
            return False
            
        except Exception as e:
            logger.error(f"Error deleting uploaded data {filename}: {e}")
            return False

class BacktestEngine:
    """Core backtesting engine"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.trades = []
        self.open_positions = []
        self.equity_curve = []
        
    def reset(self):
        """Reset engine state"""
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        self.trades = []
        self.open_positions = []
        self.equity_curve = []
        
    def generate_strategy_signals(self, data: pd.DataFrame, strategy_type: StrategyType, 
                                 parameters: Dict[str, Any]) -> List[Dict]:
        """Generate trading signals based on strategy type"""
        signals = []
        
        if strategy_type == StrategyType.TREND_FOLLOWING:
            signals = self._trend_following_signals(data, parameters)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            signals = self._mean_reversion_signals(data, parameters)
        elif strategy_type == StrategyType.BREAKOUT:
            signals = self._breakout_signals(data, parameters)
        elif strategy_type == StrategyType.SCALPING:
            signals = self._scalping_signals(data, parameters)
        else:
            logger.warning(f"Strategy type {strategy_type} not implemented")
            
        return signals
    
    def _trend_following_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict]:
        """Trend following strategy signals"""
        signals = []
        
        # Create a copy to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Parameters - Convert to integers for pandas rolling window
        ma_short = int(params.get('ma_short', 20))
        ma_long = int(params.get('ma_long', 50))
        
        # Calculate moving averages
        data['MA_Short'] = data['Close'].rolling(window=ma_short).mean()
        data['MA_Long'] = data['Close'].rolling(window=ma_long).mean()
        
        # Generate signals
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i-1]
            
            # Buy signal: MA short crosses above MA long
            if (current_row['MA_Short'] > current_row['MA_Long'] and 
                prev_row['MA_Short'] <= prev_row['MA_Long']):
                signals.append({
                    'time': current_row.name,
                    'type': 'buy',
                    'price': current_row['Close'],
                    'stop_loss': current_row['Close'] * (1 - params.get('stop_loss_pct', 0.02)),
                    'take_profit': current_row['Close'] * (1 + params.get('take_profit_pct', 0.04))
                })
            
            # Sell signal: MA short crosses below MA long
            elif (current_row['MA_Short'] < current_row['MA_Long'] and 
                  prev_row['MA_Short'] >= prev_row['MA_Long']):
                signals.append({
                    'time': current_row.name,
                    'type': 'sell',
                    'price': current_row['Close'],
                    'stop_loss': current_row['Close'] * (1 + params.get('stop_loss_pct', 0.02)),
                    'take_profit': current_row['Close'] * (1 - params.get('take_profit_pct', 0.04))
                })
                
        return signals
    
    def _mean_reversion_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict]:
        """Mean reversion strategy signals"""
        signals = []
        
        # Create a copy to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Parameters - Convert to integers for pandas rolling window
        bb_period = int(params.get('bb_period', 20))
        bb_std = params.get('bb_std', 2.0)
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
        data['BB_Std'] = data['Close'].rolling(window=bb_period).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * bb_std)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * bb_std)
        
        # Generate signals
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            
            # Buy signal: price touches lower band
            if current_row['Close'] <= current_row['BB_Lower']:
                signals.append({
                    'time': current_row.name,
                    'type': 'buy',
                    'price': current_row['Close'],
                    'stop_loss': current_row['BB_Lower'] * (1 - params.get('stop_loss_pct', 0.01)),
                    'take_profit': current_row['BB_Middle']
                })
            
            # Sell signal: price touches upper band
            elif current_row['Close'] >= current_row['BB_Upper']:
                signals.append({
                    'time': current_row.name,
                    'type': 'sell',
                    'price': current_row['Close'],
                    'stop_loss': current_row['BB_Upper'] * (1 + params.get('stop_loss_pct', 0.01)),
                    'take_profit': current_row['BB_Middle']
                })
                
        return signals
    
    def _breakout_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict]:
        """Breakout strategy signals"""
        signals = []
        
        # Create a copy to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Parameters - Convert to integers for pandas rolling window
        breakout_period = int(params.get('breakout_period', 20))
        
        # Calculate breakout levels
        data['High_Max'] = data['High'].rolling(window=breakout_period).max()
        data['Low_Min'] = data['Low'].rolling(window=breakout_period).min()
        
        # Generate signals
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i-1]
            
            # Buy signal: price breaks above recent high
            if (current_row['Close'] > prev_row['High_Max'] and 
                prev_row['Close'] <= prev_row['High_Max']):
                signals.append({
                    'time': current_row.name,
                    'type': 'buy',
                    'price': current_row['Close'],
                    'stop_loss': prev_row['Low_Min'],
                    'take_profit': current_row['Close'] + (current_row['Close'] - prev_row['Low_Min'])
                })
            
            # Sell signal: price breaks below recent low
            elif (current_row['Close'] < prev_row['Low_Min'] and 
                  prev_row['Close'] >= prev_row['Low_Min']):
                signals.append({
                    'time': current_row.name,
                    'type': 'sell',
                    'price': current_row['Close'],
                    'stop_loss': prev_row['High_Max'],
                    'take_profit': current_row['Close'] - (prev_row['High_Max'] - current_row['Close'])
                })
                
        return signals
    
    def _scalping_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> List[Dict]:
        """Scalping strategy signals"""
        signals = []
        
        # Create a copy to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Parameters - Convert to integers for pandas rolling window
        rsi_period = int(params.get('rsi_period', 14))
        rsi_overbought = params.get('rsi_overbought', 70)
        rsi_oversold = params.get('rsi_oversold', 30)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            
            # Buy signal: RSI oversold
            if current_row['RSI'] < rsi_oversold:
                signals.append({
                    'time': current_row.name,
                    'type': 'buy',
                    'price': current_row['Close'],
                    'stop_loss': current_row['Close'] * (1 - params.get('stop_loss_pct', 0.005)),
                    'take_profit': current_row['Close'] * (1 + params.get('take_profit_pct', 0.01))
                })
            
            # Sell signal: RSI overbought
            elif current_row['RSI'] > rsi_overbought:
                signals.append({
                    'time': current_row.name,
                    'type': 'sell',
                    'price': current_row['Close'],
                    'stop_loss': current_row['Close'] * (1 + params.get('stop_loss_pct', 0.005)),
                    'take_profit': current_row['Close'] * (1 - params.get('take_profit_pct', 0.01))
                })
                
        return signals
    
    def execute_trade(self, signal: Dict, lot_size: float = 0.01) -> BacktestTrade:
        """Execute a trade based on signal"""
        trade_id = str(uuid.uuid4())
        
        trade = BacktestTrade(
            trade_id=trade_id,
            symbol=signal.get('symbol', 'UNKNOWN'),
            position_type=PositionType.BUY if signal['type'] == 'buy' else PositionType.SELL,
            entry_price=signal['price'],
            entry_time=signal['time'],
            lot_size=lot_size,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            status=TradeStatus.OPEN
        )
        
        self.open_positions.append(trade)
        return trade
    
    def close_trade(self, trade: BacktestTrade, exit_price: float, exit_time: datetime) -> float:
        """Close a trade and calculate profit/loss"""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = TradeStatus.CLOSED
        
        # Calculate profit/loss
        if trade.position_type == PositionType.BUY:
            profit = (exit_price - trade.entry_price) * trade.lot_size * 100000  # Assuming forex
        else:
            profit = (trade.entry_price - exit_price) * trade.lot_size * 100000
            
        trade.profit_loss = profit
        self.current_balance += profit
        
        # Remove from open positions
        if trade in self.open_positions:
            self.open_positions.remove(trade)
            
        self.trades.append(trade)
        return profit
    
    def update_equity(self, current_time: datetime, current_prices: Dict[str, float]):
        """Update equity curve based on current prices"""
        floating_pnl = 0
        
        for trade in self.open_positions:
            if trade.symbol in current_prices:
                current_price = current_prices[trade.symbol]
                if trade.position_type == PositionType.BUY:
                    floating_pnl += (current_price - trade.entry_price) * trade.lot_size * 100000
                else:
                    floating_pnl += (trade.entry_price - current_price) * trade.lot_size * 100000
                    
        self.equity = self.current_balance + floating_pnl
        self.equity_curve.append((current_time, self.equity))
        
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
            
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return {}
            
        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.profit_loss > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = sum(t.profit_loss for t in closed_trades if t.profit_loss > 0)
        total_loss = abs(sum(t.profit_loss for t in closed_trades if t.profit_loss < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Average metrics
        average_win = total_profit / winning_trades if winning_trades > 0 else 0
        average_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        # Largest win/loss
        largest_win = max((t.profit_loss for t in closed_trades), default=0)
        largest_loss = min((t.profit_loss for t in closed_trades), default=0)
        
        # Drawdown calculation
        max_drawdown = 0
        peak = self.initial_balance
        
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        max_drawdown_percent = (max_drawdown / peak) * 100 if peak > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1][1]
                curr_equity = self.equity_curve[i][1]
                returns.append((curr_equity - prev_equity) / prev_equity)
                
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in closed_trades:
            if trade.profit_loss > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_percent,
            'sharpe_ratio': sharpe_ratio,
            'consecutive_wins': max_consecutive_wins,
            'consecutive_losses': max_consecutive_losses,
            'final_balance': self.current_balance,
            'total_return': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        }

class QNTIStrategyTester:
    """Main strategy tester class"""
    
    def __init__(self, data_dir: str = "qnti_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_manager = HistoricalDataManager(data_dir)
        self.data_store = HistoricalDataStore(data_dir)
        self.backtest_engine = BacktestEngine()
        
        # Database for storing results
        self.db_path = self.data_dir / "strategy_tests.db"
        self._init_database()
        
        logger.info("QNTI Strategy Tester initialized")
    
    def _init_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                test_id TEXT PRIMARY KEY,
                ea_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                initial_balance REAL,
                final_balance REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                max_drawdown REAL,
                max_drawdown_percent REAL,
                sharpe_ratio REAL,
                total_profit REAL,
                total_loss REAL,
                average_win REAL,
                average_loss REAL,
                largest_win REAL,
                largest_loss REAL,
                consecutive_wins INTEGER,
                consecutive_losses INTEGER,
                parameters TEXT,
                execution_time REAL,
                created_at TEXT
            )
        ''')
        
        # Individual trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_trades (
                trade_id TEXT PRIMARY KEY,
                test_id TEXT,
                symbol TEXT,
                position_type TEXT,
                entry_price REAL,
                exit_price REAL,
                entry_time TEXT,
                exit_time TEXT,
                lot_size REAL,
                profit_loss REAL,
                status TEXT,
                stop_loss REAL,
                take_profit REAL,
                commission REAL,
                swap REAL,
                comment TEXT,
                FOREIGN KEY (test_id) REFERENCES backtest_results (test_id)
            )
        ''')
        
        # Equity curve table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                timestamp TEXT,
                equity REAL,
                FOREIGN KEY (test_id) REFERENCES backtest_results (test_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def run_backtest(self, ea_name: str, symbol: str, timeframe: str, 
                    start_date: datetime, end_date: datetime, 
                    strategy_type: StrategyType, parameters: Dict[str, Any],
                    initial_balance: float = 10000.0) -> BacktestResult:
        """Run a complete backtest"""
        
        start_time = time.time()
        test_id = str(uuid.uuid4())
        
        logger.info(f"Starting backtest for {ea_name} on {symbol} {timeframe}")
        
        try:
            # Get historical data
            data = self.data_manager.get_historical_data(symbol, timeframe, start_date, end_date, 
                                                        data_store=self.data_store)
            
            if data.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Reset backtest engine
            self.backtest_engine.reset()
            self.backtest_engine.initial_balance = initial_balance
            self.backtest_engine.current_balance = initial_balance
            
            # Generate trading signals
            signals = self.backtest_engine.generate_strategy_signals(data, strategy_type, parameters)
            
            logger.info(f"Generated {len(signals)} trading signals")
            
            # Execute trades
            lot_size = parameters.get('lot_size', 0.01)
            
            for signal in signals:
                signal['symbol'] = symbol
                
                # Execute trade
                trade = self.backtest_engine.execute_trade(signal, lot_size)
                
                # Check for stop loss/take profit and close trades
                self._process_open_trades(data, signal['time'])
                
                # Update equity curve
                current_prices = {symbol: signal['price']}
                self.backtest_engine.update_equity(signal['time'], current_prices)
            
            # Close any remaining open trades at the end
            if self.backtest_engine.open_positions:
                final_price = data.iloc[-1]['Close']
                final_time = pd.Timestamp(data.index[-1]).to_pydatetime()
                
                for trade in self.backtest_engine.open_positions.copy():
                    self.backtest_engine.close_trade(trade, final_price, final_time)
            
            # Calculate performance metrics
            performance = self.backtest_engine.calculate_performance_metrics()
            
            # Create result object
            result = BacktestResult(
                test_id=test_id,
                ea_name=ea_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance,
                final_balance=self.backtest_engine.current_balance,
                total_trades=performance.get('total_trades', 0),
                winning_trades=performance.get('winning_trades', 0),
                losing_trades=performance.get('losing_trades', 0),
                win_rate=performance.get('win_rate', 0),
                profit_factor=performance.get('profit_factor', 0),
                max_drawdown=performance.get('max_drawdown', 0),
                max_drawdown_percent=performance.get('max_drawdown_percent', 0),
                sharpe_ratio=performance.get('sharpe_ratio', 0),
                total_profit=performance.get('total_profit', 0),
                total_loss=performance.get('total_loss', 0),
                average_win=performance.get('average_win', 0),
                average_loss=performance.get('average_loss', 0),
                largest_win=performance.get('largest_win', 0),
                largest_loss=performance.get('largest_loss', 0),
                consecutive_wins=performance.get('consecutive_wins', 0),
                consecutive_losses=performance.get('consecutive_losses', 0),
                trades=self.backtest_engine.trades,
                equity_curve=self.backtest_engine.equity_curve,
                parameters=parameters,
                execution_time=time.time() - start_time,
                created_at=datetime.now()
            )
            
            # Save result to database
            self._save_backtest_result(result)
            
            logger.info(f"Backtest completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _process_open_trades(self, data: pd.DataFrame, current_time: datetime):
        """Process open trades for stop loss/take profit"""
        for trade in self.backtest_engine.open_positions.copy():
            # Find current price data
            try:
                current_data = data.loc[current_time]
                
                # Check stop loss
                if trade.stop_loss:
                    if trade.position_type == PositionType.BUY and current_data['Low'] <= trade.stop_loss:
                        self.backtest_engine.close_trade(trade, trade.stop_loss, current_time)
                        continue
                    elif trade.position_type == PositionType.SELL and current_data['High'] >= trade.stop_loss:
                        self.backtest_engine.close_trade(trade, trade.stop_loss, current_time)
                        continue
                
                # Check take profit
                if trade.take_profit:
                    if trade.position_type == PositionType.BUY and current_data['High'] >= trade.take_profit:
                        self.backtest_engine.close_trade(trade, trade.take_profit, current_time)
                        continue
                    elif trade.position_type == PositionType.SELL and current_data['Low'] <= trade.take_profit:
                        self.backtest_engine.close_trade(trade, trade.take_profit, current_time)
                        continue
                        
            except KeyError:
                # Current time not in data, skip
                continue
    
    def _save_backtest_result(self, result: BacktestResult):
        """Save backtest result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Save main result
            cursor.execute('''
                INSERT OR REPLACE INTO backtest_results (
                    test_id, ea_name, symbol, timeframe, start_date, end_date,
                    initial_balance, final_balance, total_trades, winning_trades,
                    losing_trades, win_rate, profit_factor, max_drawdown,
                    max_drawdown_percent, sharpe_ratio, total_profit, total_loss,
                    average_win, average_loss, largest_win, largest_loss,
                    consecutive_wins, consecutive_losses, parameters,
                    execution_time, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.test_id, result.ea_name, result.symbol, result.timeframe,
                result.start_date.isoformat(), result.end_date.isoformat(),
                result.initial_balance, result.final_balance, result.total_trades,
                result.winning_trades, result.losing_trades, result.win_rate,
                result.profit_factor, result.max_drawdown, result.max_drawdown_percent,
                result.sharpe_ratio, result.total_profit, result.total_loss,
                result.average_win, result.average_loss, result.largest_win,
                result.largest_loss, result.consecutive_wins, result.consecutive_losses,
                json.dumps(result.parameters), result.execution_time,
                result.created_at.isoformat()
            ))
            
            # Save individual trades
            for trade in result.trades:
                cursor.execute('''
                    INSERT OR REPLACE INTO backtest_trades (
                        trade_id, test_id, symbol, position_type, entry_price,
                        exit_price, entry_time, exit_time, lot_size, profit_loss,
                        status, stop_loss, take_profit, commission, swap, comment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.trade_id, result.test_id, trade.symbol, trade.position_type.value,
                    trade.entry_price, trade.exit_price,
                    trade.entry_time.isoformat() if trade.entry_time else None,
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.lot_size, trade.profit_loss, trade.status.value,
                    trade.stop_loss, trade.take_profit, trade.commission,
                    trade.swap, trade.comment
                ))
            
            # Save equity curve
            for timestamp, equity in result.equity_curve:
                cursor.execute('''
                    INSERT INTO equity_curve (test_id, timestamp, equity)
                    VALUES (?, ?, ?)
                ''', (result.test_id, timestamp.isoformat(), equity))
            
            conn.commit()
            logger.info(f"Backtest result saved: {result.test_id}")
            
        except Exception as e:
            logger.error(f"Error saving backtest result: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_backtest_results(self, limit: int = 100) -> List[Dict]:
        """Get stored backtest results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM backtest_results
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            result = {
                'test_id': row[0],
                'ea_name': row[1],
                'symbol': row[2],
                'timeframe': row[3],
                'start_date': row[4],
                'end_date': row[5],
                'initial_balance': row[6],
                'final_balance': row[7],
                'total_trades': row[8],
                'winning_trades': row[9],
                'losing_trades': row[10],
                'win_rate': row[11],
                'profit_factor': row[12],
                'max_drawdown': row[13],
                'max_drawdown_percent': row[14],
                'sharpe_ratio': row[15],
                'total_profit': row[16],
                'total_loss': row[17],
                'average_win': row[18],
                'average_loss': row[19],
                'largest_win': row[20],
                'largest_loss': row[21],
                'consecutive_wins': row[22],
                'consecutive_losses': row[23],
                'parameters': json.loads(row[24]) if row[24] else {},
                'execution_time': row[25],
                'created_at': row[26]
            }
            results.append(result)
        
        conn.close()
        return results
    
    def get_backtest_details(self, test_id: str) -> Optional[Dict]:
        """Get detailed backtest result including trades and equity curve"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get main result
        cursor.execute('SELECT * FROM backtest_results WHERE test_id = ?', (test_id,))
        result_row = cursor.fetchone()
        
        if not result_row:
            conn.close()
            return None
        
        # Get trades
        cursor.execute('SELECT * FROM backtest_trades WHERE test_id = ?', (test_id,))
        trades_data = cursor.fetchall()
        
        # Get equity curve
        cursor.execute('SELECT timestamp, equity FROM equity_curve WHERE test_id = ? ORDER BY timestamp', (test_id,))
        equity_data = cursor.fetchall()
        
        conn.close()
        
        result = {
            'test_id': result_row[0],
            'ea_name': result_row[1],
            'symbol': result_row[2],
            'timeframe': result_row[3],
            'start_date': result_row[4],
            'end_date': result_row[5],
            'initial_balance': result_row[6],
            'final_balance': result_row[7],
            'total_trades': result_row[8],
            'winning_trades': result_row[9],
            'losing_trades': result_row[10],
            'win_rate': result_row[11],
            'profit_factor': result_row[12],
            'max_drawdown': result_row[13],
            'max_drawdown_percent': result_row[14],
            'sharpe_ratio': result_row[15],
            'total_profit': result_row[16],
            'total_loss': result_row[17],
            'average_win': result_row[18],
            'average_loss': result_row[19],
            'largest_win': result_row[20],
            'largest_loss': result_row[21],
            'consecutive_wins': result_row[22],
            'consecutive_losses': result_row[23],
            'parameters': json.loads(result_row[24]) if result_row[24] else {},
            'execution_time': result_row[25],
            'created_at': result_row[26],
            'trades': [
                {
                    'trade_id': trade[0],
                    'symbol': trade[2],
                    'position_type': trade[3],
                    'entry_price': trade[4],
                    'exit_price': trade[5],
                    'entry_time': trade[6],
                    'exit_time': trade[7],
                    'lot_size': trade[8],
                    'profit_loss': trade[9],
                    'status': trade[10],
                    'stop_loss': trade[11],
                    'take_profit': trade[12],
                    'commission': trade[13],
                    'swap': trade[14],
                    'comment': trade[15]
                }
                for trade in trades_data
            ],
            'equity_curve': [
                {
                    'timestamp': equity[0],
                    'equity': equity[1]
                }
                for equity in equity_data
            ]
        }
        
        return result
    
    def delete_backtest_result(self, test_id: str) -> bool:
        """Delete a backtest result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM backtest_results WHERE test_id = ?', (test_id,))
            cursor.execute('DELETE FROM backtest_trades WHERE test_id = ?', (test_id,))
            cursor.execute('DELETE FROM equity_curve WHERE test_id = ?', (test_id,))
            conn.commit()
            
            deleted = cursor.rowcount > 0
            logger.info(f"Deleted backtest result: {test_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting backtest result: {e}")
            conn.rollback()
            return False
        finally:
            conn.close() 