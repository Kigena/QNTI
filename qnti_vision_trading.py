#!/usr/bin/env python3
"""
QNTI Vision Trading Module
Automated trading based on Vision AI analysis with SMC (Smart Money Concepts)
"""

import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import threading
import time
import sqlite3
from pathlib import Path
import uuid

# Import from existing QNTI system
from qnti_core_system import Trade, TradeSource, TradeStatus, QNTITradeManager
from qnti_mt5_integration import QNTIMT5Bridge

logger = logging.getLogger(__name__)

class VisionTradeStatus(Enum):
    PENDING = "pending"
    WAITING_ENTRY = "waiting_entry"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class SMCAnalysis:
    """Structured SMC Analysis Data"""
    market_structure: Dict
    smc_elements: Dict
    directional_bias: Dict
    trading_plan: Dict
    risk_factors: Dict
    timestamp: datetime
    symbol: str
    timeframe: str
    confidence: float

@dataclass
class VisionChart:
    """Represents an uploaded chart with analysis"""
    id: str
    filename: str
    uploaded_at: datetime
    analysis_id: Optional[str] = None
    analysis_data: Optional[Dict] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    file_path: Optional[str] = None

@dataclass 
class VisionTrade:
    """Vision trading specific trade with tracking"""
    trade_id: str
    chart_id: str
    analysis_id: str
    symbol: str
    trade_type: str  # BUY, SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP
    lot_size: float
    open_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    entry_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    mt5_ticket: Optional[int] = None
    status: str = "PENDING"  # PENDING, FILLED, PARTIAL, CLOSED, CANCELLED
    current_price: Optional[float] = None
    profit_loss: Optional[float] = None
    confidence: Optional[float] = None
    auto_trade: bool = False
    risk_percent: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class VisionTradingDatabase:
    """Database manager for vision trading data"""
    
    def __init__(self, db_path: str = "qnti_data/vision_trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vision_charts (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    uploaded_at TIMESTAMP NOT NULL,
                    analysis_id TEXT,
                    analysis_data TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    file_path TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vision_trades (
                    trade_id TEXT PRIMARY KEY,
                    chart_id TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    lot_size REAL NOT NULL,
                    open_price REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    entry_reason TEXT,
                    created_at TIMESTAMP NOT NULL,
                    mt5_ticket INTEGER,
                    status TEXT NOT NULL,
                    current_price REAL,
                    profit_loss REAL,
                    confidence REAL,
                    auto_trade BOOLEAN,
                    risk_percent REAL,
                    FOREIGN KEY (chart_id) REFERENCES vision_charts (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vision_trade_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    price REAL NOT NULL,
                    profit_loss REAL,
                    status TEXT,
                    notes TEXT,
                    FOREIGN KEY (trade_id) REFERENCES vision_trades (trade_id)
                )
            """)
            
            conn.commit()
    
    def save_chart(self, chart: VisionChart):
        """Save chart to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO vision_charts 
                (id, filename, uploaded_at, analysis_id, analysis_data, symbol, timeframe, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chart.id, chart.filename, chart.uploaded_at,
                chart.analysis_id, 
                json.dumps(chart.analysis_data) if chart.analysis_data else None,
                chart.symbol, chart.timeframe, chart.file_path
            ))
            conn.commit()
    
    def get_chart(self, chart_id: str) -> Optional[VisionChart]:
        """Get chart by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, filename, uploaded_at, analysis_id, analysis_data, symbol, timeframe, file_path
                FROM vision_charts WHERE id = ?
            """, (chart_id,))
            row = cursor.fetchone()
            
            if row:
                return VisionChart(
                    id=row[0],
                    filename=row[1],
                    uploaded_at=datetime.fromisoformat(row[2]),
                    analysis_id=row[3],
                    analysis_data=json.loads(row[4]) if row[4] else None,
                    symbol=row[5],
                    timeframe=row[6],
                    file_path=row[7]
                )
        return None
    
    def get_all_charts(self) -> List[VisionChart]:
        """Get all charts"""
        charts = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, filename, uploaded_at, analysis_id, analysis_data, symbol, timeframe, file_path
                FROM vision_charts ORDER BY uploaded_at DESC
            """)
            
            for row in cursor.fetchall():
                charts.append(VisionChart(
                    id=row[0],
                    filename=row[1],
                    uploaded_at=datetime.fromisoformat(row[2]),
                    analysis_id=row[3],
                    analysis_data=json.loads(row[4]) if row[4] else None,
                    symbol=row[5],
                    timeframe=row[6],
                    file_path=row[7]
                ))
        
        return charts
    
    def save_trade(self, trade: VisionTrade):
        """Save trade to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO vision_trades 
                (trade_id, chart_id, analysis_id, symbol, trade_type, lot_size, open_price,
                 stop_loss, take_profit_1, take_profit_2, entry_reason, created_at, mt5_ticket,
                 status, current_price, profit_loss, confidence, auto_trade, risk_percent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id, trade.chart_id, trade.analysis_id, trade.symbol,
                trade.trade_type, trade.lot_size, trade.open_price, trade.stop_loss,
                trade.take_profit_1, trade.take_profit_2, trade.entry_reason,
                trade.created_at.isoformat(), trade.mt5_ticket, trade.status,
                trade.current_price, trade.profit_loss, trade.confidence,
                trade.auto_trade, trade.risk_percent
            ))
            conn.commit()
    
    def get_trades_by_chart(self, chart_id: str) -> List[VisionTrade]:
        """Get all trades for a chart"""
        trades = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT trade_id, chart_id, analysis_id, symbol, trade_type, lot_size, open_price,
                       stop_loss, take_profit_1, take_profit_2, entry_reason, created_at, mt5_ticket,
                       status, current_price, profit_loss, confidence, auto_trade, risk_percent
                FROM vision_trades WHERE chart_id = ? ORDER BY created_at DESC
            """, (chart_id,))
            
            for row in cursor.fetchall():
                trades.append(VisionTrade(
                    trade_id=row[0], chart_id=row[1], analysis_id=row[2], symbol=row[3],
                    trade_type=row[4], lot_size=row[5], open_price=row[6], stop_loss=row[7],
                    take_profit_1=row[8], take_profit_2=row[9], entry_reason=row[10],
                    created_at=datetime.fromisoformat(row[11]), mt5_ticket=row[12],
                    status=row[13], current_price=row[14], profit_loss=row[15],
                    confidence=row[16], auto_trade=bool(row[17]), risk_percent=row[18]
                ))
        
        return trades
    
    def update_trade_status(self, trade_id: str, status: str, current_price: float = None, 
                           profit_loss: float = None, mt5_ticket: int = None):
        """Update trade status and metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Update trade
            params = [status]
            query = "UPDATE vision_trades SET status = ?"
            
            if current_price is not None:
                query += ", current_price = ?"
                params.append(current_price)
            
            if profit_loss is not None:
                query += ", profit_loss = ?"
                params.append(profit_loss)
                
            if mt5_ticket is not None:
                query += ", mt5_ticket = ?"
                params.append(mt5_ticket)
            
            query += " WHERE trade_id = ?"
            params.append(trade_id)
            
            conn.execute(query, params)
            
            # Add trade update record
            conn.execute("""
                INSERT INTO vision_trade_updates (trade_id, timestamp, price, profit_loss, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (trade_id, datetime.now().isoformat(), current_price, profit_loss, status, 
                  f"Trade status updated to {status}"))
            
            conn.commit()

class QNTIVisionTrader:
    """Enhanced vision trader with comprehensive MT5 integration"""
    
    def __init__(self, trade_manager, mt5_bridge, config: Dict = None):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        self.config = config or {}
        self.database = VisionTradingDatabase()
        self.monitoring = False
        self.monitor_thread = None
        logger.info("QNTIVisionTrader initialized")
    
    def create_chart_record(self, analysis_id: str, filename: str) -> VisionChart:
        """Create a chart record from uploaded analysis"""
        chart = VisionChart(
            id=analysis_id,
            filename=filename,
            uploaded_at=datetime.now(),
            analysis_id=analysis_id,
            file_path=f"chart_uploads/{analysis_id}"
        )
        self.database.save_chart(chart)
        return chart
    
    def update_chart_analysis(self, chart_id: str, analysis_data: Dict, symbol: str = None, timeframe: str = None):
        """Update chart with analysis results"""
        chart = self.database.get_chart(chart_id)
        if chart:
            chart.analysis_data = analysis_data
            chart.symbol = symbol
            chart.timeframe = timeframe
            self.database.save_chart(chart)
    
    def get_all_charts(self) -> List[Dict]:
        """Get all charts with analysis for API"""
        charts = self.database.get_all_charts()
        result = []
        
        for chart in charts:
            chart_data = {
                "id": chart.id,
                "filename": chart.filename,
                "uploaded_at": chart.uploaded_at.isoformat(),
                "symbol": chart.symbol,
                "timeframe": chart.timeframe
            }
            
            if chart.analysis_data:
                # Extract key analysis information
                analysis = chart.analysis_data
                chart_data["analysis"] = {
                    "confidence": analysis.get("confidence", 0),
                    "direction": analysis.get("market_bias", ""),
                    "summary": analysis.get("analysis_notes", "")[:200] + "..." if analysis.get("analysis_notes") else "",
                    "full_text": analysis.get("analysis_notes", ""),
                    "entry_zone": self._extract_entry_zone(analysis),
                    "stop_loss": self._extract_stop_loss(analysis),
                    "take_profit": self._extract_take_profit(analysis)
                }
            
            result.append(chart_data)
        
        return result
    
    def _extract_entry_zone(self, analysis: Dict) -> str:
        """Extract entry zone from analysis"""
        scenario = analysis.get("primary_scenario", {})
        if scenario and hasattr(scenario, 'entry_levels'):
            levels = scenario.entry_levels
            if levels and len(levels) >= 2:
                return f"{levels[0]:.5f} - {levels[1]:.5f}"
        return None
    
    def _extract_stop_loss(self, analysis: Dict) -> str:
        """Extract stop loss from analysis"""
        scenario = analysis.get("primary_scenario", {})
        if scenario and hasattr(scenario, 'stop_loss'):
            return f"{scenario.stop_loss:.5f}"
        return None
    
    def _extract_take_profit(self, analysis: Dict) -> str:
        """Extract take profit from analysis"""
        scenario = analysis.get("primary_scenario", {})
        if scenario and hasattr(scenario, 'profit_targets') and scenario.profit_targets:
            return f"{scenario.profit_targets[0]:.5f}"
        return None
    
    def create_trade_from_order(self, order_data: Dict, chart_id: str = None, analysis_id: str = None) -> VisionTrade:
        """Create vision trade from order data"""
        trade = VisionTrade(
            trade_id=f"VT_{uuid.uuid4().hex[:8]}",
            chart_id=chart_id or "manual",
            analysis_id=analysis_id or "manual",
            symbol=order_data["symbol"],
            trade_type=order_data["trade_type"],
            lot_size=order_data["lot_size"],
            open_price=order_data.get("open_price"),
            stop_loss=order_data.get("stop_loss"),
            take_profit_1=order_data.get("take_profit"),
            entry_reason=order_data.get("entry_reason", "Manual trade"),
            auto_trade=order_data.get("auto_trade", False),
            risk_percent=order_data.get("risk_percent"),
            confidence=order_data.get("confidence", 0.5)
        )
        
        # Save to database
        self.database.save_trade(trade)
        
        # Execute trade on MT5 if possible
        if self.mt5_bridge and self.mt5_bridge.is_connected():
            success, message, mt5_result = self._execute_mt5_trade(trade)
            if success and mt5_result:
                trade.mt5_ticket = mt5_result.get("order", 0)
                trade.status = "FILLED" if trade.trade_type in ["BUY", "SELL"] else "PENDING"
                self.database.update_trade_status(
                    trade.trade_id, trade.status, 
                    mt5_ticket=trade.mt5_ticket
                )
        
        return trade
    
    def _execute_mt5_trade(self, trade: VisionTrade) -> Tuple[bool, str, Dict]:
        """Execute trade on MT5"""
        try:
            # Create MT5 trade request
            request = self._build_mt5_request(trade)
            
            # Send order to MT5
            import MetaTrader5 as mt5
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return True, "Trade executed successfully", {
                    "order": result.order,
                    "deal": result.deal,
                    "volume": result.volume,
                    "price": result.price
                }
            else:
                error_msg = f"MT5 Error {result.retcode}: {result.comment}"
                return False, error_msg, {}
                
        except Exception as e:
            logger.error(f"Error executing MT5 trade: {e}")
            return False, str(e), {}
    
    def _build_mt5_request(self, trade: VisionTrade) -> Dict:
        """Build MT5 order request"""
        import MetaTrader5 as mt5
        
        # Map trade types to MT5 constants
        type_mapping = {
            "BUY": mt5.ORDER_TYPE_BUY,
            "SELL": mt5.ORDER_TYPE_SELL,
            "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
            "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
            "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
            "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP
        }
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade.symbol,
            "volume": trade.lot_size,
            "type": type_mapping.get(trade.trade_type, mt5.ORDER_TYPE_BUY),
            "comment": f"QNTI_Vision_{trade.trade_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Add price for pending orders
        if trade.trade_type in ["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"]:
            request["price"] = trade.open_price
        
        # Add stop loss and take profit
        if trade.stop_loss:
            request["sl"] = trade.stop_loss
        if trade.take_profit_1:
            request["tp"] = trade.take_profit_1
            
        return request
    
    def start_monitoring(self):
        """Start trade monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_trades, daemon=True)
        self.monitor_thread.start()
        logger.info("Vision trade monitoring started")
    
    def stop_monitoring(self):
        """Stop trade monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Vision trade monitoring stopped")
    
    def _monitor_trades(self):
        """Monitor active trades and update status"""
        import time
        
        while self.monitoring:
            try:
                # Get all active trades
                active_trades = self._get_active_trades()
                
                for trade in active_trades:
                    if trade.mt5_ticket:
                        # Check MT5 position status
                        self._update_trade_from_mt5(trade)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring trades: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _get_active_trades(self) -> List[VisionTrade]:
        """Get all active vision trades"""
        all_trades = []
        charts = self.database.get_all_charts()
        
        for chart in charts:
            trades = self.database.get_trades_by_chart(chart.id)
            all_trades.extend([t for t in trades if t.status in ["PENDING", "FILLED"]])
        
        return all_trades
    
    def _update_trade_from_mt5(self, trade: VisionTrade):
        """Update trade status from MT5"""
        try:
            import MetaTrader5 as mt5
            
            # Check if position exists
            position = mt5.positions_get(ticket=trade.mt5_ticket)
            
            if position:
                pos = position[0]
                current_price = pos.price_current
                profit_loss = pos.profit
                
                # Update trade with current metrics
                self.database.update_trade_status(
                    trade.trade_id, "FILLED", 
                    current_price=current_price,
                    profit_loss=profit_loss
                )
            else:
                # Position might be closed, check history
                history = mt5.history_deals_get(ticket=trade.mt5_ticket)
                if history:
                    deal = history[-1]  # Last deal
                    if deal.type == 1:  # Sell deal (position closed)
                        self.database.update_trade_status(
                            trade.trade_id, "CLOSED",
                            current_price=deal.price,
                            profit_loss=deal.profit
                        )
                        
        except Exception as e:
            logger.error(f"Error updating trade {trade.trade_id} from MT5: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get vision trading performance statistics"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN status = 'CLOSED' AND profit_loss > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN status = 'CLOSED' AND profit_loss < 0 THEN 1 END) as losing_trades,
                        SUM(CASE WHEN profit_loss IS NOT NULL THEN profit_loss ELSE 0 END) as total_pnl,
                        AVG(CASE WHEN profit_loss IS NOT NULL THEN profit_loss ELSE 0 END) as avg_pnl,
                        AVG(confidence) as avg_confidence
                    FROM vision_trades
                    WHERE created_at >= date('now', '-30 days')
                """)
                
                row = cursor.fetchone()
                if row:
                    total_trades, winning_trades, losing_trades, total_pnl, avg_pnl, avg_confidence = row
                    closed_trades = winning_trades + losing_trades
                    win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
                    
                    return {
                        "total_trades": total_trades or 0,
                        "winning_trades": winning_trades or 0,
                        "losing_trades": losing_trades or 0,
                        "win_rate": round(win_rate, 1),
                        "total_pnl": round(total_pnl or 0, 2),
                        "avg_pnl": round(avg_pnl or 0, 2),
                        "avg_confidence": round(avg_confidence or 0, 2)
                    }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "avg_confidence": 0
        }

# Integration function to process vision analysis and create trades
def process_vision_analysis_for_trading(
    analysis_text: str, 
    symbol: str, 
    vision_trader: QNTIVisionTrader,
    auto_submit: bool = False
) -> Optional[VisionTrade]:
    """
    Main function to process vision analysis and optionally submit for trading
    
    Args:
        analysis_text: The structured SMC analysis text
        symbol: Trading symbol (e.g., "XAUUSD")
        vision_trader: QNTIVisionTrader instance
        auto_submit: Whether to automatically submit the trade for execution
    
    Returns:
        VisionTrade object if successful, None otherwise
    """
    try:
        # Parse the analysis
        analysis = vision_trader.parse_smc_analysis(analysis_text, symbol)
        if not analysis:
            logger.error("Failed to parse SMC analysis")
            return None
        
        # Create vision trade
        vision_trade = vision_trader.create_vision_trade_from_analysis(analysis)
        if not vision_trade:
            logger.error("Failed to create vision trade from analysis")
            return None
        
        # Optionally submit for execution
        if auto_submit:
            success = vision_trader.submit_vision_trade(vision_trade)
            if success:
                logger.info(f"Vision trade submitted for execution: {vision_trade.analysis_id}")
            else:
                logger.error("Failed to submit vision trade")
                return None
        
        return vision_trade
        
    except Exception as e:
        logger.error(f"Error processing vision analysis for trading: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from qnti_core_system import QNTITradeManager
    from qnti_mt5_integration import QNTIMT5Bridge
    
    # Initialize components
    trade_manager = QNTITradeManager()
    mt5_bridge = QNTIMT5Bridge(trade_manager)
    vision_trader = QNTIVisionTrader(trade_manager, mt5_bridge)
    
    # Start monitoring
    vision_trader.start_monitoring()
    
    # Example analysis text (your actual output)
    sample_analysis = """
    ## 1. MARKET STRUCTURE ANALYSIS
    - **HTF Trend Direction:** Bearish
    - **Market Phase:** Transitional
    
    ## 2. SMC ELEMENTS IDENTIFIED
    - **Order Blocks (OB):** 
      - Bearish OB: 3,338.240 - 3,360.000
    
    ## 3. DIRECTIONAL BIAS & CONFLUENCE
    - **Primary Bias:** Short (High conviction)
    
    ## 4. TRADING PLAN
    ### Entry Strategy:
    - **Entry Zone:** 
      - 3,338.240 - 3,360.000
    
    ### Risk Management:
    - **Stop Loss:** 
      - 3,365.000 (Above OB)
    
    ### Profit Targets:
    - **TP1:** 
      - 3,311.160 (Previous BOS level)
    - **TP2:** 
      - 3,300.000 (Bullish OB)
    - **TP3:** 
      - 3,280.000 (Extended target near liquidity zone)
    """
    
    # Process the analysis
    vision_trade = process_vision_analysis_for_trading(
        analysis_text=sample_analysis,
        symbol="XAUUSD",
        vision_trader=vision_trader,
        auto_submit=True
    )
    
    if vision_trade:
        print(f"Vision trade created: {vision_trade.analysis_id}")
        print(f"Direction: {vision_trade.direction}")
        print(f"Entry Zone: {vision_trade.entry_zone_min} - {vision_trade.entry_zone_max}")
        print(f"Stop Loss: {vision_trade.stop_loss}")
        print(f"Take Profits: TP1={vision_trade.take_profit_1}, TP2={vision_trade.take_profit_2}")
    
    print("Vision trader running... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(10)
            summary = vision_trader.get_vision_trades_summary()
            print(f"Vision Trades Summary: {summary['by_status']}")
    except KeyboardInterrupt:
        vision_trader.stop_monitoring()
        print("Vision trader stopped") 