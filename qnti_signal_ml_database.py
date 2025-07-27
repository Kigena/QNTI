import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import MetaTrader5 as mt5

logger = logging.getLogger('QNTI_ML_DB')

@dataclass
class SignalOutcome:
    """Signal outcome tracking"""
    signal_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # 'win', 'loss', 'pending'
    profit_loss: Optional[float] = None
    duration_minutes: Optional[int] = None
    risk_reward_achieved: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None

@dataclass
class SignalPerformanceMetrics:
    """ML-derived performance metrics"""
    signal_type: str
    symbol: str
    htf_bias: str
    zone_type: str
    confluence_score: int
    confidence: float
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    total_signals: int
    winning_signals: int
    losing_signals: int
    avg_duration: float
    best_time_of_day: str
    best_market_conditions: Dict

class QNTISignalMLDatabase:
    """Machine Learning Signal Database for Persistent Learning"""
    
    def __init__(self, db_path: str = "data/qnti_signal_ml.db"):
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
        
        # ML Learning Parameters
        self.min_signals_for_learning = 10
        self.performance_threshold = 0.6  # 60% win rate threshold
        self.confidence_boost_factor = 0.1
        self.confidence_penalty_factor = 0.15
        
        logger.info(f"🧠 ML Signal Database initialized: {db_path}")
    
    def ensure_db_directory(self):
        """Ensure database directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database with ML-optimized schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Signals table - store all generated signals
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    signal_name TEXT,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    risk_reward_ratio REAL,
                    confidence REAL,
                    htf_bias TEXT,
                    zone_type TEXT,
                    confluence_score INTEGER,
                    confluence_factors TEXT,  -- JSON array
                    market_conditions TEXT,   -- JSON object
                    signal_data TEXT,         -- Complete signal as JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signal outcomes table - track results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    outcome TEXT,  -- 'win', 'loss', 'pending'
                    profit_loss REAL,
                    duration_minutes INTEGER,
                    risk_reward_achieved REAL,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
                )
            ''')
            
            # Performance analytics table - ML insights
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_analytics (
                    analytics_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    htf_bias TEXT,
                    zone_type TEXT,
                    confluence_score INTEGER,
                    confidence_range TEXT,  -- e.g., "0.8-0.9"
                    win_rate REAL,
                    avg_profit REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    total_signals INTEGER,
                    winning_signals INTEGER,
                    losing_signals INTEGER,
                    avg_duration REAL,
                    best_time_of_day TEXT,
                    best_market_conditions TEXT,  -- JSON
                    confidence_adjustment REAL,   -- ML-derived adjustment
                    generation_priority REAL,     -- ML-derived priority
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System learning state
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_state (
                    state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_signals_processed INTEGER DEFAULT 0,
                    total_learning_cycles INTEGER DEFAULT 0,
                    last_learning_update TIMESTAMP,
                    learning_confidence REAL DEFAULT 0.5,
                    system_performance REAL DEFAULT 0.0,
                    adaptive_factors TEXT,  -- JSON of learning factors
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ ML Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ML database: {e}")
            raise
    
    def store_signal(self, signal: Any) -> bool:
        """Store a generated signal for ML analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract signal data
            signal_id = signal.signal_id
            symbol = signal.symbol
            signal_type = str(signal.signal_type)
            signal_name = signal.zone_info.get('signal_name', '')
            entry_time = signal.timestamp
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            risk_reward_ratio = signal.risk_reward_ratio
            confidence = signal.confidence
            htf_bias = signal.zone_info.get('htf_bias', '')
            zone_type = signal.zone_info.get('zone_type', '')
            confluence_score = signal.zone_info.get('confluence_score', 0)
            confluence_factors = json.dumps(signal.zone_info.get('confluence_factors', []))
            
            # Get current market conditions
            market_conditions = self._get_market_conditions(symbol)
            
            # Store complete signal as JSON
            signal_data = json.dumps({
                'zone_info': signal.zone_info,
                'additional_data': signal.additional_data,
                'alert_level': str(signal.alert_level)
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO signals (
                    signal_id, symbol, signal_type, signal_name, entry_time,
                    entry_price, stop_loss, take_profit, risk_reward_ratio,
                    confidence, htf_bias, zone_type, confluence_score,
                    confluence_factors, market_conditions, signal_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, symbol, signal_type, signal_name, entry_time,
                entry_price, stop_loss, take_profit, risk_reward_ratio,
                confidence, htf_bias, zone_type, confluence_score,
                confluence_factors, json.dumps(market_conditions), signal_data
            ))
            
            # Also create initial outcome record
            cursor.execute('''
                INSERT OR REPLACE INTO signal_outcomes (
                    signal_id, symbol, entry_time, entry_price,
                    stop_loss, take_profit, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, symbol, entry_time, entry_price,
                stop_loss, take_profit, 'pending'
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"🧠 Signal stored for ML: {signal_id} ({symbol})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing signal for ML: {e}")
            return False
    
    def update_signal_outcome(self, signal_id: str, outcome_data: SignalOutcome) -> bool:
        """Update signal outcome for ML learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE signal_outcomes SET
                    exit_time = ?, exit_price = ?, outcome = ?,
                    profit_loss = ?, duration_minutes = ?,
                    risk_reward_achieved = ?, max_favorable_excursion = ?,
                    max_adverse_excursion = ?, updated_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
            ''', (
                outcome_data.exit_time, outcome_data.exit_price, outcome_data.outcome,
                outcome_data.profit_loss, outcome_data.duration_minutes,
                outcome_data.risk_reward_achieved, outcome_data.max_favorable_excursion,
                outcome_data.max_adverse_excursion, signal_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"🎯 Signal outcome updated: {signal_id} - {outcome_data.outcome}")
            
            # Trigger ML learning update
            self._trigger_ml_learning_update()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating signal outcome: {e}")
            return False
    
    def track_signal_progress(self, signal_id: str) -> Optional[Dict]:
        """Track signal progress via MT5 integration"""
        try:
            # Get signal details
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, entry_price, stop_loss, take_profit, entry_time
                FROM signal_outcomes 
                WHERE signal_id = ? AND outcome = 'pending'
            ''', (signal_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
            
            symbol, entry_price, stop_loss, take_profit, entry_time = result
            
            # Get current price from MT5
            if not mt5.initialize():
                logger.warning("⚠️ MT5 not available for signal tracking")
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                mt5.shutdown()
                return None
            
            current_price = tick.bid
            mt5.shutdown()
            
            # Calculate progress metrics
            entry_dt = datetime.fromisoformat(entry_time) if isinstance(entry_time, str) else entry_time
            duration = (datetime.now() - entry_dt).total_seconds() / 60  # minutes
            
            # Determine if signal hit TP or SL
            outcome = None
            exit_price = None
            profit_loss = None
            
            if current_price <= stop_loss:  # Assuming SELL signal logic
                outcome = 'loss'
                exit_price = stop_loss
                profit_loss = entry_price - stop_loss
            elif current_price >= take_profit:
                outcome = 'win'
                exit_price = take_profit
                profit_loss = entry_price - take_profit
            
            # Calculate excursions
            price_move = abs(current_price - entry_price)
            max_favorable = price_move if (current_price < entry_price) else 0
            max_adverse = price_move if (current_price > entry_price) else 0
            
            progress_data = {
                'signal_id': signal_id,
                'symbol': symbol,
                'current_price': current_price,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'duration_minutes': int(duration),
                'outcome': outcome,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'max_favorable_excursion': max_favorable,
                'max_adverse_excursion': max_adverse,
                'progress_percentage': self._calculate_progress_percentage(entry_price, current_price, take_profit, stop_loss)
            }
            
            # If signal completed, update outcome
            if outcome:
                outcome_obj = SignalOutcome(
                    signal_id=signal_id,
                    symbol=symbol,
                    entry_time=entry_dt,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    exit_time=datetime.now(),
                    exit_price=exit_price,
                    outcome=outcome,
                    profit_loss=profit_loss,
                    duration_minutes=int(duration),
                    risk_reward_achieved=abs(profit_loss / (entry_price - stop_loss)) if outcome == 'win' else 0,
                    max_favorable_excursion=max_favorable,
                    max_adverse_excursion=max_adverse
                )
                
                self.update_signal_outcome(signal_id, outcome_obj)
            
            return progress_data
            
        except Exception as e:
            logger.error(f"❌ Error tracking signal progress: {e}")
            return None
    
    def get_pending_signals(self) -> List[str]:
        """Get all pending signal IDs for tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT signal_id FROM signal_outcomes 
                WHERE outcome = 'pending'
                ORDER BY entry_time ASC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"❌ Error getting pending signals: {e}")
            return []
    
    def analyze_performance_for_ml(self) -> Dict:
        """Analyze historical performance for ML insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Overall performance metrics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN outcome = 'win' THEN profit_loss ELSE 0 END) as avg_win,
                    AVG(CASE WHEN outcome = 'loss' THEN ABS(profit_loss) ELSE 0 END) as avg_loss,
                    AVG(duration_minutes) as avg_duration
                FROM signal_outcomes 
                WHERE outcome IN ('win', 'loss')
            ''')
            
            overall = cursor.fetchone()
            
            # Performance by signal characteristics
            cursor.execute('''
                SELECT 
                    s.htf_bias, s.zone_type, s.confluence_score,
                    COUNT(*) as total,
                    SUM(CASE WHEN so.outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(CASE WHEN so.outcome = 'win' THEN so.profit_loss ELSE 0 END) as avg_win_profit
                FROM signals s
                JOIN signal_outcomes so ON s.signal_id = so.signal_id
                WHERE so.outcome IN ('win', 'loss')
                GROUP BY s.htf_bias, s.zone_type, s.confluence_score
                HAVING total >= ?
            ''', (self.min_signals_for_learning,))
            
            performance_by_characteristics = cursor.fetchall()
            conn.close()
            
            # Calculate insights
            total_signals, wins, losses, avg_win, avg_loss, avg_duration = overall or (0, 0, 0, 0, 0, 0)
            
            win_rate = (wins / total_signals) if total_signals > 0 else 0
            profit_factor = (avg_win * wins) / (avg_loss * losses) if (avg_loss and losses) else 0
            
            insights = {
                'overall_performance': {
                    'total_signals': total_signals,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_duration': avg_duration,
                    'learning_confidence': min(total_signals / 100, 1.0)  # Confidence grows with data
                },
                'characteristics_performance': [],
                'recommendations': []
            }
            
            # Analyze characteristics performance
            for row in performance_by_characteristics:
                htf_bias, zone_type, confluence_score, total, wins, avg_win_profit = row
                char_win_rate = wins / total if total > 0 else 0
                
                insights['characteristics_performance'].append({
                    'htf_bias': htf_bias,
                    'zone_type': zone_type,
                    'confluence_score': confluence_score,
                    'win_rate': char_win_rate,
                    'total_signals': total,
                    'avg_profit': avg_win_profit,
                    'confidence_adjustment': self._calculate_confidence_adjustment(char_win_rate)
                })
            
            # Generate ML recommendations
            insights['recommendations'] = self._generate_ml_recommendations(insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error analyzing performance for ML: {e}")
            return {}
    
    def get_adaptive_signal_parameters(self, symbol: str, htf_bias: str, zone_type: str) -> Dict:
        """Get ML-adjusted parameters for signal generation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get historical performance for these characteristics
            cursor.execute('''
                SELECT 
                    AVG(s.confidence) as avg_confidence,
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN so.outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(CASE WHEN so.outcome = 'win' THEN so.profit_loss ELSE 0 END) as avg_profit
                FROM signals s
                JOIN signal_outcomes so ON s.signal_id = so.signal_id
                WHERE s.symbol = ? AND s.htf_bias = ? AND s.zone_type = ?
                AND so.outcome IN ('win', 'loss')
            ''', (symbol, htf_bias, zone_type))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result or not result[1]:  # No historical data
                return {
                    'confidence_adjustment': 0.0,
                    'generation_priority': 0.5,
                    'min_confluence_score': 2,
                    'risk_reward_min': 1.5
                }
            
            avg_confidence, total_signals, wins, avg_profit = result
            win_rate = wins / total_signals if total_signals > 0 else 0
            
            # Calculate ML-based adjustments
            confidence_adjustment = self._calculate_confidence_adjustment(win_rate)
            generation_priority = self._calculate_generation_priority(win_rate, avg_profit, total_signals)
            
            # Adaptive parameters based on performance
            if win_rate > 0.7:  # High performance
                min_confluence = 1
                risk_reward_min = 1.2
            elif win_rate > 0.5:  # Medium performance
                min_confluence = 2
                risk_reward_min = 1.5
            else:  # Low performance
                min_confluence = 3
                risk_reward_min = 2.0
            
            return {
                'confidence_adjustment': confidence_adjustment,
                'generation_priority': generation_priority,
                'min_confluence_score': min_confluence,
                'risk_reward_min': risk_reward_min,
                'historical_win_rate': win_rate,
                'total_historical_signals': total_signals
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting adaptive parameters: {e}")
            return {
                'confidence_adjustment': 0.0,
                'generation_priority': 0.5,
                'min_confluence_score': 2,
                'risk_reward_min': 1.5
            }
    
    def _get_market_conditions(self, symbol: str) -> Dict:
        """Get current market conditions for ML analysis"""
        try:
            if not mt5.initialize():
                return {}
            
            # Get current market data
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                mt5.shutdown()
                return {}
            
            # Get some recent price data for volatility analysis
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
            mt5.shutdown()
            
            if rates is None or len(rates) == 0:
                return {}
            
            # Calculate market conditions
            current_hour = datetime.now().hour
            volatility = np.std([r['close'] for r in rates[-10:]])
            price_range = (max([r['high'] for r in rates[-5:]]) - min([r['low'] for r in rates[-5:]]))
            
            return {
                'hour_of_day': current_hour,
                'volatility': float(volatility),
                'recent_price_range': float(price_range),
                'spread': float(tick.ask - tick.bid),
                'market_session': self._get_market_session(current_hour)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting market conditions: {e}")
            return {}
    
    def _get_market_session(self, hour: int) -> str:
        """Determine market session based on hour"""
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'london'
        elif 16 <= hour < 24:
            return 'new_york'
        else:
            return 'overnight'
    
    def _calculate_progress_percentage(self, entry: float, current: float, tp: float, sl: float) -> float:
        """Calculate signal progress percentage"""
        try:
            if entry == tp:  # Avoid division by zero
                return 0.0
            
            total_distance = abs(tp - entry)
            current_distance = abs(current - entry)
            
            progress = (current_distance / total_distance) * 100
            return min(progress, 100.0)
            
        except:
            return 0.0
    
    def _calculate_confidence_adjustment(self, win_rate: float) -> float:
        """Calculate ML-based confidence adjustment"""
        if win_rate > self.performance_threshold:
            return self.confidence_boost_factor * (win_rate - self.performance_threshold)
        else:
            return -self.confidence_penalty_factor * (self.performance_threshold - win_rate)
    
    def _calculate_generation_priority(self, win_rate: float, avg_profit: float, total_signals: int) -> float:
        """Calculate signal generation priority based on ML analysis"""
        base_priority = 0.5
        
        # Win rate factor
        win_rate_factor = (win_rate - 0.5) * 0.4  # -0.2 to +0.2
        
        # Profit factor
        profit_factor = min(avg_profit * 10, 0.2) if avg_profit > 0 else -0.1
        
        # Data confidence factor
        data_confidence = min(total_signals / 50, 0.2)  # More data = higher confidence
        
        priority = base_priority + win_rate_factor + profit_factor + data_confidence
        return max(0.1, min(1.0, priority))
    
    def _generate_ml_recommendations(self, insights: Dict) -> List[str]:
        """Generate ML-based recommendations for system improvement"""
        recommendations = []
        
        overall = insights.get('overall_performance', {})
        win_rate = overall.get('win_rate', 0)
        
        if win_rate < 0.4:
            recommendations.append("🔴 Low win rate detected. Consider increasing confluence requirements.")
        elif win_rate > 0.7:
            recommendations.append("🟢 High win rate achieved. Consider reducing confluence requirements for more signals.")
        
        # Analyze characteristics
        characteristics = insights.get('characteristics_performance', [])
        
        best_performers = [c for c in characteristics if c.get('win_rate', 0) > 0.7]
        if best_performers:
            best = max(best_performers, key=lambda x: x.get('win_rate', 0))
            recommendations.append(f"⭐ Best performing setup: {best['htf_bias']} bias + {best['zone_type']} zone")
        
        poor_performers = [c for c in characteristics if c.get('win_rate', 0) < 0.3]
        if poor_performers:
            worst = min(poor_performers, key=lambda x: x.get('win_rate', 0))
            recommendations.append(f"❌ Avoid: {worst['htf_bias']} bias + {worst['zone_type']} zone (Low win rate)")
        
        return recommendations
    
    def _trigger_ml_learning_update(self):
        """Trigger ML learning update after signal completion"""
        try:
            # Update learning state
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if learning state exists
            cursor.execute('SELECT COUNT(*) FROM learning_state')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO learning_state (
                        total_signals_processed, total_learning_cycles,
                        last_learning_update, learning_confidence, system_performance
                    ) VALUES (1, 1, CURRENT_TIMESTAMP, 0.5, 0.0)
                ''')
            else:
                cursor.execute('''
                    UPDATE learning_state SET
                        total_signals_processed = total_signals_processed + 1,
                        total_learning_cycles = total_learning_cycles + 1,
                        last_learning_update = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                ''')
            
            conn.commit()
            conn.close()
            
            logger.info("🧠 ML learning update triggered")
            
        except Exception as e:
            logger.error(f"❌ Error triggering ML learning update: {e}")
    
    def get_learning_summary(self) -> Dict:
        """Get current learning state summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get learning state
            cursor.execute('SELECT * FROM learning_state ORDER BY state_id DESC LIMIT 1')
            learning_state = cursor.fetchone()
            
            # Get recent performance
            cursor.execute('''
                SELECT COUNT(*), SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END)
                FROM signal_outcomes 
                WHERE outcome IN ('win', 'loss') 
                AND entry_time > datetime('now', '-7 days')
            ''')
            recent_performance = cursor.fetchone()
            
            conn.close()
            
            if learning_state:
                total_signals, total_cycles, last_update, confidence, performance = learning_state[1:6]
                
                recent_total, recent_wins = recent_performance or (0, 0)
                recent_win_rate = (recent_wins / recent_total) if recent_total > 0 else 0
                
                return {
                    'total_signals_processed': total_signals,
                    'total_learning_cycles': total_cycles,
                    'last_learning_update': last_update,
                    'learning_confidence': confidence,
                    'system_performance': performance,
                    'recent_win_rate': recent_win_rate,
                    'recent_signals': recent_total,
                    'learning_status': 'active' if recent_total > 0 else 'waiting_for_data'
                }
            else:
                return {
                    'learning_status': 'not_initialized',
                    'total_signals_processed': 0,
                    'learning_confidence': 0.0
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting learning summary: {e}")
            return {'learning_status': 'error'} 