#!/usr/bin/env python3
"""
QNTI Persistence Manager
Handles all system state persistence across restarts
"""

import json
import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger('qnti_persistence')

@dataclass
class StrategyState:
    """Individual strategy state"""
    name: str
    active: bool
    parameters: Dict[str, Any]
    last_signal_time: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    last_updated: str = None

@dataclass
class SystemState:
    """Complete system state"""
    unified_automation_states: Dict[str, StrategyState]
    dashboard_preferences: Dict[str, Any]
    user_settings: Dict[str, Any]
    active_trades: List[Dict[str, Any]]
    last_saved: str

class QNTIPersistenceManager:
    """Comprehensive persistence manager for QNTI system"""
    
    def __init__(self, data_dir: str = "qnti_data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # File paths
        self.system_state_file = os.path.join(data_dir, "system_state.json")
        self.settings_file = os.path.join(data_dir, "user_settings.json")
        self.preferences_file = os.path.join(data_dir, "dashboard_preferences.json")
        self.trades_db = os.path.join(data_dir, "persistent_trades.db")
        self.session_db = os.path.join(data_dir, "session_memory.db")
        
        # Initialize databases
        self.init_databases()
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("QNTI Persistence Manager initialized")
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['backups', 'exports', 'sessions']:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)
    
    def init_databases(self):
        """Initialize SQLite databases for complex data"""
        
        # Trades database
        with sqlite3.connect(self.trades_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS persistent_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket TEXT UNIQUE,
                    symbol TEXT,
                    strategy_name TEXT,
                    trade_type TEXT,
                    volume REAL,
                    open_price REAL,
                    open_time TEXT,
                    close_price REAL,
                    close_time TEXT,
                    profit REAL,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_strategy 
                ON persistent_trades(strategy_name)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                ON persistent_trades(symbol)
            ''')
        
        # Session memory database
        with sqlite3.connect(self.session_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS session_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_key TEXT,
                    data_type TEXT,
                    data_value TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_key 
                ON session_memory(session_key)
            ''')
    
    def save_strategy_state(self, strategy_name: str, state: Dict[str, Any]):
        """Save individual strategy state"""
        with self._lock:
            try:
                current_states = self.load_all_strategy_states()
                
                strategy_state = StrategyState(
                    name=strategy_name,
                    active=state.get('active', False),
                    parameters=state.get('parameters', {}),
                    last_signal_time=state.get('last_signal_time'),
                    performance_metrics=state.get('performance_metrics', {}),
                    last_updated=datetime.now().isoformat()
                )
                
                current_states[strategy_name] = asdict(strategy_state)
                
                with open(self.system_state_file, 'w') as f:
                    json.dump({
                        'unified_automation_states': current_states,
                        'last_saved': datetime.now().isoformat()
                    }, f, indent=2)
                
                logger.info(f"Strategy state saved: {strategy_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving strategy state {strategy_name}: {e}")
                return False
    
    def load_strategy_state(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Load individual strategy state"""
        try:
            all_states = self.load_all_strategy_states()
            return all_states.get(strategy_name)
        except Exception as e:
            logger.error(f"Error loading strategy state {strategy_name}: {e}")
            return None
    
    def load_all_strategy_states(self) -> Dict[str, Dict[str, Any]]:
        """Load all strategy states"""
        try:
            if os.path.exists(self.system_state_file):
                with open(self.system_state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('unified_automation_states', {})
            return {}
        except Exception as e:
            logger.error(f"Error loading strategy states: {e}")
            return {}
    
    def save_user_settings(self, settings: Dict[str, Any]):
        """Save user settings"""
        with self._lock:
            try:
                settings['last_updated'] = datetime.now().isoformat()
                
                with open(self.settings_file, 'w') as f:
                    json.dump(settings, f, indent=2)
                
                logger.info("User settings saved")
                return True
                
            except Exception as e:
                logger.error(f"Error saving user settings: {e}")
                return False
    
    def load_user_settings(self) -> Dict[str, Any]:
        """Load user settings"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading user settings: {e}")
            return {}
    
    def save_dashboard_preferences(self, preferences: Dict[str, Any]):
        """Save dashboard UI preferences"""
        with self._lock:
            try:
                preferences['last_updated'] = datetime.now().isoformat()
                
                with open(self.preferences_file, 'w') as f:
                    json.dump(preferences, f, indent=2)
                
                logger.info("Dashboard preferences saved")
                return True
                
            except Exception as e:
                logger.error(f"Error saving dashboard preferences: {e}")
                return False
    
    def load_dashboard_preferences(self) -> Dict[str, Any]:
        """Load dashboard UI preferences"""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading dashboard preferences: {e}")
            return {}
    
    def save_trade_memory(self, trade_data: Dict[str, Any]):
        """Save trade to persistent memory"""
        try:
            with sqlite3.connect(self.trades_db) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO persistent_trades 
                    (ticket, symbol, strategy_name, trade_type, volume, open_price, 
                     open_time, close_price, close_time, profit, status, 
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('ticket'),
                    trade_data.get('symbol'),
                    trade_data.get('strategy_name'),
                    trade_data.get('trade_type'),
                    trade_data.get('volume'),
                    trade_data.get('open_price'),
                    trade_data.get('open_time'),
                    trade_data.get('close_price'),
                    trade_data.get('close_time'),
                    trade_data.get('profit'),
                    trade_data.get('status', 'open'),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            logger.info(f"Trade memory saved: {trade_data.get('ticket')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade memory: {e}")
            return False
    
    def load_trade_memory(self, strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load trade memory"""
        try:
            with sqlite3.connect(self.trades_db) as conn:
                if strategy_name:
                    cursor = conn.execute(
                        'SELECT * FROM persistent_trades WHERE strategy_name = ? ORDER BY created_at DESC',
                        (strategy_name,)
                    )
                else:
                    cursor = conn.execute(
                        'SELECT * FROM persistent_trades ORDER BY created_at DESC'
                    )
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error loading trade memory: {e}")
            return []
    
    def save_session_data(self, session_key: str, data_type: str, data_value: Any):
        """Save session data"""
        try:
            with sqlite3.connect(self.session_db) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO session_memory 
                    (session_key, data_type, data_value, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_key,
                    data_type,
                    json.dumps(data_value) if not isinstance(data_value, str) else data_value,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            logger.info(f"Session data saved: {session_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            return False
    
    def load_session_data(self, session_key: str) -> List[Dict[str, Any]]:
        """Load session data"""
        try:
            with sqlite3.connect(self.session_db) as conn:
                cursor = conn.execute(
                    'SELECT * FROM session_memory WHERE session_key = ? ORDER BY updated_at DESC',
                    (session_key,)
                )
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return []
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a complete system backup"""
        try:
            if not backup_name:
                backup_name = f"qnti_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_dir = os.path.join(self.data_dir, 'backups', backup_name)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy all persistent files
            import shutil
            
            files_to_backup = [
                self.system_state_file,
                self.settings_file,
                self.preferences_file,
                self.trades_db,
                self.session_db
            ]
            
            for file_path in files_to_backup:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, backup_dir)
            
            # Create backup manifest
            manifest = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'files_backed_up': [os.path.basename(f) for f in files_to_backup if os.path.exists(f)]
            }
            
            with open(os.path.join(backup_dir, 'manifest.json'), 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"System backup created: {backup_name}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """Restore system from backup"""
        try:
            backup_dir = os.path.join(self.data_dir, 'backups', backup_name)
            
            if not os.path.exists(backup_dir):
                logger.error(f"Backup not found: {backup_name}")
                return False
            
            # Load manifest
            manifest_path = os.path.join(backup_dir, 'manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    logger.info(f"Restoring from backup: {manifest['backup_name']}")
            
            # Restore files
            import shutil
            
            files_to_restore = [
                ('system_state.json', self.system_state_file),
                ('user_settings.json', self.settings_file),
                ('dashboard_preferences.json', self.preferences_file),
                ('persistent_trades.db', self.trades_db),
                ('session_memory.db', self.session_db)
            ]
            
            for backup_file, target_path in files_to_restore:
                backup_file_path = os.path.join(backup_dir, backup_file)
                if os.path.exists(backup_file_path):
                    shutil.copy2(backup_file_path, target_path)
            
            logger.info(f"System restored from backup: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get persistence system status"""
        try:
            status = {
                'persistence_active': True,
                'data_directory': self.data_dir,
                'files_status': {},
                'database_status': {},
                'last_backup': None,
                'total_strategies': 0,
                'total_trades': 0
            }
            
            # Check file status
            files_to_check = [
                ('system_state', self.system_state_file),
                ('user_settings', self.settings_file),
                ('dashboard_preferences', self.preferences_file),
                ('trades_db', self.trades_db),
                ('session_db', self.session_db)
            ]
            
            for name, path in files_to_check:
                status['files_status'][name] = {
                    'exists': os.path.exists(path),
                    'size': os.path.getsize(path) if os.path.exists(path) else 0,
                    'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(path) else None
                }
            
            # Check strategy count
            strategies = self.load_all_strategy_states()
            status['total_strategies'] = len(strategies)
            
            # Check trade count
            trades = self.load_trade_memory()
            status['total_trades'] = len(trades)
            
            # Check for recent backups
            backups_dir = os.path.join(self.data_dir, 'backups')
            if os.path.exists(backups_dir):
                backups = [d for d in os.listdir(backups_dir) if os.path.isdir(os.path.join(backups_dir, d))]
                if backups:
                    status['last_backup'] = max(backups)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'persistence_active': False, 'error': str(e)}

def get_persistence_manager() -> QNTIPersistenceManager:
    """Get global persistence manager instance"""
    if not hasattr(get_persistence_manager, '_instance'):
        get_persistence_manager._instance = QNTIPersistenceManager()
    return get_persistence_manager._instance 