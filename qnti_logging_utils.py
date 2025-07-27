#!/usr/bin/env python3
"""
QNTI Logging Utilities
Centralized logging configuration with proper Unicode support
"""

import logging
import sys
import os
from typing import Optional

class QNTILogger:
    """Centralized logger with Unicode support for QNTI system"""
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def configure_global_logging(cls, 
                                level: int = logging.INFO,
                                log_dir: str = "logs",
                                max_file_size: int = 10 * 1024 * 1024,  # 10MB
                                backup_count: int = 5):
        """Configure global logging settings for the entire QNTI system"""
        if cls._configured:
            return
            
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Set console encoding to UTF-8 on Windows
        if sys.platform == 'win32':
            try:
                # Try to set console to UTF-8
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
            except:
                # If that fails, we'll handle encoding in the formatter
                pass
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = UnicodeConsoleFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'qnti_system.log'),
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger with proper Unicode support"""
        if name not in cls._loggers:
            if not cls._configured:
                cls.configure_global_logging()
                
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            
        return cls._loggers[name]

class UnicodeConsoleFormatter(logging.Formatter):
    """Custom formatter that handles Unicode characters properly on all platforms"""
    
    # Emoji to text mapping for fallback
    EMOJI_REPLACEMENTS = {
        '📈': '[CHART UP]',
        '📊': '[CHART]',
        '✅': '[OK]',
        '🔄': '[REFRESH]',
        '⚡': '[FAST]',
        '🎯': '[TARGET]',
        '⚠️': '[WARNING]',
        '❌': '[ERROR]',
        '🚀': '[ROCKET]',
        '💰': '[MONEY]',
        '📉': '[CHART DOWN]',
        '🔍': '[SEARCH]',
        '🛠️': '[TOOLS]',
        '📝': '[NOTE]',
        '🔔': '[BELL]',
        '💡': '[IDEA]',
        '⭐': '[STAR]',
        '🎉': '[PARTY]',
        '🔥': '[FIRE]',
        '💎': '[DIAMOND]'
    }
    
    def format(self, record):
        """Format the record, handling Unicode characters"""
        try:
            # First try normal formatting
            formatted = super().format(record)
            
            # Test if we can encode this safely
            formatted.encode('cp1252')  # Common Windows encoding
            return formatted
            
        except (UnicodeEncodeError, UnicodeDecodeError):
            # If Unicode fails, replace emojis with text equivalents
            message = record.getMessage()
            for emoji, replacement in self.EMOJI_REPLACEMENTS.items():
                message = message.replace(emoji, replacement)
                
            # Create a new record with the safe message
            safe_record = logging.LogRecord(
                record.name, record.levelno, record.pathname, record.lineno,
                message, record.args, record.exc_info, record.funcName, record.stack_info
            )
            safe_record.created = record.created
            safe_record.msecs = record.msecs
            safe_record.relativeCreated = record.relativeCreated
            safe_record.thread = record.thread
            safe_record.threadName = record.threadName
            safe_record.processName = record.processName
            safe_record.process = record.process
            
            return super().format(safe_record)

def get_qnti_logger(name: str) -> logging.Logger:
    """Convenience function to get a QNTI logger"""
    return QNTILogger.get_logger(name)

def safe_log_message(message: str) -> str:
    """Convert a message with emojis to a safe console-compatible version"""
    formatter = UnicodeConsoleFormatter()
    for emoji, replacement in formatter.EMOJI_REPLACEMENTS.items():
        message = message.replace(emoji, replacement)
    return message

# Initialize global logging when module is imported
QNTILogger.configure_global_logging() 