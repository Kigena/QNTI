#!/usr/bin/env python3
"""
QNTI Market Intelligence Engine - Standalone Version
Provides independent market intelligence and insights
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import yfinance as yf
import numpy as np
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('qnti_market_intelligence')

class QNTIMarketIntelligence:
    """Standalone Market Intelligence Engine"""
    
    def __init__(self):
        self.insights = []
        self.market_data = {}
        self.running = False
        self.update_thread = None
        
        # Market symbols to track
        self.symbols = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
            'AUDUSD=X', 'USDCAD=X', 'GC=F', 'SI=F', 
            'BTC-USD', '^GSPC', '^DJI', '^IXIC'
        ]
        
        logger.info("QNTI Market Intelligence Engine initialized")
        
    def start_monitoring(self):
        """Start market intelligence monitoring"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._monitoring_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Market Intelligence monitoring started")
    
    def stop_monitoring(self):
        """Stop market intelligence monitoring"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("Market Intelligence monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._update_market_data()
                self._generate_insights()
                time.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _update_market_data(self):
        """Update market data from Yahoo Finance"""
        try:
            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        change = current_price - hist['Close'].iloc[0]
                        change_percent = (change / hist['Close'].iloc[0]) * 100
                        
                        self.market_data[symbol] = {
                            'symbol': symbol,
                            'price': current_price,
                            'change': change,
                            'change_percent': change_percent,
                            'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _generate_insights(self):
        """Generate market insights"""
        try:
            # Simple insight generation based on market movements
            insights = []
            
            for symbol, data in self.market_data.items():
                if abs(data['change_percent']) > 1.0:  # Significant movement
                    insight = {
                        'id': f"insight_{symbol}_{int(time.time())}",
                        'symbol': symbol,
                        'title': f"{symbol} Significant Movement",
                        'content': f"{symbol} has moved {data['change_percent']:.2f}% today",
                        'confidence': 0.8,
                        'timestamp': datetime.now().isoformat(),
                        'category': 'market_movement'
                    }
                    insights.append(insight)
            
            # Keep only recent insights (last 100)
            self.insights.extend(insights)
            self.insights = self.insights[-100:]
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
    
    def get_insights(self, limit=10):
        """Get recent insights"""
        return self.insights[-limit:] if self.insights else []
    
    def get_market_data(self):
        """Get current market data"""
        return self.market_data
    
    def get_stats(self):
        """Get intelligence statistics"""
        return {
            'total_insights': len(self.insights),
            'symbols_tracked': len(self.symbols),
            'last_update': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.running else 'inactive'
        }

# Global intelligence engine instance
intelligence_engine = QNTIMarketIntelligence()

def integrate_market_intelligence_with_qnti(qnti_system):
    """Integrate market intelligence with QNTI system"""
    try:
        # Start the intelligence engine
        intelligence_engine.start_monitoring()
        
        # Add API endpoints
        @qnti_system.app.route('/api/market-intelligence/insights', methods=['GET'])
        def get_market_intelligence_insights():
            """Get market intelligence insights"""
            try:
                limit = request.args.get('limit', 10, type=int)
                insights = intelligence_engine.get_insights(limit)
                stats = intelligence_engine.get_stats()
                
                return jsonify({
                    'success': True,
                    'insights': insights,
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting insights: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @qnti_system.app.route('/api/market-intelligence/real-time-data', methods=['GET'])
        def get_real_time_market_data():
            """Get real-time market data"""
            try:
                market_data = intelligence_engine.get_market_data()
                
                return jsonify({
                    'success': True,
                    'data': market_data,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting market data: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @qnti_system.app.route('/api/market-intelligence/status', methods=['GET'])
        def get_market_intelligence_status():
            """Get market intelligence status"""
            try:
                stats = intelligence_engine.get_stats()
                
                return jsonify({
                    'success': True,
                    'status': stats,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @qnti_system.app.route('/api/market-intelligence/force-analysis', methods=['POST'])
        def force_market_analysis():
            """Force immediate market analysis"""
            try:
                intelligence_engine._update_market_data()
                intelligence_engine._generate_insights()
                
                return jsonify({
                    'success': True,
                    'message': 'Market analysis completed',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error forcing analysis: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        logger.info("Market Intelligence Engine integrated with QNTI system")
        return True
        
    except Exception as e:
        logger.error(f"Error integrating market intelligence: {e}")
        return False

if __name__ == "__main__":
    # For standalone testing
    intelligence_engine.start_monitoring()
    print("Market Intelligence Engine running...")
    try:
        while True:
            time.sleep(10)
            print(f"Insights: {len(intelligence_engine.get_insights())}")
    except KeyboardInterrupt:
        intelligence_engine.stop_monitoring()
        print("Market Intelligence Engine stopped.") 