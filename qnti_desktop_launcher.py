#!/usr/bin/env python3
"""
QNTI Desktop Application Launcher
Standalone desktop version of the Quantum Nexus Trading Intelligence system
"""

import sys
import os
import time
import threading
import webbrowser
import socket
from pathlib import Path
import logging
import signal
import atexit
from contextlib import closing

# GUI imports for system tray
try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    print("System tray not available. Install: pip install pystray pillow")

# QNTI System imports
from qnti_main_system import QNTIMainSystem

class QNTIDesktopApp:
    """Desktop wrapper for QNTI Trading System"""
    
    def __init__(self):
        self.qnti_system = None
        self.server_thread = None
        self.tray_icon = None
        self.running = False
        self.port = 5002  # Default port from memory
        
        # Setup logging for desktop app
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
        
        self.logger = logging.getLogger('QNTI_DESKTOP')
        self.logger.info("QNTI Desktop Application initialized")
    
    def setup_logging(self):
        """Setup logging for desktop application"""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / 'qnti_desktop.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def find_free_port(self, start_port=5002):
        """Find a free port starting from the preferred port"""
        for port in range(start_port, start_port + 100):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                if result != 0:  # Port is free
                    return port
        raise RuntimeError("Could not find a free port")
    
    def create_tray_icon(self):
        """Create system tray icon"""
        if not TRAY_AVAILABLE:
            return None
            
        # Create icon image
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)
        
        # Draw QNTI logo (simple Q design)
        draw.ellipse([10, 10, 54, 54], fill='blue', outline='white', width=2)
        draw.text((25, 22), "Q", fill='white', anchor="mm")
        
        # Create menu
        menu = pystray.Menu(
            pystray.MenuItem("Open Dashboard", self.open_dashboard),
            pystray.MenuItem("Restart System", self.restart_system),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Logs", self.show_logs),
            pystray.MenuItem("System Status", self.show_status),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self.quit_application)
        )
        
        return pystray.Icon("QNTI", image, "QNTI Trading System", menu)
    
    def start_qnti_server(self):
        """Start the QNTI system in a separate thread"""
        try:
            self.logger.info("Starting QNTI server...")
            
            # Find available port
            self.port = self.find_free_port(5002)
            self.logger.info(f"Using port {self.port}")
            
            # Initialize QNTI system
            self.qnti_system = QNTIMainSystem(config_file='qnti_config.json')
            
            # Disable auto-trading for safety in desktop mode
            self.qnti_system.auto_trading_enabled = False
            
            # Start the system
            self.qnti_system.start(
                host="127.0.0.1",  # Only localhost for desktop app
                port=self.port,
                debug=False  # Disable debug in production desktop app
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start QNTI server: {e}")
            self.show_error(f"Failed to start QNTI server: {e}")
    
    def wait_for_server(self, timeout=30):
        """Wait for the server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', self.port))
                    if result == 0:  # Server is ready
                        return True
            except:
                pass
            time.sleep(0.5)
        return False
    
    def open_dashboard(self, icon=None, item=None):
        """Open the QNTI dashboard in default browser"""
        url = f"http://127.0.0.1:{self.port}"
        try:
            webbrowser.open(url)
            self.logger.info(f"Opened dashboard: {url}")
        except Exception as e:
            self.logger.error(f"Failed to open dashboard: {e}")
            self.show_error(f"Failed to open dashboard: {e}")
    
    def restart_system(self, icon=None, item=None):
        """Restart the QNTI system"""
        self.logger.info("Restarting QNTI system...")
        
        # Stop current system
        if self.qnti_system:
            try:
                self.qnti_system.shutdown()
            except:
                pass
        
        # Wait a moment
        time.sleep(2)
        
        # Start new server thread
        self.server_thread = threading.Thread(target=self.start_qnti_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server and open dashboard
        if self.wait_for_server():
            time.sleep(1)  # Give it a moment to fully initialize
            self.open_dashboard()
    
    def show_logs(self, icon=None, item=None):
        """Open logs directory"""
        logs_dir = Path("logs").absolute()
        try:
            if sys.platform == "win32":
                os.startfile(str(logs_dir))
            elif sys.platform == "darwin":
                os.system(f"open '{logs_dir}'")
            else:
                os.system(f"xdg-open '{logs_dir}'")
        except Exception as e:
            self.logger.error(f"Failed to open logs directory: {e}")
    
    def show_status(self, icon=None, item=None):
        """Show system status (simplified for desktop)"""
        try:
            if self.qnti_system:
                health = self.qnti_system.get_system_health()
                status_msg = f"""QNTI System Status:
• Status: {health.get('system_status', 'Unknown')}
• Port: {self.port}
• Balance: ${health.get('account_balance', 0):.2f}
• Equity: ${health.get('account_equity', 0):.2f}
• Open Trades: {health.get('open_trades', 0)}
• Daily P&L: ${health.get('daily_pnl', 0):.2f}
"""
                print(status_msg)  # For now, print to console
                self.logger.info("Status check completed")
            else:
                print("QNTI System not running")
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
    
    def show_error(self, message):
        """Show error message (could be enhanced with GUI dialog)"""
        print(f"ERROR: {message}")
        self.logger.error(message)
    
    def quit_application(self, icon=None, item=None):
        """Quit the desktop application"""
        self.logger.info("Shutting down QNTI Desktop Application...")
        self.cleanup()
        self.running = False
        if self.tray_icon:
            self.tray_icon.stop()
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.quit_application()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.qnti_system:
                self.qnti_system.shutdown()
                self.qnti_system = None
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def run(self):
        """Run the desktop application"""
        self.running = True
        self.logger.info("=== QNTI Desktop Application Starting ===")
        
        try:
            # Start QNTI server in background thread
            self.server_thread = threading.Thread(target=self.start_qnti_server, daemon=True)
            self.server_thread.start()
            
            # Wait for server to be ready
            self.logger.info("Waiting for QNTI server to start...")
            if not self.wait_for_server():
                raise RuntimeError("QNTI server failed to start within timeout")
            
            self.logger.info(f"QNTI server is ready on port {self.port}")
            
            # Auto-open dashboard
            time.sleep(1)  # Give server a moment to fully initialize
            self.open_dashboard()
            
            # Create and run system tray
            if TRAY_AVAILABLE:
                self.tray_icon = self.create_tray_icon()
                self.logger.info("Starting system tray...")
                print("QNTI is running in the system tray. Right-click the icon for options.")
                self.tray_icon.run()  # This blocks until quit
            else:
                # Fallback: simple console interface
                print(f"\nQNTI Desktop Application is running!")
                print(f"Dashboard: http://127.0.0.1:{self.port}")
                print("Press Ctrl+C to stop\n")
                
                try:
                    while self.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.quit_application()
                    
        except Exception as e:
            self.logger.error(f"Fatal error in desktop application: {e}")
            self.show_error(str(e))
            sys.exit(1)

def main():
    """Main entry point for desktop application"""
    
    # Set working directory to script location
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Create and run desktop app
    app = QNTIDesktopApp()
    app.run()

if __name__ == "__main__":
    main() 