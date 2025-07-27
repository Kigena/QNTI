/**
 * QNTI Desktop - Electron Main Process
 * Quantum Nexus Trading Intelligence Desktop Application
 */

const { app, BrowserWindow, Menu, Tray, shell, dialog, ipcMain, Notification } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const log = require('electron-log');
const { autoUpdater } = require('electron-updater');
const Store = require('electron-store');
const { nativeImage } = require('electron');

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';

// Initialize settings store
const store = new Store({
  defaults: {
    windowState: {
      width: 1400,
      height: 900,
      x: undefined,
      y: undefined,
      maximized: false
    },
    qntiSettings: {
      port: 5003,
      autoStart: true,
      minimizeToTray: true,
      startMinimized: false
    }
  }
});

class QNTIElectronApp {
  constructor() {
    this.mainWindow = null;
    this.tray = null;
    this.qntiProcess = null;
    this.serverPort = store.get('qntiSettings.port', 5003);
    this.serverUrl = `http://localhost:${this.serverPort}`;
    this.isQuitting = false;
    this.isServerRunning = false;
    
    // Setup app event handlers
    this.setupAppEvents();
    
    // Setup auto-updater
    this.setupAutoUpdater();
    
    log.info('QNTI Desktop Application initialized');
  }

  setupAppEvents() {
    // App ready
    app.whenReady().then(() => {
      this.createWindow();
      this.createTray();
      this.createMenu();
      
      if (store.get('qntiSettings.autoStart', true)) {
        this.startQNTIServer();
      }
    });

    // Window all closed
    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        this.shutdown();
      }
    });

    // Activate (macOS)
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        this.createWindow();
      }
    });

    // Before quit
    app.on('before-quit', () => {
      this.isQuitting = true;
    });

    // Will quit
    app.on('will-quit', (event) => {
      if (this.qntiProcess && !this.isQuitting) {
        event.preventDefault();
        this.shutdown();
      }
    });
  }

  createWindow() {
    // Get stored window state
    const windowState = store.get('windowState');
    
    // Create the browser window
    this.mainWindow = new BrowserWindow({
      width: windowState.width,
      height: windowState.height,
      x: windowState.x,
      y: windowState.y,
      minWidth: 1200,
      minHeight: 800,
      show: false, // Don't show until ready
      icon: this.getIconPath(),
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, 'preload.js'),
        webSecurity: true
      },
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default'
    });

    // Restore maximized state
    if (windowState.maximized) {
      this.mainWindow.maximize();
    }

    // Show window when ready
    this.mainWindow.once('ready-to-show', () => {
      if (!store.get('qntiSettings.startMinimized', false)) {
        this.mainWindow.show();
      }
      
      // Focus on the window
      if (this.mainWindow.isVisible()) {
        this.mainWindow.focus();
      }
    });

    // Save window state on move/resize
    this.mainWindow.on('moved', () => this.saveWindowState());
    this.mainWindow.on('resized', () => this.saveWindowState());
    this.mainWindow.on('maximize', () => this.saveWindowState());
    this.mainWindow.on('unmaximize', () => this.saveWindowState());

    // Handle window close
    this.mainWindow.on('close', (event) => {
      if (!this.isQuitting && store.get('qntiSettings.minimizeToTray', true)) {
        event.preventDefault();
        this.mainWindow.hide();
        
        // Show notification on first minimize
        if (!this.hasShownTrayNotification) {
          this.showNotification('QNTI is running in the background', 'Click the tray icon to restore the window');
          this.hasShownTrayNotification = true;
        }
      }
    });

    // Handle external links
    this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      shell.openExternal(url);
      return { action: 'deny' };
    });

    // Load the application
    this.loadApplication();

    log.info('Main window created');
  }

  createTray() {
    try {
      // Create a simple base64 encoded icon if the file doesn't exist
      const iconPath = path.join(__dirname, '..', 'build', 'tray-icon.ico');
      
      // Check if icon exists, if not create a simple one
      if (!fs.existsSync(iconPath)) {
        console.log('Tray icon not found, using default system icon');
        // Use a simple system icon instead
        this.tray = new Tray(nativeImage.createEmpty());
      } else {
        this.tray = new Tray(iconPath);
      }
      
      const contextMenu = Menu.buildFromTemplate([
        {
          label: 'Open QNTI Dashboard',
          click: () => this.showWindow()
        },
        { type: 'separator' },
        {
          label: 'Server Status',
          submenu: [
            {
              label: this.isServerRunning ? '● Running' : '○ Stopped',
              enabled: false
            },
            {
              label: `Port: ${this.serverPort}`,
              enabled: false
            },
            { type: 'separator' },
            {
              label: 'Start Server',
              enabled: !this.isServerRunning,
              click: () => this.startQNTIServer()
            },
            {
              label: 'Stop Server',
              enabled: this.isServerRunning,
              click: () => this.stopQNTIServer()
            },
            {
              label: 'Restart Server',
              click: () => this.restartQNTIServer()
            }
          ]
        },
        { type: 'separator' },
        {
          label: 'Show Logs',
          click: () => this.showLogs()
        },
        {
          label: 'Settings',
          click: () => this.showSettings()
        },
        { type: 'separator' },
        {
          label: 'Quit QNTI',
          click: () => this.quit()
        }
      ]);

      this.tray.setToolTip('QNTI Trading System');
      this.tray.setContextMenu(contextMenu);
      
      // Double click to show window
      this.tray.on('double-click', () => {
        this.showWindow();
      });

      log.info('System tray created');
    } catch (error) {
      log.error('Failed to create system tray:', error);
    }
  }

  createMenu() {
    const template = [
      {
        label: 'File',
        submenu: [
          {
            label: 'New Trading Session',
            accelerator: 'CmdOrCtrl+N',
            click: () => this.newTradingSession()
          },
          { type: 'separator' },
          {
            label: 'Settings',
            accelerator: 'CmdOrCtrl+,',
            click: () => this.showSettings()
          },
          { type: 'separator' },
          {
            label: 'Exit',
            accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
            click: () => this.quit()
          }
        ]
      },
      {
        label: 'Server',
        submenu: [
          {
            label: 'Start Server',
            accelerator: 'CmdOrCtrl+S',
            click: () => this.startQNTIServer()
          },
          {
            label: 'Stop Server',
            accelerator: 'CmdOrCtrl+Shift+S',
            click: () => this.stopQNTIServer()
          },
          {
            label: 'Restart Server',
            accelerator: 'CmdOrCtrl+R',
            click: () => this.restartQNTIServer()
          },
          { type: 'separator' },
          {
            label: 'Server Status',
            click: () => this.showServerStatus()
          }
        ]
      },
      {
        label: 'Tools',
        submenu: [
          {
            label: 'Show Logs',
            accelerator: 'CmdOrCtrl+L',
            click: () => this.showLogs()
          },
          {
            label: 'Open Data Folder',
            click: () => this.openDataFolder()
          },
          { type: 'separator' },
          {
            label: 'Developer Tools',
            accelerator: 'F12',
            click: () => this.mainWindow.webContents.openDevTools()
          }
        ]
      },
      {
        label: 'Help',
        submenu: [
          {
            label: 'Documentation',
            click: () => shell.openExternal('https://github.com/your-org/qnti-trading-system')
          },
          {
            label: 'Check for Updates',
            click: () => autoUpdater.checkForUpdatesAndNotify()
          },
          { type: 'separator' },
          {
            label: 'About QNTI',
            click: () => this.showAbout()
          }
        ]
      }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
  }

  loadApplication() {
    // Show loading screen
    const loadingHtml = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>QNTI Loading...</title>
        <style>
          body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
          }
          .loader {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
          }
          .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
          }
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          .logo {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
          }
          .status {
            font-size: 1.1em;
            opacity: 0.9;
          }
        </style>
      </head>
      <body>
        <div class="loader">
          <div class="logo">QNTI</div>
          <div class="spinner"></div>
          <div class="status">Starting Trading System...</div>
        </div>
      </body>
      </html>
    `;

    this.mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(loadingHtml)}`);

    // Try to connect to QNTI server
    this.waitForServer();
  }

  async waitForServer(maxAttempts = 30) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const fetch = (await import('node-fetch')).default;
        const response = await fetch(`${this.serverUrl}/api/health`, {
          timeout: 2000
        });
        
        if (response.ok) {
          log.info('QNTI server is ready');
          this.isServerRunning = true;
          this.mainWindow.loadURL(this.serverUrl);
          this.updateTrayMenu();
          return;
        }
      } catch (error) {
        // Server not ready yet
      }
      
      // Wait before next attempt
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // Server didn't start, show error
    log.error('QNTI server failed to start');
    this.showServerError();
  }

  showServerError() {
    const errorHtml = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>QNTI Server Error</title>
        <style>
          body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
          }
          .error-container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-width: 500px;
          }
          .error-icon {
            font-size: 4em;
            color: #e74c3c;
            margin-bottom: 20px;
          }
          .error-title {
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 15px;
          }
          .error-message {
            color: #7f8c8d;
            margin-bottom: 30px;
            line-height: 1.5;
          }
          .retry-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
          }
          .retry-btn:hover {
            background: #2980b9;
          }
        </style>
      </head>
      <body>
        <div class="error-container">
          <div class="error-icon">⚠️</div>
          <div class="error-title">Server Connection Failed</div>
          <div class="error-message">
            Unable to connect to the QNTI trading system server.<br>
            Please check if Python is installed and try again.
          </div>
          <button class="retry-btn" onclick="location.reload()">Retry Connection</button>
        </div>
      </body>
      </html>
    `;

    this.mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(errorHtml)}`);
  }

  async startQNTIServer() {
    if (this.qntiProcess) {
      console.log('QNTI server already running');
      return;
    }

    try {
      console.log('Starting QNTI server...');
      
      // Check if QNTI server is already running on port 5003
      const isRunning = await this.checkServerHealth();
      if (isRunning) {
        console.log('QNTI server already running on port 5003, connecting to existing instance');
        this.showNotification('QNTI Desktop', 'Connected to existing QNTI server');
        this.updateServerStatus('running');
        return;
      }

      // Start new QNTI server process
      this.qntiProcess = spawn('python', [
        'qnti_main.py',
        '--port', '5003',
        '--debug',
        '--no-auto-trading'
      ], {
        cwd: process.cwd(),
        stdio: ['ignore', 'pipe', 'pipe']
      });

      console.log('QNTI server process started');

      // Handle server output
      if (this.qntiProcess.stdout) {
        this.qntiProcess.stdout.on('data', (data) => {
          const output = data.toString();
          console.log('QNTI Server Output:', output);
          
          // Check for server ready message
          if (output.includes('QUANTUM NEXUS TRADING INTELLIGENCE') || 
              output.includes('Dashboard URL:')) {
            this.updateServerStatus('running');
            this.showNotification('QNTI Desktop', 'QNTI Server started successfully');
          }
        });
      }

      if (this.qntiProcess.stderr) {
        this.qntiProcess.stderr.on('data', (data) => {
          const error = data.toString();
          console.log('QNTI Server Error:', error);
          
          // Handle specific errors
          if (error.includes('WinError 10048') || error.includes('address already in use')) {
            console.log('Port already in use, checking if existing server is accessible');
            setTimeout(() => this.checkAndConnectToExistingServer(), 2000);
          }
        });
      }

      this.qntiProcess.on('exit', (code) => {
        console.log(`QNTI server exited with code ${code}`);
        this.qntiProcess = null;
        this.updateServerStatus('stopped');
        
        if (code !== 0) {
          console.log('QNTI server failed to start');
          this.showNotification('QNTI Desktop', 'QNTI Server failed to start', true);
          // Try to connect to existing server
          setTimeout(() => this.checkAndConnectToExistingServer(), 1000);
        }
      });

      this.qntiProcess.on('error', (error) => {
        console.error('Failed to start QNTI server:', error);
        this.qntiProcess = null;
        this.updateServerStatus('error');
        this.showNotification('QNTI Desktop', 'Failed to start QNTI Server', true);
      });

    } catch (error) {
      console.error('Error starting QNTI server:', error);
      this.updateServerStatus('error');
      this.showNotification('QNTI Desktop', 'Error starting QNTI Server', true);
    }
  }

  async checkAndConnectToExistingServer() {
    try {
      const isRunning = await this.checkServerHealth();
      if (isRunning) {
        console.log('Connected to existing QNTI server');
        this.showNotification('QNTI Desktop', 'Connected to existing QNTI server');
        this.updateServerStatus('running');
      } else {
        this.updateServerStatus('stopped');
        this.showNotification('QNTI Desktop', 'QNTI Server is not accessible', true);
      }
    } catch (error) {
      console.error('Failed to connect to existing server:', error);
      this.updateServerStatus('error');
    }
  }

  stopQNTIServer() {
    if (this.qntiProcess) {
      log.info('Stopping QNTI server...');
      this.qntiProcess.kill('SIGTERM');
      this.qntiProcess = null;
      this.isServerRunning = false;
      this.updateTrayMenu();
    }
  }

  restartQNTIServer() {
    log.info('Restarting QNTI server...');
    this.stopQNTIServer();
    setTimeout(() => {
      this.startQNTIServer();
    }, 2000);
  }

  // Helper methods
  saveWindowState() {
    if (!this.mainWindow) return;
    
    const bounds = this.mainWindow.getBounds();
    const isMaximized = this.mainWindow.isMaximized();
    
    store.set('windowState', {
      ...bounds,
      maximized: isMaximized
    });
  }

  showWindow() {
    if (this.mainWindow) {
      if (this.mainWindow.isMinimized()) {
        this.mainWindow.restore();
      }
      this.mainWindow.show();
      this.mainWindow.focus();
    }
  }

  updateTrayMenu() {
    if (this.tray) {
      this.createTray(); // Recreate tray menu with updated status
    }
  }

  getIconPath() {
    const iconName = process.platform === 'win32' ? 'icon.ico' : 
                     process.platform === 'darwin' ? 'icon.icns' : 'icon.png';
    return path.join(__dirname, '..', 'build', iconName);
  }

  getTrayIconPath() {
    const iconName = process.platform === 'win32' ? 'tray-icon.ico' : 
                     process.platform === 'darwin' ? 'tray-iconTemplate.png' : 'tray-icon.png';
    return path.join(__dirname, '..', 'build', iconName);
  }

  showNotification(title, body, isError = false) {
    // Use Electron's notification system instead of web Notification API
    try {
      const notification = new Notification({
        title: title,
        body: body,
        icon: this.getIconPath(),
        silent: false
      });
      
      notification.show();
      
      // Log to console as well
      if (isError) {
        console.error(`NOTIFICATION ERROR: ${title} - ${body}`);
      } else {
        console.log(`NOTIFICATION: ${title} - ${body}`);
      }
    } catch (error) {
      console.error('Failed to show notification:', error);
      // Fallback to console logging
      if (isError) {
        console.error(`ERROR: ${title} - ${body}`);
      } else {
        console.log(`INFO: ${title} - ${body}`);
      }
    }
  }

  showLogs() {
    const logsPath = path.join(__dirname, '..', 'logs');
    shell.openPath(logsPath);
  }

  showSettings() {
    // TODO: Implement settings window
    dialog.showMessageBox(this.mainWindow, {
      type: 'info',
      title: 'Settings',
      message: 'Settings panel coming soon!',
      detail: 'Advanced settings will be available in a future update.'
    });
  }

  showAbout() {
    dialog.showMessageBox(this.mainWindow, {
      type: 'info',
      title: 'About QNTI',
      message: 'QNTI Trading System v1.0.0',
      detail: 'Quantum Nexus Trading Intelligence\nProfessional Desktop Trading System\n\nBuilt with Electron and Python'
    });
  }

  openDataFolder() {
    const dataPath = path.join(__dirname, '..', 'qnti_data');
    shell.openPath(dataPath);
  }

  newTradingSession() {
    // TODO: Implement new trading session
    this.showWindow();
  }

  showServerStatus() {
    const status = this.isServerRunning ? 'Running' : 'Stopped';
    dialog.showMessageBox(this.mainWindow, {
      type: 'info',
      title: 'Server Status',
      message: `QNTI Server is ${status}`,
      detail: `Port: ${this.serverPort}\nURL: ${this.serverUrl}`
    });
  }

  setupAutoUpdater() {
    autoUpdater.checkForUpdatesAndNotify();
    
    autoUpdater.on('update-available', () => {
      log.info('Update available');
    });
    
    autoUpdater.on('update-downloaded', () => {
      log.info('Update downloaded');
      dialog.showMessageBox(this.mainWindow, {
        type: 'info',
        title: 'Update Ready',
        message: 'Update downloaded. Restart the application to apply the update.',
        buttons: ['Restart Now', 'Later'],
        defaultId: 0
      }).then((result) => {
        if (result.response === 0) {
          autoUpdater.quitAndInstall();
        }
      });
    });
  }

  quit() {
    this.isQuitting = true;
    this.shutdown();
  }

  shutdown() {
    log.info('Shutting down QNTI Desktop...');
    
    // Stop server
    if (this.qntiProcess) {
      this.stopQNTIServer();
    }
    
    // Save window state
    this.saveWindowState();
    
    // Quit app
    app.quit();
  }

  async checkServerHealth() {
    try {
      const fetch = (await import('node-fetch')).default;
      const response = await fetch(`${this.serverUrl}/api/health`, {
        timeout: 2000
      });
      
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  updateServerStatus(status) {
    this.isServerRunning = status === 'running';
    this.updateTrayMenu();
  }
}

// Initialize the application
const qntiApp = new QNTIElectronApp(); 