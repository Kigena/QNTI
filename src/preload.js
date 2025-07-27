/**
 * QNTI Desktop - Preload Script
 * Secure bridge between main and renderer processes
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // App control
  quit: () => ipcRenderer.invoke('app-quit'),
  minimize: () => ipcRenderer.invoke('app-minimize'),
  maximize: () => ipcRenderer.invoke('app-maximize'),
  close: () => ipcRenderer.invoke('app-close'),
  
  // Server control
  startServer: () => ipcRenderer.invoke('server-start'),
  stopServer: () => ipcRenderer.invoke('server-stop'),
  restartServer: () => ipcRenderer.invoke('server-restart'),
  getServerStatus: () => ipcRenderer.invoke('server-status'),
  
  // Settings
  getSettings: () => ipcRenderer.invoke('settings-get'),
  setSettings: (settings) => ipcRenderer.invoke('settings-set', settings),
  
  // Files and paths
  openLogsFolder: () => ipcRenderer.invoke('open-logs'),
  openDataFolder: () => ipcRenderer.invoke('open-data'),
  
  // Notifications
  showNotification: (title, body) => ipcRenderer.invoke('show-notification', title, body),
  
  // Events
  onServerStatusChange: (callback) => {
    ipcRenderer.on('server-status-changed', (event, status) => callback(status));
  },
  
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
});

// Expose version information
contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron,
  app: () => '1.0.0' // QNTI version
});

// Platform information
contextBridge.exposeInMainWorld('platform', {
  isWindows: process.platform === 'win32',
  isMac: process.platform === 'darwin',
  isLinux: process.platform === 'linux'
});

// Security: Remove node integration from window object
delete window.require;
delete window.exports;
delete window.module;

console.log('QNTI Desktop preload script loaded'); 