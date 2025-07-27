# QNTI Electron Desktop Application Guide

## Overview

The QNTI Electron Desktop Application provides a modern, cross-platform desktop experience for the Quantum Nexus Trading Intelligence system. Built with Electron, it offers native OS integration, automatic updates, and a professional desktop app experience.

## 🚀 Quick Start

### Prerequisites

1. **Node.js 16+** - Download from [nodejs.org](https://nodejs.org/)
2. **Python 3.8+** - For QNTI backend system
3. **QNTI System** - All Python files in the same directory

### Step 1: Install Dependencies

```bash
npm install
```

### Step 2: Run in Development Mode

```bash
# Windows
.\run_electron.bat

# Command line
npm run dev
```

### Step 3: Build for Production

```bash
# Windows
.\build_electron.bat

# Command line
npm run build:win
```

## 📋 Features

### 🖥️ Native Desktop Experience
- **True Desktop App** - No browser dependency
- **System Tray Integration** - Minimize to tray with context menu
- **Native Menus** - File, Server, Tools, Help menus
- **Keyboard Shortcuts** - Full keyboard navigation
- **Window State Memory** - Remembers size, position, maximized state

### 🔧 Server Management
- **Auto-Start Server** - QNTI Python server starts automatically
- **Server Control** - Start, stop, restart from menus or tray
- **Health Monitoring** - Real-time server status monitoring
- **Error Recovery** - Automatic error detection and recovery

### 🔒 Security & Performance
- **Context Isolation** - Secure renderer process
- **No Node Integration** - Web content runs in secure sandbox
- **Preload Scripts** - Safe API exposure
- **Auto-Updates** - Built-in update mechanism

### 🎨 Professional UI
- **Loading Screens** - Beautiful startup experience
- **Error Handling** - User-friendly error messages
- **Notifications** - System notifications for events
- **Modern Design** - Clean, professional interface

## 🏗️ Project Structure

```
qnti-trading-system/
├── package.json              # Project configuration
├── src/
│   ├── main.js               # Main Electron process
│   └── preload.js            # Preload script
├── build/                    # Icons and build resources
│   ├── icon.ico             # Windows icon
│   ├── icon.icns            # macOS icon
│   └── icon.png             # Linux icon
├── dist-electron/           # Built application output
├── build_electron.bat       # Windows build script
├── run_electron.bat         # Windows dev script
└── ELECTRON_APP_GUIDE.md    # This guide
```

## 🛠️ Development

### Running in Development

```bash
npm run dev
```

**Development Features:**
- Hot reload
- Developer tools (F12)
- Debug logging
- Live server monitoring

### Building for Production

```bash
# Windows installer + portable
npm run build:win

# macOS dmg
npm run build:mac

# Linux AppImage + deb
npm run build:linux

# All platforms
npm run build
```

### Available Scripts

| Script | Description |
|--------|-------------|
| `npm start` | Start production app |
| `npm run dev` | Start development mode |
| `npm run build` | Build for all platforms |
| `npm run build:win` | Build Windows installer |
| `npm run build:mac` | Build macOS dmg |
| `npm run build:linux` | Build Linux packages |
| `npm run pack` | Package without installer |

## 📦 Distribution

### Windows
- **Installer**: `QNTI Trading System-1.0.0-x64.exe`
- **Portable**: `QNTI Trading System-1.0.0-x64-portable.exe`

### macOS
- **DMG**: `QNTI Trading System-1.0.0-x64.dmg`
- **Universal**: Supports Intel and Apple Silicon

### Linux
- **AppImage**: `QNTI Trading System-1.0.0-x64.AppImage`
- **Debian**: `QNTI Trading System-1.0.0-x64.deb`

## ⚙️ Configuration

### App Settings (stored in user data)

```javascript
{
  "windowState": {
    "width": 1400,
    "height": 900,
    "maximized": false
  },
  "qntiSettings": {
    "port": 5003,
    "autoStart": true,
    "minimizeToTray": true,
    "startMinimized": false
  }
}
```

### Build Configuration (package.json)

Key configuration options:
- **App ID**: `com.qnti.desktop`
- **Auto-updater**: GitHub releases
- **Icons**: Platform-specific icons
- **Installer**: NSIS for Windows

## 🎛️ Menu Reference

### File Menu
- **New Trading Session** (Ctrl+N)
- **Settings** (Ctrl+,)
- **Exit** (Ctrl+Q)

### Server Menu
- **Start Server** (Ctrl+S)
- **Stop Server** (Ctrl+Shift+S)
- **Restart Server** (Ctrl+R)
- **Server Status**

### Tools Menu
- **Show Logs** (Ctrl+L)
- **Open Data Folder**
- **Developer Tools** (F12)

### Help Menu
- **Documentation**
- **Check for Updates**
- **About QNTI**

## 🖱️ System Tray

Right-click the system tray icon for:
- Open QNTI Dashboard
- Server Status (with start/stop/restart)
- Show Logs
- Settings
- Quit QNTI

## 🔄 Auto-Updates

The app includes automatic update checking:
- Checks for updates on startup
- Downloads updates in background
- Prompts user to restart when ready
- Uses GitHub releases for distribution

### Update Configuration

```json
{
  "publish": {
    "provider": "github",
    "owner": "your-org",
    "repo": "qnti-trading-system"
  }
}
```

## 🧪 Advanced Configuration

### Custom Python Path

If Python is not in PATH, modify `main.js`:

```javascript
const pythonPath = 'C:\\Python39\\python.exe';
this.qntiProcess = spawn(pythonPath, [pythonScript, ...args], {
  // ... options
});
```

### Custom Server Port

Change default port in `main.js`:

```javascript
const store = new Store({
  defaults: {
    qntiSettings: {
      port: 5003  // Change this
    }
  }
});
```

### Icon Customization

Replace icons in `build/` directory:
- `icon.ico` - Windows icon (256x256)
- `icon.icns` - macOS icon (1024x1024)
- `icon.png` - Linux icon (512x512)
- `tray-icon.*` - System tray icons

## 🚨 Troubleshooting

### Common Issues

**App won't start:**
```bash
# Check Node.js version
node --version

# Reinstall dependencies
rm -rf node_modules
npm install
```

**Build fails:**
```bash
# Clear cache
npm run rebuild
electron-builder --dir
```

**Server won't start:**
- Check Python installation
- Verify QNTI files are present
- Check logs in system tray menu

**Icon not showing:**
- Ensure icons exist in `build/` directory
- Rebuild after adding icons
- Check file paths in `package.json`

### Debugging

Enable debug mode:
```bash
DEBUG=* npm run dev
```

Check Electron logs:
- Windows: `%APPDATA%\qnti-desktop\logs\`
- macOS: `~/Library/Logs/qnti-desktop/`
- Linux: `~/.config/qnti-desktop/logs/`

## 🔧 Development Tips

### Adding New Features

1. **Main Process** - Add to `src/main.js`
2. **Renderer Process** - Use `electronAPI` from preload
3. **IPC Communication** - Add to `src/preload.js`

### Security Best Practices

- ✅ Context isolation enabled
- ✅ Node integration disabled
- ✅ Remote module disabled
- ✅ Secure preload script
- ✅ External links open in browser

### Performance Optimization

- Use `will-finish-launching` for faster startup
- Lazy load heavy modules
- Minimize preload script
- Use efficient file watching

## 📊 Comparison: Electron vs PyInstaller

| Feature | Electron | PyInstaller |
|---------|----------|-------------|
| **File Size** | ~150MB | ~400MB |
| **Startup Time** | Fast | Medium |
| **Native Feel** | Excellent | Good |
| **Cross-platform** | ✅ | Limited |
| **Auto-updates** | ✅ Built-in | Manual |
| **Development** | Modern | Simple |
| **System Integration** | Excellent | Good |

## 🌟 Why Choose Electron?

1. **Cross-Platform** - Windows, macOS, Linux
2. **Modern UI** - Web technologies with native feel
3. **Auto-Updates** - Built-in update mechanism
4. **Professional** - Used by VS Code, Discord, Slack
5. **Developer Tools** - Full Chrome DevTools
6. **Ecosystem** - Rich plugin ecosystem

## 📈 Next Steps

After building your Electron app:

1. **Test thoroughly** on target platforms
2. **Set up auto-updates** with GitHub releases
3. **Code signing** for production distribution
4. **App store submission** (optional)
5. **Crash reporting** integration
6. **Analytics** for usage insights

## 🔗 Resources

- [Electron Documentation](https://electronjs.org/docs)
- [Electron Builder](https://www.electron.build/)
- [Electron Store](https://github.com/sindresorhus/electron-store)
- [Auto-updater Guide](https://electronjs.org/docs/tutorial/updates)

---

*Your QNTI trading system is now a professional cross-platform desktop application with all the features users expect from modern software!* 