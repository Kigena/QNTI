# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for QNTI Desktop Application
Creates a standalone executable with all dependencies bundled
"""

import os
from pathlib import Path

# Get current directory
current_dir = Path.cwd()

# Define data files to include
datas = [
    # Configuration files
    ('qnti_config.json', '.'),
    ('mt5_config.json', '.'),
    ('vision_config.json', '.'),
    
    # Dashboard and static files
    ('dashboard', 'dashboard'),
    ('qnti_reports/templates', 'qnti_reports/templates'),
    
    # Data directories (if they exist)
    ('qnti_data', 'qnti_data') if os.path.exists('qnti_data') else None,
    ('ea_profiles', 'ea_profiles') if os.path.exists('ea_profiles') else None,
]

# Filter out None entries
datas = [d for d in datas if d is not None]

# Hidden imports (modules that PyInstaller might miss)
hiddenimports = [
    # Flask and web dependencies
    'flask',
    'flask_socketio',
    'flask_cors',
    'socketio',
    'engineio',
    'werkzeug',
    'jinja2',
    
    # QNTI modules
    'qnti_main_system',
    'qnti_core_system',
    'qnti_mt5_integration',
    'qnti_vision_analysis',
    'qnti_web_interface',
    'qnti_llm_mcp_integration',
    
    # MT5 and trading
    'MetaTrader5',
    'numpy',
    'pandas',
    
    # AI and vision
    'openai',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    
    # System tray
    'pystray',
    'pystray.backends',
    'pystray.backends._win32',
    
    # Async and threading
    'asyncio',
    'threading',
    'concurrent.futures',
    
    # Database
    'sqlite3',
    'chromadb',
    
    # Other utilities
    'requests',
    'json',
    'uuid',
    'datetime',
    'pathlib',
    'logging',
    'signal',
    'atexit',
    'webbrowser',
    'socket',
    'contextlib',
]

# Binaries to exclude (reduce size)
excludes = [
    'tkinter',
    'matplotlib',
    'scipy',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'setuptools',
    'distutils',
]

# Analysis block
a = Analysis(
    ['qnti_desktop_launcher.py'],
    pathex=[str(current_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='QNTI_Desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress executable
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='qnti_icon.ico' if os.path.exists('qnti_icon.ico') else None,
)

# For Windows, create an app bundle
if os.name == 'nt':
    app = BUNDLE(
        exe,
        name='QNTI_Desktop.app',
        icon='qnti_icon.ico' if os.path.exists('qnti_icon.ico') else None,
        bundle_identifier='com.qnti.desktop',
    ) 