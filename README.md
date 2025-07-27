# Quantum Nexus Trading Intelligence (QNTI)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `vision_config.json` and add your OpenAI API key:
```json
{
  "vision": {
    "openai_api_key": "YOUR_OPENAI_API_KEY_HERE"
  }
}
```

### 3. Configure MT5 (Optional)
Edit `mt5_config.json` with your MetaTrader 5 account details:
```json
{
  "account": {
    "login": your_account_number,
    "password": "your_password",
    "server": "your_broker_server"
  }
}
```

### 4. Start the System

#### Windows:
```bash
start_qnti.bat
```

#### Linux/Mac:
```bash
./start_qnti.sh
```

#### Manual start:
```bash
python qnti_main_system.py --no-auto-trading --debug
```

### 5. Access Dashboard
Open your browser to: http://localhost:5000

## System Components

- **qnti_main_system.py** - Main orchestration system
- **qnti_core_system.py** - Core trade management
- **qnti_mt5_integration.py** - MetaTrader 5 integration
- **qnti_vision_analysis.py** - Chart analysis with AI
- **qnti_dashboard.html** - Web dashboard interface

## Configuration Files

- **qnti_config.json** - Main system configuration
- **mt5_config.json** - MetaTrader 5 settings
- **vision_config.json** - AI vision analysis settings

## Safety Features

- System starts in safe mode (no auto-trading) by default
- Emergency stop functionality
- Comprehensive risk management
- Full audit logging

## Troubleshooting

1. **MT5 Connection Issues**: Ensure MetaTrader 5 is installed and running
2. **Vision Analysis Errors**: Check OpenAI API key and credits
3. **Port Conflicts**: Change port in config if 5000 is in use
4. **Import Errors**: Ensure all files are in the same directory

## Support

Check the logs in the `logs/` directory for detailed error information.
