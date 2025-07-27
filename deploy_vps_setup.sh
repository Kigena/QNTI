#!/bin/bash
# QNTI Trading System - VPS Deployment Script
# Run this on a fresh Ubuntu 20.04/22.04 VPS

echo "🚀 QNTI Trading System VPS Setup Starting..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+ and dependencies
sudo apt install -y python3 python3-pip python3-venv git htop nano screen curl wget

# Install MetaTrader 5 dependencies (Wine for MT5 on Linux)
sudo dpkg --add-architecture i386
wget -nc https://dl.winehq.org/wine-builds/winehq.key
sudo apt-key add winehq.key
sudo add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ focal main'
sudo apt update
sudo apt install -y winehq-stable

# Create QNTI user and directory
sudo useradd -m -s /bin/bash qnti
sudo usermod -aG sudo qnti

# Switch to QNTI user
sudo -u qnti bash << 'EOF'
cd /home/qnti

# Clone your repository (replace with your actual repo)
# git clone https://github.com/your-username/qnti-trading-system.git
# For now, we'll create the directory structure
mkdir -p qnti-trading-system
cd qnti-trading-system

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install flask flask-socketio flask-cors
pip install MetaTrader5 numpy pandas
pip install sqlite3  # Usually built-in
pip install asyncio logging pathlib dataclasses enum
pip install opencv-python  # for cv2
pip install requests httpx

# Create systemd service for auto-start
EOF

# Create systemd service file
sudo tee /etc/systemd/system/qnti-trading.service > /dev/null << 'EOF'
[Unit]
Description=QNTI Trading System
After=network.target

[Service]
Type=simple
User=qnti
WorkingDirectory=/home/qnti/qnti-trading-system
Environment=PATH=/home/qnti/qnti-trading-system/venv/bin
ExecStart=/home/qnti/qnti-trading-system/venv/bin/python qnti_main_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set up firewall
sudo ufw allow 22    # SSH
sudo ufw allow 5002  # QNTI Web Interface
sudo ufw --force enable

# Create nginx config for reverse proxy (optional)
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/qnti > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;  # Replace with your domain if you have one
    
    location / {
        proxy_pass http://127.0.0.1:5002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/qnti /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo "✅ VPS Setup Complete!"
echo "📋 Next Steps:"
echo "1. Upload your QNTI code to /home/qnti/qnti-trading-system/"
echo "2. Configure MetaTrader 5 connection"
echo "3. Start the service: sudo systemctl start qnti-trading"
echo "4. Enable auto-start: sudo systemctl enable qnti-trading"
echo "5. Access via: http://YOUR_VPS_IP:5002"
echo ""
echo "🔧 Useful Commands:"
echo "- Check status: sudo systemctl status qnti-trading"
echo "- View logs: sudo journalctl -u qnti-trading -f"
echo "- Restart: sudo systemctl restart qnti-trading" 