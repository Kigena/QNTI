# 🌐 QNTI Trading System - Cloud Deployment Guide

## **🚀 Deployment Options Overview**

| Option | Cost/Month | Difficulty | Best For | Uptime |
|--------|------------|------------|----------|---------|
| **VPS** | $5-15 | Easy | 24/7 Trading | 99.9% |
| **AWS EC2** | $10-30 | Medium | Scalability | 99.99% |
| **DigitalOcean** | $5-20 | Easy | Simplicity | 99.9% |
| **Railway** | $5-10 | Very Easy | Quick Deploy | 99.9% |
| **Heroku** | $7-25 | Easy | App-focused | 99.95% |

---

## **📋 Pre-Deployment Checklist**

- [ ] MetaTrader 5 account and credentials
- [ ] Trading strategy configured
- [ ] ML database preferences set
- [ ] Risk management parameters defined
- [ ] Backup plan for data/configs

---

## **🔥 Method 1: VPS Deployment (RECOMMENDED)**

### **Step 1: Get a VPS**
**Best Providers:**
- **DigitalOcean**: Create $5 droplet (Ubuntu 22.04)
- **Vultr**: $5/month VPS 
- **Linode**: $5/month server
- **AWS Lightsail**: $3.50/month instance

### **Step 2: Deploy Automatically**
```bash
# SSH into your VPS
ssh root@YOUR_VPS_IP

# Download and run setup script
wget https://raw.githubusercontent.com/your-repo/qnti-trading-system/main/deploy_vps_setup.sh
chmod +x deploy_vps_setup.sh
./deploy_vps_setup.sh
```

### **Step 3: Upload Your Code**
```bash
# From your local machine
scp -r qnti-trading-system/ qnti@YOUR_VPS_IP:/home/qnti/

# Or use git
ssh qnti@YOUR_VPS_IP
cd /home/qnti/qnti-trading-system
git pull origin main
```

### **Step 4: Start & Configure**
```bash
# Start the service
sudo systemctl start qnti-trading
sudo systemctl enable qnti-trading

# Check status
sudo systemctl status qnti-trading

# View logs
sudo journalctl -u qnti-trading -f
```

### **Step 5: Access Your System**
- Web Interface: `http://YOUR_VPS_IP:5002`
- SMC Dashboard: `http://YOUR_VPS_IP:5002/smc_automation`

---

## **🐳 Method 2: Docker + Cloud Platform**

### **Railway (Easiest)**
1. Fork/upload code to GitHub
2. Connect Railway to your GitHub repo
3. Railway auto-detects Dockerfile
4. Deploy with one click
5. Get: `https://your-app.railway.app`

### **DigitalOcean App Platform**
```bash
# Build and push to container registry
docker build -t qnti-trading .
docker tag qnti-trading registry.digitalocean.com/your-registry/qnti-trading
docker push registry.digitalocean.com/your-registry/qnti-trading
```

### **AWS ECS (Elastic Container Service)**
```bash
# Create ECR repository
aws ecr create-repository --repository-name qnti-trading

# Build and push
docker build -t qnti-trading .
docker tag qnti-trading:latest YOUR_AWS_ACCOUNT.dkr.ecr.region.amazonaws.com/qnti-trading:latest
docker push YOUR_AWS_ACCOUNT.dkr.ecr.region.amazonaws.com/qnti-trading:latest
```

---

## **⚡ Method 3: One-Click Cloud Deployments**

### **Railway (Recommended for Beginners)**
1. Go to [Railway.app](https://railway.app)
2. Connect GitHub account
3. Import your QNTI repository
4. Railway auto-deploys
5. Get instant URL: `https://qnti-trading-production.up.railway.app`

### **Render**
1. Go to [Render.com](https://render.com)
2. Connect GitHub
3. Select "Web Service"
4. Auto-detects Python app
5. Deploy with generated URL

### **Fly.io**
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login and deploy
fly auth login
fly launch  # Auto-detects Dockerfile
fly deploy
```

---

## **🛡️ Security & Production Setup**

### **1. Environment Variables**
Create `.env` file:
```bash
# Trading Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# Security
SECRET_KEY=your_secret_key_here
ADMIN_PASSWORD=your_admin_password

# ML Database
ML_DATABASE_PATH=/app/data/qnti_signal_ml.db

# API Keys (if using)
TELEGRAM_BOT_TOKEN=your_telegram_token
DISCORD_WEBHOOK=your_discord_webhook
```

### **2. HTTPS Setup (with Domain)**
```nginx
# nginx.conf for SSL
server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/your_cert.pem;
    ssl_certificate_key /etc/ssl/private/your_key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:5002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### **3. Monitoring Setup**
```bash
# Add health checks
curl -f http://localhost:5002/health

# Monitor with cron
echo "*/5 * * * * curl -f http://localhost:5002/health || systemctl restart qnti-trading" | crontab -
```

---

## **💰 Cost Comparison**

### **Monthly Costs:**
- **VPS (Basic)**: $5-10/month
- **AWS t3.micro**: $8-15/month  
- **DigitalOcean Droplet**: $5-12/month
- **Railway**: $5-10/month
- **Heroku Hobby**: $7/month
- **Google Cloud Run**: $5-20/month

### **What You Get:**
- 24/7 uptime
- ML database persistence
- Real-time signal generation
- Web dashboard access
- Automatic restarts
- Log monitoring

---

## **🔧 Post-Deployment Configuration**

### **1. MetaTrader 5 Setup**
```python
# Update qnti_config.json
{
    "mt5": {
        "enabled": true,
        "login": "YOUR_MT5_LOGIN",
        "password": "YOUR_MT5_PASSWORD", 
        "server": "YOUR_MT5_SERVER",
        "path": "/path/to/terminal64.exe"  # For VPS with Wine
    }
}
```

### **2. ML Database Configuration**
```python
# qnti_smc_automation_config.json
{
    "ml_learning": {
        "enabled": true,
        "min_signals_for_learning": 10,
        "performance_threshold": 0.6,
        "confidence_boost_factor": 0.1
    }
}
```

### **3. Access Your Deployed System**
- **VPS**: `http://YOUR_VPS_IP:5002`
- **Railway**: `https://your-app.railway.app`
- **Heroku**: `https://your-app.herokuapp.com`
- **Custom Domain**: `https://trading.yourdomain.com`

---

## **🚨 Troubleshooting**

### **Common Issues:**

**1. Port Not Accessible**
```bash
# Check firewall
sudo ufw status
sudo ufw allow 5002
```

**2. Service Won't Start**
```bash
# Check logs
sudo journalctl -u qnti-trading -f
```

**3. MT5 Connection Issues**
```bash
# For Linux VPS, install Wine for MT5
sudo apt install winehq-stable
```

**4. Memory Issues**
```bash
# Monitor resources
htop
# Upgrade VPS if needed
```

---

## **📊 Monitoring Your Deployment**

### **Health Checks:**
- System status: `http://your-url/health`
- ML status: `http://your-url/api/smc-automation/ml-status`
- Signal tracking: `http://your-url/api/smc-automation/track-signals`

### **Log Monitoring:**
```bash
# Live logs
tail -f /home/qnti/qnti-trading-system/logs/qnti_system.log

# Error monitoring
grep ERROR /home/qnti/qnti-trading-system/logs/qnti_system.log
```

---

## **🎯 Recommended Setup for Serious Trading**

**For $10-15/month:**
1. **DigitalOcean $10 VPS** (2GB RAM, 1 CPU)
2. **Domain name** ($10/year) 
3. **SSL certificate** (Free with Let's Encrypt)
4. **Automated backups** (DigitalOcean snapshots)

**Result:** Professional trading system accessible 24/7 with your own domain!

---

**Ready to deploy? Choose your preferred method and follow the steps above! 🚀** 