#!/bin/bash

# Crypto Trading Bot Setup Script
echo "Setting up Crypto Trading Bot..."

# Get the static IP
STATIC_IP=$(curl -s ifconfig.me)
echo "Your static IP is: $STATIC_IP"

# Create necessary directories
mkdir -p logs data

# Create systemd service
sudo tee /etc/systemd/system/crypto-bot.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto-bot
Environment=PATH=/home/ubuntu/crypto-bot/venv/bin
ExecStart=/home/ubuntu/crypto-bot/venv/bin/python app.py
Restart=always
RestartSec=10

StandardOutput=append:/home/ubuntu/crypto-bot/logs/crypto_bot.log
StandardError=append:/home/ubuntu/crypto-bot/logs/crypto_bot_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/crypto-bot > /dev/null <<EOF
server {
    listen 80;
    server_name $STATIC_IP;
    
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
    }
    
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/crypto-bot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx

# Enable and start the crypto bot service
sudo systemctl daemon-reload
sudo systemctl enable crypto-bot
sudo systemctl start crypto-bot

# Create log rotation
sudo tee /etc/logrotate.d/crypto-bot > /dev/null <<EOF
/home/ubuntu/crypto-bot/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 ubuntu ubuntu
}
EOF

echo "Setup completed!"
echo ""
echo "🚀 Your Crypto Trading Bot is now running!"
echo "📊 Dashboard URL: http://$STATIC_IP"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status crypto-bot    # Check bot status"
echo "  sudo systemctl restart crypto-bot   # Restart bot"
echo "  tail -f logs/crypto_bot.log         # View logs"
echo "  sudo systemctl status nginx         # Check web server"
echo ""
echo "The bot will:"
echo "  ✅ Scan 20+ cryptocurrencies every 15 minutes"
echo "  ✅ Generate smart trading signals with ML analysis"
echo "  ✅ Track P&L with \$1000 virtual portfolio (10x leverage)"
echo "  ✅ Show live charts and technical analysis"
echo "  ✅ Run 24/7 automatically"
echo ""