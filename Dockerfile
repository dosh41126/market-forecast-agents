# ─────────────────────────────────────────────────────────────────────────────
# Base image
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# ─────────────────────────────────────────────────────────────────────────────
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-tk \
    libgl1-mesa-glx curl iptables dnsutils openssl \
 && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# App setup
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# ─────────────────────────────────────────────────────────────────────────────
# Firewall + launch entrypoint
RUN cat << 'EOF' > /app/firewall_start.sh
#!/usr/bin/env bash
set -e

# Flush existing OUTPUT rules
iptables -F OUTPUT

# Allow loopback
iptables -A OUTPUT -o lo -j ACCEPT

# Allow DNS (UDP & TCP port 53)
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Allow outbound to Coinbase API domain
for DOMAIN in api.exchange.coinbase.com; do
  getent ahosts "\$DOMAIN" | awk '/STREAM/ {print \$1}' | sort -u | \
    while read ip; do
      iptables -A OUTPUT -d "\$ip" -j ACCEPT
    done
done

# Reject all other outbound traffic
iptables -A OUTPUT -j REJECT

# Finally, launch the dashboard
exec python main.py
EOF

RUN chmod +x /app/firewall_start.sh

# ─────────────────────────────────────────────────────────────────────────────
# Default entrypoint
CMD ["/app/firewall_start.sh"]
