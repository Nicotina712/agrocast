#!/bin/bash
# ============================================================================
# AgroCast PRO — Setup automático en Hetzner VPS (Ubuntu 22.04/24.04)
# ============================================================================
#
# USO:
#   1. Crear VPS CX22 en Hetzner (Ubuntu 22.04, region Ashburn o Helsinki)
#   2. Copiar este script al server:
#      scp deploy/setup_vps.sh root@TU_IP:/root/
#   3. SSH al server y ejecutar:
#      ssh root@TU_IP
#      chmod +x setup_vps.sh
#      ./setup_vps.sh
#   4. Cuando pregunte, pegar las API keys
#
# El script instala todo y deja el servidor corriendo.
# Después de esto, cada push a main se auto-deploya.
# ============================================================================

set -euo pipefail

# ── Colores ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Variables ────────────────────────────────────────────────────────
APP_USER="agrocast"
APP_DIR="/opt/agrocast"
REPO_URL="https://github.com/Nicotina712/agrocast.git"
BRANCH="main"
PYTHON_VERSION="3.12"
DOMAIN=""  # se configura después si tiene dominio

echo ""
echo "============================================"
echo "  AgroCast PRO — Setup VPS Automático"
echo "============================================"
echo ""

# ── 0. Verificar que corre como root ────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    error "Ejecutar como root: sudo ./setup_vps.sh"
fi

# ── 1. Preguntar configuración ──────────────────────────────────────
echo -e "${YELLOW}Configuración:${NC}"
read -p "¿Tenés dominio para AgroCast? (ej: agrocast.tudominio.com, dejar vacío si no): " DOMAIN
echo ""
echo "Ahora necesito las API keys. Las podés dejar vacías y configurar después."
echo "(Se guardan en /opt/agrocast/.env)"
echo ""
read -p "ANTHROPIC_API_KEY: " ANTHROPIC_KEY
read -p "NEWS_API_KEY: " NEWS_KEY
read -p "TELEGRAM_BOT_TOKEN: " TG_TOKEN
read -p "TELEGRAM_CHAT_ID: " TG_CHAT
read -p "AGROCAST_API_KEY (cualquier string para proteger endpoints): " AGROCAST_KEY
echo ""

# ── 2. Actualizar sistema ───────────────────────────────────────────
info "Actualizando sistema..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    software-properties-common curl git nginx certbot python3-certbot-nginx \
    build-essential libffi-dev libssl-dev ufw fail2ban

# ── 3. Instalar Python 3.12 ─────────────────────────────────────────
info "Instalando Python ${PYTHON_VERSION}..."
add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
apt-get update -qq
apt-get install -y -qq python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
info "Python $(python${PYTHON_VERSION} --version) instalado"

# ── 4. Crear usuario de aplicación ──────────────────────────────────
info "Creando usuario ${APP_USER}..."
if ! id "${APP_USER}" &>/dev/null; then
    useradd -r -m -d /opt/agrocast -s /bin/bash ${APP_USER}
fi

# ── 5. Clonar repositorio ───────────────────────────────────────────
info "Clonando repositorio..."
if [ -d "${APP_DIR}/.git" ]; then
    warn "Repo ya existe, actualizando..."
    cd ${APP_DIR}
    sudo -u ${APP_USER} git pull origin ${BRANCH} || true
else
    git clone ${REPO_URL} ${APP_DIR}
    chown -R ${APP_USER}:${APP_USER} ${APP_DIR}
fi
cd ${APP_DIR}

# ── 6. Crear virtualenv e instalar dependencias ─────────────────────
info "Creando virtualenv..."
sudo -u ${APP_USER} python${PYTHON_VERSION} -m venv ${APP_DIR}/venv
sudo -u ${APP_USER} ${APP_DIR}/venv/bin/pip install --upgrade pip -q
info "Instalando dependencias (esto tarda ~2 min)..."
sudo -u ${APP_USER} ${APP_DIR}/venv/bin/pip install -r requirements.txt -q
info "Dependencias instaladas"

# ── 7. Crear archivo .env ────────────────────────────────────────────
info "Configurando variables de entorno..."
cat > ${APP_DIR}/.env << ENVEOF
# AgroCast PRO — Environment Variables
# Editar con: nano /opt/agrocast/.env
# Después de editar: sudo systemctl restart agrocast

ANTHROPIC_API_KEY=${ANTHROPIC_KEY}
NEWS_API_KEY=${NEWS_KEY}
NEWSAPI_KEY=${NEWS_KEY}
TELEGRAM_BOT_TOKEN=${TG_TOKEN}
TELEGRAM_CHAT_ID=${TG_CHAT}
AGROCAST_API_KEY=${AGROCAST_KEY}
TZ=America/Montevideo

# Server config
RUN_SCHEDULER=1
BOOTSTRAP_PIPELINE=1
AGROCAST_FAST_START=0
CORS_ORIGINS=*
PORT=8000
ENVEOF
chown ${APP_USER}:${APP_USER} ${APP_DIR}/.env
chmod 600 ${APP_DIR}/.env

# ── 8. Crear servicio systemd ────────────────────────────────────────
info "Configurando servicio systemd..."
cat > /etc/systemd/system/agrocast.service << 'SVCEOF'
[Unit]
Description=AgroCast PRO — Dashboard de Inteligencia de Mercado
After=network.target
Wants=network-online.target

[Service]
Type=notify
User=agrocast
Group=agrocast
WorkingDirectory=/opt/agrocast
EnvironmentFile=/opt/agrocast/.env
ExecStart=/opt/agrocast/venv/bin/gunicorn wsgi:app \
    --workers 2 \
    --threads 4 \
    --timeout 300 \
    --graceful-timeout 30 \
    --preload \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --bind 127.0.0.1:8000 \
    --access-logfile /var/log/agrocast/access.log \
    --error-logfile /var/log/agrocast/error.log \
    --log-level info

# Restart automático si crashea
Restart=always
RestartSec=5

# Límites de recursos (4 GB RAM disponible en CX22)
MemoryMax=3G
CPUWeight=80

# Seguridad
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
SVCEOF

# Crear directorio de logs
mkdir -p /var/log/agrocast
chown ${APP_USER}:${APP_USER} /var/log/agrocast

# ── 9. Configurar nginx ─────────────────────────────────────────────
info "Configurando nginx..."

SERVER_NAME="${DOMAIN:-_}"

cat > /etc/nginx/sites-available/agrocast << NGXEOF
# AgroCast PRO — nginx reverse proxy
# Sirve contenido estático desde nginx (rápido)
# Proxea API requests a gunicorn (Flask)

server {
    listen 80;
    server_name ${SERVER_NAME};

    # Logs
    access_log /var/log/nginx/agrocast_access.log;
    error_log  /var/log/nginx/agrocast_error.log;

    # Compresión gzip (reduce tamaño 70%)
    gzip on;
    gzip_types text/html text/css application/json application/javascript text/xml;
    gzip_min_length 256;

    # Cache de archivos estáticos (1 día)
    location /static/ {
        alias /opt/agrocast/MVP lectura de noticias/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }

    # API endpoints → gunicorn
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeouts generosos para endpoints lentos (/api/news)
        proxy_connect_timeout 10;
        proxy_read_timeout 180;
        proxy_send_timeout 60;

        # Buffering para respuestas grandes
        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 4 32k;
    }

    # Health check (para monitoreo externo)
    location /healthz {
        proxy_pass http://127.0.0.1:8000/healthz;
        access_log off;
    }
}
NGXEOF

# Activar site
ln -sf /etc/nginx/sites-available/agrocast /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# ── 10. Configurar SSL (si tiene dominio) ────────────────────────────
if [ -n "${DOMAIN}" ]; then
    info "Configurando SSL para ${DOMAIN}..."
    certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos -m admin@${DOMAIN} || \
        warn "SSL falló — configurar manualmente después: certbot --nginx -d ${DOMAIN}"
fi

# ── 11. Configurar firewall ─────────────────────────────────────────
info "Configurando firewall..."
ufw --force reset > /dev/null 2>&1
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow http
ufw allow https
ufw --force enable

# ── 12. Script de deploy automático ─────────────────────────────────
info "Creando script de deploy..."
cat > /opt/agrocast/deploy.sh << 'DEPLOYEOF'
#!/bin/bash
# Deploy automático — se ejecuta via webhook o cron
# IMPORTANTE: Nunca sobreescribe data/ ni artifacts/ con versiones de git.
# Los datos locales del pipeline tienen prioridad.
set -e
cd /opt/agrocast

LOCK="/tmp/agrocast-deploy.lock"
exec 200>"$LOCK"
flock -n 200 || { echo "[$(date)] Deploy ya en progreso"; exit 0; }

echo "[$(date)] Iniciando deploy..."

# Check si hay cambios nuevos
git fetch origin main 2>/dev/null
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "[$(date)] Sin cambios nuevos"
    exit 0
fi

echo "[$(date)] Nuevos cambios detectados: ${LOCAL:0:8} → ${REMOTE:0:8}"

# Proteger data/ y artifacts/ locales (el pipeline los genera frescos)
# skip-worktree le dice a git: "no toques estos archivos en pull"
git ls-files data/ artifacts/ 2>/dev/null | while read f; do
    git update-index --skip-worktree "$f" 2>/dev/null || true
done

# Pull solo código (data/ y artifacts/ no se tocan)
git pull origin main --no-edit 2>/dev/null

# Reinstalar dependencias si requirements.txt cambió
if git diff HEAD~1 --name-only 2>/dev/null | grep -q "requirements.txt"; then
    echo "[$(date)] requirements.txt cambió — instalando deps..."
    /opt/agrocast/venv/bin/pip install -r requirements.txt -q
fi

# Restart graceful (gunicorn recarga workers sin cortar conexiones)
sudo systemctl reload-or-restart agrocast

echo "[$(date)] Deploy completado ✓ ($(git rev-parse --short HEAD))"
DEPLOYEOF
chmod +x /opt/agrocast/deploy.sh
chown ${APP_USER}:${APP_USER} /opt/agrocast/deploy.sh

# ── 13. Webhook receiver para auto-deploy ────────────────────────────
info "Configurando webhook para auto-deploy..."
cat > /opt/agrocast/webhook_receiver.py << 'WHEOF'
"""
Micro-server que recibe GitHub webhooks y ejecuta deploy.sh
Corre como servicio systemd en puerto 9000.
"""
import hmac, hashlib, subprocess, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler

SECRET = os.environ.get("WEBHOOK_SECRET", "agrocast-deploy-2026")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/deploy":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Verificar firma GitHub
        sig_header = self.headers.get("X-Hub-Signature-256", "")
        if sig_header:
            expected = "sha256=" + hmac.new(
                SECRET.encode(), body, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig_header, expected):
                self.send_response(403)
                self.end_headers()
                return

        # Ejecutar deploy
        try:
            result = subprocess.run(
                ["/opt/agrocast/deploy.sh"],
                capture_output=True, text=True, timeout=120,
                cwd="/opt/agrocast",
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"OK: {result.stdout[-200:]}".encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        pass  # silencio

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 9000), Handler)
    print("[webhook] Listening on :9000")
    server.serve_forever()
WHEOF
chown ${APP_USER}:${APP_USER} /opt/agrocast/webhook_receiver.py

# Servicio systemd para webhook
cat > /etc/systemd/system/agrocast-webhook.service << 'WHSVCEOF'
[Unit]
Description=AgroCast Deploy Webhook
After=network.target

[Service]
Type=simple
User=agrocast
WorkingDirectory=/opt/agrocast
ExecStart=/opt/agrocast/venv/bin/python /opt/agrocast/webhook_receiver.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
WHSVCEOF

# Agregar webhook a nginx
cat >> /etc/nginx/sites-available/agrocast << 'WHLOC'

    # Webhook para auto-deploy desde GitHub
    location /deploy {
        proxy_pass http://127.0.0.1:9000/deploy;
        proxy_set_header X-Hub-Signature-256 $http_x_hub_signature_256;
    }
WHLOC

# Fix: insertar antes del último }
# (el webhook location necesita estar dentro del server block)
# Reescribir nginx config correctamente
python3 -c "
conf = open('/etc/nginx/sites-available/agrocast').read()
# Remove the appended block (it's outside server {})
conf = conf.rsplit('# Webhook para auto-deploy desde GitHub', 1)[0]
# Insert webhook location before closing }
parts = conf.rsplit('}', 1)
webhook = '''
    # Webhook para auto-deploy desde GitHub
    location /deploy {
        proxy_pass http://127.0.0.1:9000/deploy;
        proxy_set_header X-Hub-Signature-256 \\\$http_x_hub_signature_256;
    }
'''
conf = parts[0] + webhook + '\n}\n'
open('/etc/nginx/sites-available/agrocast', 'w').write(conf)
"
nginx -t && systemctl reload nginx

# ── 14. Cron de backup y polling ─────────────────────────────────────
info "Configurando cron jobs..."
cat > /etc/cron.d/agrocast << 'CRONEOF'
# AgroCast PRO — Cron Jobs
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin

# Auto-deploy: check cada 5 min si hay cambios en GitHub
# (backup del webhook — si el webhook falla, esto lo hace igual)
*/5 * * * * agrocast cd /opt/agrocast && /opt/agrocast/deploy.sh >> /var/log/agrocast/deploy.log 2>&1

# Logrotate diario
0 3 * * * root /usr/sbin/logrotate /etc/logrotate.d/agrocast

CRONEOF

# Logrotate
cat > /etc/logrotate.d/agrocast << 'LREOF'
/var/log/agrocast/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    postrotate
        systemctl reload agrocast 2>/dev/null || true
    endscript
}
LREOF

# ── 15. Sudoers para deploy sin password ─────────────────────────────
echo "agrocast ALL=(ALL) NOPASSWD: /bin/systemctl reload-or-restart agrocast" > /etc/sudoers.d/agrocast
chmod 440 /etc/sudoers.d/agrocast

# ── 16. Arrancar servicios ───────────────────────────────────────────
info "Arrancando servicios..."
systemctl daemon-reload
systemctl enable agrocast
systemctl enable agrocast-webhook
systemctl start agrocast
systemctl start agrocast-webhook

# Esperar a que arranque
sleep 3
if systemctl is-active --quiet agrocast; then
    info "AgroCast corriendo ✓"
else
    warn "AgroCast no arrancó. Verificar: journalctl -u agrocast -n 50"
fi

# ── 17. Resumen final ───────────────────────────────────────────────
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo "============================================"
echo "  ✅ AgroCast PRO — Setup Completado"
echo "============================================"
echo ""
echo "  🌐 Dashboard:  http://${DOMAIN:-$SERVER_IP}"
echo "  🔧 Health:     http://${DOMAIN:-$SERVER_IP}/healthz"
echo "  📦 Deploy:     http://${DOMAIN:-$SERVER_IP}/deploy"
echo ""
echo "  📁 Código:     /opt/agrocast/"
echo "  🔑 Env vars:   /opt/agrocast/.env"
echo "  📋 Logs:       /var/log/agrocast/"
echo ""
echo "  Comandos útiles:"
echo "    Ver logs:      journalctl -u agrocast -f"
echo "    Restart:       sudo systemctl restart agrocast"
echo "    Deploy manual: /opt/agrocast/deploy.sh"
echo "    Editar env:    nano /opt/agrocast/.env"
echo ""
if [ -n "${DOMAIN}" ]; then
    echo "  🔒 SSL configurado para ${DOMAIN}"
else
    echo "  ⚠️  Sin dominio. Para agregar:"
    echo "     1. Apuntar tu dominio a ${SERVER_IP}"
    echo "     2. Editar /etc/nginx/sites-available/agrocast"
    echo "     3. certbot --nginx -d tudominio.com"
fi
echo ""
echo "  📌 SIGUIENTE PASO:"
echo "  Configurar webhook en GitHub:"
echo "    → Repo → Settings → Webhooks → Add webhook"
echo "    → URL: http://${DOMAIN:-$SERVER_IP}/deploy"
echo "    → Content type: application/json"
echo "    → Events: Just the push event"
echo ""
echo "============================================"
