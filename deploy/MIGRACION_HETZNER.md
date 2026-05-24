# Migracion AgroCast PRO: Render Free → Hetzner VPS

## Por que migrar

| Problema en Render | Solucion en Hetzner |
|---|---|
| Se duerme a los 15min → cold start 30-77s | Nunca se duerme (es tu servidor) |
| 512MB RAM → OOM, scraping lento | 4GB RAM → todo fluido |
| Sin disco persistente → datos se pierden | 40GB SSD → datos persisten siempre |
| Cada push → redeploy completo → datos stale | Deploy solo reinicia el server, datos no se tocan |
| Pipeline corre en GitHub Actions → commits datos → deploy → datos stale | Pipeline corre en el VPS directo → datos siempre frescos |

**Costo**: ~$5/mes (Hetzner CX22)

---

## Paso 1: Crear cuenta y servidor en Hetzner (5 min)

1. Ir a https://www.hetzner.com/cloud
2. Crear cuenta (necesitas tarjeta de credito)
3. Ir a "Cloud Console" → "Add Server"
4. Configurar:
   - **Location**: Ashburn, VA (mas cerca del mercado US) o Helsinki
   - **Image**: Ubuntu 22.04
   - **Type**: Shared vCPU → CX22 (2 vCPU, 4GB RAM, 40GB SSD)
   - **Networking**: Public IPv4 (activado)
   - **SSH Key**: Agregar tu clave SSH publica
     - Si no tenes: en tu PC correr `ssh-keygen -t ed25519` y copiar el contenido de `~/.ssh/id_ed25519.pub`
   - **Name**: agrocast-pro
5. Click "Create & Buy Now"
6. Anotar la IP del servidor

---

## Paso 2: Ejecutar setup automatico (10 min)

Desde tu PC (PowerShell o Git Bash):

```bash
# Copiar script al servidor
scp deploy/setup_vps.sh root@TU_IP:/root/

# Conectar al servidor
ssh root@TU_IP

# Ejecutar setup
chmod +x setup_vps.sh
./setup_vps.sh
```

El script te va a pedir:
- Dominio (opcional, podes dejar vacio)
- ANTHROPIC_API_KEY
- NEWS_API_KEY
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- AGROCAST_API_KEY

El script instala TODO automaticamente:
- Python 3.12 + virtualenv
- nginx (reverse proxy + gzip + cache)
- gunicorn (2 workers, 4 threads)
- systemd (auto-restart si crashea)
- firewall (solo SSH + HTTP/HTTPS)
- fail2ban (proteccion anti-brute-force)
- certbot (SSL automatico si tenes dominio)
- webhook receiver (auto-deploy en cada push)
- cron de backup (polling cada 5 min)
- logrotate (logs no llenan el disco)

---

## Paso 3: Configurar GitHub Webhook (2 min)

Para que cada `git push` haga deploy automatico:

1. Ir a https://github.com/Nicotina712/agrocast/settings/hooks
2. Click "Add webhook"
3. Configurar:
   - **Payload URL**: `http://TU_IP/deploy`
   - **Content type**: `application/json`
   - **Secret**: `agrocast-deploy-2026`
   - **Events**: "Just the push event"
4. Click "Add webhook"

Ahora cada push a main → deploy automatico en <30 segundos.

---

## Paso 4: Modificar GitHub Actions (opcional)

El pipeline ya no necesita commitear datos a git.
Puede correr directamente en el VPS via el scheduler interno (ya configurado).

Opcion A: **Desactivar GitHub Actions** (recomendado)
- El scheduler del VPS corre el pipeline cada 6 horas automaticamente
- Borrar o desactivar `.github/workflows/pipeline.yml`

Opcion B: **Mantener GitHub Actions como backup**
- Cambiar el pipeline para que NO commitee a git
- En su lugar, hacer un curl al VPS para triggear el pipeline:
  ```yaml
  - name: Trigger VPS pipeline
    run: curl -X POST http://TU_IP/api/trigger_pipeline?key=${{ secrets.AGROCAST_API_KEY }}
  ```

---

## Paso 5: Apuntar dominio (opcional)

Si tenes dominio:
1. En tu DNS, crear registro A: `agrocast.tudominio.com` → TU_IP
2. SSH al server:
   ```bash
   sudo certbot --nginx -d agrocast.tudominio.com
   ```
3. SSL automatico y renovacion automatica

---

## Paso 6: Desactivar Render

1. Ir a https://dashboard.render.com
2. Suspender o eliminar el servicio agrocast-web
3. Listo — ya no lo necesitas

---

## Comandos utiles

```bash
# Ver estado del servicio
sudo systemctl status agrocast

# Ver logs en tiempo real
journalctl -u agrocast -f

# Restart manual
sudo systemctl restart agrocast

# Deploy manual (sin webhook)
/opt/agrocast/deploy.sh

# Editar variables de entorno
nano /opt/agrocast/.env
sudo systemctl restart agrocast

# Ver uso de recursos
htop

# Ver espacio en disco
df -h
```

---

## Arquitectura final

```
Tu PC                          Hetzner VPS (CX22 — $5/mo)
  |                               |
  | git push                      |
  +-----> GitHub ----webhook----> nginx (:80/:443)
                                    |
                                    +---> gunicorn (:8000)
                                    |       |
                                    |       +-- Flask (news_server.py)
                                    |       +-- APScheduler (pipeline cada 6h)
                                    |       +-- News fetcher (background)
                                    |
                                    +---> /opt/agrocast/data/     (persistente)
                                    +---> /opt/agrocast/artifacts/ (persistente)
                                    +---> /var/log/agrocast/      (logs rotativos)
```

**Diferencias clave vs Render:**
- Nunca se duerme → carga instantanea
- 4GB RAM → sin OOM
- Disco persistente → datos nunca se pierden
- Pipeline corre EN el server → datos siempre frescos
- Deploy no toca data/ ni artifacts/ → no mas "arreglo 1, rompo 1"
- nginx cache + gzip → respuestas rapidas
- 2 workers gunicorn → puede servir requests mientras pipeline corre
