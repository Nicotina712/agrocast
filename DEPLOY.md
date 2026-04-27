# AgroCast PRO — Deploy gratis (GitHub Actions + Render Free)

Esta guía deja AgroCast en la nube **gratis**, accesible desde cualquier
navegador. Se separa el sistema en dos piezas:

```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Actions (cron cada 6h, 2000 min/mes free)            │
│   └── corre src.pipeline → commitea data/ y artifacts/      │
└──────────────────┬──────────────────────────────────────────┘
                   │ git push automático
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ GitHub repo (main)                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │ webhook
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Render Free Web (750 h/mes, 512 MB RAM)                     │
│   └── gunicorn wsgi:app — read-only, sirve CSVs del repo    │
└─────────────────────────────────────────────────────────────┘
```

**Costo total: USD 0/mes.**

---

## Limitaciones del free tier (importante saberlas)

| Limitación | Impacto | Mitigación |
|---|---|---|
| Render web **duerme tras 15 min sin tráfico** | Primer request = ~30 s de cold start | Aceptable pre-PMF; usar UptimeRobot free para mantenerlo despierto en horario activo |
| **512 MB RAM** en Render | xgboost predict y pandas justo entran | El entrenamiento corre en GitHub Actions (7 GB RAM gratis), no acá |
| **Sin disco persistente** en free | Los artifacts se pierden al redeploy | Por eso el pipeline corre en Actions y commitea al repo |
| **Sin Postgres free** en Render | — | Seguimos con CSVs/parquet (ya funciona así) |
| Repo público o 2000 min/mes Actions free | — | 4 corridas/día × ~7 min = ~840 min/mes, cabe |
| Archivos en repo deben pesar < 100 MB c/u | — | Ya excluido `psd_oilseeds.csv` (65 MB) en `.gitignore` |

---

## Pre-requisitos

1. **Cuenta GitHub** (gratis)
2. **Cuenta Render** (gratis, sin tarjeta de crédito)
3. **API keys**:
   - `ANTHROPIC_API_KEY` — https://console.anthropic.com (free tier 5 USD inicial)
   - `NEWSAPI_KEY` — https://newsapi.org (free 100 req/día)
   - `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` — BotFather en Telegram
   - `AGROCAST_API_KEY` — generala con `python -c "import secrets; print(secrets.token_urlsafe(32))"`

---

## Paso 1 — Subir el repo a GitHub

```bash
cd "C:/Users/Lenovo/OneDrive/Escritorio/MVP"
git init
git add .
git commit -m "AgroCast PRO — initial commit (deploy-ready free tier)"
git branch -M main
# Crear el repo vacío en https://github.com/new (nombre: agrocast)
git remote add origin https://github.com/<TU_USUARIO>/agrocast.git
git push -u origin main
```

> **Importante:** `.env` está en `.gitignore` — nunca lo subas. Las secrets
> van en GitHub Settings y en Render Dashboard.

Si el push falla por archivos grandes (>100 MB), revisar `.gitignore`.

---

## Paso 2 — Configurar secrets en GitHub Actions

GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.

Agregar **una por una**:

| Name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `NEWSAPI_KEY` | tu key |
| `TELEGRAM_BOT_TOKEN` | `123456:ABC...` |
| `TELEGRAM_CHAT_ID` | `-1001234567890` |
| `AGROCAST_API_KEY` | la que generaste |

GitHub Actions las inyecta en el workflow `pipeline.yml` cada 6 h.

---

## Paso 3 — Disparar el primer pipeline en Actions

1. GitHub repo → tab **Actions** → seleccionar **AgroCast Pipeline (every 6h)**.
2. Click **Run workflow** → branch `main` → **Run workflow**.
3. Esperá ~7 min. Cuando termina exitoso, verás un commit nuevo
   `data: refresh pipeline artifacts (...)` con los CSVs actualizados en `main`.

Si falla, mirá los logs del job. Lo más común: alguna secret mal pegada.

---

## Paso 4 — Crear el web service en Render

1. https://render.com → **New +** → **Blueprint**.
2. Connect GitHub → seleccionar el repo `agrocast`.
3. Render detecta `render.yaml` y muestra:
   - `agrocast-web` (Free plan)
4. Click **Apply**. Render arranca el primer build (~5 min).

---

## Paso 5 — Configurar secrets en Render

Mientras buildea, abrí **agrocast-web** → tab **Environment**.

Pegar las **mismas 5 secrets** que pusiste en GitHub Actions:

- `ANTHROPIC_API_KEY`
- `NEWSAPI_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `AGROCAST_API_KEY`

Save → Render relanza el deploy con las nuevas vars.

---

## Paso 6 — Verificar

Render te asigna una URL tipo `https://agrocast-web.onrender.com`.

```
https://agrocast-web.onrender.com/healthz
  → {"status":"ok","service":"agrocast"}

https://agrocast-web.onrender.com/
  → dashboard (puede tardar 30 s la primera vez)

https://agrocast-web.onrender.com/api/forecast?api_key=<TU_AGROCAST_API_KEY>
  → JSON con el forecast 30d
```

---

## Paso 7 — Mantenerlo despierto (opcional, gratis)

El free tier duerme tras 15 min sin tráfico. Si querés que esté
siempre caliente:

1. **UptimeRobot** (https://uptimerobot.com — free 50 monitors).
2. Crear un monitor HTTP a `https://agrocast-web.onrender.com/healthz`
   cada 5 min.
3. Listo: el ping mantiene la instancia activa.

> Cuidado: los 750 h/mes free son una sola instancia 24/7. Si tenés
> dos servicios free, las horas se reparten.

---

## Operación diaria

| Acción | Cómo |
|---|---|
| Ver corridas del pipeline | GitHub repo → tab **Actions** |
| Forzar pipeline manual | Actions → Run workflow |
| Ver logs del web | Render Dashboard → Logs |
| Forzar redeploy | Render → Manual Deploy → Deploy latest commit |
| Cambiar frecuencia del cron | editar `cron:` en `.github/workflows/pipeline.yml` |
| Pausar el cron | Actions → Pipeline → ··· → Disable workflow |

---

## Local + nube en paralelo

Tu instancia local sigue intacta:

```bash
python "MVP lectura de noticias/news_server.py"
```

Arranca con APScheduler activo (porque local no tiene `RUN_SCHEDULER=0`),
corre el pipeline de bootstrap si falta, y sirve en `localhost:8000`.

Las dos instancias son **independientes** — local usa tu disco, la web
usa el repo. Si querés que local sincronice con lo último del cron,
hacé `git pull` y reiniciá el server.

---

## Cuándo migrar a plan pago

Mové a Render Starter (USD 7/mes) cuando ocurra alguno de:

- Tu primer cliente pago se queja del cold start.
- El repo supera 500 MB por acumulación de artifacts (Git lento).
- Necesitás Postgres para queries históricas grandes.
- 2000 min/mes de Actions no alcanzan (improbable hasta que agregues
  más commodities).

La migración es trivial: cambiar `plan: free` → `plan: starter` en
`render.yaml`, agregar un disk y la DB, push. La guía paga está en el
historial de git de este archivo.

---

## Troubleshooting

### El web service tira "Internal Server Error" pero `/healthz` funciona
Probablemente faltan los CSVs en el repo. Disparar el pipeline en
Actions → **Run workflow** → esperar al commit → Render redeployea solo.

### "ModuleNotFoundError: news_server" en build de Render
`wsgi.py` agrega `MVP lectura de noticias/` a `sys.path`. Verificá que
la carpeta no fue renombrada en el push. Linux distingue mayúsculas.

### El pipeline en Actions corre pero no commitea
Probablemente `permissions: contents: write` falta. Está en el yaml por
default. Si seguís con problemas, GitHub Settings → Actions → Workflow
permissions → "Read and write permissions".

### Cold start eterno (>60 s)
Free tier en Render a veces se ralentiza si Render está saturado.
Solución: UptimeRobot ping cada 5 min (paso 7).

### Telegram alerta de pipeline fail pero todo se ve bien
El step `Notify on failure` solo dispara si el step previo falla.
Revisar logs de Actions para detalle.

### Secrets de Anthropic agotadas
Free tier de Anthropic son ~5 USD iniciales. Cuando se acaban, el cache
permanente sigue sirviendo a las noticias ya analizadas, las nuevas caen
al fallback VADER. La señal sigue funcionando, solo pierde la capa
estructurada hasta recargar.

---

## Resumen de archivos creados para el deploy

| Archivo | Función |
|---|---|
| `wsgi.py` | Entrypoint de gunicorn (resuelve el espacio en `MVP lectura de noticias/`) |
| `render.yaml` | Blueprint de Render — free plan, sin disk ni DB |
| `.github/workflows/pipeline.yml` | Cron cada 6h en Actions, commitea artifacts |
| `Procfile` | Fallback Heroku/Railway |
| `requirements.txt` | + gunicorn, flask-cors, sqlalchemy, pyarrow |
| `.env.example` | Template de las env vars |
| `.gitignore` | Excluye .env, caches, parquet pesados, USDA dump 65MB |
