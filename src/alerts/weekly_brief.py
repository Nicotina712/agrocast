"""
src/alerts/weekly_brief.py
Brief semanal auto-generado en español via Claude API (Anthropic SDK).

Genera un resumen ejecutivo de ~400 palabras cada lunes con:
  - Performance de señales de la semana anterior
  - Top 3 drivers de mercado (qué señales movieron más)
  - Calendario de eventos próximos (WASDE, vencimientos)
  - Recomendación concreta para el productor

Entrega via Telegram + WhatsApp.
Requiere: ANTHROPIC_API_KEY en .env
"""

import json
import os
import re
from datetime import date, datetime, timedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STATE_PATH   = os.path.join(_PROJECT_ROOT, "data", "last_weekly_brief.json")


def _was_sent_this_week() -> bool:
    """Evita enviar más de un brief por semana."""
    if not os.path.exists(_STATE_PATH):
        return False
    try:
        with open(_STATE_PATH) as f:
            state = json.load(f)
        last = datetime.fromisoformat(state.get("sent_at", "2000-01-01"))
        return (datetime.now() - last).days < 6
    except Exception:
        return False


def _mark_sent():
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump({"sent_at": datetime.now().isoformat()}, f)


def _load_context() -> dict:
    """Carga todos los datos disponibles para construir el brief."""
    ctx = {}

    # Señal actual y historial
    try:
        import pandas as pd
        sig_path = os.path.join(_PROJECT_ROOT, "artifacts", "signals.csv")
        if os.path.exists(sig_path):
            df = pd.read_csv(sig_path, parse_dates=["Date"])
            df = df.sort_values("Date")
            recent = df.tail(10)
            ctx["current_signal"] = df["Signal"].iloc[-1] if not df.empty else "HOLD"
            ctx["signal_history"] = recent[["Date", "Signal", "Confidence"]].to_dict("records")
            ctx["current_price"]  = float(df["Price"].iloc[-1]) if "Price" in df.columns else None
    except Exception:
        pass

    # Backtest metrics
    try:
        bt_path = os.path.join(_PROJECT_ROOT, "artifacts", "backtest_summary.json")
        if os.path.exists(bt_path):
            with open(bt_path) as f:
                ctx["backtest"] = json.load(f)
    except Exception:
        pass

    # COT data
    try:
        cot_path = os.path.join(_PROJECT_ROOT, "data", "cot_soybeans.csv")
        if os.path.exists(cot_path):
            import pandas as pd
            cot = pd.read_csv(cot_path, parse_dates=["Date"])
            cot = cot.sort_values("Date")
            last_cot = cot.iloc[-1]
            ctx["cot"] = {
                "date":           str(last_cot.get("Date", "")[:10]),
                "noncomm_net":    float(last_cot.get("cot_noncomm_net", 0)),
                "cot_index":      float(last_cot.get("cot_index", 50)),
                "commercial_net": float(last_cot.get("cot_commercial_net", 0)),
            }
    except Exception:
        pass

    # WASDE próximos
    try:
        from src.data.wasde_dates import get_wasde_dates
        wasde = get_wasde_dates()
        ctx["wasde_upcoming"] = wasde[:2] if wasde else []
    except Exception:
        pass

    # Argentina
    try:
        ar_path = os.path.join(_PROJECT_ROOT, "data", "argentina_supply.json")
        if os.path.exists(ar_path):
            with open(ar_path) as f:
                ctx["argentina"] = json.load(f)
    except Exception:
        pass

    # Brazil exports
    try:
        br_path = os.path.join(_PROJECT_ROOT, "data", "brazil_exports.json")
        if os.path.exists(br_path):
            with open(br_path) as f:
                ctx["brazil"] = json.load(f)
    except Exception:
        pass

    # China demand
    try:
        cn_path = os.path.join(_PROJECT_ROOT, "data", "china_demand.json")
        if os.path.exists(cn_path):
            with open(cn_path) as f:
                ctx["china"] = json.load(f)
    except Exception:
        pass

    # Basis Uruguay
    try:
        basis_path = os.path.join(_PROJECT_ROOT, "data", "basis_uruguay.json")
        if os.path.exists(basis_path):
            with open(basis_path) as f:
                ctx["basis"] = json.load(f)
    except Exception:
        pass

    # Accuracy reciente
    try:
        acc_path = os.path.join(_PROJECT_ROOT, "artifacts", "accuracy.json")
        if os.path.exists(acc_path):
            with open(acc_path) as f:
                ctx["accuracy"] = json.load(f)
    except Exception:
        pass

    return ctx


# ─────────────────────────────────────────────────────────────────────────
# Mensaje del productor (determinista, lenguaje simple) — Tarea C
# Espeja lo que el productor ve en el panel, sin jerga técnica.
# No depende de LLM: siempre funciona si hay datos.
# ─────────────────────────────────────────────────────────────────────────
def _fmt_num(n, dec=0):
    if n is None:
        return "—"
    try:
        return f"{float(n):,.{dec}f}".replace(",", "@").replace(".", ",").replace("@", ".")
    except Exception:
        return str(n)


# ─────────────────────────────────────────────────────────────────────────
# Informe semanal del PRODUCTOR (Fase 4 del pivot, 2026-10)
# Esqueleto determinístico con los números del motor de decisión + resumen
# redactado por IA SOLO con esos números (validado; si falla, se omite).
# ─────────────────────────────────────────────────────────────────────────
_WEEKLY_STATE_PATH = os.path.join(_PROJECT_ROOT, "data", "producer_weekly_state.json")

# Frases de pronóstico de dirección que el redactor no puede usar.
_FORBIDDEN = re.compile(
    r"(va(n)? a (subir|bajar|caer|repuntar)|subir[áa]n?\b|bajar[áa]n?\b|caer[áa]n?\b|"
    r"esperamos que (el precio|suba|baje)|el precio (seguir[áa]|tender[áa])|"
    r"tendencia (alcista|bajista)|presi[oó]n (alcista|bajista)|rally|rebote)",
    re.IGNORECASE)


def _weekly_snapshot(b: dict) -> dict:
    dec = b.get("decision") or {}
    f = dec.get("fijar_precio") or {}
    g = dec.get("guardar_en_cosecha") or dec.get("guardar_hoy") or {}
    best = g.get("mejor") or {}
    return {
        "fecha": date.today().isoformat(),
        "precio_usd_ton": (b.get("precio_hoy") or {}).get("usd_ton"),
        "precio_fijable_neto": f.get("precio_fijable_usd_ton"),
        "margen_pct": f.get("margen_asegurable_pct"),
        "sugerido_pct": f.get("sugerido_pct"),
        "falta_pct": f.get("falta_pct"),
        "guardar_resultado": best.get("ganancia_esperada_usd_ton"),
        "guardar_hasta": best.get("hasta"),
        "vol_pct": dec.get("vol_anual_pct"),
    }


def _load_prev_snapshot() -> dict | None:
    try:
        with open(_WEEKLY_STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _save_snapshot(snap: dict) -> None:
    os.makedirs(os.path.dirname(_WEEKLY_STATE_PATH), exist_ok=True)
    with open(_WEEKLY_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1)


def _cambios(prev: dict | None, now: dict) -> list[str]:
    """Qué cambió vs el último informe enviado (en hechos, no pronósticos)."""
    if not prev:
        return []
    out = []
    p0, p1 = prev.get("precio_fijable_neto"), now.get("precio_fijable_neto")
    if p0 and p1 and abs(p1 - p0) >= 1:
        out.append(f"El precio que podés fijar pasó de {p0:.0f} a {p1:.0f} USD/ton neto ({p1 - p0:+.0f}).")
    s0, s1 = prev.get("sugerido_pct"), now.get("sugerido_pct")
    if s0 is not None and s1 is not None and s0 != s1:
        out.append(f"Lo sugerido para fijar cambió de {s0}% a {s1}% "
                   f"({'más riesgo de resignar margen' if s1 > s0 else 'menos riesgo de resignar margen'}).")
    g0, g1 = prev.get("guardar_resultado"), now.get("guardar_resultado")
    if g0 is not None and g1 is not None and abs(g1 - g0) >= 2:
        out.append(f"Guardar a cosecha pasó de {g0:+.0f} a {g1:+.0f} USD/ton según la curva.")
    if not out:
        out.append("Sin cambios relevantes en tus números respecto al informe anterior.")
    return out


def _numbers_in(text: str) -> set[str]:
    return {n.replace(".", "").replace(",", "") for n in re.findall(r"\d[\d.,]*", text)}


def _redactar_resumen(facts_text: str) -> str | None:
    """
    El LLM redacta 2-3 frases para el productor usando SOLO los hechos dados.
    Validación: sin frases de pronóstico de dirección y sin números que no
    estén en los hechos. Si no pasa (o no hay API key), devuelve None y el
    informe sale igual con su esqueleto determinístico.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0,
            system=(
                "Sos el redactor de AgroCast para productores de soja de Uruguay. Escribís en español "
                "rioplatense, simple y directo, como un asesor de confianza. Reglas estrictas: "
                "1) Usá SOLO los hechos y números que te paso; no agregues datos ni números nuevos. "
                "2) Nunca digas si el precio va a subir o bajar: AgroCast no pronostica la dirección del precio; "
                "habla de margen, riesgo y costos. 3) Máximo 3 frases y 420 caracteres. "
                "4) Sin saludos, sin firmas, sin emojis."),
            messages=[{"role": "user", "content":
                       "Hechos de esta semana:\n" + facts_text +
                       "\n\nEscribí el resumen de 2-3 frases: qué decisión le toca al productor y por qué."}],
        )
        text = "".join(blk.text for blk in msg.content if getattr(blk, "type", "") == "text").strip()
    except Exception as e:
        print(f"[Brief productor] redactor IA no disponible: {e}")
        return None
    if not text or len(text) > 480:
        return None
    if _FORBIDDEN.search(text):
        print(f"[Brief productor] resumen IA descartado (pronóstico de dirección): {text[:120]}")
        return None
    extra = _numbers_in(text) - _numbers_in(facts_text)
    if extra:
        print(f"[Brief productor] resumen IA descartado (números no provistos {extra}): {text[:120]}")
        return None
    return text


def build_producer_message(html: bool = True, use_llm: bool = True, _brief: dict | None = None) -> str | None:
    """
    Informe semanal del productor. `html=True` para Telegram (<b>),
    False para WhatsApp (*texto*). Sin pronósticos de dirección.
    """
    try:
        if _brief is None:
            from src.producer.producer_brief import build_producer_brief
            _brief = build_producer_brief()
    except Exception as e:
        print(f"[Brief productor] error build_producer_brief: {e}")
        return None
    bf = _brief
    if not bf or not bf.get("ok"):
        return None

    def B(t):  # negrita según canal
        return f"<b>{t}</b>" if html else f"*{t}*"

    dec = bf.get("decision") or {}
    f = dec.get("fijar_precio") or {}
    etapa = bf.get("etapa") or {}
    m = bf.get("momento") or {}
    p = bf.get("precio_hoy") or {}
    neto = bf.get("precio_neto") or {}
    mg = bf.get("margen") or {}
    ev = bf.get("proximo_evento")
    pre = etapa.get("nombre") in ("Siembra", "Cultivo en desarrollo")
    g = (dec.get("guardar_en_cosecha") if pre else dec.get("guardar_hoy")) or {}
    snap = _weekly_snapshot(bf)
    cambios = _cambios(_load_prev_snapshot(), snap)

    # Hechos para el redactor (los mismos números que van en el informe)
    facts = [f"Etapa: {etapa.get('nombre')} de la campaña {etapa.get('campania')}. {etapa.get('foco', '')}",
             f"Recomendación: {m.get('titulo', '')}. {m.get('explicacion', '')}"]
    if g.get("texto"):
        facts.append(g["texto"])
    facts += cambios
    resumen = _redactar_resumen("\n".join(facts)) if use_llm else None

    L = []
    L.append(f"🌾 {B('AgroCast — Tu semana')}")
    L.append(f"{etapa.get('icono', '📅')} Campaña {etapa.get('campania', '')} · {etapa.get('nombre', '')} · "
             f"{date.today().strftime('%d/%m/%Y')}")
    L.append("")
    if resumen:
        L.append(resumen)
        L.append("")

    L.append(f"💲 {B('Precio hoy')}: {_fmt_num(p.get('usd_ton'), 0)} USD/ton · neto {_fmt_num(neto.get('neto_usd_ton'), 0)} "
             f"(descontando flete y gastos)")
    if mg.get("margen_pct") is not None:
        L.append(f"💰 Tu margen: {'+' if mg['margen_pct'] >= 0 else ''}{mg['margen_pct']}% sobre tu costo "
                 f"({'+' if (mg.get('margen_total_usd') or 0) >= 0 else ''}{_fmt_num(mg.get('margen_total_usd'), 0)} USD sobre tu cosecha)")
    L.append("")

    icon = {"FIJAR": "✅", "GUARDAR": "📦", "VENDER": "💵", "NO_VENDER": "✋"}.get(m.get("momento"), "🟡")
    L.append(f"{icon} {B(m.get('titulo', '—'))}")
    if f.get("ok") and f.get("prob_perder_mitad_margen") is not None:
        L.append(f"   Si esperás a cosechar sin fijar: {f['prob_perder_mitad_margen']}% de chance de perder más de la mitad "
                 f"del margen, {f.get('prob_debajo_costo_a_cosecha')}% de quedar debajo de tu costo.")
    rq = f.get("rango_a_cosecha") or {}
    if rq.get("q10") is not None:
        L.append(f"   📏 Rango probable a cosecha (8 de cada 10 veces): {_fmt_num(rq['q10'], 0)}–{_fmt_num(rq['q90'], 0)} USD/ton. "
                 f"Hoy podés fijar {_fmt_num(f.get('precio_fijable_usd_ton'), 0)}.")
    if f.get("caja"):
        L.append(f"   💵 {f['caja']['mensaje']}")
    L.append("")

    if g.get("ok"):
        best = g["mejor"]
        L.append(f"📦 {B('Cuando coseches: ¿guardar?' if pre else '¿Guardar o vender?')}")
        if g.get("guardar_paga"):
            L.append(f"   Guardar hasta {best['hasta']} paga {best['ganancia_esperada_usd_ton']:+.0f} USD/ton "
                     f"(el mercado paga {best['paga_mercado_usd_ton']:+.0f}, te cuesta {best['costo_guardar_usd_ton']:.0f}).")
        else:
            L.append(f"   No paga con la curva de hoy: el mercado paga {best['paga_mercado_usd_ton']:+.0f} USD/ton hasta "
                     f"{best['hasta']} y guardar cuesta {best['costo_guardar_usd_ton']:.0f}. Solo pagaría si la base "
                     f"local mejora más de {best['mejora_base_necesaria_usd_ton']:.0f}.")
        L.append("")

    if cambios:
        L.append(f"🔄 {B('Qué cambió desde la semana pasada')}")
        for c in cambios:
            L.append(f"   • {c}")
        L.append("")

    drivers = bf.get("drivers_simple") or []
    if drivers:
        L.append(f"🌍 {B('Qué está pasando')}")
        for d in drivers[:4]:
            L.append(f"   {d.get('icono', '')} {d.get('etiqueta', '')}: {d.get('detalle', '')}")
        L.append("")

    if ev:
        urg = "⚠️" if ev.get("inminente") else "🗓️"
        L.append(f"{urg} {B('Fecha a tener en cuenta')}: {ev.get('nombre')} el {ev.get('fecha_es')} "
                 f"(en {ev.get('dias_para')} días). {ev.get('impacto')}.")
        L.append("")

    L.append("ℹ️ AgroCast no adivina si el precio sube o baja: te muestra tu margen, el riesgo de esperar y si guardar paga.")
    L.append("— AgroCast · decisiones para tu cosecha")
    return "\n".join(L)


def generate_producer_weekly(force: bool = False) -> str | None:
    """
    Genera y envía el informe semanal del productor en lenguaje simple.
    Determinista (no requiere LLM). Solo corre los lunes salvo force=True.
    """
    today = date.today()
    if not force and today.weekday() != 0:
        return None
    if not force and _was_sent_this_week():
        print("[Brief productor] Ya enviado esta semana — omitiendo.")
        return None

    from src.producer.producer_brief import build_producer_brief
    _b = build_producer_brief()
    msg_tg = build_producer_message(html=True, _brief=_b)
    # WhatsApp reusa el mismo resumen IA (evita 2 llamadas y textos distintos)
    msg_wa = None
    if msg_tg:
        msg_wa = re.sub(r"</?b>", "", msg_tg)
        for line in re.findall(r"<b>(.*?)</b>", msg_tg):
            msg_wa = msg_wa.replace(line, f"*{line}*", 1)
    if not msg_tg:
        print("[Brief productor] Sin datos para construir el informe.")
        return None

    # Telegram
    try:
        from src.alerts.telegram_bot import _send_message
        _send_message(msg_tg, parse_mode="HTML")
        print("[Brief productor] Enviado por Telegram ✅")
    except Exception as e:
        print(f"[Brief productor] Telegram error: {e}")

    # WhatsApp
    try:
        from src.alerts.whatsapp_bot import send_whatsapp
        send_whatsapp(msg_wa)
        print("[Brief productor] Enviado por WhatsApp ✅")
    except Exception as e:
        print(f"[Brief productor] WhatsApp error: {e}")

    # Guardar copia
    try:
        path = os.path.join(_PROJECT_ROOT, "data", f"brief_productor_{today.isoformat()}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(msg_wa)
    except Exception:
        pass

    _mark_sent()
    try:
        _save_snapshot(_weekly_snapshot(_b))   # base del "qué cambió" de la próxima semana
    except Exception as e:
        print(f"[Brief productor] no se pudo guardar snapshot: {e}")
    return msg_tg


def _build_prompt(ctx: dict) -> str:
    today = date.today()
    week  = today.isocalendar()[1]

    return f"""Eres el analista jefe de AgroCast PRO, un sistema de inteligencia de mercado para productores de soja en Uruguay.

Es lunes {today.strftime('%d de %B de %Y')} (semana {week} del año).

Datos del mercado disponibles:
{json.dumps(ctx, indent=2, default=str, ensure_ascii=False)}

Redactá un BRIEF SEMANAL en español rioplatense (voseo), con el siguiente formato exacto:

---
🌱 **AgroCast PRO — Brief Semanal {today.strftime('%d/%m/%Y')}**

📊 **Señal actual:** [BUY/SELL/HOLD] con [X]% de confianza
💲 **Precio Chicago:** [precio] USc/bu

**¿Qué pasó la semana pasada?**
[2-3 oraciones sobre los movimientos de precio y qué drivers fueron más relevantes]

**Top 3 drivers esta semana:**
1. [Driver más importante con datos concretos]
2. [Segundo driver con datos]
3. [Tercer driver con datos]

**Señales clave del modelo:**
• COT (especuladores): [interpretación del posicionamiento]
• Argentina: [cepo/spread/retenciones y su impacto]
• Brasil: [pace de exportaciones]
• Basis Uruguay: [spread con Chicago]

**📅 Calendario de la semana:**
• [Eventos importantes: WASDE, reportes USDA, vencimientos]

**💡 Recomendación para el productor:**
[Una recomendación concreta y accionable en 2-3 oraciones. Incluí si conviene vender ahora o esperar, y por qué.]

---

Tono: profesional pero directo, sin jerga técnica innecesaria. Máximo 400 palabras. Basate SOLO en los datos proporcionados."""


def generate_weekly_brief(force: bool = False) -> str | None:
    """
    Genera y envía el brief semanal. Solo corre los lunes o si force=True.

    Retorna el texto del brief o None si no corresponde enviarlo.
    """
    today = date.today()
    is_monday = today.weekday() == 0

    if not force and not is_monday:
        return None

    if not force and _was_sent_this_week():
        print("[Brief] Ya enviado esta semana — omitiendo.")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[Brief] ANTHROPIC_API_KEY no configurada — brief no generado.")
        return None

    print("[Brief] Generando brief semanal con Claude API…")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        ctx    = _load_context()
        prompt = _build_prompt(ctx)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",   # rápido y económico para texto
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        brief_text = message.content[0].text
        print(f"[Brief] Generado ({len(brief_text)} chars)")

        # Enviar por Telegram
        try:
            from src.alerts.telegram_bot import send_telegram
            send_telegram(brief_text)
            print("[Brief] Enviado por Telegram ✅")
        except Exception as e:
            print(f"[Brief] Telegram error: {e}")

        # Enviar por WhatsApp
        try:
            from src.alerts.whatsapp_bot import send_whatsapp
            send_whatsapp(brief_text)
            print("[Brief] Enviado por WhatsApp ✅")
        except Exception as e:
            print(f"[Brief] WhatsApp error: {e}")

        # Guardar copia local
        brief_path = os.path.join(_PROJECT_ROOT, "data", f"brief_{today.isoformat()}.txt")
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(brief_text)

        _mark_sent()
        return brief_text

    except Exception as e:
        print(f"[Brief] Error generando brief: {e}")
        return None
