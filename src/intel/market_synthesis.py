"""
src/intel/market_synthesis.py
Brief Ejecutivo V2 — productor y trader.

Dos briefs distintos generados con Claude Sonnet:
  - PRODUCTOR (Dashboard): lenguaje simple, acciones concretas con %,
    rango esperado diario, honesto sobre limitaciones.
  - TRADER (pestaña Trader): jerga técnica, niveles, RSI, Bollinger,
    COT, spread, señal compuesta con factores.

Ambos consumen TODA la inteligencia disponible:
  - Signal Breakdown (composite)
  - Decision Classifier (P(WAIT), partial_sell, horizonte óptimo)
  - Narrative Forecast (rango diario 1d, 7d, 15d, 30d por análogos)
  - Evento narrativo actual (tipo, dirección, strength, fade_risk)
  - News Intel (drivers, sentiment)
  - Fundamentales (China, WASDE, Argentina, Brazil, basis)
  - Forecast 30d, multi-commodity, contrato actual
  - Track record (accountability)
  - Drift monitor (calibración del modelo)
"""

import json
import os
from datetime import datetime, timedelta

# Ensure .env is loaded so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(_env_path):
        _load_dotenv(_env_path, override=True)
except ImportError:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUTPUT_DIR   = os.path.join(_PROJECT_ROOT, "data")
_TTL_HOURS    = 4

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 4000


def _output_path(brief_type: str) -> str:
    if brief_type == "trader":
        return os.path.join(_OUTPUT_DIR, "market_synthesis_trader.json")
    return os.path.join(_OUTPUT_DIR, "market_synthesis.json")


def _is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=_TTL_HOURS)


def _load_context() -> dict:
    """Junta TODA la inteligencia disponible y la pre-digiere."""
    ctx = {}

    def _read_json(rel: str, key: str) -> None:
        p = os.path.join(_PROJECT_ROOT, rel)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    ctx[key] = json.load(f)
            except Exception:
                pass

    # ── Fuentes originales ──
    _read_json("data/news_intel.json",       "news_intel")
    _read_json("data/signal_breakdown.json", "signal")
    _read_json("data/argentina_supply.json", "argentina")
    _read_json("data/brazil_exports.json",   "brazil")
    _read_json("data/china_demand.json",     "china")
    _read_json("data/basis_uruguay.json",    "basis")
    _read_json("data/wasde_official.json",   "wasde")
    _read_json("data/multi_commodity.json",  "multi_commodity")
    _read_json("data/current_contract.json", "contract")

    # ── Nuevas fuentes de inteligencia ──
    _read_json("artifacts/narrative_forecast/latest.json", "narrative_forecast")
    _read_json("artifacts/event_memory.json",              "event_memory_summary")
    _read_json("artifacts/drift_monitor.json",             "drift_monitor")

    # Forecast (CSV → resumen)
    try:
        import pandas as pd
        f_path = os.path.join(_PROJECT_ROOT, "artifacts", "forecast.csv")
        if os.path.exists(f_path):
            df = pd.read_csv(f_path)
            if not df.empty:
                ctx["forecast"] = {
                    "today":   round(float(df["Soybeans"].iloc[0]), 2),
                    "d7":      round(float(df["Soybeans"].iloc[min(6, len(df)-1)]), 2),
                    "d30":     round(float(df["Soybeans"].iloc[-1]), 2),
                    "horizon": len(df),
                }
    except Exception:
        pass

    # Accountability (track record)
    try:
        from src.trader.accountability import get_accountability_records
        recs = get_accountability_records()
        if isinstance(recs, dict) and recs.get("summary"):
            ctx["accuracy"] = recs["summary"]
    except Exception:
        pass

    # LLM accountability (hit rate del brief)
    try:
        from src.intel.llm_accountability import get_summary
        ctx["llm_track_record"] = get_summary()
    except Exception:
        pass

    return ctx


def _digest_context(ctx: dict) -> str:
    """Pre-digiere el contexto crudo en un resumen estructurado y conciso."""
    lines = []

    # ── 1. Señal compuesta ──
    sig = ctx.get("signal") or {}
    lines.append("=== SENAL COMPUESTA (FUENTE DE VERDAD) ===")
    lines.append(f"  composite_signal: {sig.get('composite_signal', '?')}")
    lines.append(f"  composite_raw: {sig.get('composite_raw', '?')} (rango -1 a +1)")
    lines.append(f"  composite_score: {sig.get('composite_score', '?')}/100")
    for f in (sig.get("factors") or []):
        w = f.get("weight", 0)
        tag = f"peso {int(w*100)}%" if w > 0 else "INFORMATIVO"
        lines.append(f"  - {f.get('name')}: score={f.get('score')} dir={f.get('direction')} ({tag})")
        lines.append(f"    detalle: {f.get('detail','')}")

    # ── 2. Decision Classifier ──
    nf = ctx.get("narrative_forecast") or {}
    fc = nf.get("forecast") or {}
    bts = nf.get("backtests") or {}
    lines.append("")
    lines.append("=== DECISION CLASSIFIER (vender vs esperar) ===")
    # We don't have the DC prediction cached separately, but the narrative forecast
    # has event info and backtests that show the DC's track record
    if bts:
        for h in ["1d", "7d", "15d", "30d"]:
            bt = bts.get(h, {})
            if bt.get("ok"):
                nar = bt.get("strategies", {}).get("narrative", {})
                aw = bt.get("strategies", {}).get("always_wait", {})
                lines.append(f"  Backtest {h}: narrativa={nar.get('mean_pnl_usd_ton','?')} USD/ton vs "
                             f"always-wait={aw.get('mean_pnl_usd_ton','?')} USD/ton "
                             f"(n={bt.get('n_decisions','?')}, activo {bt.get('narrative_active_pct','?')}%)")

    # ── 3. Evento narrativo actual + rango ──
    lines.append("")
    lines.append("=== EVENTO NARRATIVO ACTUAL ===")
    if fc.get("ok"):
        lines.append(f"  Evento activo: {fc.get('event_active')}")
        lines.append(f"  Tipo: {fc.get('event_type')} | Direccion: {fc.get('event_direction')}")
        lines.append(f"  Fuerza narrativa: {fc.get('narrative_strength')} | Fade risk: {fc.get('fade_risk')}")
        lines.append(f"  Narrativa diaria: {fc.get('daily_narrative','')}")
        lines.append("")
        lines.append("  RANGOS ESPERADOS POR HORIZONTE (basado en analogos historicos):")
        forecasts = fc.get("forecasts", {})
        for h in ["1d", "7d", "15d", "30d"]:
            hf = forecasts.get(h, {})
            if hf.get("available"):
                r = hf["range_pct"]
                usd = hf.get("usd_ton", {})
                lines.append(f"    {h}: Q10={r['q10']}% | mediana={r['median']}% | Q90={r['q90']}% | P(suba)={hf['p_up']*100:.0f}% | n={hf['n_samples']}")
                if usd:
                    lines.append(f"        en USD/ton: {usd.get('q10','')} a {usd.get('q90','')} (mediana {usd.get('median','')})")
    else:
        lines.append("  (narrative forecast no disponible)")

    # ── 4. Event memory stats ──
    em = ctx.get("event_memory_summary") or {}
    if em.get("ok"):
        lines.append("")
        lines.append(f"=== EVENT MEMORY: {em.get('n_events',0)} eventos historicos ===")
        lines.append(f"  Fade rate 7d: {em.get('fade_rate_7d_pct','?')}%")
        lines.append(f"  Tipos dominantes: {em.get('event_types',{})}")

    # ── 5. Noticias ──
    ni = ctx.get("news_intel") or {}
    if ni.get("n_articles"):
        lines.append("")
        lines.append(f"=== NOTICIAS ({ni.get('n_articles',0)} articulos analizados) ===")
        drivers = ni.get("drivers", {})
        for k, v in drivers.items():
            if isinstance(v, (int, float)) and abs(v) > 0.1:
                lines.append(f"  {k}: {v}")
            elif isinstance(v, dict) and abs(v.get("signal", 0)) > 0.1:
                lines.append(f"  {k}: signal={v.get('signal')}")

    # ── 6. Fundamentales ──
    lines.append("")
    lines.append("=== FUNDAMENTALES ===")

    china = ctx.get("china") or {}
    if china:
        cm = china.get("crush_margin") or {}
        lines.append(f"  China: crush_margin={cm.get('margin_usd_ton','?')} USD/ton ({cm.get('signal','?')}) | "
                     f"imports_yoy={china.get('imports_yoy_pct','?')}% | "
                     f"demand_score={china.get('demand_score','?')}/100 | "
                     f"as_of={china.get('as_of','?')}")

    wasde = ctx.get("wasde") or {}
    if wasde:
        lines.append(f"  WASDE: signal={wasde.get('signal','?')} | stocks={wasde.get('ending_stocks_mmt','?')} MMT | "
                     f"surprise={wasde.get('surprise_score','?')}")

    arg = ctx.get("argentina") or {}
    if arg:
        lines.append(f"  Argentina: retenciones={arg.get('retenciones','?')}% | cepo={arg.get('cepo','?')}")

    brazil = ctx.get("brazil") or {}
    if brazil:
        lines.append(f"  Brasil: exports_ytd={brazil.get('exported_ytd_mmt','?')} MMT | "
                     f"yoy={brazil.get('yoy_pct','?')}% | "
                     f"pace_semanal={brazil.get('weekly_pace_mmt','?')} MMT/sem | "
                     f"as_of={brazil.get('as_of','?')}")

    basis = ctx.get("basis") or {}
    if basis and basis.get("basis_usd_ton") is not None:
        lines.append(f"  Basis Uruguay: {basis.get('basis_usd_ton',0):+.1f} USD/ton | "
                     f"local={basis.get('fob_usd_ton','?')} | cbot_ref={basis.get('cbot_usd_ton','?')} | "
                     f"signal={basis.get('signal','?')} | zscore={basis.get('basis_zscore','?')} | "
                     f"source={basis.get('source','?')} | as_of={basis.get('as_of','?')}")
    elif basis:
        lines.append(f"  Basis UY: sin datos actualizados")

    # ── 7. Multi-commodity ──
    mc = ctx.get("multi_commodity") or {}
    if mc:
        lines.append("")
        lines.append("=== CROSS-MARKET ===")
        for k, v in mc.items():
            if isinstance(v, dict):
                lines.append(f"  {k}: price={v.get('price','?')} | RSI={v.get('rsi','?')} | "
                             f"chg5d={v.get('change_5d_pct','?')}% | signal={v.get('signal','?')}")

    # ── 8. Forecast ──
    fcast = ctx.get("forecast") or {}
    if fcast:
        lines.append("")
        lines.append(f"=== FORECAST 30d (INFORMATIVO, peso 0% en composite) ===")
        lines.append(f"  Hoy: {fcast.get('today','')} | D7: {fcast.get('d7','')} | D30: {fcast.get('d30','')} USc/bu")

    # ── 9. Contrato actual ──
    contract = ctx.get("contract") or {}
    if contract:
        lines.append("")
        lines.append(f"=== CONTRATO ACTUAL ===")
        lines.append(f"  {json.dumps(contract, default=str)}")

    # ── 10. Track record y drift ──
    acc = ctx.get("accuracy") or {}
    llm_tr = ctx.get("llm_track_record") or {}
    drift = ctx.get("drift_monitor") or {}
    lines.append("")
    lines.append("=== TRACK RECORD Y CALIBRACION ===")
    if acc:
        lines.append(f"  Accountability modelo: {json.dumps(acc, default=str)}")
    if llm_tr:
        lines.append(f"  Track record brief LLM: {json.dumps(llm_tr, default=str)}")
    regime = drift.get("regime_shift", {})
    if regime:
        lines.append(f"  Drift monitor: p_buy_recent={regime.get('p_buy_recent_90d','?')} | "
                     f"p_buy_hist={regime.get('p_buy_historical','?')} | "
                     f"z_delta={regime.get('z_delta','?')}")

    return "\n".join(lines)


def _build_prompt_producer(digest: str) -> str:
    today = datetime.now().strftime("%d/%m/%Y")
    return f"""Sos el asesor de mercado de AgroCast PRO. Tu audiencia son PRODUCTORES de soja en Uruguay y Argentina — gente del campo, no traders. Escribis en espanol rioplatense (voseo).

Hoy es {today}.

Toda la inteligencia del sistema:
{digest}

Genera un BRIEF EJECUTIVO para el PRODUCTOR. Formato JSON estricto:

{{
  "headline": "frase de 8-12 palabras, lenguaje simple, sin jerga tecnica",
  "stance": "ALCISTA" | "BAJISTA" | "NEUTRAL",
  "conviction": 0-100,
  "situacion_hoy": "2-3 oraciones explicando que pasa HOY en el mercado en lenguaje simple. Incluir el rango esperado diario si hay evento activo. Ej: 'Hoy hay un evento de tipo macro que historicamente mueve el precio entre -X% y +Y%'",
  "que_hacer": {{
    "accion_principal": "UNA oracion clara: vender / esperar / vender parcial con %",
    "porcentaje_sugerido": "X% del stock",
    "horizonte": "1d / 7d / 15d / 30d",
    "razon": "1 oracion con el POR QUE, sin jerga"
  }},
  "rangos_esperados": {{
    "hoy_manana": "rango en % y USD/ton para 1d",
    "semana": "rango en % para 7d",
    "quincenal": "rango en % para 15d"
  }},
  "contexto_clave": [
    "hecho 1 en lenguaje simple (ej: 'China no esta comprando soja, los margenes de crushing son negativos')",
    "hecho 2",
    "hecho 3 maximo"
  ],
  "riesgos": ["riesgo 1 en lenguaje simple", "riesgo 2"],
  "honestidad": "1-2 oraciones sobre las limitaciones: track record real, que tan confiable es esto, que NO sabemos",
  "data_gaps": ["que info falta para tener mas certeza"]
}}

REGLAS CRITICAS:
- El campo stance DEBE coincidir con la senal compuesta: BUY->ALCISTA, SELL->BAJISTA, HOLD->NEUTRAL.
- PROHIBIDO usar jerga tecnica: nada de RSI, Bollinger, golden cross, spread, crush margin, z-score.
  Traduci todo a lenguaje de campo: "el precio esta fuerte pero cerca de un techo", "la demanda china esta fria".
- que_hacer DEBE ser concreto con porcentaje. Usa la info del decision classifier y narrative forecast.
  Si el modelo dice SELL con X%, traducilo a "vender X% del stock".
  Si dice HOLD/INDIFFERENT, decilo honestamente: "no hay senal clara, mantener posicion".
- rangos_esperados DEBEN venir de los datos del narrative forecast (Q10/Q90). No inventes numeros.
- honestidad: menciona el track record real si lo tenes, y se claro sobre lo que el modelo NO puede predecir.
- conviction: basate en |composite_raw|*100, ajustado por coherencia entre factores y track record.
- Maximo 3 items en contexto_clave (solo lo que importa, no todo).
- Devolve SOLO el JSON, sin markdown, sin texto extra.
"""


def _build_prompt_trader(digest: str) -> str:
    today = datetime.now().strftime("%d/%m/%Y")
    return f"""Sos el analista tecnico de AgroCast PRO. Tu audiencia son TRADERS y operadores de futuros ZS (soybeans CBOT). Podes usar jerga tecnica libremente. Escribis en espanol.

Hoy es {today}.

Toda la inteligencia del sistema:
{digest}

Genera un BRIEF TECNICO para el TRADER. Formato JSON estricto:

{{
  "headline": "frase de 8-12 palabras, puede incluir niveles y jerga tecnica",
  "stance": "ALCISTA" | "BAJISTA" | "NEUTRAL",
  "conviction": 0-100,
  "senal_compuesta": {{
    "score": "composite_score/100",
    "raw": "composite_raw",
    "factores_clave": "resumen de los factores que ponderan y su score"
  }},
  "tecnico": {{
    "niveles": "soporte, resistencia, MA50, MA200, Bollinger",
    "rsi": "valor y zona",
    "momentum": "5d y 20d",
    "pattern": "golden cross, divergencia, etc."
  }},
  "narrativo": {{
    "evento_activo": "tipo, direccion, strength, fade_risk",
    "rango_1d": "Q10-Q90 del narrative forecast",
    "rango_7d": "Q10-Q90",
    "analogos": "n analogos, distribucion de tipos, fade rate"
  }},
  "cross_market": {{
    "oil": "precio, cambio, impacto en soja via biofuel",
    "dollar": "DXY, cambio, impacto",
    "corn": "ratio, arbitraje"
  }},
  "fundamentales": {{
    "china": "crush margin, imports, demand score",
    "wasde": "stocks, surprise, signal",
    "supply": "Brasil pace, Argentina retenciones"
  }},
  "decision_classifier": {{
    "horizonte_optimo": "cual horizonte tiene mas conviccion",
    "p_wait": "probabilidad de que esperar pague",
    "partial_sell": "recomendacion graduada si aplica"
  }},
  "trade_idea": "setup concreto con entry, SL, TP si hay senal. Si no hay senal, decilo.",
  "risks": ["riesgo 1 tecnico", "riesgo 2"],
  "track_record": "hit rate del brief, drift del modelo, calibracion",
  "data_gaps": ["que info falta"]
}}

REGLAS:
- stance DEBE coincidir con composite_signal: BUY->ALCISTA, SELL->BAJISTA, HOLD->NEUTRAL.
- USA toda la inteligencia disponible. No ignores el narrative forecast, el event memory, ni el track record.
- Si el drift monitor muestra cambio de regimen, mencionalo.
- Incluir el Forecast 30d SOLO como referencia informativa con la advertencia de que tiene peso 0%.
- Si no hay trade idea clara, decilo: "sin setup definido, esperar confirmacion".
- conviction basada en |composite_raw|*100 ajustada por coherencia.
- Devolve SOLO el JSON, sin markdown, sin texto extra.
"""


def synthesize(force: bool = False, brief_type: str = "producer") -> dict | None:
    """Genera o devuelve cache del brief de mercado.

    brief_type: "producer" (Dashboard) o "trader" (pestaña Trader)
    """
    path = _output_path(brief_type)

    if not force and _is_fresh(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[synthesis] ANTHROPIC_API_KEY no configurada")
        return None

    ctx = _load_context()
    if not ctx:
        print("[synthesis] sin contexto disponible")
        return None

    digest = _digest_context(ctx)

    if brief_type == "trader":
        prompt = _build_prompt_trader(digest)
    else:
        prompt = _build_prompt_producer(digest)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()

        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("` \n")

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                result = json.loads(m.group())
            else:
                raise
        result["_generated_at"] = datetime.utcnow().isoformat()
        result["_model"]        = MODEL
        result["_brief_type"]   = brief_type
        result["_input_keys"]   = list(ctx.keys())

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[synthesis:{brief_type}] OK stance={result.get('stance')} conv={result.get('conviction')}")

        if brief_type == "producer":
            try:
                from src.intel.llm_accountability import record_today, evaluate_due
                record_today()
                evaluate_due()
            except Exception as _e:
                print(f"[synthesis] accountability skip: {_e}")

        return result

    except Exception as e:
        print(f"[synthesis:{brief_type}] error: {e}")
        import traceback; traceback.print_exc()
        return None


def load_synthesis(brief_type: str = "producer") -> dict:
    path = _output_path(brief_type)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


if __name__ == "__main__":
    import sys
    bt = sys.argv[1] if len(sys.argv) > 1 else "producer"
    out = synthesize(force=True, brief_type=bt)
    print(json.dumps(out, indent=2, ensure_ascii=False) if out else "FAILED")
