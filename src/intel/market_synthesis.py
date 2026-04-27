"""
src/intel/market_synthesis.py
Agente de síntesis de mercado basado en Claude Sonnet.

Mientras que `news_analyst` analiza UN artículo a la vez (Haiku, barato),
este módulo toma TODA la inteligencia disponible (intel agregado, señal
compuesta, fundamentos, COT, basis, WASDE, forecast) y produce un brief
ejecutivo coherente para el productor.

Uso típico: una llamada por día (post-pipeline), no por refresh de UI.
Cache: data/market_synthesis.json (TTL 4h).
"""

import json
import os
from datetime import datetime, timedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUTPUT_PATH  = os.path.join(_PROJECT_ROOT, "data", "market_synthesis.json")
_TTL_HOURS    = 4

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 1500


def _is_fresh(path: str = _OUTPUT_PATH) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=_TTL_HOURS)


def _load_context() -> dict:
    """Junta todo lo que el agente necesita ver."""
    ctx = {}

    def _read_json(rel: str, key: str) -> None:
        p = os.path.join(_PROJECT_ROOT, rel)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    ctx[key] = json.load(f)
            except Exception:
                pass

    _read_json("data/news_intel.json",       "news_intel")
    _read_json("data/signal_breakdown.json", "signal")
    _read_json("data/argentina_supply.json", "argentina")
    _read_json("data/brazil_exports.json",   "brazil")
    _read_json("data/china_demand.json",     "china")
    _read_json("data/basis_uruguay.json",    "basis")
    _read_json("data/wasde_official.json",   "wasde")
    _read_json("data/multi_commodity.json",  "multi_commodity")
    _read_json("data/current_contract.json", "contract")

    # Forecast (CSV)
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

    # Accountability reciente
    try:
        from src.trader.accountability import get_accountability_records
        recs = get_accountability_records()
        if isinstance(recs, dict) and recs.get("summary"):
            ctx["accuracy"] = recs["summary"]
    except Exception:
        pass

    return ctx


def _build_prompt(ctx: dict) -> str:
    today = datetime.now().strftime("%d/%m/%Y")

    # Extraer la señal compuesta de forma destacada para que el LLM no la pierda
    sig = ctx.get("signal") or {}
    composite_signal = sig.get("composite_signal", "?")
    composite_raw    = sig.get("composite_raw", "?")
    composite_score  = sig.get("composite_score", "?")

    # Lista de factores con peso > 0 (los que SÍ pesan)
    weighted_factors = []
    informational    = []
    for f in (sig.get("factors") or []):
        line = f"- {f.get('name')}: score={f.get('score')} dir={f.get('direction')} (peso {int(f.get('weight',0)*100)}%)"
        if f.get("weight", 0) > 0:
            weighted_factors.append(line)
        else:
            informational.append(line)

    return f"""Sos el analista jefe de AgroCast PRO, sistema de inteligencia de mercado para productores de soja en Uruguay y Argentina.

Hoy es {today}.

═══════════════════════════════════════════════
SEÑAL COMPUESTA (FUENTE DE VERDAD - usá ESTO para stance):
  composite_signal: {composite_signal}
  composite_raw:    {composite_raw}  (rango -1 a +1)
  composite_score:  {composite_score}/100

Factores que PONDERAN en el composite:
{chr(10).join(weighted_factors) if weighted_factors else "(ninguno)"}

Factores INFORMATIVOS (NO ponderan, sólo referencia visual):
{chr(10).join(informational) if informational else "(ninguno)"}
═══════════════════════════════════════════════

Resto de la inteligencia (datos crudos):
{json.dumps(ctx, indent=2, default=str, ensure_ascii=False)}

Generá un BRIEF EJECUTIVO en español rioplatense (voseo), formato JSON estricto:

{{
  "headline": "una frase de 8-12 palabras que resuma la postura del mercado HOY",
  "stance": "ALCISTA" | "BAJISTA" | "NEUTRAL",
  "conviction": 0-100,
  "key_drivers": [
    {{"driver": "nombre", "direction": "+/-", "why": "explicación corta basada en datos"}}
  ],
  "risks": ["riesgo 1 corto", "riesgo 2 corto"],
  "tactical_recommendation": "qué hacer esta semana en 2-3 oraciones",
  "strategic_view": "perspectiva 30-90 días en 1-2 oraciones",
  "data_gaps": ["qué falta saber para mejorar la convicción"]
}}

REGLAS CRÍTICAS:
- El campo `stance` DEBE coincidir con composite_signal: BUY→ALCISTA, SELL→BAJISTA, HOLD→NEUTRAL. NO inventes una stance distinta porque te parezca.
- PROHIBIDO incluir "Forecast 30d", "Forecast", "Ridge+XGB" o cualquier referencia al pronóstico de 30 días en `key_drivers`. Es un factor INFORMATIVO con peso 0%. Si lo incluís, el brief queda inválido.
- key_drivers SOLO pueden venir de los factores marcados como "PONDERAN" arriba (peso > 0%).
- Si querés mencionar el Forecast 30d, hacelo en `strategic_view` con la advertencia de que satura técnicamente (cap diario ±1% acumulado).
- key_drivers: máximo 4, ordenados por |score|×peso (importancia real en el composite).
- `conviction` (0-100): cercano a |composite_raw|×100 si hay alta consistencia entre factores, más bajo si hay contradicciones.
- tactical_recommendation: concreta, niveles de precio del precio actual (no del forecast saturado).
- NO uses jerga financiera innecesaria. Lector = productor, no trader.
- Si algo falta, decilo en data_gaps.
- Devolvé SOLO el JSON, sin markdown, sin texto extra.
"""


def synthesize(force: bool = False) -> dict | None:
    """
    Genera o devuelve cache del brief de mercado.

    Returns
    -------
    dict con el JSON estructurado, o None si no se pudo generar.
    """
    if not force and _is_fresh():
        try:
            with open(_OUTPUT_PATH, "r", encoding="utf-8") as f:
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

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": _build_prompt(ctx)}],
        )
        text = msg.content[0].text.strip()

        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("` \n")

        result = json.loads(text)
        result["_generated_at"] = datetime.utcnow().isoformat()
        result["_model"]        = MODEL
        result["_input_keys"]   = list(ctx.keys())

        os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
        with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[synthesis] OK stance={result.get('stance')} conv={result.get('conviction')}")

        try:
            from src.intel.llm_accountability import record_today, evaluate_due
            record_today()
            evaluate_due()
        except Exception as _e:
            print(f"[synthesis] accountability skip: {_e}")

        return result

    except Exception as e:
        print(f"[synthesis] error: {e}")
        return None


def load_synthesis() -> dict:
    if not os.path.exists(_OUTPUT_PATH):
        return {}
    try:
        with open(_OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


if __name__ == "__main__":
    out = synthesize(force=True)
    print(json.dumps(out, indent=2, ensure_ascii=False) if out else "FAILED")
