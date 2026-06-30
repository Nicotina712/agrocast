"""
Portfolio-level guards for all robots.
Importado por cada live_runner para:
  1. cluster_cap   — bloquea si un instrumento del mismo cluster ya tiene posición abierta
  2. atr_vol_filter — bloquea si el ATR actual es >1.5σ sobre la media (mercado errático)

Clusters:
  equity  → US500 | USTEC | US30
  energy  → WTI_N6 | BRENT_N6 (y variantes de contrato)
  crypto  → BTCUSD | ETHUSD

Instrumentos sin cluster (XAUUSD, UK100) siempre pasan el cluster_cap.
"""

import os
import numpy as np

# ── Definición de clusters ───────────────────────────────────────────────────

_CLUSTERS = {
    "equity": {"US500", "USTEC", "US30"},
    "energy": {"WTI_N6", "BRENT_N6", "BRENT_Q6", "BRENT_M6", "USOIL", "WTIUSD", "BRTUSD", "UKOIL"},
    "crypto": {"BTCUSD", "ETHUSD"},
    "agri":   {"CORN_N6", "WHEAT_N6", "SUGAR_N6", "COFFEE_N6", "COCOA_N6", "COTTON_N6"},
    "europe_idx": {"UK100", "STOXX50"},   # corr 0.63 — direction-aware (solo bloquea misma direccion)
}


def _get_cluster_name(symbol: str) -> str | None:
    """Devuelve el nombre del cluster de un símbolo, o None si no pertenece a ninguno."""
    s = symbol.upper()
    for name, members in _CLUSTERS.items():
        for m in members:
            if s.startswith(m.upper()) or m.upper().startswith(s[:6]):
                return name
    return None


def cluster_cap_blocked(symbol: str, get_positions_fn,
                         proposed_direction: str | None = None) -> dict | None:
    """
    Llama a get_positions_fn() para obtener TODAS las posiciones abiertas.
    Si algún instrumento del mismo cluster (≠ symbol) tiene posición abierta,
    devuelve un dict con info del bloqueo.
    Si no hay bloqueo, devuelve None.

    proposed_direction: "LONG", "SHORT" o None.
      - None (default): bloquea siempre que haya posición en el cluster (comportamiento original).
      - "LONG"/"SHORT": bloquea SOLO si la posición existente va en la MISMA dirección.
        Permite trades opuestos (spread energy, hedge cripto, etc.).

    Dirección de posición MT5: type=0 → BUY/LONG, type=1 → SELL/SHORT.

    Ejemplo de uso en live_runner (después del LLM, direction-aware):
        if signal["signal"] != "FLAT":
            block = cluster_cap_blocked(SYMBOL, get_positions, signal["signal"])
            if block:
                _log("cluster_cap", block)
                signal["signal"] = "FLAT"
    """
    cluster_name = _get_cluster_name(symbol)
    if cluster_name is None:
        return None  # instrumento sin cluster → siempre libre

    cluster_members = _CLUSTERS[cluster_name]
    my_sym_upper = symbol.upper()

    try:
        all_positions = get_positions_fn() or []
    except Exception as e:
        return None  # si falla el check, no bloqueamos (conservador)

    for pos in all_positions:
        pos_sym = pos.get("symbol", "").upper()
        for member in cluster_members:
            m_upper = member.upper()
            # ¿Esta posición pertenece a un member del cluster?
            if pos_sym.startswith(m_upper[:6]) or m_upper.startswith(pos_sym[:6]):
                # ¿Es del propio símbolo? (ya cubierto por position_guard)
                if pos_sym.startswith(my_sym_upper[:6]) or my_sym_upper.startswith(pos_sym[:6]):
                    continue
                # Es un cluster-mate → verificar dirección si se proporciona
                if proposed_direction is not None:
                    # MT5: type 0 = BUY/LONG, type 1 = SELL/SHORT
                    pos_type = pos.get("type", -1)
                    existing_dir = "LONG" if pos_type == 0 else ("SHORT" if pos_type == 1 else None)
                    if existing_dir is not None and existing_dir != proposed_direction:
                        # Direcciones opuestas → spread legítimo, NO bloquear
                        continue
                # Misma dirección o dirección desconocida → bloquear
                return {
                    "cluster":          cluster_name,
                    "blocking_symbol":  pos_sym,
                    "ticket":           pos.get("ticket"),
                    "blocking_dir":     "LONG" if pos.get("type") == 0 else "SHORT",
                    "proposed_dir":     proposed_direction,
                    "reason":           f"cluster '{cluster_name}' already has {('LONG' if pos.get('type')==0 else 'SHORT')} on {pos_sym}",
                }

    return None  # cluster libre


def atr_vol_filter(bars_df, zscore_threshold: float = 1.5, lookback: int = 40) -> dict | None:
    """
    Calcula ATR de los últimos `lookback` bars y verifica si el ATR actual
    es más de `zscore_threshold` desviaciones estándar sobre la media.

    bars_df: DataFrame con columnas high, low, close (index temporal).
    Devuelve dict con info si se debe filtrar, None si el mercado es normal.

    Ejemplo de uso en live_runner:
        vf = atr_vol_filter(bars)
        if vf:
            _log("vol_filter", vf)
            return {"status": "vol_filter", **vf}
    """
    try:
        import pandas as pd
        h = bars_df["high"]
        lo = bars_df["low"]
        c = bars_df["close"]

        # True Range
        tr = pd.concat([
            h - lo,
            (h - c.shift(1)).abs(),
            (lo - c.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=14, adjust=False).mean()

        if len(atr) < lookback + 5:
            return None  # no enough data

        window = atr.iloc[-lookback:]
        atr_now = float(atr.iloc[-1])
        mean_atr = float(window.mean())
        std_atr = float(window.std())

        if std_atr < 1e-9:
            return None

        zscore = (atr_now - mean_atr) / std_atr

        if zscore > zscore_threshold:
            return {
                "atr_now":    round(atr_now, 5),
                "atr_mean":   round(mean_atr, 5),
                "atr_zscore": round(zscore, 2),
                "threshold":  zscore_threshold,
                "reason":     f"ATR z-score {zscore:.2f} > {zscore_threshold} — high-vol regime, skip",
            }

        return None  # mercado normal

    except Exception:
        return None  # si falla, no bloqueamos


def market_dormancy_check(bars_df, threshold_pct: float = 0.08, n_bars: int = 3) -> dict | None:
    """
    Detecta si el mercado está "dormido" y no vale la pena gastar una call LLM.

    Lógica: calcula el True Range promedio de las últimas `n_bars` barras como
    porcentaje del precio actual. Si está por debajo de `threshold_pct`, el
    mercado no se está moviendo lo suficiente para que el LLM genere señales útiles.

    Objetivo: ahorrar calls LLM en períodos de baja actividad (madrugada, pre-market
    silencioso, holiday thin trading) para tenerlas disponibles en horas activas.

    threshold_pct: 0.08 = 0.08% de precio por barra
      - XAUUSD $4400: 0.08% = $3.52/barra → dormido si rango < $3.52
      - WTI $90:      0.08% = $0.07/barra → dormido si rango < 7 cents
      - BTCUSD $62k:  0.08% = $49.6/barra → dormido si rango < $49
      - UK100 $10300: 0.08% = $8.2/barra  → dormido si rango < 8 pts

    Devuelve dict si debe saltar, None si debe evaluar normalmente.

    Uso en live_runner (después del vol_filter):
        dormant = market_dormancy_check(bars)
        if dormant:
            _log("dormant", dormant)
            return {"status": "dormant", **dormant}
    """
    try:
        import pandas as pd
        import numpy as np

        h  = bars_df["high"].iloc[-n_bars:]
        lo = bars_df["low"].iloc[-n_bars:]
        c  = bars_df["close"]
        c_prev = c.iloc[-n_bars-1:-1]

        # True Range por barra
        tr_vals = pd.concat([
            h.values - lo.values,
            abs(h.values - c_prev.values),
            abs(lo.values - c_prev.values),
        ], axis=1).max(axis=1)

        avg_tr   = float(tr_vals.mean())
        price    = float(c.iloc[-1])
        tr_pct   = avg_tr / price * 100 if price > 0 else 0

        if tr_pct < threshold_pct:
            return {
                "avg_tr_pct":  round(tr_pct, 4),
                "avg_tr_abs":  round(avg_tr, 4),
                "threshold_pct": threshold_pct,
                "n_bars":      n_bars,
                "price":       round(price, 4),
                "reason":      f"Mercado dormido: rango medio {tr_pct:.3f}% < {threshold_pct}% — ahorrando call LLM",
            }

        return None  # mercado activo, evaluar normalmente

    except Exception:
        return None  # en caso de error, no bloquear


# ═══════════════════════════════════════════════════════════════════════════
#  FASE 1 — ML MODEL CONFIRMATION FILTER
# ═══════════════════════════════════════════════════════════════════════════

_ML_CACHE = {}  # {symbol: (model, scaler)}

def ml_confirm(symbol, bars_df, artifacts_dir, proposed_direction, timeframe="15m"):
    """
    Carga el modelo GradientBoosting entrenado y predice dirección sobre las
    features actuales. Devuelve dict con la predicción y si CONFIRMA o CONTRADICE
    la señal del LLM.

    Modelo guardado por retrainer.py en {artifacts_dir}/models/{symbol}_model.pkl
    Target del modelo: 1=long, -1=short, 0=flat (6 barras adelante, umbral 0.1%)

    Devuelve:
      {"confirm": True}  si ML coincide con dirección o es neutral
      {"confirm": False, "ml_pred": X, "reason": ...} si ML contradice
      None si no hay modelo (no bloquea)
    """
    try:
        import joblib
        mdir = os.path.join(artifacts_dir, "models")
        mpath = os.path.join(mdir, f"{symbol}_model.pkl")
        spath = os.path.join(mdir, f"{symbol}_scaler.pkl")
        if not (os.path.exists(mpath) and os.path.exists(spath)):
            return None  # sin modelo → no bloquear

        if symbol not in _ML_CACHE:
            _ML_CACHE[symbol] = (joblib.load(mpath), joblib.load(spath))
        clf, scaler = _ML_CACHE[symbol]

        # Reconstruir features igual que el retrainer
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        # build_intraday_features está en cada carpeta de robot (microstructure.py)
        # El live_runner ya tiene las features; pasamos bars y reconstruimos.
        from microstructure import build_intraday_features
        feats = build_intraday_features(bars_df, timeframe)
        X_cols = [c for c in feats.columns if c not in ("signal","label","target")]
        X = feats[X_cols].fillna(0).replace([np.inf,-np.inf], 0)
        if len(X) == 0:
            return None
        x_last = X.iloc[[-1]].values
        x_scaled = scaler.transform(x_last)
        pred = int(clf.predict(x_scaled)[0])  # 1, -1, 0
        proba = clf.predict_proba(x_scaled)[0]
        conf = float(max(proba))

        pred_dir = {1: "LONG", -1: "SHORT", 0: "FLAT"}[pred]

        # Lógica: ML contradice si predice la dirección OPUESTA con confianza
        contradicts = (
            (proposed_direction == "LONG"  and pred == -1) or
            (proposed_direction == "SHORT" and pred == 1)
        )
        if contradicts and conf >= 0.45:
            return {
                "confirm":   False,
                "ml_pred":   pred_dir,
                "ml_conf":   round(conf, 3),
                "reason":    f"ML predice {pred_dir} (conf {conf:.2f}) contra señal {proposed_direction}",
            }
        return {"confirm": True, "ml_pred": pred_dir, "ml_conf": round(conf, 3)}

    except Exception as e:
        return None  # cualquier error → no bloquear (fail-safe)


# ═══════════════════════════════════════════════════════════════════════════
#  FASE 3 — ADX REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

def adx_regime(bars_df, period=14, trend_threshold=20, n_bars=200):
    """
    Calcula ADX para clasificar el régimen de mercado.
      ADX > trend_threshold → TRENDING (trend-following válido)
      ADX <= trend_threshold → RANGING (mean-reversion, trend-following falla)

    Devuelve dict {"regime": "TRENDING"|"RANGING", "adx": X, "di_plus", "di_minus"}
    """
    try:
        import pandas as pd
        df = bars_df.iloc[-n_bars:].copy()
        h, l, c = df["high"], df["low"], df["close"]

        # True Range
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        # Directional Movement
        up_move   = h.diff()
        down_move = -l.diff()
        plus_dm  = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        adx_now = float(adx.iloc[-1])
        regime = "TRENDING" if adx_now > trend_threshold else "RANGING"
        return {
            "regime":   regime,
            "adx":      round(adx_now, 1),
            "di_plus":  round(float(plus_di.iloc[-1]), 1),
            "di_minus": round(float(minus_di.iloc[-1]), 1),
            "trend_dir": "UP" if plus_di.iloc[-1] > minus_di.iloc[-1] else "DOWN",
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  FASE 3 — ECONOMIC CALENDAR FILTER
# ═══════════════════════════════════════════════════════════════════════════

# Eventos macro recurrentes (CT = America/Chicago). Bloquea entradas ±30min.
# Formato: (weekday 0=Mon, hour_ct, minute_ct, nombre, símbolos afectados)
# 'ALL' = afecta todos los símbolos.
_RECURRING_EVENTS = [
    # NFP — primer viernes del mes 07:30 CT (se chequea aparte por "primer viernes")
    # FOMC days — se cargan de calendar JSON si existe
    (2, 9, 30, "EIA Crude Inventories",  ["WTI_N6","BRENT_N6","BRENT_Q6"]),  # Wed 09:30 CT
    (1, 16, 30, "API Crude (estimate)",  ["WTI_N6","BRENT_N6","BRENT_Q6"]),  # Tue ~16:30 CT
]

def economic_calendar_block(symbol, now_ct, window_mins=30, calendar_path=None):
    """
    Bloquea nuevas entradas cerca de eventos macro de alto impacto.
    now_ct: datetime en hora Central.
    Combina eventos recurrentes + calendario JSON opcional (FOMC, NFP, etc.)

    Devuelve dict si debe bloquear, None si está libre.
    """
    try:
        wd  = now_ct.weekday()
        hm  = now_ct.hour * 60 + now_ct.minute

        # 1. Eventos recurrentes
        for ev_wd, ev_h, ev_m, name, syms in _RECURRING_EVENTS:
            if wd != ev_wd:
                continue
            if symbol not in syms and "ALL" not in syms:
                continue
            ev_min = ev_h * 60 + ev_m
            if abs(hm - ev_min) <= window_mins:
                return {
                    "event":  name,
                    "ev_time_ct": f"{ev_h:02d}:{ev_m:02d}",
                    "reason": f"Evento macro '{name}' a {ev_h:02d}:{ev_m:02d} CT (±{window_mins}min) — no nuevas entradas",
                }

        # 2. NFP: primer viernes del mes 07:30 CT
        if wd == 4 and now_ct.day <= 7:
            nfp_min = 7 * 60 + 30
            if abs(hm - nfp_min) <= window_mins:
                return {"event": "NFP", "ev_time_ct": "07:30",
                        "reason": "NFP (primer viernes) 07:30 CT ±30min — no nuevas entradas"}

        # 3. Calendario JSON opcional (FOMC, CPI, fechas específicas)
        if calendar_path and os.path.exists(calendar_path):
            import json
            with open(calendar_path, encoding="utf-8") as f:
                cal = json.load(f)
            today_str = now_ct.strftime("%Y-%m-%d")
            for ev in cal.get("events", []):
                if ev.get("date") != today_str:
                    continue
                ev_syms = ev.get("symbols", ["ALL"])
                if symbol not in ev_syms and "ALL" not in ev_syms:
                    continue
                try:
                    eh, em = map(int, ev.get("time_ct", "00:00").split(":"))
                    if abs(hm - (eh*60+em)) <= window_mins:
                        return {"event": ev.get("name","macro"), "ev_time_ct": ev.get("time_ct"),
                                "reason": f"Evento '{ev.get('name')}' {ev.get('time_ct')} CT ±{window_mins}min"}
                except Exception:
                    pass

        return None  # sin eventos cercanos
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  FASE 2 — TRAILING STOP + BREAKEVEN (gestión de posición abierta)
# ═══════════════════════════════════════════════════════════════════════════

def manage_open_position(position, atr_value, close_position_fn, modify_sl_fn=None,
                          breakeven_at_r=1.0, trail_at_r=1.5, trail_distance_r=0.5):
    """
    Gestiona una posición abierta: mueve SL a breakeven y aplica trailing stop.

    position: dict con keys ticket, type(0=long,1=short), price_open, price_current, sl, tp
    atr_value: ATR actual del instrumento (para medir R)
    close_position_fn / modify_sl_fn: funciones del mt5_bridge

    Lógica:
      - profit >= breakeven_at_r (1R) → mover SL a entry (breakeven)
      - profit >= trail_at_r    (1.5R) → trail SL a (precio - trail_distance_r×R)

    R = distancia entry→SL original. Devuelve dict con acción tomada, o None.
    """
    try:
        entry = position["price_open"]
        cur   = position["price_current"]
        sl    = position.get("sl", 0)
        # BUG corregido 2026-06-29: get_positions devuelve type como STRING "BUY"/"SELL"
        # (no int 0/1). `== 0` daba False siempre -> todo LONG se trataba como SHORT ->
        # profit_r negativo -> BE@1R/trailing NUNCA disparaba en longs (caso CHINA50 +1.2R
        # sin mover SL -> el gap del finde lo mato). Aceptar ambas representaciones.
        is_long = position["type"] in (0, "BUY", "buy")

        # R inicial = distancia entry→SL, PERO solo si el SL sigue del lado perdedor.
        # BUG corregido 2026-06-10: tras mover SL a breakeven, |entry-SL|≈0 y el R
        # colapsaba → trailing ultra-apretado. Si el SL ya cruzó el entry (post-BE),
        # usar el fallback ATR como unidad de R estable.
        _sl_losing_side = sl and sl > 0 and ((is_long and sl < entry) or (not is_long and sl > entry))
        if _sl_losing_side:
            r_unit = abs(entry - sl)
        else:
            r_unit = atr_value * 1.5 if atr_value else 0
        if r_unit <= 0:
            return None

        # Profit actual en unidades de R
        if is_long:
            profit_r = (cur - entry) / r_unit
        else:
            profit_r = (entry - cur) / r_unit

        new_sl = None
        action = None

        # Trailing (prioritario si profit alto)
        if profit_r >= trail_at_r:
            if is_long:
                candidate = cur - trail_distance_r * r_unit
                if candidate > sl:
                    new_sl = candidate; action = "trail"
            else:
                candidate = cur + trail_distance_r * r_unit
                if sl == 0 or candidate < sl:
                    new_sl = candidate; action = "trail"
        # Breakeven
        elif profit_r >= breakeven_at_r:
            be = entry + (r_unit * 0.05 if is_long else -r_unit * 0.05)  # entry + pequeño buffer
            if is_long and be > sl:
                new_sl = be; action = "breakeven"
            elif not is_long and (sl == 0 or be < sl):
                new_sl = be; action = "breakeven"

        if new_sl and modify_sl_fn:
            ok = modify_sl_fn(position["ticket"], round(new_sl, 5))
            return {
                "action":   action,
                "ticket":   position["ticket"],
                "old_sl":   round(sl, 5),
                "new_sl":   round(new_sl, 5),
                "profit_r": round(profit_r, 2),
                "ok":       ok,
            }
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  SEÑALES MECANICAS (mechanical-primary + LLM-veto)
#  Validadas IS/OOS 2026-06-08:
#   - trend (WTI): EMA20/100 cross + RSI mid + bar dir. OOS PF 1.15-1.31
#   - vwap_reversion (HK50): fade extension de VWAP. OOS PF 1.61
# ═══════════════════════════════════════════════════════════════════════════

def _indicators_for_signal(bars_df, ema_fast=20, ema_slow=100):
    """Calcula EMA/RSI/VWAP/ATR sobre el df de barras (open/high/low/close/volume).
    ema_fast/ema_slow parametrizables: cada robot DEBE pasar su EMA validada
    (p.ej. BTC 9/21, Corn 20/50). Default 20/100."""
    import numpy as np
    import pandas as pd
    df = bars_df.copy()
    c = df["close"]
    df["ema20"]  = c.ewm(span=ema_fast, adjust=False).mean()
    df["ema100"] = c.ewm(span=ema_slow, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - 100/(1 + g/l.replace(0, np.nan))
    tr = pd.concat([df["high"]-df["low"], (df["high"]-c.shift()).abs(), (df["low"]-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    tp = (df["high"]+df["low"]+c)/3
    df["vwap"] = (tp*vol).rolling(96, min_periods=20).sum() / vol.rolling(96, min_periods=20).sum()
    return df


def mechanical_signal(bars_df, mode="trend", sl_atr=2.0, tp_atr=4.5, vwap_th=0.5,
                      ema_fast=20, ema_slow=100):
    """
    Genera señal mecánica + niveles SL/TP. Devuelve dict o None (FLAT).
    mode:
      'trend'          -> EMA_fast>EMA_slow & RSI 45-55 & barra a favor (WTI/BTC/US500/Corn)
      'vwap_reversion' -> fade: LONG si precio < VWAP-th%, SHORT si > VWAP+th% (HK50)
    ema_fast/ema_slow: cada robot pasa su EMA validada (BTC 9/21, Corn 20/50, default 20/100).
    Devuelve: {signal, entry, sl, tp, atr, rr, basis}
    """
    try:
        import numpy as np
        if bars_df is None or len(bars_df) < 110:
            return None
        df = _indicators_for_signal(bars_df, ema_fast=ema_fast, ema_slow=ema_slow)
        r = df.iloc[-1]
        atr = float(r["atr"])
        close = float(r["close"])
        if atr <= 0 or np.isnan(atr):
            return None
        direction = None
        basis = ""

        if mode == "trend":
            if np.isnan(r["ema100"]) or np.isnan(r["rsi"]):
                return None
            bull = r["ema20"] > r["ema100"]; bear = r["ema20"] < r["ema100"]
            rsi_ok = 45 < r["rsi"] < 55
            if bull and rsi_ok and r["close"] > r["open"]:
                direction = "LONG";  basis = f"trend UP EMA20>100 RSI{r['rsi']:.0f}"
            elif bear and rsi_ok and r["close"] < r["open"]:
                direction = "SHORT"; basis = f"trend DOWN EMA20<100 RSI{r['rsi']:.0f}"

        elif mode == "vwap_reversion":
            v = float(r["vwap"])
            if np.isnan(v) or v <= 0:
                return None
            dist = (close - v) / v * 100
            if dist < -vwap_th:
                direction = "LONG";  basis = f"VWAP-rev: {dist:.2f}% bajo VWAP"
            elif dist > vwap_th:
                direction = "SHORT"; basis = f"VWAP-rev: {dist:.2f}% sobre VWAP"

        if direction is None:
            return None

        if direction == "LONG":
            sl = close - sl_atr*atr; tp = close + tp_atr*atr
        else:
            sl = close + sl_atr*atr; tp = close - tp_atr*atr
        return {
            "signal": direction, "entry": round(close, 5),
            "sl": round(sl, 5), "tp": round(tp, 5),
            "atr": round(atr, 5), "rr": round(tp_atr/sl_atr, 2), "basis": basis,
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  RESILIENCIA LLM — fallback mecánico cuando el LLM cae (sin créditos/timeout)
# ═══════════════════════════════════════════════════════════════════════════

def resilient_decision(mech, llm_dir=None, llm_conf="LOW", llm_failed=False):
    """
    Decide la señal final combinando mecánica (primaria) + LLM (veto opcional).
    Diseñado para que el robot SIGA OPERANDO su edge mecánico aunque el LLM caiga.

    mech: dict de mechanical_signal() o None (sin setup mecánico).
    llm_dir: 'LONG'/'SHORT'/'FLAT'/None — dirección que propuso el LLM.
    llm_conf: confianza del LLM ('HIGH' para que el veto cuente).
    llm_failed: True si la llamada LLM falló (crédito/timeout) — entonces NO se vetea.

    Lógica:
      - Sin señal mecánica → FLAT.
      - LLM caído → tomar mecánica SIN veto (resiliencia: el edge mecánico es +EV).
      - LLM OK y propone dirección OPUESTA con conf HIGH → vetar (FLAT).
      - Caso contrario → tomar mecánica.

    Devuelve (decision: 'TAKE'|'FLAT', reason: str). El robot arma el signal dict.
    """
    if not mech:
        return ("FLAT", "sin señal mecánica")
    if llm_failed or llm_dir is None:
        return ("TAKE", f"mecánico {mech.get('signal')} (LLM caído — modo resiliencia)")
    if llm_dir in ("LONG", "SHORT") and llm_dir != mech["signal"] and llm_conf == "HIGH":
        return ("FLAT", f"LLM veto {llm_dir} HIGH vs mecánico {mech['signal']}")
    return ("TAKE", f"mecánico {mech.get('signal')} | LLM={llm_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  STORM BREAKER — bloqueo de entradas trend tras tormenta cross-asset
#  Validado 2026-06-11: entradas 0-12h post-tormenta = lotería negativa
#  (expR -0.03, 59% stops; bloque WTI -0.18, ETH -0.14, CHINA50 -0.10,
#   BRENT -0.00). BTC (+0.19) y UK100 (+0.04) EXENTOS — no aplicarles.
#  Tormenta = >=3 símbolos vigía con barra-shock 15m (rango>4x mediana 400,
#  cuerpo >=50%) dentro de la misma ventana de 1h.
# ═══════════════════════════════════════════════════════════════════════════

_STORM_SENTINELS = ["US500", "USTEC", "XAUUSD", "BTCUSD", "UK100", "WTI_N6"]
_STORM_CACHE_FILE = None  # se resuelve en runtime (artifacts/storm_state.json)
_STORM_TTL_S = 600        # recalcular cada 10 min como mucho


def _storm_cache_path():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, "artifacts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "storm_state.json")


def _scan_storms(lookback_hours=14):
    """Escanea shocks recientes en los vigías y devuelve epoch de la última tormenta (o None)."""
    import MetaTrader5 as mt5
    import numpy as np
    import pandas as pd
    import time as _t
    now = _t.time()
    hits = []   # (epoch, symbol)
    nbars = 400 + int(lookback_hours * 4) + 4
    for s in _STORM_SENTINELS:
        try:
            b = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, nbars)
            if b is None or len(b) < 420:
                continue
            df = pd.DataFrame(b)
            h, lo, c, o = df["high"], df["low"], df["close"], df["open"]
            rng = (h - lo) / c
            med = rng.rolling(400, min_periods=100).median()
            for i in range(400, len(df) - 1):           # solo barras cerradas
                if rng.iloc[i] <= 4 * med.iloc[i]:
                    continue
                if abs(c.iloc[i] - o.iloc[i]) < 0.5 * (h.iloc[i] - lo.iloc[i]):
                    continue
                ep = int(df["time"].iloc[i]) - 3 * 3600   # broker UTC+3 -> UTC
                if now - ep <= lookback_hours * 3600:
                    hits.append((ep, s))
        except Exception:
            continue
    last_storm = None
    for ep, s in hits:
        syms = {s2 for ep2, s2 in hits if abs(ep2 - ep) <= 3600}
        if len(syms) >= 3:
            if last_storm is None or ep > last_storm:
                last_storm = ep
    return last_storm


def storm_breaker_check(window_hours=12.0):
    """
    Devuelve dict si hay que bloquear entradas (tormenta cross-asset hace < window_hours),
    None si está libre. Cachea el scan en artifacts/storm_state.json (TTL 10 min) para
    no repetir el trabajo en cada robot/ciclo.
    """
    import json, os, time as _t
    try:
        path = _storm_cache_path()
        now = _t.time()
        st = {}
        if os.path.exists(path):
            try:
                st = json.loads(open(path, encoding="utf-8").read())
            except Exception:
                st = {}
        if now - st.get("checked_at", 0) > _STORM_TTL_S:
            st = {"checked_at": now, "last_storm": _scan_storms()}
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(st, f)
            except Exception:
                pass
        ls = st.get("last_storm")
        if ls:
            hrs = (now - ls) / 3600
            if hrs <= window_hours:
                from datetime import datetime as _dt
                return {"reason": f"tormenta cross-asset hace {hrs:.1f}h (< {window_hours}h) — régimen post-shock, no entradas trend",
                        "storm_utc": _dt.utcfromtimestamp(ls).isoformat(), "hours_ago": round(hrs, 1)}
        return None
    except Exception:
        return None   # fail-open: nunca bloquear por error propio


# ═══════════════════════════════════════════════════════════════════════════
#  RIESGO DINAMICO — presupuesto de riesgo $ relativo al BALANCE real de MT5
#  (2026-06-13). Cada config calcula MAX_RISK_USD = live_balance * risk_pct,
#  asi el sizing nunca queda desactualizado. Usa BALANCE (no equity: el tamaño
#  no debe fluctuar por P&L flotante). Cache 60s. Fallback al capital fijo si
#  MT5 no responde. Compounding en ambas direcciones (protector en drawdown).
# ═══════════════════════════════════════════════════════════════════════════
_BAL_CACHE = {"bal": None, "ts": 0.0}

def live_risk_usd(fallback_capital: float, risk_pct: float, ttl: float = 60.0) -> float:
    """Devuelve balance_real * risk_pct. Si MT5 no responde -> fallback_capital * risk_pct."""
    import time as _t
    now = _t.time()
    bal = None
    if _BAL_CACHE["bal"] is not None and (now - _BAL_CACHE["ts"]) < ttl:
        bal = _BAL_CACHE["bal"]
    else:
        try:
            import MetaTrader5 as _mt5
            a = _mt5.account_info()
            if a and a.balance and a.balance > 0:
                bal = float(a.balance)
                _BAL_CACHE["bal"] = bal; _BAL_CACHE["ts"] = now
        except Exception:
            bal = None
    if bal is None or bal <= 0:
        bal = fallback_capital
    return round(bal * risk_pct, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  FILTRO DE CONTEXTO CROSS-ASSET (2026-06-14) — confirmado multi-lookback + placebo.
#  Las señales mecánicas rinden mejor si su DIRECCION se alinea con el momentum de un
#  instrumento PROXY economicamente relacionado (que puede NO operarse — solo lectura).
#  Hallazgos validados (Δ expR, placebo no-relacionado ~0): XAU<-EURUSD(+0.78), STOXX50<-DE40,
#  BRENT<-XAGUSD, BTC<-CHINA50, CHINA50<-HK50 (este ultimo ya inline). Proxy mom 20 barras M30.
# ═══════════════════════════════════════════════════════════════════════════
def context_aligned(proxy_sym, trade_dir, lookback=20, stale_hours=3.0):
    """True=alineado(tomar), False=opuesto(skip), None=sin dato/proxy stale(fail-open=tomar).
    trade_dir: 'LONG'/'SHORT' o +1/-1. Compara con el signo del momentum del proxy en M30.
    2026-06-21: FAIL-OPEN si el proxy esta STALE (ultima barra > stale_hours = mercado cerrado).
    Antes vetaba con momentum congelado (ej: BTC 24/7 vs CHINA50 cerrado el finde -> bloqueaba el
    edge de finde cripto validado). Usa BTCUSD (24/7) como reloj de referencia (mismo basis broker)."""
    try:
        import MetaTrader5 as mt5
        b = mt5.copy_rates_from_pos(proxy_sym, mt5.TIMEFRAME_M30, 0, lookback + 5)
        if b is None or len(b) < lookback + 1:
            return None
        # staleness: comparar la ultima barra del proxy contra el reloj live de BTCUSD (24/7)
        ref = mt5.symbol_info_tick("BTCUSD")
        if ref and getattr(ref, "time", 0):
            age_h = (ref.time - int(b["time"][-1])) / 3600.0
            if age_h > stale_hours:
                return None   # proxy cerrado/stale -> no vetar con momentum congelado
        c = b["close"]
        ctx = 1 if c[-1] > c[-(lookback + 1)] else -1
        sd = 1 if (trade_dir == "LONG" or trade_dir == 1) else -1
        return ctx == sd
    except Exception:
        return None
