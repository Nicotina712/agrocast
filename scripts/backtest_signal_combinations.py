"""
scripts/backtest_signal_combinations.py
Backtest combinatorio de fuentes de señal AgroCast.

Reconstruye históricamente cada factor de señal desde features.csv y signals.csv,
luego testea TODAS las combinaciones posibles para determinar cuál contribuye
y cuál degrada el sistema.

Factores testeados:
  - ML (XGBoost 14d): probabilidad del modelo, desde signals.csv
  - Technical: RSI, momentum, volatilidad → score técnico
  - China: china_score desde features.csv
  - WASDE: wasde_surprise_proxy desde features.csv
  - Fundamental Composite: news_sentiment + macro + weather combinados

IE no puede testearse históricamente (solo 2 días de datos).
"""

import sys
import os
import itertools

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Parámetros ────────────────────────────────────────────────────────
HORIZON = 14          # días forward para evaluar dirección
COST_BPS = 5          # 5 bps round-trip cost
BUY_THRESH = 0.15     # composite_raw > 0.15 → BUY
SELL_THRESH = -0.15   # composite_raw < -0.15 → SELL

# ── Cargar datos ──────────────────────────────────────────────────────
print("=" * 70)
print("BACKTEST COMBINATORIO DE SEÑALES AGROCAST")
print("=" * 70)

feat_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
sig_path  = os.path.join(PROJECT_ROOT, "artifacts", "signals.csv")

df = pd.read_csv(feat_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ML signals
sig = pd.read_csv(sig_path)
sig["Date"] = pd.to_datetime(sig["Date"])
sig = sig.sort_values("Date").reset_index(drop=True)

# Merge
df = df.merge(sig[["Date", "expected_return", "signal", "confidence"]], on="Date", how="left")

# Target: dirección real del precio a HORIZON días (net de costos)
df["ret_fwd"] = df["Soybeans"].shift(-HORIZON) / df["Soybeans"] - 1
df["direction_actual"] = ((df["ret_fwd"] - COST_BPS / 10000) > 0).astype(int)

# Drop rows sin target
df = df.dropna(subset=["ret_fwd"]).copy()
print(f"\nDatos: {len(df)} filas ({df['Date'].min().date()} → {df['Date'].max().date()})")
print(f"Horizonte: {HORIZON}d | Costo: {COST_BPS} bps")
print(f"Target: {df['direction_actual'].mean()*100:.1f}% subidas | {(1-df['direction_actual'].mean())*100:.1f}% bajadas")


# ── Reconstruir señales históricas de cada factor ─────────────────────

def _score_to_signal(score_series: pd.Series) -> pd.Series:
    """Convierte score -1..1 a BUY/HOLD/SELL."""
    return pd.cut(score_series, bins=[-np.inf, SELL_THRESH, BUY_THRESH, np.inf],
                  labels=["SELL", "HOLD", "BUY"])


def factor_ml(df: pd.DataFrame) -> pd.Series:
    """Factor ML: expected_return escalado a -1..1."""
    er = df["expected_return"].fillna(0)
    return np.clip(er * 5, -1, 1)


def factor_technical(df: pd.DataFrame) -> pd.Series:
    """Factor técnico: RSI + momentum + volatility."""
    rsi = df["rsi_14"].fillna(50)
    mom5 = df["mom_5d"].fillna(0)
    mom20 = df["mom_20d"].fillna(0)
    price_vs_ma90 = df["price_vs_ma90"].fillna(0) if "price_vs_ma90" in df.columns else 0
    price_vs_ma20 = df["price_vs_ma20"].fillna(0) if "price_vs_ma20" in df.columns else 0

    # RSI: 30→+1 (oversold=bullish), 70→-1 (overbought=bearish), 50→0
    rsi_score = np.clip(-(rsi - 50) / 20, -1, 1)

    # Momentum: positivo = bullish
    mom_score = np.clip((mom5 + mom20) / 2 * 10, -1, 1)

    # MA position: above MA = bullish
    ma_score = np.clip(price_vs_ma20 * 5 + price_vs_ma90 * 3, -1, 1)

    # Composite técnico (equal weight)
    tech = (rsi_score + mom_score + ma_score) / 3
    return np.clip(tech, -1, 1)


def factor_china(df: pd.DataFrame) -> pd.Series:
    """Factor China: china_score + crush spread."""
    china = df["china_score"].fillna(50) if "china_score" in df.columns else pd.Series(50, index=df.index)
    score = (china - 50) / 50  # 0-100 → -1..+1

    if "crush_spread_dev" in df.columns:
        crush_dev = df["crush_spread_dev"].fillna(0)
        crush_score = np.clip(crush_dev / 20, -1, 1)
        score = 0.7 * score + 0.3 * crush_score

    return np.clip(score, -1, 1)


def factor_wasde(df: pd.DataFrame) -> pd.Series:
    """Factor WASDE: surprise proxy + bull bias."""
    surprise = df["wasde_surprise_proxy"].fillna(0) if "wasde_surprise_proxy" in df.columns else pd.Series(0, index=df.index)
    bull_bias = df["wasde_bull_bias"].fillna(0) if "wasde_bull_bias" in df.columns else pd.Series(0, index=df.index)

    score = np.clip(surprise * 2 + bull_bias * 0.5, -1, 1)
    return score.fillna(0)


def factor_fundamental(df: pd.DataFrame) -> pd.Series:
    """Factor fundamental compuesto: news + weather + macro + COT."""
    components = []

    if "news_sentiment" in df.columns:
        components.append(np.clip(df["news_sentiment"].fillna(0) * 2, -1, 1))
    if "weather_score" in df.columns:
        components.append(np.clip((df["weather_score"].fillna(50) - 50) / 50, -1, 1))
    if "cot_index" in df.columns:
        components.append(np.clip((df["cot_index"].fillna(50) - 50) / 50, -1, 1))
    if "intel_composite" in df.columns:
        components.append(np.clip(df["intel_composite"].fillna(0) * 2, -1, 1))

    if components:
        return pd.concat(components, axis=1).mean(axis=1).clip(-1, 1)
    return pd.Series(0, index=df.index)


# ── Calcular todos los factores ──────────────────────────────────────
print("\n── Reconstruyendo señales históricas ──")

df["f_ml"]          = factor_ml(df)
df["f_technical"]   = factor_technical(df)
df["f_china"]       = factor_china(df)
df["f_wasde"]       = factor_wasde(df)
df["f_fundamental"] = factor_fundamental(df)

factors = {
    "ML":          {"col": "f_ml",          "default_w": 0.15},
    "Technical":   {"col": "f_technical",   "default_w": 0.15},
    "China":       {"col": "f_china",       "default_w": 0.20},
    "WASDE":       {"col": "f_wasde",       "default_w": 0.15},
    "Fundamental": {"col": "f_fundamental", "default_w": 0.15},
}

for name, info in factors.items():
    s = df[info["col"]]
    print(f"  {name:15s}: mean={s.mean():+.3f} std={s.std():.3f} "
          f"min={s.min():+.3f} max={s.max():+.3f}")


# ── Split temporal para evaluación OOS ────────────────────────────────
# Usamos 70% train / 30% test (los últimos ~3 años son OOS)
split_idx = int(len(df) * 0.70)
embargo = HORIZON + 5
df_train = df.iloc[:split_idx].copy()
df_test  = df.iloc[split_idx + embargo:].copy()

print(f"\n── Split temporal ──")
print(f"  Train: {len(df_train)} filas ({df_train['Date'].min().date()} → {df_train['Date'].max().date()})")
print(f"  Test:  {len(df_test)} filas ({df_test['Date'].min().date()} → {df_test['Date'].max().date()})")
print(f"  Embargo: {embargo} días")


# ── Función de evaluación ─────────────────────────────────────────────

def evaluate_combination(df_eval: pd.DataFrame, factor_names: list,
                         weights: dict = None) -> dict:
    """
    Evalúa una combinación de factores.

    Si weights=None, usa equal-weight entre los factores seleccionados.
    """
    if not factor_names:
        return {"accuracy": 0.5, "n_trades": 0, "hit_rate": 0.5}

    cols = [factors[f]["col"] for f in factor_names]

    if weights is None:
        # Equal weight
        w = {f: 1.0 / len(factor_names) for f in factor_names}
    else:
        total_w = sum(weights[f] for f in factor_names)
        w = {f: weights[f] / total_w for f in factor_names}

    # Composite score
    composite = pd.Series(0.0, index=df_eval.index)
    for f in factor_names:
        composite += w[f] * df_eval[factors[f]["col"]]

    # Signal
    signal = pd.Series("HOLD", index=df_eval.index)
    signal[composite > BUY_THRESH] = "BUY"
    signal[composite < SELL_THRESH] = "SELL"

    actual = df_eval["direction_actual"]

    # Accuracy: BUY when up, SELL when down, HOLD neutral
    buy_mask  = signal == "BUY"
    sell_mask = signal == "SELL"
    hold_mask = signal == "HOLD"

    correct_buy  = (buy_mask & (actual == 1)).sum()
    correct_sell = (sell_mask & (actual == 0)).sum()
    wrong_buy    = (buy_mask & (actual == 0)).sum()
    wrong_sell   = (sell_mask & (actual == 1)).sum()

    n_trades = buy_mask.sum() + sell_mask.sum()
    n_correct = correct_buy + correct_sell

    # Direction accuracy (solo trades, no HOLD)
    if n_trades > 0:
        hit_rate = n_correct / n_trades
    else:
        hit_rate = 0.5

    # P&L simulado (retorno promedio por trade)
    pnl_trades = []
    for idx in df_eval.index:
        sig_val = signal[idx]
        ret = df_eval.loc[idx, "ret_fwd"]
        if sig_val == "BUY":
            pnl_trades.append(ret - COST_BPS / 10000)
        elif sig_val == "SELL":
            pnl_trades.append(-ret - COST_BPS / 10000)

    avg_pnl = np.mean(pnl_trades) if pnl_trades else 0
    total_pnl = np.sum(pnl_trades) if pnl_trades else 0

    # Sharpe (anualizado)
    if pnl_trades and len(pnl_trades) > 5:
        pnl_arr = np.array(pnl_trades)
        trades_per_year = len(pnl_arr) / ((df_eval["Date"].max() - df_eval["Date"].min()).days / 365.25)
        sharpe = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(trades_per_year)) if pnl_arr.std() > 0 else 0
    else:
        sharpe = 0

    # Signal distribution
    buy_pct  = buy_mask.mean() * 100
    sell_pct = sell_mask.mean() * 100
    hold_pct = hold_mask.mean() * 100

    return {
        "accuracy":    round(hit_rate * 100, 1),
        "n_trades":    int(n_trades),
        "n_buys":      int(buy_mask.sum()),
        "n_sells":     int(sell_mask.sum()),
        "avg_pnl_bps": round(avg_pnl * 10000, 1),
        "total_pnl_pct": round(total_pnl * 100, 2),
        "sharpe":      round(sharpe, 2),
        "buy_pct":     round(buy_pct, 1),
        "sell_pct":    round(sell_pct, 1),
        "hold_pct":    round(hold_pct, 1),
    }


# ── Testear TODAS las combinaciones ──────────────────────────────────
print("\n" + "=" * 70)
print("RESULTADOS: TODAS LAS COMBINACIONES (Out-of-Sample)")
print("=" * 70)

factor_names = list(factors.keys())
results = []

# Test cada combinación posible (1 a 5 factores)
for r in range(1, len(factor_names) + 1):
    for combo in itertools.combinations(factor_names, r):
        combo_list = list(combo)
        label = " + ".join(combo_list)

        # Equal-weight
        res_ew = evaluate_combination(df_test, combo_list)

        # Default weights (production weights)
        default_w = {f: factors[f]["default_w"] for f in combo_list}
        res_dw = evaluate_combination(df_test, combo_list, default_w)

        results.append({
            "combination": label,
            "n_factors":   r,
            "has_ml":      "ML" in combo_list,
            # Equal weight results
            "ew_accuracy":   res_ew["accuracy"],
            "ew_trades":     res_ew["n_trades"],
            "ew_avg_pnl":    res_ew["avg_pnl_bps"],
            "ew_total_pnl":  res_ew["total_pnl_pct"],
            "ew_sharpe":     res_ew["sharpe"],
            "ew_buy_pct":    res_ew["buy_pct"],
            "ew_sell_pct":   res_ew["sell_pct"],
            "ew_hold_pct":   res_ew["hold_pct"],
            # Default weight results
            "dw_accuracy":   res_dw["accuracy"],
            "dw_trades":     res_dw["n_trades"],
            "dw_avg_pnl":    res_dw["avg_pnl_bps"],
            "dw_total_pnl":  res_dw["total_pnl_pct"],
            "dw_sharpe":     res_dw["sharpe"],
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ew_accuracy", ascending=False)

# ── Mostrar tabla principal ───────────────────────────────────────────
print(f"\n{'Combinación':<45} {'Acc%':>6} {'Trades':>7} {'AvgPnL':>8} {'TotPnL%':>9} {'Sharpe':>7} {'B/S/H%':>15}")
print("-" * 100)

for _, row in results_df.iterrows():
    bsh = f"{row['ew_buy_pct']:.0f}/{row['ew_sell_pct']:.0f}/{row['ew_hold_pct']:.0f}"
    ml_flag = " [+ML]" if row["has_ml"] else "      "
    print(f"{row['combination']:<45}{ml_flag} {row['ew_accuracy']:>5.1f}% {row['ew_trades']:>6d} "
          f"{row['ew_avg_pnl']:>+7.1f} {row['ew_total_pnl']:>+8.2f}% {row['ew_sharpe']:>+6.2f}  {bsh:>15}")


# ── Análisis ML vs no-ML ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("ANÁLISIS: IMPACTO DEL ML EN CADA COMBINACIÓN")
print("=" * 70)

# Para cada combinación sin ML, comparar con la misma + ML
no_ml_combos = results_df[~results_df["has_ml"]].copy()
ml_combos    = results_df[results_df["has_ml"]].copy()

print(f"\n{'Base (sin ML)':<40} {'Acc':>6} → {'Con ML':<40} {'Acc':>6} {'Delta':>7}")
print("-" * 110)

for _, row_no in no_ml_combos.iterrows():
    base_factors = set(row_no["combination"].split(" + "))
    # Find the same combo + ML
    for _, row_ml in ml_combos.iterrows():
        ml_factors = set(row_ml["combination"].split(" + "))
        if ml_factors == base_factors | {"ML"}:
            delta = row_ml["ew_accuracy"] - row_no["ew_accuracy"]
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"{row_no['combination']:<40} {row_no['ew_accuracy']:>5.1f}% → "
                  f"{row_ml['combination']:<40} {row_ml['ew_accuracy']:>5.1f}% {arrow}{abs(delta):>+5.1f}pp")
            break


# ── Comparación por periodos (régimen) ────────────────────────────────
print("\n\n" + "=" * 70)
print("ANÁLISIS POR PERIODOS (OOS)")
print("=" * 70)

# Dividir test en sub-periodos
test_years = df_test.groupby(df_test["Date"].dt.year)
best_combos = results_df.head(5)["combination"].tolist()

# Test las top-5 combinaciones por año
print(f"\n{'Año':>6}", end="")
for combo in best_combos:
    print(f" | {combo[:25]:>25}", end="")
print()
print("-" * (6 + 28 * len(best_combos)))

for year, grp in test_years:
    print(f"{year:>6}", end="")
    for combo in best_combos:
        combo_factors = combo.split(" + ")
        res = evaluate_combination(grp, combo_factors)
        print(f" | {res['accuracy']:>5.1f}% ({res['n_trades']:>4}tr)", end="")
    print()


# ── Análisis especial: ML solo vs todo menos ML ──────────────────────
print("\n\n" + "=" * 70)
print("COMPARACIÓN DIRECTA CLAVE")
print("=" * 70)

configs = {
    "ML Only":                   ["ML"],
    "Technical Only":            ["Technical"],
    "China Only":                ["China"],
    "WASDE Only":                ["WASDE"],
    "Fundamental Only":          ["Fundamental"],
    "Sin ML (Tech+China+WASDE+Fund)": ["Technical", "China", "WASDE", "Fundamental"],
    "Con ML (FULL)":             ["ML", "Technical", "China", "WASDE", "Fundamental"],
    "China+WASDE (fund core)":   ["China", "WASDE"],
    "China+WASDE+Tech":          ["China", "WASDE", "Technical"],
    "ML+Technical":              ["ML", "Technical"],
}

print(f"\n{'Configuración':<35} {'Acc%':>6} {'Trades':>7} {'AvgPnL':>8} {'TotPnL%':>9} {'Sharpe':>7}")
print("-" * 75)

for name, combo in configs.items():
    res = evaluate_combination(df_test, combo)
    print(f"{name:<35} {res['accuracy']:>5.1f}% {res['n_trades']:>6d} "
          f"{res['avg_pnl_bps']:>+7.1f} {res['total_pnl_pct']:>+8.2f}% {res['sharpe']:>+6.2f}")


# ── Volatility regime analysis ────────────────────────────────────────
print("\n\n" + "=" * 70)
print("IMPACTO ML POR RÉGIMEN DE VOLATILIDAD")
print("=" * 70)

if "vol_30d" in df_test.columns:
    vol_median = df_test["vol_30d"].median()
    low_vol  = df_test[df_test["vol_30d"] <= vol_median]
    high_vol = df_test[df_test["vol_30d"] > vol_median]

    for regime, grp in [("BAJA volatilidad", low_vol), ("ALTA volatilidad", high_vol)]:
        print(f"\n  {regime} ({len(grp)} filas):")
        for name, combo in [("Sin ML", ["Technical", "China", "WASDE", "Fundamental"]),
                            ("Con ML", ["ML", "Technical", "China", "WASDE", "Fundamental"])]:
            res = evaluate_combination(grp, combo)
            print(f"    {name:<20} Acc={res['accuracy']:>5.1f}% | Trades={res['n_trades']:>5} | "
                  f"AvgPnL={res['avg_pnl_bps']:>+6.1f}bps | Sharpe={res['sharpe']:>+5.2f}")


# ── Conclusión automática ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("CONCLUSIÓN")
print("=" * 70)

# Get full-no-ml and full configs
full_noml = evaluate_combination(df_test, ["Technical", "China", "WASDE", "Fundamental"])
full_ml   = evaluate_combination(df_test, ["ML", "Technical", "China", "WASDE", "Fundamental"])
ml_only   = evaluate_combination(df_test, ["ML"])

delta_acc = full_ml["ew_accuracy" if "ew_accuracy" in full_ml else "accuracy"] - full_noml["accuracy"]
delta_pnl = full_ml["avg_pnl_bps"] - full_noml["avg_pnl_bps"]

# Best overall combination
best_row = results_df.iloc[0]
best_combo = best_row["combination"]
best_acc   = best_row["ew_accuracy"]

print(f"""
1. MEJOR COMBINACIÓN (OOS): {best_combo}
   Accuracy: {best_acc:.1f}% | Sharpe: {best_row['ew_sharpe']:.2f}

2. IMPACTO DEL ML:
   - Sin ML: Acc={full_noml['accuracy']:.1f}% | Sharpe={full_noml['sharpe']:.2f}
   - Con ML: Acc={full_ml['accuracy']:.1f}% | Sharpe={full_ml['sharpe']:.2f}
   - Delta:  {delta_acc:+.1f}pp accuracy | {delta_pnl:+.1f}bps avg PnL

3. ML SOLO: Acc={ml_only['accuracy']:.1f}% | Sharpe={ml_only['sharpe']:.2f}
   → {'CONTRIBUYE' if delta_acc > 0.5 else 'NEUTRAL' if abs(delta_acc) <= 0.5 else 'DEGRADA'} el sistema

4. NOTA: IE (Intelligence Engine) no pudo testearse históricamente
   (solo 2 días de datos). IE es la fuente primaria (35% peso).
   Este análisis evalúa los factores complementarios.
""")

# Guardar resultados
out_dir = os.path.join(PROJECT_ROOT, "artifacts_eval")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "signal_combination_backtest.csv")
results_df.to_csv(out_path, index=False)
print(f"Resultados guardados en: {out_path}")
