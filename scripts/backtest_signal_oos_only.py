"""
scripts/backtest_signal_oos_only.py
Backtest HONESTO: solo periodo verdaderamente OOS para el modelo ML.

El modelo XGB se entrenó con 80% de los datos (~hasta 2024).
Solo 2025+ es verdaderamente out-of-sample.
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

HORIZON = 14
COST_BPS = 5
BUY_THRESH = 0.15
SELL_THRESH = -0.15

print("=" * 70)
print("BACKTEST HONESTO: SOLO PERIODO OOS (2025+)")
print("=" * 70)

feat_path = os.path.join(PROJECT_ROOT, "data", "features.csv")
sig_path  = os.path.join(PROJECT_ROOT, "artifacts", "signals.csv")

df = pd.read_csv(feat_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

sig = pd.read_csv(sig_path)
sig["Date"] = pd.to_datetime(sig["Date"])
df = df.merge(sig[["Date", "expected_return", "signal", "confidence"]], on="Date", how="left")

df["ret_fwd"] = df["Soybeans"].shift(-HORIZON) / df["Soybeans"] - 1
df["direction_actual"] = ((df["ret_fwd"] - COST_BPS / 10000) > 0).astype(int)
df = df.dropna(subset=["ret_fwd"]).copy()

# ── Factores (misma lógica que antes) ──────────────────────────────────
def factor_ml(df):
    er = df["expected_return"].fillna(0)
    return np.clip(er * 5, -1, 1)

def factor_technical(df):
    rsi = df["rsi_14"].fillna(50)
    mom5 = df["mom_5d"].fillna(0)
    mom20 = df["mom_20d"].fillna(0)
    price_vs_ma90 = df["price_vs_ma90"].fillna(0) if "price_vs_ma90" in df.columns else 0
    price_vs_ma20 = df["price_vs_ma20"].fillna(0) if "price_vs_ma20" in df.columns else 0
    rsi_score = np.clip(-(rsi - 50) / 20, -1, 1)
    mom_score = np.clip((mom5 + mom20) / 2 * 10, -1, 1)
    ma_score = np.clip(price_vs_ma20 * 5 + price_vs_ma90 * 3, -1, 1)
    return np.clip((rsi_score + mom_score + ma_score) / 3, -1, 1)

def factor_china(df):
    china = df["china_score"].fillna(50) if "china_score" in df.columns else pd.Series(50, index=df.index)
    score = (china - 50) / 50
    if "crush_spread_dev" in df.columns:
        crush_dev = df["crush_spread_dev"].fillna(0)
        score = 0.7 * score + 0.3 * np.clip(crush_dev / 20, -1, 1)
    return np.clip(score, -1, 1)

def factor_wasde(df):
    surprise = df["wasde_surprise_proxy"].fillna(0) if "wasde_surprise_proxy" in df.columns else pd.Series(0, index=df.index)
    bull_bias = df["wasde_bull_bias"].fillna(0) if "wasde_bull_bias" in df.columns else pd.Series(0, index=df.index)
    return np.clip(surprise * 2 + bull_bias * 0.5, -1, 1).fillna(0)

def factor_fundamental(df):
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


# ── SOLO OOS: 2025+ ──────────────────────────────────────────────────
# El modelo XGB se entrena con ~80% de datos (hasta ~mayo 2024),
# más embargo de 18 días. Solo datos desde julio 2024 son OOS.
# Para ser conservador, usamos 2025+.

OOS_START = "2025-01-01"
df_oos = df[df["Date"] >= OOS_START].copy()
print(f"\nPeriodo OOS: {df_oos['Date'].min().date()} → {df_oos['Date'].max().date()}")
print(f"Filas: {len(df_oos)}")
print(f"Target: {df_oos['direction_actual'].mean()*100:.1f}% subidas | {(1-df_oos['direction_actual'].mean())*100:.1f}% bajadas")

# Also test semi-OOS (2024-H2 + 2025+)
SEMI_OOS_START = "2024-07-01"
df_semi = df[df["Date"] >= SEMI_OOS_START].copy()


def evaluate(df_eval, factor_names, weights=None):
    if not factor_names:
        return {"accuracy": 50, "n_trades": 0, "avg_pnl_bps": 0, "sharpe": 0,
                "total_pnl_pct": 0, "buy_pct": 0, "sell_pct": 0, "hold_pct": 100}

    if weights is None:
        w = {f: 1.0 / len(factor_names) for f in factor_names}
    else:
        total_w = sum(weights[f] for f in factor_names)
        w = {f: weights[f] / total_w for f in factor_names}

    composite = sum(w[f] * df_eval[factors[f]["col"]] for f in factor_names)

    signal = pd.Series("HOLD", index=df_eval.index)
    signal[composite > BUY_THRESH] = "BUY"
    signal[composite < SELL_THRESH] = "SELL"

    buy_mask  = signal == "BUY"
    sell_mask = signal == "SELL"
    actual = df_eval["direction_actual"]

    correct_buy  = (buy_mask & (actual == 1)).sum()
    correct_sell = (sell_mask & (actual == 0)).sum()
    n_trades = buy_mask.sum() + sell_mask.sum()
    n_correct = correct_buy + correct_sell
    hit_rate = n_correct / n_trades if n_trades > 0 else 0.5

    pnl = []
    for idx in df_eval.index:
        ret = df_eval.loc[idx, "ret_fwd"]
        if signal[idx] == "BUY":
            pnl.append(ret - COST_BPS / 10000)
        elif signal[idx] == "SELL":
            pnl.append(-ret - COST_BPS / 10000)

    avg_pnl = np.mean(pnl) if pnl else 0
    total_pnl = np.sum(pnl) if pnl else 0
    if pnl and len(pnl) > 5:
        pnl_arr = np.array(pnl)
        tpy = len(pnl_arr) / max(1, (df_eval["Date"].max() - df_eval["Date"].min()).days / 365.25)
        sharpe = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(tpy)) if pnl_arr.std() > 0 else 0
    else:
        sharpe = 0

    return {
        "accuracy": round(hit_rate * 100, 1),
        "n_trades": int(n_trades),
        "avg_pnl_bps": round(avg_pnl * 10000, 1),
        "total_pnl_pct": round(total_pnl * 100, 2),
        "sharpe": round(sharpe, 2),
        "buy_pct": round(buy_mask.mean() * 100, 1),
        "sell_pct": round(sell_mask.mean() * 100, 1),
        "hold_pct": round((~buy_mask & ~sell_mask).mean() * 100, 1),
    }


# ── Comparación directa: configuraciones clave ───────────────────────
configs = {
    "ML Only":                      ["ML"],
    "Technical Only":               ["Technical"],
    "China Only":                   ["China"],
    "WASDE Only":                   ["WASDE"],
    "Fundamental Only":             ["Fundamental"],
    "Sin ML (Tech+China+WASDE+Fund)": ["Technical", "China", "WASDE", "Fundamental"],
    "Con ML (FULL 5 factores)":     ["ML", "Technical", "China", "WASDE", "Fundamental"],
    "China+WASDE":                  ["China", "WASDE"],
    "China+WASDE+Tech":             ["China", "WASDE", "Technical"],
    "ML+Technical":                 ["ML", "Technical"],
    "ML+China+WASDE":               ["ML", "China", "WASDE"],
    "ML+Fundamental":               ["ML", "Fundamental"],
    "China+Fundamental":            ["China", "Fundamental"],
    "Tech+Fundamental":             ["Technical", "Fundamental"],
}

for period_name, df_eval in [("OOS ESTRICTO (2025+)", df_oos), ("SEMI-OOS (Jul 2024+)", df_semi)]:
    print(f"\n\n{'=' * 70}")
    print(f"  {period_name}  |  {len(df_eval)} filas  |  {df_eval['Date'].min().date()} → {df_eval['Date'].max().date()}")
    print(f"{'=' * 70}")
    print(f"  Baseline (random): ~{df_eval['direction_actual'].mean()*100:.1f}% subidas")

    print(f"\n{'Configuración':<35} {'Acc%':>6} {'Trades':>7} {'AvgPnL':>8} {'TotPnL%':>9} {'Sharpe':>7} {'B/S/H%':>15}")
    print("-" * 90)

    for name, combo in configs.items():
        res = evaluate(df_eval, combo)
        bsh = f"{res['buy_pct']:.0f}/{res['sell_pct']:.0f}/{res['hold_pct']:.0f}"
        print(f"{name:<35} {res['accuracy']:>5.1f}% {res['n_trades']:>6d} "
              f"{res['avg_pnl_bps']:>+7.1f} {res['total_pnl_pct']:>+8.2f}% {res['sharpe']:>+6.2f}  {bsh:>15}")


# ── Impacto del ML: comparación pareada ──────────────────────────────
print(f"\n\n{'=' * 70}")
print("IMPACTO DEL ML: COMPARACIÓN PAREADA (OOS 2025+)")
print(f"{'=' * 70}")

pairs = [
    ("China+WASDE",           ["China", "WASDE"],             ["ML", "China", "WASDE"]),
    ("Tech+China+WASDE",      ["Technical", "China", "WASDE"],["ML", "Technical", "China", "WASDE"]),
    ("Tech+China+WASDE+Fund", ["Technical", "China", "WASDE", "Fundamental"],
                              ["ML", "Technical", "China", "WASDE", "Fundamental"]),
    ("Fundamental",           ["Fundamental"],                ["ML", "Fundamental"]),
    ("Technical",             ["Technical"],                  ["ML", "Technical"]),
    ("China",                 ["China"],                      ["ML", "China"]),
]

print(f"\n{'Base':<30} {'Sin ML':>8} {'Con ML':>8} {'Delta':>8} {'Veredicto':>12}")
print("-" * 75)

for label, without, with_ml in pairs:
    r_no = evaluate(df_oos, without)
    r_ml = evaluate(df_oos, with_ml)
    delta = r_ml["accuracy"] - r_no["accuracy"]
    verdict = "MEJORA" if delta > 1 else ("DEGRADA" if delta < -1 else "NEUTRAL")
    print(f"{label:<30} {r_no['accuracy']:>6.1f}% → {r_ml['accuracy']:>6.1f}% {delta:>+6.1f}pp   {verdict:>10}")


# ── Todas las combinaciones OOS ──────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("RANKING COMPLETO: TODAS LAS COMBINACIONES (OOS 2025+)")
print(f"{'=' * 70}")

factor_names = list(factors.keys())
results = []

for r in range(1, len(factor_names) + 1):
    for combo in itertools.combinations(factor_names, r):
        combo_list = list(combo)
        label = " + ".join(combo_list)
        res = evaluate(df_oos, combo_list)
        results.append({"combination": label, "has_ml": "ML" in combo_list, **res})

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)

print(f"\n{'#':>3} {'Combinación':<45} {'ML':>3} {'Acc%':>6} {'Trades':>7} {'AvgPnL':>8} {'Sharpe':>7}")
print("-" * 85)

for i, (_, row) in enumerate(results_df.iterrows(), 1):
    ml_flag = "SI" if row["has_ml"] else "  "
    print(f"{i:>3}. {row['combination']:<45} {ml_flag:>3} {row['accuracy']:>5.1f}% "
          f"{row['n_trades']:>6d} {row['avg_pnl_bps']:>+7.1f} {row['sharpe']:>+6.2f}")


# ── Monthly breakdown ────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("DESEMPEÑO MENSUAL (OOS): ML vs Sin ML vs Baseline")
print(f"{'=' * 70}")

df_oos_m = df_oos.copy()
df_oos_m["ym"] = df_oos_m["Date"].dt.to_period("M")

print(f"\n{'Mes':>8} {'Actual%':>8} {'ML_Only':>8} {'SinML':>8} {'FULL':>8} {'Best':>12}")
print("-" * 55)

for ym, grp in df_oos_m.groupby("ym"):
    if len(grp) < 10:
        continue
    actual_pct = grp["direction_actual"].mean() * 100
    r_ml   = evaluate(grp, ["ML"])
    r_noml = evaluate(grp, ["Technical", "China", "WASDE", "Fundamental"])
    r_full = evaluate(grp, ["ML", "Technical", "China", "WASDE", "Fundamental"])

    accs = {"ML_Only": r_ml["accuracy"], "SinML": r_noml["accuracy"], "FULL": r_full["accuracy"]}
    best = max(accs, key=accs.get)

    print(f"{str(ym):>8} {actual_pct:>7.1f}% {r_ml['accuracy']:>7.1f}% {r_noml['accuracy']:>7.1f}% "
          f"{r_full['accuracy']:>7.1f}% {best:>12}")


# ── Conclusión final ─────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("CONCLUSIÓN FINAL (basada SOLO en datos OOS)")
print(f"{'=' * 70}")

r_ml_only = evaluate(df_oos, ["ML"])
r_noml    = evaluate(df_oos, ["Technical", "China", "WASDE", "Fundamental"])
r_full    = evaluate(df_oos, ["ML", "Technical", "China", "WASDE", "Fundamental"])
r_best    = results_df.iloc[0]

# Count: in how many combos does ML improve vs degrade
ml_impact = []
for _, row in results_df[results_df["has_ml"]].iterrows():
    combo_without = row["combination"].replace("ML + ", "").replace(" + ML", "").replace("ML", "").strip()
    if combo_without:
        match = results_df[results_df["combination"] == combo_without]
        if not match.empty:
            delta = row["accuracy"] - match.iloc[0]["accuracy"]
            ml_impact.append(delta)

n_improve = sum(1 for d in ml_impact if d > 0.5)
n_degrade = sum(1 for d in ml_impact if d < -0.5)
n_neutral = len(ml_impact) - n_improve - n_degrade

print(f"""
PERIODO: {df_oos['Date'].min().date()} → {df_oos['Date'].max().date()} ({len(df_oos)} observaciones)
BASELINE (random): ~{df_oos['direction_actual'].mean()*100:.1f}% subidas

1. ML SOLO:
   Accuracy: {r_ml_only['accuracy']:.1f}% | Sharpe: {r_ml_only['sharpe']:.2f} | Trades: {r_ml_only['n_trades']}
   vs baseline {df_oos['direction_actual'].mean()*100:.1f}%: {'MEJOR' if r_ml_only['accuracy'] > df_oos['direction_actual'].mean()*100 + 2 else 'PEOR' if r_ml_only['accuracy'] < df_oos['direction_actual'].mean()*100 - 2 else 'SIMILAR'}

2. SIN ML (4 factores fundamentales/tecnico):
   Accuracy: {r_noml['accuracy']:.1f}% | Sharpe: {r_noml['sharpe']:.2f} | Trades: {r_noml['n_trades']}

3. FULL (5 factores con ML):
   Accuracy: {r_full['accuracy']:.1f}% | Sharpe: {r_full['sharpe']:.2f} | Trades: {r_full['n_trades']}

4. MEJOR COMBINACION OOS:
   {r_best['combination']}
   Accuracy: {r_best['accuracy']:.1f}% | Sharpe: {r_best['sharpe']:.2f}

5. IMPACTO DEL ML (en {len(ml_impact)} comparaciones pareadas):
   Mejora: {n_improve}/{len(ml_impact)} | Degrada: {n_degrade}/{len(ml_impact)} | Neutral: {n_neutral}/{len(ml_impact)}

6. VEREDICTO:""")

if n_degrade > n_improve:
    print("   >>> EL ML DEGRADA el sistema en la mayoría de combinaciones OOS.")
    print("   >>> Recomendación: REDUCIR o ELIMINAR el peso del ML.")
elif n_improve > n_degrade:
    print("   >>> EL ML MEJORA el sistema en la mayoría de combinaciones OOS.")
    print("   >>> Recomendación: MANTENER el ML pero ajustar peso según contexto.")
else:
    print("   >>> EL ML es NEUTRAL en promedio.")
    print("   >>> Recomendación: Mantener con peso bajo o condicionar a régimen.")

print(f"""
CAVEAT IMPORTANTE:
- El periodo OOS es corto ({len(df_oos)} observaciones, ~{(df_oos['Date'].max() - df_oos['Date'].min()).days} días).
- El backtest anterior (2023-2026) mostraba ML con 71-85% accuracy,
  pero eso incluía datos in-sample del modelo (overfitting).
- La degradación de 2023 (95%) a 2026 (47%) es señal clara de overfitting.
- IE (35% del peso actual) NO pudo evaluarse históricamente.
""")

# Save
out_path = os.path.join(PROJECT_ROOT, "artifacts_eval", "signal_combination_backtest_oos.csv")
results_df.to_csv(out_path, index=False)
print(f"Guardado en: {out_path}")
