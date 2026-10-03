"""
Auditoría del clasificador de retornos 14d (research, no producción).

Walk-forward honesto:
  - Re-entrena cada trimestre desde 2019 con ventana rolling de 5 años.
  - Embargo de 18 días entre fin de train y comienzo de test (target = ret 14d fwd).
  - Sin early stopping ni calibración sobre el test (eso inflaba el 72.5% interno).
  - Métricas solo sobre fechas que el modelo NO vio.

Responde:
  1. ¿BUY y SELL tienen edge fuera de muestra? (por separado, por mitades)
  2. ¿Qué grupos de variables suman / restan? (ablation drop-one-group y only-group)
  3. Cobertura real de cada variable (las de historia corta se rellenan con 0).

Uso:  python -m scripts.research_model_audit [coverage|side|ablation|all]
Salida: artifacts/research/model_audit_<parte>.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.train_returns import NON_FEATURE_COLS, _select_return_features  # noqa: E402

OUT_DIR = os.path.join(ROOT, "artifacts", "research")
TARGET = "ret_14d_fwd"
COST = 0.0005          # round-trip aprox (mismo fallback que train_returns)
EMBARGO = 18
WINDOW_Y = 5
BUY_T, SELL_T = 0.58, 0.42
START_TEST = "2019-01-01"

GROUPS = {
    "precio_tecnico":  lambda c: c.startswith(("Soybeans_", "mom_", "rsi_", "vol_", "price_vs", "ma90", "ma20"))
                                 or c in ("is_spike_5d", "spike_with_oil_codriver", "spike_without_oil",
                                          "spike_magnitude", "is_drop_5d", "extreme_move_5d"),
    "cross_asset":     lambda c: c.startswith(("Maize", "Wheat", "Oil", "Dollar", "SoybeanMeal", "SoybeanOil",
                                               "soy_corn", "soy_oil", "crush_")),
    "descomposicion":  lambda c: c.startswith(("decomp_", "ms_")),
    "estacionalidad":  lambda c: c in ("month_sin", "month_cos", "quarter"),
    "basis_local":     lambda c: c.startswith(("basis_", "local_price", "fx_basis")),
    "fx_externos":     lambda c: c in ("brl_usd", "cny_usd", "ext_brazil_signal", "ext_china_signal"),
    "inspecciones":    lambda c: c.startswith("insp_"),
    "cot":             lambda c: c.startswith("cot_"),
    "google_trends":   lambda c: c.startswith("trends_"),
    "noticias":        lambda c: c.startswith(("news_", "intel_")) or c in ("china_score", "weather_score", "macro_score"),
    "wasde":           lambda c: c.startswith("wasde_"),
    "curva":           lambda c: c.startswith(("front_next", "roll_yield", "curve_")) or c == "days_between",
    "eventos":         lambda c: c in ("days_to_next_event", "days_since_event", "in_event_window", "post_event_drift",
                                       "event_cost_mult", "is_roll_window", "days_to_wasde", "days_since_wasde",
                                       "days_to_acreage", "days_to_stocks", "event_intensity"),
    "cme_oi_vol":      lambda c: c.startswith("cme_"),
    "opciones_iv":     lambda c: c.startswith("cvol_"),
    "satelite":        lambda c: c.startswith("sat_"),
    "clima_enso":      lambda c: c.startswith(("enso_", "drought_", "clim_fwd")),
    "crop_progress":   lambda c: c.startswith("cp_"),
}


def load():
    df = pd.read_csv(os.path.join(ROOT, "data", "features.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=[TARGET]).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    df["y"] = ((df[TARGET] - COST) > 0).astype(int)
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS | {"y", "direction"}]
    return df, feats


def group_of(col):
    for g, f in GROUPS.items():
        if f(col):
            return g
    return "otros"


def coverage(df, feats):
    recent = df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=WINDOW_Y)]
    rows = []
    for c in feats:
        s = recent[c]
        real = s.notna() & (s != 0)
        first = df.loc[df[c].notna() & (df[c] != 0), "Date"].min()
        rows.append({"feature": c, "group": group_of(c),
                     "pct_real_5y": round(real.mean() * 100, 1),
                     "n_unique_5y": int(s.nunique()),
                     "desde": str(first.date()) if pd.notna(first) else None})
    return rows


def walk_forward(df, feats, select=True, max_feats=40, seed=42):
    """Devuelve DataFrame con Date, p (prob OOS), ret, y para el período de test."""
    test_starts = pd.date_range(START_TEST, df["Date"].max(), freq="QS")
    out = []
    for i, ts in enumerate(test_starts):
        te = test_starts[i + 1] if i + 1 < len(test_starts) else df["Date"].max() + pd.Timedelta(days=1)
        train_end = ts - pd.Timedelta(days=EMBARGO)
        tr = df[(df["Date"] < train_end) & (df["Date"] >= train_end - pd.DateOffset(years=WINDOW_Y))]
        tst = df[(df["Date"] >= ts) & (df["Date"] < te)]
        if len(tr) < 300 or tst.empty:
            continue
        X_tr = tr[feats].fillna(0)
        cols = _select_return_features(X_tr, tr["y"], tr["Date"].reset_index(drop=True), max_feats) \
            if select and len(feats) > max_feats else feats
        n_pos, n_neg = tr["y"].sum(), (1 - tr["y"]).sum()
        m = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, gamma=0.05, scale_pos_weight=n_neg / max(n_pos, 1),
                          eval_metric="logloss", random_state=seed, n_jobs=4, verbosity=0)
        m.fit(X_tr[cols], tr["y"])
        p = m.predict_proba(tst[cols].fillna(0))[:, 1]
        out.append(pd.DataFrame({"Date": tst["Date"].values, "p": p,
                                 "ret": tst[TARGET].values, "y": tst["y"].values}))
    return pd.concat(out, ignore_index=True)


def side_metrics(oos, buy_t=BUY_T, sell_t=SELL_T):
    """Edge por lado. 'trades' = muestreo cada 14 días hábiles (no solapado)."""
    def _block(o):
        res = {"n_dias": int(len(o))}
        try:
            res["auc"] = round(roc_auc_score(o["y"], o["p"]), 4)
        except ValueError:
            res["auc"] = None
        res["ret_medio_todos_pct"] = round(o["ret"].mean() * 100, 3)   # benchmark "siempre comprado"
        for side, mask, sgn in (("BUY", o["p"] > buy_t, 1), ("SELL", o["p"] < sell_t, -1)):
            s = o[mask]
            res[side] = {
                "pct_dias": round(mask.mean() * 100, 1),
                "hit_rate": round(((s["ret"] * sgn) > COST).mean() * 100, 1) if len(s) else None,
                "ret_medio_pct": round((s["ret"] * sgn - COST).mean() * 100, 3) if len(s) else None,
            }
        # trades no solapados
        t = o.iloc[::14]
        for side, mask, sgn in (("BUY", t["p"] > buy_t, 1), ("SELL", t["p"] < sell_t, -1)):
            s = t[mask]
            r = s["ret"] * sgn - COST
            res[side]["trades"] = int(len(s))
            res[side]["trades_pnl_pct"] = round(r.sum() * 100, 2) if len(s) else 0
            res[side]["trades_t_stat"] = round(r.mean() / (r.std() / np.sqrt(len(r))), 2) if len(r) > 2 and r.std() > 0 else None
        return res

    mid = oos["Date"].quantile(0.5)
    return {"total": _block(oos), "mitad_1": _block(oos[oos["Date"] < mid]),
            "mitad_2": _block(oos[oos["Date"] >= mid]),
            "corte_mitad": str(pd.Timestamp(mid).date())}


def run_side(df, feats):
    print("== Walk-forward modelo actual (selección top-40 por fold) ==")
    oos = walk_forward(df, feats, select=True)
    res = side_metrics(oos)
    # sensibilidad de umbrales SELL
    res["sell_umbral"] = {}
    for st in (0.42, 0.38, 0.35, 0.30):
        s = side_metrics(oos, sell_t=st)["total"]["SELL"]
        res["sell_umbral"][str(st)] = s
    # ¿el SELL acierta más en mercado bajista? (régimen = precio < MA90 al momento)
    m = df.set_index("Date")["price_vs_ma90"]
    oos["bajo_ma90"] = oos["Date"].map(m).lt(0)
    res["por_regimen"] = {k: side_metrics(g)["total"] for k, g in
                          (("precio_bajo_ma90", oos[oos["bajo_ma90"]]), ("precio_sobre_ma90", oos[~oos["bajo_ma90"]]))}
    oos.to_csv(os.path.join(OUT_DIR, "model_audit_oos_actual.csv"), index=False)
    return res


def run_ablation(df, feats):
    groups = sorted({group_of(c) for c in feats})
    base = walk_forward(df, feats, select=True)
    base_auc = roc_auc_score(base["y"], base["p"])
    mid = base["Date"].quantile(0.5)
    def _aucs(o):
        return (round(roc_auc_score(o["y"], o["p"]), 4),
                round(roc_auc_score(o[o["Date"] < mid]["y"], o[o["Date"] < mid]["p"]), 4),
                round(roc_auc_score(o[o["Date"] >= mid]["y"], o[o["Date"] >= mid]["p"]), 4))
    res = {"base": _aucs(base), "drop": {}, "only": {}}
    print(f"base AUC {res['base']}")
    for g in groups:
        keep = [c for c in feats if group_of(c) != g]
        only = [c for c in feats if group_of(c) == g]
        d = _aucs(walk_forward(df, keep, select=True))
        o = _aucs(walk_forward(df, only, select=len(only) > 40))
        res["drop"][g] = {"auc": d, "delta_vs_base": round(d[0] - res["base"][0], 4), "n_feats": len(only)}
        res["only"][g] = {"auc": o}
        print(f"{g:16} n={len(only):3}  sin-grupo AUC {d}  Δ={d[0]-res['base'][0]:+.4f} | solo-grupo AUC {o}")
    return res


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT_DIR, exist_ok=True)
    df, feats = load()
    print(f"{len(df)} filas, {len(feats)} features, {df['Date'].min().date()} → {df['Date'].max().date()}")
    jobs = {"coverage": lambda: coverage(df, feats), "side": lambda: run_side(df, feats),
            "ablation": lambda: run_ablation(df, feats)}
    for name, fn in jobs.items():
        if part in (name, "all"):
            res = fn()
            with open(os.path.join(OUT_DIR, f"model_audit_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1, ensure_ascii=False, default=str)
            if name == "side":
                print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
