"""
Variantes simplificadas del clasificador 14d, mismo walk-forward que
research_model_audit (trimestral, rolling 5y, embargo 18d, 2019→hoy).

Compara: modelo actual (199 → top-40) vs solo variables con historia completa
vs modelos chicos (logística regularizada / XGB poco profundo) sobre pocas
variables con sentido económico.

Uso: python -m scripts.research_model_variants
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.research_model_audit import (  # noqa: E402
    load, group_of, walk_forward, side_metrics, EMBARGO, WINDOW_Y, START_TEST, OUT_DIR)

SPARSE_GROUPS = {"satelite", "cme_oi_vol", "opciones_iv", "noticias", "curva", "crop_progress"}
CORE = ["mom_20d", "mom_60d", "price_vs_ma90", "ma20_vs_ma90", "vol_30d", "rsi_14",
        "soy_corn_ratio_dev", "crush_spread_dev", "Dollar_chg7", "Oil_chg7",
        "insp_soy_zscore", "cot_index", "wasde_surprise_ma3", "enso_oni", "month_sin", "month_cos"]


def wf_logit(df, feats, C=0.1):
    test_starts = pd.date_range(START_TEST, df["Date"].max(), freq="QS")
    out = []
    for i, ts in enumerate(test_starts):
        te = test_starts[i + 1] if i + 1 < len(test_starts) else df["Date"].max() + pd.Timedelta(days=1)
        train_end = ts - pd.Timedelta(days=EMBARGO)
        tr = df[(df["Date"] < train_end) & (df["Date"] >= train_end - pd.DateOffset(years=WINDOW_Y))]
        tst = df[(df["Date"] >= ts) & (df["Date"] < te)]
        if len(tr) < 300 or tst.empty:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        m.fit(tr[feats].fillna(0), tr["y"])
        out.append(pd.DataFrame({"Date": tst["Date"].values, "p": m.predict_proba(tst[feats].fillna(0))[:, 1],
                                 "ret": tst["ret_14d_fwd"].values, "y": tst["y"].values}))
    return pd.concat(out, ignore_index=True)


def summary(name, oos):
    s = side_metrics(oos)
    t, h1, h2 = s["total"], s["mitad_1"], s["mitad_2"]
    row = {"variante": name, "auc": t["auc"], "auc_h1": h1["auc"], "auc_h2": h2["auc"],
           "buy_ret_pct": t["BUY"]["ret_medio_pct"], "buy_t": t["BUY"]["trades_t_stat"],
           "sell_ret_pct": t["SELL"]["ret_medio_pct"], "sell_t": t["SELL"]["trades_t_stat"]}
    print(f"{name:34} AUC {t['auc']:.3f} (H1 {h1['auc']:.3f} / H2 {h2['auc']:.3f}) | "
          f"BUY {t['BUY']['ret_medio_pct']:+.2f}% t={t['BUY']['trades_t_stat']} | "
          f"SELL {t['SELL']['ret_medio_pct']:+.2f}% t={t['SELL']['trades_t_stat']}")
    return row


def main():
    df, feats = load()
    dense = [c for c in feats if group_of(c) not in SPARSE_GROUPS]
    core = [c for c in CORE if c in df.columns]
    rows = [
        summary("actual (199 → top40, XGB)", walk_forward(df, feats, select=True)),
        summary(f"densas ({len(dense)} → top40, XGB)", walk_forward(df, dense, select=True)),
        summary(f"densas ({len(dense)} → top15, XGB)", walk_forward(df, dense, select=True, max_feats=15)),
        summary(f"core {len(core)} (XGB)", walk_forward(df, core, select=False)),
        summary(f"core {len(core)} (logística C=0.1)", wf_logit(df, core, 0.1)),
        summary(f"core {len(core)} (logística C=0.01)", wf_logit(df, core, 0.01)),
    ]
    with open(os.path.join(OUT_DIR, "model_variants.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
