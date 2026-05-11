"""
scripts/test_post_spike_fade.py
Test empírico de la hipótesis del usuario:
  "Spikes de precio anticipan caídas posteriores"

Diseño:
  1. Definir SPIKE: ret_5d en percentil > X (varios cortes)
  2. Para cada spike, medir ret_30d_fwd
  3. Comparar contra ret_30d_fwd no-spike
  4. Test estadístico (Mann-Whitney U)

Análisis condicional:
  - Spikes con oil también subiendo (co-driver) vs solo soja
  - Spikes en régimen alcista vs bajista
  - Spikes con sentiment negativo (divergencia) vs positivo

Output: ¿bajo qué condiciones específicas el spike anticipa fade?
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # Recalcular ret_5d (returns últimos 5d) y ret_30d_fwd (return próximos 30d)
    df["ret_5d_past"] = df["Soybeans"].pct_change(5)
    df["ret_30d_fwd"] = df["Soybeans"].pct_change(30).shift(-30)

    # Para análisis condicional
    df["oil_5d_past"] = df["Oil"].pct_change(5)
    df["soyoil_5d_past"] = df["SoybeanOil"].pct_change(5)
    df["rsi_14"] = df.get("rsi_14", 50)
    df["mom_20d"] = df.get("mom_20d", 0)

    df = df.dropna(subset=["ret_5d_past", "ret_30d_fwd"]).copy()
    print(f"N total observaciones: {len(df)}")
    print(f"Período: {df['Date'].min().date()} → {df['Date'].max().date()}\n")

    # ── 1. Test base: post-spike vs no-spike ──────────────────────
    print("══════════════════════════════════════════════════════════════")
    print("  Test 1: post-spike fade (¿precio cae después de spikes?)")
    print("══════════════════════════════════════════════════════════════")

    from scipy.stats import mannwhitneyu, ttest_ind

    # Probar varios cortes para definir spike
    cuts = [0.90, 0.95, 0.97, 0.99]
    print(f"\n  Cut    Threshold   N_spike   N_normal   E[r30 spike]   E[r30 normal]   Δ        p-value")
    print(f"  {'-'*5}  {'-'*9}   {'-'*7}   {'-'*8}   {'-'*12}   {'-'*13}   {'-'*5}    {'-'*7}")
    for cut in cuts:
        thr = df["ret_5d_past"].quantile(cut)
        spike_mask = df["ret_5d_past"] >= thr
        n_spike  = int(spike_mask.sum())
        n_normal = int((~spike_mask).sum())
        r30_spike  = df.loc[spike_mask, "ret_30d_fwd"]
        r30_normal = df.loc[~spike_mask, "ret_30d_fwd"]
        m_spike  = float(r30_spike.mean()) * 100
        m_normal = float(r30_normal.mean()) * 100
        try:
            stat, pval = mannwhitneyu(r30_spike, r30_normal, alternative="two-sided")
        except Exception:
            pval = float("nan")
        sign = "✓" if pval < 0.05 else "×"
        print(f"  p>{cut*100:.0f}  {thr*100:>+6.2f}%   {n_spike:>5}     {n_normal:>5}     "
              f"{m_spike:>+8.2f}%      {m_normal:>+8.2f}%      {m_spike-m_normal:>+5.2f}    {pval:.4f} {sign}")

    # ── 2. Spikes condicionales: oil también subiendo ──────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  Test 2: spike soja CON oil también subiendo (co-driver)")
    print("══════════════════════════════════════════════════════════════")
    thr_soy = df["ret_5d_past"].quantile(0.95)
    thr_oil = df["oil_5d_past"].quantile(0.75)  # oil también subiendo
    cond1 = (df["ret_5d_past"] >= thr_soy) & (df["oil_5d_past"] >= thr_oil)
    cond2 = (df["ret_5d_past"] >= thr_soy) & (df["oil_5d_past"] <  thr_oil)
    nrm   = ~(df["ret_5d_past"] >= thr_soy)

    for label, mask in [("Spike soja + oil↑", cond1),
                         ("Spike soja + oil≈", cond2),
                         ("No-spike",         nrm)]:
        sub = df.loc[mask, "ret_30d_fwd"]
        if len(sub) > 5:
            print(f"  {label:<25} N={len(sub):>4}  E[r30]={sub.mean()*100:>+5.2f}%  "
                  f"std={sub.std()*100:>5.2f}%  pos%={(sub>0).mean()*100:>4.1f}%")

    # ── 3. Spikes RSI overbought ──────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  Test 3: spike + RSI sobrecomprado (>70 vs <70)")
    print("══════════════════════════════════════════════════════════════")
    rsi_high = (df["ret_5d_past"] >= thr_soy) & (df["rsi_14"] > 70)
    rsi_norm = (df["ret_5d_past"] >= thr_soy) & (df["rsi_14"].between(50, 70))
    rsi_low  = (df["ret_5d_past"] >= thr_soy) & (df["rsi_14"] < 50)

    for label, mask in [("Spike + RSI > 70 (overbought)", rsi_high),
                         ("Spike + RSI 50-70",            rsi_norm),
                         ("Spike + RSI < 50",             rsi_low)]:
        sub = df.loc[mask, "ret_30d_fwd"]
        if len(sub) > 3:
            print(f"  {label:<35} N={len(sub):>4}  E[r30]={sub.mean()*100:>+5.2f}%  "
                  f"pos%={(sub>0).mean()*100:>4.1f}%")

    # ── 4. Divergencia sentimiento ────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  Test 4: spike con divergencia bajista de sentimiento")
    print("══════════════════════════════════════════════════════════════")
    if "news_sentiment" in df.columns:
        sub_div = df[(df["ret_5d_past"] >= thr_soy) & (df["news_sentiment"] < -0.3)]
        sub_pos = df[(df["ret_5d_past"] >= thr_soy) & (df["news_sentiment"] > 0.3)]
        sub_neu = df[(df["ret_5d_past"] >= thr_soy) & df["news_sentiment"].between(-0.3, 0.3)]
        for label, sub in [("Spike + sentiment <-0.3 (divergencia)", sub_div),
                            ("Spike + sentiment > 0.3 (confirmación)", sub_pos),
                            ("Spike + sentiment neutral",              sub_neu)]:
            if len(sub) > 3:
                r = sub["ret_30d_fwd"]
                print(f"  {label:<43} N={len(sub):>3}  E[r30]={r.mean()*100:>+5.2f}%  "
                      f"pos%={(r>0).mean()*100:>4.1f}%")

    # ── 5. Análisis dual: spike up vs spike down ──────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  Test 5: simetría — fade después de spike up vs después de drop down")
    print("══════════════════════════════════════════════════════════════")
    thr_up   = df["ret_5d_past"].quantile(0.95)
    thr_down = df["ret_5d_past"].quantile(0.05)
    spike_up   = df["ret_5d_past"] >= thr_up
    spike_down = df["ret_5d_past"] <= thr_down
    normal     = df["ret_5d_past"].between(df["ret_5d_past"].quantile(0.40),
                                             df["ret_5d_past"].quantile(0.60))

    for label, mask in [("Spike UP (top 5%)",      spike_up),
                         ("Drop DOWN (bottom 5%)", spike_down),
                         ("Mid-range (40-60%)",    normal)]:
        sub = df.loc[mask, "ret_30d_fwd"]
        if len(sub) > 5:
            t_stat, t_p = ttest_ind(sub, df.loc[~mask, "ret_30d_fwd"])
            print(f"  {label:<27} N={len(sub):>4}  E[r30]={sub.mean()*100:>+5.2f}%  "
                  f"std={sub.std()*100:>5.2f}%  pos%={(sub>0).mean()*100:>4.1f}%  "
                  f"p_vs_rest={t_p:.4f}")


if __name__ == "__main__":
    main()
