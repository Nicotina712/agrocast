"""
scripts/shock_half_life_analysis.py
Test empírico: ¿los shocks de soja tienen half-life predecible?

Idea: detectar shocks ex-post (price-based, no necesita LLM) y medir cómo evoluciona
el precio en los días siguientes. Si la persistencia tiene un patrón estable
(half-life consistente), entonces el "shock catalog + analog engine" es viable.

Pasos:
  1. Detectar shocks: ret_5d > 95th percentile (top 5%)
  2. Clasificar por tipo según co-driver dominante:
        - "oil_driven":      oil_5d > 75th percentile
        - "soyoil_driven":   soyoil_5d > 75th percentile (biofuels)
        - "speculative":     ni oil ni soyoil
  3. Para cada shock, medir trayectoria 30d:
        - day_to_peak: día cuando se alcanza el máximo
        - peak_gain_pct: cuánto subió desde el shock hasta el peak
        - day_to_half_fade: día cuando precio cae 50% del peak
        - day_to_full_fade: día cuando precio retorna a pre-shock
        - day_30_persistence: ¿cuánto del shock queda vivo a 30d?
  4. Agregados por tipo: half-life, % fade, distribución de outcomes
  5. Análogo del shock actual (oil-driven, +4.42% soja en 7d, +5.96% oil)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(ROOT, "data", "features.csv")


def detect_shocks(df: pd.DataFrame, q_spike: float = 0.95) -> pd.DataFrame:
    """Detecta shocks como spikes de retorno 5d > q_spike. Clasifica por tipo
    según co-driver dominante."""
    d = df.copy()
    d["ret_5d"]    = d["Soybeans"].pct_change(5)
    d["oil_5d"]    = d["Oil"].pct_change(5)
    d["soyoil_5d"] = d["SoybeanOil"].pct_change(5)
    d["maize_5d"]  = d["Maize"].pct_change(5)

    thr_soy    = float(d["ret_5d"].quantile(q_spike))
    thr_oil    = float(d["oil_5d"].quantile(0.75))
    thr_soyoil = float(d["soyoil_5d"].quantile(0.75))

    # Spikes
    d["is_spike"] = d["ret_5d"] >= thr_soy
    spikes = d[d["is_spike"]].copy()

    # Clasificación
    def classify(row):
        if row["oil_5d"] >= thr_oil:
            return "oil_driven"
        if row["soyoil_5d"] >= thr_soyoil:
            return "soyoil_driven"
        return "speculative"
    spikes["shock_type"] = spikes.apply(classify, axis=1)

    return spikes[["Date", "Soybeans", "ret_5d", "oil_5d", "soyoil_5d",
                    "shock_type"]].reset_index(drop=True)


def measure_trajectory(df_full: pd.DataFrame, shock_idx: int, days: int = 30) -> dict:
    """Para un shock detectado en df_full[shock_idx], mide la trayectoria
    de precio los próximos `days` días."""
    if shock_idx + days >= len(df_full):
        return {"ok": False}
    p0 = float(df_full.iloc[shock_idx]["Soybeans"])
    # Pre-shock = precio 5 días antes (start del spike)
    if shock_idx - 5 < 0:
        return {"ok": False}
    p_pre = float(df_full.iloc[shock_idx - 5]["Soybeans"])
    spike_size = (p0 - p_pre) / p_pre

    future = df_full.iloc[shock_idx:shock_idx + days + 1]["Soybeans"].values
    if len(future) < days:
        return {"ok": False}

    rels = (future - p_pre) / p_pre   # cambio relativo desde pre-shock
    # Peak post-shock
    peak_idx = int(np.argmax(rels))
    peak_rel = float(rels[peak_idx])
    peak_day = int(peak_idx)

    # Half-fade: cuándo el precio cae al 50% de la ganancia desde el peak
    fade_target = peak_rel - (peak_rel - 0) * 0.5  # 50% del peak gain (vs pre-shock)
    half_fade_day = None
    for i in range(peak_idx + 1, len(rels)):
        if rels[i] <= fade_target:
            half_fade_day = i
            break

    # Full fade: cuándo retorna a pre-shock (rel=0)
    full_fade_day = None
    for i in range(peak_idx + 1, len(rels)):
        if rels[i] <= 0:
            full_fade_day = i
            break

    # Persistencia a 30d: cuánto del peak queda
    persistence_30d = float(rels[-1] / peak_rel) if peak_rel > 1e-6 else 0.0
    survived_30d   = bool(rels[-1] > 0)

    return {
        "ok":               True,
        "spike_size_pct":   round(spike_size * 100, 2),
        "peak_day":         peak_day,
        "peak_gain_pct":    round(peak_rel * 100, 2),
        "half_fade_day":    half_fade_day,
        "full_fade_day":    full_fade_day,
        "persistence_30d_pct": round(persistence_30d * 100, 1),
        "ret_30d_pct":      round(float(rels[-1]) * 100, 2),
        "survived_30d":     survived_30d,
    }


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"N total: {len(df)}  rango: {df['Date'].min().date()} → {df['Date'].max().date()}\n")

    shocks = detect_shocks(df, q_spike=0.95)
    print(f"Shocks detectados (top 5% spikes 5d): {len(shocks)}")
    print(f"  oil_driven:    {(shocks['shock_type'] == 'oil_driven').sum()}")
    print(f"  soyoil_driven: {(shocks['shock_type'] == 'soyoil_driven').sum()}")
    print(f"  speculative:   {(shocks['shock_type'] == 'speculative').sum()}\n")

    # Medir trayectoria de cada shock
    rows = []
    for _, sh in shocks.iterrows():
        idx = int(df.index[df["Date"] == sh["Date"]][0])
        traj = measure_trajectory(df, idx, days=30)
        if not traj.get("ok"):
            continue
        rows.append({"Date": sh["Date"], "shock_type": sh["shock_type"],
                      "ret_5d_at_shock": sh["ret_5d"],
                      "oil_5d_at_shock": sh["oil_5d"], **traj})
    R = pd.DataFrame(rows)
    if R.empty:
        print("Sin shocks medibles.")
        return

    # ── Agregados por tipo ──────────────────────────────────────
    print("══════════════════════════════════════════════════════════════")
    print("  Trayectorias post-shock por tipo")
    print("══════════════════════════════════════════════════════════════")
    print(f"{'Tipo':<18} {'N':>4} {'peak_day':>8} {'peak_gain%':>10} {'half_fade_d':>11} "
          f"{'full_fade_d':>11} {'ret_30d%':>9} {'survived%':>9}")
    print("-" * 95)
    for stype in ("oil_driven", "soyoil_driven", "speculative"):
        sub = R[R["shock_type"] == stype]
        if sub.empty:
            continue
        peak_day_med  = float(sub["peak_day"].median())
        peak_gain_med = float(sub["peak_gain_pct"].median())
        # Half/full fade days (filtrar None)
        hf = sub["half_fade_day"].dropna()
        ff = sub["full_fade_day"].dropna()
        hf_med = float(hf.median()) if not hf.empty else None
        ff_med = float(ff.median()) if not ff.empty else None
        ret_30d_med = float(sub["ret_30d_pct"].median())
        survived_pct = float(sub["survived_30d"].mean() * 100)

        hf_str = f"{hf_med:>11.0f}" if hf_med is not None else f"{'>30d':>11}"
        ff_str = f"{ff_med:>11.0f}" if ff_med is not None else f"{'>30d':>11}"
        print(f"{stype:<18} {len(sub):>4} {peak_day_med:>8.0f} "
              f"{peak_gain_med:>+9.2f}% {hf_str} {ff_str} "
              f"{ret_30d_med:>+8.2f}% {survived_pct:>8.1f}%")

    # ── Distribución detallada oil-driven ───────────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  Distribución detallada — shocks oil-driven (caso de hoy)")
    print("══════════════════════════════════════════════════════════════")
    sub = R[R["shock_type"] == "oil_driven"]
    if not sub.empty:
        print(f"  N={len(sub)} eventos históricos")
        print(f"  Spike size original (med): {sub['spike_size_pct'].median():.2f}%")
        print(f"  Peak day distribution: q25={sub['peak_day'].quantile(0.25):.0f}d, "
              f"med={sub['peak_day'].median():.0f}d, q75={sub['peak_day'].quantile(0.75):.0f}d")
        print(f"  Peak gain distribution: q25={sub['peak_gain_pct'].quantile(0.25):+.1f}%, "
              f"med={sub['peak_gain_pct'].median():+.1f}%, q75={sub['peak_gain_pct'].quantile(0.75):+.1f}%")
        # ¿Cuánto del shock dura?
        hf = sub["half_fade_day"].dropna()
        if not hf.empty:
            print(f"  Half-fade reached in: q25={hf.quantile(0.25):.0f}d, med={hf.median():.0f}d, "
                  f"q75={hf.quantile(0.75):.0f}d (n={len(hf)}/{len(sub)})")
        else:
            print(f"  Half-fade no alcanzado en 30d en ninguno → shock muy persistente")

        # Últimos 5 shocks oil-driven
        print(f"\n  Últimos 10 shocks oil-driven (más recientes):")
        latest = sub.sort_values("Date").tail(10)
        for _, row in latest.iterrows():
            hf = row["half_fade_day"]
            hf_s = f"{hf:.0f}d" if pd.notna(hf) else ">30d"
            print(f"    {pd.Timestamp(row['Date']).date()}  spike {row['spike_size_pct']:+5.2f}%  "
                  f"oil5d {row['oil_5d_at_shock']*100:+5.2f}%  peak day {row['peak_day']:>2.0f} "
                  f"({row['peak_gain_pct']:+5.2f}%)  half-fade {hf_s}  ret30d {row['ret_30d_pct']:+5.2f}%")

    # ── Análogo del shock actual ────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  Análogo del shock actual (oil-driven, soja +4.42%/7d, oil +5.96%/7d)")
    print("══════════════════════════════════════════════════════════════")
    # Buscar shocks similares: oil-driven con ret_5d en [3%, 6%] y oil_5d en [4%, 8%]
    similar = R[(R["shock_type"] == "oil_driven") &
                 (R["ret_5d_at_shock"].between(0.03, 0.06)) &
                 (R["oil_5d_at_shock"].between(0.04, 0.08))]
    print(f"  N análogos: {len(similar)}")
    if not similar.empty:
        print(f"  Outcomes a 30d:")
        print(f"    ret_30d med: {similar['ret_30d_pct'].median():+.2f}%")
        print(f"    ret_30d q25: {similar['ret_30d_pct'].quantile(0.25):+.2f}%")
        print(f"    ret_30d q75: {similar['ret_30d_pct'].quantile(0.75):+.2f}%")
        print(f"    survived 30d: {similar['survived_30d'].mean() * 100:.1f}%")
        peak_d = similar['peak_day'].dropna()
        if not peak_d.empty:
            print(f"    peak day med: {peak_d.median():.0f}d (q25={peak_d.quantile(0.25):.0f}, q75={peak_d.quantile(0.75):.0f})")
        hf = similar['half_fade_day'].dropna()
        if not hf.empty:
            print(f"    half-fade day med: {hf.median():.0f}d (n={len(hf)}/{len(similar)})")

    # ── Persistir como CSV ──────────────────────────────────────
    out_path = os.path.join(ROOT, "artifacts_eval", "shock_catalog.csv")
    R.to_csv(out_path, index=False)
    print(f"\n💾 {out_path}")


if __name__ == "__main__":
    main()
