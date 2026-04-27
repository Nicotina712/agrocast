"""
notebooks/intraday_data_quality.py
Exploración de calidad de datos intradía — Fase 0 (yfinance ZS=F).

Objetivo: decidir GO/NO-GO antes de invertir tiempo construyendo el resto del
módulo intraday/. Si yfinance no entrega cobertura suficiente, hay que pasar
directo a CME DataMine (Fase 1, paga) sin perder tiempo en Fase 0.

Outputs:
  - stdout: reporte legible
  - artifacts/intraday/
      coverage_5m.json, coverage_15m.json, coverage_60m.json
      hour_distribution_<interval>.png    (si matplotlib disponible)
      feature_summary_<interval>.csv
      verdict.json                        (resumen final)

Uso:
  python notebooks/intraday_data_quality.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime

# Forzar UTF-8 en stdout para Windows (consola cp1252 rompe con flechas Unicode)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Path setup: este script vive en notebooks/, importa src/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd  # noqa: E402

from src.intraday.data.tick_feed import (  # noqa: E402
    fetch_intraday_bars, diagnose_coverage,
    print_coverage_report, viability_score,
)
from src.intraday.features.microstructure import (  # noqa: E402
    build_intraday_features, quick_summary, feature_columns,
)


_OUT_DIR = os.path.join(_ROOT, "artifacts", "intraday")
os.makedirs(_OUT_DIR, exist_ok=True)


def _save_hour_plot(diag: dict, interval: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    hours = sorted(diag["hour_distribution_ct"].items())
    if not hours:
        return
    h, n = zip(*hours)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(h, n, color="steelblue")
    ax.axvspan(8.5, 13.33, alpha=0.15, color="green", label="RTH (08:30–13:20 CT)")
    ax.set_xlabel("Hora (CT)")
    ax.set_ylabel("Barras observadas")
    ax.set_title(f"ZS=F {interval} — distribución por hora CT")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, f"hour_distribution_{interval}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  📊 plot guardado: {path}")


def _save_feature_correlation_plot(feat: pd.DataFrame, interval: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    cols = [c for c in feature_columns() if c in feat.columns]
    cols = [c for c in cols if feat[c].notna().sum() > 100]
    if len(cols) < 5:
        return
    corr = feat[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    ax.set_title(f"Correlación de features — ZS=F {interval}")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, f"feature_corr_{interval}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  📊 plot guardado: {path}")


def explore(interval: str) -> dict:
    print(f"\n{'#'*72}")
    print(f"#  ZS=F  intervalo {interval}")
    print(f"{'#'*72}\n")

    df = fetch_intraday_bars(interval=interval, use_cache=True)
    if df.empty:
        return {"interval": interval, "verdict": "NO-GO", "reason": "EMPTY"}

    # Diagnóstico de cobertura
    diag = diagnose_coverage(df, interval=interval)
    print_coverage_report(diag)

    # Veredicto
    verdict = viability_score(diag)
    print(f"\n  VEREDICTO Fase 0: {verdict['verdict']}")
    for r in verdict.get("reasons", []):
        print(f"    ❌ {r}")
    for w in verdict.get("warnings", []):
        print(f"    ⚠️  {w}")

    # Persistir diag
    with open(os.path.join(_OUT_DIR, f"coverage_{interval}.json"), "w") as f:
        json.dump({**diag, "verdict": verdict}, f, indent=2, default=str)

    _save_hour_plot(diag, interval)

    # Construir features y reportar
    print(f"\n  → construyendo features intradía ...")
    feat = build_intraday_features(df, interval=interval)
    summ = quick_summary(feat)
    print(f"\n  Features ({feat.shape[0]} barras × {len(feature_columns())} cols):")
    print(summ.to_string())

    summ.to_csv(os.path.join(_OUT_DIR, f"feature_summary_{interval}.csv"))
    _save_feature_correlation_plot(feat, interval)

    # Sanity check: porcentaje de NaN en features esenciales
    essentials = ["ret_1", "rsi_14", "atr_14", "vwap_session", "vol_zscore_30"]
    nan_pct = {c: float(feat[c].isna().mean()) for c in essentials if c in feat.columns}
    print(f"\n  NaN% en features esenciales: " +
          ", ".join(f"{k}={v:.1%}" for k, v in nan_pct.items()))

    # Cantidad de barras RTH con TODOS los features esenciales válidos
    rth = feat[feat["is_rth"] == 1]
    valid = rth.dropna(subset=essentials)
    print(f"  Barras RTH usables (sin NaN en esenciales): "
          f"{len(valid):,} / {len(rth):,} "
          f"({len(valid)/max(len(rth),1):.1%})")

    return {
        "interval":           interval,
        "verdict":            verdict["verdict"],
        "reasons":            verdict.get("reasons", []),
        "warnings":           verdict.get("warnings", []),
        "n_bars_total":       diag["n_bars_total"],
        "n_bars_rth":         diag["n_bars_rth"],
        "n_sessions_rth":     diag["n_sessions_rth"],
        "rth_usable_for_ml":  int(len(valid)),
        "essentials_nan_pct": nan_pct,
    }


def main() -> None:
    print("=" * 72)
    print("  AGROCAST — Fase 0 Intradía: Auditoría de calidad de datos")
    print(f"  Run: {datetime.now().isoformat()}")
    print("=" * 72)

    results = []
    for interval in ("5m", "15m", "60m"):
        try:
            r = explore(interval)
            results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {interval}: {e}")
            results.append({"interval": interval, "verdict": "ERROR", "error": str(e)})

    # Resumen final
    print("\n" + "=" * 72)
    print("  RESUMEN FINAL")
    print("=" * 72)
    print(f"  {'interval':<10} {'verdict':<8} {'sessions':<10} "
          f"{'rth_bars':<10} {'usable_ml':<10}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r.get('interval','?'):<10} {r.get('verdict','?'):<8} "
              f"{r.get('n_sessions_rth','?'):<10} "
              f"{r.get('n_bars_rth','?'):<10} "
              f"{r.get('rth_usable_for_ml','?'):<10}")

    # Persistir veredicto global
    overall = "NO-GO"
    if any(r.get("verdict") == "GO" for r in results):
        overall = "GO"
    elif any(r.get("verdict") == "WARN" for r in results):
        overall = "WARN"

    final = {
        "run_at":   datetime.now().isoformat(),
        "overall":  overall,
        "by_interval": results,
        "decision_rule":
            "GO si al menos 1 intervalo cumple criterios; WARN si solo warnings; "
            "NO-GO si ninguno califica.",
        "next_step": {
            "GO":     "Avanzar a Fase 0.1: construir target intradía (ret futuro N barras) "
                      "y entrenar XGBoost piloto.",
            "WARN":   "Revisar warnings; puede que convenga limitar a un intervalo "
                      "específico o esperar más historia.",
            "NO-GO":  "Saltar a Fase 1 (CME DataMine). yfinance insuficiente.",
        }[overall],
    }
    with open(os.path.join(_OUT_DIR, "verdict.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    print(f"\n  VEREDICTO GLOBAL: {overall}")
    print(f"  → {final['next_step']}")
    print(f"\n  Artifacts en: {_OUT_DIR}")


if __name__ == "__main__":
    main()
