"""
Demo del nuevo predict_horizons vs el forecast.csv productivo actual.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.predict_horizons import forecast_curve, forecast_anchors

FEATURES = os.path.join(ROOT, "data", "features.csv")
OLD_FCST = os.path.join(ROOT, "artifacts", "forecast.csv")

df = pd.read_csv(FEATURES, parse_dates=["Date"])
print(f"Features: {df['Date'].min().date()} → {df['Date'].max().date()}\n")

# Anchors
a = forecast_anchors(df)
print(f"Precio actual: ${a['current_price']:.2f}\n")
for h, info in a["horizons"].items():
    print(f"  {h}d → ${info['price']:.2f}  "
          f"[{info['q10']:.1f} ─ {info['q90']:.1f}]  "
          f"ret={info['return_pct']:+.2f}%  "
          f"(modelo puro={info['model_return_pct']:+.2f}%, α={info['alpha']:.2f}, δ=${info['delta']:.1f})")

# Curva
new = forecast_curve(df, steps=30)
old = pd.read_csv(OLD_FCST)

print(f"\nCurva diaria — comparativa nuevo vs producción actual:")
print(f"{'día':>4} │ {'fecha':>10} │ {'NUEVO':>10}  [{'q10':>7} ─ {'q90':>7}] │ {'OLD':>10}  [{'old_lo':>7} ─ {'old_hi':>7}]")
print("─" * 96)
for i in (0, 4, 9, 14, 19, 24, 29):
    nrow = new.iloc[i]
    orow = old.iloc[i] if i < len(old) else None
    if orow is not None:
        print(f"{i+1:>4} │ {str(nrow['Date'])[:10]} │ "
              f"{nrow['Soybeans']:>10.2f}  [{nrow['lower']:>7.1f} ─ {nrow['upper']:>7.1f}] │ "
              f"{orow['Soybeans']:>10.2f}  [{orow['lower']:>7.1f} ─ {orow['upper']:>7.1f}]")

# Métricas de forma
def total_change(p):
    return (p[-1] / p[0] - 1) * 100 if len(p) > 0 else 0

new_p = new["Soybeans"].values
old_p = old["Soybeans"].values
new_rets = np.diff(new_p) / new_p[:-1] * 100
old_rets = np.diff(old_p) / old_p[:-1] * 100

print(f"\n── Forma de la curva ──")
print(f"  NUEVO  → cambio total = {(new_p[-1]/a['current_price']-1)*100:+.2f}%, "
      f"std(rets diarios) = {np.std(new_rets):.3f}%, "
      f"% pasos saturando ±1% = {(np.abs(new_rets) >= 0.99).mean()*100:.0f}%")
print(f"  OLD    → cambio total = {(old_p[-1]/a['current_price']-1)*100:+.2f}%, "
      f"std(rets diarios) = {np.std(old_rets):.3f}%, "
      f"% pasos saturando ±1% = {(np.abs(old_rets) >= 0.99).mean()*100:.0f}%")

print(f"\n── Ancho promedio de bandas ──")
print(f"  NUEVO  banda media = ${(new['upper']-new['lower']).mean():.1f}  "
      f"(crece de ${new['upper'].iloc[0]-new['lower'].iloc[0]:.1f} día 1 a ${new['upper'].iloc[-1]-new['lower'].iloc[-1]:.1f} día 30)")
print(f"  OLD    banda media = ${(old['upper']-old['lower']).mean():.1f}  "
      f"(crece de ${old['upper'].iloc[0]-old['lower'].iloc[0]:.1f} día 1 a ${old['upper'].iloc[-1]-old['lower'].iloc[-1]:.1f} día 30)")
