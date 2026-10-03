"""
¿Qué motores aportan inteligencia NO direccional al productor? (research)

1. Volatilidad: ¿se puede pronosticar el riesgo (no la dirección)?
   Compara, fuera de muestra, pronósticos de vol realizada a 30d:
   naive (vol 30d pasada), EWMA (RiskMetrics λ=0.94), vol 60d, y el vol-head ML.
2. Rangos: ¿las bandas basadas en vol EWMA cubren lo que dicen (80%)?
3. Eventos: ¿los días WASDE mueven más el precio que un día normal?
4. Estacionalidad: ¿hay meses sistemáticamente mejores para vender? (por mitades)

Uso: python -m scripts.research_producer_engines
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "artifacts", "research", "producer_engines.json")


def load_prices():
    df = pd.read_csv(os.path.join(ROOT, "data", "raw_market.csv"), parse_dates=["Date"])
    df = df.sort_values("Date").dropna(subset=["Soybeans"]).reset_index(drop=True)
    # rollovers enmascarados: retornos diarios absurdos (>8%) fuera
    df["r"] = np.log(df["Soybeans"]).diff()
    df.loc[df["r"].abs() > 0.08, "r"] = np.nan
    return df


def vol_forecasting(df, h=30):
    r = df["r"]
    fwd = r[::-1].rolling(h, min_periods=h - 5).std()[::-1].shift(-1) * np.sqrt(252)   # vol realizada próximos h días
    cands = {
        "naive_30d": r.rolling(30).std() * np.sqrt(252),
        "vol_60d":   r.rolling(60).std() * np.sqrt(252),
        "ewma_094":  np.sqrt((r.fillna(0) ** 2).ewm(alpha=0.06).mean()) * np.sqrt(252),
        "media_larga_1y": r.rolling(252).std() * np.sqrt(252),
    }
    d = pd.DataFrame({"Date": df["Date"], "fwd": fwd, **cands}).dropna()
    d = d[d["Date"] >= "2018-01-01"].iloc[::h]          # muestras no solapadas
    mid = d["Date"].quantile(0.5)
    res = {}
    for k in cands:
        e = (d[k] - d["fwd"]).abs()
        res[k] = {"mae_total": round(e.mean(), 4),
                  "mae_h1": round(e[d["Date"] < mid].mean(), 4),
                  "mae_h2": round(e[d["Date"] >= mid].mean(), 4),
                  "corr": round(d[k].corr(d["fwd"]), 3)}
    res["n"] = int(len(d))
    res["vol_media_realizada"] = round(d["fwd"].mean(), 3)
    return res


def band_coverage(df, horizons=(30, 60, 90, 180), z=1.2816):
    """Banda 80% log-normal sin drift con vol EWMA: ¿cubre ~80%?"""
    r = df["r"]
    sig_d = np.sqrt((r.fillna(0) ** 2).ewm(alpha=0.06).mean())
    out = {}
    for h in horizons:
        fut = df["Soybeans"].shift(-h)
        lr = np.log(fut / df["Soybeans"])
        width = z * sig_d * np.sqrt(h)
        ok = (lr.abs() <= width)
        d = pd.DataFrame({"Date": df["Date"], "ok": ok, "lr": lr, "w": width}).dropna()
        d = d[d["Date"] >= "2018-01-01"]
        mid = d["Date"].quantile(0.5)
        out[f"{h}d"] = {"cobertura_pct": round(d["ok"].mean() * 100, 1),
                        "h1": round(d[d["Date"] < mid]["ok"].mean() * 100, 1),
                        "h2": round(d[d["Date"] >= mid]["ok"].mean() * 100, 1),
                        "por_debajo_pct": round((d["lr"] < -d["w"]).mean() * 100, 1),
                        "por_encima_pct": round((d["lr"] > d["w"]).mean() * 100, 1),
                        "ancho_medio_pct": round(d["w"].mean() * 100, 1)}
    return out


def event_vol(df):
    from src.data.event_calendar import build_event_calendar
    cal = pd.DataFrame(build_event_calendar(2016))
    cal["date"] = pd.to_datetime(cal["date"])
    res = {}
    absr = df.set_index("Date")["r"].abs()
    base = absr.median()
    for typ, g in cal.groupby("kind"):
        days = absr.reindex(g["date"]).dropna()
        if len(days) < 10:
            continue
        res[str(typ)] = {"n": int(len(days)),
                         "mov_medio_pct": round(days.mean() * 100, 2),
                         "mov_mediana_pct": round(days.median() * 100, 2),
                         "vs_dia_normal_x": round(days.median() / base, 2),
                         "p90_pct": round(days.quantile(0.9) * 100, 2)}
    res["dia_normal_mediana_pct"] = round(base * 100, 2)
    return res


def seasonality(df):
    m = df.set_index("Date")["Soybeans"].resample("ME").last()
    ret = np.log(m).diff().dropna()
    d = pd.DataFrame({"ret": ret, "month": ret.index.month, "year": ret.index.year})
    d = d[d["year"] >= 2010] if d["year"].min() < 2010 else d
    mid = d["year"].median()
    out = {}
    for mo, g in d.groupby("month"):
        h1, h2 = g[g["year"] < mid]["ret"], g[g["year"] >= mid]["ret"]
        t = g["ret"].mean() / (g["ret"].std() / np.sqrt(len(g))) if len(g) > 2 else None
        out[int(mo)] = {"ret_medio_pct": round(g["ret"].mean() * 100, 2), "t": round(t, 2) if t else None,
                        "h1_pct": round(h1.mean() * 100, 2), "h2_pct": round(h2.mean() * 100, 2),
                        "pct_meses_sube": round((g["ret"] > 0).mean() * 100), "n": int(len(g))}
    out["periodo"] = f"{d.index.min().date()} → {d.index.max().date()}"
    return out


def main():
    df = load_prices()
    res = {"vol_forecast_30d": vol_forecasting(df), "bandas_80": band_coverage(df),
           "eventos": event_vol(df), "estacionalidad_mensual": seasonality(df)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
