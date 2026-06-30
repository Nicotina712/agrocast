"""Generate boilerplate files for all remaining instruments."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# (symbol, magic, port, color, emoji, regime_field, extra_field, extra_label, bat_name, comment)
INSTRUMENTS = [
    ("US500",   20260604, 8083, "#00c853",  "📈", "sp500_regime",  "risk_on_off",      "Risk",    "run_us500.bat",  "US500_QA"),
    ("USTEC",   20260605, 8084, "#00b0ff",  "💻", "ustec_regime",  "ai_sentiment",     "AI Sent", "run_ustec.bat",  "USTEC_QA"),
    ("US30",    20260606, 8085, "#ff6f00",  "🏭", "us30_regime",   "risk_on_off",      "Risk",    "run_us30.bat",   "US30_QA"),
    ("WTI_N6",  20260607, 8086, "#795548",  "🛢", "oil_regime",    "opec_dynamic",     "OPEC",    "run_wti.bat",    "WTI_QA"),
    ("BRENT_N6",20260608, 8087, "#546e7a",  "🛢", "oil_regime",    "brent_wti_spread", "Spread",  "run_brent.bat",  "BRENT_QA"),
    ("UK100",   20260609, 8088, "#e91e63",  "🇬🇧","uk100_regime",  "gbp_impact",       "GBP",     "run_uk100.bat",  "UK100_QA"),
]

# ── execution_tracker.py ────────────────────────────────────────────────────

TRACKER_TMPL = '''"""{{SYM}} — Execution Tracker"""

import os, sys, json, argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import SYMBOL, ARTIFACTS_DIR, EXEC_LOG_FILE, PAPER_LOG_FILE

OUR_MAGIC = {{MAGIC}}


def _read_jsonl(path):
    if not os.path.exists(path): return []
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: lines.append(json.loads(line))
                except: pass
    return lines


def compute_performance(trades):
    if not trades: return {"total_trades": 0}
    longs  = [t for t in trades if t.get("signal") == "LONG"]
    shorts = [t for t in trades if t.get("signal") == "SHORT"]
    rr_vals   = [float(t["rr"])       for t in trades if t.get("rr")]
    risk_vals = [float(t["risk_usd"]) for t in trades if t.get("risk_usd")]
    confs     = [t.get("confidence","?") for t in trades]
    avg_rr   = round(sum(rr_vals)/len(rr_vals), 2)    if rr_vals   else None
    avg_risk = round(sum(risk_vals)/len(risk_vals), 2) if risk_vals else None
    conf_dist = {}
    for c in confs: conf_dist[c] = conf_dist.get(c,0)+1
    regimes = {}
    for t in trades:
        r = t.get("{{REGIME}}","unknown"); regimes[r] = regimes.get(r,0)+1
    extra_dist = {}
    for t in trades:
        e = t.get("{{EXTRA}}","unknown"); extra_dist[e] = extra_dist.get(e,0)+1
    return {
        "total_trades": len(trades), "longs": len(longs), "shorts": len(shorts),
        "avg_rr": avg_rr, "avg_risk_usd": avg_risk,
        "confidence_dist": conf_dist,
        "regime_dist": regimes, "extra_dist": extra_dist,
    }


def sync_fills():
    try:
        from mt5_bridge import initialize, get_positions, is_connected
        if not is_connected(): initialize()
        positions = get_positions(SYMBOL)
        return [p for p in (positions or []) if p.get("magic") == OUR_MAGIC]
    except Exception as e:
        print(f"MT5 sync error: {e}"); return []


def print_report():
    paper = _read_jsonl(PAPER_LOG_FILE)
    exec_ = _read_jsonl(EXEC_LOG_FILE)
    trades = [t for t in paper if t.get("signal") not in (None,"FLAT")]
    flat   = [t for t in paper if t.get("signal") == "FLAT"]
    stats  = compute_performance(trades)
    print("\\n" + "="*58)
    print(f"  {{SYM}} Robot — Performance Report")
    print("="*58)
    print(f"  Paper trades total : {stats.get(\'total_trades\',0)}")
    print(f"    LONG             : {stats.get(\'longs\',0)}")
    print(f"    SHORT            : {stats.get(\'shorts\',0)}")
    print(f"    FLAT (skipped)   : {len(flat)}")
    print(f"  Avg R:R            : {stats.get(\'avg_rr\',\'N/A\')}")
    print(f"  Avg Risk/trade     : ${stats.get(\'avg_risk_usd\',\'N/A\')}")
    print(f"  Confidence dist    : {stats.get(\'confidence_dist\',{})}")
    print(f"  Regime dist        : {stats.get(\'regime_dist\',{})}")
    print(f"  {{EXTRA_LABEL}} dist    : {stats.get(\'extra_dist\',{})}")
    print(f"  MT5 executions     : {len(exec_)}")
    print("="*58)
    if trades:
        print("\\nRecent paper trades (last 5):")
        for t in trades[-5:]:
            ts  = t.get("timestamp","")[:16].replace("T"," ")
            sig = t.get("signal","?")
            ent = t.get("entry"); sl = t.get("sl"); tp = t.get("tp")
            ent_s = f"${ent:,.2f}" if isinstance(ent,(int,float)) else str(ent)
            sl_s  = f"${sl:,.2f}"  if isinstance(sl,(int,float))  else str(sl)
            tp_s  = f"${tp:,.2f}"  if isinstance(tp,(int,float))  else str(tp)
            print(f"  {ts} | {sig:5s} @ {ent_s:>12} | SL:{sl_s:>12} | TP:{tp_s:>12} | R:R {t.get(\'rr\',\'?\')}")


def main():
    parser = argparse.ArgumentParser(description="{{SYM}} Execution Tracker")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--sync",   action="store_true")
    args = parser.parse_args()
    if args.sync:
        fills = sync_fills()
        print(f"Open {{SYM}} positions (magic={OUR_MAGIC}): {len(fills)}")
        for p in fills: print(f"  {p}")
    else:
        print_report()

if __name__ == "__main__":
    main()
'''

# ── retrainer.py ────────────────────────────────────────────────────────────

RETRAINER_TMPL = '''"""{{SYM}} — Walk-Forward Retrainer"""

import os, sys, json, argparse, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

from config import SYMBOL, TIMEFRAME, N_BARS_LIVE, ARTIFACTS_DIR
from mt5_bridge import initialize as mt5_init, fetch_mt5_bars, is_connected as mt5_connected
from microstructure import build_intraday_features

N_SPLITS  = 5
EMBARGO   = 24
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")


def fetch_training_bars(n=5000):
    if not mt5_connected():
        if not mt5_init():
            raise RuntimeError("MT5 connection failed")
    bars = fetch_mt5_bars(TIMEFRAME, n, SYMBOL)
    if bars is None or len(bars) < 200:
        raise ValueError(f"Not enough bars: {len(bars) if bars is not None else 0}")
    return bars


def build_features(bars):
    feats = build_intraday_features(bars, TIMEFRAME)
    X_cols = [c for c in feats.columns if c not in ("signal","label","target")]
    X = feats[X_cols].fillna(0).replace([np.inf,-np.inf], 0)
    # Build target: 1=long profitable, -1=short profitable, 0=flat
    close = bars["close"].values
    target = []
    fwd = 6
    for i in range(len(close)):
        if i + fwd >= len(close):
            target.append(0)
        else:
            ret = (close[i+fwd] - close[i]) / close[i]
            if ret > 0.001:   target.append(1)
            elif ret < -0.001: target.append(-1)
            else:              target.append(0)
    y = pd.Series(target, index=feats.index)
    mask = X.index.isin(feats.index)
    return X[mask], y[mask]


def walk_forward_cv(X, y):
    n = len(X)
    fold_size = n // (N_SPLITS + 1)
    results = []
    for i in range(N_SPLITS):
        train_end = fold_size * (i + 1)
        test_start = train_end + EMBARGO
        test_end   = test_start + fold_size
        if test_end > n: break
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        sc  = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        acc   = accuracy_score(y_te, preds)
        results.append({"fold": i+1, "train_n": train_end, "test_n": len(y_te), "accuracy": round(acc,4)})
        print(f"  Fold {i+1}: acc={acc:.4f} | train={train_end} | test={len(y_te)}")
    return results


def train_final(X, y):
    os.makedirs(MODEL_DIR, exist_ok=True)
    sc  = StandardScaler()
    X_s = sc.fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    clf.fit(X_s, y)
    mpath = os.path.join(MODEL_DIR, f"{SYMBOL}_model.pkl")
    spath = os.path.join(MODEL_DIR, f"{SYMBOL}_scaler.pkl")
    joblib.dump(clf, mpath)
    joblib.dump(sc,  spath)
    print(f"  Model saved: {mpath}")
    print(f"  Scaler saved: {spath}")
    return clf, sc


def main():
    parser = argparse.ArgumentParser(description="{{SYM}} Retrainer")
    parser.add_argument("--cv-only", action="store_true", help="Cross-validation only, no final fit")
    parser.add_argument("--bars",    type=int, default=5000)
    args = parser.parse_args()

    print(f"\\n{'='*55}")
    print(f"  {{SYM}} Walk-Forward Retrainer")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    print("\\nFetching bars...")
    bars = fetch_training_bars(args.bars)
    print(f"  Got {len(bars)} bars of {TIMEFRAME}")

    print("\\nBuilding features...")
    X, y = build_features(bars)
    print(f"  X shape: {X.shape} | Classes: {dict(y.value_counts())}")

    print(f"\\nWalk-forward CV ({N_SPLITS} splits, embargo={EMBARGO})...")
    cv_results = walk_forward_cv(X, y)
    accs = [r["accuracy"] for r in cv_results]
    print(f"  Mean acc: {np.mean(accs):.4f} | Std: {np.std(accs):.4f}")

    if not args.cv_only:
        print("\\nTraining final model on full dataset...")
        train_final(X, y)

    rpath = os.path.join(ARTIFACTS_DIR, "retrain_results.json")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(rpath,"w",encoding="utf-8") as f:
        json.dump({"symbol":SYMBOL,"timestamp":datetime.now().isoformat(),
                   "n_bars":len(bars),"cv_results":cv_results,
                   "mean_acc":round(np.mean(accs),4) if accs else None}, f, indent=2)
    print(f"\\nResults saved: {rpath}")
    print("\\nDone.")

if __name__ == "__main__":
    main()
'''

# ── dashboard.py ────────────────────────────────────────────────────────────

DASHBOARD_TMPL = '''"""{{SYM}} — Web Dashboard (port {{PORT}})"""

import os, sys, json, threading
from datetime import datetime, date
from http.server import HTTPServer, BaseHTTPRequestHandler

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import SYMBOL, ARTIFACTS_DIR, PAPER_LOG_FILE, LIVE_LOG_FILE, SIGNAL_FILE

PORT   = {{PORT}}
COLOR  = "{{COLOR}}"
EMOJI  = "{{EMOJI}}"


def _read_jsonl(path, n=50):
    if not os.path.exists(path): return []
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: lines.append(json.loads(line))
                except: pass
    return lines[-n:]


def _read_signal():
    if not os.path.exists(SIGNAL_FILE): return {}
    try:
        with open(SIGNAL_FILE, encoding="utf-8") as f: return json.load(f)
    except: return {}


def _fmt(v):
    return f"${v:,.2f}" if isinstance(v,(int,float)) else str(v) if v else "—"


def _build_html():
    sig      = _read_signal()
    paper    = _read_jsonl(PAPER_LOG_FILE, 200)
    logs     = _read_jsonl(LIVE_LOG_FILE,  30)
    trades   = [t for t in paper if t.get("signal") not in (None,"FLAT")]
    flat_ct  = sum(1 for t in paper if t.get("signal") == "FLAT")
    ls       = trades[-1] if trades else {}

    total = len(trades)
    rr_vals = [float(t["rr"]) for t in trades if t.get("rr")]
    risk_vals = [float(t["risk_usd"]) for t in trades if t.get("risk_usd")]
    avg_rr   = round(sum(rr_vals)/len(rr_vals),2)   if rr_vals   else "—"
    avg_risk = round(sum(risk_vals)/len(risk_vals),2) if risk_vals else "—"
    longs  = sum(1 for t in trades if t.get("signal")=="LONG")
    shorts = sum(1 for t in trades if t.get("signal")=="SHORT")

    sig_color = "#00c853" if sig.get("signal")=="LONG" else "#f44336" if sig.get("signal")=="SHORT" else "#888"
    sig_label = sig.get("signal","—")

    rows = ""
    for t in reversed(trades[-20:]):
        ts  = t.get("timestamp","")[:16].replace("T"," ")
        s   = t.get("signal","?")
        sc  = "#00c853" if s=="LONG" else "#f44336"
        ent = _fmt(t.get("entry")); sl_ = _fmt(t.get("sl")); tp_ = _fmt(t.get("tp"))
        rr_ = t.get("rr","?"); cf_ = t.get("confidence","?")
        reg = t.get("{{REGIME}}","?"); ext = t.get("{{EXTRA}}","?")
        rows += f"""<tr>
          <td>{ts}</td>
          <td style="color:{sc};font-weight:bold">{s}</td>
          <td>{ent}</td><td>{sl_}</td><td>{tp_}</td>
          <td>{rr_}</td><td>{cf_}</td>
          <td>{reg}</td><td>{ext}</td>
        </tr>"""

    log_rows = ""
    for e in reversed(logs[-15:]):
        t = e.get("ct_time",""); ty = e.get("type","")
        data = {k:v for k,v in e.items() if k not in ("timestamp","ct_time","type","trend_analysis","risk_analysis")}
        log_rows += f"<tr><td>{t} CT</td><td><b>{ty}</b></td><td style=\\'font-size:11px\\'>{json.dumps(data,default=str)[:120]}</td></tr>"

    last_ts = sig.get("timestamp","")[:19].replace("T"," ") if sig else "—"
    price   = sig.get("price","—")

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{EMOJI} {{SYM}} Dashboard</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;margin:0;padding:20px}}
h1{{color:{COLOR};margin:0 0 4px}}
.sub{{color:#888;font-size:13px;margin-bottom:20px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;min-width:150px}}
.card .val{{font-size:28px;font-weight:bold;color:{COLOR}}}
.card .lbl{{font-size:12px;color:#888;margin-top:4px}}
.sig-card{{background:#161b22;border:2px solid {COLOR};border-radius:8px;padding:16px 20px;margin-bottom:24px}}
.sig-card h3{{margin:0 0 8px;color:{COLOR}}}
table{{width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;overflow:hidden;margin-bottom:24px}}
th{{background:#21262d;color:#888;font-size:12px;text-transform:uppercase;padding:10px 12px;text-align:left}}
td{{padding:8px 12px;border-bottom:1px solid #21262d;font-size:13px}}
tr:last-child td{{border-bottom:none}}
</style></head><body>
<h1>{EMOJI} {{SYM}} Robot</h1>
<div class="sub">Live at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Last signal: {last_ts} | Price: {price}</div>
<div class="cards">
  <div class="card"><div class="val">{total}</div><div class="lbl">Total Signals</div></div>
  <div class="card"><div class="val">{longs}</div><div class="lbl">LONG</div></div>
  <div class="card"><div class="val">{shorts}</div><div class="lbl">SHORT</div></div>
  <div class="card"><div class="val">{flat_ct}</div><div class="lbl">FLAT</div></div>
  <div class="card"><div class="val">{avg_rr}</div><div class="lbl">Avg R:R</div></div>
  <div class="card"><div class="val">${avg_risk}</div><div class="lbl">Avg Risk/Trade</div></div>
</div>
<div class="sig-card">
  <h3>Latest Signal</h3>
  <span style="color:{sig_color};font-size:24px;font-weight:bold">{sig_label}</span>
  &nbsp;@ {_fmt(sig.get("entry"))} | SL: {_fmt(sig.get("sl"))} | TP: {_fmt(sig.get("tp"))}<br>
  <small>R:R: {sig.get("rr","—")} | Risk: {_fmt(sig.get("risk_usd"))} | Conf: {sig.get("confidence","—")}
  | {{REGIME_LABEL}}: {sig.get("{{REGIME}}","—")} | {{EXTRA_LABEL}}: {sig.get("{{EXTRA}}","—")}</small><br>
  <small style="color:#888">{str(sig.get("reasoning",""))[:200]}</small>
</div>
<h2 style="color:#888;font-size:14px;text-transform:uppercase;margin-bottom:8px">Recent Trades</h2>
<table><thead><tr>
  <th>Time</th><th>Signal</th><th>Entry</th><th>SL</th><th>TP</th>
  <th>R:R</th><th>Conf</th><th>{{REGIME_LABEL}}</th><th>{{EXTRA_LABEL}}</th>
</tr></thead><tbody>{rows}</tbody></table>
<h2 style="color:#888;font-size:14px;text-transform:uppercase;margin-bottom:8px">Live Log</h2>
<table><thead><tr><th>Time</th><th>Type</th><th>Data</th></tr></thead>
<tbody>{log_rows}</tbody></table>
</body></html>"""
    return html


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = _build_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, fmt, *args): pass


def start_dashboard(port=PORT):
    srv = HTTPServer(("0.0.0.0", port), Handler)
    print(f"{{SYM}} dashboard: http://localhost:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    start_dashboard()
'''

# ── run_*.bat ───────────────────────────────────────────────────────────────

BAT_TMPL = '''@echo off
REM {{SYM}} Robot Launcher
cd /d "%~dp0"

set CMD=%1
if "%CMD%"=="" set CMD=--report

if "%CMD%"=="--loop"      goto loop
if "%CMD%"=="--execute"   goto execute
if "%CMD%"=="--diagnose"  goto diagnose
if "%CMD%"=="--report"    goto report
if "%CMD%"=="--retrain"   goto retrain
if "%CMD%"=="--dashboard" goto dashboard
if "%CMD%"=="--stop"      goto stop

:usage
echo Usage: {{BAT}} [--loop^|--execute^|--diagnose^|--report^|--retrain^|--dashboard^|--stop]
goto end

:loop
echo Starting {{SYM}} robot (paper mode, loop)...
start "{{SYM}}_robot" python live_runner.py --loop
goto end

:execute
echo Starting {{SYM}} robot (LIVE EXECUTION)...
start "{{SYM}}_robot" python live_runner.py --loop --execute
goto end

:diagnose
python live_runner.py --diagnose
goto end

:report
python execution_tracker.py --report
goto end

:retrain
python retrainer.py
goto end

:dashboard
echo Starting {{SYM}} dashboard on port {{PORT}}...
start "{{SYM}}_dash" python dashboard.py
timeout /t 2 >nul
start http://localhost:{{PORT}}
goto end

:stop
echo Stopping {{SYM}} robot...
taskkill /FI "WINDOWTITLE eq {{SYM}}_robot" /F 2>nul
taskkill /FI "WINDOWTITLE eq {{SYM}}_dash"  /F 2>nul
echo Done.
goto end

:end
'''


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  CREATED: {path}")


def generate(sym, magic, port, color, emoji, regime, extra, extra_label, bat_name, comment):
    folder = os.path.join(BASE, sym)
    os.makedirs(folder, exist_ok=True)
    regime_label = regime.replace("_regime","").upper()

    def fill(tmpl):
        return (tmpl
            .replace("{{SYM}}", sym)
            .replace("{{MAGIC}}", str(magic))
            .replace("{{PORT}}", str(port))
            .replace("{{COLOR}}", color)
            .replace("{{EMOJI}}", emoji)
            .replace("{{REGIME}}", regime)
            .replace("{{EXTRA}}", extra)
            .replace("{{EXTRA_LABEL}}", extra_label)
            .replace("{{REGIME_LABEL}}", regime_label)
            .replace("{{BAT}}", bat_name)
        )

    # execution_tracker.py — skip if exists and is non-trivial
    tracker_path = os.path.join(folder, "execution_tracker.py")
    if not os.path.exists(tracker_path) or os.path.getsize(tracker_path) < 500:
        write_file(tracker_path, fill(TRACKER_TMPL))
    else:
        print(f"  SKIP (exists): {tracker_path}")

    write_file(os.path.join(folder, "retrainer.py"),        fill(RETRAINER_TMPL))
    write_file(os.path.join(folder, "dashboard.py"),         fill(DASHBOARD_TMPL))
    write_file(os.path.join(folder, bat_name),               fill(BAT_TMPL))


if __name__ == "__main__":
    print("Generating boilerplate files...")
    for args in INSTRUMENTS:
        sym = args[0]
        print(f"\n[{sym}]")
        generate(*args)
    print("\nAll done.")
