"""US30 — Web Dashboard (port 8085)"""

import os, sys, json, threading
from datetime import datetime, date
from http.server import HTTPServer, BaseHTTPRequestHandler

_HERE = os.path.dirname(os.path.abspath(__file__))
_MVP_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _MVP_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import SYMBOL, ARTIFACTS_DIR, PAPER_LOG_FILE, LIVE_LOG_FILE, SIGNAL_FILE

PORT   = 8085
COLOR  = "#ff6f00"
EMOJI  = "🏭"


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
        reg = t.get("us30_regime","?"); ext = t.get("risk_on_off","?")
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
        log_rows += f"<tr><td>{t} CT</td><td><b>{ty}</b></td><td style=\'font-size:11px\'>{json.dumps(data,default=str)[:120]}</td></tr>"

    last_ts = sig.get("timestamp","")[:19].replace("T"," ") if sig else "—"
    price   = sig.get("price","—")

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{EMOJI} US30 Dashboard</title>
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
<h1>{EMOJI} US30 Robot</h1>
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
  | Régimen: {sig.get("us30_regime","—")} | Contexto: {sig.get("risk_on_off","—")}</small><br>
  <small style="color:#888">{str(sig.get("reasoning",""))[:200]}</small>
</div>
<h2 style="color:#888;font-size:14px;text-transform:uppercase;margin-bottom:8px">Recent Trades</h2>
<table><thead><tr>
  <th>Time</th><th>Signal</th><th>Entry</th><th>SL</th><th>TP</th>
  <th>R:R</th><th>Conf</th><th>Régimen</th><th>Contexto</th>
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
    print(f"US30 dashboard: http://localhost:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    start_dashboard()
