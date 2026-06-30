# Prop Firms & Challenges — Investigación para escalar a ~USD $3,000/mes

> Fecha: 2026-05-31 · Estado: investigación, **no ejecutado**.
> Objetivo: conseguir *buying power* para que el edge del portfolio produzca ~$3k/mes
> con el **mínimo capital propio en riesgo** (fee de challenge ~$500 vs. $100k+ de capital propio).
> ⚠️ Las fees y reglas cambian seguido — **verificá en la web de cada firma antes de pagar.**

---

## 1. Por qué prop firm (la ruta de menor capital)

`$/mes neto = capital × retorno_mensual% × profit_split`

Para **$3,000/mes neto** al split 80% → ~$3,750 brutos/mes:

| Capital fondeado | Retorno/mes requerido | Viabilidad |
|---|---|---|
| $25k  | ~15%/mes | Ruina segura |
| $50k  | ~7.5%/mes | Muy agresivo |
| **$100k** | **~3.75%/mes** | Plausible con edge real |
| $150k | ~2.5%/mes | Cómodo |
| $200k | ~1.9%/mes | Holgado |

**Capital propio en riesgo por la ruta prop: ~$500–$1,500** (fee de 1–3 intentos de challenge,
reembolsable en muchas firmas al primer payout) — vs. los $100k–$150k que necesitarías aportar tú mismo.
Esa es la respuesta a "menor cantidad de capital inicial".

---

## 2. Evidencia del propio backtest (cuenta $10k, 5 meses, dic-2025→may-2026)

Motor: `backtest_portfolio_10k.py` (mismas configs optimizadas que el sistema vivo; in-sample, sin slippage/comisión).

| Riesgo/trade | Política | Trades | P&L | Ret% | MaxDD$ | DD% | Sharpe | Calmar |
|---|---|---|---|---|---|---|---|---|
| 2.0% ($200) | NO-CAP | 757 | +$40,255 | +402% | -$3,300 | **-33%** | 2.64 | 12.2 |
| 2.0% ($200) | cluster_cap (vivo) | 579 | +$27,572 | +276% | -$2,900 | **-29%** | 2.37 | 9.5 |
| 0.3% ($30)  | NO-CAP | 766 | +$5,645 | +56% | -$615 | -6.2% | 2.44 | 9.2 |
| 0.3% ($30)  | cluster_cap (vivo) | 588 | +$3,743 | +37% | -$555 | **-5.5%** | 2.12 | 6.7 |

**Lectura clave:** el retorno y la DD escalan linealmente con el riesgo; Sharpe/WR/Calmar son invariantes.
- A 2%/trade el backtest es espectacular pero con **DD ~29-33%** → **incompatible** con cualquier prop firm
  (max DD típico 6-10%). Una racha correlacionada te liquida la cuenta.
- A **0.3%/trade** la DD baja a **~5.5%** (prop-compatible) y el sistema da ~$3,743/5mo ≈ **~$750/mes** sobre $10k.
- Para llegar a $3k/mes hay que **multiplicar el buying power** (no el riesgo): ~$100k-$150k fondeados a 0.3%/trade.

---

## 3. Comparativa de firmas (verificar fees/reglas vigentes)

| Firma | Fee challenge $100k (aprox) | EA / bots | Notas |
|---|---|---|---|
| [FTMO](https://ftmo.com) | ~$540 | Sí | Cuenta **Swing** permite holding de fin de semana/overnight sin restricción |
| [FundedNext](https://fundednext.com) | ~$499 | Sí | Varios modelos (Evaluation / Express / Stellar) |
| [E8 Markets](https://e8markets.com) | ~$488 | Sí | EA-friendly, reglas flexibles |
| [FXIFY](https://fxify.com) | ~$499 | Sí | Reglas configurables, multi-fase o 1-fase |
| [The5ers](https://the5ers.com) | varía | Sí | Orientada a swing/holding |
| [Apex Trader Funding](https://apextraderfunding.com) | (solo futuros CME) | Sí | **No aplica** a CFD/spot de MT5; sería para ZS/ZC reales en CME |

**Reglas típicas a chequear en cada una:**
- Profit target: 8-10% (fase 1), 4-5% (fase 2)
- Daily loss limit: 4-5%
- Max drawdown (total/trailing): 6-10%
- Profit split: 80-90% (suele subir con scaling)
- Regla de consistencia: ningún día > 30-50% de las ganancias totales
- Min días de trading, EA/news/weekend rules (varían por firma)

---

## 4. Restricciones que hoy nos descalifican (lo crítico)

1. **Riesgo por trade ~5x demasiado alto.** A 2%/trade la DD del backtest es ~29-33% → hay que
   bajar a **~0.3%/trade** para que la DD entre en la zona prop (~5.5%). Esto **no reduce el Sharpe**
   (invariante), solo el tamaño absoluto → por eso se necesita más buying power, no más riesgo.
2. **Fin de semana: ya NO es requisito duro.** El edge de finde en cripto **no dio resultados positivos
   en vivo**, así que dejó de ser una restricción para elegir firma. Esto **amplía** el universo de firmas
   viables (ya no obliga a FTMO Swing). → Operar solo en horario de semana es aceptable.
3. **Regla de consistencia.** Sistema multi-robot puede tener días de P&L grande puntual. cluster_cap
   ayuda a suavizar, pero hay que monitorear que ningún día supere el 30-50% del profit total.
4. **EA rules.** Confirmar que la firma permite ejecución 100% automatizada (EA) sin "manual trading requirement".

---

## 5. Brecha de preparación (honestidad)

- **Track record en vivo demasiado fino** (~50 paper trades repartidos). Se necesitan ~100-150 trades/robot
  para confiar estadísticamente. El backtest es el ancla **optimista** (in-sample, sin slippage/comisión/costo LLM,
  posible overfit a ventana mayormente alcista dic-may).
- Lo que se arriesga de verdad son las **fees de challenge** (~$500/intento), no $100k.

---

## 6. Plan recomendado (menor capital → $3k/mes)

1. **De-riskear a ~0.3%/trade** en `templates_nuevos_mercados/` y reconfirmar MaxDD < 6% (✓ backtest ya lo muestra: -5.5%).
2. **Acumular track record real** con ese riesgo hasta ~100-150 trades/robot; validar Sharpe OOS.
3. **Elegir firma EA-friendly** (FundedNext / E8 / FXIFY / FTMO). Finde ya no es requisito → más opciones.
4. **Pasar UN challenge $100k** (~$500). No múltiples a la vez hasta probar consistencia.
5. **En cuenta fondeada:** correr a 3-5%/mes objetivo → $3,750 brutos × 80% = ~$3,000 neto.
6. **Escalar:** usar scaling plans de la firma ($200k-$400k) para que $3k/mes sea holgado (~1-2%/mes).

**Costo de entrada realista: ~$500-$1,500** (1-3 intentos).

---

## 7. Fuentes

- [FTMO](https://ftmo.com) · [FundedNext](https://fundednext.com) · [E8 Markets](https://e8markets.com)
- [FXIFY](https://fxify.com) · [The5ers](https://the5ers.com) · [Apex Trader Funding](https://apextraderfunding.com)

> Comparadores y reseñas cambian seguido; cruzar siempre con la web oficial y los Términos vigentes de cada firma.
