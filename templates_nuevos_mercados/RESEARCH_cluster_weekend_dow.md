# Hallazgos: Cluster Cap, Edge de Fin de Semana y Riesgo por Día

**Fecha:** 2026-05-30
**Alcance:** robots cripto (BTCUSD, ETHUSD) y la política de `cluster_cap` del portfolio.
**Restricción:** todo en `templates_nuevos_mercados/`; no se tocó el sistema de soja ni los robots vivos.

---

## Pregunta de origen

¿Conviene **agregar más instrumentos cripto** que operen los fines de semana para "no perder días
sin operaciones"? Y, derivado de eso, ¿conviene cerrar una posición cripto abierta para liberar
otra mejor del mismo cluster, o ajustar el riesgo por día de la semana?

---

## 1. Política de `cluster_cap` — ¿el cap duro ayuda o estorba?

**Script:** `backtest_cluster_policy.py` · 759 trades, ~5 meses, configs optimizados.

| Política | Ops | Net P&L | Max DD | Sharpe | Calmar |
|----------|----:|--------:|-------:|-------:|-------:|
| Sin cap (todo libre) | 759 | +$3,970 | −$405 | 2.56 | 9.80 |
| **Cap duro (live actual)** | 580 | +$2,744 | **−$246** | 2.32 | **11.17** |
| Swap por RR | 580 | +$2,744 | −$246 | 2.32 | 11.17 |

**Conclusión:** el cap duro descarta ~24% de señales y deja ~$1,226 de P&L bruto sobre la mesa,
pero **corta el drawdown casi a la mitad** y da el mejor Calmar. Para una cuenta chica es la opción
correcta. **Mantener el cap duro.**

El swap por RR salió idéntico al cap duro porque **el RR es homogéneo dentro de cada cluster**
(equity ≈ 2.0, crypto ≈ 2.5, energy ≈ 3.0) → nunca se dispara. Un swap real necesitaría una señal
que *diferencie* trades (confidence del LLM), no RR.

---

## 2. Swap por confidence del LLM — no validable todavía

**Scripts:** `backtest_swap_confidence.py` (proxy técnico) · `confidence_analyzer.py` (datos vivos).

- El **proxy técnico de confidence** (separación EMA + momentum + dist. VWAP, normalizado por ATR)
  resultó **inversamente** predictivo: los trades de "mayor convicción técnica" ganan MENOS
  (Q4 33% WR vs Q1 41% WR). Señales técnicas más "ruidosas" ≠ mejores señales.
- En vivo solo hay ~47 trades acumulados, **casi todos MEDIUM** (44 MEDIUM, 3 HIGH, 0 LOW).
  Hint temprano: las 3 HIGH ganaron 3/3, pero muestra insuficiente.
- **Prerequisito:** acumular ~100-150 trades/robot etiquetados por confidence (coincide con el
  prerequisito del orquestador de portfolio) antes de poder validar/implementar un swap por
  confidence.
- **Monitoreo automático:** tarea de Windows `ConfidenceAnalyzer_Weekly` (domingos 18:00) corre
  `confidence_analyzer.py` y deja reportes en `confidence_reports/`. Su veredicto cambia solo
  cuando haya ≥15 HIGH y ≥15 LOW.

---

## 3. Edge de fin de semana (cripto) — REAL y robusto

**Script:** `backtest_weekend_crypto.py` · BTC+ETH, spread modelado.

Neto de spread (semana 5 bps / finde 15 bps = 3×):

| Bucket | Ops | WR% | Exp$/trade | Sharpe |
|--------|----:|----:|-----------:|-------:|
| Semana | 225 | 33.3% | $0.47 | 0.24 |
| **Finde** | 61 | 50.8% | **$5.08** | **2.43** |

**Sensibilidad:** el edge de finde sobrevive hasta ~**5× el spread de semana (25 bps)**, muy por
encima de lo típico de ICMarkets (~4-9 bps). Robusto.

**Mecanismo:** la estrategia es de **reversión/rango** (EMA+RSI+VWAP). Prospera en mercados quietos
(finde) y sufre en alta volatilidad (**miércoles** = peor día, −$3.25/trade, el pico de actividad
según la literatura). No contradice a la literatura: describe el *régimen*; nuestra estrategia es
de un tipo que se beneficia de ese régimen.

**Conclusión:** **mantener cripto operando los fines de semana.** Es el mejor período.

---

## 4. Riesgo por día de la semana — NO generaliza (no implementar)

**Script:** `backtest_dow_risk.py` · test out-of-sample (train 60% / test 40%).

**In-sample** (datos completos): ponderar por día casi duplica Sharpe (0.67 → 1.20). **Tentador
pero circular** (overfitting: ya sabíamos qué días fueron buenos en esta muestra).

**Out-of-sample** (pesos aprendidos en feb-2may, aplicados a 2-30may):

| En TEST set | Net P&L | Max DD | Sharpe |
|-------------|--------:|-------:|-------:|
| **Flat (actual)** | $121 | −$306 | **0.52** |
| Día-ponderado (aprendido) | $52 | −$366 | 0.19 |
| Solo finde 1.5× | $124 | −$361 | 0.48 |

**Conclusión:** el flat gana. Ponderar por día **empeora** el Sharpe OOS. Incluso el sobrepeso solo
de finde queda neutro-a-peor. **Mantener riesgo plano.**

**Distinción clave:** "el finde tiene mayor expectativa" (cierto, ya entra al equity con riesgo
plano) ≠ "subir el riesgo en finde mejora el portfolio" (falso: escala retorno *y* varianza/DD a la
par, y el patrón fino día-a-día es ruido inestable entre períodos).

---

## Decisiones finales

| Idea evaluada | Veredicto |
|---------------|-----------|
| Agregar más instrumentos cripto p/ findes | ❌ `cluster_cap` (1 slot) impide que sumen trades; edge ya capturado por BTC/ETH |
| Cerrar ETHUSD para liberar BTCUSD | ❌ No conviene; el cap protege drawdown correlacionado |
| Operar fines de semana (cripto) | ✅ Mantener — mejor período, robusto a spreads |
| Swap por RR dentro de cluster | ❌ RR homogéneo → nunca dispara |
| Swap por confidence del LLM | ⏸️ No validable aún; reactivar con ~100-150 trades/robot |
| Ponderar riesgo por día de la semana | ❌ No generaliza OOS; mantener riesgo plano |

**Acción neta: NO cambiar el setup cripto.** Está bien como está (riesgo plano, operando findes,
cluster_cap duro).

## Caveats generales
- Muestras chicas (test OOS: 115 trades, ~24 de finde).
- El backtest modela spread pero NO el efecto de **barras delgadas** de finde (rangos high/low
  comprimidos → puede subestimar toques de SL → WR de finde quizás algo inflado).
- Revisar cuando se acumule más historial vivo.

## Scripts (en `templates_nuevos_mercados/`)
- `backtest_cluster_policy.py` — comparación de políticas de cluster_cap
- `backtest_swap_confidence.py` — swap por proxy de confidence (exploratorio)
- `confidence_analyzer.py` — WR por confidence sobre trades vivos (+ tarea semanal)
- `backtest_weekend_crypto.py` — finde vs semana, con modelo de spread
- `backtest_dow_risk.py` — ponderación de riesgo por día con test OOS
