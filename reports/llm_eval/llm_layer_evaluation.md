# Explanation layer evaluation (Module 13)

Language model backend in this run: **template** (the layer works with it switched off).

| Test | Result |
|---|---|
| 13.1 Faithfulness, rule-based text, 50 real windows | 100% have every number grounded |
| 13.1 Faithfulness, language model | not measured (no model configured) |
| Guardrail, a model that invents numbers | 100% caught and replaced |
| 13.2 Questions the data cannot answer | 10/10 declined with no figures |
| 13.3 Retrieval, top-1 / top-3 | 85% / 100% on 20 questions (optimistic: terms tuned on these) |
| 13.3 Answers citing a retrieved note | 100% |
| 13.5 Selective calls | 555 of 1309 windows (42%); 58% fewer calls than always-call |
| 13.5 Delay, rule-based answer | median 3 ms, p95 5 ms |
| 13.5 Delay, language model | not measured: no language model configured in this run |
| 13.6 Readability (Flesch), ML only vs ML + explanation | 63 vs 51 (proxy) |
| 13.4 Usefulness ratings from people | **not collected yet** |

## Where it fails or is unmeasured
- Usefulness (13.4) needs 3 to 5 raters; the sheet is generated but empty. The ablation (13.6) is therefore a readability proxy only.
- The number checker also accepts any whole number 0 to 20, 50 and 100, so an invented count in that range would pass.
- Retrieval questions were used to tune query expansion, so the retrieval score is optimistic.
- Language-model delay and quality are unmeasured here without an API key or a local model.

Fires by trigger: {'ML_TELEMETRY_ONLY': 0, 'HEAVY_FATIGUE_WARNING_THRESHOLD': 359, 'CRITICAL_EXHAUSTION_THRESHOLD_EXCEEDED': 139, 'SESSION_CONCLUDED_SUMMARY': 46, 'SHARP_FATIGUE_TREND_ACCELERATION': 11}
