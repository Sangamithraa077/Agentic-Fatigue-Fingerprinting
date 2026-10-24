# End-to-end run with the hardware simulator (Modules 14.2 and 14.5)

Synthetic signals in the dataset's units and rates feed the live pipeline; this validates the software path, not accuracy on people.

| | LLM off | mock LLM | lying LLM |
|---|---|---|---|
| windows processed | 16 | 16 | 16 |
| first estimate at (s) | 20.0 | 20.0 | 20.0 |
| estimate vs simulated effort, r | 0.854 | 0.854 | 0.854 |
| window processing, mean / p95 / max (ms; budget 10000) | 104 / 151 / 162 | 99 / 144 / 157 | 83 / 138 / 146 |
| alerts | warning@60s, stop@80s | warning@60s, stop@80s | warning@60s, stop@80s |
| EMG dropout flagged in windows | [6, 7] | [6, 7] | [6, 7] |
| loop waited for explanations? | no | no | no |
| explanations (model / fell back) | 0 / 11 | 11 / 0 | 0 / 11 |
| fallback reasons | no_llm_configured | - | ungrounded_numbers |
| logged calls, all numbers grounded | 11/11 | 11/11 | 11/11 |
