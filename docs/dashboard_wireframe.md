# Dashboard wireframe (Module 12.1)

A text sketch of the pages, written before the Session lab was built. The existing five tabs are unchanged;
the Session lab is the sixth (key `6`). A person should still sketch it on paper and check it with a non-technical reader.

```
┌ sidebar ─────┐ ┌ top bar: page title · theme ───────────────────────────────────────────┐
│ Live monitor │ │ SESSION LAB                                                              │
│ Effort anal. │ │ ┌ controls ─────────────────────────────────────────────────────────────┐│
│ Drivers      │ │ │ Participant [P05 ▾]  Exercise [Internal rotation 30-40% ▾]  (Gen|Pers)││
│ Athlete prof.│ │ │ [▶ Play] [↺]  ────────●──────────────  0 s    [1×][4×][10×]           ││
│ AI coach     │ │ └───────────────────────────────────────────────────────────────────────┘│
│ ▶ Session lab│ │ ┌ Effort now ───────┐ ┌ Estimated vs self-reported ─────────────────────┐│
│              │ │ │     ╭──────╮      │ │ 20 ┤ ░░░ stop (18) ░░░░░░░░░░░░░░░░░░░░░ ___▁▃▅ ││
│ Athlete [▾]  │ │ │   ╱ gauge   ╲     │ │ 15 ┤ ▒▒▒ warning (15) ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒_▃▅▇     ││
│ ● connected  │ │ │      16.2         │ │ 11 ┤──────────────────▂▃▅▆▇                     ││
│ AI engine    │ │ │  [Fatigued] chip  │ │  6 ┤▁▂▃▄     own rating ── gen. ── pers. ‥‥     ││
│              │ │ │ own rating  16    │ │     0 s                                 100 s   ││
│              │ │ │ difference +0.2   │ └─────────────────────────────────────────────────┘│
│              │ │ │ Warning at 60 s   │                                                    │
│              │ │ └───────────────────┘                                                    │
│              │ │ ┌ Signals vs own start (%) ─────────────┐ ┌ Behind this estimate ───────┐│
│              │ │ │ [chips: pick signals]  line chart     │ │ rotator cuff speed  ▇▇ lowers││
│              │ │ │ muscle speed ↓, pulse ↑ …             │ │ pulse rate          ▇  raises││
│              │ │ └───────────────────────────────────────┘ │ DATA QUALITY: all clean      ││
│              │ │                                           │ PROFILE: age, sex, baseline  ││
│              │ │ ┌ Explanation ──────────────[Explain now]┐ └─────────────────────────────┘│
│              │ │ │ [GENERATED TEXT · rule-based]           │ ┌ Ask your data ─────────────┐│
│              │ │ │ "Effort is hard, 16.2 of 20 …"          │ │ [ question…      ][Look up]││
│              │ │ │ trigger · delay · numbers checked ✓     │ │ chips: example questions   ││
│              │ │ │ ▸ numbers allowed  ▸ sources  ▸ notice  │ │ [READ-ONLY LOOKUP] answer  ││
│              │ │ │ [Open report] [Download .md]            │ └────────────────────────────┘│
│              │ │ └─────────────────────────────────────────┘                                │
│              │ │ ▸ How accurate is this? Published results (tables)                         │
└──────────────┘ └──────────────────────────────────────────────────────────────────────────┘
```

Reading order: controls, then the gauge (how hard it is right now), then the chart (is the estimate tracking what the athlete
said), then why (signals and drivers), then words (explanation), then questions (ask your data), then the evidence.

Rules the design follows:

- Colour carries meaning only: green fine, amber warning (15), red stop (18).
- Generated text is always labelled, shown next to the numbers it was allowed to use, and never replaces them.
- The self-reported rating is shown for comparison and is never an input to the estimate.
- Nothing waits for the language model: the replay keeps running while an explanation is written.
- With the language model off, every panel works; the explanation is rule-based and says so.
- Phone width: the three grids collapse to one column and the controls wrap.
