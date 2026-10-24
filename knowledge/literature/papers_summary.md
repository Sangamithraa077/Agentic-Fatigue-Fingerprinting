---
chunk_id: LIT-HYBRID-01
title: Selective RAG and Representation Before Retrieval in Clinical AI
category: literature
source: Bioengineering (2026) / medRxiv (2026)
date: 2026-02-15
licence: own summary, no text copied; check each paper's licence at its source before quoting
tags: [hybrid_ml_llm, rag, representation_before_retrieval, guardrails]
---
# Selective RAG and Representation Before Retrieval in Clinical AI

Recent literature establishes two foundational principles for explainable biomedical AI:

1. **Selective RAG-Enhanced Hybrid ML-LLM Framework (Bioengineering 2026):**
   ML handles numeric prediction with high temporal resolution and low computational cost. The LLM is invoked only selectively—when predefined triggers fire (e.g. fatigue index crossing warning thresholds, sudden trajectory shifts, or explicit user explanation requests).

2. **Representation Before Retrieval (medRxiv 2026):**
   Passing unstructured raw waveforms or uncurated text to an LLM produces hallucinations. Converting high-dimensional signals into structured, typed data summaries (such as JSON artifacts detailing baseline deviations, top SHAP features, and quality flags) prior to LLM prompting dramatically eliminates numerical hallucinations and ensures strict grounding.
