# HotpotQA Experiment Results — Full Wikipedia Corpus

**Date**: 2026-03-18
**Model**: Qwen2.5-7B-Instruct
**Dataset**: HotpotQA Dev Validation (500 samples, seed=233)
**Retrieval Corpus**: TIGER-Lab/LongRAG Wikipedia (~5.2M articles, BM25)
**Max ReAct Steps**: 7
**BM25 Top-K**: 5

## Key Differences from Previous Experiments

- **Previous**: Used distractor setting (10 per-sample context paragraphs)
- **Current**: Uses full Wikipedia corpus (~5.2M articles) for open-domain retrieval
- **Original ReAct paper**: Uses live Wikipedia API; we approximate with offline BM25 over full Wikipedia
- **Sampling**: Follows original paper — shuffle with seed=233, take first 500

## Results Summary

| Method | EM (%) | F1 (%) | Avg Time/Sample (s) | Total Time |
|--------|--------|--------|---------------------|------------|
| Single Model (Direct QA) | N/A | N/A | N/A | N/A |
| RAG (Full Wiki) | N/A | N/A | N/A | N/A |
| ReAct (Full Wiki) | 28.00 | 39.19 | 14.8 | 7417s (2.1h) |
| ReAct-KV (none) | N/A | N/A | N/A | N/A |
| ReAct-KV (H2O) | N/A | N/A | N/A | N/A |
| ReAct-KV (SnapKV) | N/A | N/A | N/A | N/A |

## Analysis

### Comparison vs ReAct Baseline

| Method | EM Δ | F1 Δ |
|--------|------|------|
| ReAct (Full Wiki) | +0.00 | +0.00 |

## Methodology Notes

1. **Single Model**: Direct question → answer, no retrieval
2. **RAG**: Single BM25 retrieval pass → read → answer
3. **ReAct**: Multi-step interleaved reasoning with search/lookup/finish actions
4. **ReAct-KV (none)**: ReAct with KV cache reuse across steps (no pruning)
5. **ReAct-KV (H2O)**: ReAct-KV with Heavy Hitter Oracle pruning (keep_ratio=0.5)
6. **ReAct-KV (SnapKV)**: ReAct-KV with SnapKV attention pooling pruning

All ReAct variants use the same 6-shot prompt from the original ReAct paper (Yao et al., 2022).
