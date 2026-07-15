# SIEVE Performance Improvements: CPU to GPU Pipeline

## Problem

At 1M token context (LongMemEval, BEAM), the CPU-bound heuristic pipeline becomes the bottleneck. Processing thousands of conversation turns serially through spaCy/NLTK doesn't scale.

## Current bottlenecks

| Stage | Current approach | Time complexity | Why it's slow |
|-------|-----------------|-----------------|---------------|
| Retrieval | BM25 (CPU, sklearn) | O(V * Q) per query | Scores every turn against query serially |
| Memory object building | spaCy `nlp()` per turn | O(N) serial calls | 5k+ turns, each through full NLP pipeline |
| Evidence unit construction | NLTK tokenize + spaCy POS | O(K * L) | Runs on every retrieved passage |
| Proposal/evidence scoring | Word overlap heuristics | O(K * T) pairwise | No batching, token-level comparisons |
| Schema slot filling | Deterministic matching | O(K * S) | Regex + heuristic per slot per unit |

N = total turns, V = vocabulary, Q = query terms, K = retrieved passages, L = passage length, T = targets, S = slots

## Proposed GPU pipeline

### Stage 1: Dense retrieval (replaces BM25)

Encode all conversation turns once, query-time is a matmul.

```
Library:    sentence-transformers (all-MiniLM-L6-v2 or BGE-M3)
Index:      FAISS-GPU
Throughput: ~10k turns/sec encoding, sub-ms search
```

- Pre-encode entire conversation history on GPU in batches
- Query = single encode + cosine similarity against index
- Can retrieve top-100 instead of top-20 (cheap at this speed)
- Hybrid: dense + sparse (BM25) fusion via RRF for best recall

### Stage 2: Batched tokenization (replaces serial spaCy/NLTK)

```
Library:    HuggingFace tokenizers (Rust backend)
Speedup:    ~100x over NLTK word_tokenize
```

- Use `tokenizers` for raw tokenization (needed for budget counting, overlap)
- Keep spaCy only for POS/NER on the final selected passages (not all retrieved)
- Use `spacy.pipe()` with batching instead of single `nlp()` calls
- Move WordNet lemmatization to spaCy's built-in lemmatizer (one fewer dependency)

### Stage 3: Learned evidence scorer (replaces heuristic selection)

This is the key research contribution.

```
Model:      Cross-encoder (MiniLM-L6 or deberta-v3-small fine-tuned)
Input:      (query, evidence_passage) pairs
Output:     usefulness score [0, 1]
Training:   Use existing compilation cache as silver labels
```

- Batch all (query, candidate_passage) pairs, score on GPU in one forward pass
- Replaces: query target extraction, evidence unit role assignment, heuristic ranking, rescue passes
- Training data: the `data/compilation_cache/paper_sieve.json` already has query-evidence pairs with implicit labels (selected vs not selected)
- Can also train reader-adaptive: condition on reader model ID to predict what each reader needs

### Stage 4: Embedding-based slot filling (replaces regex/heuristic matching)

```
Model:      Same cross-encoder or a small extractive QA model
Input:      (slot_description, evidence_text)
Output:     extracted span + confidence
```

- Replaces the deterministic schema matching with learned extraction
- Handles edge cases that regex misses (paraphrases, implicit references)
- Falls back to deterministic for high-confidence pattern matches

## Architecture comparison

```
CURRENT (CPU, serial):
  1M tokens --> BM25 (CPU)
           --> top-20 --> spaCy per-turn (CPU, serial)
           --> heuristic selection (CPU, O(n^2))
           --> reader LLM

PROPOSED (GPU, batched):
  1M tokens --> dense retrieval (GPU, instant)
           --> top-100 --> learned scorer (GPU, batched)
           --> top-10 compiled evidence --> reader LLM
```

## Expected speedup

| Stage | Current | Proposed | Speedup |
|-------|---------|----------|---------|
| Retrieval (5k turns) | ~2-5s (BM25) | ~50ms (FAISS-GPU) | 40-100x |
| NLP processing (top-100) | ~10-20s (spaCy serial) | ~200ms (batched) | 50-100x |
| Evidence scoring | ~5-10s (heuristics) | ~100ms (cross-encoder) | 50-100x |
| Total per query | ~20-35s | ~500ms | 40-70x |

## Implementation plan

### Phase 1: Drop-in dense retrieval
- Add sentence-transformers + FAISS to requirements
- Write `retrieval/dense.py` alongside existing BM25
- A/B test: dense vs BM25 vs hybrid on LongMemEval recall@20
- No changes to compiler, pure retrieval swap

### Phase 2: Learned evidence scorer
- Extract training pairs from compilation cache
- Fine-tune cross-encoder on (query, passage, selected?) triples
- Replace `compiler/execution/selection.py` greedy loop with scorer
- Evaluate: does learned selection match or beat heuristic on strong readers?

### Phase 3: Reader-adaptive compilation
- Add reader model ID as a conditioning signal to the scorer
- Train on runs from multiple readers (Llama-8B, Llama-70B, GPT-4)
- Hypothesis: learned compiler discovers that strong readers need different evidence density
- This is the main conference contribution

### Phase 4: End-to-end optimization
- Joint training: retriever + compiler optimized for reader accuracy
- Distillation: use strong reader judgments to train the compiler
- Target: consistent improvement on both weak AND strong readers

## Dependencies to add

```
sentence-transformers>=3.0
faiss-gpu>=1.9  (or faiss-cpu for dev)
tokenizers>=0.21
```

## Evaluation

- LongMemEval: multi-session QA, 1M+ token histories
- BEAM: long-context benchmark
- Metrics: accuracy (LLM judge), latency per query, token efficiency
- Baselines: raw BM25 top-k, current SIEVE heuristic, LLM summarize
