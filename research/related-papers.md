# SIEVE: Related Papers

Organized bibliography for the SIEVE post-retrieval evidence compilation pipeline. Papers are grouped by the four active research directions; a paper may be listed once under its primary relevance.

---

## 1. Post-Retrieval Compression for RAG

These papers are the closest methodological neighbors to SIEVE's core idea: compress or filter retrieved passages before passing them to a reader LLM.

---

### RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
**Authors:** Fangyuan Xu, Weijia Shi, Eunsol Choi  
**Year:** 2023  
**Venue:** ICLR 2024  
**arXiv:** [2310.04408](https://arxiv.org/abs/2310.04408)  
**Relevance:** The most direct prior work: trains an extractive compressor (contrastive learning over sentences) and an abstractive compressor (distilled from a large LM) to reduce retrieved context to 5-10% of tokens before reader inference. SIEVE's schema-matching compiler is a domain-specific instantiation of the abstractive compression idea. RECOMP's training recipe (end-task signal + faithful summaries) is the template for a learned SIEVE compiler.

---

### LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
**Authors:** Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu  
**Year:** 2023  
**Venue:** EMNLP 2023  
**arXiv:** [2310.05736](https://arxiv.org/abs/2310.05736)  
**Relevance:** Introduces perplexity-based token pruning using a small proxy LM, achieving up to 20x compression with minimal quality loss. Establishes the benchmark for token-level hard compression that SIEVE's pipeline must beat (or complement) when handling conversational history passages.

---

### LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression
**Authors:** Huiqiang Jiang et al.  
**Year:** 2023 / ACL 2024  
**Venue:** ACL 2024  
**arXiv:** [2310.06839](https://arxiv.org/abs/2310.06839)  
**Relevance:** Extends LLMLingua to the long-context RAG regime by conditioning token importance on the query, then pruning irrelevant chunks before fine-grained token removal. Directly applicable to SIEVE's retrieval-then-compress pipeline on 1M+ token conversation histories; SIEVE's domain-schema approach is an alternative to this perplexity heuristic.

---

### ECoRAG: Evidentiality-guided Compression for Long Context RAG
**Authors:** Yeonseok Jeong, Jinsu Kim, Dohyeon Lee, Seung-won Hwang  
**Year:** 2025  
**Venue:** Findings of ACL 2025  
**arXiv:** [2506.05167](https://arxiv.org/abs/2506.05167)  
**Relevance:** Trains a compressor that explicitly filters on evidentiality (whether a passage supports the correct answer), which is exactly what SIEVE's schema-based extraction approximates heuristically. The evidentiality signal is a strong training objective candidate for a learned SIEVE compiler.

---

### EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation
**Authors:** Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, SeungYoon Han, Jong C. Park  
**Year:** 2024  
**Venue:** arXiv preprint  
**arXiv:** [2412.12559](https://arxiv.org/abs/2412.12559)  
**Relevance:** Extractive sentence classifier for RAG that preserves inter-sentence dependencies and adapts to query complexity. Outperforms LLMLingua and RECOMP on QA accuracy while reducing token count; parallels SIEVE's extractive compilation step and is an ablation baseline target.

---

### OSCAR: Online Soft Compression and Reranking
**Authors:** Maxime Louis et al. (NAVER LABS Europe)  
**Year:** 2025  
**Venue:** arXiv preprint  
**arXiv:** [2504.07109](https://arxiv.org/abs/2504.07109)  
**Relevance:** Combines soft (embedding-space) compression with reranking in a single online pass, achieving 2-5x speedup with no accuracy loss. Relevant to SIEVE's GPU-acceleration direction: soft compilation into compressed embeddings is an alternative to SIEVE's text-level schema extraction.

---

### SEER: Self-Aligned Evidence Extraction for Retrieval-Augmented Generation
**Authors:** Zhao et al.  
**Year:** 2024  
**Venue:** EMNLP 2024  
**arXiv:** [2410.11315](https://arxiv.org/abs/2410.11315)  
**Relevance:** Learns an evidence extractor using self-aligned training (no gold evidence labels required), reducing context by 9.25x while improving faithfulness and conciseness. The self-alignment training strategy is directly applicable to the learned SIEVE compiler direction, since SIEVE also lacks annotated extraction spans.

---

### RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
**Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning  
**Year:** 2024  
**Venue:** ICLR 2024  
**arXiv:** [2401.18059](https://arxiv.org/abs/2401.18059)  
**Relevance:** Builds a multi-level summary tree over a corpus at index time, enabling retrieval at multiple abstraction levels. Contrasts with SIEVE's query-time compilation: RAPTOR compresses offline, SIEVE compresses online per-query. Relevant to the scaling direction where offline compilation could pre-aggregate conversation history.

---

## 2. Conversational Memory / Long-Context QA Benchmarks

Benchmarks that define the task space SIEVE operates in.

---

### LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
**Authors:** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu  
**Year:** 2024  
**Venue:** ICLR 2025  
**arXiv:** [2410.10813](https://arxiv.org/abs/2410.10813)  
**Relevance:** The primary benchmark SIEVE is evaluated on. Provides 500 questions over simulated conversation histories of 115K tokens (small split) to 1.5M tokens (large split), covering information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. The 30-60% accuracy drop from oracle to full-history retrieval quantifies the problem SIEVE is solving.

---

### Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs (BEAM)
**Authors:** (Multiple authors)  
**Year:** 2025  
**Venue:** ICLR 2026  
**arXiv:** [2510.27246](https://arxiv.org/abs/2510.27246)  
**Relevance:** Pushes conversational memory evaluation to 1M and 10M token contexts with 100 conversations and 2,000 questions across 10 memory capability types. BEAM-1M and BEAM-10M are the scale targets for SIEVE's 1M+ scaling direction; the finding that structured memory consistently outperforms raw long-context LLMs motivates SIEVE's compilation approach.

---

### LoCoMo: Evaluating Long-Context Models on Long Conversations and More
**Authors:** Aditi Maharana et al.  
**Year:** 2024  
**Venue:** ACL 2024  
**arXiv:** (ACL Anthology: 2024.acl-long)  
**Relevance:** 50 multi-session conversations averaging 300 turns and 17K tokens per conversation, with 1,540 QA pairs across single-hop, multi-hop, temporal, and open-domain categories. Provides a smaller-scale but human-validated complement to LongMemEval and BEAM for evaluating SIEVE's retrieval and compilation quality.

---

## 3. Retrieval and Reranking

Sparse and dense retrieval methods that feed into or could replace SIEVE's BM25 retrieval stage.

---

### ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
**Authors:** Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia  
**Year:** 2021 / NAACL 2022  
**Venue:** NAACL 2022  
**arXiv:** [2112.01488](https://arxiv.org/abs/2112.01488)  
**Relevance:** Token-level multi-vector retrieval (late interaction) that achieves state-of-the-art quality at 6-10x lower storage than prior late-interaction models. The primary candidate for replacing SIEVE's BM25 retriever in the GPU-accelerated pipeline direction; ColBERT's per-token MaxSim can also serve as a learned relevance signal for the compiler.

---

### BGE M3-Embedding / BGE Reranker (FlagEmbedding)
**Authors:** BAAI (Beijing Academy of AI)  
**Year:** 2023-2024  
**Venue:** HuggingFace / technical reports  
**arXiv:** (bge-m3: 2402.03216; bge-reranker: HF model cards)  
**Relevance:** BGE-M3 provides multilingual dense retrieval with 1024-dim embeddings; BGE-reranker-v2 is a cross-encoder that jointly scores (query, passage) pairs for reranking. Both are drop-in replacements for BM25 in SIEVE's retrieval stage and are widely used as off-the-shelf baselines in RAG evaluation.

---

### BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval
**Authors:** Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, et al.  
**Year:** 2024  
**Venue:** arXiv preprint (under review)  
**arXiv:** [2407.12883](https://arxiv.org/abs/2407.12883)  
**Relevance:** 1,398 real-world queries where correct retrieval requires multi-step reasoning, not surface-level keyword matching. Relevant to SIEVE's retrieval quality: BM25 fails badly on reasoning-intensive queries, motivating learned dense retrieval in the GPU pipeline direction.

---

## 4. Reader-Adaptive RAG / Compression Conditioned on Reader Capability

Work that shows compression effects are not reader-neutral — directly motivating SIEVE's reader-adaptive compilation direction.

---

### Fixed RAG Compression Collapses Measured Reader Scaling
**Authors:** Sugam Panthi, Rabab Abdelfattah  
**Year:** 2025  
**Venue:** arXiv preprint  
**arXiv:** [2606.21807](https://arxiv.org/abs/2606.21807)  
**Relevance:** This is SIEVE's companion diagnostic paper. Across 20 readers and 10 domain-method settings, compression gain decreases with reader baseline (9/10 settings significant, p < 0.05): compression rescues weak readers by removing noise they cannot filter, and hurts strong readers by dropping details they would have used. Directly motivates SIEVE's reader-adaptive compilation direction and provides the ragscale toolkit for auditing.

---

### Enhancing RAG Efficiency with Adaptive Context Compression
**Authors:** (Multiple authors)  
**Year:** 2025  
**Venue:** Findings of EMNLP 2025  
**arXiv:** [2507.22931](https://arxiv.org/abs/2507.22931)  
**Relevance:** Proposes ACC-RAG, which dynamically adjusts compression ratio based on input complexity rather than applying a fixed ratio. Addresses the same interaction between content difficulty and compression that SIEVE faces in the reader-adaptive direction, though it focuses on query complexity rather than reader capability.

---

## 5. Fusion-in-Decoder and Multi-Document Reading

Seminal work on how reader architectures handle multiple retrieved passages.

---

### Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (FiD)
**Authors:** Gautier Izacard, Edouard Grave  
**Year:** 2021  
**Venue:** EACL 2021  
**arXiv:** [2007.01282](https://arxiv.org/abs/2007.01282)  
**Relevance:** Fusion-in-Decoder encodes each retrieved passage independently then fuses them in the decoder — the foundational architecture that demonstrated strong readers can integrate noisy multi-passage evidence effectively. SIEVE's finding that compression helps weak readers but not strong ones directly echoes FiD's advantage: strong readers (large decoders) already do implicit compilation.

---

*Last updated: July 2026*
