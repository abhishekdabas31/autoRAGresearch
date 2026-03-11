# autoRAGresearch — Research Program

## Dataset
BeIR/SciFact — 5,183 scientific paper abstracts, 30 fixed eval queries with passage-level relevance labels.
SciFact abstracts are 150–250 words each. Naive 512-token chunks often span two abstracts.

## Objective
Maximize the composite RAGAS score on the 30-query eval set.
Current best score: (run `python eval.py` to measure baseline)

## Research Strategy
Start with the highest-leverage changes first:
1. Experiment with chunking strategies — SciFact abstracts are short, so chunk_size is critical
2. After chunking is stable, try hybrid retrieval (BM25 + dense)
3. Only try reranking after basic retrieval is tuned
4. Query expansion is high-cost — only try if retrieval precision is already high
5. Prompt engineering is free — try it any time

## Hypotheses to Explore (in rough priority order)
- Smaller chunks (256 or even 200) may align better with abstract boundaries, improving context_precision
- Paragraph chunking may naturally respect abstract boundaries
- Hybrid retrieval with alpha=0.3 may help with rare scientific terms (BM25 excels here)
- A SYSTEM_PROMPT that asks for evidence-based answers may improve faithfulness on scientific text
- Cross-encoder reranking (top-3) should boost context_precision
- Reducing chunk overlap to 0 when chunk_size fits within abstract length
- RETRIEVAL_TOP_K=3 may outperform top_k=5 if context_precision is low (less noise)

## Constraints
- Do not use any closed API (no OpenAI, no Anthropic) — local models only
- Keep eval runtime under 5 minutes
- Do not modify corpus_prep.py or eval.py
- One change per iteration — do not make multiple simultaneous changes
- If a direction fails 3 times in a row, abandon it and try something new

## What NOT to do
- Do not overfit to the eval set by hardcoding answers
- Do not reduce MAX_CONTEXT_LENGTH below 512
- Do not increase RETRIEVAL_TOP_K above 20 (latency will blow up)
- Do not remove the imports from corpus_prep.py
- Do not add external API calls to the pipeline
- Do not remove source_ids from the run_query return value
