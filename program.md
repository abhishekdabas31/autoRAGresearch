# autoRAGresearch — Research Program

## Objective
Maximize the composite RAGAS score on the eval set.
Current best score: (run `python eval.py` to measure baseline)

## Research Strategy
Start with the highest-leverage changes first:
1. Experiment with chunking strategies before touching retrieval parameters
2. After chunking is stable, try hybrid retrieval (BM25 + dense)
3. Only try reranking after basic retrieval is tuned
4. Query expansion is high-cost — only try if retrieval precision is already high
5. Prompt engineering is free — try it any time

## Hypotheses to Explore (in rough priority order)
- Smaller chunks (256) may improve context_precision for factual queries
- Sentence-level chunking may improve faithfulness vs fixed-size chunking
- Paragraph chunking may preserve natural semantic boundaries
- Hybrid retrieval with alpha=0.3 may help with rare-term queries
- A more explicit SYSTEM_PROMPT with chain-of-thought may improve faithfulness
- Cross-encoder reranking (top-3) should boost context_precision
- Increasing CHUNK_OVERLAP to 100 may prevent information loss at boundaries
- HyDE may improve retrieval for abstract or conceptual queries
- Reducing GENERATION_TEMPERATURE to 0.0 may improve faithfulness

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
