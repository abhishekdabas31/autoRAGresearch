"""
setup_data.py — One-time data generation

Creates the default corpus (text files in data/corpus/) and evaluation set
(data/eval_set.json). Run once before using autoRAGresearch.

Usage:
    python setup_data.py
"""

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"
EVAL_SET_PATH = Path(__file__).parent / "data" / "eval_set.json"

# ============================================================
# CORPUS: ~40 articles on Transformers and modern LLMs
# Each article is a focused, information-dense passage.
# ============================================================

ARTICLES = {
    "transformer_architecture": """\
The Transformer Architecture

The Transformer is a deep learning architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google Brain. Unlike recurrent neural networks (RNNs) that process tokens sequentially, the Transformer processes all positions in a sequence simultaneously through self-attention mechanisms. This parallelism makes Transformers significantly faster to train on modern GPU hardware.

The architecture consists of an encoder and a decoder, each composed of stacked layers. Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network. Each decoder layer adds a third sub-layer for cross-attention over the encoder output. Residual connections and layer normalization are applied around each sub-layer.

The Transformer introduced the scaled dot-product attention formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V, where Q, K, and V are query, key, and value matrices, and d_k is the dimension of the keys. This scaling factor prevents the dot products from growing too large, which would push the softmax into regions with extremely small gradients.

The original Transformer used 6 encoder layers and 6 decoder layers, with a model dimension of 512 and 8 attention heads. It was trained on the WMT 2014 English-to-German and English-to-French translation tasks and achieved state-of-the-art BLEU scores while requiring significantly less training time than previous models.\
""",

    "self_attention": """\
Self-Attention Mechanism

Self-attention, also called intra-attention, is the core mechanism of the Transformer architecture. It allows each position in a sequence to attend to all other positions, computing a weighted representation where the weights reflect relevance. Unlike convolutional layers that have a fixed receptive field, self-attention has a global receptive field from the first layer.

In self-attention, each input token is projected into three vectors: a query (Q), a key (K), and a value (V) using learned weight matrices. The attention score between any two positions is computed as the dot product of the query of one position with the key of another, scaled by the square root of the dimension. These scores are passed through a softmax function to obtain attention weights, which are then used to compute a weighted sum of the value vectors.

Self-attention has O(n^2) complexity with respect to sequence length n, since every position attends to every other position. This quadratic scaling is the main computational bottleneck for processing very long sequences and has motivated research into efficient attention variants such as linear attention, sparse attention, and Flash Attention.

The ability of self-attention to directly model long-range dependencies without the vanishing gradient problem of RNNs is considered the key advantage that made Transformers successful across NLP, computer vision, and other domains.\
""",

    "multi_head_attention": """\
Multi-Head Attention

Multi-head attention is an extension of the basic attention mechanism that allows the model to jointly attend to information from different representation subspaces at different positions. Instead of computing a single attention function, multi-head attention runs multiple attention operations in parallel, each with its own learned projection matrices.

Formally, given h attention heads, the input is projected h times with different learned linear projections for queries, keys, and values. Each head operates on a lower-dimensional subspace (d_model / h dimensions). The outputs of all heads are concatenated and projected through a final linear layer: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O.

In the original Transformer, 8 attention heads were used with d_model = 512, giving each head a dimension of 64. Different heads learn to attend to different types of relationships: some heads might focus on syntactic dependencies, others on semantic similarity, and others on positional proximity. This diversity of attention patterns is crucial for the model's representational power.

Multi-head attention is used in three ways in the Transformer: encoder self-attention (each encoder position attends to all encoder positions), decoder self-attention (each decoder position attends to previous decoder positions), and encoder-decoder cross-attention (each decoder position attends to all encoder positions).\
""",

    "positional_encoding": """\
Positional Encoding

Since the Transformer architecture processes all tokens in parallel without any inherent notion of order, positional encodings are added to the input embeddings to provide information about the position of each token in the sequence. Without positional encoding, the Transformer would treat the input as a bag of tokens with no sequential structure.

The original Transformer uses sinusoidal positional encodings. For each position pos and each dimension i, the encoding is: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)). This scheme was chosen because it allows the model to learn to attend to relative positions, since PE(pos+k) can be expressed as a linear function of PE(pos) for any fixed offset k.

Later work introduced learned positional encodings, where position embeddings are trained parameters rather than fixed functions. BERT and GPT use learned absolute position embeddings. More recent models use relative positional encodings such as RoPE (Rotary Position Embedding) used in LLaMA, and ALiBi (Attention with Linear Biases) which adds a linear bias to attention scores based on the distance between positions.

RoPE encodes position by rotating the query and key vectors in pairs of dimensions, which naturally encodes relative position information in the dot product attention scores. This approach generalizes better to sequence lengths not seen during training.\
""",

    "layer_normalization": """\
Layer Normalization

Layer normalization is a technique used in Transformers to stabilize training and improve convergence. Unlike batch normalization, which normalizes across the batch dimension, layer normalization normalizes across the feature dimension for each individual example. This makes it independent of batch size and well-suited for sequence models where batch sizes may vary.

For a given input vector x, layer normalization computes: LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta, where gamma and beta are learned scale and shift parameters, and epsilon is a small constant for numerical stability.

In the original Transformer, layer normalization is applied after each sub-layer (post-norm): output = LayerNorm(x + Sublayer(x)). However, subsequent research found that pre-norm placement, where normalization is applied before the sub-layer, leads to more stable training especially for deep models: output = x + Sublayer(LayerNorm(x)). GPT-2 and most modern LLMs use pre-norm.

RMSNorm (Root Mean Square Layer Normalization) is a simplified variant used in LLaMA and other recent models. It removes the mean centering step: RMSNorm(x) = gamma * x / sqrt(mean(x^2) + epsilon). This is computationally cheaper while performing comparably to standard layer normalization.\
""",

    "feed_forward_networks": """\
Feed-Forward Networks in Transformers

Each Transformer layer contains a position-wise feed-forward network (FFN) applied independently to each position. The FFN consists of two linear transformations with a nonlinear activation in between: FFN(x) = W_2 * activation(W_1 * x + b_1) + b_2. In the original Transformer, the inner dimension is 2048 (4 times the model dimension of 512) and the activation function is ReLU.

The FFN is where most of the model's parameters reside. In a standard Transformer layer, the FFN parameters outnumber the attention parameters by a factor of about 2-3x. Research suggests that the FFN layers serve as key-value memories that store factual knowledge learned during pre-training.

Modern Transformers often use different activation functions in the FFN. GELU (Gaussian Error Linear Unit) is used in BERT and GPT models. SwiGLU, a gated variant combining Swish activation with a Gated Linear Unit, is used in LLaMA and PaLM. The gated variants have been shown to improve performance without additional computational cost when the inner dimension is adjusted.

The choice of FFN inner dimension affects the model's capacity. The standard ratio of 4x the model dimension is widely used, but some architectures like Chinchilla use different ratios. Mixture of Experts (MoE) architectures replace the dense FFN with a sparse set of expert FFNs, activating only a subset for each token.\
""",

    "bert": """\
BERT: Bidirectional Encoder Representations from Transformers

BERT, introduced by Google in 2018, is a pre-trained language model based on the Transformer encoder architecture. BERT's key innovation is bidirectional pre-training: unlike previous models that read text left-to-right or right-to-left, BERT reads the entire sequence simultaneously, allowing each token to attend to both left and right context.

BERT is pre-trained with two objectives. Masked Language Modeling (MLM) randomly masks 15% of input tokens and trains the model to predict them from context. Next Sentence Prediction (NSP) trains the model to predict whether two sentences appear consecutively in the original text. Later research showed that NSP provides minimal benefit and can be removed.

BERT-base has 12 layers, 768 hidden dimensions, 12 attention heads, and 110 million parameters. BERT-large scales to 24 layers, 1024 hidden dimensions, 16 attention heads, and 340 million parameters. Both models are pre-trained on the BooksCorpus (800M words) and English Wikipedia (2,500M words).

BERT established the pre-train-then-fine-tune paradigm in NLP. After pre-training, BERT is fine-tuned on downstream tasks by adding a task-specific output layer. It achieved state-of-the-art results on 11 NLP benchmarks including GLUE, SQuAD question answering, and named entity recognition. BERT's success demonstrated that large-scale unsupervised pre-training captures rich linguistic knowledge.\
""",

    "gpt_series": """\
GPT: Generative Pre-trained Transformer Series

The GPT series from OpenAI uses the Transformer decoder architecture for autoregressive language modeling. GPT-1 (2018) demonstrated that generative pre-training on a large corpus followed by discriminative fine-tuning achieves strong results on diverse NLP tasks. It had 117 million parameters and was trained on the BooksCorpus.

GPT-2 (2019) scaled to 1.5 billion parameters and was trained on WebText, a dataset of 8 million web pages. GPT-2 showed that language models can perform tasks zero-shot, without any task-specific fine-tuning, by framing tasks as text generation. OpenAI initially withheld the full model due to concerns about misuse.

GPT-3 (2020) scaled dramatically to 175 billion parameters and introduced few-shot learning via in-context learning. By providing a few examples in the prompt, GPT-3 could perform tasks it was never explicitly trained for. The paper showed that performance scales predictably with model size, dataset size, and compute.

GPT-4 (2023) is a multimodal model accepting both text and image inputs. While OpenAI did not disclose architectural details, GPT-4 demonstrated significant improvements in reasoning, factual accuracy, and instruction following compared to GPT-3.5. It achieves human-level performance on many professional and academic benchmarks.\
""",

    "tokenization": """\
Tokenization in Language Models

Tokenization is the process of converting raw text into a sequence of tokens that a language model can process. Modern LLMs use subword tokenization algorithms that split text into a vocabulary of common subword units, balancing between character-level and word-level representations.

Byte-Pair Encoding (BPE), originally a data compression algorithm, was adapted for NLP by Sennrich et al. in 2016. BPE starts with a character-level vocabulary and iteratively merges the most frequent adjacent pairs of tokens until a desired vocabulary size is reached. GPT-2, GPT-3, and GPT-4 use BPE tokenization. GPT-2's tokenizer has a vocabulary of 50,257 tokens.

WordPiece is a similar subword algorithm used by BERT. It differs from BPE in that it selects merges based on the likelihood of the training data rather than raw frequency. WordPiece starts with characters and greedily adds subwords that maximize the likelihood of the corpus. BERT's vocabulary contains 30,522 WordPiece tokens.

SentencePiece by Google treats the input as a raw stream of Unicode characters and learns a subword vocabulary without requiring pre-tokenization into words. This makes it language-agnostic and particularly useful for multilingual models. T5 and LLaMA use SentencePiece tokenization. The choice of tokenizer and vocabulary size significantly affects model efficiency: a larger vocabulary reduces sequence length but increases embedding table size.\
""",

    "transfer_learning_nlp": """\
Transfer Learning in NLP

Transfer learning in NLP refers to the practice of pre-training a model on a large, general-purpose text corpus and then fine-tuning it on a specific downstream task. This approach leverages the linguistic knowledge captured during pre-training and dramatically reduces the amount of task-specific labeled data needed.

The pre-train-then-fine-tune paradigm was popularized by ELMo (2018), which used bidirectional LSTMs, and then revolutionized by BERT (2018) and GPT (2018) using Transformers. The key insight is that language models trained on massive text corpora develop rich internal representations of syntax, semantics, and even world knowledge that transfer well to diverse tasks.

Fine-tuning typically involves adding a task-specific head (such as a classification layer) on top of the pre-trained model and training the entire model end-to-end on labeled data for the target task. Learning rates for fine-tuning are typically much smaller than for pre-training (e.g., 2e-5 vs 1e-4) to avoid catastrophically forgetting pre-trained knowledge.

Parameter-efficient fine-tuning methods like LoRA (Low-Rank Adaptation), adapters, and prefix tuning modify only a small number of parameters while keeping the base model frozen. LoRA adds trainable low-rank matrices to attention layers, typically training less than 1% of the original parameters while achieving comparable performance to full fine-tuning.\
""",

    "scaling_laws": """\
Scaling Laws for Neural Language Models

Scaling laws describe the empirical relationship between model performance and three key factors: model size (number of parameters), dataset size (number of tokens), and compute budget (FLOPs). Research by Kaplan et al. at OpenAI (2020) showed that loss scales as a power law with each of these factors.

The key findings were: performance improves smoothly and predictably as models get larger; larger models are more sample-efficient, requiring fewer tokens to reach the same loss; and for a fixed compute budget, there is an optimal allocation between model size and data size.

The Chinchilla paper by Hoffmann et al. at DeepMind (2022) refined these scaling laws. They found that previous models like GPT-3 were significantly undertrained relative to their size. Chinchilla, with 70 billion parameters trained on 1.4 trillion tokens, outperformed the 280 billion parameter Gopher trained on 300 billion tokens. The Chinchilla-optimal ratio is approximately 20 tokens per parameter.

These scaling laws have profoundly influenced LLM development. They allow researchers to predict performance before training, optimize compute allocation, and make informed decisions about model architecture. The laws also suggest that performance improvements from scaling alone will eventually plateau, motivating research into more efficient architectures and training methods.\
""",

    "emergent_abilities": """\
Emergent Abilities of Large Language Models

Emergent abilities are capabilities that appear in large language models but are absent in smaller models. They represent qualitative changes in model behavior that arise from quantitative scaling. The term was popularized by Wei et al. (2022) in a survey of behaviors that emerge at specific scale thresholds.

Examples of emergent abilities include: multi-step arithmetic (emerging around 100B parameters), chain-of-thought reasoning (enabling complex multi-step problem solving), code generation, analogical reasoning, and the ability to follow complex instructions. These abilities appear suddenly rather than improving gradually with scale.

Few-shot learning is considered an emergent ability. GPT-3 (175B parameters) demonstrated that providing a few examples in the prompt enables the model to perform new tasks without any parameter updates. Smaller models in the GPT-3 family showed little to no few-shot capability, while GPT-3 showed dramatic improvement.

Some researchers have questioned whether emergent abilities are truly sudden or whether they appear sudden due to the choice of evaluation metrics. Schaeffer et al. (2023) argued that with appropriate metrics, many seemingly emergent abilities actually improve smoothly with scale. Regardless, the practical impact is clear: sufficiently large models can perform tasks that smaller models cannot, and predicting which abilities will emerge at which scale remains an open research problem.\
""",

    "instruction_tuning": """\
Instruction Tuning

Instruction tuning is the process of fine-tuning a pre-trained language model on a dataset of (instruction, response) pairs to improve its ability to follow human instructions. This technique bridges the gap between the next-token-prediction objective used in pre-training and the instruction-following behavior users expect.

FLAN (Fine-tuned Language Net) by Google (2022) demonstrated that instruction tuning on a diverse set of tasks framed as instructions dramatically improves zero-shot performance on unseen tasks. FLAN-T5 and FLAN-PaLM showed consistent gains across model scales.

InstructGPT by OpenAI (2022) combined instruction tuning with reinforcement learning from human feedback (RLHF). Human labelers wrote demonstrations of desired behavior and ranked model outputs, creating training signals that align the model with human preferences. The resulting model was preferred by humans despite being much smaller than GPT-3.

Modern instruction-tuned models like LLaMA-Chat, Vicuna, and Mistral Instruct are typically created in multiple stages: first pre-training on large text corpora, then supervised fine-tuning on curated instruction data, and optionally RLHF or DPO (Direct Preference Optimization) for further alignment. The quality and diversity of instruction data are often more important than quantity.\
""",

    "rlhf": """\
Reinforcement Learning from Human Feedback (RLHF)

RLHF is a technique for aligning language models with human preferences by training a reward model from human feedback and using it to fine-tune the language model via reinforcement learning. RLHF was a key component in training ChatGPT and InstructGPT.

The RLHF pipeline has three stages. First, supervised fine-tuning (SFT) trains the model on high-quality demonstration data written by human labelers. Second, reward modeling collects comparison data where labelers rank model outputs from best to worst, and a reward model is trained to predict these human preferences. Third, the language model is optimized against the reward model using Proximal Policy Optimization (PPO).

The reward model learns a scalar score for any (prompt, response) pair that correlates with human preferences. During PPO training, the language model generates responses and receives reward signals from the reward model. A KL-divergence penalty prevents the policy from diverging too far from the SFT model, maintaining response quality and diversity.

Direct Preference Optimization (DPO), introduced by Rafailov et al. in 2023, simplifies RLHF by directly optimizing the language model on preference pairs without training a separate reward model. DPO reparameterizes the reward model as a function of the language model itself, reducing training complexity while achieving comparable or better alignment results.\
""",

    "chain_of_thought": """\
Chain-of-Thought Prompting

Chain-of-thought (CoT) prompting is a technique where the model is encouraged to generate intermediate reasoning steps before producing a final answer. Introduced by Wei et al. (2022), CoT prompting significantly improves performance on arithmetic, commonsense, and symbolic reasoning tasks.

In standard prompting, the model receives a question and generates an answer directly. In CoT prompting, the few-shot examples include step-by-step reasoning. For example, instead of "Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. A: 11", CoT uses "Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. A: He bought 2 * 3 = 6 balls. So he has 5 + 6 = 11 balls. The answer is 11."

Zero-shot CoT, discovered by Kojima et al. (2022), shows that simply adding "Let's think step by step" to the prompt triggers chain-of-thought reasoning without any examples. This suggests that the reasoning capability is latent in large language models and can be elicited with minimal prompting.

CoT reasoning is considered an emergent ability — it primarily benefits models above approximately 100 billion parameters. Smaller models tend to produce incoherent reasoning chains. Self-consistency, introduced by Wang et al. (2023), samples multiple reasoning paths and selects the most common answer, further improving CoT performance.\
""",

    "in_context_learning": """\
In-Context Learning

In-context learning (ICL) is the ability of large language models to learn tasks from examples provided in the input prompt, without any gradient updates to the model's parameters. This capability was first systematically studied in GPT-3, where providing a few input-output examples in the prompt enabled the model to perform diverse tasks.

ICL operates through three regimes. Zero-shot learning provides only a task description. Few-shot learning includes several (typically 1-32) input-output examples. Many-shot learning provides hundreds or thousands of examples within the context window.

The mechanism behind ICL is not fully understood. One hypothesis is that ICL performs implicit Bayesian inference, using the examples to identify the task and then applying knowledge learned during pre-training. Another view, supported by Akyurek et al. (2023), suggests that Transformer attention layers implement mesa-optimization, essentially running a learning algorithm (similar to gradient descent) within their forward pass.

The effectiveness of ICL depends on several factors: the order of examples matters (different orderings can cause up to 30% variance in accuracy), the format of examples affects performance, and the choice of examples (selecting examples similar to the test query) can dramatically improve results. ICL is sensitive to surface-level features like label distribution and prompt formatting.\
""",

    "retrieval_augmented_generation": """\
Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that enhances language model outputs by first retrieving relevant documents from an external knowledge base and then conditioning the generation on both the query and the retrieved context. RAG was introduced by Lewis et al. at Facebook AI Research in 2020.

The RAG pipeline consists of three stages: indexing, retrieval, and generation. During indexing, documents are chunked, embedded using a dense encoder, and stored in a vector database. At inference time, the query is embedded and used to retrieve the top-k most similar document chunks. These chunks are prepended to the prompt as context for the language model to generate a grounded answer.

RAG addresses several limitations of standalone LLMs. It reduces hallucination by grounding responses in retrieved evidence. It allows the model to access up-to-date information beyond its training cutoff. It makes the model's knowledge source transparent and verifiable. And it enables domain-specific QA without fine-tuning the model.

Key hyperparameters in RAG include chunk size, overlap between chunks, the number of documents retrieved (top-k), the embedding model, and the generation prompt. Advanced RAG techniques include hybrid retrieval combining dense and sparse methods, cross-encoder reranking of retrieved documents, query expansion, and Hypothetical Document Embedding (HyDE) which generates a hypothetical answer to use as the retrieval query.\
""",

    "vector_databases": """\
Vector Databases and Similarity Search

Vector databases are specialized storage systems designed for efficient similarity search over high-dimensional embedding vectors. They are a critical component of RAG systems, enabling fast nearest-neighbor retrieval from potentially millions of document embeddings.

FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search. It supports multiple index types: Flat indices perform exact brute-force search, IVF (Inverted File) indices partition the vector space into clusters for approximate search, and HNSW (Hierarchical Navigable Small World) indices build a graph structure for fast approximate nearest neighbor lookup. FAISS supports both CPU and GPU computation.

The choice of similarity metric matters. Cosine similarity measures the angle between vectors and is most common for normalized embeddings. Euclidean (L2) distance measures the straight-line distance and is sensitive to vector magnitude. Inner product (dot product) combines magnitude and direction and is used when vector norms carry semantic meaning.

Other vector database solutions include Pinecone (managed cloud service), Weaviate (open-source with hybrid search), Chroma (lightweight, optimized for RAG), and Milvus (distributed, scalable). The choice depends on scale, latency requirements, and whether hybrid search (combining dense vectors with sparse keyword matching) is needed.\
""",

    "embedding_models": """\
Embedding Models for NLP

Embedding models convert text into dense vector representations that capture semantic meaning, enabling similarity-based retrieval and comparison. The quality of embeddings directly determines the quality of retrieval in RAG systems.

Early word embeddings like Word2Vec (2013) and GloVe (2014) produced static embeddings where each word has a single vector regardless of context. BERT introduced contextualized embeddings where the same word gets different vectors depending on surrounding context.

Sentence-transformers, based on Siamese BERT networks, are specifically trained for producing high-quality sentence and paragraph embeddings. The training uses contrastive learning objectives that pull semantically similar texts closer in embedding space. Popular models include all-MiniLM-L6-v2 (fast, good quality), nomic-embed-text-v1.5 (strong open-source embedding model), and BGE (BAAI General Embedding) models.

The MTEB (Massive Text Embedding Benchmark) evaluates embedding models across diverse tasks including retrieval, classification, clustering, and semantic similarity. Embedding dimension (typically 384-1536) affects the trade-off between representational capacity and computational cost. Matryoshka representation learning allows a single model to produce embeddings that work well at multiple dimensions, enabling flexible deployment.\
""",

    "bm25_sparse_retrieval": """\
BM25 and Sparse Retrieval

BM25 (Best Matching 25) is a ranking function used in information retrieval that scores documents based on term frequency and document length. It is the most widely used sparse retrieval method and remains competitive with dense retrieval for many tasks, particularly when queries contain rare or specific terms.

BM25 computes a relevance score based on three components: term frequency (TF) with diminishing returns, inverse document frequency (IDF) to weight rare terms higher, and document length normalization to avoid biasing toward longer documents. The formula is parameterized by k1 (controls TF saturation, typically 1.2-2.0) and b (controls length normalization, typically 0.75).

Unlike dense retrieval which matches based on semantic similarity in embedding space, sparse retrieval matches based on exact lexical overlap. This makes BM25 particularly effective for queries with rare technical terms, proper nouns, or specific codes that dense models might not represent well in their embedding space.

Hybrid retrieval combines dense and sparse methods, typically using a weighted sum of normalized scores. Research consistently shows that hybrid retrieval outperforms either method alone. The weighting parameter alpha controls the balance: higher alpha favors dense retrieval (better for semantic matching) while lower alpha favors sparse retrieval (better for keyword matching). Reciprocal Rank Fusion is an alternative combination method that doesn't require score normalization.\
""",

    "cross_encoder_reranking": """\
Cross-Encoder Reranking

Cross-encoder reranking is a retrieval technique where an initial set of candidate documents retrieved by a fast first-stage retriever (like BM25 or dense retrieval) is re-scored by a more expensive but more accurate cross-encoder model. This two-stage approach balances efficiency with retrieval quality.

Unlike bi-encoders that independently embed the query and document, cross-encoders jointly process the query-document pair through a single Transformer. This allows full token-level interaction between the query and document, capturing fine-grained relevance signals that bi-encoders miss. Cross-encoders typically achieve 5-15% higher accuracy but are 100-1000x slower than bi-encoders.

Popular cross-encoder models include ms-marco-MiniLM-L-6-v2 (fast, trained on MS MARCO passage ranking), BGE-reranker-v2-m3 (multilingual, strong accuracy), and Cohere Rerank (commercial API). The choice of reranker depends on the trade-off between latency and accuracy requirements.

In a RAG pipeline, reranking works as follows: retrieve top-k candidates (typically 20-50) using a fast retriever, then re-score all candidates with the cross-encoder, and finally keep the top-n (typically 3-5) highest-scoring documents for the generation step. The reranking step adds latency but significantly improves context quality, which directly impacts the faithfulness and accuracy of generated answers.\
""",

    "hyde": """\
HyDE: Hypothetical Document Embedding

Hypothetical Document Embedding (HyDE) is a retrieval technique introduced by Gao et al. (2022) that uses a language model to generate a hypothetical answer to the query, then uses the embedding of this hypothetical document as the retrieval query instead of the original query embedding.

The intuition behind HyDE is that a hypothetical answer, even if inaccurate, will be more lexically and semantically similar to the actual relevant documents than the short query itself. This bridges the gap between the query space and the document space, improving retrieval recall especially for abstract or conceptual queries.

The HyDE pipeline works as follows: given a user query, prompt the LLM to generate a hypothetical passage that would answer the query. Embed this generated passage using the same embedding model used for the document index. Use the hypothetical document embedding to retrieve actual documents from the index. Pass the retrieved real documents (not the hypothetical one) to the LLM for final answer generation.

HyDE is most effective when queries are abstract or when there is a significant vocabulary mismatch between how users phrase questions and how information is stated in the corpus. However, it adds latency (one extra LLM call) and can hurt performance when the generated hypothetical document is misleading. HyDE works best when combined with other retrieval strategies like hybrid search.\
""",

    "knowledge_distillation": """\
Knowledge Distillation

Knowledge distillation is a model compression technique where a smaller student model is trained to mimic the behavior of a larger teacher model. Introduced by Hinton et al. (2015), the key insight is that the soft probability distribution output by the teacher contains more information than hard labels, including relationships between classes.

In the standard distillation setup, the student is trained on a weighted combination of two losses: the cross-entropy with the hard labels and the KL-divergence between the student and teacher soft outputs. A temperature parameter T controls the softness of the distributions: higher temperatures produce softer distributions that reveal more about the teacher's learned structure.

For language models, distillation takes several forms. DistilBERT is a 66-million parameter model distilled from BERT-base (110M) that retains 97% of BERT's performance while being 60% faster. TinyLLaMA trains a 1.1B parameter model using the LLaMA architecture on the same data distribution as larger models.

Modern LLM distillation often uses the teacher to generate training data rather than matching output distributions directly. The student is fine-tuned on high-quality outputs generated by the teacher model. Alpaca was trained on 52K instruction-following demonstrations generated by GPT-3.5. This approach is simpler than traditional distillation and can transfer complex reasoning capabilities.\
""",

    "model_quantization": """\
Model Quantization

Quantization reduces the memory footprint and computational cost of neural networks by representing weights and activations with lower-precision data types. Standard Transformer models use 32-bit (FP32) or 16-bit (FP16/BF16) floating-point numbers. Quantization converts these to 8-bit integers (INT8) or even 4-bit (INT4) representations.

Post-training quantization (PTQ) converts a pre-trained model to lower precision without retraining. GPTQ is a popular one-shot quantization method that minimizes the layer-wise reconstruction error when quantizing to 3-4 bits. AWQ (Activation-Aware Weight Quantization) identifies salient weights based on activation patterns and quantizes them with higher precision.

Quantization-aware training (QAT) simulates low-precision arithmetic during training, allowing the model to adapt to quantization effects. QAT typically produces higher-quality quantized models than PTQ but requires access to training data and compute.

The impact of quantization on model quality depends on the target precision. FP16/BF16 quantization is essentially lossless. INT8 quantization typically causes less than 1% degradation. 4-bit quantization can cause 1-5% quality loss on benchmarks but dramatically reduces memory requirements — a 7B parameter model drops from 14GB (FP16) to 3.5GB (4-bit), enabling it to run on consumer GPUs. The GGUF format, used by llama.cpp and Ollama, supports various quantization levels.\
""",

    "lora": """\
LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method introduced by Hu et al. (2021) that freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each layer. Instead of updating the full weight matrix W, LoRA adds a low-rank update: W' = W + BA, where B and A are small matrices with rank r << d.

For a weight matrix W of shape d x d, full fine-tuning updates d^2 parameters. LoRA only trains 2*d*r parameters (for matrices A and B), where r is typically 4-64. With r=16 and d=4096, this reduces trainable parameters from 16.7M to 131K per layer — a 128x reduction.

LoRA is applied to attention projection matrices (Q, K, V, and output projections) and sometimes to FFN layers. The low-rank constraint acts as a regularizer, often preventing overfitting on small datasets. LoRA adapters can be merged into the base model at inference time, adding zero latency. Multiple LoRA adapters can also be swapped or combined for different tasks.

QLoRA (Quantized LoRA) combines 4-bit quantization of the base model with LoRA fine-tuning, enabling fine-tuning of 65B parameter models on a single 48GB GPU. This democratized LLM fine-tuning, making it accessible to researchers and practitioners without access to large compute clusters.\
""",

    "flash_attention": """\
Flash Attention

Flash Attention, introduced by Dao et al. (2022), is an IO-aware exact attention algorithm that significantly speeds up Transformer training and inference. Standard attention materializes the full N x N attention matrix in GPU high-bandwidth memory (HBM), which is both memory-intensive (O(N^2)) and slow due to memory bandwidth limitations.

Flash Attention tiles the attention computation into blocks that fit in SRAM (the fast on-chip memory), avoiding the need to materialize the full attention matrix in HBM. It fuses the attention operations (matrix multiply, softmax, masking, dropout) into a single GPU kernel. This reduces HBM reads/writes by up to 9x, resulting in 2-4x wall-clock speedup for attention operations.

Flash Attention 2 (2023) further improves performance by better work partitioning between GPU thread blocks and warps. It achieves close to the theoretical maximum FLOPs utilization on modern GPUs (up to 73% on A100). Flash Attention 2 is used by default in most modern LLM training frameworks.

The practical impact is significant. Flash Attention enables training with longer context lengths without running out of memory. Models like LLaMA 2 (4K context), Mistral (8K context with sliding window), and GPT-4 (128K context) all benefit from Flash Attention variants. The algorithm is mathematically equivalent to standard attention — it produces identical outputs while being faster and more memory-efficient.\
""",

    "mixture_of_experts": """\
Mixture of Experts (MoE)

Mixture of Experts is an architecture pattern where the feed-forward network in each Transformer layer is replaced by multiple expert FFNs, with a gating network that routes each token to a subset of experts. This allows the model to have more total parameters while only activating a fraction of them for each input, achieving better compute efficiency.

The Switch Transformer by Google (2021) simplified MoE by routing each token to a single expert. Mixtral 8x7B by Mistral AI (2023) uses 8 expert FFNs per layer with top-2 routing, meaning each token is processed by 2 of the 8 experts. Despite having 47B total parameters, Mixtral only uses 13B parameters per forward pass, achieving performance comparable to LLaMA 2 70B at much lower inference cost.

The gating function, typically a simple linear layer followed by softmax or top-k selection, determines which experts process each token. Load balancing is crucial: if some experts are used much more than others, the model wastes capacity. Auxiliary losses encourage balanced expert utilization during training.

MoE presents unique challenges for deployment. The full model must fit in memory even though only a fraction of parameters are active per token. Expert parallelism across GPUs requires careful placement to minimize communication. Despite these challenges, MoE has become the dominant approach for scaling models beyond a trillion parameters, as demonstrated by GPT-4 (rumored to use MoE) and Google's DeepMind models.\
""",

    "vision_transformers": """\
Vision Transformers (ViT)

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2020) at Google, applies the Transformer architecture directly to image recognition. An image is divided into fixed-size patches (typically 16x16 pixels), each patch is linearly embedded into a vector, and the sequence of patch embeddings is processed by a standard Transformer encoder.

A special [CLS] token is prepended to the patch sequence, and its final representation is used for classification. Position embeddings are added to encode spatial information. With sufficient training data, ViT matches or exceeds the performance of state-of-the-art convolutional neural networks while being conceptually simpler.

ViT initially required pre-training on very large datasets (JFT-300M) to outperform CNNs. DeiT (Data-efficient Image Transformers) showed that ViT can be trained effectively on ImageNet alone using strong data augmentation and knowledge distillation. This democratized vision transformers for researchers without access to proprietary datasets.

Subsequent work extended ViT in many directions: Swin Transformer introduced hierarchical feature maps with shifted window attention for dense prediction tasks. DINO and DINOv2 demonstrated powerful self-supervised learning with ViTs. MAE (Masked Autoencoder) adapted the masked language modeling paradigm to vision by masking random image patches and reconstructing them.\
""",

    "multimodal_models": """\
Multimodal Models

Multimodal models process and generate content across multiple modalities — text, images, audio, and video. The Transformer architecture has proven to be a flexible foundation for multimodal learning, as sequences of tokens from different modalities can be concatenated and processed jointly.

CLIP (Contrastive Language-Image Pre-training) by OpenAI (2021) learns a shared embedding space for images and text through contrastive learning. Trained on 400 million image-text pairs, CLIP can perform zero-shot image classification by computing similarity between image embeddings and text embeddings of class descriptions. CLIP's image encoder has become a standard component in many multimodal systems.

LLaVA (Large Language and Vision Assistant) connects a pre-trained vision encoder (CLIP ViT) with a large language model through a simple projection layer. The model is trained on visual instruction data where GPT-4 generates conversations about images. This approach achieves strong visual question answering performance with minimal architectural changes.

GPT-4V and Gemini represent the frontier of multimodal models, natively processing interleaved image and text inputs. These models can describe images, answer questions about visual content, read text from images, and reason about spatial relationships. The trend is toward unified models that handle all modalities within a single architecture and training framework.\
""",

    "prompt_engineering": """\
Prompt Engineering

Prompt engineering is the practice of designing input prompts to elicit desired behaviors from language models. As LLMs became capable of in-context learning, the way tasks are framed in the prompt became a critical factor in performance.

Key prompting strategies include: zero-shot prompting (describe the task without examples), few-shot prompting (provide examples of desired input-output pairs), chain-of-thought prompting (include reasoning steps in examples), and system prompts (define the model's role and constraints at the beginning of the conversation).

The structure of prompts matters significantly. Clear task descriptions improve performance. Separating instructions from context with delimiters reduces confusion. Specifying the output format (JSON, markdown, etc.) improves consistency. Providing negative examples of what not to do can be as helpful as positive examples.

Advanced techniques include: ReAct (Reasoning + Acting) where the model alternates between thinking and tool use; Tree of Thoughts where the model explores multiple reasoning paths; and retrieval-augmented prompting where relevant context is dynamically inserted. For RAG systems, the system prompt is particularly important: it instructs the model to answer only from provided context, reducing hallucination. Prompt sensitivity remains a challenge — small wording changes can cause large performance differences.\
""",

    "hallucination": """\
Hallucination in Language Models

Hallucination refers to language models generating text that is fluent and plausible-sounding but factually incorrect or unsupported by the input context. Hallucination is one of the most significant challenges for deploying LLMs in production, particularly for knowledge-intensive tasks.

Hallucinations are classified into two types. Intrinsic hallucinations contradict the provided source material. Extrinsic hallucinations contain information that cannot be verified from the source, neither confirmed nor contradicted. Both types are problematic for applications requiring factual accuracy.

Several factors contribute to hallucination. During pre-training, the model learns statistical patterns that may include factual errors in the training data. The training objective (next-token prediction) rewards fluency over accuracy. Models may also "confuse" information from different contexts or generate plausible-sounding but invented details to maintain coherence.

RAG (Retrieval-Augmented Generation) is one of the most effective approaches to reducing hallucination. By providing relevant documents as context and instructing the model to answer only from that context, RAG grounds the model's outputs in verifiable information. Other mitigation strategies include: calibrating confidence (the model expresses uncertainty when unsure), factual consistency checking (comparing generated text against source documents), and self-consistency (generating multiple responses and selecting the most consistent one).\
""",

    "context_window": """\
Context Window and Long-Context Models

The context window is the maximum number of tokens a language model can process in a single forward pass. Early Transformers had short context windows (512 tokens for BERT, 1024 for GPT-2) due to the quadratic memory complexity of self-attention. Modern models have dramatically expanded context windows.

GPT-4 supports 128K tokens. Claude 3 handles 200K tokens. Gemini 1.5 Pro processes up to 1 million tokens. These expansions were enabled by architectural innovations like Flash Attention, sparse attention patterns, and positional encoding improvements such as RoPE (Rotary Position Embedding) and ALiBi.

Long-context models face the "lost in the middle" problem: they tend to pay more attention to information at the beginning and end of the context, sometimes missing relevant information in the middle. This has implications for RAG systems, where the placement of retrieved documents in the context can affect generation quality.

Techniques for extending context length include: sliding window attention (Mistral), where attention is limited to a local window; grouped-query attention (GQA), which reduces KV-cache memory by sharing key-value heads; and retrieval-augmented approaches that selectively load relevant context from a larger document store. The trade-off between context length and inference cost is a key design decision for deployed LLM systems.\
""",

    "attention_variants": """\
Efficient Attention Variants

The quadratic complexity of standard self-attention (O(n^2) in sequence length) has motivated extensive research into more efficient attention mechanisms. These variants trade off some modeling quality for improved computational and memory efficiency, enabling longer sequence processing.

Sparse attention limits each token to attend to a fixed subset of other tokens rather than all tokens. Longformer uses a combination of local sliding window attention and global attention on special tokens. BigBird combines random, window, and global attention patterns. These approaches achieve O(n) or O(n*sqrt(n)) complexity.

Linear attention replaces the softmax attention kernel with a linear kernel function, decomposing the attention computation into separate query-key and key-value products. This enables O(n) complexity. Performers use random feature approximations of the softmax kernel. RetNet (Retentive Network) uses exponential decay attention that can be computed recurrently, combining the training parallelism of Transformers with O(1) inference complexity per token.

Multi-query attention (MQA) and grouped-query attention (GQA) reduce the memory bandwidth requirements of attention by sharing key and value projections across attention heads. GQA, used in LLaMA 2 and Mistral, divides heads into groups where each group shares a single key-value head. This reduces KV-cache size by 4-8x with minimal quality degradation, significantly improving inference throughput.\
""",

    "pretraining_objectives": """\
Pre-training Objectives for Language Models

Pre-training objectives define the self-supervised task a language model learns from unlabeled text. The choice of objective determines what knowledge and capabilities the model acquires and how it can be used downstream.

Causal language modeling (CLM) trains the model to predict the next token given all previous tokens. This autoregressive objective is used by GPT models. CLM naturally supports text generation and in-context learning. The loss is computed only on the next token prediction: L = -sum(log P(x_t | x_<t)).

Masked language modeling (MLM) randomly masks tokens and trains the model to predict them from bidirectional context. Used by BERT, MLM produces representations that capture both left and right context, making them strong for understanding tasks like classification and extraction. However, MLM models are not natural text generators.

Span corruption, used by T5, masks contiguous spans of tokens and trains the model to generate the missing spans. This objective unifies understanding and generation in an encoder-decoder framework. Prefix language modeling, used by UniLM, combines causal and bidirectional objectives.

Denoising objectives like those in BART add various types of noise (token masking, deletion, permutation, rotation) and train the model to reconstruct the original text. Different corruptions encourage different capabilities. The trend in modern LLMs is toward causal language modeling with massive scale, as it provides the most natural interface for instruction following and generation.\
""",

    "sequence_to_sequence": """\
Sequence-to-Sequence Models

Sequence-to-sequence (seq2seq) models transform an input sequence into an output sequence, making them suitable for tasks like translation, summarization, and question answering. The Transformer architecture supports three main paradigms: encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5, BART).

The encoder-decoder architecture processes the input through the encoder, which produces contextualized representations, and then generates the output autoregressively through the decoder, which attends to the encoder output via cross-attention. T5 (Text-to-Text Transfer Transformer) frames all NLP tasks as text-to-text problems: translation, summarization, classification, and question answering all use the same architecture and training procedure.

BART (Bidirectional and Auto-Regressive Transformer) is a denoising autoencoder that uses the encoder-decoder architecture. It is pre-trained by corrupting text with various noise functions and learning to reconstruct the original. BART excels at generation tasks like summarization and is the foundation for the mBART multilingual model.

Despite the theoretical advantages of encoder-decoder models for conditional generation tasks, decoder-only models have become dominant for general-purpose LLMs. Scaling experiments showed that decoder-only models achieve comparable or better performance with simpler architectures and training pipelines. However, encoder-decoder models remain competitive for specific tasks like translation and structured output generation.\
""",

    "fine_tuning_techniques": """\
Fine-Tuning Techniques for Language Models

Fine-tuning adapts a pre-trained language model to a specific task or domain by continuing training on task-specific data. The approach to fine-tuning has evolved significantly as models have grown larger, with different strategies offering trade-offs between performance, cost, and simplicity.

Full fine-tuning updates all model parameters on the downstream task. This is most effective when sufficient task-specific data is available but requires storing a separate copy of all parameters for each task. For a 7B parameter model, this means 14GB per task in FP16.

Parameter-efficient fine-tuning (PEFT) methods update only a small fraction of parameters. LoRA adds low-rank matrices to attention layers. Adapters insert small trainable modules between existing layers. Prefix tuning prepends trainable vectors to the attention keys and values. P-tuning learns continuous prompt embeddings. These methods typically achieve 90-99% of full fine-tuning performance while training less than 1% of parameters.

Instruction tuning is a form of fine-tuning on diverse (instruction, response) pairs that improves the model's ability to follow instructions generally. RLHF and DPO further align the model with human preferences. The typical pipeline for creating a production chat model is: pre-train on large corpus, supervised fine-tune on high-quality instruction data, and align with preference optimization. Each stage uses different data, objectives, and hyperparameters.\
""",

    "evaluation_metrics_nlp": """\
Evaluation Metrics for NLP and RAG Systems

Evaluating the quality of language model outputs and RAG systems requires a combination of automated metrics and human evaluation. Different metrics capture different aspects of quality, and no single metric is sufficient.

For question answering, common metrics include Exact Match (EM), which checks if the predicted answer exactly matches the ground truth, and F1 score, which measures token-level overlap between prediction and ground truth. These metrics are simple and deterministic but may penalize valid paraphrases.

RAG-specific evaluation frameworks like RAGAS assess multiple dimensions: faithfulness measures whether the answer is grounded in the retrieved context, answer relevance measures whether the answer addresses the query, context precision measures whether the retrieved documents are relevant, and context recall measures whether all relevant information was retrieved. These dimensions are computed using LLM-as-judge evaluations or token overlap heuristics.

Perplexity measures how well a language model predicts a held-out test set and is the standard metric for comparing language models. BLEU and ROUGE are n-gram overlap metrics used for translation and summarization. BERTScore uses contextualized embeddings to compute semantic similarity between predictions and references. Human evaluation remains the gold standard but is expensive and slow, making automated metrics essential for rapid iteration.\
""",

    "document_chunking": """\
Document Chunking Strategies for RAG

Document chunking is the process of splitting documents into smaller segments for indexing and retrieval in RAG systems. The chunking strategy significantly affects both retrieval precision and the quality of generated answers.

Fixed-size chunking splits text into segments of a constant number of characters or tokens, optionally with overlap between consecutive chunks. This is the simplest strategy and works well as a baseline. Typical chunk sizes range from 256 to 1024 tokens. Overlap (usually 10-20% of chunk size) prevents information loss at chunk boundaries.

Sentence-level chunking splits text at sentence boundaries, grouping sentences until a maximum size is reached. This preserves natural semantic units and avoids splitting mid-sentence, which can confuse the embedding model and the generation model. Paragraph-level chunking is similar but uses paragraph boundaries.

Semantic chunking uses an embedding model to detect topic shifts within a document. Adjacent sentences with similar embeddings are grouped together, and splits occur where the embedding similarity drops below a threshold. This produces chunks that are more semantically coherent than fixed-size chunks.

Recursive chunking (used in LangChain) tries multiple separators in order of priority: first paragraph breaks, then sentence breaks, then word breaks, falling back to character-level splitting only if necessary. The choice of chunking strategy should be guided by the nature of the corpus and the types of queries expected.\
""",

    "query_expansion": """\
Query Expansion Techniques

Query expansion improves retrieval performance by augmenting the original user query with additional terms or reformulations before searching the document index. This addresses the vocabulary mismatch problem, where users and documents may use different terms for the same concept.

Traditional query expansion uses pseudo-relevance feedback: retrieve initial results, extract important terms from top-ranked documents, and add these terms to the query for a second retrieval pass. This unsupervised approach improves recall but can introduce topic drift if the initial results are off-topic.

LLM-based query expansion generates multiple reformulations of the original query using a language model. For example, given the query "What causes diabetes?", the LLM might generate: "What are the risk factors for diabetes?", "How does insulin resistance lead to diabetes?", and "What is the etiology of type 2 diabetes?". Each reformulation is used to retrieve documents independently, and the results are merged.

Multi-query retrieval is a specific form of query expansion where the LLM generates N different versions of the query, retrieves top-k documents for each version, and takes the union of all retrieved documents. This increases recall at the cost of N additional retrieval calls. The retrieved documents can then be reranked using a cross-encoder to restore precision. Query expansion is most effective when the original query is short or ambiguous.\
""",

    "ollama_local_inference": """\
Ollama and Local LLM Inference

Ollama is an open-source tool for running large language models locally on consumer hardware. It provides a simple command-line interface and HTTP API for downloading, managing, and running models. Ollama supports a wide range of models including LLaMA, Mistral, Gemma, Phi, and many others in the GGUF quantized format.

Local inference offers several advantages over cloud APIs: complete data privacy (no data leaves your machine), zero per-token cost, no rate limits, and the ability to work offline. The trade-off is that local models are generally smaller and less capable than cloud-hosted models like GPT-4 or Claude.

Ollama automatically handles model quantization and memory management. A 7B parameter model in 4-bit quantization requires approximately 4GB of RAM, running comfortably on modern laptops. Larger models (13B, 34B, 70B) require proportionally more memory and benefit from GPU acceleration. Ollama supports GPU offloading on both NVIDIA and Apple Silicon hardware.

For RAG systems, local inference means the entire pipeline — embedding, retrieval, and generation — can run without any cloud dependency. Models like llama3.2:3b provide reasonable quality for QA tasks while running fast on CPU. For embeddings, Ollama can also serve embedding models, though sentence-transformers running locally is typically faster for batch embedding operations.\
""",

    "cosine_similarity": """\
Cosine Similarity and Distance Metrics

Cosine similarity is the most commonly used metric for comparing embedding vectors in information retrieval and RAG systems. It measures the cosine of the angle between two vectors, ranging from -1 (opposite directions) to 1 (same direction), with 0 indicating orthogonality.

The formula is: cos_sim(A, B) = (A · B) / (||A|| * ||B||), where A · B is the dot product and ||A||, ||B|| are the L2 norms. For normalized vectors (unit length), cosine similarity equals the dot product, making it computationally efficient. Most modern embedding models produce normalized vectors by default.

Euclidean (L2) distance measures the straight-line distance between two points in vector space. It is related to cosine similarity for normalized vectors: L2_distance^2 = 2 - 2 * cos_sim. L2 distance is sensitive to vector magnitude, which can be problematic when comparing embeddings of different-length texts.

Manhattan (L1) distance sums the absolute differences across dimensions and is more robust to outliers than L2. Inner product (dot product) combines both magnitude and direction similarity. When vectors are not normalized, dot product and cosine similarity can give different rankings. For retrieval tasks with normalized embeddings, cosine similarity and dot product produce identical rankings, so FAISS's IndexFlatIP (inner product) is commonly used for efficiency.\
""",

    "t5_model": """\
T5: Text-to-Text Transfer Transformer

T5 (Text-to-Text Transfer Transformer), introduced by Raffel et al. at Google in 2019, is an encoder-decoder model that frames every NLP task as a text-to-text problem. Translation, summarization, classification, and question answering all use the same input-output format: the model receives text and produces text.

For example, translation becomes "translate English to German: The house is wonderful." → "Das Haus ist wunderbar." Sentiment classification becomes "sst2 sentence: The movie was excellent." → "positive." This unified format allows a single model architecture and training procedure for all tasks.

T5 was pre-trained on the C4 (Colossal Clean Crawled Corpus), a cleaned version of Common Crawl containing about 750GB of text. The pre-training objective is span corruption: random spans of input tokens are replaced with sentinel tokens, and the model is trained to generate the missing spans. T5 comes in multiple sizes: Small (60M), Base (220M), Large (770M), 3B, and 11B parameters.

FLAN-T5 is an instruction-tuned version that significantly improves T5's zero-shot and few-shot capabilities. T5's encoder-decoder architecture makes it particularly effective for tasks that require both understanding the input (encoder) and generating a structured output (decoder), such as summarization, question answering with extractive answers, and data-to-text generation.\
""",

    "llama_models": """\
LLaMA: Large Language Model Meta AI

LLaMA (Large Language Model Meta AI), released by Meta in February 2023, is a family of open-weight language models that demonstrated smaller, well-trained models can match or exceed larger models. LLaMA was trained following the Chinchilla scaling laws, using more data relative to model size than previous open models.

LLaMA 1 came in four sizes: 7B, 13B, 33B, and 65B parameters, trained on 1-1.4 trillion tokens from publicly available datasets. LLaMA-13B outperformed GPT-3 (175B) on most benchmarks, demonstrating the importance of training data quantity and quality over raw model size.

LLaMA 2, released in July 2023, expanded to sizes of 7B, 13B, and 70B with a 4K context window. It was trained on 2 trillion tokens and included LLaMA 2 Chat models fine-tuned with RLHF for dialogue applications. LLaMA 2 used grouped-query attention (GQA) in the 70B variant for improved inference efficiency.

LLaMA 3, released in 2024, further scaled the training data to 15 trillion tokens and introduced an 8K context window. The LLaMA 3 8B model approaches the performance of LLaMA 2 70B. LLaMA 3.2 includes 1B and 3B models optimized for edge deployment. The LLaMA family has become the foundation for numerous derivative models including Vicuna, Alpaca, CodeLlama, and Mistral (which uses a similar architecture).\
""",

    "mistral_models": """\
Mistral AI Models

Mistral AI, a French AI company founded in 2023, has released several influential open-weight language models. Mistral 7B, their first release, outperformed LLaMA 2 13B on all benchmarks despite being half the size, demonstrating significant architectural and training improvements.

Mistral 7B introduced sliding window attention (SWA) with a window size of 4096 tokens, allowing it to handle sequences much longer than the window by letting information propagate through layers. Combined with Flash Attention 2 and grouped-query attention, Mistral 7B achieved excellent inference speed while maintaining quality.

Mixtral 8x7B is Mistral's Mixture of Experts model with 8 expert FFNs per layer and top-2 routing. Despite 47B total parameters, it activates only 13B per forward pass, matching LLaMA 2 70B performance at a fraction of the inference cost. Mixtral demonstrated that MoE can be practical at moderate scale.

Mistral models are available through Ollama and can be run locally with reasonable hardware requirements. Mistral 7B in 4-bit quantization requires about 4GB of RAM, making it accessible on most modern computers. The efficiency-to-quality ratio of Mistral models makes them popular choices for local RAG systems where inference speed matters.\
""",

    "data_preprocessing": """\
Data Preprocessing for RAG Systems

Effective data preprocessing is crucial for RAG system performance. Raw documents often contain noise, inconsistent formatting, and irrelevant content that can degrade retrieval quality and confuse the generation model.

Text extraction from PDFs is a common challenge. PDFs may contain multi-column layouts, tables, headers, footers, and images. Tools like PyPDF, pdfplumber, and unstructured handle different PDF types with varying accuracy. OCR (optical character recognition) may be needed for scanned documents. Maintaining document structure (headings, sections, lists) during extraction improves chunking quality.

Text cleaning involves removing or standardizing artifacts: extra whitespace, special characters, encoding errors, HTML tags, and boilerplate content (navigation menus, copyright notices). However, aggressive cleaning can remove meaningful formatting cues. The best approach depends on the corpus — legal documents need different treatment than blog posts.

Metadata enrichment adds structured information to documents: source URL, creation date, document type, section headings, and author. This metadata can be used for filtered retrieval (search only recent documents, only from specific sources) and can be included in the LLM prompt to provide additional context. For RAG systems, the quality of preprocessing often matters more than the sophistication of the retrieval algorithm.\
""",

    "diffusion_models": """\
Diffusion Models

Diffusion models are a class of generative models that learn to create data by gradually denoising a signal, reversing a process that progressively adds Gaussian noise until the data becomes pure noise. They have achieved state-of-the-art results in image generation, surpassing GANs in quality and diversity.

The forward diffusion process adds small amounts of Gaussian noise to data over T timesteps, gradually transforming it into pure noise. The reverse process learns to denoise at each step, transforming noise back into data. The model is trained to predict the noise added at each timestep, using a U-Net or Transformer architecture.

Stable Diffusion by Stability AI operates in a compressed latent space rather than pixel space, making it computationally efficient. A variational autoencoder first encodes images into a lower-dimensional latent representation. The diffusion process operates in this latent space, and the final latent is decoded back to pixel space.

Text-to-image diffusion models like DALL-E 2, Midjourney, and Stable Diffusion use CLIP text embeddings to condition the generation process, enabling generation from text descriptions. Classifier-free guidance balances generation quality with prompt adherence. While not directly related to text-based RAG, diffusion models share key concepts with Transformers and demonstrate the power of iterative refinement — a principle that also applies to autonomous pipeline optimization.\
""",

    "neural_architecture_search": """\
Neural Architecture Search

Neural Architecture Search (NAS) automates the design of neural network architectures, replacing manual architecture engineering with algorithmic search. NAS methods explore a defined search space of possible architectures and use a search strategy to find architectures that maximize performance on a validation set.

The search space defines what architectures are possible: which operations (convolution, attention, pooling), how they connect, and what hyperparameters they have. The search strategy can be random search, evolutionary algorithms, reinforcement learning (the controller generates architectures and receives accuracy as reward), or gradient-based methods (DARTS differentiates through the architecture choice).

Early NAS methods required thousands of GPU hours to search. One-shot NAS methods like DARTS train a single super-network that contains all possible architectures as subgraphs, then search for the best subgraph. This reduces search cost from thousands to a few GPU hours.

NAS has produced several notable architectures: EfficientNet (discovered by NAS on ImageNet) and NASNet (architecture for image classification). For Transformers, NAS has been applied to find optimal attention patterns, FFN dimensions, and layer configurations. AutoML pipelines extend NAS to jointly optimize architecture, hyperparameters, and training procedures.\
""",

    "semantic_search": """\
Semantic Search

Semantic search retrieves documents based on meaning rather than keyword matching. Unlike traditional lexical search (BM25, TF-IDF) that matches exact terms, semantic search uses dense vector representations to find conceptually related content even when different words are used.

The semantic search pipeline consists of: encoding documents into dense vectors using an embedding model, storing vectors in an index (like FAISS), encoding the query into a vector using the same model, and retrieving the nearest vectors by similarity. The quality depends heavily on the embedding model's ability to capture semantic relationships.

Bi-encoder models (used in semantic search) independently encode queries and documents, enabling pre-computation of document embeddings and fast retrieval. Cross-encoder models jointly process query-document pairs for more accurate scoring but cannot pre-compute document representations. The common pattern is bi-encoder for initial retrieval followed by cross-encoder for reranking.

Semantic search excels when users express needs differently from how information is stored: searching for "how to fix a broken pipe" retrieves documents about "plumbing repair methods." It struggles with exact-match queries, rare terms, and highly specific technical vocabulary. This complementarity with lexical search is why hybrid retrieval (combining both approaches) consistently outperforms either method alone.\
""",

    "training_compute": """\
Training Compute and Infrastructure for LLMs

Training large language models requires massive computational resources. GPT-3 (175B parameters) required approximately 3.14 × 10^23 FLOPs, estimated at 1,024 A100 GPUs for about 34 days. GPT-4's training is estimated to have used over 10,000 GPUs for several months.

Data parallelism distributes training across GPUs by replicating the model on each GPU and splitting the data batch. ZeRO (Zero Redundancy Optimizer) by DeepSpeed reduces memory by partitioning optimizer states, gradients, and parameters across GPUs rather than replicating them.

Model parallelism splits the model itself across GPUs. Tensor parallelism splits individual layers across GPUs. Pipeline parallelism assigns different layers to different GPUs and overlaps computation across micro-batches. Megatron-LM by NVIDIA combines all three parallelism strategies for efficient training at scale.

Mixed-precision training uses FP16 or BF16 for most computations while maintaining a FP32 master copy of weights for updates. BF16 (Brain Float16) has the same exponent range as FP32, avoiding overflow issues that sometimes occur with FP16. Modern GPU architectures (A100, H100) include specialized tensor cores for fast FP16/BF16 matrix multiplication, achieving 2-4x speedup over FP32.\
""",

    "constitutional_ai": """\
Constitutional AI

Constitutional AI (CAI) is an alignment technique developed by Anthropic that trains AI systems to be helpful, harmless, and honest using a set of principles (a "constitution") rather than relying entirely on human feedback labels. CAI aims to reduce the need for human labelers while producing well-aligned models.

The CAI process has two phases. In the first phase (supervised), the model generates responses, then critiques and revises its own responses according to constitutional principles. For example, a principle might be "choose the response that is least likely to encourage illegal activity." The model generates pairs of (original, revised) responses, and the revised responses are used as training data.

In the second phase (reinforcement learning), the model generates response pairs and uses the constitution to determine which response is preferred. This preference data trains a reward model, which is then used for RLHF. The key innovation is that both the critique and the preference labeling are done by the AI itself, guided by the constitution.

CAI produces models that are more transparent in their reasoning about safety. Instead of learning implicit rules from human labels, the model explicitly references principles when deciding how to respond. This makes the alignment process more interpretable and auditable. The constitution can be updated to adjust behavior without retraining from scratch.\
""",

    "few_shot_learning": """\
Few-Shot and Zero-Shot Learning

Few-shot learning enables models to perform tasks with only a handful of examples. In the context of large language models, few-shot learning is achieved through in-context learning: examples are provided in the prompt, and the model infers the task pattern without parameter updates.

Zero-shot learning provides no examples at all — only a task description. The model relies entirely on knowledge from pre-training to understand and perform the task. GPT-3 showed that zero-shot performance improves with model scale, with the largest models performing competitively with fine-tuned smaller models on many tasks.

Few-shot learning in LLMs depends on several factors: the number and quality of examples, their ordering in the prompt, their similarity to the test input, and the label distribution across examples. Research shows that even random or misleading labels in few-shot examples can sometimes maintain performance, suggesting the model uses the format more than the specific labels.

Meta-learning approaches train models to be good few-shot learners. MAML (Model-Agnostic Meta-Learning) optimizes initial parameters to enable fast adaptation. Prototypical networks learn to classify by computing distances to class prototypes in embedding space. While LLM in-context learning has largely supplanted these methods for NLP tasks, meta-learning principles inform how instruction tuning datasets are designed.\
""",

    "model_evaluation_benchmarks": """\
Model Evaluation Benchmarks

Benchmarks provide standardized evaluation of language model capabilities across diverse tasks. They are essential for comparing models, tracking progress, and identifying strengths and weaknesses.

GLUE (General Language Understanding Evaluation) and SuperGLUE are collections of NLU tasks including sentiment analysis, textual entailment, and sentence similarity. BERT-level models saturated GLUE, leading to the harder SuperGLUE. Both benchmarks are solved by modern LLMs.

MMLU (Massive Multitask Language Understanding) evaluates knowledge across 57 subjects from STEM to humanities. It tests the model's ability to recall factual knowledge and apply reasoning across diverse domains. GPT-4 achieves 86.4% on MMLU; human expert performance is around 89%.

HumanEval evaluates code generation by testing whether model-generated Python functions pass unit tests. The pass@k metric measures the probability that at least one of k generated solutions is correct. MT-Bench and Chatbot Arena use human preference judgments to evaluate conversational ability. The MTEB (Massive Text Embedding Benchmark) evaluates embedding models across retrieval, classification, and clustering tasks. No single benchmark captures all aspects of model quality, and benchmark performance does not always correlate with real-world usefulness.\
""",
}


# ============================================================
# EVAL SET: 30 questions with ground truth answers and contexts
# ============================================================

EVAL_SET = [
    {
        "query": "What is the core mechanism of the Transformer architecture?",
        "ground_truth_answer": "Self-attention (also called intra-attention) is the core mechanism of the Transformer architecture, allowing each position in a sequence to attend to all other positions.",
        "ground_truth_contexts": [ARTICLES["self_attention"], ARTICLES["transformer_architecture"]],
    },
    {
        "query": "What is the formula for scaled dot-product attention?",
        "ground_truth_answer": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V, where Q, K, and V are query, key, and value matrices, and d_k is the dimension of the keys.",
        "ground_truth_contexts": [ARTICLES["transformer_architecture"]],
    },
    {
        "query": "How does multi-head attention work?",
        "ground_truth_answer": "Multi-head attention runs multiple attention operations in parallel with different learned projections, then concatenates the outputs. Each head attends to different representation subspaces, allowing the model to capture diverse relationship patterns.",
        "ground_truth_contexts": [ARTICLES["multi_head_attention"]],
    },
    {
        "query": "Why are positional encodings needed in Transformers?",
        "ground_truth_answer": "Positional encodings provide information about token order because the Transformer processes all tokens in parallel without any inherent notion of sequence. Without them, the model would treat input as a bag of tokens.",
        "ground_truth_contexts": [ARTICLES["positional_encoding"]],
    },
    {
        "query": "What is the difference between pre-norm and post-norm in Transformers?",
        "ground_truth_answer": "Post-norm applies layer normalization after the sub-layer (original Transformer), while pre-norm applies it before the sub-layer. Pre-norm leads to more stable training for deep models and is used by GPT-2 and most modern LLMs.",
        "ground_truth_contexts": [ARTICLES["layer_normalization"]],
    },
    {
        "query": "What are BERT's pre-training objectives?",
        "ground_truth_answer": "BERT uses two pre-training objectives: Masked Language Modeling (MLM), which randomly masks 15% of tokens and predicts them from context, and Next Sentence Prediction (NSP), which predicts whether two sentences are consecutive. Later research showed NSP provides minimal benefit.",
        "ground_truth_contexts": [ARTICLES["bert"]],
    },
    {
        "query": "How many parameters does GPT-3 have and what is its key capability?",
        "ground_truth_answer": "GPT-3 has 175 billion parameters. Its key capability is few-shot learning through in-context learning, where providing a few examples in the prompt enables it to perform tasks without task-specific fine-tuning.",
        "ground_truth_contexts": [ARTICLES["gpt_series"]],
    },
    {
        "query": "What is the difference between BPE and WordPiece tokenization?",
        "ground_truth_answer": "BPE (Byte-Pair Encoding) selects merges based on raw frequency of adjacent token pairs, while WordPiece selects merges based on the likelihood of the training data. BPE is used by GPT models; WordPiece is used by BERT.",
        "ground_truth_contexts": [ARTICLES["tokenization"]],
    },
    {
        "query": "What did the Chinchilla scaling paper find?",
        "ground_truth_answer": "The Chinchilla paper found that previous models like GPT-3 were significantly undertrained relative to their size. The Chinchilla-optimal ratio is approximately 20 tokens per parameter. Chinchilla (70B) outperformed Gopher (280B) by training on more data.",
        "ground_truth_contexts": [ARTICLES["scaling_laws"]],
    },
    {
        "query": "What are emergent abilities in large language models?",
        "ground_truth_answer": "Emergent abilities are capabilities that appear in large models but are absent in smaller ones, representing qualitative changes from quantitative scaling. Examples include multi-step arithmetic, chain-of-thought reasoning, and code generation.",
        "ground_truth_contexts": [ARTICLES["emergent_abilities"]],
    },
    {
        "query": "How does RLHF align language models with human preferences?",
        "ground_truth_answer": "RLHF has three stages: supervised fine-tuning on demonstration data, training a reward model from human comparison data, and optimizing the language model against the reward model using PPO with a KL-divergence penalty.",
        "ground_truth_contexts": [ARTICLES["rlhf"]],
    },
    {
        "query": "What is chain-of-thought prompting and when does it work?",
        "ground_truth_answer": "Chain-of-thought prompting encourages the model to generate intermediate reasoning steps before the final answer. It significantly improves arithmetic and reasoning tasks. It primarily benefits models above approximately 100 billion parameters.",
        "ground_truth_contexts": [ARTICLES["chain_of_thought"]],
    },
    {
        "query": "What are the three stages of the RAG pipeline?",
        "ground_truth_answer": "The RAG pipeline consists of indexing (documents are chunked, embedded, and stored in a vector database), retrieval (the query is embedded and used to find similar document chunks), and generation (retrieved chunks are used as context for the LLM to generate a grounded answer).",
        "ground_truth_contexts": [ARTICLES["retrieval_augmented_generation"]],
    },
    {
        "query": "How does BM25 differ from dense retrieval?",
        "ground_truth_answer": "BM25 matches based on exact lexical overlap using term frequency and inverse document frequency, while dense retrieval matches based on semantic similarity in embedding space. BM25 is better for rare terms and specific keywords; dense retrieval is better for semantic matching.",
        "ground_truth_contexts": [ARTICLES["bm25_sparse_retrieval"]],
    },
    {
        "query": "What is cross-encoder reranking and how does it improve retrieval?",
        "ground_truth_answer": "Cross-encoder reranking jointly processes query-document pairs through a single Transformer for more accurate scoring than bi-encoders. It is typically 5-15% more accurate but 100-1000x slower, so it is applied to a small set of candidates from a fast first-stage retriever.",
        "ground_truth_contexts": [ARTICLES["cross_encoder_reranking"]],
    },
    {
        "query": "What is HyDE and when is it useful?",
        "ground_truth_answer": "HyDE (Hypothetical Document Embedding) uses an LLM to generate a hypothetical answer, then uses its embedding as the retrieval query. It bridges the gap between query and document spaces and is most effective for abstract queries or vocabulary mismatch situations.",
        "ground_truth_contexts": [ARTICLES["hyde"]],
    },
    {
        "query": "How does LoRA achieve parameter-efficient fine-tuning?",
        "ground_truth_answer": "LoRA freezes pre-trained weights and adds trainable low-rank matrices (B and A where rank r << d) to each layer. Instead of updating d^2 parameters, it trains only 2*d*r parameters, achieving a 128x reduction with r=16. LoRA adapters can be merged at inference with zero added latency.",
        "ground_truth_contexts": [ARTICLES["lora"]],
    },
    {
        "query": "How does Flash Attention improve Transformer performance?",
        "ground_truth_answer": "Flash Attention tiles the attention computation into blocks that fit in SRAM (fast on-chip memory), avoiding materializing the full N x N attention matrix in HBM. This reduces memory reads/writes by up to 9x, resulting in 2-4x wall-clock speedup while producing mathematically identical outputs.",
        "ground_truth_contexts": [ARTICLES["flash_attention"]],
    },
    {
        "query": "How does Mixtral use Mixture of Experts?",
        "ground_truth_answer": "Mixtral 8x7B uses 8 expert FFNs per layer with top-2 routing, meaning each token is processed by 2 of 8 experts. Despite 47B total parameters, it activates only 13B per forward pass, matching LLaMA 2 70B performance at lower inference cost.",
        "ground_truth_contexts": [ARTICLES["mixture_of_experts"]],
    },
    {
        "query": "What are the different document chunking strategies for RAG?",
        "ground_truth_answer": "Main strategies include fixed-size chunking (constant token count with overlap), sentence-level chunking (preserving sentence boundaries), paragraph-level chunking (using paragraph breaks), semantic chunking (detecting topic shifts via embedding similarity), and recursive chunking (trying multiple separators in priority order).",
        "ground_truth_contexts": [ARTICLES["document_chunking"]],
    },
    {
        "query": "What is the advantage of hybrid retrieval over dense-only retrieval?",
        "ground_truth_answer": "Hybrid retrieval combines dense (semantic) and sparse (BM25 keyword) methods using a weighted sum. Dense retrieval handles semantic matching while sparse handles exact terms. Research consistently shows hybrid outperforms either method alone, especially for queries with rare terms.",
        "ground_truth_contexts": [ARTICLES["bm25_sparse_retrieval"], ARTICLES["semantic_search"]],
    },
    {
        "query": "What is knowledge distillation and how is DistilBERT created?",
        "ground_truth_answer": "Knowledge distillation trains a smaller student model to mimic a larger teacher model using soft probability distributions. DistilBERT is a 66M parameter model distilled from BERT-base (110M), retaining 97% of performance while being 60% faster.",
        "ground_truth_contexts": [ARTICLES["knowledge_distillation"]],
    },
    {
        "query": "How does model quantization affect LLM deployment?",
        "ground_truth_answer": "Quantization reduces precision from FP32/FP16 to INT8 or INT4, dramatically reducing memory requirements. A 7B parameter model drops from 14GB (FP16) to 3.5GB (4-bit), enabling consumer GPU deployment. 4-bit quantization causes 1-5% quality loss.",
        "ground_truth_contexts": [ARTICLES["model_quantization"]],
    },
    {
        "query": "What is DPO and how does it simplify RLHF?",
        "ground_truth_answer": "Direct Preference Optimization (DPO) directly optimizes the language model on preference pairs without training a separate reward model. It reparameterizes the reward model as a function of the language model itself, reducing training complexity while achieving comparable alignment results.",
        "ground_truth_contexts": [ARTICLES["rlhf"]],
    },
    {
        "query": "What is the lost-in-the-middle problem in long-context models?",
        "ground_truth_answer": "Long-context models tend to pay more attention to information at the beginning and end of the context while missing relevant information in the middle. This affects RAG systems where the placement of retrieved documents in the context impacts generation quality.",
        "ground_truth_contexts": [ARTICLES["context_window"]],
    },
    {
        "query": "How did LLaMA demonstrate the importance of training data over model size?",
        "ground_truth_answer": "LLaMA-13B outperformed GPT-3 (175B) on most benchmarks by following Chinchilla scaling laws and training on more data relative to model size. This demonstrated that smaller, well-trained models can match much larger models.",
        "ground_truth_contexts": [ARTICLES["llama_models"], ARTICLES["scaling_laws"]],
    },
    {
        "query": "What are the key metrics used to evaluate RAG systems?",
        "ground_truth_answer": "Key RAG metrics include faithfulness (whether answers are grounded in context), answer relevance (whether answers address the query), context precision (whether retrieved documents are relevant), and context recall (whether all relevant information was retrieved).",
        "ground_truth_contexts": [ARTICLES["evaluation_metrics_nlp"]],
    },
    {
        "query": "How does Mistral 7B achieve efficiency through sliding window attention?",
        "ground_truth_answer": "Mistral 7B uses sliding window attention with a 4096-token window, limiting each token's attention to a local window. Information propagates through layers, allowing effective processing of sequences longer than the window size. Combined with Flash Attention 2 and grouped-query attention, it achieves excellent speed.",
        "ground_truth_contexts": [ARTICLES["mistral_models"], ARTICLES["attention_variants"]],
    },
    {
        "query": "What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?",
        "ground_truth_answer": "Encoder-only (BERT) processes bidirectional context for understanding tasks. Decoder-only (GPT) generates text autoregressively. Encoder-decoder (T5) first encodes the input then generates the output with cross-attention. Decoder-only models have become dominant for general-purpose LLMs.",
        "ground_truth_contexts": [ARTICLES["sequence_to_sequence"], ARTICLES["bert"], ARTICLES["gpt_series"]],
    },
    {
        "query": "How does RAG reduce hallucination in language models?",
        "ground_truth_answer": "RAG reduces hallucination by providing relevant retrieved documents as context and instructing the model to answer only from that context. This grounds the model's outputs in verifiable information rather than relying solely on potentially inaccurate knowledge from pre-training.",
        "ground_truth_contexts": [ARTICLES["hallucination"], ARTICLES["retrieval_augmented_generation"]],
    },
]


def main():
    print("Setting up autoRAGresearch default data...")

    # Create corpus files
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in ARTICLES.items():
        filepath = CORPUS_DIR / f"{name}.txt"
        filepath.write_text(text, encoding="utf-8")
    print(f"  Created {len(ARTICLES)} corpus articles in {CORPUS_DIR}")

    # Create eval set (strip large context texts to save space in JSON)
    eval_entries = []
    for item in EVAL_SET:
        eval_entries.append({
            "query": item["query"],
            "ground_truth_answer": item["ground_truth_answer"],
            "ground_truth_contexts": item["ground_truth_contexts"],
        })

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_entries, f, indent=2, ensure_ascii=False)
    print(f"  Created eval set with {len(eval_entries)} queries at {EVAL_SET_PATH}")

    print("\nDone! Next steps:")
    print("  1. ollama pull llama3.2:3b && ollama pull nomic-embed-text")
    print("  2. python eval.py          # see baseline score")
    print("  3. python loop.py          # start optimizing")


if __name__ == "__main__":
    main()
