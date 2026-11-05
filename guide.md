# Architecture Guide

This guide explains the components in `main.py`. The model predicts the next token in a sequence:

```
tokens → embeddings → transformer blocks → final norm → logits
```

The output has shape `(batch, seq, vocab_size)`.

## 1. RMSNorm

```python
rms = sqrt(mean(x²) + eps)
output = (x / rms) * learnable_weight
```

RMSNorm keeps activations at a stable scale as they pass through many layers. It skips the mean-centering step of LayerNorm, so it needs fewer operations. LLaMA and Mistral use it.

## 2. Rotary Position Embeddings (RoPE)

Many earlier transformers add position information as a separate embedding. RoPE instead rotates the query and key vectors according to their position:

```python
freqs = position * (1 / 10000^(2i/d))
q_rotated = q * cos(freqs) + rotate_half(q) * sin(freqs)
k_rotated = k * cos(freqs) + rotate_half(k) * sin(freqs)
```

- The attention score depends on the relative distance between two tokens.
- It has no learned parameters.

`cos` and `sin` have shape `(seq, head_dim)`, while `q` and `k` have shape `(batch, seq, heads, head_dim)`. The code reshapes them to `(1, seq, 1, head_dim)` so they broadcast correctly.

## 3. Grouped Query Attention (GQA)

Standard attention has one key and value head for each query head. GQA lets several query heads share one key and value head:

```
Standard: 8 Q heads → 8 K heads, 8 V heads
GQA:      8 Q heads → 2 K heads, 2 V heads (each K/V head serves 4 Q heads)
```

This reduces parameters and the memory needed to cache keys and values during generation. LLaMA 2 and Mistral use this idea.

## 4. SwiGLU

```python
Traditional: FFN(x) = W2(ReLU(W1(x)))
SwiGLU:      FFN(x) = W3(SiLU(W1(x)) ⊙ W2(x))
```

- `W1` and `W2` create two parallel projections.
- `SiLU` is the smooth activation `x * sigmoid(x)`.
- `⊙` is element-wise multiplication, which acts as a gate on the information flow.
- `W3` projects back to the model dimension.

## 5. Shared Input and Output Embeddings

```python
self.lm_head.weight = self.embed.weight
```

The matrix that turns token IDs into vectors is reused to produce the final logits. This saves `vocab_size × dim` parameters.

## 6. Attention, Step by Step

```python
q, k, v = Wq(x), Wk(x), Wv(x)      # split into heads
q, k = RoPE(q, k)                  # add position information
k, v = repeat_kv(k), repeat_kv(v)  # match the number of query heads
scores = (q @ k^T) / sqrt(head_dim)
scores = scores.masked_fill(mask == 0, -inf)
output = softmax(scores) @ v
```

**Causal mask:** a token at position `i` may only attend to positions `≤ i`. Without this mask, the model could see the tokens it is supposed to predict. `TransformerLM` builds this mask automatically when no mask is passed.

## 7. Transformer Block

```python
x = x + Attention(RMSNorm(x))
x = x + FeedForward(RMSNorm(x))
```

The block normalizes before each layer (pre-norm) and adds a residual connection around it. Residual connections help gradients flow through deep networks.
