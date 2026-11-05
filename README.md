# Modern Architecture of LLMs

A small PyTorch implementation of a transformer language model with the components used in LLaMA and Mistral style models:

- Grouped query attention (GQA)
- Rotary position embeddings (RoPE)
- SwiGLU feed-forward layers
- RMSNorm
- Shared input and output embeddings
- A causal attention mask

This is an educational implementation of the architecture. It does not include a trained model or a training pipeline.

## Files

| File | Purpose |
|---|---|
| `main.py` | The model |
| `test_main.py` | Tests for the model's behavior |
| `guide.md` | An explanation of each component |

## Run

```bash
pip install -r requirements.txt
python main.py
pytest test_main.py
```

`python main.py` builds a model with random weights and prints the output shape: `torch.Size([2, 64, 32000])`.

## What the tests check

- The output has shape `(batch, sequence, vocabulary)`.
- Changing a future token does not change the logits of earlier positions, so the causal mask works.
- Grouped attention has fewer parameters than full attention.
- The model can overfit a single sequence, so it is able to learn.
