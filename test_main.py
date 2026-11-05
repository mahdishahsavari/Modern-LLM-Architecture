import torch
import torch.nn.functional as F

from main import GroupedQueryAttention, TransformerLM


def small_model():
    return TransformerLM(vocab_size=50, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)


def test_output_shape():
    assert small_model()(torch.randint(0, 50, (2, 16))).shape == (2, 16, 50)


def test_future_tokens_do_not_change_earlier_logits():
    model = small_model().eval()
    tokens = torch.randint(0, 50, (1, 16))
    changed = tokens.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 50
    a, b = model(tokens), model(changed)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-5)


def test_grouped_attention_has_fewer_parameters_than_full_attention():
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(GroupedQueryAttention(64, 8, 2)) < count(GroupedQueryAttention(64, 8, 8))


def test_can_overfit_one_sequence():
    torch.manual_seed(0)
    model = small_model()
    tokens = torch.randint(0, 50, (1, 17))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        loss = F.cross_entropy(model(tokens[:, :-1]).squeeze(0), tokens[0, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1
