from __future__ import annotations

import math

import torch

from modlehr import registry
from modlehr.config import ModelConfig
from modlehr.model.heads import CompetingRiskHead
from modlehr.model.model import ModLEHR


def batch(b=2, t=10, n_tokens=20):
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(1, n_tokens, (b, t), generator=g)
    tokens[1, 7:] = 0  # padding
    return {"tokens": tokens, "ages": torch.linspace(0, 60, t).repeat(b, 1), "values": torch.randn(b, t, generator=g),
            "has_value": torch.rand(b, t, generator=g) > 0.5, "next_target": torch.randint(-1, 5, (b, t), generator=g),
            "dt": torch.rand(b, t, generator=g) * 5, "loss_mask": tokens != 0}


def model(**kw):
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, dropout=0.0, max_len=16, value_encoding="mlp", **kw)
    return ModLEHR(cfg, n_tokens=20, n_targets=5).eval()


def test_registered_parts():
    for kind in ("attention", "norm", "ffn", "time_encoding", "value_encoding", "head"):
        assert registry.available(kind)


def test_forward_shape_and_loss():
    m = model()
    out = m(batch())
    assert out.shape == (2, 10, 5)
    loss, parts = m.loss(batch())
    assert torch.isfinite(loss) and set(parts) == {"nll_event", "nll_time"}


def test_swapping_parts_by_name():
    m = model(norm="rmsnorm", ffn="swiglu")
    assert torch.isfinite(m(batch())).all()


def test_causal():
    m, x = model(), batch()
    base = m(x)
    y = {k: v.clone() for k, v in x.items()}
    y["tokens"][0, 6] = (y["tokens"][0, 6] % 19) + 1
    y["values"][0, 6] += 3.0
    changed = m(y)
    assert torch.allclose(base[0, :6], changed[0, :6], atol=1e-5)  # earlier positions do not see later events
    assert not torch.allclose(base[0, 6], changed[0, 6])


def test_competing_risk_loss_matches_formula():
    log_rates = torch.log(torch.tensor([[[0.1, 0.3]], [[0.1, 0.3]]]))  # two positions, rates 0.1 and 0.3 per year
    next_target = torch.tensor([[1], [-1]])
    dt = torch.tensor([[2.0], [4.0]])
    loss, _ = CompetingRiskHead.loss(log_rates, next_target, dt, torch.ones(2, 1, dtype=torch.bool))
    event = -math.log(0.3) + 0.4 * 2.0  # target 1 after 2 years
    censored = 0.4 * 4.0  # nothing for 4 years
    assert abs(float(loss) - (event + censored) / 2) < 1e-5


def test_risk_is_a_probability():
    log_rates = torch.log(torch.tensor([[0.1, 0.3]]))
    marginal = CompetingRiskHead.risk(log_rates, horizon_years=5.0)  # each target on its own hazard
    assert torch.allclose(marginal, torch.tensor([[1 - math.exp(-0.5), 1 - math.exp(-1.5)]]))
    nxt = CompetingRiskHead.risk(log_rates, horizon_years=5.0, kind="next_event")  # one target at a time
    assert torch.all(nxt >= 0) and abs(float(nxt.sum()) - (1 - math.exp(-0.4 * 5))) < 1e-6
    assert torch.all(nxt < marginal)


def test_init_rates():
    head = CompetingRiskHead(8, 3)
    head.init_rates([9, 0, 99], exposure_years=100.0)
    rates = head(torch.zeros(1, 8)).exp()
    assert torch.allclose(rates, torch.tensor([[0.1, 0.01, 1.0]]), atol=1e-6)


def test_every_attention_is_causal():
    for name in registry.available("attention"):
        m, x = model(attention=name), batch()
        base = m(x)
        y = {k: v.clone() for k, v in x.items()}
        y["tokens"][0, 6] = (y["tokens"][0, 6] % 19) + 1
        assert torch.allclose(base[0, :6], m(y)[0, :6], atol=1e-5), name


def test_gqa_has_fewer_parameters():
    full = model(attention="mha").n_parameters()
    grouped = model(attention="gqa", attention_options={"n_kv_heads": 1}).n_parameters()
    assert grouped < full


def test_rope_age_depends_only_on_age_differences():
    # with no absolute age encoding, shifting every age by the same amount changes nothing
    m, x = model(attention="rope_age", time_encoding="none"), batch()
    y = {k: v.clone() for k, v in x.items()}
    y["ages"] = y["ages"] + 7.5
    assert torch.allclose(m(x), m(y), atol=1e-4)
    # but changing the gap between events does
    y["ages"][:, 5:] += 3.0
    assert not torch.allclose(m(x), m(y), atol=1e-4)


def test_alibi_age_depends_only_on_age_differences():
    m, x = model(attention="alibi_age", time_encoding="none"), batch()
    y = {k: v.clone() for k, v in x.items()}
    y["ages"] = y["ages"] + 12.0
    assert torch.allclose(m(x), m(y), atol=1e-4)
    y["ages"][:, 5:] += 3.0
    assert not torch.allclose(m(x), m(y), atol=1e-4)


def test_alibi_age_penalizes_older_events():
    from modlehr.model.layers import alibi_age

    attn = alibi_age(d_model=8, n_heads=2)
    assert torch.all(attn.slopes > 0) and attn.slopes[0, 0] > attn.slopes[0, 1]  # each head its own time span


def test_new_parts_build_and_train_step():
    for kw in ({"attention": "mla"}, {"attention": "local_age"}, {"attention_options": {"qk_norm": True}},
               {"ffn": "moe", "ffn_options": {"n_experts": 4, "top_k": 2}}):
        m = model(**kw).train()
        loss, parts = m.loss(batch())
        loss.backward()
        assert torch.isfinite(loss), kw
        if kw.get("ffn") == "moe":
            assert parts["aux"] > 0


def test_mla_uses_a_low_rank_latent():
    from modlehr.model.layers import mla

    attn = mla(d_model=32, n_heads=4, kv_rank=8)
    assert attn.kv[0].out_features == 8


def test_local_age_ignores_events_outside_the_window():
    m, x = model(attention="local_age", attention_options={"window_years": 10.0}), batch()
    # ages run 0 to 60 over 10 events, so the last event (age 60) cannot see the first (age 0)
    y = {k: v.clone() for k, v in x.items()}
    y["tokens"][0, 0] = (y["tokens"][0, 0] % 19) + 1
    out_x, out_y = m(x), m(y)
    assert torch.allclose(out_x[0, -1], out_y[0, -1], atol=1e-5)
    assert not torch.allclose(out_x[0, 1], out_y[0, 1], atol=1e-5)  # but an event at age 6.7 can


def test_moe_dense_and_sparse_dispatch_agree():
    """Running every expert and zeroing the unchosen ones gives the same outputs and gradients as
    sending each event only to its chosen experts."""
    from modlehr.model.layers import MixtureOfExperts

    torch.manual_seed(0)
    sparse = MixtureOfExperts(32, n_experts=4, top_k=2, dispatch="sparse").double()
    dense = MixtureOfExperts(32, n_experts=4, top_k=2, dispatch="dense").double()
    dense.load_state_dict(sparse.state_dict())
    x = torch.randn(3, 10, 32, dtype=torch.float64)
    results = []
    for m in (sparse, dense):
        y = m(x)
        (y.square().mean() + m.aux_loss).backward()
        results.append((y.detach(), {n: p.grad for n, p in m.named_parameters()}))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-12)
    assert all(torch.allclose(results[0][1][n], results[1][1][n], atol=1e-12) for n in results[0][1])


def test_padding_changes_no_prediction():
    """A row's outputs are the same whether its batch is padded to 64 or, next to a longer row, to 128."""
    import numpy as np

    from modlehr.prepare.dataset import collate

    def row(n, seed):
        r = np.random.default_rng(seed)
        return {"tokens": r.integers(1, 20, n), "ages": np.sort(r.uniform(0, 80, n)).astype(np.float32),
                "values": np.full(n, np.nan, np.float32), "next_target": np.full(n, -1), "dt": np.zeros(n, np.float32),
                "loss_mask": np.zeros(n, bool)}

    torch.manual_seed(0)
    model = ModLEHR(ModelConfig(d_model=32, n_layers=2, n_heads=2, dropout=0.0, max_len=256), n_tokens=20,
                    n_targets=5).eval()
    short, long = row(5, 0), row(100, 1)
    alone, beside = collate([short]), collate([short, long])
    assert alone["tokens"].shape[1] == 64 and beside["tokens"].shape[1] == 128
    with torch.no_grad():
        assert torch.allclose(model(alone)[0, :5], model(beside)[0, :5], atol=1e-5)


def test_moe_balancing_ignores_padding():
    from modlehr.model.layers import MixtureOfExperts

    torch.manual_seed(0)
    moe = MixtureOfExperts(16, n_experts=4, top_k=2)
    x = torch.randn(1, 6, 16)
    moe(x)
    real_only = moe.aux_loss.clone()
    moe.valid = torch.tensor([True] * 6 + [False] * 4)  # the same 6 events and 4 padding positions
    moe(torch.cat([x, torch.randn(1, 4, 16)], dim=1))
    assert torch.allclose(moe.aux_loss, real_only)
