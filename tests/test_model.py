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
    r = CompetingRiskHead.risk(torch.log(torch.tensor([[0.1, 0.3]])), horizon_years=5.0)
    assert torch.all(r >= 0) and float(r.sum()) <= 1.0
    assert abs(float(r.sum()) - (1 - math.exp(-0.4 * 5))) < 1e-6


def test_init_rates():
    head = CompetingRiskHead(8, 3)
    head.init_rates([9, 0, 99], exposure_years=100.0)
    rates = head(torch.zeros(1, 8)).exp()
    assert torch.allclose(rates, torch.tensor([[0.1, 0.01, 1.0]]), atol=1e-6)
