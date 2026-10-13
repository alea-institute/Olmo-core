import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3MoeConfig

from olmo_core.nn.hf.convert import (
    convert_state_from_hf,
    convert_state_to_hf,
    infer_dropless_expert_layout,
)
from olmo_core.nn.moe.mlp import DroplessMoEMLP, MoEMLP

D_MODEL, EXPERT_HIDDEN, NUM_EXPERTS, NUM_LAYERS = 16, 8, 4, 2


def _hf_moe_config(**kwargs) -> Qwen3MoeConfig:
    defaults = dict(
        vocab_size=64,
        hidden_size=D_MODEL,
        intermediate_size=EXPERT_HIDDEN,
        moe_intermediate_size=EXPERT_HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
    )
    defaults.update(kwargs)
    return Qwen3MoeConfig(**defaults)


def _build_expert_mlp(dropless: bool, d_model: int = D_MODEL, hidden_size: int = EXPERT_HIDDEN):
    torch.manual_seed(0)
    cls = DroplessMoEMLP if dropless else MoEMLP
    mlp = cls(d_model=d_model, hidden_size=hidden_size, num_experts=NUM_EXPERTS, init_device="cpu")
    if dropless:
        mlp._gmm = None  # type: ignore[assignment]  # exact per-expert loop, runs on CPU
    return mlp


def _olmo_core_expert_outputs(mlp, x: torch.Tensor) -> torch.Tensor:
    """Per-expert outputs of shape ``(num_experts, N, d_model)`` computed by the Olmo-core module."""
    num_experts, n = x.shape[0], x.shape[1]
    if isinstance(mlp, DroplessMoEMLP):
        out = mlp(x.reshape(num_experts * n, -1), torch.full((num_experts,), n))
        return out.reshape(num_experts, n, -1)
    return mlp(x)


def _hf_expert_outputs(hf_state, layer: int, x: torch.Tensor) -> torch.Tensor:
    """Per-expert outputs with the HF SwiGLU expert semantics: ``down(silu(gate(x)) * up(x))``."""
    outs = []
    for e in range(x.shape[0]):
        prefix = f"model.layers.{layer}.mlp.experts.{e}."
        gate = F.linear(x[e], hf_state[prefix + "gate_proj.weight"])
        up = F.linear(x[e], hf_state[prefix + "up_proj.weight"])
        outs.append(F.linear(F.silu(gate) * up, hf_state[prefix + "down_proj.weight"]))
    return torch.stack(outs)


def _expert_state(mlp, num_layers: int = NUM_LAYERS):
    return {
        f"blocks.{i}.feed_forward_moe.experts.mlp.{name}": getattr(mlp, name).detach().clone()
        for i in range(num_layers)
        for name in ("w1", "w2", "w3")
    }


@pytest.mark.parametrize(
    "dropless", [pytest.param(True, id="dropless"), pytest.param(False, id="moe_mlp")]
)
@pytest.mark.parametrize("explicit_layout", [True, False])
def test_convert_expert_weights_to_hf_matches_expert_outputs(dropless: bool, explicit_layout: bool):
    mlp = _build_expert_mlp(dropless)
    olmo_core_state = _expert_state(mlp)
    hf_config = _hf_moe_config()

    hf_state = convert_state_to_hf(
        hf_config, olmo_core_state, dropless_experts=dropless if explicit_layout else None
    )

    for e in range(NUM_EXPERTS):
        prefix = f"model.layers.0.mlp.experts.{e}."
        assert hf_state[prefix + "gate_proj.weight"].shape == (EXPERT_HIDDEN, D_MODEL)
        assert hf_state[prefix + "up_proj.weight"].shape == (EXPERT_HIDDEN, D_MODEL)
        assert hf_state[prefix + "down_proj.weight"].shape == (D_MODEL, EXPERT_HIDDEN)

    x = torch.randn(NUM_EXPERTS, 5, D_MODEL)
    with torch.no_grad():
        expected = _olmo_core_expert_outputs(mlp, x)
    torch.testing.assert_close(_hf_expert_outputs(hf_state, 0, x), expected)


@pytest.mark.parametrize(
    "dropless", [pytest.param(True, id="dropless"), pytest.param(False, id="moe_mlp")]
)
def test_convert_expert_weights_roundtrip(dropless: bool):
    mlp = _build_expert_mlp(dropless)
    olmo_core_state = _expert_state(mlp)
    hf_config = _hf_moe_config()

    hf_state = convert_state_to_hf(hf_config, olmo_core_state, dropless_experts=dropless)
    roundtrip = convert_state_from_hf(
        hf_config, hf_state, model_type="qwen3_moe", dropless_experts=dropless
    )

    assert set(roundtrip) == set(olmo_core_state)
    for key, value in olmo_core_state.items():
        torch.testing.assert_close(roundtrip[key], value, rtol=0, atol=0)


def test_infer_dropless_expert_layout():
    hf_config = _hf_moe_config()
    assert infer_dropless_expert_layout(hf_config, _expert_state(_build_expert_mlp(True))) is True
    assert infer_dropless_expert_layout(hf_config, _expert_state(_build_expert_mlp(False))) is False
    assert infer_dropless_expert_layout(hf_config, {"embeddings.weight": torch.zeros(1)}) is None

    # Ambiguous: the two layouts coincide when the expert hidden size equals d_model.
    square_config = _hf_moe_config(moe_intermediate_size=D_MODEL, intermediate_size=D_MODEL)
    square_state = _expert_state(_build_expert_mlp(True, hidden_size=D_MODEL))
    assert infer_dropless_expert_layout(square_config, square_state) is None

    with pytest.raises(ValueError, match="matches neither"):
        infer_dropless_expert_layout(
            hf_config, {"blocks.0.feed_forward_moe.experts.mlp.w1": torch.zeros(3, 3)}
        )
