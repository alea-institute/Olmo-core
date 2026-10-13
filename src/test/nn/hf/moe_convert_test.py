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


def _tiny_moe_transformer_config(*, reordered_norm: bool, dropless: bool = True, **kwargs):
    from olmo_core.nn.transformer.config import TransformerConfig

    return TransformerConfig.llama_like_moe(
        d_model=D_MODEL,
        vocab_size=64,
        n_layers=NUM_LAYERS,
        n_heads=2,
        n_kv_heads=kwargs.pop("n_kv_heads", 1),
        num_experts=NUM_EXPERTS,
        top_k=2,
        expert_hidden_size=EXPERT_HIDDEN,
        dropless=dropless,
        reordered_norm=reordered_norm,
        qk_norm=True,
        layer_norm_eps=1e-6,
        tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
        **kwargs,
    )


def _per_expert_state(hf_model) -> dict:
    from olmo_core.nn.hf.convert import unfuse_hf_moe_experts

    return unfuse_hf_moe_experts({k: v.detach() for k, v in hf_model.state_dict().items()})


def test_fuse_unfuse_hf_moe_experts_roundtrip():
    from olmo_core.nn.hf.convert import (
        fuse_hf_moe_experts,
        has_fused_hf_moe_experts,
        unfuse_hf_moe_experts,
    )

    per_expert = {"model.norm.weight": torch.randn(D_MODEL)}
    for i in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            prefix = f"model.layers.{i}.mlp.experts.{e}."
            per_expert[prefix + "gate_proj.weight"] = torch.randn(EXPERT_HIDDEN, D_MODEL)
            per_expert[prefix + "up_proj.weight"] = torch.randn(EXPERT_HIDDEN, D_MODEL)
            per_expert[prefix + "down_proj.weight"] = torch.randn(D_MODEL, EXPERT_HIDDEN)

    fused = fuse_hf_moe_experts(per_expert)
    assert has_fused_hf_moe_experts(fused) and not has_fused_hf_moe_experts(per_expert)
    assert fused["model.layers.0.mlp.experts.gate_up_proj"].shape == (
        NUM_EXPERTS,
        2 * EXPERT_HIDDEN,
        D_MODEL,
    )
    assert fused["model.layers.0.mlp.experts.down_proj"].shape == (
        NUM_EXPERTS,
        D_MODEL,
        EXPERT_HIDDEN,
    )
    roundtrip = unfuse_hf_moe_experts(fused)
    assert set(roundtrip) == set(per_expert)
    for key, value in per_expert.items():
        torch.testing.assert_close(roundtrip[key], value, rtol=0, atol=0)

    del per_expert["model.layers.1.mlp.experts.2.up_proj.weight"]
    with pytest.raises(ValueError, match="non-contiguous"):
        fuse_hf_moe_experts(per_expert)


def test_fused_hf_moe_experts_match_transformers_layout(tmp_path):
    """Our fused layout must be the one ``transformers`` uses in memory (when it fuses at all)."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from olmo_core.nn.hf.convert import fuse_hf_moe_experts, has_fused_hf_moe_experts

    hf_model = AutoModelForCausalLM.from_config(_hf_moe_config())
    in_memory = hf_model.state_dict()
    if not has_fused_hf_moe_experts(in_memory):
        pytest.skip("The installed transformers version keeps per-expert tensors in memory")
    hf_model.save_pretrained(tmp_path)
    on_disk = load_file(str(tmp_path / "model.safetensors"))
    assert not has_fused_hf_moe_experts(on_disk)
    fused = fuse_hf_moe_experts(on_disk)
    for key, value in in_memory.items():
        if "lm_head" in key and key not in fused:
            continue
        torch.testing.assert_close(fused[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dropless", [pytest.param(True, id="dropless"), pytest.param(False, id="moe_mlp")]
)
def test_save_and_load_hf_moe_model(tmp_path, dropless: bool):
    from transformers import AutoModelForCausalLM

    from olmo_core.nn.hf.checkpoint import load_hf_model, save_hf_model

    model = _tiny_moe_transformer_config(reordered_norm=True, dropless=dropless).build(
        init_device="cpu"
    )
    model.init_weights(device=torch.device("cpu"))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_hf_model(tmp_path / "hf", state, model)

    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path / "hf")
    hf_state = _per_expert_state(hf_model)
    x = torch.randn(NUM_EXPERTS, 3, D_MODEL)
    for i in range(NUM_LAYERS):
        mlp = model.blocks[str(i)].feed_forward_moe.experts.mlp
        if dropless:
            mlp._gmm = None
        with torch.no_grad():
            expected = _olmo_core_expert_outputs(mlp, x)
        torch.testing.assert_close(_hf_expert_outputs(hf_state, i, x), expected)

    # HF -> Olmo-core: the loaded state must equal the original exactly.
    target = {k: torch.zeros_like(v) for k, v in state.items()}
    load_hf_model(tmp_path / "hf", target)
    for key, value in state.items():
        torch.testing.assert_close(target[key], value, rtol=0, atol=0, msg=key)
