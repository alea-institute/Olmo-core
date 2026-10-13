from pathlib import Path
from typing import Callable

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3MoeConfig

from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.hf.checkpoint import load_hf_model, save_hf_model
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.layer_norm import LayerNormType
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig
from olmo_core.testing import requires_gpu

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None

VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM = 64, 16, 2, 4, 2, 8
NUM_EXPERTS, TOP_K, EXPERT_HIDDEN = 4, 2, 8


def _qwen3_moe_like_config(**kwargs) -> TransformerConfig:
    """A tiny Olmo-core MoE that is exactly ``Qwen3MoeForCausalLM`` (``head_dim * n_heads != d_model``)."""
    config = TransformerConfig.llama_like_moe(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_hidden_size=EXPERT_HIDDEN,
        dropless=kwargs.pop("dropless", True),
        reordered_norm=False,
        qk_norm=True,
        use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
        layer_norm_name=LayerNormType.qwen_rms,
        layer_norm_eps=1e-6,
        rope_theta=10_000,
        tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
        **kwargs,
    )
    config.block.feed_forward_moe.router.normalize_expert_weights = 1.0  # type: ignore[union-attr]
    return config


def _flex_olmo_like_config() -> TransformerConfig:
    return TransformerConfig.llama_like_moe(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=2,
        n_kv_heads=1,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_hidden_size=EXPERT_HIDDEN,
        dropless=True,
        reordered_norm=True,
        qk_norm=True,
        layer_norm_eps=1e-6,
        rope_theta=10_000,
    )


def test_get_hf_config_qwen3_moe():
    model = _qwen3_moe_like_config().build(init_device="cpu")
    hf_config = get_hf_config(model, max_sequence_length=256)
    assert isinstance(hf_config, Qwen3MoeConfig)
    assert hf_config.hidden_size == D_MODEL
    assert hf_config.num_hidden_layers == N_LAYERS
    assert hf_config.num_attention_heads == N_HEADS
    assert hf_config.num_key_value_heads == N_KV_HEADS
    assert hf_config.head_dim == HEAD_DIM
    assert hf_config.num_experts == NUM_EXPERTS
    assert hf_config.num_experts_per_tok == TOP_K
    assert hf_config.moe_intermediate_size == EXPERT_HIDDEN
    assert hf_config.norm_topk_prob is True
    assert hf_config.decoder_sparse_step == 1
    assert hf_config.mlp_only_layers == []
    assert hf_config.rms_norm_eps == 1e-6
    assert hf_config.max_position_embeddings == 256
    assert hf_config.tie_word_embeddings is True
    assert not hf_config.attention_bias


def test_get_hf_config_qwen3_moe_without_renormalization():
    config = _qwen3_moe_like_config()
    config.block.feed_forward_moe.router.normalize_expert_weights = None  # type: ignore[union-attr]
    hf_config = get_hf_config(config.build(init_device="cpu"))
    assert isinstance(hf_config, Qwen3MoeConfig)
    assert hf_config.norm_topk_prob is False


def _with_full_width_qk_norm(config: TransformerConfig) -> None:
    config.block.sequence_mixer.use_head_qk_norm = False  # type: ignore[union-attr]


def _without_qk_norm(config: TransformerConfig) -> None:
    config.block.sequence_mixer.qk_norm = None  # type: ignore[union-attr]


def _with_per_head_gains(config: TransformerConfig) -> None:
    config.block.sequence_mixer.qk_norm_per_head_gains = True  # type: ignore[union-attr]


def _with_shared_expert(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.shared_mlp = FeedForwardConfig(hidden_size=32, bias=False)  # type: ignore[union-attr]


def _with_sigmoid_router(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.router.gating_function = MoERouterGatingFunction.sigmoid  # type: ignore[union-attr]


def _with_score_bias(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.router.bias_gamma = 0.001  # type: ignore[union-attr]


def _with_l2_normalized_weights(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.router.normalize_expert_weights = 2.0  # type: ignore[union-attr]


def _with_dense_layers(config: TransformerConfig) -> None:
    config.block.name = TransformerBlockType.moe_hybrid  # type: ignore[union-attr]
    config.block.feed_forward = FeedForwardConfig(hidden_size=32, bias=False)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "modify, message",
    [
        pytest.param(_with_full_width_qk_norm, "full-width QK-norm", id="full-width-qk-norm"),
        pytest.param(_without_qk_norm, "no QK-norm", id="no-qk-norm"),
        pytest.param(_with_per_head_gains, "per-head QK-norm gains", id="per-head-gains"),
        pytest.param(_with_shared_expert, "shared expert", id="shared-expert"),
        pytest.param(_with_sigmoid_router, "gating", id="sigmoid-router"),
        pytest.param(_with_score_bias, "score bias", id="score-bias"),
        pytest.param(_with_l2_normalized_weights, "normalize_expert_weights", id="l2-norm"),
        pytest.param(_with_dense_layers, "MoETransformerBlock", id="hybrid-dense"),
    ],
)
def test_get_hf_config_qwen3_moe_refuses_unsupported_features(
    modify: Callable[[TransformerConfig], None], message: str
):
    config = _qwen3_moe_like_config()
    modify(config)
    with pytest.raises(NotImplementedError, match=message):
        get_hf_config(config.build(init_device="cpu"))


def test_save_hf_model_qwen3_moe(tmp_path: Path):
    model = _qwen3_moe_like_config().build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_hf_model(tmp_path / "hf", state, model, max_sequence_length=256)

    assert AutoConfig.from_pretrained(tmp_path / "hf").model_type == "qwen3_moe"
    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path / "hf")
    assert type(hf_model).__name__ == "Qwen3MoeForCausalLM"

    target = {k: torch.zeros_like(v) for k, v in state.items()}
    load_hf_model(tmp_path / "hf", target)
    for key, value in state.items():
        torch.testing.assert_close(target[key], value, rtol=0, atol=0, msg=key)


def _olmo_core_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    for block in model.blocks.values():
        block.feed_forward_moe.experts.mlp._gmm = None  # exact fp32 per-expert loop
    model.eval()
    with torch.no_grad():
        return model(input_ids=input_ids)


@pytest.fixture
def _no_tf32():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


@requires_gpu
@pytest.mark.parametrize(
    "build_config, hf_class",
    [
        pytest.param(_qwen3_moe_like_config, "Qwen3MoeForCausalLM", id="qwen3_moe"),
        pytest.param(_flex_olmo_like_config, "FlexOlmoForCausalLM", id="flex_olmo"),
    ],
)
def test_save_hf_moe_model_logits_match(tmp_path: Path, _no_tf32, build_config, hf_class: str):
    if hf_class == "FlexOlmoForCausalLM" and FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")
    torch.manual_seed(0)
    model = build_config().build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    save_hf_model(tmp_path / "hf", model.state_dict(), model, max_sequence_length=64)

    input_ids = torch.randint(0, VOCAB_SIZE, (2, 24), generator=torch.Generator().manual_seed(0))
    logits = _olmo_core_logits(model.to("cuda"), input_ids.to("cuda"))

    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path / "hf", dtype=torch.float32)
    assert type(hf_model).__name__ == hf_class
    hf_model = hf_model.to("cuda").eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids.to("cuda")).logits

    torch.testing.assert_close(hf_logits, logits, rtol=1e-4, atol=1e-4)


@requires_gpu
def test_load_hf_qwen3_moe_model_logits_match(tmp_path: Path, _no_tf32):
    torch.manual_seed(0)
    model = _qwen3_moe_like_config().build(init_device="cpu")
    hf_config = get_hf_config(model, max_sequence_length=64)
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    # Non-trivial norm gains so that a mis-mapped norm cannot pass.
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if "norm" in name:
                param.uniform_(0.5, 1.5)
    hf_model.save_pretrained(tmp_path / "hf")

    state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    load_hf_model(tmp_path / "hf", state)
    model.load_state_dict(state)

    input_ids = torch.randint(0, VOCAB_SIZE, (2, 24), generator=torch.Generator().manual_seed(0))
    logits = _olmo_core_logits(model.to("cuda"), input_ids.to("cuda"))
    hf_model = hf_model.to("cuda").eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids.to("cuda")).logits
    torch.testing.assert_close(hf_logits, logits, rtol=1e-4, atol=1e-4)
