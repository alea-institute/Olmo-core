import pytest
from transformers import Olmo2Config

from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None


def test_get_hf_config():
    vocab_size = 4096
    model_config = TransformerConfig.olmo2_190M(vocab_size)
    model = model_config.build()

    hf_config = get_hf_config(model)
    assert isinstance(hf_config, Olmo2Config)
    assert hf_config.hidden_size == model_config.d_model
    assert hf_config.intermediate_size == 3072
    assert hf_config.num_hidden_layers == model_config.n_layers


def test_get_hf_config_default_block():
    vocab_size = 4096
    model_config = TransformerConfig.llama2_271M(vocab_size)
    model = model_config.build()

    with pytest.raises(NotImplementedError):
        get_hf_config(model)


def test_get_hf_config_moe():
    vocab_size = 4096
    model_config = TransformerConfig.smallmoe(vocab_size)
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert model_config.block.feed_forward_moe is not None
    # FlexOlmo has no shared expert.
    model_config.block.feed_forward_moe.shared_mlp = None
    model = model_config.build()

    if FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")

    hf_config = get_hf_config(model, max_sequence_length=4096)
    assert isinstance(hf_config, FlexOlmoConfig)
    assert hf_config.hidden_size == model_config.d_model
    assert hf_config.intermediate_size == model_config.block.feed_forward_moe.hidden_size
    assert hf_config.num_hidden_layers == model_config.n_layers
    assert hf_config.num_experts == model_config.block.feed_forward_moe.num_experts
    assert hf_config.num_experts_per_tok == model_config.block.feed_forward_moe.router.top_k
    assert hf_config.max_position_embeddings == 4096
    assert hf_config.norm_topk_prob is False
    assert hf_config.rms_norm_eps == 1e-6


def test_get_hf_config_moe_norm_topk_prob():
    if FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")
    model_config = TransformerConfig.smallmoe(4096)
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert model_config.block.feed_forward_moe is not None
    model_config.block.feed_forward_moe.shared_mlp = None
    model_config.block.feed_forward_moe.router.normalize_expert_weights = 1.0
    hf_config = get_hf_config(model_config.build())
    assert hf_config.norm_topk_prob is True
    assert hf_config.max_position_embeddings == -1


def _smallmoe_without_shared_expert() -> TransformerConfig:
    model_config = TransformerConfig.smallmoe(4096, n_layers=2)
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert model_config.block.feed_forward_moe is not None
    model_config.block.feed_forward_moe.shared_mlp = None
    return model_config


def _with_shared_expert(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.shared_mlp = FeedForwardConfig(hidden_size=64, bias=False)  # type: ignore[union-attr]


def _with_sigmoid_router(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.router.gating_function = MoERouterGatingFunction.sigmoid  # type: ignore[union-attr]


def _with_score_bias(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.router.bias_gamma = 0.001  # type: ignore[union-attr]


def _with_l2_normalized_weights(config: TransformerConfig) -> None:
    config.block.feed_forward_moe.router.normalize_expert_weights = 2.0  # type: ignore[union-attr]


def _with_head_qk_norm(config: TransformerConfig) -> None:
    config.block.sequence_mixer.use_head_qk_norm = True  # type: ignore[union-attr]


def _with_pre_norm_blocks(config: TransformerConfig) -> None:
    config.block.name = TransformerBlockType.moe  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "modify, message",
    [
        pytest.param(_with_shared_expert, "shared expert", id="shared-expert"),
        pytest.param(_with_sigmoid_router, "gating", id="sigmoid-router"),
        pytest.param(_with_score_bias, "score bias", id="score-bias"),
        pytest.param(_with_l2_normalized_weights, "normalize_expert_weights", id="l2-norm"),
        pytest.param(_with_head_qk_norm, "per-head QK-norm", id="head-qk-norm"),
        pytest.param(_with_pre_norm_blocks, "MoEReorderedNormTransformerBlock", id="pre-norm"),
    ],
)
def test_get_hf_config_flex_olmo_refuses_unsupported_features(modify, message: str):
    if FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")
    model_config = _smallmoe_without_shared_expert()
    modify(model_config)
    with pytest.raises(NotImplementedError, match=message):
        get_hf_config(model_config.build())
