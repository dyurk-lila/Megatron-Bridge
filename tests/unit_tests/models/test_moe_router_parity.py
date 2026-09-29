# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HF -> Megatron router config parity for the bridged MoE families.

Each case builds a provider from an offline HF config that keeps the released checkpoint's
router hyperparameters, routes random logits through MCore with the provider's router fields,
and compares against a reference written from the HF modeling code's routing. Routing replay
must reproduce the HF combine weights at the replayed experts, fused or not.
"""

import dataclasses
import inspect
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import (
    DeepseekV3Config,
    DeepseekV4Config,
    Glm4MoeConfig,
    Glm4MoeLiteConfig,
    GlmMoeDsaConfig,
    MiMoV2FlashConfig,
    Qwen3_5MoeTextConfig,
    Qwen3MoeConfig,
    Qwen3NextConfig,
)

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge


pytestmark = pytest.mark.unit

NUM_TOKENS = 32
_SMALL = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=128,
    moe_intermediate_size=32,
    vocab_size=512,
)


@dataclasses.dataclass(frozen=True)
class HFGate:
    """Router hyperparameters as the HF modeling code reads them from the config."""

    score_function: str
    num_experts: int
    topk: int
    expert_bias: bool
    normalize: bool
    scaling_factor: float = 1.0
    num_groups: int = 1
    group_topk: int = 1


_SCORE_FUNCTIONS = {
    "softmax": lambda logits: logits.softmax(dim=-1),
    "sigmoid": torch.sigmoid,
    "sqrtsoftplus": lambda logits: F.softplus(logits).sqrt(),
}


def _hf_weights(scores: torch.Tensor, gate: HFGate, indices: torch.Tensor) -> torch.Tensor:
    weights = scores.gather(1, indices)
    if gate.normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * gate.scaling_factor


def _hf_route(logits: torch.Tensor, expert_bias: torch.Tensor, gate: HFGate) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing of the HF TopkRouter/SparseMoeBlock implementations: (weights, indices)."""
    scores = _SCORE_FUNCTIONS[gate.score_function](logits.float())
    choice = scores + expert_bias if gate.expert_bias else scores
    if gate.num_groups > 1:
        # DeepSeek-V3 node-limited routing: a group scores as the sum of its top-2 biased scores.
        grouped = choice.view(len(choice), gate.num_groups, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        keep = torch.zeros_like(group_scores, dtype=torch.bool)
        keep.scatter_(1, group_scores.topk(gate.group_topk, dim=-1).indices, True)
        choice = grouped.masked_fill(~keep.unsqueeze(-1), float("-inf")).flatten(1)
    indices = choice.topk(gate.topk, dim=-1).indices
    return _hf_weights(scores, gate, indices), indices


def _deepseek_gate(cfg) -> HFGate:
    return HFGate(
        score_function="sigmoid",
        num_experts=cfg.n_routed_experts,
        topk=cfg.num_experts_per_tok,
        expert_bias=True,
        normalize=cfg.norm_topk_prob,
        scaling_factor=cfg.routed_scaling_factor,
        num_groups=cfg.n_group,
        group_topk=cfg.topk_group,
    )


def _qwen_gate(cfg, *, normalize: bool) -> HFGate:
    return HFGate("softmax", cfg.num_experts, cfg.num_experts_per_tok, expert_bias=False, normalize=normalize)


def _auto_provider(cfg):
    return AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)


def _deepseek_v4_config() -> DeepseekV4Config:
    compress_rope = {
        "rope_type": "yarn",
        "rope_theta": 160000.0,
        "factor": 16.0,
        "original_max_position_embeddings": 65536,
    }
    return DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        **_SMALL,
        n_routed_experts=256,
        num_experts_per_tok=6,
        routed_scaling_factor=1.5,
        mlp_layer_types=["hash_moe", "moe"],
        rope_parameters={"main": {"rope_type": "default", "rope_theta": 10000.0}, "compress": compress_rope},
    )


def _kimi_k3_provider_and_gate():
    # K3's modeling code is remote; its gate is the DeepSeek-V3 sigmoid gate with renormalization.
    text_config = SimpleNamespace(
        **_SMALL,
        attention_bias=False,
        attn_res_block_size=12,
        first_k_dense_replace=1,
        head_dim=16,
        hidden_act="situ",
        initializer_range=0.006,
        kv_lora_rank=16,
        linear_attn_config={
            "gate_lower_bound": -5.0,
            "head_dim": 16,
            "kda_layers": [1],
            "num_heads": 4,
            "short_conv_kernel_size": 4,
        },
        max_position_embeddings=4096,
        moe_layer_freq=1,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        num_experts=896,
        num_experts_per_token=16,
        num_shared_experts=2,
        q_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        rms_norm_eps=1e-5,
        routed_expert_hidden_size=32,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
        topk_group=1,
        torch_dtype="bfloat16",
        use_grouped_topk=True,
        v_head_dim=16,
    )
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"], text_config=text_config, torch_dtype="bfloat16"
    )
    gate = HFGate(
        score_function=text_config.moe_router_activation_func,
        num_experts=text_config.num_experts,
        topk=text_config.num_experts_per_token,
        expert_bias=True,
        normalize=True,
        scaling_factor=text_config.routed_scaling_factor,
        num_groups=text_config.num_expert_group,
        group_topk=text_config.topk_group,
    )
    return KimiK3Bridge().provider_bridge(pretrained), gate


def _from_hf(make_config: Callable, make_gate: Callable) -> Callable:
    def build():
        cfg = make_config()
        return _auto_provider(cfg), make_gate(cfg)

    return build


# Router hyperparameters follow the released checkpoints; everything else is shrunk.
CASES = {
    "deepseek_v3": _from_hf(
        lambda: DeepseekV3Config(
            architectures=["DeepseekV3ForCausalLM"],
            **_SMALL,
            n_routed_experts=256,
            num_experts_per_tok=8,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
            first_k_dense_replace=1,
        ),
        _deepseek_gate,
    ),
    "kimi_k2": _from_hf(
        lambda: DeepseekV3Config(
            architectures=["KimiK2ForCausalLM"],
            **_SMALL,
            n_routed_experts=384,
            num_experts_per_tok=8,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=2.827,
            first_k_dense_replace=1,
            scoring_func="sigmoid",
        ),
        _deepseek_gate,
    ),
    "kimi_k3": _kimi_k3_provider_and_gate,
    "deepseek_v4": _from_hf(
        _deepseek_v4_config,
        lambda cfg: HFGate(
            cfg.scoring_func,
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            expert_bias=True,
            normalize=True,
            scaling_factor=cfg.routed_scaling_factor,
        ),
    ),
    "qwen3_moe": _from_hf(
        lambda: Qwen3MoeConfig(
            architectures=["Qwen3MoeForCausalLM"],
            **_SMALL,
            num_experts=128,
            num_experts_per_tok=8,
            norm_topk_prob=True,
        ),
        lambda cfg: _qwen_gate(cfg, normalize=cfg.norm_topk_prob),
    ),
    "qwen3_moe_unnormalized": _from_hf(
        lambda: Qwen3MoeConfig(
            architectures=["Qwen3MoeForCausalLM"],
            **_SMALL,
            num_experts=128,
            num_experts_per_tok=8,
            norm_topk_prob=False,
        ),
        lambda cfg: _qwen_gate(cfg, normalize=cfg.norm_topk_prob),
    ),
    "qwen3_next": _from_hf(
        lambda: Qwen3NextConfig(
            architectures=["Qwen3NextForCausalLM"],
            **_SMALL,
            num_experts=512,
            num_experts_per_tok=10,
            norm_topk_prob=True,
        ),
        lambda cfg: _qwen_gate(cfg, normalize=cfg.norm_topk_prob),
    ),
    "qwen3_next_no_norm": _from_hf(
        lambda: Qwen3NextConfig(
            architectures=["Qwen3NextForCausalLM"],
            **_SMALL,
            num_experts=512,
            num_experts_per_tok=10,
            norm_topk_prob=False,
        ),
        lambda cfg: _qwen_gate(cfg, normalize=cfg.norm_topk_prob),
    ),
    "qwen3_5_moe": _from_hf(
        lambda: Qwen3_5MoeTextConfig(
            architectures=["Qwen3_5MoeForCausalLM"], **_SMALL, num_experts=512, num_experts_per_tok=10
        ),
        # Qwen3.5 always renormalizes the top-k probabilities.
        lambda cfg: _qwen_gate(cfg, normalize=True),
    ),
    "glm4_5": _from_hf(
        lambda: Glm4MoeConfig(
            architectures=["Glm4MoeForCausalLM"],
            **_SMALL,
            n_routed_experts=160,
            num_experts_per_tok=8,
            routed_scaling_factor=2.5,
        ),
        _deepseek_gate,
    ),
    "glm4_7_flash": _from_hf(
        lambda: Glm4MoeLiteConfig(
            architectures=["Glm4MoeLiteForCausalLM"],
            **_SMALL,
            n_routed_experts=64,
            num_experts_per_tok=4,
            routed_scaling_factor=1.8,
        ),
        _deepseek_gate,
    ),
    "glm5": _from_hf(
        lambda: GlmMoeDsaConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            **_SMALL,
            n_routed_experts=256,
            num_experts_per_tok=8,
            routed_scaling_factor=2.5,
            first_k_dense_replace=1,
            indexer_rope_interleave=True,
        ),
        _deepseek_gate,
    ),
    "mimo_v2_flash": _from_hf(
        # Native configs have no scoring_func; MiMo always gates with sigmoid.
        lambda: MiMoV2FlashConfig(
            architectures=["MiMoV2FlashForCausalLM"],
            **_SMALL,
            n_routed_experts=256,
            num_experts_per_tok=8,
            routed_scaling_factor=None,
            rope_theta=5000000,
            swa_rope_theta=10000,
            hybrid_layer_pattern=[0, 1],
            sliding_window_size=128,
            layernorm_epsilon=1e-5,
            v_head_dim=16,
            swa_num_key_value_heads=4,
            attention_value_scale=0.707,
            moe_layer_freq=[0, 1],
        ),
        _deepseek_gate,
    ),
}


def _fused_router_supports(arg_name: str | None = None) -> bool:
    fused_topk = moe_utils.fused_topk_with_score_function
    if not torch.cuda.is_available() or fused_topk is None:
        return False
    return arg_name is None or arg_name in inspect.signature(fused_topk).parameters


def _fused_params(arg_name: str | None = None) -> list:
    reason = "needs CUDA and TE's fused router" + (f" with `{arg_name}`" if arg_name else "")
    fused = pytest.param(
        True,
        id="fused",
        marks=[pytest.mark.gpu, pytest.mark.skipif(not _fused_router_supports(arg_name), reason=reason)],
    )
    return [pytest.param(False, id="unfused"), fused]


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def provider_and_gate(request):
    return CASES[request.param]()


@pytest.fixture(autouse=True)
def _clear_router_replay():
    yield
    RouterReplay.clear_global_router_replay_instances()


def _inputs(gate: HFGate, fused: bool) -> tuple[torch.Tensor, torch.Tensor]:
    device = "cuda" if fused else "cpu"
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(NUM_TOKENS, gate.num_experts, generator=generator).to(device)
    expert_bias = (0.1 * torch.randn(gate.num_experts, generator=generator)).to(device)
    return logits, expert_bias


def _mcore_route(provider, logits, expert_bias, *, fused, router_replay=None):
    """Route with MCore exactly as TopKRouter does for this provider's router fields."""
    return moe_utils.topk_routing_with_score_function(
        logits,
        provider.moe_router_topk,
        use_pre_softmax=provider.moe_router_pre_softmax,
        num_groups=provider.moe_router_num_groups,
        group_topk=provider.moe_router_group_topk,
        scaling_factor=provider.moe_router_topk_scaling_factor,
        score_function=provider.moe_router_score_function,
        expert_bias=expert_bias if provider.moe_router_enable_expert_bias else None,
        fused=fused,
        router_replay=router_replay,
    )


def _dense(values: torch.Tensor, indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.zeros(len(indices), num_experts, dtype=values.dtype, device=values.device).scatter(1, indices, values)


def _assert_routing(probs, routing_map, weights, indices, num_experts):
    expected_map = _dense(torch.ones_like(weights, dtype=torch.bool), indices, num_experts)
    torch.testing.assert_close(routing_map, expected_map, rtol=0, atol=0)
    torch.testing.assert_close(probs.float(), _dense(weights, indices, num_experts), rtol=1e-5, atol=1e-6)


def test_router_config_matches_hf(provider_and_gate):
    provider, gate = provider_and_gate
    assert provider.num_moe_experts == gate.num_experts
    assert provider.moe_router_topk == gate.topk
    assert provider.moe_router_score_function == gate.score_function
    assert provider.moe_router_enable_expert_bias == gate.expert_bias
    assert (provider.moe_router_topk_scaling_factor or 1.0) == pytest.approx(gate.scaling_factor)
    assert (provider.moe_router_num_groups or 1, provider.moe_router_group_topk or 1) == (
        gate.num_groups,
        gate.group_topk,
    )


@pytest.mark.parametrize("fused", _fused_params())
def test_routing_matches_hf(provider_and_gate, fused):
    provider, gate = provider_and_gate
    logits, expert_bias = _inputs(gate, fused)
    weights, indices = _hf_route(logits, expert_bias, gate)
    probs, routing_map = _mcore_route(provider, logits, expert_bias, fused=fused)
    _assert_routing(probs, routing_map, weights, indices, gate.num_experts)


@pytest.mark.parametrize("fused", _fused_params("precomputed_indices"))
def test_replay_forward_matches_hf_weights_at_replayed_experts(provider_and_gate, fused):
    """Replayed experts get the HF combine weights; this is also DeepSeek-V4's hash-layer math."""
    provider, gate = provider_and_gate
    logits, expert_bias = _inputs(gate, fused)
    generator = torch.Generator().manual_seed(1)
    targets = torch.rand(NUM_TOKENS, gate.num_experts, generator=generator).argsort(dim=-1)[:, : gate.topk]
    targets = targets.to(logits.device)
    router_replay = RouterReplay()
    router_replay.set_target_indices(targets)
    router_replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    probs, routing_map = _mcore_route(provider, logits, expert_bias, fused=fused, router_replay=router_replay)

    scores = _SCORE_FUNCTIONS[gate.score_function](logits.float())
    _assert_routing(probs, routing_map, _hf_weights(scores, gate, targets), targets, gate.num_experts)


@pytest.mark.parametrize("fused", _fused_params("topk_indices"))
def test_record_captures_hf_selection(provider_and_gate, fused):
    provider, gate = provider_and_gate
    logits, expert_bias = _inputs(gate, fused)
    router_replay = RouterReplay()
    router_replay.set_router_replay_action(RouterReplayAction.RECORD)

    _mcore_route(provider, logits, expert_bias, fused=fused, router_replay=router_replay)

    recorded = router_replay.get_recorded_indices()
    assert recorded is not None, "RECORD did not record the selected experts"
    _, indices = _hf_route(logits, expert_bias, gate)
    torch.testing.assert_close(recorded.long().sort(dim=-1).values, indices.sort(dim=-1).values, rtol=0, atol=0)


_TRANSFORMER_CONFIG_FIELDS = {field.name for field in dataclasses.fields(TransformerConfig)}
# (num hash layers, hash vocab size) field names on MCore main and on MCore dev.
_HASH_FIELD_NAMES = [("moe_num_hash_layers", "hash_moe_vocab_size"), ("moe_n_hash_layers", "actual_vocab_size")]


@pytest.mark.xfail(
    "moe_n_hash_layers" not in _TRANSFORMER_CONFIG_FIELDS,
    strict=True,
    reason="the DeepSeek-V4 bridge sets MCore dev's hash-routing field names, which MCore main does not define, "
    "so hash routing is silently disabled; fixed by NVIDIA-NeMo/Megatron-Bridge#5860",
)
def test_deepseek_v4_hash_layers_reach_provider():
    cfg = _deepseek_v4_config()
    provider = _auto_provider(cfg)
    layers_field, vocab_field = next(names for names in _HASH_FIELD_NAMES if names[0] in _TRANSFORMER_CONFIG_FIELDS)
    assert getattr(provider, layers_field) == cfg.mlp_layer_types.count("hash_moe")
    assert getattr(provider, vocab_field) == cfg.vocab_size
