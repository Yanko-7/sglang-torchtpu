# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 TPU runtime hooks (compress ratios 0/1/2).

Every hook here passes V4 (no ratio-1/2 layers, no Engram, no vision layers)
through to ``original``. They are appended after sglang_tpu/hooks/dsv4.py so
that, on shared targets, they are the outer wrapper.

build_tpu_dsv41_kv_pool, get_tpu_dsv41_bytes_per_swa_token,
get_tpu_dsv41_bytes_per_full_token
  Why: KVCacheConfigurator._build_dsv4_kv_pool hardcodes DeepSeekV4TokenToKVPool
       and the V4 TPU hook rebinds it to DSV4TPUTokenToKVPool, which rejects
       ratios 1/2; DSV4PoolConfigurator budgets FlashMLA bytes per token.
  How: AROUND the three targets, bodies in
       sglang_tpu.models.deepseek_v41.memory_pool (owned by the pool module).
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: KVCacheConfigurator and DSV4PoolConfigurator query
       current_platform for the DSV4 pool class and its byte sizing.

select_tpu_dsv41_loader
  Why: each ~100 GB Engram table is one tensor, far above a per-rank Run:AI
       buffer, and every rank keeps only its TP row range of it.
  How: AFTER get_model_loader swaps in DSV41RunaiModelStreamerLoader for a
       target model with Engram layers. Routed experts take the generic
       EP-local Run:AI read (sglang_tpu.ep_weight_shard); Engram tables leave
       the stream and are read as this rank's row range.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: EngramEmbedding loads a pre-sliced row range from a
       model-owned checkpoint tensor filter.

load_tpu_dsv41_engram_rows
  Why: EngramEmbedding._load_rows slices its TP row range out of the full
       ~100 GB table, so the loader must hand it the whole table. The V4.1
       loader reads only this rank's rows (one contiguous range, concurrent
       pread), which the original would slice a second time.
  How: AROUND EngramEmbedding._load_rows; tensors tagged with ROW_START_ATTR
       are copied as-is after a row-range check, others take the original path.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: EngramEmbedding accepts a pre-sliced row range.
trace_tpu_platform_fact
  Why: the V4.1 shifted-mHC loop in DeepseekV4Model reads
       get_platform().is_blackwell inside the compiled forward;
       PlatformContext.__getattr__ uses object.__getattribute__, which Dynamo
       cannot trace, so torch.compile fails during TPU warmup.
  How: AROUND PlatformContext.__getattr__ resolves the fact through a
       torch._dynamo.assume_constant_result helper; facts are fixed once the
       process starts, before any model is traced.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: PlatformContext is Dynamo-traceable or V4.1 forward reads
       platform facts outside the compiled region.
tpu_dsv41_hc_combine
  Why: the shifted-mHC forward (DeepseekV4DecoderLayer._hc_combine and the
       model's final hc_head) calls the Triton hc_combine with no device
       guard; Dynamo cannot trace a Triton launch on TPU.
  How: AROUND sglang.kernels.ops.layernorm.mhc.hc_combine computes the same
       fp32 weighted sum of the hc_mult copies in torch for non-CUDA tensors.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: hc_combine dispatches on device like the other mHC kernels.

tpu_dsv41_engram_gate
  Why: the torch fallback of sglang.srt.layers.engram.engram_gate uses
       torch.copysign, which TorchTPU does not implement (aten::copysign.out).
  How: AROUND engram_gate keeps the original on CUDA; elsewhere computes the
       same gate with torch.where(dot >= 0, |dot|', -|dot|') for the sign.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: TorchTPU implements aten::copysign or engram_gate avoids it.

tpu_dsv41_engram_embed_init, tpu_dsv41_engram_embed_forward,
tpu_dsv41_hasher_forward
  Why: TPU serves Engram only from one host table per node shared by the DP
       ranks (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1, layout "shared"); the
       in-graph lookup of HBM tables returned wrong outputs under attention DP
       and its offloaded gather halted a SparseCore. _HostTable pins its
       mapping with cudaHostRegister, and the core lookup reads it from a GPU.
  How: AROUND EngramEmbedding.__init__ rejects any other placement on TPU and
       builds the table with cudaHostRegister stubbed out. The V4 compile
       policy (deepseek_v41/compile_policy.py) hashes on the host and stages
       raw rows after padding; inside the graph EngramHasher.forward returns
       the staged row ids and EngramEmbedding.forward dequantizes the staged
       rows.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: the device can read host tables, or the in-graph HBM lookup
       is correct under attention DP.

allow_tpu_dsv41_encoder_replay
  Why: validate_deepseek_v41_features rejects --enable-encoder-swa-bounded-replay
       on hardware other than CUDA and gfx950 with one inline check, so the TPU
       request window and folded replay rows cannot be enabled. Every other check
       (prefill graphs, CP, ROCm DP, PD, mixed chunk, LoRA, radix sessions) must
       still apply.
  How: AROUND the validator; for a V4.1 model on TPU with the flag it runs the
       original with the module's get_platform reporting CUDA and not ROCm, the
       only two platform facts the validator reads.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR3).
  Remove when: the validator asks current_platform whether it folds encoder
       SWA replay.

derive_tpu_dsv41_pool_sizes, build_tpu_dsv41_allocator, get_tpu_dsv41_swa_fixed_bytes
  Why: under --enable-encoder-swa-bounded-replay the core sizes the paged SWA pool
       at 0 tokens (SWA lives in the request window), which the TPU 8-page SWA
       floor in _derive_pool_sizes refuses; the allocator builder then takes the
       core PagedTokenToKVPoolAllocator, whose alloc paths launch Triton; and the
       configurator prices core's FlashMLA-packed window instead of the TPU
       window's bf16 pages.
  How: AROUND each target, bodies in sglang_tpu.models.deepseek_v41.memory_pool:
       skip the floor for a 0-token SWA size only, bind the module's
       PagedTokenToKVPoolAllocator to the TPU paged allocator, and swap the
       window bytes. Every other case calls the original unchanged.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR3).
  Remove when: the SWA floor skips pools without a paged SWA allocator, that
       allocator arm asks current_platform for its class, and the configurator
       asks the pool for its window bytes.

trace_tpu_env_default
  Why: EnvField._resolve_default calls callable(self.default) for lazily
       computed defaults (e.g. SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)),
       which Dynamo cannot trace; the V4.1 MoE DP path reads it in the forward.
  How: AROUND EnvField._resolve_default resolves the default through a
       torch._dynamo.assume_constant_result helper keyed by the env name;
       defaults depend only on the platform, fixed before any model is traced.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR1).
  Remove when: callable env defaults are resolved at import or are traceable.

tpu_dsv41_dspark_attention, tpu_dsv41_dspark_forward
  Why: DSparkAttention.forward writes its block KV with the CUDA FlashMLA
       writer and inverse-rotates its output, which CSA2 already does; the
       draft forward returns no next_draft_tokens, which TpuBlockSpecWorker
       reads for greedy drafts.
  How: AROUND both when the forward runs on the CSA2 backend
       (DSV41TPUAttnBackend; dspark.py): attention goes through the backend's
       prepare_mqa_qkv and non-causal forward, then wo_a / wo_b; the forward
       appends the greedy Markov chain in-graph, as vllm-torchtpu
       models/vllm/deepseek_v41/v2/dspark.py does.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: DSparkAttention reads its KV write and inverse RoPE from the
       attention backend and the V4 draft returns draft tokens.

tpu_dsv41_flush_engram_before_prebuilt
  Why: TpuDsv41DsparkDraftWorker.after_commit queues each verify's Engram
       history commit and the next target forward's host Engram staging
       applies the queue (engram_host); a PD prebuilt batch writes history rows
       in NgramEmbeddingManager._prepare_engram_history without staging.
  How: AROUND _prepare_engram_history applies the queued commits first for a
       prebuilt batch.
  Upstream PR or issue: not filed (V4.1 TPU bring-up, PR2).
  Remove when: the Engram history lives on the device and commits in-graph.
"""

from __future__ import annotations

import torch


def build_tpu_dsv41_kv_pool(original, kvc, *args, **kwargs):
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        build_tpu_dsv41_kv_pool as _impl,
    )

    return _impl(original, kvc, *args, **kwargs)


def get_tpu_dsv41_bytes_per_swa_token(original, configurator) -> float:
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        get_tpu_dsv41_bytes_per_swa_token as _impl,
    )

    return _impl(original, configurator)


def get_tpu_dsv41_bytes_per_full_token(original, configurator) -> float:
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        get_tpu_dsv41_bytes_per_full_token as _impl,
    )

    return _impl(original, configurator)


def allow_tpu_dsv41_encoder_replay(original, server_args):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.arg_groups import deepseek_v4_hook as validator
    from sglang.srt.runtime_context import get_platform

    cfg = validator.resolving_view(server_args)
    if (
        not get_platform().is_tpu
        or validator.model_config_of(server_args).hf_config.model_type != "deepseek_v41"
        or not cfg.enable_encoder_swa_bounded_replay
    ):
        return original(server_args)
    cuda = SimpleNamespace(is_cuda=True, is_hip=False)
    with patch.object(validator, "get_platform", lambda: cuda):
        return original(server_args)


def derive_tpu_dsv41_pool_sizes(original, kvc, *, config):
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        derive_tpu_dsv41_pool_sizes as _impl,
    )

    return _impl(original, kvc, config=config)


def build_tpu_dsv41_allocator(original, kvc, **kwargs):
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        build_tpu_dsv41_allocator as _impl,
    )

    return _impl(original, kvc, **kwargs)


def get_tpu_dsv41_swa_fixed_bytes(original, configurator) -> float:
    from sglang_tpu.models.deepseek_v41.memory_pool import (
        get_tpu_dsv41_swa_fixed_bytes as _impl,
    )

    return _impl(original, configurator)


def select_tpu_dsv41_loader(result, load_config, model_config=None):
    from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
    from sglang_tpu.ep_weight_shard import EPLocalRunaiModelStreamerLoader

    # The generic EP-local Run:AI hook may already have wrapped the stock loader.
    if (
        type(result) not in (RunaiModelStreamerLoader, EPLocalRunaiModelStreamerLoader)
        or model_config is None
        or load_config.draft_model_idx is not None
    ):
        return result
    from sglang_tpu.models.deepseek_v41.loader import (
        DSV41RunaiModelStreamerLoader,
        is_bundled_dspark_draft,
    )

    # Without Engram tables the generic EP-local loader does everything; a
    # bundled DSpark draft comes here to stream only its mtp.* tensors.
    if not (model_config.use_engram or is_bundled_dspark_draft(model_config)):
        return result

    return DSV41RunaiModelStreamerLoader(load_config)


def load_tpu_dsv41_engram_rows(original, embedding, param, loaded_weight):
    from sglang_tpu.models.deepseek_v41.engram_row_reader import load_local_rows

    return load_local_rows(original, embedding, param, loaded_weight)


_ORIGINAL_PLATFORM_GETATTR = None


@torch._dynamo.assume_constant_result
def _resolve_platform_fact(name: str):
    from sglang.srt.runtime_context import get_platform

    return _ORIGINAL_PLATFORM_GETATTR(get_platform(), name)


def trace_tpu_platform_fact(original, platform, name):
    """AROUND PlatformContext.__getattr__: facts are fixed per process, so
    Dynamo bakes them into the graph instead of tracing the override dict."""
    global _ORIGINAL_PLATFORM_GETATTR
    _ORIGINAL_PLATFORM_GETATTR = original
    return _resolve_platform_fact(name)


@torch._dynamo.assume_constant_result
def _resolve_env_default(name: str):
    from sglang.srt.environ import envs
    from sglang_tpu.envs import envs as tpu_envs

    # Class-level lookup: Envs instances refuse bare field access, and an
    # EnvField refuses bool(), so test membership explicitly.
    for owner in (type(envs), type(tpu_envs)):
        for cls in owner.__mro__:
            if name in vars(cls):
                default = vars(cls)[name].default
                return default() if callable(default) else default
    raise AttributeError(f"unknown env field {name}")


def trace_tpu_env_default(original, field):
    """AROUND EnvField._resolve_default: env defaults are fixed per process, so
    under Dynamo they are baked into the graph instead of tracing
    callable(default); eager reads take the original path."""
    if not torch.compiler.is_compiling():
        return original(field)
    return _resolve_env_default(field.name)


def tpu_dsv41_hc_combine(original, x_flat, pre, hc, out_dtype):
    """y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h], accumulated in fp32."""
    if x_flat.is_cuda:
        return original(x_flat, pre, hc, out_dtype)
    m = x_flat.shape[0]
    copies = x_flat.reshape(m, hc, -1).float()
    return (pre.float().unsqueeze(-1) * copies).sum(1).to(out_dtype)


def tpu_dsv41_engram_gate(original, x, kv, q_weight, k_weight, eps, clamp_value):
    """engram_gate without torch.copysign (no TorchTPU kernel)."""
    if x.is_cuda:
        return original(x, kv, q_weight, k_weight, eps, clamp_value)
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(clamp_value).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, magnitude, -magnitude))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def tpu_dsv41_hasher_forward(original, hasher, input_ids, forward_batch):
    """AROUND EngramHasher.forward: in the TPU graph, the host-staged row ids."""
    if forward_batch.tpu_engram_local_ids is not None:
        return forward_batch.tpu_engram_local_ids
    return original(hasher, input_ids, forward_batch)


def tpu_dsv41_engram_embed_init(original, embedding, *args, **kwargs):
    """AROUND EngramEmbedding.__init__: build host tables without the CUDA pin."""
    if torch.cuda.is_available():
        return original(embedding, *args, **kwargs)
    from sglang.srt.environ import envs

    if not (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get()
        and envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get() == "shared"
    ):
        raise ValueError(
            "DeepSeek-V4.1 Engram on TPU needs the shared host table: set "
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 and "
            "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared"
        )
    from sglang_tpu.models.deepseek_v41.engram_host import without_cuda_host_register

    with without_cuda_host_register():
        return original(embedding, *args, **kwargs)


def tpu_dsv41_engram_embed_forward(
    original, embedding, indices, forward_batch=None, **kwargs
):
    """AROUND EngramEmbedding.forward: dequantize the host-staged rows."""
    if indices.is_cuda or embedding.host_table is None:
        return original(embedding, indices, forward_batch, **kwargs)
    from sglang_tpu.models.deepseek_v41.engram_host import staged_lookup

    # The core host-table lookup reads a raw host pointer from the device.
    if forward_batch is None or forward_batch.tpu_engram_staged is None:
        raise RuntimeError(
            "DeepSeek-V4.1 Engram on TPU reads rows staged by the compiled-forward "
            "shim; this forward did not stage them"
        )
    return staged_lookup(embedding, indices, forward_batch)


def tpu_dsv41_dspark_attention(original, attn, positions, hidden_states, forward_batch):
    """AROUND DSparkAttention.forward: the draft on the CSA2 backend."""
    from sglang_tpu.models.deepseek_v41.dspark import (
        dspark_attention_forward,
        serves_dspark_draft,
    )

    if hidden_states.is_cuda or not serves_dspark_draft():
        return original(attn, positions, hidden_states, forward_batch)
    return dspark_attention_forward(attn, positions, hidden_states, forward_batch)


def tpu_dsv41_dspark_forward(original, model, input_ids, positions, *args, **kwargs):
    """AROUND DeepseekV4ForCausalLMDSpark.forward: add the greedy drafts."""
    from sglang_tpu.models.deepseek_v41.dspark import (
        dspark_forward,
        serves_dspark_draft,
    )

    if input_ids.is_cuda or not serves_dspark_draft():
        return original(model, input_ids, positions, *args, **kwargs)
    return dspark_forward(original, model, input_ids, positions, *args, **kwargs)


def tpu_dsv41_flush_engram_before_prebuilt(original, manager, batch):
    """AROUND NgramEmbeddingManager._prepare_engram_history: a PD prebuilt batch
    writes history rows directly, so queued verify commits land first."""
    from sglang_tpu.models.deepseek_v41 import engram_host

    hasher = manager.engram_hasher
    if batch.forward_mode.is_prebuilt() and engram_host.has_pending_engram_commits(
        hasher
    ):
        engram_host.flush_engram_commits(hasher)
    return original(manager, batch)


def _serves_dspark_on_csa2(target_runner) -> bool:
    """A DeepSeek-V4-family target on the CSA2 backend (ratio-1/2 layers)."""
    from sglang_tpu.models.deepseek_v41.memory_pool import uses_dsv41_pool

    return uses_dsv41_pool(getattr(target_runner.model_config, "compress_ratios", ()))


def _make_dspark_draft_worker(target_runner):
    from sglang_tpu.models.deepseek_v41.dspark_drafter import (
        TpuDsv41DsparkDraftWorker,
    )

    return TpuDsv41DsparkDraftWorker(target_runner)


def register_tpu_dsv41_draft_workers() -> None:
    """Serve DSPARK on a CSA2 target with the MoE draft drafter.

    TpuBlockSpecWorker's built-in DSPARK drafter takes only the dense
    DSparkDraftModel; the bundled V4-family draft is DeepseekV4ForCausalLMDSpark.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang_tpu.experimental import (
        register_block_draft_worker,
        registered_block_draft_workers,
    )

    algorithm = SpeculativeAlgorithm.DSPARK
    if any(
        factory is _make_dspark_draft_worker
        for _, factory in registered_block_draft_workers(algorithm)
    ):
        return
    register_block_draft_worker(
        algorithm=algorithm,
        matches=_serves_dspark_on_csa2,
        factory=_make_dspark_draft_worker,
    )


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    # Declares the ForwardBatch fields the Engram hooks read.
    from sglang_tpu.models.deepseek_v41 import engram_host  # noqa: F401

    kvc = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    cfg = "sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator"
    return [
        (f"{kvc}._build_dsv4_kv_pool", build_tpu_dsv41_kv_pool, HookType.AROUND),
        (
            f"{cfg}._get_bytes_per_swa_token",
            get_tpu_dsv41_bytes_per_swa_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_bytes_per_full_token",
            get_tpu_dsv41_bytes_per_full_token,
            HookType.AROUND,
        ),
        (
            f"{cfg}._get_swa_fixed_bytes",
            get_tpu_dsv41_swa_fixed_bytes,
            HookType.AROUND,
        ),
        (f"{kvc}._derive_pool_sizes", derive_tpu_dsv41_pool_sizes, HookType.AROUND),
        (
            f"{kvc}._build_token_to_kv_pool_allocator",
            build_tpu_dsv41_allocator,
            HookType.AROUND,
        ),
        (
            "sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v41_features",
            allow_tpu_dsv41_encoder_replay,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_loader.loader.get_model_loader",
            select_tpu_dsv41_loader,
            HookType.AFTER,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding._load_rows",
            load_tpu_dsv41_engram_rows,
            HookType.AROUND,
        ),
        (
            "sglang.srt.runtime_context.PlatformContext.__getattr__",
            trace_tpu_platform_fact,
            HookType.AROUND,
        ),
        (
            "sglang.srt.environ.EnvField._resolve_default",
            trace_tpu_env_default,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.layernorm.mhc.hc_combine",
            tpu_dsv41_hc_combine,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.engram_gate",
            tpu_dsv41_engram_gate,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher.forward",
            tpu_dsv41_hasher_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding.__init__",
            tpu_dsv41_engram_embed_init,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding.forward",
            tpu_dsv41_engram_embed_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4_dspark.DSparkAttention.forward",
            tpu_dsv41_dspark_attention,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.forward",
            tpu_dsv41_dspark_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager"
            ".NgramEmbeddingManager._prepare_engram_history",
            tpu_dsv41_flush_engram_before_prebuilt,
            HookType.AROUND,
        ),
    ]
