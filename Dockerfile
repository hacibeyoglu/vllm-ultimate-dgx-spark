# Ultimate vLLM image for DGX Spark (GB10 / sm_121a) — vLLM main base
# - Source: vLLM main @ 2bd895762 (2026-07-15) merged into the AEON tree
#     (aeon-main-dspark). The V1 DFlash carries are GONE BY DESIGN: upstream main
#     provides hybrid SWA+full DFlash drafters natively on Model-Runner-V2
#     (#47914 + #48135 causal-metadata fix), V2 handles rejected-token slots via
#     num_rejected + PAD_SLOT_ID by construction (no #41703 ctx-mask needed), and
#     V2's BlockTables design has no #43982-class padded-shape bug.
# - Still carried (NOT upstream): Triton NVFP4-KV (#44389 lineage, MRv2-wired,
#     kernel-optimized) — upstream's nvfp4 KV is FlashInfer trtllm-gen SM100-only;
#     aeon-cudagraph-align-nonfull (port of PR #46324, V1-gated);
#     aeon-uma-negative-estimate-clamp (port of PR #46932, GB10 unified memory).
# - NVFP4 KV layout note: upstream's #44455 packed-KV refactor moved K/V into the
#     content dim; the Triton NVFP4 path keeps its proven 5-dim split-KV layout via
#     a dtype-aware get_kv_cache_stride_order (see merge d06f8cb25).
# - MRv2: NO V1 pin. Main force-routes mixed SWA/full DFlash drafters (z-lab
#     Qwen3.6-27B: 4/5 layers SWA-2048) to Model-Runner-V2; hybrid GDN targets are
#     supported there (model_states/mamba_hybrid.py). Do NOT set
#     VLLM_USE_V2_MODEL_RUNNER=0 — the V1 tree no longer carries the DFlash fixes.
# - Deps: torch 2.11.0 + FlashInfer 0.6.13 + cutlass-dsl 4.5.2 unchanged from the
#     v0.25.0 image (main pins match). torchcodec (>=0.14 in cuda.txt) still
#     SKIPPED — 0.14.0 is ABI-broken on torch 2.11; vLLM guards the import
#     (PlaceholderModule) so absence degrades gracefully. apt ffmpeg kept.
# - Keep: TORCH_CUDA_ARCH_LIST=12.1a (NOT bare 12.1 — 12.0f x 12.1a -> 12.1a keeps
#     native NVFP4 CUTLASS; bare 12.1 -> FP4 kernels #ifdef out -> Marlin fallback),
#     GCC 12, transformers 5.12.1, xgrammar>=0.2.1, TurboQuant fork, humming-stub.
# - VALIDATION GATES before promoting to :latest (V2 runner on GB10 is unproven):
#     long-context DFlash acceptance at 2k/16k/33k, prefix-caching soak, c=64
#     concurrency sweep, throughput A/B vs :2026-07-14-v0.25.0, tools + multimodal.

FROM ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV TORCH_CUDA_ARCH_LIST="12.1a"
ENV ENABLE_NVFP4_SM100=0
ENV CCACHE_DISABLE=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV MAX_JOBS=12
ENV NVCC_THREADS=2
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.26.0.dev0+aeon.sm121a.main20260715"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.26.0.dev0+aeon.sm121a.main20260715"

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates ffmpeg \
        gcc-12 g++-12 \
        cuda-nvrtc-dev-13-0 \
        libcusparse-dev-13-0 \
        libcublas-dev-13-0 \
        libcusolver-dev-13-0 \
        libcufft-dev-13-0 \
        libcurand-dev-13-0 \
        libnvjitlink-dev-13-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# GCC >= 12 for C++20 (#44923); base ships 11.4
ENV CC=gcc-12
ENV CXX=g++-12
ENV CUDAHOSTCXX=g++-12

RUN CUDA_LIB=/usr/local/cuda-13.0/targets/sbsa-linux/lib && \
    if [ ! -f $CUDA_LIB/libnvrtc.so ] && [ -f $CUDA_LIB/libnvrtc.so.13.0.88 ]; then \
        ln -sf $CUDA_LIB/libnvrtc.so.13.0.88 $CUDA_LIB/libnvrtc.so; \
        echo "Created libnvrtc.so symlink"; \
    fi && \
    ls -la $CUDA_LIB/libnvrtc.so* | head -5

# Re-pin CUDA 13.0 toolkit + assert nvcc==13.0 before compiling extensions (r0b0tlab gotcha:
# newer deps can pull cu13.x python compiler pkgs that shadow the system nvcc -> ABI mismatch).
RUN update-alternatives --set cuda /usr/local/cuda-13.0 2>/dev/null || true; \
    export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH; \
    nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/nvcc release \1/p' | head -1; \
    test "$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -1)" = "13.0"

COPY vllm-src/ /build/vllm-src/

RUN pip uninstall -y vllm 2>&1 | tail -3

WORKDIR /build/vllm-src
RUN CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
    pip install --no-deps -v . 2>&1 | tee /tmp/vllm-install.log | tail -120 && \
    echo "[vllm pip install] exit=$?"

# torchcodec: INTENTIONALLY NOT INSTALLED. main lists torchcodec>=0.14 in cuda.txt (video
# decode), but the newest published torchcodec (0.14.0) is ABI-INCOMPATIBLE with torch
# 2.11.0+cu130 (verified 2026-07-14), and no torch-2.11-matched torchcodec exists on PyPI.
# vLLM guards the import (vllm/multimodal/video.py: try/except ImportError ->
# PlaceholderModule), so ABSENT torchcodec degrades GRACEFULLY (video-decode unavailable)
# while text/image/audio/voice are unaffected — a PRESENT-but-broken torchcodec would
# instead throw an uncaught OSError and crash. ffmpeg is still installed (apt, above) for
# other codec paths. Revisit when torchcodec publishes a torch-2.11 build.

# nvidia-cutlass-dsl pinned to the upstream-tested 4.5.2 (matches main requirements/cuda.txt)
RUN pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.5.2" 2>&1 | tail -3 && \
    python3 -c "import importlib.metadata as m; print('cutlass-dsl:', m.version('nvidia-cutlass-dsl'))"

# FlashInfer 0.6.13 (main pin, unchanged from the v0.25.0 image). Purge any stale
# jit-cache first (it shadows matching-version kernels); rely on runtime JIT or the cubin.
RUN pip uninstall -y flashinfer-jit-cache 2>&1 | tail -2 || true; \
    pip install --no-cache-dir "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13" 2>&1 | tail -3 && \
    (flashinfer download-cubin 2>&1 | tail -3 || echo "[WARN] cubin download failed; runtime JIT fallback") && \
    python3 -c "import flashinfer, importlib.metadata as m; assert flashinfer.__version__=='0.6.13', flashinfer.__version__; \
      import importlib.util; assert importlib.util.find_spec('flashinfer_jit_cache') is None, 'stale jit-cache present'; \
      print('flashinfer:', flashinfer.__version__, '(jit-cache absent)')"

# xgrammar >= 0.2.1 (tool-choice 500s with ImportError: normalize_tool_choice otherwise)
RUN pip install --no-cache-dir "xgrammar>=0.2.1,<1.0.0" 2>&1 | tail -2 && \
    python3 -c "from xgrammar import normalize_tool_choice; print('xgrammar: normalize_tool_choice OK')"

RUN pip install --no-cache-dir "scipy>=1.11" 2>&1 | tail -3 && \
    pip install --no-cache-dir --no-deps \
      "turboquant @ git+https://github.com/AEON-7/turboquant.git@fix/cuda-graph-safe-qjl-powers" \
      2>&1 | tail -3 || \
    echo "[WARN] turboquant install attempted; check logs above if needed"

# Pinned stable transformers (main floor is >=5.5.3; 5.12.1 fleet-validated)
RUN pip install --no-cache-dir --upgrade "transformers==5.12.1" 2>&1 | tail -3 && \
    python3 -c "import transformers; print('transformers:', transformers.__version__)"

COPY humming-stub/ /tmp/humming-stub/
RUN pip install --no-cache-dir /tmp/humming-stub && rm -rf /tmp/humming-stub && \
    python3 -c "from humming.dtypes import DataType; print('humming-stub: importable')"

COPY verify.py /tmp/verify.py
RUN python3 /tmp/verify.py && rm /tmp/verify.py

# Smoke (from WORKDIR / so cwd doesn't shadow the installed pkg; dlopen stable-ABI vs the
# driver stub): confirm the load-bearing AEON carries survived the main merge, the V2
# DFlash stack is present, and MRv2 is NOT pinned off (main must route hybrid DFlash
# drafters to V2).
WORKDIR /
RUN mkdir -p /tmp/cuda-stub && \
    ln -s /usr/local/cuda-13.0/targets/sbsa-linux/lib/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 && \
    LD_LIBRARY_PATH=/tmp/cuda-stub:$LD_LIBRARY_PATH python3 -c "\
import vllm._C_stable_libtorch; import vllm._moe_C_stable_libtorch; \
import vllm, inspect; assert vllm.__version__.startswith('0.26.0'), vllm.__version__; \
from vllm import LLM, SamplingParams; from vllm.config import VllmConfig; \
import vllm.v1.attention.backends.triton_attn as t; assert 'aeon-triton-nvfp4-kv' in inspect.getsource(t), 'Triton NVFP4-KV carry lost'; \
import vllm.v1.attention.ops.triton_unified_attention as tu; assert 'KV_QUANT_MODE == 5' in inspect.getsource(tu), 'NVFP4 kernel literal lost'; \
from vllm.v1.kv_cache_interface import KVQuantMode; assert int(KVQuantMode.NVFP4) == 5, 'KVQuantMode.NVFP4 renumbered — kernel literal now stale'; \
from vllm.utils.torch_utils import nvfp4_kv_cache_split_views; \
import inspect as _i; _src = _i.getsource(t.TritonAttentionBackend.get_kv_cache_stride_order); \
assert 'cache_dtype_str' in _i.signature(t.TritonAttentionBackend.get_kv_cache_stride_order).parameters, 'stride order missing cache_dtype_str kwarg'; \
assert 'cache_dtype_str == \"nvfp4\"' in _src, 'nvfp4 5-dim stride-order branch lost'; \
import vllm.config.compilation as cc; assert 'aeon-cudagraph-align-nonfull' in inspect.getsource(cc), 'cudagraph align carry lost'; \
import vllm.v1.worker.gpu_model_runner as gmr; assert 'aeon-uma-negative-estimate-clamp' in inspect.getsource(gmr), 'UMA clamp carry lost'; \
import vllm.v1.worker.gpu.spec_decode.dflash.speculator; \
import vllm.config.vllm as cv; assert '_dflash_needs_multi_kv_group' in inspect.getsource(cv), 'hybrid-DFlash V2 forcing missing'; \
import vllm.envs as e; assert e.VLLM_USE_V2_MODEL_RUNNER is None, 'MRv2 must not be pinned (V1 tree has no DFlash carries)'; \
import vllm.multimodal.video as vid; \
print('vllm', vllm.__version__, '+aeon import OK; Triton NVFP4-KV + UMA clamp + cudagraph-align present; V2 DFlash stack present; MRv2 unpinned; video.py graceful-degrade OK')" && \
    rm -rf /tmp/cuda-stub

RUN rm -rf /build /root/.cache/pip

LABEL ai.aeon.vllm_base="vLLM main @ 2bd895762 2026-07-15 (from-source, sm_121a, aeon-main-dspark merge)" \
      ai.aeon.model="fleet: Gemma-4-26B-A4B, Qwen3.6-27B, Qwen3.6-35B-A3B" \
      ai.aeon.hardware="NVIDIA DGX Spark GB10 SM121" \
      ai.aeon.features="gemma4,qwen3.6,dflash-mrv2,dflash-swa-upstream,nvfp4,triton-nvfp4-kv,fp8-kv,flashinfer-0.6.13,cutlass-dsl-4.5.2,uma-clamp,cudagraph-align,mrv2-default,dspark-ready,turboquant,tool-calling" \
      org.opencontainers.image.description="AEON vLLM Ultimate — vLLM main (2026-07-15) built from source for DGX Spark / Blackwell (sm_121a/GB10). DFlash SWA/prefix-cache/concurrency handled natively by upstream Model-Runner-V2 (#47914/#48135); carries Triton NVFP4-KV (#44389 lineage), UMA clamp (#46932 port), cudagraph-align (#46324 port). MRv2 unpinned. torchcodec omitted (torch-2.11 ABI)." \
      org.opencontainers.image.documentation="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["/bin/bash"]
