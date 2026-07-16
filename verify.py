"""Self-contained verification of the aeon-vllm-ultimate image."""
import sys

print("=" * 60)
print("AEON vLLM Ultimate — environment verification")
print("=" * 60)

try:
    import vllm
    print(f"  vllm:        {vllm.__version__}  ({vllm.__file__})")
except Exception as e:
    print(f"  vllm:        FAIL ({e})")
    sys.exit(1)

try:
    import flashinfer
    print(f"  flashinfer:  {flashinfer.__version__}")
except Exception as e:
    print(f"  flashinfer:  WARN ({e})")

try:
    import torch
    print(f"  torch:       {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"  cuda avail:  {torch.cuda.is_available()}")
except Exception as e:
    print(f"  torch:       FAIL ({e})")
    sys.exit(1)

try:
    import modelopt
    print(f"  modelopt:    {modelopt.__version__}")
except Exception as e:
    print(f"  modelopt:    WARN ({e})")

try:
    import turboquant
    ver = getattr(turboquant, "__version__", "unknown")
    print(f"  turboquant:  {ver}")
except ImportError as e:
    print(f"  turboquant:  not installed ({e})")

try:
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig
    print(f"  vllm.LLM:    importable")
except Exception as e:
    print(f"  vllm.LLM:    FAIL ({e})")
    sys.exit(1)

try:
    # Exercises the actual eager import path the model registry hits at boot
    # (vllm/model_executor/models/qwen3_5.py -> ...gdn.qwen_gdn_linear_attn
    # -> auto_awq -> kernels.linear -> humming_utils), not just a bare
    # `from humming.dtypes import DataType`. humming_utils.py builds
    # {humming_dtypes.<name>: dtype} tables at import time whenever the
    # `humming` package is importable at all — regardless of chosen
    # quantization method — so every dtype name it references must exist on
    # the stub or model loading crashes before engine startup.
    import vllm.model_executor.layers.quantization.utils.humming_utils  # noqa: F401
    print(f"  humming_utils: importable (stub dtypes cover all referenced attrs)")
except Exception as e:
    print(f"  humming_utils: FAIL ({e})")
    sys.exit(1)

print()
print("GREEN - aeon-vllm-ultimate ready")
