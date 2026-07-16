class DataType:
    """Stub. Calling this will fail — only used to satisfy `from humming.dtypes import DataType`."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "humming-stub: real humming library is not installed. "
            "Choose a quantization method other than `humming` (e.g. modelopt, awq, gguf)."
        )

    @classmethod
    def from_torch_dtype(cls, *args, **kwargs):
        raise RuntimeError(
            "humming-stub: real humming library is not installed. "
            "Choose a quantization method other than `humming` (e.g. modelopt, awq, gguf)."
        )


class _StubDTypeConstant:
    """Inert placeholder for a humming dtype enum member.

    vllm.model_executor.layers.quantization.utils.humming_utils builds
    {humming_dtypes.<name>: torch_dtype} lookup tables at MODULE IMPORT TIME
    (inside `if has_humming():`), unconditionally — i.e. even when the
    selected quantization method isn't humming. These objects only need to
    exist and be hashable as dict keys; they are never expected to do
    anything (any real use of humming would already have failed earlier, at
    HummingMethod.__init__ / DataType.__init__).
    """
    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"<humming-stub dtype {self._name!r}>"


# Every module-level constant vllm's humming_utils.py / kernels reference as
# `humming_dtypes.<name>` for dict-key purposes. Keep in sync with:
#   grep -rn "humming_dtypes\.\w\+" vllm/ -o | sed 's/.*\.//' | sort -u
bfloat16 = _StubDTypeConstant("bfloat16")
float16 = _StubDTypeConstant("float16")
float32 = _StubDTypeConstant("float32")
float4e2m1 = _StubDTypeConstant("float4e2m1")
float8e4m3 = _StubDTypeConstant("float8e4m3")
float8e5m2 = _StubDTypeConstant("float8e5m2")
float8e8m0 = _StubDTypeConstant("float8e8m0")
int4 = _StubDTypeConstant("int4")
int8 = _StubDTypeConstant("int8")
uint2 = _StubDTypeConstant("uint2")
uint3 = _StubDTypeConstant("uint3")
uint4 = _StubDTypeConstant("uint4")
uint8 = _StubDTypeConstant("uint8")
