import inspect
from functools import lru_cache
from importlib import import_module

import torch
from packaging import version


FLASH_ATTENTION_2 = "flash_attention_2"
FLASH_ATTENTION_3 = "flash_attention_3"


@lru_cache()
def _import_flash_attn_2_interface():
    return import_module("flash_attn.flash_attn_interface")


@lru_cache()
def _import_flash_attn_2_modules_mha():
    return import_module("flash_attn.modules.mha")


@lru_cache()
def _import_flash_attn_3_interface():
    last_error = None
    for module_name in ("flash_attn_interface", "hopper.flash_attn_interface"):
        try:
            return import_module(module_name)
        except ImportError as exc:
            last_error = exc

    raise ImportError(
        "FlashAttention 3 is not installed. Install the standard flash-attn package and then "
        "build the Hopper beta kernels from the flash-attention repo's `hopper/` directory."
    ) from last_error


@lru_cache()
def is_flash_attn_2_available():
    try:
        _import_flash_attn_2_interface()
        return True
    except ImportError:
        return False


@lru_cache()
def is_flash_attn_3_available():
    try:
        _import_flash_attn_3_interface()
        return True
    except ImportError:
        return False


def _get_flash_attn_callable(implementation: str, function_name: str):
    if implementation == FLASH_ATTENTION_2:
        if function_name == "flash_attn_varlen_kvpacked_func":
            module = _import_flash_attn_2_modules_mha()
        else:
            module = _import_flash_attn_2_interface()
    elif implementation == FLASH_ATTENTION_3:
        module = _import_flash_attn_3_interface()
    else:
        raise ValueError(f"Unsupported FlashAttention implementation: {implementation}")

    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise ImportError(
            f"{implementation} does not expose `{function_name}`. "
            "Please install a compatible flash-attn build."
        ) from exc


@lru_cache()
def _get_parameter_names(func):
    try:
        return frozenset(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return None


def _supports_argument(func, argument_name: str) -> bool:
    parameter_names = _get_parameter_names(func)
    return parameter_names is None or argument_name in parameter_names


def _call_with_supported_kwargs(func, **kwargs):
    parameter_names = _get_parameter_names(func)
    if parameter_names is None:
        return func(**kwargs)

    return func(**{key: value for key, value in kwargs.items() if key in parameter_names})


def _ensure_flash_attn_3_runtime_requirements():
    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention 3 requires CUDA.")

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError("FlashAttention 3 requires Hopper-class GPUs (SM90 / H100 / H800 or newer).")

    if torch.version.cuda is not None and version.parse(torch.version.cuda) < version.parse("12.3"):
        raise RuntimeError(
            f"FlashAttention 3 requires CUDA >= 12.3, but the current PyTorch build uses CUDA {torch.version.cuda}."
        )


def flash_attn_supports_argument(implementation: str, function_name: str, argument_name: str) -> bool:
    try:
        func = _get_flash_attn_callable(implementation, function_name)
    except ImportError:
        return False
    return _supports_argument(func, argument_name)


def flash_attn_func(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    implementation: str,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    return_attn_probs: bool = False,
    **kwargs,
):
    func = _get_flash_attn_callable(implementation, "flash_attn_func")

    if implementation == FLASH_ATTENTION_3:
        _ensure_flash_attn_3_runtime_requirements()
        if dropout_p not in (0, 0.0) and not _supports_argument(func, "dropout_p"):
            raise NotImplementedError("The installed FlashAttention 3 build does not support non-zero attention dropout.")

    if return_attn_probs and not _supports_argument(func, "return_attn_probs"):
        raise NotImplementedError(
            f"The installed {implementation} build does not support returning attention probabilities."
        )

    return _call_with_supported_kwargs(
        func,
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        return_attn_probs=return_attn_probs,
        **kwargs,
    )


def flash_attn_varlen_func(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    implementation: str,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    return_attn_probs: bool = False,
    **kwargs,
):
    func = _get_flash_attn_callable(implementation, "flash_attn_varlen_func")

    if implementation == FLASH_ATTENTION_3:
        _ensure_flash_attn_3_runtime_requirements()
        if dropout_p not in (0, 0.0) and not _supports_argument(func, "dropout_p"):
            raise NotImplementedError("The installed FlashAttention 3 build does not support non-zero attention dropout.")

    if return_attn_probs and not _supports_argument(func, "return_attn_probs"):
        raise NotImplementedError(
            f"The installed {implementation} build does not support returning attention probabilities."
        )

    return _call_with_supported_kwargs(
        func,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        return_attn_probs=return_attn_probs,
        **kwargs,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    *,
    implementation: str,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    return_attn_probs: bool = False,
    **kwargs,
):
    if implementation == FLASH_ATTENTION_2:
        func = _get_flash_attn_callable(implementation, "flash_attn_varlen_kvpacked_func")
        if return_attn_probs and not _supports_argument(func, "return_attn_probs"):
            raise NotImplementedError(
                f"The installed {implementation} build does not support returning attention probabilities."
            )
        return _call_with_supported_kwargs(
            func,
            q=q,
            kv=kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=return_attn_probs,
            **kwargs,
        )

    if implementation != FLASH_ATTENTION_3:
        raise ValueError(f"Unsupported FlashAttention implementation for packed KV attention: {implementation}")

    return flash_attn_varlen_func(
        q=q,
        k=kv[:, 0],
        v=kv[:, 1],
        implementation=implementation,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=return_attn_probs,
        **kwargs,
    )
